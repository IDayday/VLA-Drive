"""Train spatial BEV tasks with verified pretrained current-image features only."""
import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path

import torch
from torch import nn

from starVLA.model.modules.joint_world.bev_tasks import (TaskBEVEncoder, BEVGraphFusion,
    InteractionGeometryHead, bev_loss_sums, interaction_loss_sum, raster_targets)
from starVLA.model.modules.structured_world.providers import calibration_fingerprint
from tools.joint_world.train_graph import load_samples
from tools.structured_world.provider_cli import current_observation
from tools.structured_world.runtime import seed_all
from tools.structured_world_v1p1.budget import start, record
from tools.structured_world_v1p1.reaudit_metrics import write_csv


WEIGHT_SHA = 'a7ea19fa0ed99244e67b624c72b8580b7e9553043245905be58796a608eb9345'


def load_bev(samples, root, target_root):
    fingerprint = hashlib.sha256()
    for sample in samples:
        token = sample['cache']['token']; path = Path(root) / (token + '.pt')
        fingerprint.update(token.encode() + hashlib.sha256(path.read_bytes()).digest())
        payload = torch.load(path, map_location='cuda', weights_only=True)
        if set(payload) != {'metadata','features','coordinates','observation_support'}:
            raise ValueError('Unexpected BEV cache fields')
        meta = payload['metadata']
        inputs, images = current_observation(Path(target_root)/'observations'/(token+'.npz'), '/', 'cpu')
        expected = {'scene_token': token, 'decision_time': int(inputs.decision_time[0]),
                    'image_sha256': images, 'calibration_sha256': calibration_fingerprint(inputs),
                    'backbone_weights_sha256': WEIGHT_SHA, 'pretrained_bev': False,
                    'sensor_contract': {'cameras':['CAM_F0','CAM_L0','CAM_R0'],'time':'current_only'}}
        for key, value in expected.items():
            if meta[key] != value: raise ValueError('BEV identity mismatch: '+key)
        if payload['features'].shape != (1,1960,1024) or payload['coordinates'].shape != (1,1960,3):
            raise ValueError('Incorrect BEV shape')
        if payload['observation_support'].shape != (1,1960) or payload['observation_support'].dtype != torch.bool:
            raise ValueError('Incorrect BEV support')
        for name in ['features','coordinates','observation_support']:
            if not torch.isfinite(payload[name]).all(): raise ValueError('Invalid BEV values')
        sample['bev'] = payload
        # Targets stay separate and are never fed to encoder/fusion.
        t = sample['target']
        for name, value in vars(t).items():
            if torch.is_tensor(value): setattr(t, name, value.cuda())
    return fingerprint.hexdigest()


def predict(model, sample):
    bev, cache = sample['bev'], sample['cache']
    pred = model['encoder'](bev['features'], bev['coordinates'], bev['observation_support'])
    _, actor_bev = model['fusion'](cache['actor_features'].cuda(), cache['current_xy'].cuda(), pred['memory'], bev['observation_support'])
    pair = model['interaction'](actor_bev)
    return pred, pair


@torch.no_grad()
def evaluate(model, samples, out, tag):
    model.eval(); rows=[]
    for sample in samples:
        pred, pair = predict(model, sample)
        bev = sample['bev']
        occ, covered, motion, valid = raster_targets(sample['target'], bev['coordinates'][0], bev['observation_support'][0])
        positive = pred['occupancy_logits'][0] >= 0
        errors = (pred['future_displacement'][0] - motion).norm(dim=-1)
        stationary = motion.norm(dim=-1)
        separation, pairs = interaction_loss_sum(pair, sample['xy'].cuda(), sample['valid'].cuda())
        rows.append({'token':sample['cache']['token'], 'status':'ok', 'tp':int((positive & occ & covered).sum()),
                     'fp':int((positive & ~occ & covered).sum()), 'fn':int((~positive & occ & covered).sum()),
                     'covered_cells':int(covered.sum()), 'motion_error_sum':float(errors[valid].sum()),
                     'stationary_error_sum':float(stationary[valid].sum()),'motion_points':int(valid.sum()),
                     'pair_loss_sum':float(separation),'pair_count':pairs})
    count=sum(r['motion_points'] for r in rows);tp=sum(r['tp'] for r in rows);fp=sum(r['fp'] for r in rows);fn=sum(r['fn'] for r in rows)
    summary={'scenes':len(rows),'failed':0,'occupied_precision':tp/max(tp+fp,1),'occupied_recall':tp/max(tp+fn,1),
             'occupied_iou':tp/max(tp+fp+fn,1),'dense_motion_ADE':sum(r['motion_error_sum'] for r in rows)/count if count else None,
             'stationary_dense_ADE':sum(r['stationary_error_sum'] for r in rows)/count if count else None,
             'dense_motion_points':count,'pair_geometry_loss':sum(r['pair_loss_sum'] for r in rows)/max(sum(r['pair_count'] for r in rows),1),
             'motion_evaluation':'oracle GT current occupied cells with valid tracked futures; not end-to-end actor recall',
             'future_free_space_supervision':False,'planning_evaluation':False}
    write_csv(out/(tag+'.csv'),rows);(out/(tag+'.json')).write_text(json.dumps(summary,indent=2))
    print(json.dumps(dict(tag=tag,**summary)),flush=True);model.train()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['train-cache','holdout-cache','train-bev','holdout-bev','targets','data-root','output','ledger','run-id']:
        p.add_argument('--'+key,required=True)
    p.add_argument('--steps',type=int,default=600);a=p.parse_args()
    if not 1 <= a.steps <= 1000: raise ValueError('Task probe cap1000')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);seed_all(42)
    identity={'arguments':vars(a),'code_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'provider':'frozen pretrained visual backbone + new calibrated BEV','backbone_sha256':WEIGHT_SHA,
              'batch':8,'seed':42,'labels':'current occupancy / valid future track displacement / pair separation; no PDMS'}
    start(a.ledger,a.run_id,a.steps,identity);completed=0
    try:
        train,_,train_hash=load_samples(a.train_cache,a.targets,a.data_root,8)
        holdout,_,holdout_hash=load_samples(a.holdout_cache,a.targets,a.data_root,8)
        identity['train_label_sha256']=train_hash;identity['holdout_label_sha256']=holdout_hash
        identity['train_bev_sha256']=load_bev(train,a.train_bev,a.targets)
        identity['holdout_bev_sha256']=load_bev(holdout,a.holdout_bev,a.targets)
        assert not ({s['cache']['token'] for s in train}&{s['cache']['token'] for s in holdout})
        dim=train[0]['cache']['actor_features'].shape[-1]
        model=nn.ModuleDict({'encoder':TaskBEVEncoder(),'fusion':BEVGraphFusion(dim),'interaction':InteractionGeometryHead()}).cuda()
        # A task-only probe trains the BEV reader, not its graph-use gate/output projection.
        model['fusion'].gate.requires_grad_(False);model['fusion'].output.requires_grad_(False)
        parameters=[x for x in model.parameters() if x.requires_grad]
        opt=torch.optim.AdamW(parameters,lr=1e-4,weight_decay=.01)
        identity['trainable_parameters']=sum(p.numel() for p in parameters)
        (out/'manifest.json').write_text(json.dumps(identity,indent=2))
        evaluate(model,train,out,'train_0');order=list(range(len(train)));random.shuffle(order);position=0
        for step in range(1,a.steps+1):
            if not record(a.ledger,a.run_id,completed):raise RuntimeError('Budget cap')
            opt.zero_grad(set_to_none=True);sums={k:[] for k in ['occupied','free','motion','interaction']};counts={k:0 for k in sums}
            for _ in range(8):
                if position==len(order):random.shuffle(order);position=0
                sample=train[order[position]];position+=1;pred,pair=predict(model,sample);bev=sample['bev']
                s,c=bev_loss_sums(pred,[sample['target']],bev['coordinates'],bev['observation_support'])
                s['interaction'],c['interaction']=interaction_loss_sum(pair,sample['xy'].cuda(),sample['valid'].cuda())
                for k in sums:sums[k].append(s[k]);counts[k]+=c[k]
            loss=sum(sum(sums[k])/max(counts[k],1) for k in sums)
            if not torch.isfinite(loss):raise FloatingPointError('BEV task loss')
            loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True));opt.step();completed=step
            record(a.ledger,a.run_id,step)
            row={'step':step,'loss':float(loss.detach()),'gradient_before_clip':norm,'presentations':step*8,'effective_epochs':step*8/len(train),'peak_gpu_bytes':torch.cuda.max_memory_allocated()}
            with (out/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if step%100==0:print(json.dumps(row),flush=True)
            if step in {300,a.steps}:
                evaluate(model,train,out,'train_'+str(step));torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,'identity':identity},out/f'checkpoint_{step}.pt')
        evaluate(model,holdout,out,'holdout_final');record(a.ledger,a.run_id,completed,'complete')
    except BaseException:record(a.ledger,a.run_id,completed,'failed');raise


if __name__=='__main__':main()
