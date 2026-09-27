"""Matched current/ALL/MASK representation transfer through the original frozen DiT."""
import argparse
import csv
import json
import math
from pathlib import Path
import random
import subprocess
import time
import numpy as np
import torch
from tools.local_interaction_mask_v2.foundation import rng_state, restore_rng, collate
from tools.local_interaction_mask_v2.graph_runtime import LocalCorpus, batch_current, cached_worker_init
from tools.local_interaction_mask_v2.planner_runtime import load_public_head, predict_payload, decode_actions
from tools.local_interaction_mask_v2.train_foundation import atomic_json
from starVLA.model.modules.joint_world.local_planner import LocalPlanningBridge
from starVLA.model.modules.joint_world.public_baseline import sha256


@torch.no_grad()
def evaluate(head, bridge, corpus, out, tag, check_gate_zero=False):
    bridge.eval(); rows=[]; max_condition_error=0.; max_action_error=0.
    for sample in (corpus[i] for i in range(len(corpus))):
        result=predict_payload(head, bridge, sample['payload'], corpus.identity)
        trajectory=decode_actions(result['normalized_actions'])[0].cpu()
        target=decode_actions(sample['action'])
        error=(trajectory[:,:2]-target[:,:2]).norm(dim=-1)
        if not torch.isfinite(trajectory).all():raise FloatingPointError('Invalid DiT prediction')
        if check_gate_zero:
            native=sample['payload']['native_actions'].cuda().float()
            max_condition_error=max(max_condition_error,float((native-result['action_conditions']).abs().max()))
            baseline=predict_payload(head,None,sample['payload'],corpus.identity)
            max_action_error=max(max_action_error,float((baseline['normalized_actions']-result['normalized_actions']).abs().max()))
        rows.append({'token':sample['token'],'status':'ok','ego_ADE_m':float(error.mean()),'ego_FDE_m':float(error[-1]),
                     'active_neighbors':int(sample['current']['local_graph'].active_actor_mask[:,1:].sum())})
    with (out/f'holdout_{tag}.csv').open('w') as f:
        writer=csv.DictWriter(f,list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={'scenes':len(rows),'failed':0,'ego_ADE_m':float(np.mean([r['ego_ADE_m'] for r in rows])),
             'ego_FDE_m':float(np.mean([r['ego_FDE_m'] for r in rows])), 'gate':float(bridge.adapter.gate),
             'scope':'Actual DiT imitation error on training-domain holdout, not PDMS'}
    if check_gate_zero:
        summary.update(gate_zero_condition_max_error=max_condition_error,gate_zero_DiT_max_error=max_action_error)
        if max_condition_error!=0 or max_action_error!=0:raise AssertionError('Gate zero failed original DiT parity')
    atomic_json(out/f'holdout_{tag}.json',summary);bridge.train();return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('cache','targets','meta-root','foundation','holdout','graph-config','output'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--mode',choices=['current','all','mask'],required=True)
    p.add_argument('--graph-checkpoint');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=8);p.add_argument('--schedule-epochs',type=int,default=16)
    p.add_argument('--batch',type=int,default=32);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int)
    a=p.parse_args()
    cached_worker_init(0)
    if a.batch<1 or not 1<=a.epochs<=a.schedule_epochs:raise ValueError('Invalid finite transfer plan')
    if (a.mode=='current')!=(a.graph_checkpoint is None):raise ValueError('Only ALL/MASK load a graph checkpoint')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'checkpoint.pt').exists() and not a.resume:raise FileExistsError('Use new run or explicit resume')
    corpus=LocalCorpus(a.cache,a.targets,a.meta_root)
    spec=json.loads(Path(a.holdout).read_text());holdout=LocalCorpus(spec['cache'],spec['targets'],spec['meta_root'])
    if corpus.manifest['identity_sha256']!=holdout.manifest['identity_sha256']:
        raise ValueError('Train/holdout current foundation or graph differs')
    cfg=json.loads(Path(a.graph_config).read_text())
    if cfg!=corpus.identity['graph_config']:raise ValueError('Cache graph configuration differs')
    head,_=load_public_head(a.foundation,corpus.identity['foundation_sha256'])
    dim=corpus[0]['current']['context'].shape[-1]
    model=LocalPlanningBridge(dim,cfg,a.mode,a.seed).cuda()
    if a.graph_checkpoint:
        graph=torch.load(a.graph_checkpoint,map_location='cpu',weights_only=False)
        if graph['identity']['cache_identity']!=corpus.manifest['identity_sha256']:
            raise ValueError('Graph training and planner current contract differ')
        if graph['identity']['arguments']['mode']!=a.mode or graph['identity']['config']!=cfg:
            raise ValueError('Wrong ALL/MASK graph architecture or supervision arm')
        model.graph.load_state_dict(graph['model'],strict=True)
    # Loading frozen modules does not shift the common optimizer/noise stream.
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    params=[v for v in model.parameters() if v.requires_grad]
    opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01)
    horizon=math.ceil(len(corpus)/a.batch)*a.schedule_epochs
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:min((s+1)/100,1)*(.1+.9*.5*(1+math.cos(math.pi*min(s,horizon)/horizon))))
    identity={'schema_version':2,'code_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'arguments':{k:v for k,v in vars(a).items() if k not in ('output','resume','stop_after','epochs')},
        'current_identity':corpus.manifest['identity_sha256'],'labels':corpus.label_fingerprint,
        'holdout_labels':holdout.label_fingerprint,'graph_config':cfg,
        'graph_checkpoint_sha256':sha256(a.graph_checkpoint) if a.graph_checkpoint else None,
        'foundation_sha256':corpus.identity['foundation_sha256'],
        'trainable_parameters':sum(p.numel() for p in params),'frozen_DiT_parameters':sum(p.numel() for p in head.parameters()),
        'trainable':['current_projection','graph_to_world','adapter'],
        'frozen':['Qwen including common LoRA/tokens','Reader/current heads','original DiT','trajectory graph'],
        'condition':'current memory and optional ALL-HIDDEN generated features; original visual/risk context in every arm',
        'targets_in_input':False,'pdms_training':False,'single_training_seed':42}
    step=epoch=offset=presentations=0;seen=set()
    if a.resume:
        saved=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=False)
        if saved['identity']!=identity:raise ValueError('Planner resume identity differs')
        model.load_state_dict(saved['model'],strict=True);opt.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        step,epoch,offset,presentations=[saved[k] for k in ('step','epoch','offset','presentations')]
        seen=set(saved['seen']);restore_rng(saved['rng'])
    atomic_json(out/'manifest.json',identity)
    def save():
        torch.save({'identity':identity,'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),
            'step':step,'epoch':epoch,'offset':offset,'presentations':presentations,'seen':sorted(seen),'rng':rng_state(),
            'resume_boundary':'optimizer step, same code/data/device topology; frozen DiT remains eval'},out/'checkpoint.tmp')
        (out/'checkpoint.tmp').replace(out/'checkpoint.pt')
    if not a.resume:evaluate(head,model,holdout,out,'0',check_gate_zero=True)
    paused=False
    while epoch<a.epochs:
        order=np.random.default_rng(np.random.SeedSequence([a.seed,epoch])).permutation(len(corpus)).tolist()
        batches=[order[i:i+a.batch] for i in range(offset,len(order),a.batch)]
        loader=torch.utils.data.DataLoader(corpus,batch_sampler=batches,num_workers=a.workers,collate_fn=collate,
            worker_init_fn=cached_worker_init,prefetch_factor=1 if a.workers else None,
            multiprocessing_context='spawn' if a.workers else None,generator=torch.Generator().manual_seed(a.seed+epoch))
        for samples in loader:
            start=time.monotonic();current=batch_current(samples)
            native=torch.cat([s['payload']['native_actions'] for s in samples]).cuda()
            actions=torch.stack([s['action'] for s in samples]).cuda()
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',enabled=False):
                condition,_=model(native,current,[s['token'] for s in samples])
                loss=head(condition,actions,None)  # Frozen DiT still differentiates its conditioning.
            if not torch.isfinite(loss):raise FloatingPointError('Planner nonfinite FM loss')
            loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
            grads={name:float(sum((p.grad.float().square().sum() for p in module.parameters() if p.grad is not None),torch.zeros((),device='cuda')).sqrt())
                   for name,module in [('current_projection',model.current_projection),('graph_to_world',model.graph_to_world),('adapter',model.adapter)]}
            gate_grad=float(model.adapter.gate.grad)
            opt.step();scheduler.step();step+=1;offset+=len(samples);presentations+=len(samples);seen.update(s['token'] for s in samples)
            row={'step':step,'epoch':epoch,'presentations':presentations,'effective_epochs':presentations/len(corpus),'unique_scenes':len(seen),
                'ego_FM_loss':float(loss.detach()),'gradient_before_clip':norm,'gradients_after_clip':grads,'gate_gradient':gate_grad,
                'gate':float(model.adapter.gate),'lr':opt.param_groups[0]['lr'],'seconds':time.monotonic()-start,'peak_gpu_bytes':torch.cuda.max_memory_allocated()}
            with (out/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if step%100==0:print(json.dumps(row),flush=True)
            end_epoch=offset==len(corpus)
            if end_epoch:epoch+=1;offset=0
            atomic_json(out/'progress.json',{'step':step,'epoch':epoch,'offset':offset,'presentations':presentations})
            paused=((a.stop_after is not None and step>=a.stop_after) or (out/'STOP_REQUESTED').exists()) and epoch<a.epochs
            if end_epoch:evaluate(head,model,holdout,out,str(epoch))
            if end_epoch or paused:save()
            if end_epoch or paused:break
        if paused:break
    save();atomic_json(out/'status.json',{'status':'paused' if paused else 'complete','step':step,'epochs':epoch,'presentations':presentations})


if __name__=='__main__':main()
