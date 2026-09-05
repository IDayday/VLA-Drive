#!/usr/bin/env python3
"""Bounded log-disjoint task learnability, frozen upstream, no Navtest feedback."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from navsim.planning.training.dataset import CacheOnlyDataset,drivevla_cached_collate
from navsim.agents.EpisodeDrive.layers.world_model.task_future_loss import (
    task_future_lite_loss,physical_element_losses,global_task_mean,HINDSIGHT_BINS,HORIZONS,sample_training_candidates,
)
from navsim.agents.EpisodeDrive.layers.world_model.physical_label_sidecar import score_with_physical_sidecar
from smoke_task_future_lite_real import agent_config
from planreg_audit_runtime import to_device_non_paths
from export_planreg_student_checkpoint import sha256_file


def metrics(out,values,classes,valid):
    prob=out['gap_logits'].float().softmax(-1)
    gapmask=valid[...,0]
    truth=F.one_hot(classes,5)
    gap_ce=-(prob.clamp_min(1e-8).log()*truth).sum(-1)
    brier=(prob-truth).square().sum(-1)
    entropy=-(prob*prob.clamp_min(1e-8).log()).sum(-1)
    def mean(x,mask):return float(x[mask].mean()) if mask.any() else None
    pred=prob.argmax(-1)
    near=(classes<=1)&gapmask
    road=out['road_margin']*2
    reference=values[...,1].clamp(-2,2)
    confidence=prob.max(-1).values
    accuracy=(pred==classes).float()
    calibration=[]
    for lo in np.arange(0,1,.1):
        mask=gapmask&(confidence>=lo)&(confidence<lo+.1)
        calibration.append(dict(lower=float(lo),count=int(mask.sum()),confidence=mean(confidence,mask),accuracy=mean(accuracy,mask)))
    return dict(gap_ce=mean(gap_ce,gapmask),gap_brier=mean(brier,gapmask),
        gap_accuracy=mean(accuracy,gapmask),near_contact_recall=mean((pred<=1).float(),near),
        near_contact_count=int(near.sum()),gap_calibration=calibration,
        road_mae_m_clipped=mean((road-reference).abs(),valid[...,1]),
        road_sign_accuracy=mean((road.sign()==reference.sign()).float(),valid[...,1]),
        road_near_boundary_mae_m=mean((road-reference).abs(),valid[...,1]&(reference.abs()<=.5)),
        progress_mae_m=mean((out['route_progress']*40-values[...,2]).abs(),valid[...,2]),
        gap_entropy_per_horizon=[mean(entropy[:,:,h],gapmask[:,:,h]) for h in range(entropy.shape[2])])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','shared-init','cache-root','scene-manifest','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--steps',type=int,default=150)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.manual_seed(0);torch.set_num_threads(4)
    cfg=agent_config('base',args.shared_init)
    agent=instantiate(cfg);agent.initialize();agent.to('cuda:0')
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    agent.load_state_dict({k.removeprefix('agent.'):v for k,v in payload['state_dict'].items()},strict=True)
    del payload
    agent.eval();agent.requires_grad_(False)
    manifest=json.loads(args.scene_manifest.read_text());rows=manifest['rows']
    dataset=CacheOnlyDataset(str(args.cache_root),agent.get_feature_builders(),agent.get_target_builders(),
        log_names=[row['log'] for row in rows],preprocess_images=True,preprocess_future_images=True,
        input_only_cache_name='planreg_input_only',reject_dynamic_feature_keys=True)
    data=[]
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for start in range(0,len(rows),2):
            subset=rows[start:start+2]
            samples=[dataset[dataset.tokens.index(row['token'])] for row in subset]
            features,targets=drivevla_cached_collate(samples)
            features=to_device_non_paths(features,torch.device('cuda:0'));targets=to_device_non_paths(targets,torch.device('cuda:0'))
            pred=agent(features)
            current,future,future_valid=agent._encode_ema_register_targets(features,targets,batch_size=len(subset))
            cand,idx=sample_training_candidates(targets['trajectory'],pred['proposals'])
            label_rows=[score_with_physical_sidecar(row['metric_cache'],pred['proposals'][i].float().cpu().numpy(),
                targets['trajectory'][i].float().cpu().numpy(),idx[i].cpu().numpy(),targets['physical_map_name'][i])[1]
                for i,row in enumerate(subset)]
            data.append(dict(reg=pred['planning_registers'].detach().float(),cand=cand.float(),status=features['status_feature'][:,:8].float(),
                current=current.float(),future=future.float(),future_valid=future_valid,
                poses=targets['logged_future_poses'],
                values=torch.as_tensor(np.stack([row['physical_values'] for row in label_rows]),device='cuda'),
                classes=torch.as_tensor(np.stack([row['gap_class'] for row in label_rows]),device='cuda'),
                valid=torch.as_tensor(np.stack([row['valid'] for row in label_rows]),device='cuda')))
            print(f'encoded_probe_scenes={start+len(subset)}',flush=True)
    bank={key:torch.cat([row[key] for row in data]) for key in data[0]}
    full=copy.deepcopy(agent.physical_query_decoder).float().requires_grad_(True).train()
    action=copy.deepcopy(full)
    del agent,dataset,data;gc.collect();torch.cuda.empty_cache()
    train=torch.tensor([i for i,row in enumerate(rows) if row['partition']=='train'],device='cuda')
    dev=torch.tensor([i for i,row in enumerate(rows) if row['partition']=='development'],device='cuda')
    assert not {rows[i]['log'] for i in train.tolist()} & {rows[i]['log'] for i in dev.tolist()}
    optim=torch.optim.AdamW(full.parameters(),lr=1e-4,weight_decay=.01)
    action_optim=torch.optim.AdamW(action.parameters(),lr=1e-4,weight_decay=.01)
    def take(indices):return {key:value[indices] for key,value in bank.items()}
    tr=take(train)
    curve=[]
    for step in range(args.steps):
        optim.zero_grad(set_to_none=True);action_optim.zero_grad(set_to_none=True)
        losses=task_future_lite_loss(full,tr['reg'],tr['cand'],tr['status'],tr['current'],tr['future'],
            tr['future_valid'],tr['poses'],tr['values'],tr['classes'],tr['valid'])
        action_loss,_,_=global_task_mean(physical_element_losses(action(torch.zeros_like(tr['reg']),tr['cand'],tr['status']),
            tr['values'],tr['classes']),tr['valid'])
        losses['wm_loss'].backward();action_loss.backward()
        torch.nn.utils.clip_grad_norm_(full.parameters(),1.);torch.nn.utils.clip_grad_norm_(action.parameters(),1.)
        optim.step();action_optim.step()
        if step%25==0 or step==args.steps-1:
            curve.append(dict(step=step,full_loss=float(losses['wm_loss']),action_loss=float(action_loss)))
    results={}
    with torch.no_grad():
        for label,indices in [('train',train),('development',dev)]:
            batch=take(indices)
            out=full(batch['reg'],batch['cand'],batch['status'])
            zero=action(torch.zeros_like(batch['reg']),batch['cand'],batch['status'])
            shuffled=full(batch['reg'].roll(1,0),batch['cand'],batch['status'])
            result={name:metrics(pred,batch['values'],batch['classes'],batch['valid']) for name,pred in
                    [('current',out),('action_ego_only',zero),('wrong_current',shuffled)]}
            for name,future in [('hindsight',batch['future']),('wrong_future_same_pose',batch['future'].roll(1,0))]:
                outputs=[]
                for h,t in enumerate(HORIZONS):
                    pred=full.forward_hindsight(batch['current'],future[:,h],batch['cand'],batch['status'],batch['poses'][:,h],t)
                    outputs.append({k:v[:,:,HINDSIGHT_BINS[h]] for k,v in pred.items()})
                teacher={k:torch.stack([p[k] for p in outputs],2) for k in outputs[0]}
                mask=batch['valid'][:,:,HINDSIGHT_BINS]&batch['future_valid'][:,None,:,None]
                result[name]=metrics(teacher,batch['values'][:,:,HINDSIGHT_BINS],batch['classes'][:,:,HINDSIGHT_BINS],mask)
            results[label]=result
    report=dict(kind='train_only_frozen_upstream_task_learnability_probe',steps=args.steps,
        checkpoint_sha256=sha256_file(args.checkpoint),rows=rows,log_disjoint=True,
        upstream_exposure='New Base VLM-only planning stack saw only these pilot-train logs in the numerical smoke; public VLM pretraining exposure unknown',
        dynamic_cache_boundary='Registers/EMA targets retained only in RAM during this frozen-upstream diagnostic; never read by formal training',
        extra_compute_not_formal_epochs=True,curve=curve,metrics=results,
        limitation='Small log split, random/new readout, no PDMS or causal planning improvement claim. No automatic WM reweighting.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='COMPLETED',curve=curve),indent=2))


if __name__=='__main__':main()
