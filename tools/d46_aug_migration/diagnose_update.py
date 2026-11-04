"""Read-only TRAIN replay: isolate a saved Adam update's numerical effect.

No optimizer, no new trajectories/scores, no training state writes. Comparison
copies live only in this process and never become checkpoints or teachers.
"""
import argparse
import json
import os
from pathlib import Path
import time

import torch

from starVLA.rl.d44_native_rl.parent import load
from starVLA.rl.d44_native_rl.config import source_config
from starVLA.rl.d44_native_rl.data import DataStream
from starVLA.rl.d44_native_rl.conditions import DConditions
from starVLA.rl.d46_augustus.actor import AugustusActor
from starVLA.rl.d46_augustus.deployment import inspect
from starVLA.rl.d46_augustus.runtime import AugustusExtension
from starVLA.rl.d46_augustus.kernel import stats
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.qwen_world_compare.contracts import require_dispatch, file_sha


def role(name):
    if '.language_model.' in name: return 'text_lora'
    if '.time_' in name: return 'time_lora'
    if name.endswith('action_position'): return 'action_position'
    return 'action_io'


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--before',required=True);p.add_argument('--after',required=True)
    p.add_argument('--material',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();host=require_dispatch();configure_numerics();torch.set_num_threads(2)
    device=torch.device('cuda',0);torch.cuda.set_device(device)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'SUMMARY.json').exists():raise FileExistsError('Already completed')
    ident,seal,_=inspect(args.before);after,aseal,_=inspect(args.after)
    if ident!=after or aseal['updates']!=seal['updates']+1:raise ValueError('Exact same-run consecutive Adam snapshots required')
    cfg=ident['recipe'];stream=DataStream(cfg,cfg['global_scenes'])
    batch=torch.load(args.material,map_location='cpu',weights_only=False,mmap=True)
    if batch['behavior_update']!=seal['updates']:raise ValueError('Behavior must match BEFORE policy')
    torch.manual_seed(42);model,info=load(cfg['parent_checkpoint'],device,binding=cfg['parent_registration'])
    actor=AugustusActor(model,source_config(cfg),cfg)
    actor.action_expert.decision.activation_checkpointing=False
    conditions=DConditions(model,stream.data,info,group=cfg['group'],capacity=cfg['condition_cache_scenes'])
    object.__setattr__(actor,'conditions',conditions)
    ext=AugustusExtension();ext.cfg=cfg;ext.configure_actor(actor)
    old=torch.load(Path(args.before)/'optimizer.pt',map_location='cpu',weights_only=False,mmap=True)['adapters']
    new=torch.load(Path(args.after)/'optimizer.pt',map_location='cpu',weights_only=False,mmap=True)['adapters']
    named=dict(actor.named_parameters());rows=[];start=time.monotonic()
    modes=('before','repeat_before','after','action_io','action_position','text_lora','time_lora','text_tiny')
    for index,saved in enumerate(batch['scenes']):
        if saved['padding'] or saved['token'] not in stream.data.records:raise ValueError('Only valid TRAIN')
        r={k:v.to(device) if torch.is_tensor(v) else v for k,v in saved.items()}
        row=dict(token=saved['token'],modes={});baseline=None
        for mode in modes:
            require_dispatch()
            with torch.no_grad():
                for name,x in old.items():
                    value=new[name] if mode=='after' or role(name)==mode else x
                    if mode=='text_tiny' and role(name)=='text_lora':value=x+0.001*(new[name]-x)
                    named[name].copy_(value.to(device))
                preds=[]
                with actor._context_call(r['condition']) as material:
                    prefix=actor._prepare(material,False)
                    for k,sigma in enumerate(r['timesteps'][:-1]):
                        preds.append(actor.predict_clean(r['chain'][:,k],sigma.expand(cfg['group']),prefix))
                preds=torch.stack(preds,1)
                if baseline is None:baseline=preds.clone()
                logr=[];kl=[]
                for k,(t,nxt) in enumerate(zip(r['timesteps'][:-1],r['timesteps'][1:])):
                    lr,kval=stats(preds[:,k],r['old_clean_x'][:,k],r['old_clean_x'][:,k],r['chain'][:,k],r['chain'][:,k+1],t,nxt,noise_scale=cfg['augustus']['noise_scale'])
                    logr.append(lr);kl.append(kval)
                row['modes'][mode]=dict(clean_max_abs=float((preds-baseline).abs().max()),
                    clean_rms=float((preds-baseline).square().mean().sqrt()),
                    saved_clean_max_abs=float((preds-r['old_clean_x']).abs().max()),
                    KL=float(torch.stack(kl).mean()),ratio_mean=float(torch.stack(logr).exp().mean()))
            print(json.dumps(dict(scene=index,mode=mode,**row['modes'][mode])),flush=True)
        rows.append(row);atomic_json(out/f'scene{index:03}.json',row)
    summary=dict(status='COMPLETE_TRAIN_NUMERICAL_REPLAY',host=host,before=args.before,after=args.after,
        before_update=seal['updates'],after_update=aseal['updates'],scenes=len(rows),rows=rows,
        optimizer_steps=0,PDM_queries=0,NAVTEST_used=False,seconds=time.monotonic()-start,
        peak_GPU_bytes=torch.cuda.max_memory_allocated(),source_sha256=file_sha(__file__))
    atomic_json(out/'SUMMARY.json',summary)


if __name__=='__main__':main()
