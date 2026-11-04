"""Read-only real TRAIN replay: policy-vs-reference gradients at a saved policy.

No optimizer steps, score queries, checkpoint writes or test-set inputs.
The replay must use the checkpoint's exact fresh behavior version.
"""
import argparse
import json
import os
from pathlib import Path
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

from starVLA.rl.d44_native_rl.parent import load
from starVLA.rl.d44_native_rl.config import source_config
from starVLA.rl.d44_native_rl.data import DataStream
from starVLA.rl.d44_native_rl.conditions import DConditions
from starVLA.rl.d46_augustus.actor import AugustusActor
from starVLA.rl.d46_augustus.deployment import inspect
from starVLA.rl.d46_augustus.runtime import AugustusExtension
from starVLA.rl.d46_augustus.kernel import policy_loss
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.qwen_world_compare.contracts import require_dispatch


def cosine(a,b):
    scale=a.norm()*b.norm()
    return None if scale==0 else float(torch.dot(a,b)/scale)


def vector(loss,parameters,retain):
    grads=torch.autograd.grad(loss,parameters,retain_graph=retain,allow_unused=True)
    return torch.cat([(torch.zeros(p.numel())if g is None else g.detach().float().reshape(-1).cpu())for p,g in zip(parameters,grads)])


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--collection',required=True)
    p.add_argument('--output',required=True);p.add_argument('--count',type=int,default=16);args=p.parse_args()
    host=require_dispatch();configure_numerics();torch.set_num_threads(2);device=torch.device('cuda',0);torch.cuda.set_device(device)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if(out/'SUMMARY.json').exists():raise FileExistsError('Already complete')
    ident,seal,_=inspect(args.checkpoint);cfg=ident['recipe'];stream=DataStream(cfg,cfg['global_scenes'])
    saved=[]
    for i in range(ident['world']):
        b=torch.load(Path(args.collection)/f'rank{i}.pt',map_location='cpu',weights_only=False,mmap=True)
        if b['behavior_update']!=seal['updates']:raise ValueError('Not this checkpoint fresh behavior')
        saved.extend(s for s in b['scenes']if not s['padding'])
    def category(s):
        safe=(s['factors'][:,:3]==1).all(1)
        return 'all_safe'if safe.all()else'all_unsafe'if not safe.any()else'mixed'
    # Include every rare unsafe/mixed group first, then active and inactive safe
    # groups. This is a stratified diagnostic, not a population mean.
    saved.sort(key=lambda s:(category(s)=='all_safe',float(s['advantages'].abs().max())==0,s['token']))
    selected=saved[:args.count]
    torch.manual_seed(42);model,info=load(cfg['parent_checkpoint'],device,binding=cfg['parent_registration'])
    actor=AugustusActor(model,source_config(cfg),cfg);actor.action_expert.decision.activation_checkpointing=False
    conditions=DConditions(model,stream.data,info,group=cfg['group'],capacity=cfg['condition_cache_scenes']);object.__setattr__(actor,'conditions',conditions)
    ext=AugustusExtension();ext.cfg=cfg;ext.configure_actor(actor)
    payload=torch.load(Path(args.checkpoint)/'optimizer.pt',map_location='cpu',weights_only=False,mmap=True)
    named=dict(actor.named_parameters())
    with torch.no_grad():
        for name,value in payload['adapters'].items():named[name].copy_(value.to(device))
    parameters=[p for p in actor.parameters()if p.requires_grad];totals={};rows=[];started=time.monotonic()
    for s in selected:
        require_dispatch();assert s['token']in stream.data.records
        r={k:v.to(device)if torch.is_tensor(v)else v for k,v in s.items()};r['expect_fresh']=True
        total,metrics,parts=actor(r,return_components=True)
        scores=torch.tensor([q['metrics']['score']for q in s['records']],device=device,dtype=torch.float64)*5
        adv=((scores-scores.mean())/(scores.std(unbiased=True)+1e-6)).float()
        alternate,_=policy_loss(parts['logratio'],adv,parts['reference_KL_steps'],clip=cfg['augustus']['clip'],beta=0.)
        grads={'SafetyEP':vector(parts['policy'],parameters,True),'PDMS_group':vector(alternate,parameters,True),'reference_beta01':vector(parts['reference'],parameters,False)}
        for k,v in grads.items():totals[k]=totals.get(k,torch.zeros_like(v))+v
        pg,reference=grads['SafetyEP'],grads['reference_beta01']
        row=dict(token=s['token'],category=category(s),norms={k:float(v.norm())for k,v in grads.items()},
                 reference_to_policy_norm=float(reference.norm()/pg.norm())if pg.norm()>0 else None,
                 cosine_policy_reference=cosine(pg,reference),cosine_SafetyEP_PDMS=cosine(pg,grads['PDMS_group']),
                 fresh_ratio_error=float(metrics['fresh_ratio_error']),reference_KL=float(metrics['reference_KL']))
        rows.append(row);atomic_json(out/f'scene{len(rows):03}.json',row);print(json.dumps(row),flush=True)
        del total,parts,grads,alternate
    result=dict(status='COMPLETE_REAL_TRAIN_GRADIENT_REPLAY',host=host,checkpoint=args.checkpoint,updates=seal['updates'],
                scenes=len(rows),collection_scenes=len(saved),full_collection=len(rows)==len(saved),rows=rows,aggregate_norms={k:float(v.norm())for k,v in totals.items()},
                aggregate_cosine_policy_reference=cosine(totals['SafetyEP'],totals['reference_beta01']),
                aggregate_cosine_SafetyEP_PDMS=cosine(totals['SafetyEP'],totals['PDMS_group']),
                reference_beta0001_gradient_is_beta01_div100=True,optimizer_updates=0,PDM_queries=0,NAVTEST_used=False,
                seconds=time.monotonic()-started,peak_GPU_bytes=torch.cuda.max_memory_allocated(),
                limits=('Full single fresh collection'if len(rows)==len(saved)else'Stratified subset of one fresh collection')+'; un-clipped true parameter gradients, not Adam steps or deployment gains.')
    atomic_json(out/'SUMMARY.json',result)


if __name__=='__main__':main()
