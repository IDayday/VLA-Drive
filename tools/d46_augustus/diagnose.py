"""Bounded TRAIN-only native K10 versus old K3 deployment diagnostic.

Never exports priorities, labels or teachers to a trainer. Not NAVTEST, not a
checkpoint-selection score. Uses the same frozen scene population and root.
"""
import argparse
import json
import os
from pathlib import Path
import time
import torch
from starVLA.rl.d44_native_rl.parent import load
from starVLA.rl.d44_native_rl.data import DataStream
from starVLA.rl.d44_native_rl.conditions import DConditions
from starVLA.rl.d46_augustus.config import defaults
from starVLA.rl.d46_augustus.deployment import load_policy
from starVLA.rl.qwen_world_compare.contracts import require_dispatch,file_sha
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.simwam_recipe.reward_readouts import RewardService
from starVLA.model.modules.qwen_world_postcot.flow import shift_sigma,clean_x_step
from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--checkpoint');p.add_argument('--output',required=True)
    p.add_argument('--count',type=int,default=64);p.add_argument('--grids',default='K10,K3')
    a=p.parse_args();os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    os.environ.setdefault('FLASH_ATTENTION_DETERMINISTIC','1');host=require_dispatch();configure_numerics();torch.set_num_threads(2)
    if not 1<=a.count<=128:raise ValueError('Bounded first128 TRAIN scenes only')
    names=a.grids.split(',')
    if set(names)-{'K10','K3'}:raise ValueError('Unknown diagnostic grid')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);device=torch.device('cuda',0)
    cfg=defaults('AUG2_LORA');stream=DataStream(cfg,128)
    if a.checkpoint:
        model,identity,seal,model_sha=load_policy(a.checkpoint,device)
        info=identity['parent'];version=dict(checkpoint=a.checkpoint,updates=seal['updates'],model_sha=model_sha)
    else:
        model,info=load(cfg['parent_checkpoint'],device,binding=cfg['parent_registration'])
        version=dict(parent=info['model_sha256'],updates=0)
    conditions=DConditions(model,stream.data,info,group=1,capacity=2)
    service=RewardService(Path.cwd(),workers=2,cache_roots=stream.data.manifest['canonical_metric_cache_roots'])
    grids=dict(K10=shift_sigma(torch.linspace(1,0,11,dtype=torch.float32),5.).tolist(),K3=[1.,2/3,1/3,0.])
    start=time.monotonic();rows=[]
    try:
        for j in range(a.count):
            token,padding=stream.at(0,0,j,1)
            if padding:raise ValueError('Unexpected TRAIN padding')
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                material=conditions.materialize(conditions.get(token),device)
                prefix=model.prepare_prefix(material['current'],auxiliary=False)
                trajectories=[]
                for name in names:
                    z=torch.randn((1,8,3),generator=torch.Generator().manual_seed(2101)).to(device)
                    for t,nxt in zip(grids[name][:-1],grids[name][1:]):
                        x0=model.denoise_x0(z,z.new_full((1,),t),prefix)
                        z=clean_x_step(z,x0,t,nxt,validate=False)
                    trajectories.append(ActionCodec.denorm_odo(z).float().cpu())
            records=service.score((token,stream.logs[token],torch.cat(trajectories).numpy()))
            if len(records)!=len(names)or any(r['status']!='OK'for r in records):raise ValueError('Incomplete canonical TRAIN score')
            row=dict(token=token,log=stream.logs[token],scores={n:r['metrics']for n,r in zip(names,records)})
            atomic_json(out/f'scene{j:04}.json',row);rows.append(row)
        summary=dict(status='COMPLETE_TRAIN_DIAGNOSTIC',host=host,count=len(rows),version=version,
            NAVTEST=False,seed=2101,data=stream.identity,seconds=time.monotonic()-start,
            means={name:{k:sum(r['scores'][name][k]for r in rows)/len(rows)for k in rows[0]['scores'][name]}for name in names},
            source_file_sha256=file_sha(__file__),rows=rows)
        atomic_json(out/'SUMMARY.json',summary)
    finally:service.close()


if __name__=='__main__':main()
