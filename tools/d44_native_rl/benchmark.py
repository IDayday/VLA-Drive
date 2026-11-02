"""Full D44 GPU execution-policy qualification; no optimizer/scorer calls."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time
import torch
from starVLA.rl.d44_native_rl.parent import load
from starVLA.rl.d44_native_rl.actor import D44Actor
from starVLA.rl.d44_native_rl.conditions import DConditions
from starVLA.rl.d44_native_rl.config import defaults,source_config
from starVLA.rl.d44_native_rl.data import DataStream
from starVLA.rl.d44_native_rl.gaussian_objective import gaussian_terms,clipped_loss
from starVLA.rl.qwen_world_compare.contracts import require_dispatch
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.simwam_recipe.lora_adapter import restore_adapters
from starVLA.rl.flow_grpo.transactions import atomic_json


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True);p.add_argument('--adapter-checkpoint');a=p.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ.setdefault('FLASH_ATTENTION_DETERMINISTIC','1')
    require_dispatch();configure_numerics();torch.set_num_threads(2);device=torch.device('cuda',0)
    cfg=defaults('D0');stream=DataStream(cfg,64);model,info=load(cfg['parent_checkpoint'],device)
    actor=D44Actor(model,source_config(cfg),cfg)
    if a.adapter_checkpoint:restore_adapters(actor,torch.load(Path(a.adapter_checkpoint)/'optimizer.pt',map_location='cpu',weights_only=False)['adapters'])
    conds=DConditions(model,stream.data,info);object.__setattr__(actor,'conditions',conds)
    material=conds.materialize(conds.get(stream.tokens[0]),device);trainable=[p for p in actor.parameters()if p.requires_grad]
    records=[]
    for group in (8,16,32):
        condition=dict(material,context=torch.zeros(group,1,1,device=device));gen=torch.Generator().manual_seed(42)
        with torch.no_grad():
            chain,times,deltas,_=actor.sample_action_chain(condition,8,generator=gen)
            old,std=actor.transition_terms(condition,chain,times,deltas)
            ref,_=actor.transition_terms(condition,chain,times,deltas,use_ref=True)
        baseline=None
        for policy in ('source','short_off','all_off','all_off_flat_W3'):
            actor.study_config['probability_layout']='flat_W3'if policy=='all_off_flat_W3'else'sequential'
            actor.action_expert.decision.activation_checkpointing=policy not in('all_off','all_off_flat_W3')
            actor.action_expert.decision.checkpoint_short_suffix=policy=='source'
            elapsed=[]
            for repeat in range(3):
                actor.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();begin=time.monotonic()
                mean,actualstd=actor.transition_terms(condition,chain,times,deltas)
                ratio,kl=gaussian_terms(mean,old,ref,actualstd,chain[:,[8,9,10]])
                loss,_=clipped_loss(ratio,torch.linspace(-1,1,group,device=device));loss=loss+.001*kl.mean()
                loss.backward();torch.cuda.synchronize();elapsed.append(time.monotonic()-begin)
            grads=torch.cat([p.grad.detach().float().flatten()for p in trainable]);row=dict(group=group,policy=policy,
                median_forward_backward_seconds=statistics.median(elapsed),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                fresh_ratio_error=float(ratio.detach().expm1().abs().max()),gradient_norm=float(grads.norm()))
            if baseline is None:baseline=(mean.detach().clone(),grads.clone())
            row['mean_max_abs_vs_source']=float((mean.detach()-baseline[0]).abs().max())
            row['gradient_max_abs_vs_source']=float((grads-baseline[1]).abs().max())
            row['gradient_relative_norm_error']=float((grads-baseline[1]).norm()/baseline[1].norm().clamp_min(1e-30))
            row['status']='PASS'if row['mean_max_abs_vs_source']==0 and row['fresh_ratio_error']<=1e-4 and row['gradient_relative_norm_error']<=1e-6 else'FAIL'
            records.append(row);atomic_json(a.output,dict(status='RUNNING',records=records,scorer_calls=0,optimizer_updates=0))
    atomic_json(a.output,dict(status='PASS'if all(r['status']=='PASS'for r in records)else'FAIL',records=records,
        scorer_calls=0,optimizer_updates=0,precision='FP32 storage and gradients/BF16 forward, identical source kernel/mask/state'))


if __name__=='__main__':main()
