"""Finite TRAIN-only SafetyEP gradient diagnosis; zero optimizer updates.

Loads one sealed native policy while keeping the original complete IL reference.
Three fixed groups of eight TRAIN observations are sampled/scored normally.
Each true PG/physical-anchor gradient is measured before any optimizer exists.
The running training invocation and its state are never opened for writing.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist

from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.simwam_recipe import checkpoint
from starVLA.rl.simwam_recipe.lora_adapter import restore_adapters
from starVLA.rl.qwen_sourceflow.collection import collect
from starVLA.rl.simwam_recipe.contracts import seed_for
from starVLA.rl.qwen_world_research.runtime import ResearchRuntime
from starVLA.rl.qwen_world_research_algorithms.safety_efficiency_entry import SafetyEfficiencyRuntimeMixin
from starVLA.rl.qwen_world_research_algorithms.gradient_probe import measure_components
from starVLA.rl.qwen_world_research_algorithms.policy import ResearchLoss
from scripts.analysis.simwam_recipe.train import parameter_hash


class ProbeLoss(ResearchLoss):
    def record_policy_components(self, pg, anchor):
        self.parts={'pg':pg,'anchor':anchor}


def main():
    p=argparse.ArgumentParser(__doc__)
    for arg in ('config','manifest','output','il-checkpoint','parent-selection','data','rl-checkpoint','panel'):
        p.add_argument('--'+arg,required=True)
    a=p.parse_args();a.qualification=False
    class Runtime(SafetyEfficiencyRuntimeMixin,ResearchRuntime):
        pass
    runtime=Runtime();host=runtime.require_dispatch()
    configure_numerics();rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
    if world!=8:raise ValueError('This finite probe requires the native eight-rank reduction')
    local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local)
    dist.init_process_group('nccl',timeout=timedelta(minutes=20));checkpoint.init_control_group()
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    device=torch.device('cuda',local);root=Path(a.output)
    cfg=runtime.load_recipe(a);actor,_,_=runtime.make_actor(cfg,a,device)
    actor._loss=ProbeLoss(cfg);actor.eval()
    cp=Path(a.rl_checkpoint);seal=json.loads((cp/'COMPLETE').read_text())
    native=json.loads((cp/'identity.json').read_text());files=json.loads((cp/'files.json').read_text())
    from starVLA.rl.qwen_world_postcot.contracts import file_sha
    if (file_sha(cp/'files.json')!=seal['files_sha256'] or seal['world']!=8
            or native['recipe']['research']['arm']!='PPO_SAFETY_EP'
            or native['selected_parent_updates']!=70000
            or native['recipe']!=cfg):
        raise ValueError('Probe must bind the exact original SafetyEP recipe/native checkpoint')
    error=None
    if rank==0 and file_sha(cp/'optimizer.pt')!=files['optimizer.pt']:
        error='Native policy file changed'
    checkpoint.check_errors(error)
    payload=torch.load(cp/'optimizer.pt',map_location='cpu',weights_only=False)
    restore_adapters(actor,payload['adapters']);del payload
    policy_sha=parameter_hash(actor,trainable=True)
    all_policy=checkpoint.gather(policy_sha)
    if len(set(all_policy))!=1:raise ValueError('Probe policies differ across ranks')
    panel=json.loads(Path(a.panel).read_text());tokens=panel['tokens'][:24]
    if len(tokens)!=24 or len(set(tokens))!=24 or not set(tokens)<=set(runtime.conditions.data.tokens):
        raise ValueError('Probe requires 24 distinct legal TRAIN diagnostic tokens')
    generator=torch.Generator(device='cpu').manual_seed(seed_for('SafetyEP_component_probe',42,rank))
    service=runtime.make_reward_service();results=[];started=time.monotonic()
    if rank==0:
        root.mkdir(parents=True,exist_ok=False)
        atomic_json(root/'PROTOCOL.json',dict(schema='world_finite_SafetyEP_gradient_probe_v1',
            status='RUNNING',host=host,checkpoint=str(cp),native_updates=seal['updates'],
            native_files_sha256=seal['files_sha256'],policy_sha256=policy_sha,
            panel=str(a.panel),panel_sha256=file_sha(a.panel),tokens=tokens,
            selection='first24 from existing fixed TRAIN1024; no selection using NAVTEST',
            world=world,groups=24,candidates=192,optimizer_steps=0,
            original_running_training_untouched=True))
    checkpoint.check_errors(None)
    try:
        for case in range(3):
            runtime.require_dispatch();token=tokens[case*8+rank]
            rollout=collect(actor,runtime.conditions.get(token,device),token,runtime.conditions.data.logs[token],
                service,generator,variant=cfg['variant'],
                version=dict(run=native['sha256'],update=seal['updates'],adapter_sha256=policy_sha),recipe=cfg)
            runtime.finalize_advantages(rollout)
            actor.zero_grad(set_to_none=True)
            total,metrics=actor(rollout)
            parts=actor._loss.parts;actor._loss.parts=None
            report=measure_components(actor,parts)
            report.update(case=case,native_updates=seal['updates'],tokens=checkpoint.gather(token),
                candidate_material_by_rank=checkpoint.gather({
                    'rewards':rollout['rewards'].cpu().tolist(),
                    'advantages':rollout['advantages'].cpu().tolist(),
                    **{k:rollout['structured_advantage'][k].cpu().tolist()
                        for k in ('feasible','cost','A_safe','A_EP','A_raw')}}),
                loss_scalars_by_rank=checkpoint.gather({k:float(v) for k,v in metrics.items()}))
            checkpoint.atomic_torch(root/f'case{case}_rank{rank}.pt',rollout)
            if rank==0:atomic_json(root/f'case{case}.json',report)
            results.append(report);del total,parts,metrics,rollout
        if rank==0:
            atomic_json(root/'SUMMARY.json',dict(schema='world_finite_SafetyEP_gradient_probe_v1',status='COMPLETE',
                checkpoint=str(cp),native_updates=seal['updates'],groups=24,candidates=192,optimizer_steps=0,
                seconds=time.monotonic()-started,cases=[dict(case=x['case'],scopes=x['scopes']) for x in results],
                limitation='bounded fixed TRAIN diagnostic; no claim of population representativeness or causal training gain'))
    finally:
        service.close();dist.destroy_process_group()


if __name__=='__main__':main()
