"""Operator-allocated single-GPU full IL70 reference-cache equivalence probe.

Replays existing scored first-collection TRAIN material. No sampler, scorer,
new deployment result or live checkpoint modification. This is not world8 DDP
qualification; the operator must bind its PASS to a typed execution migration.
"""
import argparse
import copy
import json
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import torch

from starVLA.rl.simwam_recipe import checkpoint
from starVLA.rl.qwen_world_research_algorithms.policy import ResearchLoss
from starVLA.rl.qwen_world_research_algorithms import frozen_reference as C
from starVLA.rl.qwen_world_performance.batch_runtime import PerformanceRuntime
from starVLA.rl.flow_grpo.transactions import atomic_json


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state())


def restore_rng(value):
    random.setstate(value['python']);np.random.set_state(value['numpy']);torch.set_rng_state(value['cpu'])
    torch.cuda.set_rng_state(value['cuda'])


def trainables(actor):
    return {n:p.detach().cpu().clone() for n,p in actor.named_parameters() if p.requires_grad}


def restore_trainables(actor,value):
    with torch.no_grad():
        for n,p in actor.named_parameters():
            if p.requires_grad:p.copy_(value[n].to(p.device))


def main():
    parser=argparse.ArgumentParser(__doc__)
    for name in ('config','manifest','data','il-checkpoint','parent-selection','material','run-identity','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--maximum-scenes',type=int,default=1)
    args=parser.parse_args();args.qualification=True;args.qualification_force_auxiliary=False;args.resume=None
    if not 1<=args.maximum_scenes<=16:raise ValueError('Bounded1..16 existing TRAIN scenes only')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    identity=json.loads(Path(args.run_identity).read_text())
    rt=PerformanceRuntime();cfg=rt.load_recipe(args)
    if C.digest(cfg)!=C.digest(identity['recipe']):raise ValueError('Probe must use the exact existing registered cfg')
    torch.cuda.set_device(0);device=torch.device('cuda',0)
    actor,_,_=rt.make_actor(cfg,args,device)
    actor.record_sampling_terms=False;actor._loss=ResearchLoss(cfg)
    params=[p for p in actor.parameters() if p.requires_grad]
    if sum(p.numel() for p in params)!=52_707_840:raise ValueError('Real complete52.7M Scope required')
    if rt.parent['model_sha256']!=identity['IL_parent']['model_sha256']:raise ValueError('IL parent mismatch')
    material=torch.load(args.material,map_location='cpu',weights_only=False)
    leaves=material.get('samples',[material])[:args.maximum_scenes]
    if any(s['behavior_version']['update']!=0 or s['behavior_version']['run']!=identity['sha256'] for s in leaves):
        raise ValueError('Use existing first-collection IL70 material, not a different policy/version')
    for leaf in leaves:
        if C.KEY in leaf:raise ValueError('Baseline material must not already contain an unvalidated cache')
        if leaf['cond_bg']['identity']!=rt.conditions.sha:raise ValueError('Condition identity changed')
    initial=trainables(actor);initial_rng=rng_state();results={}
    # Warm frozen features and these exact tensor shapes before both paths.
    warm=checkpoint.device_tree(leaves[0],device)
    with torch.no_grad():
        actor.action_chain_logprobs(warm['cond_bg'],warm['chain'],warm['timesteps'],warm['deltas'],stoch_idx=warm['stoch_idx'],return_terms=True)
        C.reference_terms(actor,warm)
    del warm;torch.cuda.synchronize()
    for enabled in (False,True):
        restore_trainables(actor,initial);actor.zero_grad(set_to_none=True);restore_rng(initial_rng)
        local=checkpoint.device_tree(copy.deepcopy(leaves),device)
        optimizer=rt.make_optimizer(actor,cfg);scheduler=rt.make_scheduler(optimizer,cfg)
        exposures=material.get('valid_global_scenes',len(leaves))
        scheduler.set_exposures(exposures)
        if enabled:C.enable_reference_cache(actor,identity)
        else:C.disable_reference_cache(actor)
        original_material=C.digest(local)
        refs_before=actor._reference_calls;memory_before=actor.current_memory_builds
        torch.cuda.synchronize();start=time.monotonic()
        if enabled:
            for leaf in local:C.reference_terms(actor,leaf)
        torch.cuda.synchronize();cache_build=time.monotonic()-start
        rows=[];serialized_pending=False
        for inner in range(4):
            optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.monotonic()
            total=None;fresh_error=0.
            for leaf in local:
                value,_=actor._loss._policy_loss(actor,leaf)
                if inner==0:
                    current=actor._probability_audit['current']['source_clamped_logprob'].mean((-1,-2))
                    fresh_error=max(fresh_error,float(torch.expm1(current-leaf['old_logp']).abs().max()))
                    if fresh_error>1e-4:raise ValueError('Original first-collection fresh ratio tolerance failed')
                (value/len(local)).backward();total=value.detach()/len(local) if total is None else total+value.detach()/len(local)
            torch.cuda.synchronize();forward_backward_seconds=time.monotonic()-start
            gradient_sha=C.digest({n:p.grad for n,p in actor.named_parameters() if p.requires_grad})
            start=time.monotonic()
            norm=torch.nn.utils.clip_grad_norm_(params,cfg['gradient_clipping_effective'])
            used=[g['lr'] for g in optimizer.param_groups];optimizer.step();scheduler.step()
            torch.cuda.synchronize();optimizer_seconds=time.monotonic()-start
            row=dict(inner=inner+1,loss=float(total),gradient_sha256=gradient_sha,gradient_norm=float(norm),
                fresh_ratio_max_error=fresh_error if inner==0 else None,
                used_lrs=used,parameters_sha256=C.digest(trainables(actor)),optimizer_sha256=C.digest(optimizer.state_dict()),
                scheduler_sha256=C.digest(scheduler.state_dict()),RNG_sha256=C.digest(rng_state()),
                forward_backward_seconds=forward_backward_seconds,optimizer_seconds=optimizer_seconds)
            rows.append(row)
            if enabled and inner==1:
                # Same tensor cpu_tree/device_tree as native pending material;
                # scratch is unlinked on close, not a formal checkpoint.
                with tempfile.TemporaryFile(dir=output) as f:
                    torch.save(dict(trainables=trainables(actor),optimizer=checkpoint.cpu_tree(optimizer.state_dict()),
                        scheduler=scheduler.state_dict(),RNG=rng_state(),leaves=checkpoint.cpu_tree(local)),f)
                    f.seek(0);saved=torch.load(f,map_location='cpu',weights_only=False)
                restore_trainables(actor,saved['trainables']);optimizer.load_state_dict(saved['optimizer'])
                scheduler.load_state_dict(saved['scheduler']);restore_rng(saved['RNG'])
                local=checkpoint.device_tree(saved['leaves'],device)
                for leaf in local:C.validate_reference_cache(leaf,identity)
                del saved;serialized_pending=True
        unchanged=C.digest([{k:v for k,v in leaf.items() if k!=C.KEY} for leaf in local])==original_material
        results['cached' if enabled else 'original']=dict(rows=rows,reference_velocity_calls=actor._reference_calls-refs_before,
            current_memory_builds=actor.current_memory_builds-memory_before,cache_build_seconds=cache_build,
            compute_seconds_including_cache=cache_build+sum(r['forward_backward_seconds']+r['optimizer_seconds'] for r in rows),
            material_unchanged_except_cache=unchanged,pending_serialization_roundtrip=serialized_pending)
        del optimizer,scheduler,local
    fields=('loss','gradient_sha256','gradient_norm','used_lrs','parameters_sha256','optimizer_sha256','scheduler_sha256','RNG_sha256','fresh_ratio_max_error')
    equal=all(a[k]==b[k] for a,b in zip(results['original']['rows'],results['cached']['rows']) for k in fields)
    want=3*len(leaves)
    passed=(equal and all(r['material_unchanged_except_cache'] for r in results.values())
        and results['original']['reference_velocity_calls']==4*want and results['cached']['reference_velocity_calls']==want
        and results['original']['current_memory_builds']==results['cached']['current_memory_builds']==4*len(leaves))
    proof=dict(status='PASS' if passed else 'FAIL',schema='world_real_full_IL70_reference_cache_singleGPU_four_inner_v1',
        real_full_IL70=True,formal_training=False,new_generation=0,new_scoring=0,optimizer_updates_per_path=4,
        trainable_numel=sum(p.numel() for p in params),group_size=leaves[0]['chain'].shape[0],TRAIN_scenes=len(leaves),
        identity=str(Path(args.run_identity).resolve()),material=str(Path(args.material).resolve()),
        exact_equal_fields=list(fields),all_four_inner_bitexact=equal,results=results,
        peak_GPU_bytes=torch.cuda.max_memory_allocated(),
        limits='SingleGPU completeScope, same saved first collection; not world8DDP or formal driving-score evidence. Sequential bounded timings exclude digest/CPU-transfer/pending-serialization overhead, not a long-run throughput claim.')
    atomic_json(output/'COMPLETE.json',proof)
    if not passed:raise RuntimeError('Full reference-cache equivalence failed; no execution migration allowed')
    print(json.dumps(dict(status=proof['status'],output=str(output/'COMPLETE.json'))))


if __name__=='__main__':main()
