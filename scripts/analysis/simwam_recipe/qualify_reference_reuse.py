"""Bounded cache-contract tests, or real eight-GPU saved-chain parity/profile.

No new trajectories or scores. Qualification updates never initialize a run.
"""
import argparse
import copy
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.simwam_recipe.reference_reuse import FrozenReferenceReuse


def digest(value):
    h=hashlib.sha256()
    def walk(x):
        if torch.is_tensor(x):
            a=x.detach().cpu().contiguous();h.update(str((a.dtype,tuple(a.shape))).encode());h.update(a.numpy().tobytes())
        elif isinstance(x,np.ndarray):h.update(str((x.dtype,x.shape)).encode());h.update(x.tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());walk(x[k])
        elif isinstance(x,(list,tuple)):
            h.update(type(x).__name__.encode())
            for a in x:walk(a)
        else:h.update(repr(x).encode())
    walk(value);return h.hexdigest()


def cpu(output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    checks=[];ref=torch.nn.Sequential(torch.nn.Linear(4,4),torch.nn.Tanh()).eval().requires_grad_(False)
    ref.register_buffer('marker',torch.zeros(1));cache=FrozenReferenceReuse();x=torch.arange(8.).reshape(2,4)
    inputs={'x':x,'time':torch.tensor([.2,.3]),'indices':torch.tensor([7,8,9])}
    def produce():return {'mean':ref(inputs['x']),'std':torch.ones(2,4)}
    with torch.no_grad():
        before=torch.get_rng_state();a=cache.get(ref,inputs,{'std':.1},produce)
        for _ in range(3):assert torch.equal(cache.get(ref,inputs,{'std':.1},produce)['mean'],a['mean'])
        assert (cache.misses,cache.hits)==(1,3);assert torch.equal(before,torch.get_rng_state())
        checks += ['four_inner_one_reference_evaluation','exact_output_reuse','no_RNG_consumption']
        for name,mutate in [
            ('chain_version',lambda:inputs['x'].add_(.1)),
            ('time_version',lambda:inputs['time'].add_(.1)),
            ('window_version',lambda:inputs['indices'].sub_(1)),
            ('reference_parameter_version',lambda:next(ref.parameters()).add_(.1)),
            ('reference_buffer_version',lambda:ref.marker.add_(1)),
            ('new_tensor_identity',lambda:inputs.update(x=inputs['x'].clone())),
            ('input_layout',lambda:inputs.update(x=inputs['x'].t().contiguous().t())),
        ]:
            n=cache.misses;mutate();v=cache.get(ref,inputs,{'std':.1},produce)
            assert cache.misses==n+1 and torch.equal(v['mean'],produce()['mean']);checks.append(name+'_invalidates')
        n=cache.misses;cache.get(ref,inputs,{'std':.2},produce);assert cache.misses==n+1;checks.append('sampler_change_invalidates')
        cache.value['mean'].add_(1)
        try:cache.get(ref,inputs,{'std':.2},produce)
        except ValueError:checks.append('mutated_output_rejected')
        else:raise AssertionError('Mutated output accepted')
        cache.clear();v=cache.get(ref,inputs,{'std':.2},produce);assert torch.equal(v['mean'],produce()['mean']);checks.append('cold_restart_recomputes_exactly')
        ref.train()
        try:cache.get(ref,inputs,{'std':.2},produce)
        except ValueError:checks.append('training_mode_rejected')
        else:raise AssertionError('Training reference reused')
        ref.eval();next(ref.parameters()).requires_grad_(True)
        try:cache.get(ref,inputs,{'std':.2},produce)
        except ValueError:checks.append('trainable_reference_rejected')
        else:raise AssertionError('Trainable reference reused')
        next(ref.parameters()).requires_grad_(False)
    try:cache.get(ref,inputs,{'std':.2},produce)
    except ValueError:checks.append('autograd_context_rejected')
    else:raise AssertionError('Autograd reference reused')
    v=dict(status='PASS',checks=checks,new_candidates=0,new_scores=0)
    atomic_json(output,v);print(json.dumps(v))


def real(a):
    from starVLA.rl.flow_grpo.reproducibility import configure_numerics
    from starVLA.rl.simwam_recipe import checkpoint
    from starVLA.rl.simwam_recipe.contracts import require_dispatch,file_sha,registered_stochastic_steps
    from starVLA.rl.simwam_recipe.lora_adapter import adapter_state
    from starVLA.rl.simwam_recipe.recovery_state import rollout_digest
    from starVLA.rl.simwam_recipe.trainer_adapter import make_actor
    started=time.monotonic();host=require_dispatch();configure_numerics()
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    if world!=8:raise ValueError('Qualification must use the actual world8 reduction')
    torch.cuda.set_device(local);dist.init_process_group('nccl',timeout=timedelta(minutes=10));checkpoint.init_control_group()
    out=Path(a.output)
    if rank==0:out.mkdir(parents=True,exist_ok=False)
    dist.barrier();run=Path(a.run);identity=json.loads((run/'identity.json').read_text());cfg=identity['recipe']
    actor,_,_=make_actor(cfg,device=torch.device('cuda',local));trainable=[p for p in actor.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(trainable,lr=cfg['learning_rate'],betas=tuple(cfg['optimizer_betas']),eps=cfg['optimizer_eps'],weight_decay=cfg['weight_decay'])
    actor._loss.optimizer=optimizer;actor._loss.learning_rate=cfg['learning_rate']
    scheduler=actor._loss._build_scheduler(cfg['lr_scheduler_type'],cfg['max_steps'],min(int(cfg['max_steps']*.05),cfg['lr_warmup_steps']))
    generator=torch.Generator(device='cpu')
    model=DistributedDataParallel(actor,device_ids=[local],broadcast_buffers=False,
        find_unused_parameters=cfg['DDP_find_unused_parameters'],static_graph=cfg['DDP_static_graph'],bucket_cap_mb=cfg['DDP_bucket_cap_mb'])
    model.eval()
    frozen_before=digest({n:p for n,p in actor.named_parameters() if not p.requires_grad})
    records=[];snapshots={};private_hashes={}
    # Zero LoRA is checked once per mode; nonzero LoRA has an AB/BA order and
    # a cold-cache path. Only the AB/BA runs are used for timing comparisons.
    trials=[('zero_uncached',0,False,1,False),('zero_reused',0,True,1,False),
            ('AB_uncached',a.start_update,False,2,False),('AB_reused',a.start_update,True,2,False),
            ('BA_reused',a.start_update,True,2,False),('BA_uncached',a.start_update,False,2,False),
            ('cold_reused',a.start_update,True,2,True)]
    for label,start,reuse,ncollections,cold in trials:
        cp=run/'checkpoints'/f'update{start:06}_inner0'
        actor.enable_frozen_reference_reuse(reuse)
        state,pending=checkpoint.restore(cp,actor,optimizer,scheduler,generator,identity)
        if pending is not None or state['updates']!=start or state['next_inner']!=0:raise ValueError('Need a sealed READY boundary')
        baseline_adapter=digest(adapter_state(actor));ref_start=actor._reference_calls;current_start=actor._velocity_calls
        step_records=[];times=[];cache_clears=0
        for c in range(start//4+1,start//4+ncollections+1):
            p=run/'behavior'/f'collection{c:06}'/f'rank{rank}.pt'
            row=torch.load(p,map_location='cpu',weights_only=False)
            if row['behavior_version']['run']!=identity['sha256'] or row['behavior_version']['update']!=(c-1)*4:
                raise ValueError('Invalid saved behavior version')
            if row['behavior_version']['adapter_sha256']!=__import__('scripts.analysis.simwam_recipe.train',fromlist=['parameter_hash']).parameter_hash(actor,trainable=True):
                raise ValueError('Replay did not reproduce the original pre-collection adapter')
            if row['stoch_idx'].tolist()!=registered_stochastic_steps(cfg['variant']):raise ValueError('Changed stochastic window')
            if 'chain_identity' not in row or row['chain_identity']!=rollout_digest(row):raise ValueError('Changed committed chain')
            private_hashes[str(p)]=file_sha(p)
            row=checkpoint.device_tree(row,actor.device)
            for inner in range(4):
                if cold and inner==2:actor._frozen_reference_reuse.clear();cache_clears+=1
                optimizer.zero_grad(set_to_none=True);dist.barrier();torch.cuda.synchronize();t=time.monotonic()
                total,metrics=model(row);total.backward()
                grads=[p.grad for p in trainable]
                if len(grads)!=192 or any(g is None for g in grads):raise ValueError('Missing LoRA gradient')
                grad_norm=float(torch.stack([g.float().norm() for g in grads]).norm())
                if not np.isfinite(grad_norm):raise ValueError('Nonfinite LoRA gradient')
                optimizer.step();scheduler.step();torch.cuda.synchronize();end=time.monotonic()
                # AdamW does not mutate .grad. Evidence copies happen after
                # timing, before the next zero_grad, on both compared paths.
                times.append(end-t)
                gradient_sha=digest(grads)
                current=actor._probability_audit['current'];ratio=(current['source_clamped_logprob'].mean((-1,-2))-row['old_logp']).exp()
                if inner==0 and float((ratio-1).abs().max())>1e-4:raise ValueError('Fresh saved-state ratio is not on-policy')
                step_records.append(dict(collection=c,inner=inner+1,loss=float(total),metrics={k:float(v) for k,v in metrics.items()},
                    gradient_norm=grad_norm,gradient_sha256=gradient_sha,adapters_sha256=digest(adapter_state(actor)),ratio_sha256=digest(ratio)))
        snapshot=dict(adapters=checkpoint.cpu_tree(adapter_state(actor)),optimizer=checkpoint.cpu_tree(optimizer.state_dict()),scheduler=scheduler.state_dict(),
                      RNG=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),generator=generator.get_state()))
        key='zero' if start==0 else 'nonzero'
        if key not in snapshots:snapshots[key]=(digest(snapshot),step_records)
        else:
            expected,steps=snapshots[key]
            if digest(snapshot)!=expected or steps!=step_records:raise ValueError('Reference reuse changed actual gradients, updates, metrics or RNG')
        expected_reference=ncollections*3*(4 if not reuse else 2 if cold else 1)
        if actor._reference_calls-ref_start!=expected_reference or actor._velocity_calls-current_start!=ncollections*12:
            raise ValueError('Unexpected velocity/reference computation count')
        records.append(dict(trial=label,start_update=start,reuse=reuse,collections=ncollections,optimizer_updates=ncollections*4,
            reference_velocity_calls=actor._reference_calls-ref_start,current_velocity_calls=actor._velocity_calls-current_start,
            update_seconds=times,cache_clears=cache_clears,final_state_sha256=digest(snapshot),step_evidence_sha256=digest(step_records),
            initial_adapters_sha256=baseline_adapter))
    if digest({n:p for n,p in actor.named_parameters() if not p.requires_grad})!=frozen_before:raise ValueError('Frozen parameters changed')
    atomic_json(out/f'rank{rank}.json',dict(status='PASS',rank=rank,host=host,run_identity=identity['sha256'],variant=cfg['variant'],
        trials=records,saved_files_sha256=private_hashes,frozen_before_after_sha256=frozen_before,
        new_candidates=0,new_scores=0,qualification_optimizer_updates=sum(x['optimizer_updates'] for x in records),
        elapsed=time.monotonic()-started,peak_torch_GPU_bytes=torch.cuda.max_memory_allocated(),analysis_source_sha256=file_sha(__file__),
        limitations=['Saved-chain replay qualification; no policy efficacy claim.','Cold cache clear checks ephemeral-state semantics, not a new process-level fault injection.','All qualification weights are discarded; no formal source migration performed.']))
    dist.barrier()
    if rank==0:
        ranks=[json.loads((out/f'rank{i}.json').read_text()) for i in range(8)]
        summary=dict(status='PASS_REAL_WORLD8_REFERENCE_REUSE',variant=cfg['variant'],rank_count=8,
                     exact_loss_metrics_gradients_parameters_optimizer_scheduler_RNG=True,
                     original_behavior_adapter_reproduced=True,unchanged_frozen_parameters=True,
                     new_candidates=0,new_scores=0,logical_qualification_updates=sum(x['optimizer_updates'] for x in records),
                     cumulative_worker_wall_seconds=sum(r['elapsed'] for r in ranks),ranks=[{k:r[k] for k in ('rank','elapsed','peak_torch_GPU_bytes','trials')} for r in ranks])
        atomic_json(out/'summary.json',summary);print(json.dumps({k:summary[k] for k in ('status','variant','rank_count','logical_qualification_updates','new_scores')}))
    dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser(__doc__);s=p.add_subparsers(dest='command',required=True)
    c=s.add_parser('cpu');c.add_argument('--output',required=True)
    r=s.add_parser('real');r.add_argument('--run',required=True);r.add_argument('--start-update',type=int,required=True);r.add_argument('--output',required=True)
    a=p.parse_args()
    if a.command=='cpu':cpu(a.output)
    else:real(a)


if __name__=='__main__':main()
