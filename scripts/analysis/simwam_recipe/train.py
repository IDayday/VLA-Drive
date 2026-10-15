"""Two locked source-recipe variants, four real updates per actual fresh chain.

Run with torchrun. Qualification is a separate run identity; its weights can
never initialize a formal run. SIGUSR1 requests an exact inner-boundary save.
"""
import argparse
from datetime import timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import time
import traceback
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from starVLA.rl.simwam_recipe.contracts import *
from starVLA.rl.simwam_recipe.trainer_adapter import make_actor
from starVLA.rl.simwam_recipe.condition import ProjectedConditions
from starVLA.rl.simwam_recipe.reward_readouts import RewardService
from starVLA.rl.simwam_recipe.collection import collect,finalize_advantages,exploration_summary
from starVLA.rl.simwam_recipe.data_stream import SceneStream
from starVLA.rl.simwam_recipe import checkpoint
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.transactions import atomic_json

STOP=False
def stop(*_):
    global STOP
    STOP=True


def all_ok(error,device):
    flag=torch.tensor(int(error is not None),device=device)
    if dist.is_initialized():dist.all_reduce(flag,op=dist.ReduceOp.MAX)
    return not bool(flag)


def stop_requested(device):
    flag=torch.tensor(int(STOP),device=device)
    if dist.is_initialized():dist.all_reduce(flag,op=dist.ReduceOp.MAX)
    return bool(flag)


def parameter_hash(actor,*,trainable):
    h=hashlib.sha256()
    for n,p in actor.named_parameters():
        if p.requires_grad==trainable:
            h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def source_files():
    roots=['starVLA/rl/simwam_recipe']
    paths=[p for r in roots for p in Path(r).rglob('*.py')]
    paths += [Path(p) for p in ['scripts/analysis/simwam_recipe/train.py','starVLA/model/modules/action_model/GR00T_ActionHeader.py',
        'starVLA/model/modules/action_model/flow_matching_head/cross_attention_dit.py',
        'starVLA/rl/constrained_flow_noise/loading.py','scripts/analysis/constrained_flow_noise/batched_canonical_score.py']]
    return {str(p):file_sha(p) for p in sorted(paths)}


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--config',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--resume');p.add_argument('--qualification',action='store_true')
    p.add_argument('--stop-after-update',type=int);p.add_argument('--save-every',type=int)
    a=p.parse_args();cfg=load_recipe(a.config);host=require_dispatch();configure_numerics()
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    if world!=8 and not a.qualification:raise ValueError('Formal recipe requires eight GPUs')
    torch.cuda.set_device(local)
    if world>1:dist.init_process_group('nccl',timeout=timedelta(minutes=20))
    device=torch.device('cuda',local)
    for s in (signal.SIGUSR1,signal.SIGTERM):signal.signal(s,stop)
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    lock=(root/f'rank{rank}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stream=SceneStream(a.manifest,world,qualification=a.qualification)
    actor,oldcfg,sft=make_actor(cfg,device=device)
    conditions=ProjectedConditions(Path(ROOT)/'projected_condition',actor.action_expert)
    service=RewardService(Path.cwd(),workers=1)
    trainable=[p for p in actor.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(trainable,lr=cfg['learning_rate'],betas=tuple(cfg['optimizer_betas']),eps=cfg['optimizer_eps'],weight_decay=cfg['weight_decay'])
    actor._loss.optimizer=optimizer;actor._loss.learning_rate=cfg['learning_rate']
    warmup=min(int(cfg['max_steps']*.05),cfg['lr_warmup_steps'])
    scheduler=actor._loss._build_scheduler(cfg['lr_scheduler_type'],cfg['max_steps'],warmup)
    generator=torch.Generator(device='cpu').manual_seed(seed_for('simwam_source_rank_rollout',42,rank))
    identity=recipe_identity(cfg,a.manifest,conditions.sha)
    identity.update(world=world,purpose='qualification_not_formal_initialization' if a.qualification else 'formal',
                    score_protocol=service.protocol,source_files=source_files(),parameter_contract=actor.closure,
                    rng_protocol='independent persistent CPU FP32 generator per rank; source _randn; no repeated candidate seeds',
                    scene_reduction='source local effective mean, then equal mean across ranks',
                    actual_optimizer=dict(name='torch.optim.AdamW',lr=cfg['learning_rate'],betas=cfg['optimizer_betas'],
                                          eps=cfg['optimizer_eps'],weight_decay=cfg['weight_decay'],warmup_updates=warmup,
                                          max_grad_norm_effective=cfg['gradient_clipping_effective'],FP32_master_and_moments=True))
    identity['sha256']=fingerprint(identity)
    if rank==0:
        if (root/'identity.json').exists():
            if json.loads((root/'identity.json').read_text())!=identity:raise ValueError('Existing run identity changed')
            if not a.resume:raise ValueError('Existing run requires explicit native resume')
        else:
            if a.resume:raise ValueError('Resume must preserve the original run directory')
            atomic_json(root/'identity.json',identity)
            for rel in identity['source_files']:
                dest=root/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(rel,dest)
            shutil.copyfile(a.config,root/'resolved_recipe.yaml')
    if world>1:dist.barrier()
    # PyTorch rebuilds reduction buckets after the first backward when unused
    # detection is disabled. Restarting between inner updates repeats that
    # transition and changes FP32 reduction order. The installed Reducer
    # explicitly does not rebuild with find_unused=True/static_graph=False.
    # All 192 gradients are still required below; this does not allow omissions.
    model=DistributedDataParallel(actor,device_ids=[local],broadcast_buffers=False,
        find_unused_parameters=cfg['DDP_find_unused_parameters'],static_graph=cfg['DDP_static_graph'],
        bucket_cap_mb=cfg['DDP_bucket_cap_mb']) if world>1 else actor
    model.eval()
    state=dict(updates=0,collections=0,raw_scene_exposures=0,new_candidates=0,next_inner=0,
               RL_updates_local=0,anchor_only_updates_local=0,scene_cursor=0,
               seconds_rollout=0.,seconds_scoring_wait=0.,seconds_scoring_cpu=0.,seconds_old_probability=0.,seconds_update=0.)
    rollout=None
    if a.resume:state,rollout=checkpoint.restore(a.resume,actor,optimizer,scheduler,generator,identity)
    frozen_before=parameter_hash(actor,trainable=False) if a.qualification else None
    first_adapters=checkpoint.cpu_tree({n:p.clone() for n,p in actor.named_parameters() if p.requires_grad}) if a.qualification else None
    started=time.monotonic();save_every=a.save_every or cfg['save_every_optimizer_updates'];last_saved=None
    journal=(root/f'metrics_rank{rank}.jsonl').open('a',buffering=1)
    try:
        while state['updates']<cfg['max_steps']:
            require_dispatch()
            if rollout is None:
                token,position=stream.at(state['scene_cursor'],rank)
                error=None
                try:
                    version=dict(run=identity['sha256'],update=state['updates'],adapter_sha256=parameter_hash(actor,trainable=True))
                    rollout=collect(actor,conditions.get(token,device),token,stream.logs[token],service,generator,
                                    act_norm=int(sft.datasets.vla_data.act_norm),variant=cfg['variant'],version=version)
                    rollout['stream_position']=position
                except Exception:
                    error=traceback.format_exc();atomic_json(root/f'failure_rank{rank}_{state["updates"]}.json',dict(error=error,state=state,token=token))
                if not all_ok(error,device):
                    checkpoint.save(root,actor,optimizer,scheduler,generator,state,rollout,identity,tag=f'failure{state["updates"]:06}')
                    raise RuntimeError('At least one rank failed actual sampling/scoring; no zero replacement or update')
                finalize_advantages(rollout)
                state['collections']+=1;state['raw_scene_exposures']+=world;state['new_candidates']+=world*8;state['scene_cursor']+=world
                for k,v in rollout['seconds'].items():state['seconds_'+k]+=v
                state['next_inner']=0
                # Every fresh chain remains replayable, including negative candidates.
                checkpoint.atomic_torch(root/'behavior'/f'collection{state["collections"]:06}'/f'rank{rank}.pt',rollout)
            optimizer.zero_grad(set_to_none=True)
            begin=time.monotonic();error=None
            try:total,metrics=model(rollout)
            except Exception:
                error=traceback.format_exc()
                atomic_json(root/f'forward_failure_rank{rank}_{state["updates"]}.json',dict(error=error,state=state))
            if not all_ok(error,device):
                checkpoint.save(root,actor,optimizer,scheduler,generator,state,rollout,identity,tag=f'forward_failure{state["updates"]:06}')
                raise RuntimeError('At least one rank failed source loss; native state saved without updating')
            current=actor._probability_audit['current'];old=rollout['old_logp']
            ratio=(current['source_clamped_logprob'].mean((-1,-2))-old).exp()
            fresh_error=float((ratio-1).abs().max())
            error=None
            if state['next_inner']==0 and fresh_error>1e-4:error='Fresh saved-state ratio differs by '+str(fresh_error)
            if not all_ok(error,device):
                checkpoint.save(root,actor,optimizer,scheduler,generator,state,rollout,identity,tag=f'ratio_failure{state["updates"]:06}')
                raise ValueError('Fresh probability numerical error exceeds 1e-4; no rewritten old density')
            total.backward()
            grads=torch.stack([p.grad.float().norm() for p in trainable if p.grad is not None])
            grad_norm=float(grads.norm())
            error=None if np.isfinite(grad_norm) and len(grads)==len(trainable) else 'Missing/nonfinite adapter gradient'
            if not all_ok(error,device):
                checkpoint.save(root,actor,optimizer,scheduler,generator,state,rollout,identity,tag=f'gradient_failure{state["updates"]:06}')
                raise ValueError('Invalid LoRA gradient; native state saved and update aborted')
            if cfg['gradient_clipping_effective']>0:torch.nn.utils.clip_grad_norm_(trainable,cfg['gradient_clipping_effective'])
            lr=optimizer.param_groups[0]['lr'];optimizer.step();scheduler.step()
            torch.cuda.synchronize();elapsed=time.monotonic()-begin
            state['updates']+=1;state['next_inner']+=1;state['seconds_update']+=elapsed
            state['RL_updates_local']+=int(bool(rollout['adv_mask'].any()))
            state['anchor_only_updates_local']+=int(not bool(rollout['adv_mask'].any()))
            row=dict(updates=state['updates'],collection=state['collections'],inner_completed=state['next_inner'],
                     raw_scene_exposures=state['raw_scene_exposures'],new_candidates=state['new_candidates'],
                     token=rollout['token'],behavior=rollout['behavior_version'],lr=lr,loss=float(total.detach()),
                     gradient_norm=grad_norm,ratio_max_abs_from_one=fresh_error,
                     source_clamp_hit_rate=current['source_clamp_hit_rate'],seconds_update=elapsed,
                     metrics={k:float(v) for k,v in metrics.items()},exploration=exploration_summary(rollout))
            journal.write(json.dumps(row,allow_nan=False)+'\n')
            if state['next_inner']==4:rollout=None;state['next_inner']=0
            stop_now=stop_requested(device) or (a.stop_after_update is not None and state['updates']>=a.stop_after_update)
            milestone=state['updates'] in cfg['evaluate_updates']
            if state['updates']%save_every==0 or milestone or stop_now or state['updates']==cfg['max_steps']:
                last_saved=checkpoint.save(root,actor,optimizer,scheduler,generator,state,rollout,identity)
                if rank==0:atomic_json(root/'progress.json',dict(state,status='PAUSED' if stop_now else 'RUNNING',checkpoint=last_saved,host=host,wall_seconds_this_invocation=time.monotonic()-started))
            if stop_now:break
        result=dict(state,status='COMPLETE' if state['updates']==cfg['max_steps'] else 'STOPPED_AT_SAVED_BOUNDARY',
                    host=host,rank=rank,checkpoint=last_saved,wall_seconds_this_invocation=time.monotonic()-started,
                    peak_GPU_bytes=torch.cuda.max_memory_allocated(),qualification=a.qualification)
        if world>1:
            data=model._get_ddp_logging_data()
            result['DDP_bucket_profile']={k:data.get(k) for k in ('bucket_sizes','has_rebuilt_buckets','find_unused_parameters','num_parameter_tensors','iteration')}
            if data.get('has_rebuilt_buckets',0):raise ValueError('Unexpected DDP bucket rebuild breaks registered recovery layout')
        if a.qualification:
            after=parameter_hash(actor,trainable=False)
            if after!=frozen_before:raise ValueError('Frozen SFT/reference state changed')
            changed={n:bool(torch.count_nonzero(p.detach().cpu()-first_adapters[n])) for n,p in actor.named_parameters() if p.requires_grad}
            result.update(frozen_SHA_before=frozen_before,frozen_SHA_after=after,changed_adapters=changed)
        atomic_json(root/f'result_rank{rank}.json',result)
    finally:
        journal.close();service.close()
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
