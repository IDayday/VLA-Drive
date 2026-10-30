"""Independent World NFT learner; root runtime owns parent/data authorization.

Run under torchrun with --runtime-factory module:Class, the same --config,
--manifest, --data, --il-checkpoint and --parent-selection interfaces as the
research Runtime. Checkpoints are deployment-compatible World scope tensors,
but exact recovery uses the explicit NFT EMA/forward-process transaction.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import fcntl
import importlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import time
import traceback
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.simwam_recipe import checkpoint as shared
from starVLA.rl.simwam_recipe.source_advantages import advantages
from starVLA.rl.simwam_recipe.reward_readouts import RewardService
from starVLA.rl.simwam_recipe.invocation import Invocation
from starVLA.rl.simwam_recipe.contracts import seed_for
from starVLA.rl.qwen_world_postcot.contracts import fingerprint,file_sha
from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
from scripts.analysis.simwam_recipe.train import parameter_hash,all_ok

from .nft import BehaviorEMA,NFTLoss,make_forward_batch,optimality_probabilities,trainable_parameters
from . import nft_checkpoint

STOP = False


def request_stop(*_):
    global STOP
    STOP = True


def should_stop(device):
    flag=torch.tensor(int(STOP),device=device)
    if dist.is_initialized():dist.all_reduce(flag,op=dist.ReduceOp.MAX)
    return bool(flag)


@torch.no_grad()
def collect_ode(actor,conditions,stream,token,position,ema,service,generator,identity,progress=None):
    condition=conditions.get(token,actor.device)
    started=time.monotonic()
    with ema.use(actor):
        chain,timesteps,deltas,indices=actor.sample_action_chain(
            condition,8,deterministic=True,generator=generator)
    if chain.shape!=(8,11,8,3) or indices.numel()!=0 or not bool(torch.isfinite(chain).all()):
        raise ValueError('NFT collection must be eight complete pure-ODE10 trajectories')
    torch.cuda.synchronize(actor.device)
    rollout_seconds=time.monotonic()-started
    if progress:progress('generated',rollout_seconds,8)
    clean=chain[:,-1].detach()
    requested=ActionCodec.denorm_odo(clean).cpu().numpy()
    started=time.monotonic()
    records=service.score((token,stream.logs[token],requested))
    scoring_seconds=time.monotonic()-started
    if progress:progress('scored',scoring_seconds,8)
    if len(records)!=8:
        raise ValueError('NFT scorer must return every generated candidate')
    rewards=torch.tensor([[r['readouts']['canonical_PDMS'] for r in records]],
                         device=actor.device,dtype=torch.float32)
    if not bool(torch.isfinite(rewards).all()):
        raise ValueError('Infrastructure/scoring failure cannot become zero reward')
    return dict(schema='world_nft_ODE_collection_v1',cond_bg=condition,clean_actions=clean,
        rewards=rewards,records=records,token=token,log=stream.logs[token],
        stream_position=position,ema_version=ema.updates,run=identity['sha256'],
        sampling=dict(ODE_steps=10,shift=5.,SDE_steps=0,root_noise='persistent CPU normal; FP32'),
        seconds=dict(rollout=rollout_seconds,scoring_wait=scoring_seconds,
                     scoring_cpu=sum(r['scoring_seconds'] for r in records),old_probability=0.))


def parse_arguments():
    initial=argparse.ArgumentParser(add_help=False)
    initial.add_argument('--runtime-factory',default='starVLA.rl.qwen_world_research.runtime:ResearchRuntime')
    known,_=initial.parse_known_args()
    module,name=known.runtime_factory.split(':',1)
    runtime=getattr(importlib.import_module(module),name)()
    p=argparse.ArgumentParser(__doc__,parents=[initial])
    p.add_argument('--config',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--output',required=True);p.add_argument('--resume')
    p.add_argument('--qualification',action='store_true')
    p.add_argument('--stop-after-update',type=int);p.add_argument('--save-every',type=int)
    runtime.add_arguments(p)
    return p.parse_args(),runtime


def main():
    global STOP
    STOP=False
    started_process=time.monotonic();args,runtime=parse_arguments()
    cfg=runtime.load_recipe(args)
    if cfg['research'].get('policy_objective')!='nft' or cfg['grpo']['train']['kl_beta']!=0:
        raise ValueError('NFT entry requires the independent registered no-IL-anchor NFT recipe')
    if cfg['grpo']['sample']['group_size']!=8 or cfg['grpo']['train']['num_inner_epochs']!=4:
        raise ValueError('First NFT comparison preserves G8 and four inner updates')
    host=runtime.require_dispatch();configure_numerics()
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    if world!=8 and not args.qualification:
        raise ValueError('Formal first-family comparison requires eight ranks')
    torch.cuda.set_device(local)
    device=torch.device('cuda',local)
    if world>1:dist.init_process_group('nccl',timeout=timedelta(minutes=20))
    shared.init_control_group()
    for signum in (signal.SIGUSR1,signal.SIGTERM):signal.signal(signum,request_stop)
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    lock=(root/f'rank{rank}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stream=runtime.make_scene_stream(cfg,args,world)
    actor,_,_=runtime.make_actor(cfg,args,device)
    actor._loss=NFTLoss(cfg)
    conditions=runtime.make_conditions(actor)
    optimizer=runtime.make_optimizer(actor,cfg)
    actor._loss.optimizer=optimizer;actor._loss.learning_rate=cfg['learning_rate']
    warmup=min(int(cfg['max_steps']*.05),cfg['lr_warmup_steps'])
    scheduler=actor._loss._build_scheduler(cfg['lr_scheduler_type'],cfg['max_steps'],warmup)
    ema=BehaviorEMA(actor,decay=cfg['research'].get('nft_ema_decay',.9))
    generator=torch.Generator(device='cpu').manual_seed(seed_for('simwam_source_rank_rollout',cfg['seed'],rank))
    service=(runtime.make_reward_service() if hasattr(runtime,'make_reward_service')
             else RewardService(Path.cwd(),workers=1))
    identity=runtime.recipe_identity(cfg,args,conditions.sha)
    sources=runtime.source_files()
    for p in Path(__file__).parent.glob('*.py'):sources[str(p.relative_to(Path.cwd()))]=file_sha(p)
    identity.update(schema='world_postcot_cleanx_scope_RL_v1',world=world,
        training_protocol='world_nft_forward_v1',policy_ratio='none; positive/negative forward-process flow matching',
        purpose='qualification_not_formal_initialization' if args.qualification else 'formal',
        source_files=dict(sorted(sources.items())),score_protocol=service.protocol,parameter_contract=actor.closure,
        research_protocol=dict(algorithm='DiffusionNFT Eq5',behavior='EMA of all current trainables; frozen Qwen',
            EMA_decay=ema.decay,EMA_update='once after each complete four-inner collection',
            rollout='pure ODE10 shift5; eight independent normal root noises',
            time_sampling='uniform sigma on [nft_sigma_min,1), independent every inner update',
            velocity_target='epsilon-clean_actions; clean_x_to_velocity=(state-pred_x0)/sigma',
            weighting=cfg['research'].get('nft_loss','velocity_mse'),
            explicit_IL_anchor=False,no_learned_critic=True,no_SDE_likelihood=True),
        rng_protocol='persistent per-rank CPU FP32 generator for ODE roots and each forward-process noise/sigma',
        scene_reduction='equal mean of eight candidates per rank, then DDP rank mean',
        actual_optimizer=dict(name='torch.optim.AdamW',parameter_groups=[dict(name=g.get('name'),lr=g['lr'],
            tensors=len(g['params']),numel=sum(p.numel() for p in g['params'])) for g in optimizer.param_groups],
            warmup_updates=warmup,max_grad_norm=cfg['gradient_clipping_effective']))
    if hasattr(stream,'identity'):identity['data_stream_identity']=stream.identity
    identity['sha256']=fingerprint(identity)
    error=None
    try:
        if rank==0:
            if (root/'identity.json').exists():
                if not args.resume or json.loads((root/'identity.json').read_text())!=identity:
                    raise ValueError('NFT run exists; exact native resume of the same identity is required')
            else:
                if args.resume:raise ValueError('NFT resume must preserve the existing run directory')
                atomic_json(root/'identity.json',identity)
                shutil.copyfile(args.config,root/'resolved_recipe.yaml')
                for rel in sources:
                    destination=root/'source'/rel;destination.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(rel,destination)
    except Exception:error=traceback.format_exc()
    shared.check_errors(error)
    model=DistributedDataParallel(actor,device_ids=[local],broadcast_buffers=False,
        find_unused_parameters=cfg['DDP_find_unused_parameters'],static_graph=cfg['DDP_static_graph'],
        bucket_cap_mb=cfg['DDP_bucket_cap_mb']) if world>1 else actor
    model.eval()
    state=dict(updates=0,collections=0,raw_scene_exposures=0,new_candidates=0,next_inner=0,scene_cursor=0,
               ema_updates=0,phase='READY',seconds_rollout=0.,seconds_scoring_wait=0.,seconds_scoring_cpu=0.,
               seconds_old_probability=0.,seconds_update=0.)
    rollout=None
    if args.resume:
        state,rollout=nft_checkpoint.restore(args.resume,actor,optimizer,scheduler,generator,ema,identity)
    invocation=Invocation(root,rank,state,identity,args.resume)
    frozen_before=parameter_hash(actor,trainable=False) if args.qualification else None
    before={n:p.detach().cpu().clone() for n,p in trainable_parameters(actor).items()} if args.qualification else None
    save_every=args.save_every or cfg['save_every_optimizer_updates']
    journal=(root/f'metrics_rank{rank}.jsonl').open('a',buffering=1)
    last_saved=None

    def save():
        begin=time.monotonic()
        result=nft_checkpoint.save(root,actor,optimizer,scheduler,generator,state,rollout,ema,identity)
        invocation.saved(result,time.monotonic()-begin)
        return result

    try:
        if not args.resume:last_saved=save()
        while state['updates']<cfg['max_steps']:
            runtime.require_dispatch()
            if rollout is None:
                token,position=stream.at(state['scene_cursor'],rank)
                state['phase']='COLLECTING';error=None
                invocation.event('collection_start')
                try:rollout=collect_ode(actor,conditions,stream,token,position,ema,service,generator,identity,invocation.event)
                except Exception:error=traceback.format_exc()
                if not all_ok(error,device):raise RuntimeError('NFT ODE collection failed: '+str(error))
                stats=advantages(rollout['rewards'])
                rollout['global_std']=stats['global_std']
                rollout['optimality']=optimality_probabilities(rollout['rewards'],stats['global_std']).reshape(8)
                state['collections']+=1;state['raw_scene_exposures']+=world
                state['new_candidates']+=world*8;state['scene_cursor']+=world
                for k,v in rollout['seconds'].items():state['seconds_'+k]+=v
                error=None
                try:
                    behavior_path=root/'behavior'/f"collection{state['collections']:06}"/f'rank{rank}.pt'
                    if behavior_path.exists():raise FileExistsError('NFT behavior publication cannot overwrite original samples')
                    shared.atomic_torch(behavior_path,rollout)
                except Exception:error=traceback.format_exc()
                if not all_ok(error,device):raise RuntimeError('NFT behavior publication failed: '+str(error))
            optimizer.zero_grad(set_to_none=True)
            begin=time.monotonic();error=None
            try:
                batch=make_forward_batch(actor,rollout,ema,generator,cfg)
                total,metrics=model(batch)
            except Exception:error=traceback.format_exc()
            if not all_ok(error,device):raise RuntimeError('NFT forward objective failed: '+str(error))
            total.backward()
            params=trainable_parameters(actor)
            grads=[p.grad.float().norm() for p in params.values() if p.grad is not None]
            norm=float(torch.stack(grads).norm()) if grads else float('nan')
            error=None if len(grads)==len(params) and np.isfinite(norm) else 'Missing/nonfinite NFT trainable gradient'
            if not all_ok(error,device):raise ValueError(error)
            if cfg['gradient_clipping_effective']>0:
                torch.nn.utils.clip_grad_norm_(list(params.values()),cfg['gradient_clipping_effective'])
            lr=optimizer.param_groups[0]['lr'];optimizer.step();scheduler.step()
            torch.cuda.synchronize(device)
            elapsed=time.monotonic()-begin
            state['updates']+=1;state['next_inner']+=1;state['seconds_update']+=elapsed
            row=dict(updates=state['updates'],collection=state['collections'],inner_completed=state['next_inner'],
                     invocation=invocation.id,run=identity['sha256'],training_protocol='world_nft_forward_v1',
                     raw_scene_exposures=state['raw_scene_exposures'],new_candidates=state['new_candidates'],
                     token=rollout['token'],ema_behavior_version=rollout['ema_version'],lr=lr,
                     behavior=dict(run=identity['sha256'],update=rollout['ema_version']*4,
                                   policy_kind='EMA_sampling_policy',ema_version=rollout['ema_version']),
                     optimizer_group_lrs={g['name']:g['lr'] for g in optimizer.param_groups},
                     loss=float(total.detach()),gradient_norm=norm,seconds_update=elapsed,
                     metrics={k:float(v) for k,v in metrics.items()},
                     exploration=dict(canonical_random_mean=float(rollout['rewards'].mean()),
                                      global_reward_std=float(rollout['global_std']),
                                      ODE_training_sampler=True))
            journal.write(json.dumps(row,allow_nan=False)+'\n')
            if state['next_inner']==4:
                ema.update(actor);state['ema_updates']=ema.updates;state['next_inner']=0;rollout=None
            state['phase']='COMPLETE' if state['updates']==cfg['max_steps'] else 'INNER_COMMITTED' if state['next_inner'] else 'READY'
            stop_now=should_stop(device) or (args.stop_after_update is not None and state['updates']>=args.stop_after_update)
            if state['updates']%save_every==0 or stop_now or state['updates']==cfg['max_steps']:
                last_saved=save()
                if rank==0:atomic_json(root/'progress.json',dict(state,status='COMPLETE' if state['updates']==cfg['max_steps'] else 'PAUSED' if stop_now else 'RUNNING',checkpoint=last_saved,host=host))
            if stop_now:break
        result=dict(status='COMPLETE' if state['updates']==cfg['max_steps'] else 'STOPPED',state=state,
                    checkpoint=last_saved,training_protocol='world_nft_forward_v1',EMA_updates=ema.updates,
                    actual_ODE_candidates=state['new_candidates'],preparation_and_train_seconds=time.monotonic()-started_process,
                    peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(device))
        if args.qualification:
            after=parameter_hash(actor,trainable=False)
            if after!=frozen_before:raise ValueError('NFT qualification changed frozen Qwen/backbone/reference')
            result.update(frozen_SHA_before=frozen_before,frozen_SHA_after=after,
                changed_trainables={n:bool(torch.count_nonzero(p.detach().cpu()-before[n])) for n,p in params.items()})
            if hasattr(runtime,'finish_qualification'):runtime.finish_qualification(result)
        atomic_json(root/f'result_rank{rank}.json',result)
        invocation.finish(state,result['status'])
    except BaseException:
        error=traceback.format_exc()
        shared.failure_evidence(root,state,error,invocation.id,rollout=rollout)
        invocation.finish(state,'ABORTED',error)
        raise
    finally:
        journal.close();service.close()
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
