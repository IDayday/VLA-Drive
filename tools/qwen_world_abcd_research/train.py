"""Own-best-parent architecture-aware RL; four updates per recoverable chain.

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
import sys
if __package__ in (None,''):
    script_dir=Path(__file__).resolve().parent
    sys.path[:]=[x for x in sys.path if Path(x or os.curdir).resolve()!=script_dir]
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from starVLA.rl.simwam_recipe.contracts import *
from starVLA.rl.simwam_recipe.trainer_adapter import make_actor
from starVLA.rl.simwam_recipe.condition import ProjectedConditions
from starVLA.rl.simwam_recipe.reward_readouts import RewardService
from starVLA.rl.simwam_recipe.collection import collect,finalize_advantages,exploration_summary
from starVLA.rl.simwam_recipe.data_stream import make_scene_stream
from starVLA.rl.qwen_world_abcd_research import checkpoint
from starVLA.rl.qwen_world_compare.recovery_state import boundary_phase,rollout_digest,validate_boundary,validate_ranks
from starVLA.rl.simwam_recipe.invocation import Invocation
from starVLA.rl.simwam_recipe.lineage import scientific_identity,execution_identity,verify_migration
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.transactions import atomic_json

sys.modules.setdefault('tools.qwen_world_abcd_research.train',sys.modules[__name__])
STOP=False
def stop(*_):
    global STOP
    STOP=True


def all_ok(error,device):
    if error and checkpoint.fatal_device_error(error):raise RuntimeError(error)
    return not any(e is not None for e in checkpoint.gather(error))


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


def main(runtime=None):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    if os.environ['CUBLAS_WORKSPACE_CONFIG']!=':4096:8':raise ValueError('Qualified deterministic CUBLAS workspace changed')
    if runtime is None:
        preliminary=argparse.ArgumentParser(add_help=False)
        preliminary.add_argument('--architecture',choices=('A','B','C','D'),required=True)
        known,_=preliminary.parse_known_args()
        from starVLA.rl.qwen_world_abcd_research.runtime import ABCDBestRuntime
        if known.architecture=='D':
            from starVLA.rl.qwen_world_abcd_research.d_entry import DRuntimeMixin as ObjectiveMixin
        else:
            from starVLA.rl.qwen_world_abcd_research.abc import ABCObjectiveMixin as ObjectiveMixin
        class Runtime(ObjectiveMixin,ABCDBestRuntime):pass
        runtime=Runtime()
    process_started=time.monotonic();process_started_unix=time.time()
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--config',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--resume');p.add_argument('--qualification',action='store_true')
    p.add_argument('--migration-receipt',help='Explicit audited source migration; never inferred from a dirty worktree')
    p.add_argument('--stop-after-update',type=int);p.add_argument('--save-every',type=int)
    p.add_argument('--fault',help='Qualification only: before_collect, after_collect, before_finalize, rank_write, before_complete')
    p.add_argument('--fault-rank',type=int,default=0);p.add_argument('--fault-at-update',type=int,default=0)
    # Explicit interfaces for a separately identified supervised base. The
    # default historical entry and its strict recipe/asset assertions remain
    # unchanged; no global monkeypatching or implicit checkpoint substitution.
    if runtime is not None:runtime.add_arguments(p)
    a=p.parse_args()
    dispatch=runtime.require_dispatch if runtime is not None else require_dispatch
    cfg=runtime.load_recipe(a) if runtime is not None else load_recipe(a.config)
    host=dispatch();configure_numerics()
    if a.fault and not a.qualification:raise ValueError('Fault injection cannot enter formal runs')
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    if world!=8 and not a.qualification:raise ValueError('Formal recipe requires eight GPUs')
    torch.cuda.set_device(local)
    if world>1:dist.init_process_group('nccl',timeout=timedelta(minutes=20))
    checkpoint.init_control_group()
    device=torch.device('cuda',local)
    for s in (signal.SIGUSR1,signal.SIGTERM):signal.signal(s,stop)
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    lock=(root/f'rank{rank}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stream=(runtime.make_scene_stream(cfg,a,world) if runtime is not None else
            make_scene_stream(cfg,a.manifest,world,qualification=a.qualification))
    actor,oldcfg,sft=(runtime.make_actor(cfg,a,device) if runtime is not None else make_actor(cfg,device=device))
    conditions=(runtime.make_conditions(actor) if runtime is not None else
                ProjectedConditions(Path(ROOT)/'projected_condition',actor.action_expert))
    service=runtime.make_reward_service()
    trainable=[p for p in actor.parameters() if p.requires_grad]
    optimizer=(runtime.make_optimizer(actor,cfg) if runtime is not None and hasattr(runtime,'make_optimizer') else
        torch.optim.AdamW(trainable,lr=cfg['learning_rate'],betas=tuple(cfg['optimizer_betas']),eps=cfg['optimizer_eps'],weight_decay=cfg['weight_decay']))
    explicit_optimizer_groups=([dict(name=g['name'],lr=g['lr'],tensors=len(g['params']),numel=sum(p.numel() for p in g['params']))
        for g in optimizer.param_groups] if runtime is not None and hasattr(runtime,'make_optimizer') else None)
    actor._loss.optimizer=optimizer;actor._loss.learning_rate=cfg['learning_rate']
    warmup=min(int(cfg['max_steps']*.05),cfg['lr_warmup_steps'])
    scheduler=actor._loss._build_scheduler(cfg['lr_scheduler_type'],cfg['max_steps'],warmup)
    generator=torch.Generator(device='cpu').manual_seed(seed_for('simwam_source_rank_rollout',42,rank))
    identity=(runtime.recipe_identity(cfg,a,conditions.sha) if runtime is not None else
              recipe_identity(cfg,a.manifest,conditions.sha))
    identity.update(world=world,purpose='qualification_not_formal_initialization' if a.qualification else 'formal',
                    score_protocol=service.protocol,source_files=runtime.source_files() if runtime is not None else source_files(),parameter_contract=actor.closure,
                    rng_protocol='independent persistent CPU FP32 generator per rank; source _randn; no repeated candidate seeds',
                    scene_reduction='source local effective mean, then equal mean across ranks',
                    actual_optimizer=dict(name='torch.optim.AdamW',lr=cfg['learning_rate'],betas=cfg['optimizer_betas'],
                                          eps=cfg['optimizer_eps'],weight_decay=cfg['weight_decay'],warmup_updates=warmup,
                                          max_grad_norm_effective=cfg['gradient_clipping_effective'],FP32_master_and_moments=True))
    if hasattr(stream,'identity'):identity['data_stream_identity']=stream.identity
    if explicit_optimizer_groups is not None:identity['actual_optimizer']['parameter_groups']=explicit_optimizer_groups
    identity['scientific_identity']=scientific_identity(identity)
    identity['execution_identity']=execution_identity(identity['source_files'])
    identity['sha256']=fingerprint(identity)
    current_identity=identity;migration=None
    if (root/'identity.json').exists():
        previous=json.loads((root/'identity.json').read_text())
        if previous!=identity:
            if not (a.resume and a.migration_receipt):raise ValueError('Source identity changed: an explicit tested migration receipt is required')
            migration=verify_migration(a.migration_receipt,previous,current_identity,a.resume)
            identity=previous
    checkpoint.EXECUTION_IDENTITY=current_identity['execution_identity'];checkpoint.MIGRATION_RECEIPT=migration
    execution=root/'executions'/current_identity['execution_identity']['sha256']
    error=None
    try:
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
            execution.mkdir(parents=True,exist_ok=True)
            atomic_json(execution/'identity.json',current_identity)
            if migration:atomic_json(execution/'migration_receipt.json',migration)
            for rel in current_identity['source_files']:
                dest=execution/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(rel,dest)
    except Exception:error=traceback.format_exc()
    checkpoint.check_errors(error)
    modules={}
    for name,module in list(sys.modules.items()):
        path=getattr(module,'__file__',None)
        if path and name.startswith(('starVLA.','scripts.analysis.simwam_recipe','scripts.analysis.fm_transfer')) and Path(path).is_file():
            path=Path(path).resolve()
            if not path.is_relative_to(Path.cwd().resolve()):raise ValueError('Imported project module from outside the execution snapshot: '+str(path))
            modules[name]=dict(path=str(path),sha256=file_sha(path))
    atomic_json(execution/f'runtime_modules_rank{rank}.json',modules)
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
               phase='READY',
               RL_updates_local=0,anchor_only_updates_local=0,scene_cursor=0,
               seconds_rollout=0.,seconds_scoring_wait=0.,seconds_scoring_cpu=0.,seconds_old_probability=0.,seconds_update=0.)
    rollout=None
    if a.resume:state,rollout=checkpoint.restore(a.resume,actor,optimizer,scheduler,generator,identity,migration=migration,runtime=runtime)
    if rollout is not None and 'chain_identity' not in rollout:rollout['chain_identity']=rollout_digest(rollout)
    invocation=Invocation(root,rank,state,identity,a.resume)
    invocation.record.update(process_start_unix=process_started_unix,preparation_seconds=time.monotonic()-process_started)
    def fault(point,worker=rank):
        if a.fault==point and worker==a.fault_rank and state['updates']==a.fault_at_update:
            raise RuntimeError('Qualification injected failure: '+point)
    def save_boundary():
        begin=time.monotonic()
        result=checkpoint.save(root,actor,optimizer,scheduler,generator,state,rollout,identity,fault=fault,runtime=runtime)
        invocation.saved(result,time.monotonic()-begin)
        return result
    frozen_before=parameter_hash(actor,trainable=False) if a.qualification else None
    first_adapters=checkpoint.cpu_tree({n:p.clone() for n,p in actor.named_parameters() if p.requires_grad}) if a.qualification else None
    started=time.monotonic();save_every=a.save_every or cfg['save_every_optimizer_updates'];last_saved=None
    journal=(root/f'metrics_rank{rank}.jsonl').open('a',buffering=1)
    try:
        if not a.resume:last_saved=save_boundary()
        while state['updates']<cfg['max_steps']:
            actor.set_policy_version(f'RL:{identity["sha256"]}:{state["updates"]}')
            dispatch()
            if rollout is None:
                state['phase']='COLLECTING';invocation.event('collection_start')
                token,position=stream.at(state['scene_cursor'],rank)
                error=None
                try:
                    fault('before_collect')
                    version=dict(run=identity['sha256'],update=state['updates'],adapter_sha256=parameter_hash(actor,trainable=True))
                    if runtime is not None and hasattr(runtime,'collect'):
                        rollout=runtime.collect(actor,conditions.get(token,device),token,stream.logs[token],service,generator,
                                                variant=cfg['variant'],version=version,progress=invocation.event)
                        if hasattr(runtime,'validate_collection'):runtime.validate_collection(rollout)
                    else:
                        rollout=collect(actor,conditions.get(token,device),token,stream.logs[token],service,generator,
                                        act_norm=int(sft.datasets.vla_data.act_norm),variant=cfg['variant'],version=version,progress=invocation.event)
                    rollout['stream_position']=position
                    fault('after_collect')
                except Exception:
                    error=traceback.format_exc()
                    checkpoint.failure_evidence(root,state,error,invocation.id,rollout=rollout)
                if not all_ok(error,device):
                    raise RuntimeError('At least one rank failed actual sampling/scoring; no zero replacement or update')
                error=None
                try:fault('before_finalize')
                except Exception:error=traceback.format_exc()
                if not all_ok(error,device):raise RuntimeError('Collection not committed before advantage finalization')
                runtime.finalize_advantages(rollout)
                state['collections']+=1;state['raw_scene_exposures']+=world;state['new_candidates']+=world*8;state['scene_cursor']+=world
                for k,v in rollout['seconds'].items():state['seconds_'+k]+=v
                state['next_inner']=0
                state['phase']='ROLLOUT_COMMITTED'
                rollout['invocation']=invocation.id
                rollout['chain_identity']=rollout_digest(rollout)
                # This check is cheap (metadata only), and commits the SAME source advantages.
                error=None;summary=None
                try:summary=runtime.validate_boundary(state,rollout,identity)
                except Exception:error=traceback.format_exc()
                checkpoint.check_errors(error)
                runtime.validate_ranks(checkpoint.gather(summary))
                # Every fresh chain remains replayable, including negative candidates.
                behavior_path=root/'behavior'/f'collection{state["collections"]:06}'/f'rank{rank}.pt'
                error=None
                try:
                    if behavior_path.exists():raise FileExistsError('Behavior must retain its original attempt: '+str(behavior_path))
                    checkpoint.atomic_torch(behavior_path,rollout)
                except Exception:error=traceback.format_exc()
                if not all_ok(error,device):raise RuntimeError('Behavior publication failed; no optimizer update')
            optimizer.zero_grad(set_to_none=True)
            begin=time.monotonic();error=None
            try:total,metrics=model(rollout)
            except Exception:
                error=traceback.format_exc()
            if not all_ok(error,device):
                raise RuntimeError('At least one rank failed source loss; last sealed boundary retained')
            current=actor._probability_audit['current'];old=rollout['old_logp']
            ratio=(current['source_clamped_logprob'].mean((-1,-2))-old).exp()
            fresh_error=float((ratio-1).abs().max())
            error=None
            if state['next_inner']==0 and fresh_error>1e-4:error='Fresh saved-state ratio differs by '+str(fresh_error)
            if not all_ok(error,device):
                raise ValueError('Fresh probability numerical error exceeds 1e-4; no rewritten old density')
            total.backward()
            grads=torch.stack([p.grad.float().norm() for p in trainable if p.grad is not None])
            grad_norm=float(grads.norm())
            error=None if np.isfinite(grad_norm) and len(grads)==len(trainable) else 'Missing/nonfinite adapter gradient'
            if not all_ok(error,device):
                raise ValueError('Invalid LoRA gradient; last sealed boundary retained')
            if cfg['gradient_clipping_effective']>0:torch.nn.utils.clip_grad_norm_(trainable,cfg['gradient_clipping_effective'])
            lr=optimizer.param_groups[0]['lr'];optimizer.step();scheduler.step()
            torch.cuda.synchronize();elapsed=time.monotonic()-begin
            state['updates']+=1;state['next_inner']+=1;state['seconds_update']+=elapsed
            state['RL_updates_local']+=int(bool(rollout['adv_mask'].any()))
            state['anchor_only_updates_local']+=int(not bool(rollout['adv_mask'].any()))
            row=dict(updates=state['updates'],collection=state['collections'],inner_completed=state['next_inner'],
                     invocation=invocation.id,run=identity['sha256'],chain_identity=rollout['chain_identity'],
                     raw_scene_exposures=state['raw_scene_exposures'],new_candidates=state['new_candidates'],
                     token=rollout['token'],behavior=rollout['behavior_version'],lr=lr,loss=float(total.detach()),
                     gradient_norm=grad_norm,ratio_max_abs_from_one=fresh_error,
                     source_clamp_hit_rate=current['source_clamp_hit_rate'],seconds_update=elapsed,
                     metrics={k:float(v) for k,v in metrics.items()},exploration=exploration_summary(rollout))
            if explicit_optimizer_groups is not None:
                row['optimizer_group_lrs']={g['name']:g['lr'] for g in optimizer.param_groups}
            journal.write(json.dumps(row,allow_nan=False)+'\n')
            if state['next_inner']==4:rollout=None;state['next_inner']=0
            state['phase']=boundary_phase(state,cfg['max_steps'])
            stop_now=stop_requested(device) or (a.stop_after_update is not None and state['updates']>=a.stop_after_update)
            milestone=state['updates'] in cfg['evaluate_updates']
            if state['updates']%save_every==0 or milestone or stop_now or state['updates']==cfg['max_steps']:
                last_saved=save_boundary()
                if rank==0:atomic_json(root/'progress.json',dict(state,status='COMPLETE' if state['updates']==cfg['max_steps'] else 'PAUSED' if stop_now else 'RUNNING',checkpoint=last_saved,host=host,invocation=invocation.id,invocation_start_update=invocation.record['start_update'],wall_seconds_this_invocation=time.monotonic()-started))
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
            if runtime is not None and hasattr(runtime,'finish_qualification'):
                runtime.finish_qualification(result)
        atomic_json(root/f'result_rank{rank}.json',result)
        invocation.finish(state,result['status'])
    except BaseException:
        error=traceback.format_exc()
        checkpoint.failure_evidence(root,state,error,invocation.id,rollout=rollout)
        invocation.finish(state,'ABORTED',error)
        raise
    finally:
        journal.close();service.close()
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
