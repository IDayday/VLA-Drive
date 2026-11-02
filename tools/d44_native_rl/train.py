"""Multi-scene D-native training; audited sampler and atomic recovery reused."""
import argparse
import contextlib
import copy
from datetime import timedelta
import fcntl
import json
import math
import os
from pathlib import Path
import random
import signal
import time
import traceback
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from starVLA.rl.d44_native_rl import parent
from starVLA.rl.d44_native_rl.config import read,source_config
from starVLA.rl.d44_native_rl.actor import D44Actor
from starVLA.rl.d44_native_rl.conditions import DConditions
from starVLA.rl.d44_native_rl.data import DataStream
from starVLA.rl.d44_native_rl.batch_runtime import collect_scene
from starVLA.rl.d44_native_rl.safety_credit import raw_credit,normalize
from starVLA.rl.d44_native_rl.adaptive_kl import AdaptiveKL
from starVLA.rl.d44_native_rl.expanded_scope import groups
from starVLA.rl.d44_native_rl.scheduler import scene_lr
from starVLA.rl.d44_native_rl.recovery import Boundary,Attempt,material_digest,checkpoint
from starVLA.rl.qwen_world_compare.contracts import require_dispatch,boundary_stop_due,fingerprint,file_sha
from starVLA.rl.simwam_recipe.reward_readouts import RewardService
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.transactions import atomic_json

STOP=False
def stop(*_):
    global STOP
    STOP=True


class SceneScheduler:
    def __init__(self,optimizer):self.optimizer=optimizer;self.last_epoch=0
    def step(self):self.last_epoch+=1
    def state_dict(self):return {'last_epoch':self.last_epoch}
    def load_state_dict(self,value):self.last_epoch=value['last_epoch']


def code_identity():
    roots=['starVLA/rl/d44_native_rl','tools/d44_native_rl','starVLA/model/modules/qwen_world_compare',
           'starVLA/rl/qwen_world_compare','starVLA/rl/qwen_world_abcd_research','starVLA/rl/simwam_recipe',
           'starVLA/model/modules/qwen_world_postcot','starVLA/model/modules/qwen_sourceflow']
    return {str(p):file_sha(p)for folder in roots for p in sorted(Path(folder).glob('*.py'))}


def consensus_stop(device):
    x=torch.tensor(int(STOP or boundary_stop_due()),device=device)
    if dist.is_initialized():dist.all_reduce(x,op=dist.ReduceOp.MAX)
    return bool(x)


def main():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    os.environ.setdefault('FLASH_ATTENTION_DETERMINISTIC','1')
    if os.environ['CUBLAS_WORKSPACE_CONFIG']!=':4096:8':raise ValueError('Qualified deterministic CuBLAS execution changed')
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--resume');parser.add_argument('--qualification',action='store_true');parser.add_argument('--stop-after-update',type=int)
    parser.add_argument('--scenes-per-rank',type=int,help='qualification only');parser.add_argument('--fault',choices=['before_finalize','after_inner1'])
    a=parser.parse_args();cfg=read(a.config)
    if a.scenes_per_rank and not a.qualification:raise ValueError('Formal scene batch cannot be overridden')
    host=require_dispatch();configure_numerics();torch.set_num_threads(2)
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    if world!=8 and not a.qualification:raise ValueError('Formal native D44 uses eight GPUs')
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    if world>1:dist.init_process_group('nccl',timeout=timedelta(minutes=20))
    checkpoint.init_control_group()
    for sig in (signal.SIGTERM,signal.SIGUSR1):signal.signal(sig,stop)
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    if a.scenes_per_rank:cfg['global_scenes']=a.scenes_per_rank*world
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True);lock=(root/f'rank{rank}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    print(json.dumps(dict(event='loading_native_D44',rank=rank,at=time.time())),flush=True)
    stream=DataStream(cfg,cfg['global_scenes']);model,info=parent.load(cfg['parent_checkpoint'],device,readouts=cfg['post_feedback'])
    actor=D44Actor(model,source_config(cfg),cfg)
    print(json.dumps(dict(event='native_D44_and_scope_loaded',rank=rank,at=time.time(),trainable=actor.closure['trainable_numel'])),flush=True)
    conditions=DConditions(model,stream.data,info,group=cfg['group'],capacity=cfg['condition_cache_scenes']);object.__setattr__(actor,'conditions',conditions)
    if a.qualification:
        token,_=stream.at(0,rank,0,world)
        probe_generator=torch.Generator(device='cpu').manual_seed(2101)
        root_noise=torch.randn((cfg['group'],8,3),generator=probe_generator).to(device)
        with torch.no_grad():
            current_chain,_,_,_=actor.sample_action_chain(conditions.get(token),8,deterministic=True,init_actions=root_noise)
            reference_chain,_,_,_=actor.sample_action_chain(conditions.get(token),8,deterministic=True,init_actions=root_noise,velocity_use_ref=True)
        error=float((current_chain-reference_chain).abs().max())
        if error!=0:raise FloatingPointError('Zero LoRA must exactly preserve native D44 ODE: '+str(error))
        atomic_json(root/f'initial_native_parity_rank{rank}.json',dict(status='PASS',ODE_chain_max_abs=error,
            trainable=actor.closure,reference_frozen=True,parent=info['model_sha256']))
    optimizer=torch.optim.AdamW(groups(actor,cfg),betas=(.9,.95),eps=1e-8,weight_decay=.01)
    scheduler=SceneScheduler(optimizer);generator=torch.Generator(device='cpu').manual_seed(420000+rank)
    service=RewardService(Path.cwd(),workers=cfg['scoring_workers'],cache_roots=stream.data.manifest['canonical_metric_cache_roots'])
    identity=dict(schema='d44_native_rl_v1',architecture_id='D',world=world,recipe=cfg,parent=info,
        parent_model_sha256=info['model_sha256'],conditions=conditions.identity,data=stream.identity,
        scope=actor.closure,source_files=code_identity(),score_protocol=service.protocol,
        qualification=a.qualification,fresh_Adam=True,stage_reuse='same chains/old means/std/A for all inner updates')
    identity=json.loads(json.dumps(identity));identity['sha256']=fingerprint(identity)
    error=None
    if rank==0:
        try:
            if (root/'identity.json').exists():
                if json.loads((root/'identity.json').read_text())!=identity or not a.resume:raise ValueError('Existing run requires identical explicit resume')
            else:atomic_json(root/'identity.json',identity)
        except Exception:error=traceback.format_exc()
    checkpoint.check_errors(error)
    ddp=DDP(actor,device_ids=[local],broadcast_buffers=False,find_unused_parameters=True,static_graph=False,bucket_cap_mb=25)if world>1 else actor
    ddp.eval();dual=AdaptiveKL(target=cfg['target_kl'])
    state=dict(updates=0,collections=0,scene_cursor=0,next_inner=0,phase='READY',
        valid_scene_exposures=0,raw_scene_exposures=0,new_candidates=0,valid_PG_candidates=0,extra_B1_queries=0,adaptive_G32_queries=0,
        padding_scene_exposures=0,adaptive_kl=dual.state_dict())
    batch=None
    if a.resume:state,batch=checkpoint.restore(a.resume,actor,optimizer,scheduler,generator,identity,runtime=Boundary())
    dual.load_state_dict(state['adaptive_kl']);attempt=Attempt(root,rank,state,identity,a.resume)
    trainables=[p for p in actor.parameters()if p.requires_grad]
    def save():
        begin=time.monotonic();result=checkpoint.save(root,actor,optimizer,scheduler,generator,state,batch,identity,runtime=Boundary())
        attempt.record['seconds_save']+=time.monotonic()-begin;return result
    journal=(root/f'metrics_rank{rank}.jsonl').open('a',buffering=1);last_saved=a.resume;started=time.monotonic()
    try:
        if not a.resume:last_saved=save()
        while state['valid_scene_exposures']<cfg['target_epochs']*stream.n or batch is not None:
            if consensus_stop(device)and batch is None:break
            if batch is None:
                state['phase']='COLLECTING';epoch=state['valid_scene_exposures']/stream.n
                version=dict(run=identity['sha256'],update=state['updates'],collection=state['collections']+1)
                actor.set_policy_version(str(version));scenes=[];error=None
                batch=dict(scenes=scenes,uncommitted=True)
                try:
                    for j in range(cfg['global_scenes']//world):
                        token,padding=stream.at(state['scene_cursor'],rank,j,world)
                        r=collect_scene(actor,conditions.get(token),token,stream.logs[token],service,generator,cfg,epoch,version)
                        r['padding']=padding;r['beta']=dual.beta;r['expect_fresh']=True;scenes.append(r)
                        attempt.record['physical_candidates']+=r['physical_candidates']
                except Exception:error=traceback.format_exc()
                checkpoint.check_errors(error)
                if a.fault=='before_finalize':raise RuntimeError('Injected collection not finalized; latest unchanged')
                factors=torch.stack([r['factors']for r in scenes]);endpoint=None
                if cfg['arm']=='D1'and epoch>=2:
                    endpoints=[]
                    for r in scenes:
                        gt=stream.data.target(r['token']).to(device)
                        distance=(torch.as_tensor(np.stack([x['requested_trajectory']for x in r['records']]),device=device)[:,-1,:2]-gt[-1,:2]).norm(dim=-1)
                        if not torch.isfinite(distance).all():raise ValueError('Invalid TRAIN endpoint')
                        endpoints.append(torch.exp(-distance/3.))
                    endpoint=torch.stack(endpoints)
                raw,feasible=raw_credit(factors,endpoint=endpoint,stage2=endpoint is not None)
                valid=torch.tensor([not r['padding']for r in scenes],device=device,dtype=torch.bool)
                stats=torch.stack(((raw*valid[:,None]).sum(),(raw.square()*valid[:,None]).sum(),valid.sum().double()*cfg['group']))
                if world>1:dist.all_reduce(stats)
                std=(stats[1]/stats[2]-(stats[0]/stats[2]).square()).clamp_min(0).sqrt()
                advantages=(raw/(std+1e-8)).clamp(-5,5).float()
                counts=torch.tensor([int(valid.sum()),sum(r['extra_B1']for r in scenes),sum(r['extra_G32']for r in scenes)],device=device,dtype=torch.long)
                if world>1:dist.all_reduce(counts)
                for r,adv in zip(scenes,advantages):r['advantages']=adv.detach()
                batch=dict(scenes=scenes,behavior_update=state['updates'],global_std=float(std),valid_scenes=int(counts[0]),epoch=epoch)
                batch['digest']=material_digest(batch)
                state['collections']+=1;state['scene_cursor']+=cfg['global_scenes'];state['raw_scene_exposures']+=cfg['global_scenes']
                state['valid_scene_exposures']+=int(counts[0]);state['padding_scene_exposures']+=cfg['global_scenes']-int(counts[0])
                state['new_candidates']+=cfg['global_scenes']*cfg['group'];state['extra_B1_queries']+=int(counts[1]);state['adaptive_G32_queries']+=int(counts[2])
                state['valid_PG_candidates']+=int(counts[0])*cfg['group']
                state['phase']='ROLLOUT_COMMITTED';state['next_inner']=0
                Boundary.validate_ranks(checkpoint.gather(Boundary.validate_boundary(state,batch,identity)))
                path=root/'behavior'/attempt.id/f'collection{state["collections"]:06}'/f'rank{rank}.pt';checkpoint.atomic_torch(path,batch)
            begin=time.monotonic();optimizer.zero_grad(set_to_none=True);totals={}
            multiplier=scene_lr(state['valid_scene_exposures']/stream.n,warmup=cfg['warmup_epochs'],hold=cfg['hold_until_epoch'],end=cfg['cosine_end_epoch'],floor=cfg['min_lr_ratio'])
            for group in optimizer.param_groups:
                stage_multiplier=.5 if cfg['arm']=='D1'and batch['epoch']>=2 and group['name']=='text_lora'else 1.
                group['lr']=group['initial_lr']*multiplier*stage_multiplier
            for j,r in enumerate(batch['scenes']):
                r['expect_fresh']=state['next_inner']==0
                sync=ddp.no_sync()if world>1 and j<len(batch['scenes'])-1 else contextlib.nullcontext()
                with sync:
                    loss,metrics=ddp(r)
                    weighted=loss*(0. if r['padding']else world/batch['valid_scenes']);weighted.backward()
                for key,value in metrics.items():totals[key]=totals.get(key,0.)+float(value.detach())/len(batch['scenes'])
            norm=torch.nn.utils.clip_grad_norm_(trainables,1.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite native gradient')
            grad_roles={}
            for group in optimizer.param_groups:grad_roles[group['name']]=float(torch.stack([p.grad.float().norm()for p in group['params']if p.grad is not None]).norm())if any(p.grad is not None for p in group['params'])else None
            optimizer.step();scheduler.step();torch.cuda.synchronize();state['updates']+=1;state['next_inner']+=1;state['phase']='INNER_COMMITTED'
            row=dict(updates=state['updates'],run=identity['sha256'],collection=state['collections'],inner_completed=state['next_inner'],
                behavior_update=batch['behavior_update'],invocation=attempt.id,scene_epochs=state['valid_scene_exposures']/stream.n,
                raw_scene_exposures=state['raw_scene_exposures'],scene_exposures=state['valid_scene_exposures'],PG_candidates=state['valid_PG_candidates'],
                generated_PG_candidates_including_padding=state['new_candidates'],
                extra_B1_queries=state['extra_B1_queries'],adaptive_G32_queries=state['adaptive_G32_queries'],
                losses=totals,gradient_norm=float(norm),role_gradient_norms=grad_roles,lr=[g['lr']for g in optimizer.param_groups],
                update_seconds=time.monotonic()-begin,collection_seconds_generation=sum(r['seconds_generation']for r in batch['scenes']),
                collection_seconds_scoring=sum(r['seconds_scoring']for r in batch['scenes']),peak_GPU_bytes=torch.cuda.max_memory_allocated(),
                global_advantage_std=batch['global_std'],at=time.time(),beta=dual.beta)
            journal.write(json.dumps(row,allow_nan=False)+'\n')
            print(json.dumps(dict(event='optimizer_committed',rank=rank,updates=state['updates'],collection=state['collections'],
                scene_epochs=row['scene_epochs'],grad_norm=float(norm),fresh_ratio_error=totals['fresh_ratio_error'])),flush=True)
            if state['next_inner']==cfg['inner_updates']:
                kval=torch.tensor(totals['reference_KL_joint'],device=device,dtype=torch.float64)
                if world>1:dist.all_reduce(kval);kval/=world
                dual.update(float(kval));state['adaptive_kl']=dual.state_dict();batch=None;state['next_inner']=0;state['phase']='READY'
            if a.fault=='after_inner1'and state['next_inner']==1:stop()
            stopping=consensus_stop(device)or(a.stop_after_update is not None and state['updates']>=a.stop_after_update)
            if stopping or state['updates']%cfg['save_every']==0:last_saved=save()
            if stopping:break
        if last_saved is None or json.loads((Path(last_saved)/'COMPLETE').read_text())['updates']!=state['updates']:last_saved=save()
        result=dict(state,status='QUALIFICATION_STOPPED'if a.qualification else'COMPLETE'if state['valid_scene_exposures']>=cfg['target_epochs']*stream.n and batch is None else'STOPPED_AT_SAVED_BOUNDARY',
            checkpoint=last_saved,rank=rank,host=host,seconds=time.monotonic()-started,qualification=a.qualification,scope=actor.closure)
        atomic_json(root/f'result_rank{rank}.json',result);attempt.finish(state,result['status'])
    except BaseException:
        checkpoint.failure_evidence(root,state,traceback.format_exc(),attempt.id,rollout=batch);attempt.finish(state,'ABORTED');raise
    finally:
        journal.close();service.close()
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
