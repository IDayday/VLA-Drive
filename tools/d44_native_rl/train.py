"""Multi-scene D-native training; audited sampler and atomic recovery reused."""
import argparse
import contextlib
import copy
from datetime import timedelta
import fcntl
import hashlib
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
from starVLA.rl.d44_native_rl.conditions import DConditions,D6Conditions
from starVLA.rl.d44_native_rl.data import DataStream
from starVLA.rl.d44_native_rl.batch_runtime import collect_scene
from starVLA.rl.d44_native_rl.safety_credit import raw_credit,normalize
from starVLA.rl.d44_native_rl.two_stage_reward import endpoint_quality
from starVLA.rl.d44_native_rl.adaptive_kl import AdaptiveKL
from starVLA.rl.d44_native_rl.expanded_scope import groups
from starVLA.rl.d44_native_rl.scheduler import scene_lr
from starVLA.rl.d44_native_rl.stages import stage_recipe
from starVLA.rl.d44_native_rl import history as self_history
from starVLA.rl.d44_native_rl.recovery import Boundary,Attempt,material_digest,checkpoint
from starVLA.rl.qwen_world_compare.contracts import require_dispatch,boundary_stop_due,fingerprint,file_sha
from starVLA.rl.simwam_recipe.reward_readouts import RewardService
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.transactions import atomic_json

STOP=False
def stop(*_):
    global STOP
    STOP=True


def frozen_parameter_hash(actor):
    result=hashlib.sha256()
    for name,p in actor.named_parameters():
        if not p.requires_grad:result.update(name.encode());result.update(p.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


class SceneScheduler:
    def __init__(self,optimizer):self.optimizer=optimizer;self.last_epoch=0
    def step(self):self.last_epoch+=1
    def state_dict(self):return {'last_epoch':self.last_epoch}
    def load_state_dict(self,value):self.last_epoch=value['last_epoch']


def code_identity():
    roots=['starVLA/rl/d44_native_rl','tools/d44_native_rl','starVLA/model/modules/qwen_world_compare',
           'starVLA/rl/qwen_world_compare','starVLA/rl/qwen_world_abcd_research','starVLA/rl/simwam_recipe',
           'starVLA/model/modules/qwen_world_postcot','starVLA/model/modules/qwen_sourceflow',
           'starVLA/rl/flow_grpo','starVLA/rl/safe_flow_improvement',
           'scripts/analysis/constrained_flow_noise']
    return {str(p):file_sha(p)for folder in roots for p in sorted(Path(folder).rglob('*.py'))}


def consensus_stop(device):
    x=torch.tensor(int(STOP or boundary_stop_due()),device=device)
    if dist.is_initialized():dist.all_reduce(x,op=dist.ReduceOp.MAX)
    return bool(x)


def main(*,extension=None):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    os.environ.setdefault('FLASH_ATTENTION_DETERMINISTIC','1')
    if os.environ['CUBLAS_WORKSPACE_CONFIG']!=':4096:8':raise ValueError('Qualified deterministic CuBLAS execution changed')
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--resume');parser.add_argument('--qualification',action='store_true');parser.add_argument('--stop-after-update',type=int)
    parser.add_argument('--scenes-per-rank',type=int,help='qualification only');parser.add_argument('--fault',choices=['before_finalize','after_inner1'])
    a=parser.parse_args();cfg=read(a.config)if extension is None else extension.read_recipe(a)
    if cfg['arm']=='D6'and not a.qualification:
        proof_path=Path('/mnt/project/d44-native-rl-20261103/qualification/D6_REAL_STAGE_INNER_RESUME.json')
        proof=json.loads(proof_path.read_text())if proof_path.exists()else{}
        expected={p:file_sha(p)for p in ('tools/d44_native_rl/train.py','starVLA/rl/d44_native_rl/stages.py','starVLA/rl/d44_native_rl/history.py','starVLA/rl/d44_native_rl/recovery.py')}
        if proof.get('status')!='PASS'or proof.get('source_files')!=expected:raise ValueError('D6 requires same-source real staged inner-resume qualification')
    if a.scenes_per_rank and not a.qualification:raise ValueError('Formal scene batch cannot be overridden')
    host=require_dispatch();configure_numerics();torch.set_num_threads(2)
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    if extension is None:
        if world!=8 and not a.qualification:raise ValueError('Formal native D44 uses eight GPUs')
    else:extension.validate_world(world)
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    if world>1:dist.init_process_group('nccl',timeout=timedelta(minutes=20))
    if extension is None:checkpoint.init_control_group()
    else:extension.init_control_group()
    for sig in (signal.SIGTERM,signal.SIGUSR1):signal.signal(sig,stop)
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    if a.scenes_per_rank:cfg['global_scenes']=a.scenes_per_rank*world
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True);lock=(root/f'rank{rank}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    print(json.dumps(dict(event='loading_native_D44',rank=rank,at=time.time())),flush=True)
    stream=DataStream(cfg,cfg['global_scenes'])if extension is None else extension.make_stream(cfg)
    model,info=parent.load(cfg['parent_checkpoint'],device,readouts=cfg['post_feedback'],binding=cfg.get('parent_registration'))
    original_prediction=None
    if a.qualification:
        from starVLA.model.modules.qwen_sourceflow.qwen_encoder import QwenObservationProcessor
        from starVLA.rl.qwen_sourceflow.contracts import QWEN_PATH
        token,_=stream.at(0,rank,0,world);observation=stream.data.observation(token)
        original_inputs=QwenObservationProcessor(str(QWEN_PATH))([observation])
        original_ego=torch.tensor([observation['proprio']],device=device,dtype=torch.float32)
        original_noise=torch.randn((1,8,3),generator=torch.Generator().manual_seed(2101)).to(device)
        original_prediction=model.predict(original_inputs,original_ego,original_noise)
    actor_factory=D44Actor if extension is None else getattr(extension,"actor_factory",D44Actor)
    actor=actor_factory(model,source_config(cfg),cfg)
    checkpoint_policy=cfg.get('checkpoint_policy','source')
    if checkpoint_policy not in('source','short_off','all_off'):raise ValueError('Unqualified checkpoint execution policy')
    model.decision.activation_checkpointing=checkpoint_policy!='all_off'
    model.decision.checkpoint_short_suffix=checkpoint_policy=='source'
    print(json.dumps(dict(event='native_D44_and_scope_loaded',rank=rank,at=time.time(),trainable=actor.closure['trainable_numel'])),flush=True)
    condition_type=D6Conditions if cfg['arm']=='D6'else DConditions
    conditions=condition_type(model,stream.data,info,group=cfg['group'],capacity=cfg['condition_cache_scenes']);object.__setattr__(actor,'conditions',conditions)
    if a.qualification:
        token,_=stream.at(0,rank,0,world)
        probe_generator=torch.Generator(device='cpu').manual_seed(2101)
        root_noise=torch.randn((cfg['group'],8,3),generator=probe_generator).to(device)
        with torch.no_grad():
            current_chain,_,_,_=actor.sample_action_chain(conditions.get(token),8,deterministic=True,init_actions=root_noise)
            reference_chain,_,_,_=actor.sample_action_chain(conditions.get(token),8,deterministic=True,init_actions=root_noise,velocity_use_ref=True)
        error=float((current_chain-reference_chain).abs().max())
        if error!=0:raise FloatingPointError('Zero LoRA must exactly preserve native D44 ODE: '+str(error))
        wrapped_prediction=actor.action_expert.predict(original_inputs,original_ego,original_noise)
        for key in ('normalized_action','requested_trajectory'):
            if not torch.equal(original_prediction[key],wrapped_prediction[key]):raise FloatingPointError('Wrapped native clean-x deployment changed '+key)
        atomic_json(root/f'initial_native_parity_rank{rank}.json',dict(status='PASS',ODE_chain_max_abs=error,
            original_cleanx_deployment_max_abs=0.,trainable=actor.closure,reference_frozen=True,parent=info['model_sha256']))
    parameter_groups=groups if extension is None else getattr(extension,"optimizer_groups",groups)
    optimizer=torch.optim.AdamW(parameter_groups(actor,cfg),betas=(.9,.95),eps=1e-8,weight_decay=.01)
    scheduler=SceneScheduler(optimizer);generator=torch.Generator(device='cpu').manual_seed(420000+rank)
    service=RewardService(Path.cwd(),workers=cfg['scoring_workers'],cache_roots=stream.data.manifest['canonical_metric_cache_roots'])
    identity=dict(schema='d44_native_rl_v1',architecture_id='D',world=world,recipe=cfg,parent=info,
        parent_model_sha256=info['model_sha256'],conditions=conditions.identity,data=stream.identity,
        scope=actor.closure,source_files=code_identity(),score_protocol=service.protocol,
        qualification=a.qualification,fresh_Adam=True,stage_reuse='same chains/old means/std/A for all inner updates')
    identity=json.loads(json.dumps(identity));identity['sha256']=fingerprint(identity)
    if extension is not None:identity=extension.identity(identity)
    runtime=Boundary()if extension is None else extension.boundary
    error=None
    if rank==0:
        try:
            if (root/'identity.json').exists():
                if json.loads((root/'identity.json').read_text())!=identity or not a.resume:raise ValueError('Existing run requires identical explicit resume')
            else:atomic_json(root/'identity.json',identity)
        except Exception:error=traceback.format_exc()
    checkpoint.check_errors(error)
    ddp=DDP(actor,device_ids=[local],broadcast_buffers=False,find_unused_parameters=True,static_graph=False,bucket_cap_mb=25)if world>1 else actor
    ddp.eval();dual=AdaptiveKL(target=cfg['target_kl']) if extension is None or not hasattr(extension,"make_dual") else extension.make_dual(cfg)
    state=dict(updates=0,collections=0,scene_cursor=0,next_inner=0,phase='READY',
        valid_scene_exposures=0,raw_scene_exposures=0,new_candidates=0,valid_PG_candidates=0,extra_B1_queries=0,adaptive_G32_queries=0,
        padding_scene_exposures=0,adaptive_kl=dual.state_dict(),evaluation_gates=[])
    if cfg['arm']=='D6':state.update(d6_history=self_history.initial(identity['sha256']),stage_id='I',stage_transitions=[])
    batch=None
    if a.resume:state,batch=checkpoint.restore(a.resume,actor,optimizer,scheduler,generator,identity,runtime=runtime)
    if extension is not None:
        error=None
        try:
            extension.after_restore(actor,conditions,stream,state,identity)
            configure_actor=getattr(extension,'configure_actor',None)
            if configure_actor is not None:configure_actor(actor)
        except Exception:
            error=traceback.format_exc()
            if checkpoint.fatal_device_error(error):raise
        checkpoint.check_errors(error)
    dual.load_state_dict(state['adaptive_kl'])
    attempt=Attempt(root,rank,state,identity,a.resume)if extension is None else extension.make_attempt(root,rank,state,identity,a.resume)
    trainables=[p for p in actor.parameters()if p.requires_grad]
    frozen_before=frozen_parameter_hash(actor)if a.qualification else None
    def save():
        begin=time.monotonic();result=checkpoint.save(root,actor,optimizer,scheduler,generator,state,batch,identity,runtime=runtime)
        attempt.record['seconds_save']+=time.monotonic()-begin;return result
    journal=(root/f'metrics_rank{rank}.jsonl').open('a',buffering=1);last_saved=a.resume;started=time.monotonic()
    try:
        if not a.resume:last_saved=save()
        while state['valid_scene_exposures']<cfg['target_epochs']*stream.n or batch is not None:
            if consensus_stop(device)and batch is None:break
            if batch is None:
                state['phase']='COLLECTING';epoch=state['valid_scene_exposures']/stream.n
                collection_cfg,stage=stage_recipe(cfg,epoch)if cfg['arm']=='D6'else(cfg,None)
                global_scenes=collection_cfg['global_scenes'];actual_group=collection_cfg['group']
                plan=None
                if stage is not None:
                    if state['stage_id']!=stage.name:
                        state['stage_transitions'].append(dict(from_stage=state['stage_id'],to_stage=stage.name,
                            updates=state['updates'],scene_epochs=epoch,optimizer_reset=False))
                        state['stage_id']=stage.name
                    if rank==0:
                        uniform=[]
                        for pos in range(global_scenes):
                            token,padding=stream.at(state['scene_cursor'],pos%world,pos//world,world)
                            uniform.append(dict(token=token,padding=padding))
                        plan=self_history.select(state['d6_history'],uniform,collection=state['collections']+1,enabled=stage.mixture)
                    plan=checkpoint.gather(plan)[0]
                version=dict(run=identity['sha256'],update=state['updates'],collection=state['collections']+1)
                actor.set_policy_version(str(version));scenes=[];error=None
                if extension is not None:extension.before_collection(state,generator,rank,world)
                batch=dict(scenes=scenes,uncommitted=True)
                try:
                    for j in range(global_scenes//world):
                        if plan is None:token,padding=stream.at(state['scene_cursor'],rank,j,world);sampling_source='uniform'
                        else:entry=plan[j*world+rank];token,padding,sampling_source=entry['token'],entry['padding'],entry['source']
                        descriptor=conditions.get_for_group(token,actual_group)if stage is not None else conditions.get(token)
                        if extension is not None:extension.before_scene(generator,j)
                        collector=collect_scene if extension is None else getattr(extension,"collect_scene",collect_scene)
                        r=collector(actor,descriptor,token,stream.logs[token],service,generator,collection_cfg,epoch,version,
                            cost=attempt.add_cost,history=state.get('d6_history'))
                        if extension is not None:extension.after_scene(generator,j)
                        r['sampling_source']=sampling_source
                        r['padding']=padding;r['beta']=dual.beta;r['expect_fresh']=True;scenes.append(r)
                except Exception:
                    error=traceback.format_exc()
                    if checkpoint.fatal_device_error(error):raise
                checkpoint.check_errors(error)
                if extension is not None:extension.after_collection(state,generator,rank,world)
                if a.fault=='before_finalize':raise RuntimeError('Injected collection not finalized; latest unchanged')
                factors=torch.stack([r['factors']for r in scenes]);endpoint=None;endpoint_confidence=None
                if(cfg['arm']=='D1'and epoch>=2)or(stage is not None and stage.endpoint):
                    endpoints=[];confidence=[]
                    for r in scenes:
                        gt=stream.data.target(r['token']).to(device)
                        quality,weight=endpoint_quality(gt,torch.as_tensor(np.stack([x['requested_trajectory']for x in r['records']]),device=device))
                        endpoints.append(quality);confidence.append(weight)
                    endpoint=torch.stack(endpoints);endpoint_confidence=torch.stack(confidence)[:,None]
                raw,feasible=raw_credit(factors,endpoint=endpoint,endpoint_confidence=endpoint_confidence,stage2=endpoint is not None) if extension is None or not hasattr(extension,"credit") else extension.credit(scenes,stream,epoch)
                if stage is not None and stage.temporal:
                    # A bounded native full-window DAC tie-cost for ALL-failed
                    # groups. NC/TTC events additionally train R, not a made-up
                    # minimum-TTC/distance surrogate or first-event-only reward.
                    road=torch.tensor([[sum(bool(v)for v in q['non_drivable_body_by_time'])/len(q['non_drivable_body_by_time'])for q in r['records']]for r in scenes],device=device,dtype=torch.float64)
                    offsets=road-road[:,:1]
                    raw+=torch.where(feasible.any(1,keepdim=True),torch.zeros_like(raw),.05*(offsets.mean(1,keepdim=True)-offsets))
                if cfg['arm']=='D0_CANONICAL_CREDIT':
                    canonical=torch.tensor([[v['metrics']['score']for v in r['records']]for r in scenes],device=device,dtype=torch.float64)
                    offsets=canonical-canonical[:,:1];raw=offsets-offsets.mean(1,keepdim=True)
                valid=torch.tensor([not r['padding']for r in scenes],device=device,dtype=torch.bool)
                stats=torch.stack(((raw*valid[:,None]).sum(),(raw.square()*valid[:,None]).sum(),valid.sum().double()*actual_group))
                if world>1:dist.all_reduce(stats)
                std=(stats[1]/stats[2]-(stats[0]/stats[2]).square()).clamp_min(0).sqrt()
                advantages=(raw/(std+1e-8)).clamp(-5,5).float()
                if extension is not None and hasattr(extension,"advantages"):
                    advantages,std=extension.advantages(raw,valid,actual_group,world)
                counts=torch.tensor([int(valid.sum()),sum(r['extra_B1']for r in scenes),sum(r['extra_G32']for r in scenes)],device=device,dtype=torch.long)
                if world>1:dist.all_reduce(counts)
                for r,adv in zip(scenes,advantages):r['advantages']=adv.detach()
                scores=torch.tensor([[q['metrics']['score']for q in r['records']]for r in scenes],device=device,dtype=torch.float64)
                diagnostic=torch.tensor([scores.sum().item(),scores.max(1).values.sum().item(),feasible.double().sum().item(),
                    int((raw.abs().amax(1)>0).sum()),int((scores==1).all(1).sum()),int((scores==0).all(1).sum()),len(scenes)],device=device,dtype=torch.float64)
                if world>1:dist.all_reduce(diagnostic)
                diagnostic=diagnostic.cpu().tolist();nscene=diagnostic[-1]
                summary=dict(random_candidate_mean=diagnostic[0]/(nscene*actual_group),oracle_mean=diagnostic[1]/nscene,
                    feasible_candidate_fraction=diagnostic[2]/(nscene*actual_group),effective_PG_group_fraction=diagnostic[3]/nscene,
                    all_one_groups=diagnostic[4],all_zero_groups=diagnostic[5],raw_groups=nscene,
                    sigma_by_stochastic_step=scenes[0]['std'].detach().cpu().tolist())
                batch=dict(scenes=scenes,behavior_update=state['updates'],global_std=float(std),valid_scenes=int(counts[0]),epoch=epoch,diagnostics=summary,
                    active_global_scenes=global_scenes,active_group=actual_group,active_stage=collection_cfg.get('active_stage'))
                if stage is not None:
                    delta=[self_history.delta_from_scene(r)for r in scenes]
                    delta=[value for part in checkpoint.gather(delta)for value in part if value is not None]
                    self_history.update(state['d6_history'],delta,conditions.tokens,capacity=cfg['historical_teacher_capacity'])
                batch['digest']=material_digest(batch)
                state['collections']+=1;state['scene_cursor']+=global_scenes;state['raw_scene_exposures']+=global_scenes
                state['valid_scene_exposures']+=int(counts[0]);state['padding_scene_exposures']+=global_scenes-int(counts[0])
                state['new_candidates']+=global_scenes*actual_group;state['extra_B1_queries']+=int(counts[1]);state['adaptive_G32_queries']+=int(counts[2])
                state['valid_PG_candidates']+=int(counts[0])*actual_group
                state['phase']='ROLLOUT_COMMITTED';state['next_inner']=0
                runtime.validate_ranks(checkpoint.gather(runtime.validate_boundary(state,batch,identity)))
                path=root/'behavior'/attempt.id/f'collection{state["collections"]:06}'/f'rank{rank}.pt';checkpoint.atomic_torch(path,batch)
            begin=time.monotonic();optimizer.zero_grad(set_to_none=True);totals={};valid_kl_sum=0.
            multiplier=scene_lr(state['valid_scene_exposures']/stream.n,warmup=cfg['warmup_epochs'],hold=cfg['hold_until_epoch'],end=cfg['cosine_end_epoch'],floor=cfg['min_lr_ratio'])
            for group in optimizer.param_groups:
                stage_multiplier=.5 if cfg['arm']=='D1'and batch['epoch']>=2 and group['name']=='text_lora'else 1.
                group['lr']=group['initial_lr']*multiplier*stage_multiplier
                if cfg['arm']=='D6':
                    current_stage=stage_recipe(cfg,batch['epoch'])[1]
                    rate=current_stage.text_lr if group['name']=='text_lora'else current_stage.film_lr if group['name']=='film_lora'else 2e-5 if current_stage.post else 0.
                    warm=min(1.,max(1e-4,(state['valid_scene_exposures']/stream.n)/cfg['warmup_epochs']))
                    group['lr']=rate*warm
            for j,r in enumerate(batch['scenes']):
                r['expect_fresh']=state['next_inner']==0
                sync=ddp.no_sync()if world>1 and j<len(batch['scenes'])-1 else contextlib.nullcontext()
                with sync:
                    loss,metrics=ddp(r)
                    weighted=loss*(0. if r['padding']else world/batch['valid_scenes']);weighted.backward()
                for key,value in metrics.items():totals[key]=totals.get(key,0.)+float(value.detach())/len(batch['scenes'])
                if not r['padding']:valid_kl_sum+=float(metrics['reference_KL_joint'].detach())
            if cfg['arm']=='D6':
                for group in optimizer.param_groups:
                    if group['lr']==0:
                        for parameter in group['params']:parameter.grad=None
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
            row['exploration']=batch['diagnostics']
            row['active_batch']=dict(scenes=batch['active_global_scenes'],group=batch['active_group'],stage=batch['active_stage'])
            row['model_work']=dict(velocity_calls=actor._velocity_calls,reference_velocity_calls=actor._reference_calls,
                current_prefix_builds=actor.prepared_context_builds['current'],reference_prefix_builds=actor.prepared_context_builds['reference'],
                frozen_vision_prefills=conditions.prefill_calls,frozen_vision_prefill_seconds=conditions.prefill_seconds)
            journal.write(json.dumps(row,allow_nan=False)+'\n')
            print(json.dumps(dict(event='optimizer_committed',rank=rank,updates=state['updates'],collection=state['collections'],
                scene_epochs=row['scene_epochs'],grad_norm=float(norm),fresh_ratio_error=totals['fresh_ratio_error'])),flush=True)
            if state['next_inner']==cfg['inner_updates']:
                kval=torch.tensor(valid_kl_sum,device=device,dtype=torch.float64)
                if world>1:dist.all_reduce(kval)
                kval/=batch['valid_scenes']
                dual.update(float(kval));state['adaptive_kl']=dual.state_dict();batch=None;state['next_inner']=0;state['phase']='READY'
            gates=[]if batch is not None or a.qualification else [e for e in(.5,1,2,3,4,6,8,12,20)if e not in state['evaluation_gates']and state['valid_scene_exposures']>=e*stream.n]
            state['evaluation_gates']+=gates
            if a.fault=='after_inner1'and state['next_inner']==1:stop()
            stopping=consensus_stop(device)or(a.stop_after_update is not None and state['updates']>=a.stop_after_update)
            if stopping or gates or state['updates']%cfg['save_every']==0 or(a.qualification and state['updates']==1):last_saved=save()
            if gates and rank==0:
                for gate in gates:atomic_json(root/'evaluation_requests'/f'epoch{gate:g}.json',dict(epoch=gate,checkpoint=last_saved,
                    source_run=identity['sha256'],updates=state['updates'],seed=2101,protocol='single_B1_cleanx_ODE_K10_shift5_canonical_v1.1',test_selected_research=True))
            if stopping:break
        if last_saved is None or json.loads((Path(last_saved)/'COMPLETE').read_text())['updates']!=state['updates']:last_saved=save()
        result=dict(state,status='QUALIFICATION_STOPPED'if a.qualification else'COMPLETE'if state['valid_scene_exposures']>=cfg['target_epochs']*stream.n and batch is None else'STOPPED_AT_SAVED_BOUNDARY',
            checkpoint=last_saved,rank=rank,host=host,seconds=time.monotonic()-started,qualification=a.qualification,scope=actor.closure)
        if a.qualification:
            after=frozen_parameter_hash(actor)
            if after!=frozen_before:raise ValueError('Frozen D44/reference parameter bytes changed')
            result.update(frozen_before=frozen_before,frozen_after=after)
        result_json=result if extension is None else extension.json_result(result)
        atomic_json(root/f'result_rank{rank}.json',result_json);attempt.finish(state,result['status'])
    except BaseException:
        failure_error=traceback.format_exc()
        failure_state=state if extension is None else extension.json_result(state)
        checkpoint.failure_evidence(root,failure_state,failure_error,attempt.id,rollout=batch);attempt.finish(state,'ABORTED');raise
    finally:
        journal.close();service.close()
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
