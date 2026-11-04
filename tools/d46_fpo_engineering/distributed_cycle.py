"""Finite full B64/G16/M2 benchmark using the native optimizer implementation.

Native actor/Adam/RNG are loaded into disposable processes. Formal outputs are
read-only. No benchmark policy is exported or adopted as a trained result.
"""
import argparse
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import shutil
import time


def main():
    import faulthandler
    faulthandler.enable();faulthandler.dump_traceback_later(300,repeat=True)
    p=argparse.ArgumentParser(__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--cycles',type=int,default=2);p.add_argument('--modes',nargs='+',default=['baseline','exact','fast']);a=p.parse_args()
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from starVLA.rl.d44_native_rl import parent
    from starVLA.rl.d44_native_rl.actor import D44Actor
    from starVLA.rl.d44_native_rl.config import source_config
    from starVLA.rl.d44_native_rl.conditions import DConditions
    from starVLA.rl.d44_native_rl.data import DataStream
    from starVLA.rl.d44_native_rl.batch_runtime import collect_scene
    from starVLA.rl.d44_native_rl.safety_credit import raw_credit
    from starVLA.rl.d44_native_rl.expanded_scope import groups
    from starVLA.rl.d44_native_rl.adaptive_kl import AdaptiveKL
    from starVLA.rl.d44_native_rl.update import optimizer_update
    from starVLA.rl.d44_native_rl.recovery import checkpoint
    from starVLA.rl.d44_native_rl.recovery import material_digest
    from starVLA.rl.d46_world16.native import World16Boundary
    from starVLA.rl.d46_world16.runtime import World16Extension,VirtualSceneStream
    from starVLA.rl.d46_fpo_verified.runtime import VerifiedExtension,VerifiedBoundary
    from starVLA.rl.d46_fpo_engineering.joint_flow import JointVerifiedActor,OptimizedVerifiedCollection
    from starVLA.rl.d46_fpo_engineering.prefix_cache import collection_prefix_cache
    from starVLA.rl.d46_fpo_engineering.frozen_weights import FrozenLinearCache
    from starVLA.rl.d44_native_rl.execution_cache import install_invariant_cache
    from starVLA.rl.simwam_recipe.reward_readouts import RewardService
    from starVLA.rl.simwam_recipe.lora_adapter import restore_adapters
    from starVLA.rl.flow_grpo.reproducibility import configure_numerics
    from starVLA.rl.flow_grpo.transactions import atomic_json
    from starVLA.rl.qwen_world_compare.contracts import fingerprint
    from tools.d44_native_rl.train import SceneScheduler
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');configure_numerics();torch.set_num_threads(2)
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    dist.init_process_group('nccl',timeout=timedelta(minutes=20))
    if world==16:World16Extension.init_control_group()
    else:VerifiedExtension.init_control_group()
    cp=Path(a.checkpoint);identity=json.loads((cp/'identity.json').read_text());cfg=identity['recipe']
    assert identity['world']==world and(cfg['global_scenes'],cfg['group'],cfg['inner_updates'])==(64,16,2)
    assert 1<=a.cycles<=4 and a.modes[0]=='baseline'and len(a.modes)<=3 and len(set(a.modes))==len(a.modes)
    native=torch.load(cp/'optimizer.pt',map_location='cpu',weights_only=False)
    own=torch.load(cp/f'rank{rank}.pt',map_location='cpu',weights_only=False)
    verified='verified_improvement'in cfg
    model,info=parent.load(cfg['parent_checkpoint'],device,readouts=False,binding=cfg.get('parent_registration'))
    actor=(JointVerifiedActor if verified else D44Actor)(model,source_config(cfg),cfg)
    for policy in(actor.action_expert,actor.ref_action_expert):
        policy.decision.activation_checkpointing=cfg['checkpoint_policy']!='all_off'
        policy.decision.checkpoint_short_suffix=cfg['checkpoint_policy']=='source'
    stream=DataStream(cfg,64)
    if world==16:stream=VirtualSceneStream(stream)
    conditions=DConditions(model,stream.data,info,group=16,capacity=8);object.__setattr__(actor,'conditions',conditions)
    service=RewardService(Path.cwd(),workers=cfg['scoring_workers'],cache_roots=stream.data.manifest['canonical_metric_cache_roots'])
    ddp=DDP(actor,device_ids=[local],broadcast_buffers=False,find_unused_parameters=True,static_graph=False,bucket_cap_mb=25)
    params=[v for v in actor.parameters()if v.requires_grad];output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    reference=None;rows=[]
    def update(optimizer,scheduler,state,batch,bulk):
        for r in batch['scenes']:r['expect_fresh']=state['next_inner']==0
        value=optimizer_update(ddp,actor,optimizer,scheduler,batch,cfg,state['valid_scene_exposures']/103288,world,params,bulk_metrics=bulk)
        state['updates']+=1;state['next_inner']+=1
        return value
    def end_collection(state,dual,valid_kl,valid_scenes):
        kl=torch.tensor(valid_kl,device=device,dtype=torch.float64);dist.all_reduce(kl);dual.update(float(kl/valid_scenes))
        state['adaptive_kl']=dual.state_dict();state['next_inner']=0;state['phase']='READY'
    try:
        for mode in a.modes:
            assert mode in('baseline','exact','fast')
            restore_adapters(actor,native['adapters']);actor.zero_grad(set_to_none=True)
            conditions.lru.clear();conditions.token=None;conditions.cached=None
            optimizer=torch.optim.AdamW(groups(actor,cfg),betas=(.9,.95),eps=1e-8,weight_decay=.01)
            optimizer.load_state_dict(copy.deepcopy(native['optimizer']));scheduler=SceneScheduler(optimizer);scheduler.load_state_dict(native['scheduler'])
            state=copy.deepcopy(own['state']);rng=own['rng'];random.setstate(rng['python']);np.random.set_state(rng['numpy'])
            torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state(rng['cuda'])
            generator=torch.Generator().set_state(rng['rollout_generator'])
            dual=AdaptiveKL();dual.load_state_dict(state['adaptive_kl'])
            actor.execution_fast_prefix=verified;actor.execution_fuse_fm_mc=False;actor.execution_joint_flow_aux=False
            caches=[]
            if verified:
                for policy in(actor.action_expert,actor.ref_action_expert):caches.append(install_invariant_cache(policy.decision,rotary=True,additive=True))
            pending=checkpoint.device_tree(copy.deepcopy(own['rollout']),device)
            if pending is not None:
                for scene in pending['scenes']:scene['condition']=conditions.get(scene['token'])
                for _ in range(2-state['next_inner']):value=update(optimizer,scheduler,state,pending,False)
                end_collection(state,dual,value['valid_kl_sum'],pending['valid_scenes'])
            for cache in reversed(caches):cache.close()
            caches=[]
            actor.execution_fast_prefix=verified or mode!='baseline';actor.execution_fuse_fm_mc=mode=='fast';actor.execution_joint_flow_aux=mode=='fast'
            if verified and mode in('baseline','exact'):
                for policy in(actor.action_expert,actor.ref_action_expert):caches.append(install_invariant_cache(policy.decision,rotary=True,additive=True))
            if mode!='baseline':caches.append(FrozenLinearCache(actor))
            torch.cuda.reset_peak_memory_stats()
            if verified:
                extension=VerifiedExtension()if mode=='baseline'else OptimizedVerifiedCollection();extension.cfg=cfg;extension.stream=stream
            else:extension=World16Extension();extension.saved_identity=identity
            cycles=[]
            for iteration in range(a.cycles):
                dist.barrier();torch.cuda.synchronize();tick=time.perf_counter()
                epoch=state['valid_scene_exposures']/103288;version=dict(run=identity['sha256'],update=state['updates'],collection=state['collections']+1)
                actor.set_policy_version(str(version));extension.before_collection(state,generator,rank,world);scenes=[];cost={}
                def add(**values):
                    for k,v in values.items():cost[k]=cost.get(k,0)+v
                collection_start=time.perf_counter()
                for j in range(64//world):
                    token,padding=stream.at(state['scene_cursor'],rank,j,world);descriptor=conditions.get(token)
                    extension.before_scene(generator,j)
                    if verified:r=extension.collect_scene(actor,descriptor,token,stream.logs[token],service,generator,cfg,epoch,version,cost=add)
                    elif mode=='baseline':r=collect_scene(actor,descriptor,token,stream.logs[token],service,generator,cfg,epoch,version,cost=add)
                    else:
                        with collection_prefix_cache(actor):r=collect_scene(actor,descriptor,token,stream.logs[token],service,generator,cfg,epoch,version,cost=add,validate_sample_mean=False)
                    extension.after_scene(generator,j);r.update(padding=padding,beta=dual.beta);scenes.append(r)
                extension.after_collection(state,generator,rank,world)
                raw,_=raw_credit(torch.stack([r['factors']for r in scenes]));valid=torch.tensor([not r['padding']for r in scenes],device=device)
                stats=torch.stack(((raw*valid[:,None]).sum(),(raw.square()*valid[:,None]).sum(),valid.sum().double()*16));dist.all_reduce(stats)
                std=(stats[1]/stats[2]-(stats[0]/stats[2]).square()).clamp_min(0).sqrt()
                for r,adv in zip(scenes,(raw/(std+1e-8)).clamp(-5,5).float()):r['advantages']=adv.detach()
                counts=torch.tensor([int(valid.sum()),sum(r['extra_B1']for r in scenes),sum(r['extra_G32']for r in scenes)],device=device);dist.all_reduce(counts)
                state['collections']+=1;state['scene_cursor']+=64;state['raw_scene_exposures']+=64;state['valid_scene_exposures']+=int(counts[0]);state['padding_scene_exposures']+=64-int(counts[0])
                state['new_candidates']+=1024;state['valid_PG_candidates']+=int(counts[0])*16;state['extra_B1_queries']+=int(counts[1]);state['adaptive_G32_queries']+=int(counts[2])
                batch=dict(scenes=scenes,epoch=epoch,valid_scenes=int(counts[0]));torch.cuda.synchronize();collection_seconds=time.perf_counter()-collection_start
                batch.update(behavior_update=state['updates'],active_global_scenes=64,active_group=16,active_stage=None)
                batch['digest']=material_digest(batch);state['phase']='ROLLOUT_COMMITTED'
                boundary=VerifiedBoundary()if verified else World16Boundary()
                boundary.validate_ranks(checkpoint.gather(boundary.validate_boundary(state,batch,identity)))
                archive_start=time.perf_counter()
                checkpoint.atomic_torch(output/'temporary_behavior'/mode/f'collection{iteration}'/f'rank{rank}.pt',batch)
                archive_seconds=time.perf_counter()-archive_start
                updates=[]
                for _ in range(2):updates.append(update(optimizer,scheduler,state,batch,mode!='baseline'))
                end_collection(state,dual,updates[-1]['valid_kl_sum'],int(counts[0]));dist.barrier();torch.cuda.synchronize()
                row=dict(iteration=iteration,seconds=time.perf_counter()-tick,collection_seconds=collection_seconds,
                    behavior_archive_seconds=archive_seconds,
                    optimizer_seconds=[x['seconds']for x in updates],cost=cost,source_fresh_ratio_errors=[x['totals'].get('fresh_ratio_error')for x in updates])
                cycles.append(row);print(json.dumps(dict(rank=rank,mode=mode,cycle=row)),flush=True)
            current={n:p.detach().cpu().clone()for n,p in actor.named_parameters()if p.requires_grad}
            if reference is None:reference=current
            exact=all(torch.equal(reference[n],v)for n,v in current.items())
            maximum=max(float((reference[n]-v).abs().max())for n,v in current.items())
            # Exercise the actual native transaction and restore with an
            # explicitly disposable execution identity. Scientific teacher
            # versions keep their original own-run lineage.
            from tools.d44_native_rl.train import code_identity
            native_identity=copy.deepcopy(identity);native_identity.pop('sha256',None)
            sources=code_identity()
            for directory in('starVLA/rl/d46_fpo_engineering','tools/d46_fpo_engineering'):
                from starVLA.rl.qwen_world_compare.contracts import file_sha
                sources.update({str(p):file_sha(p)for p in Path(directory).glob('*.py')})
            native_identity.update(qualification=True,fresh_Adam=False,source_files=sources,training_run_id=identity['sha256'],
                engineering_execution=dict(mode=mode,origin_checkpoint=str(cp),formal_continuation=False,disposable=True))
            native_identity['sha256']=fingerprint(native_identity)
            boundary=VerifiedBoundary()if verified else World16Boundary()
            dist.barrier();save_start=time.perf_counter()
            temporary=output/'temporary_native'/mode
            saved=checkpoint.save(temporary,actor,optimizer,scheduler,generator,state,None,native_identity,runtime=boundary)
            save_seconds=time.perf_counter()-save_start
            expected_rng=generator.get_state().clone();expected_cpu=torch.get_rng_state().clone();expected_cuda=torch.cuda.get_rng_state().clone()
            with torch.no_grad():params[0].add_(.001)
            torch.rand(1,generator=generator);torch.rand(1);torch.rand(1,device=device)
            restore_start=time.perf_counter()
            restored,pending=checkpoint.restore(saved,actor,optimizer,scheduler,generator,native_identity,runtime=boundary)
            restore_seconds=time.perf_counter()-restore_start
            restored_parameters={n:p.detach().cpu()for n,p in actor.named_parameters()if p.requires_grad}
            resume_exact=(pending is None and all(torch.equal(current[n],v)for n,v in restored_parameters.items())
                and torch.equal(expected_rng,generator.get_state())and torch.equal(expected_cpu,torch.get_rng_state())
                and torch.equal(expected_cuda,torch.cuda.get_rng_state())and restored['updates']==state['updates'])
            assert resume_exact,'Actual native save/restore changed parameters or RNG'
            rows.append(dict(mode=mode,cycles=cycles,final_parameters_bitexact_to_baseline=exact,parameter_max_abs=maximum,
                final_updates=state['updates'],scene_exposures=state['valid_scene_exposures'],peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                native_save_seconds=save_seconds,native_restore_seconds=restore_seconds,native_roundtrip_exact=resume_exact))
            atomic_json(output/f'rank{rank}.json',dict(status='IN_PROGRESS',rank=rank,world=world,rows=rows,source_checkpoint=str(cp),formal_outputs_modified=False))
            for cache in reversed(caches):cache.close()
            dist.barrier()
            if rank==0:shutil.rmtree(temporary)
            dist.barrier()
        atomic_json(output/f'rank{rank}.json',dict(status='COMPLETE',rank=rank,world=world,rows=rows,source_checkpoint=str(cp),
            disposable_Adam_updates=len(a.modes)*(int(own['rollout']is not None)*(2-own['state']['next_inner'])+a.cycles*2),formal_outputs_modified=False,
            policy_weights_exported=False,scope='full B64/G16/M2 canonical collection + global advantage + DDP/Adam + behavior I/O; native checkpoint timed separately'))
        dist.barrier()
        if rank==0:
            ranks=[json.loads((output/f'rank{i}.json').read_text())for i in range(world)]
            assert all(r['status']=='COMPLETE'and r['world']==world for r in ranks)
            summary=[]
            for index,mode in enumerate(a.modes):
                measurements=[r['rows'][index]for r in ranks]
                assert all(r['mode']==mode for r in measurements)
                seconds=sum(max(r['cycles'][j]['seconds']for r in measurements)for j in range(a.cycles))
                save=max(r['native_save_seconds']for r in measurements)
                per_update=seconds/(2*a.cycles)
                summary.append(dict(mode=mode,seconds_per_update=per_update,updates_per_hour=3600/per_update,
                    updates_per_hour_with_save100=3600/(per_update+save/100),native_save_seconds=save,
                    native_restore_seconds=max(r['native_restore_seconds']for r in measurements),
                    native_roundtrip_exact=all(r['native_roundtrip_exact']for r in measurements),
                    final_parameters_bitexact_to_baseline=all(r['final_parameters_bitexact_to_baseline']for r in measurements),
                    parameter_max_abs=max(r['parameter_max_abs']for r in measurements),
                    peak_allocated_bytes=max(r['peak_allocated_bytes']for r in measurements)))
            atomic_json(output/'RESULT.json',dict(status='COMPLETE',world=world,route='verified'if verified else'mature',
                rows=summary,cycles_per_mode=a.cycles,source_checkpoint=str(cp),formal_outputs_modified=False,
                scope='full B64/G16/M2 collection, canonical scoring, global advantage, DDP, Adam and behavior I/O; finite benchmark, not long-run throughput'))
            shutil.rmtree(output/'temporary_behavior')
    finally:service.close();dist.destroy_process_group()


if __name__=='__main__':main()
