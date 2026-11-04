"""Compare real TRAIN collection including canonical scores, without Adam updates."""
import argparse
import copy
import json
import os
from pathlib import Path
import time


def main():
    import faulthandler
    faulthandler.enable();faulthandler.dump_traceback_later(180,repeat=True)
    p=argparse.ArgumentParser(__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--rank',type=int,default=0);p.add_argument('--scenes',type=int,default=2)
    a=p.parse_args()
    import torch
    from starVLA.rl.d44_native_rl import parent
    from starVLA.rl.d44_native_rl.actor import D44Actor
    from starVLA.rl.d44_native_rl.config import source_config
    from starVLA.rl.d44_native_rl.conditions import DConditions
    from starVLA.rl.d44_native_rl.data import DataStream
    from starVLA.rl.d44_native_rl.batch_runtime import collect_scene
    from starVLA.rl.d46_fpo_verified.runtime import VerifiedExtension
    from starVLA.rl.d46_fpo_engineering.joint_flow import JointVerifiedActor,OptimizedVerifiedCollection
    from starVLA.rl.d46_fpo_engineering.prefix_cache import collection_prefix_cache
    from starVLA.rl.d44_native_rl.execution_cache import install_invariant_cache
    from starVLA.rl.simwam_recipe.lora_adapter import restore_adapters
    from starVLA.rl.simwam_recipe.reward_readouts import RewardService
    from starVLA.rl.flow_grpo.reproducibility import configure_numerics
    from starVLA.rl.flow_grpo.transactions import atomic_json
    from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');configure_numerics();torch.set_num_threads(2)
    torch.cuda.set_device(0);device=torch.device('cuda',0);cp=Path(a.checkpoint)
    identity=json.loads((cp/'identity.json').read_text());cfg=identity['recipe'];verified='verified_improvement'in cfg
    own=torch.load(cp/f'rank{a.rank}.pt',map_location='cpu',weights_only=False)
    assert own['rollout']is not None and 1<=a.scenes<=len(own['rollout']['scenes'])
    print(json.dumps(dict(event='loading_parent',at=time.time())),flush=True)
    model,info=parent.load(cfg['parent_checkpoint'],device,readouts=False,binding=cfg.get('parent_registration'))
    actor=(JointVerifiedActor if verified else D44Actor)(model,source_config(cfg),cfg)
    for policy in(actor.action_expert,actor.ref_action_expert):
        policy.decision.activation_checkpointing=cfg['checkpoint_policy']!='all_off'
        policy.decision.checkpoint_short_suffix=cfg['checkpoint_policy']=='source'
    payload=torch.load(cp/'optimizer.pt',map_location='cpu',weights_only=False);restore_adapters(actor,payload['adapters']);del payload
    stream=DataStream(cfg,64);conditions=DConditions(model,stream.data,info,group=16,capacity=8)
    object.__setattr__(actor,'conditions',conditions)
    inputs=own['rollout']['scenes'][:a.scenes]
    for row in inputs:conditions.materialize(conditions.get(row['token']),device)
    service=RewardService(Path.cwd(),workers=cfg['scoring_workers'],cache_roots=stream.data.manifest['canonical_metric_cache_roots'])
    version=dict(run=identity['sha256'],update=own['state']['updates'],collection=own['state']['collections']+1)
    actor.set_policy_version(str(version));reference=None;results=[];output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    try:
        first=inputs[0];sample=first['chain'][:1,-1]
        # Warm process/map startup using one real saved TRAIN candidate.
        service.score((first['token'],first['log'],ActionCodec.denorm_odo(sample).cpu().numpy()))
        for mode in('baseline','exact_cache','optimized'):
            actor.execution_fast_prefix=verified or mode!='baseline'
            actor.execution_fuse_fm_mc=mode=='optimized';actor.execution_joint_flow_aux=mode=='optimized'
            caches=[]
            if verified and mode=='baseline':
                for policy in(actor.action_expert,actor.ref_action_expert):caches.append(install_invariant_cache(policy.decision,rotary=True,additive=True))
            if verified:
                extension=OptimizedVerifiedCollection()if mode=='optimized'else VerifiedExtension()
                extension.bank=copy.deepcopy(own['state']['verified_history']);extension.rows=[]
                # Uniform epoch-one workload: score initial IL B1 on first visit.
                for row in inputs:
                    extension.bank['rows'].pop(row['token'],None);extension.bank['teachers'].pop(row['token'],None)
            generator=torch.Generator().set_state(own['rng']['rollout_generator'])
            rows=[];cost={};before=dict(actor.prepared_context_builds)
            def add(**values):
                for k,v in values.items():cost[k]=cost.get(k,0)+v
            torch.cuda.synchronize();tick=time.perf_counter()
            try:
                for row in inputs:
                    descriptor=conditions.get(row['token'])
                    if verified:
                        if mode=='exact_cache':
                            with collection_prefix_cache(actor):r=extension.collect_scene(actor,descriptor,row['token'],row['log'],service,generator,cfg,0.,version,cost=add)
                        else:r=extension.collect_scene(actor,descriptor,row['token'],row['log'],service,generator,cfg,0.,version,cost=add)
                    elif mode!='baseline':
                        with collection_prefix_cache(actor):r=collect_scene(actor,descriptor,row['token'],row['log'],service,generator,cfg,0.,version,cost=add,validate_sample_mean=False)
                    else:r=collect_scene(actor,descriptor,row['token'],row['log'],service,generator,cfg,0.,version,cost=add)
                    rows.append(r)
                torch.cuda.synchronize();seconds=time.perf_counter()-tick
                comparisons=[]
                if reference is None:reference=rows;reference_rng=generator.get_state().clone()
                for old,new in zip(reference,rows):
                    comparisons.append({key:float((old[key]-new[key]).abs().max())for key in('chain','old_mean','ref_mean','std','factors','fm_state','fm_sigma','fm_target','old_energy')})
                assert torch.equal(reference_rng,generator.get_state()),'Main sampler RNG changed'
                assert all(all(v==0 for k,v in row.items()if k!='old_energy')for row in comparisons),'Actual trajectories/score/source terms changed'
                value=dict(mode=mode,seconds=seconds,scenes=len(rows),seconds_per_scene=seconds/len(rows),cost=cost,
                    prefix_builds={k:actor.prepared_context_builds[k]-before[k]for k in before},comparison=comparisons,
                    identical_main_sampler_RNG=True,first_visit=True)
                results.append(value);atomic_json(output/'RESULT.json',dict(status='IN_PROGRESS',route='verified'if verified else'mature',rows=results,Adam_updates=0))
                print(json.dumps(value),flush=True)
            finally:
                for cache in reversed(caches):cache.close()
        atomic_json(output/'RESULT.json',dict(status='COMPLETE',route='verified'if verified else'mature',checkpoint=str(cp),rows=results,Adam_updates=0,
            scope='real canonical scoring and native collection only; excludes gradient/optimizer/DDP/checkpoint'))
    finally:service.close()


if __name__=='__main__':main()
