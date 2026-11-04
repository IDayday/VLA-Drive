"""Real saved rank F/B with gradient comparison; no Adam updates or weights output."""
import argparse
from contextlib import ExitStack
import copy
import json
import os
from pathlib import Path
import statistics
import time


def main():
    import faulthandler
    faulthandler.enable();faulthandler.dump_traceback_later(180,repeat=True)
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--modes',nargs='+',default=['baseline','invariant','fused'])
    p.add_argument('--rank',type=int,default=0);p.add_argument('--iterations',type=int,default=2)
    p.add_argument('--profile',action='store_true');a=p.parse_args()
    import torch
    from starVLA.rl.d44_native_rl import parent
    from starVLA.rl.d44_native_rl.actor import D44Actor
    from starVLA.rl.d46_fpo_engineering.joint_flow import JointVerifiedActor
    from starVLA.rl.d44_native_rl.config import source_config
    from starVLA.rl.d44_native_rl.conditions import DConditions
    from starVLA.rl.d44_native_rl.data import DataStream
    from starVLA.rl.d44_native_rl.recovery import checkpoint,material_digest
    from starVLA.rl.simwam_recipe.lora_adapter import restore_adapters
    from starVLA.rl.flow_grpo.reproducibility import configure_numerics
    from starVLA.rl.flow_grpo.transactions import atomic_json
    from starVLA.rl.d44_native_rl.execution_cache import install_invariant_cache
    from starVLA.rl.d46_fpo_engineering.frozen_weights import FrozenLinearCache
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');configure_numerics()
    torch.set_num_threads(2);torch.cuda.set_device(0);device=torch.device('cuda',0)
    cp=Path(a.checkpoint);identity=json.loads((cp/'identity.json').read_text());cfg=identity['recipe']
    assert cfg['arm']=='D3_FPO'and 1<=a.iterations<=8 and a.modes[0]=='baseline'
    own=torch.load(cp/f'rank{a.rank}.pt',map_location='cpu',weights_only=False)
    print(json.dumps(dict(event='native_material_loaded',at=time.time())),flush=True)
    assert own['identity_sha256']==identity['sha256']and own['rollout']is not None
    batch=own['rollout'];assert batch['digest']==material_digest(batch)
    model,info=parent.load(cfg['parent_checkpoint'],device,readouts=False,binding=cfg.get('parent_registration'))
    print(json.dumps(dict(event='parent_loaded',at=time.time())),flush=True)
    verified='verified_improvement'in cfg
    actor=(JointVerifiedActor if verified else D44Actor)(model,source_config(cfg),cfg)
    for policy in(actor.action_expert,actor.ref_action_expert):
        policy.decision.activation_checkpointing=cfg['checkpoint_policy']!='all_off'
        policy.decision.checkpoint_short_suffix=cfg['checkpoint_policy']=='source'
    payload=torch.load(cp/'optimizer.pt',map_location='cpu',weights_only=False)
    restore_adapters(actor,payload['adapters']);del payload
    actor.set_policy_version(str(batch['scenes'][0]['behavior_version']))
    stream=DataStream(cfg,cfg['global_scenes']);conditions=DConditions(model,stream.data,info,group=cfg['group'],capacity=max(8,len(batch['scenes'])))
    object.__setattr__(actor,'conditions',conditions)
    batch=checkpoint.device_tree(batch,device)
    for scene in batch['scenes']:
        saved=scene['condition'];current=conditions.get(scene['token'])
        assert {k:v for k,v in saved.items()if k!='identity'}=={k:v for k,v in current.items()if k!='identity'}
        scene['condition']=current;scene['expect_fresh']=own['state']['next_inner']==0
        conditions.materialize(current,device)
    params={n:x for n,x in actor.named_parameters()if x.requires_grad}
    versions={n:x._version for n,x in params.items()};baseline=None;rows=[]
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    def step():
        actor.zero_grad(set_to_none=True);values=[]
        for scene in batch['scenes']:
            with torch.profiler.record_function('FPO_forward'):
                loss,_=actor(scene)
            with torch.profiler.record_function('FPO_backward'):
                (loss/len(batch['scenes'])).backward()
            values.append(float(loss.detach()))
        return values
    for mode in a.modes:
        assert mode in('baseline','lean','invariant','fused','joint','weights','weights_joint','weights_fused')
        actor.execution_fast_prefix=verified or mode!='baseline'
        actor.execution_fuse_fm_mc=mode in('fused','joint','weights_joint','weights_fused')
        actor.execution_joint_flow_aux=mode in('joint','weights_joint')
        caches=[]
        if(mode in('baseline','weights')and verified)or mode=='invariant':
            for decision in(actor.action_expert.decision,actor.ref_action_expert.decision):
                caches.append(install_invariant_cache(decision,rotary=True,additive=True))
        weight_cache=FrozenLinearCache(actor)if mode.startswith('weights')else None
        if weight_cache:caches.append(weight_cache)
        try:
            step();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
            for _ in range(a.iterations):
                tick=time.perf_counter();losses=step();torch.cuda.synchronize();times.append(time.perf_counter()-tick)
            print(json.dumps(dict(event='timed_steps_finished',mode=mode,seconds=times,at=time.time())),flush=True)
            gradients={n:x.grad.detach().cpu().clone()if x.grad is not None else None for n,x in params.items()}
            if baseline is None:baseline={'gradients':gradients,'losses':losses}
            error=refnorm=newnorm=dot=0.;maximum=0.;exact=True;missing=[]
            for name,x in baseline['gradients'].items():
                y=gradients[name]
                if(x is None)!=(y is None):missing.append(name)
                if x is None or y is None:continue
                diff=(x-y).double();error+=float(diff.square().sum());refnorm+=float(x.double().square().sum())
                newnorm+=float(y.double().square().sum());dot+=float((x.double()*y.double()).sum())
                maximum=max(maximum,float(diff.abs().max()));exact&=torch.equal(x,y)
            row=dict(mode=mode,seconds=times,median_seconds=statistics.median(times),scenes=len(batch['scenes']),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),losses=losses,
                comparison=dict(gradient_relative_L2=(error/max(refnorm,1e-30))**.5,gradient_cosine=dot/max((refnorm*newnorm)**.5,1e-30),
                    gradient_max_abs=maximum,bitexact=exact,missing_gradients=missing,loss_max_abs=max(abs(x-y)for x,y in zip(losses,baseline['losses']))))
            if weight_cache:row['frozen_weight_cache']=dict(bytes=weight_cache.bytes,hits=weight_cache.hits,misses=weight_cache.misses)
            print(json.dumps(dict(event='timing_measured',mode=mode,seconds=times,at=time.time())),flush=True)
            if a.profile and mode=='baseline':
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA])as prof:
                    actor.zero_grad(set_to_none=True)
                    loss,_=actor(batch['scenes'][0]);loss.backward();torch.cuda.synchronize()
                entries=prof.key_averages()
                row['profiler_top_GPU']=[dict(name=e.key,calls=e.count,cpu_us=e.cpu_time_total,GPU_us=e.device_time_total)for e in sorted(entries,key=lambda e:e.device_time_total,reverse=True)[:20]]
                row['profiler_syncs']=[dict(name=e.key,calls=e.count,cpu_us=e.cpu_time_total)for e in entries if e.key in('aten::item','aten::_local_scalar_dense','cudaDeviceSynchronize','cudaStreamSynchronize')]
            assert not missing and all(torch.isfinite(x).all()for x in gradients.values()if x is not None)
            assert versions=={n:x._version for n,x in params.items()}
            rows.append(row)
            atomic_json(output/'RESULT.json',dict(status='IN_PROGRESS',route='verified'if verified else'mature',checkpoint=str(cp),rows=rows,Adam_updates=0))
            print(json.dumps(row),flush=True)
        finally:
            for cache in reversed(caches):cache.close()
    atomic_json(output/'RESULT.json',dict(status='COMPLETE',route='verified'if verified else'mature',checkpoint=str(cp),rows=rows,
        Adam_updates=0,scope='actual saved per-rank forward/backward only; excludes collection/scoring/DDP/optimizer/checkpoint',
        trainable_parameters=sum(x.numel()for x in params.values())))


if __name__=='__main__':main()
