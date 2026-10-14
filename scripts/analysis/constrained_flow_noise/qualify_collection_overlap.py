"""Real full-head, chain/score/RNG equality and paired collection wall timings."""
import argparse
import copy
import json
import os
from pathlib import Path
import statistics
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch

from starVLA.rl.constrained_flow_noise.condition_cache import ConditionStore
from starVLA.rl.constrained_flow_noise.contracts import file_sha,require_dispatch
from starVLA.rl.constrained_flow_noise.effective_group_collector import EffectiveGroupCollector
from starVLA.rl.constrained_flow_noise.loading import make_noise_policy
from starVLA.rl.constrained_flow_noise.rollout_graph import configure
from starVLA.rl.constrained_flow_noise.runtime import collect_batch as serial,OrderedScenes
from starVLA.rl.constrained_flow_noise.safety_costs import ConstrainedRewardService
from starVLA.rl.flow_grpo.checkpoint import capture_rng,restore_rng
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.transactions import atomic_json
from .pipeline_collection import collect_batch as overlap


TIMINGS={'rollout_seconds','scoring_seconds','reference_seconds','scoring_cpu_seconds',
         'scene_scoring_cpu_seconds','scene_scoring_wall_seconds','geometry_seconds',
         'score_wait_seconds','collection_execution'}


def exact(a,b,path='root'):
    if isinstance(a,dict):
        keys=set(a)-TIMINGS
        if keys != set(b)-TIMINGS:raise AssertionError(path+' keys')
        for k in sorted(keys):exact(a[k],b[k],path+'.'+str(k))
    elif isinstance(a,(list,tuple)):
        if len(a)!=len(b):raise AssertionError(path+' length')
        for i,(x,y) in enumerate(zip(a,b)):exact(x,y,path+'.'+str(i))
    elif isinstance(a,torch.Tensor):
        if a.dtype!=b.dtype or not torch.equal(a,b):raise AssertionError(path+' tensor')
    elif isinstance(a,np.ndarray):
        if a.dtype!=b.dtype or not np.array_equal(a,b):raise AssertionError(path+' array')
    elif a!=b:raise AssertionError(path+' value')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run-root',required=True)
    p.add_argument('--output',required=True);p.add_argument('--repeats',type=int,default=4)
    p.add_argument('--fast-scoring',action='store_true',help='Compare original serial scoring against overlap plus vector geometry')
    p.add_argument('--frozen-casts',action='store_true',help='Also use the qualified frozen Linear cast cache on the optimized behavior')
    a=p.parse_args();host=require_dispatch();configure_numerics();torch.cuda.set_device(0)
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    root=Path(a.run_root)
    cal=json.loads((root/'calibration/calibration_lock.json').read_text())
    pop=json.loads(Path('/mnt/project/flow-safe-improvement-20261013/population/manifest.json').read_text())
    cache=root/'condition_cache'
    store=ConditionStore(cache,json.loads((cache/'identity.json').read_text()),readonly=True)
    result=dict(status='RUNNING',host=host,source_sha256=file_sha(__file__),
                adapter_sha256=file_sha(Path(__file__).with_name('pipeline_collection.py')),variants={},
                scope='TRAIN-only real full-head execution benchmark; no optimizer or new formal exposure')
    atomic_json(out,result)
    for variant,raw_position in [('C_SDE',12760),('N_FLOW_NOISE',12744)]:
        reference,_,_=make_noise_policy(variant,cal['noise_std_bounds'])
        if reference.noise_head is not None:
            reference.noise_head.load_state_dict(torch.load(cal['noise_initial_path'],map_location='cpu',weights_only=True))
        reference.requires_grad_(False).eval()
        actor=copy.deepcopy(reference)
        cp=root/f'formal_{variant}_seed42_v1/checkpoints'/f'update001024_raw{raw_position:06d}'
        identity=json.loads((cp/'identity.json').read_text())
        model=cp/'pytorch_model/mp_rank_00_model_states.pt'
        if file_sha(model)!=json.loads((cp/'files.json').read_text())[str(model.relative_to(cp))]:raise ValueError('Changed checkpoint')
        actor.load_state_dict(torch.load(model,map_location='cpu',mmap=True,weights_only=False)['module'],strict=True)
        configure(actor,True)
        if a.frozen_casts:
            from .frozen_weight_cast import install
            optimized_actor=copy.deepcopy(actor)
            casts=install(optimized_actor);configure(optimized_actor,True)
        else:optimized_actor=actor
        spec=dict(repo=str(Path.cwd()),maps_root='/mnt/navsim/maps',
            cache_roots=['/mnt/project/flow-safe-improvement-20261013/metric_cache_v1',str(root/'metric_cache_v1')],
            metric_manifest=str(root/'full_TRAIN_assets.json'),metric_manifest_sha256=file_sha(root/'full_TRAIN_assets.json'))
        service=ConstrainedRewardService(spec,workers=2)
        if a.fast_scoring:
            from .train_efficient import EfficientRewardService
            optimized_service=EfficientRewardService(spec,workers=2)
        else:optimized_service=service
        stream=OrderedScenes(pop['train_tokens'],42)
        collector=EffectiveGroupCollector(dict(collection=512,benchmark=True,variant=variant),torch.tensor([1.,1.,1.,.1,.1]))
        measurements=[];expected=None
        try:
            # Explicit warmup is counted separately: it warms both CPU workers
            # and the unchanged B1 CUDA graph, never trains on their outcomes.
            warmups=['serial','overlap'] if a.fast_scoring else ['overlap']
            for iteration,name in enumerate([*warmups,*['serial','overlap','overlap','serial'][:a.repeats]]):
                rng=capture_rng();t0=time.monotonic()
                selected_service=service if name=='serial' else optimized_service
                selected_actor=actor if name=='serial' else optimized_actor
                rows=(serial if name=='serial' else overlap)(selected_actor,reference,store,selected_service,stream,pop,collector,
                            raw_position,16,cal,0,8)
                torch.cuda.synchronize();seconds=time.monotonic()-t0
                after=capture_rng()
                if expected is None:expected=rows
                else:exact(expected,rows)
                exact(rng,after,'RNG')
                measurements.append(dict(iteration=iteration,mode=name,warmup=iteration<len(warmups),
                                         seconds=seconds,real_groups=len(rows),candidate_scores=len(rows)*16,
                                         output_and_RNG_exact=True))
                result['variants'][variant]=dict(checkpoint=str(cp),identity=identity['sha256'],measurements=measurements)
                atomic_json(out,result)
                print(json.dumps(dict(variant=variant,**measurements[-1])),flush=True)
        finally:
            service.close()
            if optimized_service is not service:optimized_service.close()
        serial_time=statistics.median(x['seconds'] for x in measurements if not x['warmup'] and x['mode']=='serial')
        overlap_time=statistics.median(x['seconds'] for x in measurements if not x['warmup'] and x['mode']=='overlap')
        result['variants'][variant].update(serial_median_seconds=serial_time,overlap_median_seconds=overlap_time,
                                          collection_speedup=serial_time/overlap_time)
        if a.frozen_casts:casts.close()
        del reference,actor,optimized_actor,expected,rows
        import gc
        gc.collect();torch.cuda.empty_cache()
    result.update(status='PASS',vector_geometry=a.fast_scoring,frozen_linear_casts=a.frozen_casts,
                  actual_candidate_score_calls=sum(x['candidate_scores'] for v in result['variants'].values() for x in v['measurements']),
                  limitation='Paired single-rank collection benchmark; full eight-rank training and restore equivalence still required before deployment')
    atomic_json(out,result)


if __name__=='__main__':main()
