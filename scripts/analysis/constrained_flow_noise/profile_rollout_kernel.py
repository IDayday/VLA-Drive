"""One real saved TRAIN condition: inspect warm unchanged B1 rollout execution."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from torch.profiler import profile,ProfilerActivity

from starVLA.rl.constrained_flow_noise.contracts import require_dispatch,file_sha
from starVLA.rl.constrained_flow_noise.loading import make_noise_policy
from starVLA.rl.constrained_flow_noise.rollout_graph import configure
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.transactions import atomic_json


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--checkpoint',required=True)
    p.add_argument('--collection',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();host=require_dispatch();configure_numerics();torch.cuda.set_device(0)
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    cp=Path(a.checkpoint);identity=json.loads((cp/'identity.json').read_text())
    model=cp/'pytorch_model/mp_rank_00_model_states.pt'
    if file_sha(model)!=json.loads((cp/'files.json').read_text())[str(model.relative_to(cp))]:raise ValueError('Changed model')
    actor,_,_=make_noise_policy(identity['variant'],(identity['kernel']['min_std'],identity['kernel']['max_std']))
    actor.load_state_dict(torch.load(model,map_location='cpu',mmap=True,weights_only=False)['module'],strict=True)
    actor.requires_grad_(False).eval();configure(actor,True)
    collection=Path(a.collection)
    if file_sha(collection)!=json.loads(collection.with_suffix('.json').read_text())['sha256']:raise ValueError('Changed collection')
    row=torch.load(collection,map_location='cpu',weights_only=False)['rows'][0]
    raw=row['chain']['raw'].cuda()
    def sample():return actor.sample_chain(raw,[row['token']],[row['position']],version={'read_only_profile':True})
    original=sample();times=[]
    for _ in range(3):
        t0=time.monotonic();current=sample();torch.cuda.synchronize();times.append(time.monotonic()-t0)
        for k in ('chain','old_mean','old_std','old_elementwise_logp'):
            if not torch.equal(original[k],current[k]):raise AssertionError('Nondeterministic profile chain')
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=True) as prof:
        sample();torch.cuda.synchronize()
    rows=[]
    for event in prof.key_averages():
        rows.append(dict(name=event.key,calls=event.count,self_CPU_ms=event.self_cpu_time_total/1000,
                         CPU_total_ms=event.cpu_time_total/1000,
                         self_GPU_ms=getattr(event,'self_device_time_total',0)/1000,
                         GPU_total_ms=getattr(event,'device_time_total',0)/1000))
    atomic_json(out,dict(status='COMPLETE_READONLY_PROFILE',host=host,variant=identity['variant'],
                        checkpoint=str(cp),warm_seconds=times,velocity_calls_per_scene=160,
                        top_self_CPU=sorted(rows,key=lambda x:x['self_CPU_ms'],reverse=True)[:30],
                        top_self_GPU=sorted(rows,key=lambda x:x['self_GPU_ms'],reverse=True)[:30],
                        source_sha256=file_sha(__file__),no_scores_or_optimizer=True,
                        limitation='Profiler overhead excluded from warm timers; one fixed TRAIN scene, not an overall throughput claim'))
    print(json.dumps(dict(status='COMPLETE',warm_seconds=times,output=str(out))),flush=True)


if __name__=='__main__':main()
