"""Bounded CPU-only dense-time evidence audit on immutable v5 input plans.

No solver calls. The original target receipt is retained independently. Full
footprint facts are stored by scene/plan/time; events retain actor identities.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import gzip
import json
import multiprocessing
import os
from pathlib import Path
import resource
import sys
import time
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
import numpy as np
import torch
from safetensors import safe_open
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.decision_local.common import atomic_json,read,digest,identity,source_sha,check_host
from starVLA.dataloader.foresight_dataset import decode_ego
from starVLA.dataloader.local_evidence_scene import load_local_evidence_scene
from starVLA.model.modules.decision_local.recovery_plans import PlanCurve
from starVLA.model.modules.decision_local.local_evidence import extract_local_evidence,SCHEMA,UnknownReason


def worker(task):
    row,labels,anchor,out,ident=task
    torch.set_num_threads(1);cpu=time.process_time();start=time.monotonic()
    old=read(Path(labels)/'records'/(row['token']+'.json'))
    contract=read(Path(labels)/'identity.json');split=contract['split']
    scene=load_local_evidence_scene(anchor,split,row)
    with np.load(Path(labels)/'scenes'/(row['token']+'.npz'),allow_pickle=False) as f:
        plans=f['plan_in_physical'].copy()
    with safe_open(old['source_feature_chunk'],framework='pt',device='cpu') as f:
        state=f.get_slice('state')[old['source_feature_offset']].clone()
    velocity=decode_ego(state[None])[0,:2].numpy()/.5
    times=np.linspace(0,4,81);records=[];arrays={k:[] for k in ('values','valid','positive','negative','reasons','support','observed_free')}
    events=[]
    for si,seedplans in enumerate(plans):
        for pi,plan in enumerate(seedplans):
            curve=PlanCurve(plan,velocity,velocity_boundary='measured_vector_v2')
            facts=extract_local_evidence(scene,curve.poses(times),times)
            for key,value in zip(arrays,(facts.values,facts.label_valid,facts.positive_known,facts.negative_known,
                                        facts.unknown_reason,facts.local_constraint_support,facts.observed_free_support)):
                arrays[key].append(value)
            for event in facts.events:events.append({'anchor_index':si,'plan_index':pi,**event})
            case=next(c for c in old['cases'] if c['anchor_index']==si and c['plan_index']==pi)
            records.append({'anchor_index':si,'plan_index':pi,'distribution':'NATURAL' if pi==0 else 'SYNTHETIC',
                'strict_status':case['status_name'],'strict_target_valid':case['target_valid'],
                'old_input_check':case['input_check'],
                'local_road_positive':bool(facts.positive_known[:,0].any()),
                'local_overlap_positive':bool(facts.positive_known[:,2].any()),
                'per_time_support':{str(t):facts.label_valid[int(round(t/.05))].tolist() for t in np.arange(.5,4.01,.5)},
                'prefix_support':{str(t):facts.label_valid[times<=t+1e-9].all(0).tolist() for t in (.5,1.,2.,3.,4.)},
                'unknown_reasons':{reason.name:bool((facts.unknown_reason&int(reason)).any()) for reason in UnknownReason}})
    path=Path(out)/'scenes'/(row['token']+'.npz');tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as stream:np.savez_compressed(stream,times=times,**{k:np.stack(v) for k,v in arrays.items()})
    tmp.replace(path)
    receipt={'token':row['token'],'log':row['log'],'identity':ident,'source_label_record_sha256':digest(Path(labels)/'records'/(row['token']+'.json')),
        'array_sha256':digest(path),'cases':records,'events':events,'scene_quality':scene.quality,
        'worker_CPU_seconds':time.process_time()-cpu,'worker_wall_seconds':time.monotonic()-start,'bytes':path.stat().st_size}
    path=Path(out)/'records'/(row['token']+'.json.gz');tmp=path.with_suffix('.tmp')
    with gzip.open(tmp,'wt') as stream:json.dump(receipt,stream,allow_nan=False)
    tmp.replace(path)
    return {k:receipt[k] for k in ('token','worker_CPU_seconds','worker_wall_seconds','bytes')}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('anchor','labels','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--max-seconds',type=float,required=True);p.add_argument('--cpu-hours-cap',type=float,required=True)
    a=p.parse_args();host=check_host();start=time.monotonic();cpu=time.process_time();before=resource.getrusage(resource.RUSAGE_CHILDREN)
    parent=read(a.labels/'identity.json')
    if parent['constraints'].get('velocity_boundary')!='measured_vector_v2':raise ValueError('Require corrected v5')
    rows=parent['index'][:a.limit or None];rows=sorted(rows,key=lambda r:(r['log'],r['token']))
    contract={'schema':SCHEMA,'source_label_identity':parent['identity'],'index':rows,'source_sha':source_sha(ROOT),
        'source_files':{str(f.relative_to(ROOT)):digest(f) for f in [Path(__file__),ROOT/'starVLA/model/modules/decision_local/local_evidence.py',ROOT/'starVLA/dataloader/local_evidence_scene.py']},
        'times':'0 through 4s at 0.05s','strict_targets':'read-only old v5; no solver calls','split':parent['split']}
    ident=identity(contract);a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'identity.json').exists() and read(a.output/'identity.json')['identity']!=ident:raise ValueError('Preserve prior audit version')
    atomic_json(a.output/'identity.json',dict(contract,identity=ident))
    for name in ('scenes','records'):(a.output/name).mkdir(exist_ok=True)
    atomic_json(a.output/'COMMAND.json',dict(argv=sys.argv,resource=host,max_seconds=a.max_seconds,cpu_hours_cap=a.cpu_hours_cap))
    done=[];tasks=[];errors=[];anchor=read(a.anchor)
    for row in rows:
        path=a.output/'records'/(row['token']+'.json.gz')
        if path.exists():
            with gzip.open(path,'rt') as stream:r=json.load(stream)
            if r['identity']!=ident or digest(a.output/'scenes'/(row['token']+'.npz'))!=r['array_sha256']:raise ValueError('Changed evidence artifact')
            done.append({k:r[k] for k in ('token','worker_CPU_seconds','worker_wall_seconds','bytes')})
        else:tasks.append((row,str(a.labels),anchor,str(a.output),ident))
    with ProcessPoolExecutor(a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        pending={};it=iter(tasks)
        def fill():
            while len(pending)<a.workers and time.monotonic()-start<a.max_seconds and sum(r['worker_CPU_seconds'] for r in done)<a.cpu_hours_cap*3600 and not (a.output/'STOP_REQUESTED').exists():
                task=next(it,None)
                if task is None:break
                pending[pool.submit(worker,task)]=task[0]['token']
        fill()
        while pending:
            ready,_=wait(pending,timeout=10,return_when=FIRST_COMPLETED)
            for future in ready:
                token=pending.pop(future)
                try:done.append(future.result())
                except Exception as exc:errors.append(dict(token=token,error=repr(exc)))
            atomic_json(a.output/'STATUS.json',dict(completed=len(done),expected=len(rows),errors=errors[-5:],wall_seconds=time.monotonic()-start))
            if errors:break
            fill()
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    complete=len(done)==len(rows) and not errors
    result=dict(identity=ident,completed=len(done),expected=len(rows),errors=errors,
        wall_seconds=time.monotonic()-start,CPU_parent_seconds=time.process_time()-cpu,
        CPU_waited_children_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
        worker_CPU_seconds=sum(r['worker_CPU_seconds'] for r in done),records=done,
        scope='worker CPU and waited-child CPU are overlapping measurements; do not sum them')
    atomic_json(a.output/('COMPLETE.json' if complete else 'INCOMPLETE.json'),result)
    print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
    if not complete:raise SystemExit(1)


if __name__=='__main__':main()
