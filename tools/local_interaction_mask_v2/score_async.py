"""Bounded resumable CPU scoring of atomic GPU exports using official full-cache v1 PDMS."""
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import csv
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import lzma
import multiprocessing
import os
from pathlib import Path
import pickle
import sys
import time
import numpy as np


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data)
    return h.hexdigest()


def python_tree_digest(root):
    root=Path(root);digest=hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        digest.update(str(path.relative_to(root)).encode()+b'\0'+path.read_bytes())
    return digest.hexdigest()


def initialize(devkit):
    sys.path.insert(0,devkit)
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'


def score_chunk(items):
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    sampling=TrajectorySampling(num_poses=40,interval_length=.1)
    simulator=PDMSimulator(sampling);scorer=PDMScorer(sampling);rows=[]
    for item in items:
        row={'token':item['token'],'log':item['log'],'variant':item['variant'],'status':'ok',
             'proposal_sha256':item['proposal_sha256']}
        try:
            path=Path(item['proposal_path'])
            if digest(path)!=item['proposal_sha256']:raise ValueError('Proposal changed after export')
            with np.load(path,allow_pickle=False) as f:poses=f['trajectory'].astype(np.float64)
            if poses.shape!=(8,3) or not np.isfinite(poses).all():raise ValueError('Invalid trajectory')
            row['metric_cache_sha256']=digest(item['cache_path'])
            with lzma.open(item['cache_path'],'rb') as f:cache=pickle.load(f)
            if not hasattr(cache,'trajectory'):raise ValueError('This endpoint requires official full reference cache')
            result=pdm_score(cache,Trajectory(poses,TrajectorySampling(num_poses=8,interval_length=.5)),sampling,simulator,scorer)
            row.update({k:float(v) for k,v in asdict(result).items()})
            if not all(np.isfinite(v) and -.0000001<=v<=1.0000001 for v in asdict(result).values()):
                raise ValueError('Invalid official PDM result')
        except Exception as error:row.update(status='failed',error=repr(error),score=0.)
        rows.append(row)
    return rows


def atomic_json(path,value):
    path=Path(path);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('devkit','index','predictions','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--workers',type=int,default=12);p.add_argument('--chunk',type=int,default=8)
    p.add_argument('--export-shards',type=int,default=8);p.add_argument('--timeout-seconds',type=int,default=21600)
    p.add_argument('--log-shard',type=int,default=0);p.add_argument('--log-shards',type=int,default=1)
    p.add_argument('--benchmark-navtest',action='store_true');a=p.parse_args()
    if not 1<=a.workers<=32 or not 1<=a.chunk<=32:raise ValueError('Bound CPU processes and in-flight work')
    if not 0<=a.log_shard<a.log_shards:raise ValueError('Invalid log partition')
    index=json.loads(Path(a.index).read_text());tokens={r['token'] for r in index};logs=sorted({r['log'] for r in index})
    if len(tokens)!=len(index):raise ValueError('Duplicate scoring scenes')
    if a.benchmark_navtest and (len(index)!=12146 or len(logs)!=136):raise ValueError('Incomplete full Navtest endpoint')
    assigned=set(logs[a.log_shard::a.log_shards]);index=[r for r in index if r['log'] in assigned]
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);(out/'records').mkdir(exist_ok=True)
    initialize(a.devkit)
    import nuplan
    bank=Path(a.predictions);identity={'arguments':vars(a),'index_sha256':digest(a.index),
        'evaluator_sha256':digest(Path(a.devkit)/'navsim/evaluate/pdm_score.py'),
        'simulator_sha256':digest(Path(a.devkit)/'navsim/planning/simulation/planner/pdm_planner/simulation/pdm_simulator.py'),
        'scorer_sha256':digest(Path(a.devkit)/'navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py'),
        'navsim_python_tree_sha256':python_tree_digest(Path(a.devkit)/'navsim'),
        'nuplan_python_tree_sha256':python_tree_digest(Path(nuplan.__file__).parent),
        'runtime_versions':{name:importlib.metadata.version(name) for name in ('numpy','scipy','shapely')},
        'protocol':'official NAVSIM v1 full reference cache; one predicted trajectory per scene',
        'failure_policy':'retain all requested rows; failures scored zero and invalidate benchmark'}
    ident=out/'identity.json'
    if ident.exists() and json.loads(ident.read_text())!=identity:raise ValueError('Scoring resume identity changed')
    atomic_json(ident,identity);done={};pending={};start=time.monotonic()
    for r in index:
        f=out/'records'/(r['token']+'.json')
        if f.exists():
            previous=json.loads(f.read_text())
            if previous['status']=='ok':
                if digest(bank/'predictions'/(r['token']+'.npz'))!=previous['proposal_sha256'] or digest(r['cache_path'])!=previous['metric_cache_sha256']:
                    raise ValueError('Immutable scored artifact changed')
            done[r['token']]=previous
    initialize(a.devkit)
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(a.devkit,)) as pool:
        while len(done)<len(index):
            inflight={token for group in pending.values() for token in group}
            ready=[];capacity=max(0,2*a.workers-len(pending))
            for r in (index if capacity else []):
                if r['token'] in done or r['token'] in inflight:continue
                f=bank/'predictions'/(r['token']+'.json')
                if f.exists():
                    exported=json.loads(f.read_text())
                    if exported['status']!='ok':
                        row={'token':r['token'],'log':r['log'],'variant':bank.name,'status':'failed','score':0.,'error':'GPU export: '+exported['error']}
                        done[r['token']]=row;atomic_json(out/'records'/(r['token']+'.json'),row)
                    else:ready.append(dict(r,variant=bank.name,proposal_path=str(f.with_suffix('.npz')),proposal_sha256=exported['proposal_sha256']))
                if len(ready)>=a.chunk*capacity:break
            for i in range(0,len(ready),a.chunk):
                items=ready[i:i+a.chunk];pending[pool.submit(score_chunk,items)]=[r['token'] for r in items]
            if pending:
                finished,_=wait(pending,timeout=2,return_when=FIRST_COMPLETED)
                for future in finished:
                    group=pending.pop(future)
                    try:rows=future.result()
                    except Exception as error:
                        lookup={r['token']:r for r in index}
                        rows=[{'token':t,'log':lookup[t]['log'],'variant':bank.name,'status':'failed','score':0.,'error':repr(error)} for t in group]
                    for row in rows:done[row['token']]=row;atomic_json(out/'records'/(row['token']+'.json'),row)
            elif len(done)<len(index):
                finished_export=all((bank/f'shard_{i}.json').exists() and json.loads((bank/f'shard_{i}.json').read_text())['status']=='complete' for i in range(a.export_shards))
                if finished_export:raise RuntimeError('Export claims complete but requested tokens missing')
                if time.monotonic()-start>a.timeout_seconds:raise TimeoutError('Incomplete export; partial scores retained for resume')
                time.sleep(2)
            atomic_json(out/'progress.json',{'completed':len(done),'requested':len(index),'inflight_chunks':len(pending),'failed':sum(r['status']!='ok' for r in done.values())})
    rows=[done[r['token']] for r in index];keys=sorted(set().union(*(r.keys() for r in rows)))
    with (out/'scenes.csv').open('w') as f:w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
    failed=sum(r['status']!='ok' for r in rows)
    summary={'scenes':len(rows),'logs':len(assigned),'failed':failed,'valid':failed==0,
        'PDMS':sum(r['score'] for r in rows)/len(rows),'zero_fraction':sum(r['score']==0 for r in rows)/len(rows),
        'protocol':identity['protocol'],'log_shard':a.log_shard,'log_shards':a.log_shards,'full_navtest':a.benchmark_navtest and a.log_shards==1}
    atomic_json(out/'summary.json',summary);print(json.dumps(summary),flush=True)
    if failed:raise RuntimeError('Failed scenes retained; PDMS benchmark invalid')


if __name__=='__main__':main()
