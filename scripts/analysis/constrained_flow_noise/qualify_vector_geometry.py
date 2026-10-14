"""Exact body geometry/cost comparison on real saved TRAIN consequences."""
import argparse
import cProfile
import json
from pathlib import Path
import pstats
import time

import numpy as np
import torch

from starVLA.rl.constrained_flow_noise.contracts import require_dispatch,file_sha
from starVLA.rl.constrained_flow_noise import safety_costs as original
from starVLA.rl.safe_flow_improvement import v1_reward
from starVLA.rl.flow_grpo.transactions import atomic_json
from .vector_geometry import body_margins
from .qualify_collection_overlap import exact


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run-root',required=True)
    p.add_argument('--output',required=True);p.add_argument('--groups',type=int,default=32)
    a=p.parse_args();host=require_dispatch();root=Path(a.run_root);out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    spec=dict(repo=str(Path.cwd()),maps_root='/mnt/navsim/maps',
              cache_roots=['/mnt/project/flow-safe-improvement-20261013/metric_cache_v1',str(root/'metric_cache_v1')])
    spec['protocol_sha256']=v1_reward.protocol_identity(spec['repo'])['sha256'];v1_reward.initialize(spec)
    cal=json.loads((root/'calibration/calibration_lock.json').read_text())
    path=root/'formal_N_FLOW_NOISE_seed42_v1/behavior_collections/001100.pt'
    digest=file_sha(path)
    if digest!=json.loads(path.with_suffix('.json').read_text())['sha256']:raise ValueError('Changed archive')
    rows=torch.load(path,map_location='cpu',weights_only=False)['rows']
    # Fixed consecutive collections; no selection by score or geometry result.
    if len(rows)<a.groups:
        following=path.with_name('001101.pt')
        if file_sha(following)!=json.loads(following.with_suffix('.json').read_text())['sha256']:raise ValueError('Changed archive')
        rows+=torch.load(following,map_location='cpu',weights_only=False)['rows']
    rows=rows[:a.groups]
    entries=[];prof=cProfile.Profile()
    for row in rows:
        road,observations,identity=original._geometry_context(row['token'],row['records'][0]['log'])
        old_seconds=new_seconds=0.
        for record in row['records']:
            coords=record['simulated_body_coordinates']
            t0=time.monotonic();old=original.body_margins(coords,road,observations);old_seconds+=time.monotonic()-t0
            t0=time.monotonic();new=body_margins(coords,road,observations);new_seconds+=time.monotonic()-t0
            exact(old,new,'geometry');exact(record['geometry'],new,'saved_geometry')
            expected=original.costs_from_record(record,cal['thresholds'])
            got=original.costs_from_record({**record,'geometry':new},cal['thresholds'])
            if not np.array_equal(expected,got):raise AssertionError('Cost changed')
        entries.append(dict(position=row['position'],candidates=len(row['records']),
                            scalar_seconds=old_seconds,vector_seconds=new_seconds,geometry_and_cost_exact=True))
    # Separate profile from wall comparison; profiling overhead is excluded.
    row=rows[0];road,observations,_=original._geometry_context(row['token'],row['records'][0]['log'])
    prof.enable()
    for r in row['records']:original.body_margins(r['simulated_body_coordinates'],road,observations)
    prof.disable()
    stats=pstats.Stats(prof)
    top=sorted([dict(function=f'{k[0]}:{k[1]}:{k[2]}',calls=v[1],self_seconds=v[2],cumulative_seconds=v[3])
                for k,v in stats.stats.items()],key=lambda x:x['cumulative_seconds'],reverse=True)[:20]
    old=sum(x['scalar_seconds'] for x in entries);new=sum(x['vector_seconds'] for x in entries)
    atomic_json(out,dict(status='PASS',host=host,groups=len(entries),candidates=sum(x['candidates'] for x in entries),
                scalar_seconds=old,vector_seconds=new,geometry_speedup=old/new,checks=entries,
                scalar_CPU_profile_top=top,source_sha256=file_sha(__file__),
                adapter_sha256=file_sha(Path(__file__).with_name('vector_geometry.py')),
                scope='Existing real simulated TRAIN body states; no new policy/scorer calls, exact FP64 geometry/cost equality'))
    print(json.dumps(dict(status='PASS',groups=len(entries),scalar_seconds=old,vector_seconds=new,speedup=old/new)),flush=True)


if __name__=='__main__':main()
