"""Score immutable trajectory exports with the repository's official NAVSIM v1 PDM implementation."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import asdict
import hashlib
import json
import lzma
import multiprocessing
from pathlib import Path
import pickle
import sys
import numpy as np


def score_log(task):
    devkit,items=task
    sys.path.insert(0,devkit)
    from navsim.common.dataclasses import Trajectory
    from pdm_cache_adapter import compatible_pdm_score as pdm_score
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    sampling=TrajectorySampling(num_poses=40,interval_length=.1)
    simulator=PDMSimulator(sampling);scorer=PDMScorer(sampling)
    rows=[]
    for item in items:
        token,log,cache_path,proposal_path=item
        row={'token':token,'log':log,'status':'ok'}
        try:
            with lzma.open(cache_path,'rb') as f:cache=pickle.load(f)
            with np.load(proposal_path,allow_pickle=False) as f:poses=f['trajectory'].astype(np.float64)
            if poses.shape!=(8,3) or not np.isfinite(poses).all():raise ValueError('Invalid trajectory shape/value')
            result=pdm_score(metric_cache=cache,model_trajectory=Trajectory(poses,TrajectorySampling(num_poses=8,interval_length=.5)),future_sampling=sampling,simulator=simulator,scorer=scorer)
            row.update({k:float(v) for k,v in asdict(result).items()})
            row['proposal_sha256']=hashlib.sha256(Path(proposal_path).read_bytes()).hexdigest()
        except Exception as error:row.update(status='failed',error=repr(error),score=0.)
        rows.append(row)
    return rows


def main():
    p=argparse.ArgumentParser()
    for name in ['devkit','cache-metadata','manifest','predictions','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--workers',type=int,default=16);a=p.parse_args()
    requested=json.loads(Path(a.manifest).read_text());index={}
    with open(a.cache_metadata) as f:
        for row in csv.DictReader(f):
            path=Path(row['file_name']);index[path.parent.name]=(path.parts[-4],str(path))
    grouped={}
    for token in requested:
        log,path=index.get(token,('MISSING',''))
        grouped.setdefault(log,[]).append((token,log,path,str(Path(a.predictions)/(token+'.npz'))))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);rows=[]
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for result in pool.map(score_log,[(a.devkit,x) for x in grouped.values()]):
            rows.extend(result)
            with (out/'progress.jsonl').open('a') as f:
                for row in result:f.write(json.dumps(row)+'\n')
    assert len(rows)==len(requested) and len({r['token'] for r in rows})==len(requested)
    keys=sorted(set(k for row in rows for k in row))
    with (out/'scenes.csv').open('w') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
    failed=sum(r['status']!='ok' for r in rows)
    summary={'requested':len(requested),'failed':failed,'PDMS':sum(r.get('score',0.) for r in rows)/len(rows),
             'zero_fraction':sum(r.get('score',0.)==0 for r in rows)/len(rows),'protocol':'NAVSIM v1; validated compact training-cache adapter with fixed official PDM progress; one candidate','failure_policy':'failed rows retained; zero assigned, complete result invalid if any failure','arguments':vars(a),
             'cache_adapter_sha256':hashlib.sha256(Path(__file__).with_name('pdm_cache_adapter.py').read_bytes()).hexdigest(),'evaluator_sha256':hashlib.sha256((Path(a.devkit)/'navsim/evaluate/pdm_score.py').read_bytes()).hexdigest()}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
    if failed:raise RuntimeError('Incomplete/invalid PDMS result; inspect retained failed rows')

if __name__=='__main__':main()
