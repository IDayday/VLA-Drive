"""Complete scene/log pairing across equal inference-seed lists; no oracle.

Bootstrap resamples whole logs after averaging scores (not trajectories) over
the registered inference seeds. Each invocation compares one training seed.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .prepare_data import atomic_json
from starVLA.model.modules.vehicle_joint.initialization import file_sha256


def log_bootstrap(values, logs, samples=10000, seed=9028):
    values = np.asarray(values, dtype=np.float64)
    logs = np.asarray(logs)
    if not len(values) or len(values) != len(logs) or not np.isfinite(values).all(): raise ValueError("Invalid paired values")
    groups = [np.flatnonzero(logs == log) for log in sorted(set(logs))]
    sums = np.asarray([values[g].sum() for g in groups]); counts = np.asarray([len(g) for g in groups])
    if len(groups) < 2: return {"mean":float(values.mean()), "logs":len(groups), "ci95":None}
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        selection = rng.integers(len(groups), size=len(groups))
        estimates.append(sums[selection].sum()/counts[selection].sum())
    return {"mean":float(values.mean()), "logs":len(groups), "ci95":np.quantile(estimates,[.025,.975]).tolist(),
            "method":"scene-weighted paired log-cluster percentile bootstrap", "bootstrap_seed":seed,
            "training_seed_variability_included":False}


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--left',nargs='+',required=True);p.add_argument('--right',nargs='+',required=True)
    p.add_argument('--index',required=True);p.add_argument('--output',required=True)
    p.add_argument('--training-seed',type=int,required=True)
    p.add_argument('--sampling-seeds',nargs='+',type=int,default=[42,43,44,45,46])
    a=p.parse_args()
    if len(a.left)!=len(a.right) or len(a.left)!=len(a.sampling_seeds):raise ValueError('Unequal inference protocol')
    index=json.loads(Path(a.index).read_text());tokens=[r['token'] for r in index]
    if len(set(tokens))!=len(tokens):raise ValueError('Duplicate population token')
    banks=[]
    for paths in (a.left,a.right):
        group=[]
        for path in paths:
            with Path(path).open() as f: rows=list(csv.DictReader(f))
            data={r['token']:r for r in rows}
            if len(data)!=len(rows) or set(data)!=set(tokens):raise ValueError('Missing/duplicate scenes cannot be filtered from pairing')
            if any(data[r['token']]['log']!=r['log'] for r in index):raise ValueError('Log identity mismatch')
            group.append(data)
        banks.append(group)
    metrics=('score','no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','ego_progress','comfort')
    paired=[];failures=0
    for scene in index:
        token=scene['token'];row=dict(scene)
        for side,group in zip(('left','right'),banks):
            failed=sum(data[token]['status']!='ok' for data in group);failures+=failed
            row[side+'_failures']=failed
            row[side+'_zero_fraction']=np.mean([float(data[token].get('score',0))==0 for data in group])
            for metric in metrics:
                values=[float(data[token][metric]) if data[token]['status']=='ok' else 0. for data in group]
                if not np.isfinite(values).all():raise ValueError('Nonfinite official metric')
                row[side+'_'+metric]=float(np.mean(values))
        for metric in metrics:row['delta_'+metric]=row['right_'+metric]-row['left_'+metric]
        paired.append(row)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    with (out/'paired_scenes.csv').open('w') as f:
        writer=csv.DictWriter(f,list(paired[0]));writer.writeheader();writer.writerows(paired)
    summary={"training_seed":a.training_seed,"sampling_seeds":a.sampling_seeds,"scenes":len(paired),
             "failure_observations":failures,"valid":failures==0,"score_averaging":"equal inference seeds; no trajectory averaging or selection",
             "inputs":{str(p):file_sha256(p) for p in [*a.left,*a.right]},"index_sha256":file_sha256(a.index)}
    for metric in metrics:
        summary[metric]={"left":float(np.mean([r['left_'+metric] for r in paired])),"right":float(np.mean([r['right_'+metric] for r in paired])),
                         "right_minus_left":log_bootstrap([r['delta_'+metric] for r in paired],[r['log'] for r in paired])}
    atomic_json(out/'summary.json',summary)
    if failures:raise RuntimeError('Failed scenarios retained; aggregate is not a valid benchmark result')


if __name__=='__main__':main()
