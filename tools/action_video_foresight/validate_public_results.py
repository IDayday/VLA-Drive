"""Validate the public C/S tables and bootstrap without private data/checkpoints."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def read_csv(path):
    with path.open() as stream:return list(csv.DictReader(stream))


def validate(root):
    root=Path(root);identity=json.loads((root/'IDENTITY.json').read_text())
    history=read_csv(root/'ALL_RESULTS.csv');endpoints=read_csv(root/'ENDPOINT_100K.csv')
    if (len(history)!=identity['complete_evaluations'] or len(endpoints)!=8
            or len({(r['split'],r['arm'],r['updates']) for r in history})!=len(history)):
        raise ValueError('Missing/duplicate reported evaluation')
    if any(int(r['failed']) or r['precision']!='FP32' or int(r['FM_steps'])!=10
           or int(r['candidates'])!=1 for r in history):
        raise ValueError('Invalid or mixed evaluation protocol')
    arms={r['arm']:r for r in endpoints}
    for split,count,logs in [('dev',1696,16),('navtest',12146,136)]:
        rows=read_csv(root/(split.upper()+'_100K_SCENE_PDMS.csv'))
        if (len(rows)!=count or len({r['scene_id'] for r in rows})!=count
                or len({r['log_id'] for r in rows})!=logs):
            raise ValueError('Public scene/log denominator changed')
        for arm,r in arms.items():
            values=np.asarray([float(row[arm]) for row in rows])
            expected=float(r['PDMS'] if split=='navtest' else r['dev_PDMS'])
            zeros=int(r['zero_scenes'] if split=='navtest' else r['dev_zero_scenes'])
            if (not np.isfinite(values).all() or values.min()<0 or values.max()>100
                    or abs(values.mean()-expected)>1e-10 or int((values==0).sum())!=zeros):
                raise ValueError('Public scene matrix disagrees with endpoint')
    pairs=json.loads((root/'PAIRED_COMPARISONS.json').read_text());checked=0
    for key,pair in pairs.items():
        if not key.endswith('_100000'):continue
        rows=read_csv(root/(key+'_logs.csv'))
        counts=np.asarray([int(r['scenes']) for r in rows])
        deltas=np.asarray([float(r['score_delta']) for r in rows])
        # CSV row order is original canonical log order, preserved after hashing.
        if len(rows)!=pair['logs'] or counts.sum()!=pair['scenes']:
            raise ValueError('Paired log denominator changed')
        weights=np.random.default_rng(pair['bootstrap_seed']).multinomial(
            len(rows),[1/len(rows)]*len(rows),size=pair['bootstrap_draws'])
        means=(weights@(counts*deltas))/(weights@counts)
        metric=pair['metrics']['score']
        if (abs(np.average(deltas,weights=counts)-metric['delta'])>1e-12
                or not np.allclose(np.quantile(means,[.025,.975]),
                                   metric['log_cluster_95_interval'],rtol=0,atol=1e-12)):
            raise ValueError('Public log data cannot reproduce the reported interval')
        checked+=1
    training=json.loads((root/'training/COMPLETED_RUNS.json').read_text())
    bins=read_csv(root/'training/TRAINING_CURVES_1000_UPDATE_BINS.csv')
    for arm,run in training['runs'].items():
        rows=[r for r in bins if r['arm']==arm]
        if (run['status']['status']!='COMPLETE' or run['status']['completed']!=100000
                or len(rows)!=100 or sum(int(r['averaged_updates']) for r in rows)!=100000
                or int(rows[-1]['exposure'])!=run['status']['exposure']):
            raise ValueError('Training curves/progress mismatch')
        if any(not math.isfinite(float(v)) for r in rows for k,v in r.items() if k.endswith('_mean') and v):
            raise ValueError('Nonfinite public training curve')
    return dict(valid=True,evaluations=len(history),complete_100k_models=len(arms),
                reproduced_100k_log_intervals=checked,training_curve_bins=len(bins),
                private_data_required=False,new_optimizer_updates=0,new_model_inference=0)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--report-dir',required=True)
    a=p.parse_args();print(json.dumps(validate(a.report_dir)))


if __name__=='__main__':main()
