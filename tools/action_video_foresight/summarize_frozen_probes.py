"""Paired fixed-target probe results; never rank different teacher raw MSE."""
import argparse
import json
from pathlib import Path

import numpy as np

from tools.ddpolicy_vehicle.prepare_data import atomic_json


def paired_interval(left,right,column,seed=73,draws=10000):
    if [r['token'] for r in left]!=[r['token'] for r in right] or any(
            a['log']!=b['log'] or a['failure'] or b['failure'] for a,b in zip(left,right)):
        raise ValueError('Fixed query population/failures mismatch')
    groups={};invalid=0
    for a,b in zip(left,right):
        x,y=a[column],b[column]
        if (x is None)!=(y is None):raise ValueError('Matched target eligibility changed')
        if x is None:invalid+=1;continue
        if not np.isfinite([x,y]).all():raise ValueError('Invalid successful metric')
        groups.setdefault(a['log'],[]).append(x-y)
    if not groups:raise ValueError('No valid matched target population')
    values=list(groups.values());sums=np.array([sum(v) for v in values]);counts=np.array([len(v) for v in values])
    rng=np.random.default_rng(seed);sample=rng.integers(len(values),size=(draws,len(values)))
    difference=sums[sample].sum(1)/counts[sample].sum(1)
    return {'left_minus_right_normalized_mse':float(sums.sum()/counts.sum()),
        'log_cluster_bootstrap_95':np.quantile(difference,[.025,.975]).tolist(),
        'requested_scenes':len(left),'valid_matched_scenes':int(counts.sum()),
        'missing_target_scenes_retained':invalid,'logs_with_valid_targets':len(groups),
        'bootstrap_draws':draws,'bootstrap_seed':seed,
        'interpretation':'One frozen-checkpoint/train-seed probe. Log sampling uncertainty is not cross-training-seed stability or PDMS.'}


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--campaign-root',required=True)
    parser.add_argument('--target',choices=('video','dino','both'),required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();root=Path(args.campaign_root);out=Path(args.output)
    if out.exists():raise ValueError('Preserve prior probe summaries')
    report={'scope':'Frozen C1@100k future-head probes, not formal ego planning; no raw-MSE ranking across teacher types',
        'main_model_optimizer_updates':0,'families':{}}
    for prefix in (('V','F') if args.target=='both' else ('V',) if args.target=='video' else ('F',)):
        names=(prefix+'0',prefix+'1',prefix+'_ACTION_ONLY');summary={};rows={};identities={}
        for name in names:
            directory=root/f'frozen_C1_100k_future_{name}_v1'
            summary[name]=json.loads((directory/'SUMMARY.json').read_text())
            identities[name]=json.loads((directory/'identity.json').read_text())
            if summary[name]['failed'] or summary[name]['scenes']!=summary[name]['requested']:
                raise ValueError('Incomplete probe evaluation')
            rows[name]=[json.loads(line) for line in (directory/'dev_queries.jsonl').read_text().splitlines()]
            if len(rows[name])!=summary[name]['requested']:raise ValueError('Summary/query denominator mismatch')
        anchor=identities[names[0]]
        common=('frozen_checkpoint','train_queries','dev_queries','train_targets','dev_targets','updates','batch','seed')
        if any(any(identities[name][k]!=anchor[k] for k in common) for name in names):
            raise ValueError('Probe training protocol differs')
        pairs={left+'-'+right:paired_interval(rows[left],rows[right],'normal')
            for left,right in [(names[1],names[0]),(names[1],names[2]),(names[0],names[2])]}
        report['families'][prefix]={'identities':identities,'results':summary,'paired':pairs,
            'teacher_and_target_structure':json.loads((root/(('targets_video_clip' if prefix=='V' else 'targets_dino_sequence')+'_dev_v1')/'identity.json').read_text())}
    atomic_json(out,report)


if __name__=='__main__':main()
