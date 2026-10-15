"""Matched field quality with paired log-cluster resampling of sufficient sums."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.decision_local.common import atomic_json,digest,source_sha


def matched_sensitivity_protocol(first,baseline):
    legacy='any_supported_sample_v1'
    protocols={r.get('sensitivity_protocol',legacy) for rows in (first,baseline) for r in rows.values()}
    if len(protocols)!=1:raise ValueError('Relation sensitivity protocols differ; rescore both saved predictions first')
    return next(iter(protocols))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('first','baseline','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--bootstrap',type=int,default=10000);a=p.parse_args()
    def load(path):
        if path.name.endswith('.csv.gz'):
            with gzip.open(path,'rt') as stream:
                rows=[{k:v if k in ('token','log','condition','sensitivity_protocol') else float(v) for k,v in r.items()} for r in csv.DictReader(stream)]
        else:rows=[json.loads(s) for s in path.read_text().splitlines()]
        rows=[r for r in rows if r.get('condition','normal')=='normal']
        by={r['token']:r for r in rows}
        if len(rows)!=len(by):raise ValueError('Duplicate scenes cannot replicate sample size')
        return by
    x,y=load(a.first),load(a.baseline)
    if set(x)!=set(y) or any(x[t]['log']!=y[t]['log'] for t in x):raise ValueError('Unmatched fixed scenes')
    sensitivity_protocol=matched_sensitivity_protocol(x,y)
    groups=['near_sensitive','near_insensitive','far_sensitive','far_insensitive']
    if sensitivity_protocol=='local_known_sensitive_complete_insensitive_unknown_v2':
        groups+=['near_unknown','far_unknown']
    logs=sorted({r['log'] for r in x.values()});index={l:i for i,l in enumerate(logs)}
    weights=np.random.default_rng(20260926).multinomial(len(logs),[1/len(logs)]*len(logs),size=a.bootstrap)
    def sums(rows,keys):
        total=np.zeros(len(logs))
        for r in rows.values():total[index[r['log']]]+=sum(r[k] for k in keys)
        return total
    metrics={}
    for channel in ['road_margin_m','current_clearance_m','future_clearance_m']:
        metrics[channel+'_absolute_MAE']=([channel+'_abs_error_sum'],[channel+'_valid_points'])
        metrics[channel+'_pair_MAE']=([channel+'_pair_abs_error_sum'],[channel+'_pair_valid_times'])
        metrics[channel+'_direction_accuracy']=([channel+'_direction_correct_pairs'],[channel+'_direction_valid_pairs'])
        for group in groups:
            metrics[group+'_'+channel+'_pair_MAE']=([group+'_'+channel+'_pair_abs_error_sum'],[group+'_'+channel+'_valid_times'])
    for channel in ['current_overlap','future_overlap']:
        metrics[channel+'_recall']=([channel+'_TP'],[channel+'_TP',channel+'_FN'])
        metrics[channel+'_precision']=([channel+'_TP'],[channel+'_TP',channel+'_FP'])
    result={}
    for name,(num,den) in metrics.items():
        ax,bx=sums(x,num),sums(x,den);ay,by=sums(y,num),sums(y,den)
        dx,dy=weights@bx,weights@by;valid=(dx>0)&(dy>0)
        draws=(weights[valid]@ax)/dx[valid]-(weights[valid]@ay)/dy[valid]
        result[name]={'first':float(ax.sum()/bx.sum()) if bx.sum() else None,
            'baseline':float(ay.sum()/by.sum()) if by.sum() else None,
            'delta':float(ax.sum()/bx.sum()-ay.sum()/by.sum()) if bx.sum() and by.sum() else None,
            'paired_log_cluster_95_interval':np.quantile(draws,[.025,.975]).tolist() if valid.any() and len(logs)>1 else None,
            'valid_bootstrap_draws':int(valid.sum()),'first_denominator':float(bx.sum()),'baseline_denominator':float(by.sum())}
    atomic_json(a.output,{'source_sha':source_sha(ROOT),'tool_sha256':digest(__file__),
        'first_sha256':digest(a.first),'baseline_sha256':digest(a.baseline),'scenes':len(x),'logs':len(logs),'metrics':result,
        'sensitivity_protocol':sensitivity_protocol,
        'aggregation':'Original scene sufficient sums and valid query denominators retained within sampled logs; no scene×seed pseudo-replication',
        'limits':'Fixed-query analysis of the declared checkpoint banks (or explicitly registered opposite-fold pooling); log resampling is not foundation-training seed replication. Query type and distance can be confounded; no causal planning attribution or multiplicity correction.'})


if __name__=='__main__':main()
