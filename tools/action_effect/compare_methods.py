"""Compare complete matched native evaluations; no model/threshold selection.

All result paths are explicit NAME=PATH arguments. Invalid predictions remain in
the requested denominator and prevent a full-population score/gain claim.
"""
import argparse
import csv
import io
from pathlib import Path

import numpy as np

from .common import artifact, atomic_bytes, digest, read_json, verify_artifact, verify_identity, write_json

METRICS=('score','no_at_fault_collisions','drivable_area_compliance',
         'time_to_collision_within_bound','ego_progress','comfort','driving_direction_compliance')


def load(path):
    path=Path(path);result=read_json(path)
    completion=read_json(path.parent/'COMPLETE.json')
    if result.get('status')!='COMPLETE' or completion.get('result')!=artifact(path):
        raise ValueError('Complete checksummed native evaluation required: '+str(path))
    reg=verify_identity(read_json(path.parent/'EVALUATION_REGISTRATION.json'))
    if completion['registration_identity']!=reg['identity'] or result['registration_identity']!=reg['identity']:
        raise ValueError('Evaluation/completion identity changed')
    pred=verify_identity(read_json(verify_artifact(reg['predictions'])))
    population=verify_identity(read_json(verify_artifact(pred['registration'])))
    if pred['requested_scenes']!=len(result['entries']) or len(pred['entries'])!=len(result['entries']):
        raise ValueError('Incomplete population denominator')
    rows={r['scene_id']:r for r in result['entries']};prediction={r['scene_id']:r for r in pred['entries']}
    if len(rows)!=len(result['entries']) or set(rows)!=set(prediction):raise ValueError('Duplicate/missing ID join')
    pins={p['path']:p for p in completion['scenes']}
    for sid,row in rows.items():
        verify_identity(row)
        pin=artifact(path.parent/'scenes'/sid/'RESULT.json')
        if pins.get(pin['path'])!=pin or row['prediction_identity']!=prediction[sid]['identity']:
            raise ValueError('Native result/prediction/scene receipt changed')
    return result,rows,population


def paired_summary(base,other,*,repetitions=2000,seed=42):
    if set(base)!=set(other):raise ValueError('Compared scenes differ')
    ids=sorted(base)
    if any(base[s]['source_log']!=other[s]['source_log'] for s in ids):raise ValueError('Compared source logs differ')
    valid=[s for s in ids if all(r[s].get('status')=='VALID_SCORED' and r[s].get('metrics') is not None
                               for r in (base,other))]
    n=len(valid);out={'requested_scenes':len(ids),'common_valid_scenes':n,'invalid_or_unresolved':len(ids)-n,
        'full_population_score_claim':n==len(ids),'score_units':'0..1'}
    if not n:return out
    b=np.array([base[s]['metrics']['score'] for s in valid]);a=np.array([other[s]['metrics']['score'] for s in valid]);d=a-b
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Nonfinite official score')
    logs=np.array([base[s]['source_log'] for s in valid]);groups=sorted(set(logs));rng=np.random.default_rng(seed)
    group_sums=np.array([d[logs==g].sum() for g in groups]);group_n=np.array([(logs==g).sum() for g in groups])
    boots=[]
    for _ in range(repetitions):
        selected=rng.integers(len(groups),size=len(groups));boots.append(group_sums[selected].sum()/group_n[selected].sum())
    out.update(base_mean=float(b.mean()),method_mean=float(a.mean()),mean_delta=float(d.mean()),
        delta_ci95=np.quantile(boots,[.025,.975]).tolist(),source_logs=len(groups),bootstrap_repetitions=repetitions,
        bootstrap_seed=seed,original_zero=int((b<=1e-8).sum()),zero_repaired=int(((b<=1e-8)&(a>1e-8)).sum()),
        original_high=int((b>=.8).sum()),high_to_zero=int(((b>=.8)&(a<=1e-8)).sum()),
        nonzero_to_zero=int(((b>1e-8)&(a<=1e-8)).sum()),method_zero=int((a<=1e-8).sum()),
        positive_gain=float(np.maximum(d,0).mean()),negative_cost=float(np.maximum(-d,0).mean()),
        changed_score_scenes=int((abs(d)>1e-8).sum()),
        mean_trajectory_change_m=float(np.mean([other[s]['distance_m'] for s in valid])))
    for short,key in (('NC','no_at_fault_collisions'),('DAC','drivable_area_compliance'),('TTC','time_to_collision_within_bound')):
        before=np.array([base[s]['metrics'][key] for s in valid]);after=np.array([other[s]['metrics'][key] for s in valid])
        out['new_'+short+'_failures']=int(((before>=1-1e-8)&(after<1-1e-8)).sum())
        out['repaired_'+short+'_failures']=int(((before<1-1e-8)&(after>=1-1e-8)).sum())
    out['components']={key:{'base_mean':float(np.mean([base[s]['metrics'][key] for s in valid])),
                            'method_mean':float(np.mean([other[s]['metrics'][key] for s in valid]))} for key in METRICS}
    out['DDC_weight']=0
    return out


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',required=True)
    p.add_argument('--method',action='append',default=[],help='NAME=/actual/score/RESULT.json')
    p.add_argument('--output',required=True);p.add_argument('--bootstrap-repetitions',type=int,default=2000)
    p.add_argument('--seed',type=int,default=42);a=p.parse_args(argv)
    if not 1<=a.bootstrap_repetitions<=10000 or not a.method:raise ValueError('Bounded bootstrap and explicit methods required')
    base,base_rows,base_reg=load(a.baseline);results={};pins={};csv_rows=[]
    for named in a.method:
        name,path=named.split('=',1)
        if not name or name in results:raise ValueError('Unique named methods required')
        result,rows,reg=load(path)
        for key in ('base_checkpoint','score_protocol_hash','numeric_runtime_hash','context_manifest','role','scene_ids'):
            if reg[key]!=base_reg[key]:raise ValueError('Matched evaluation contract differs: '+key)
        results[name]=paired_summary(base_rows,rows,repetitions=a.bootstrap_repetitions,seed=a.seed)
        pins[name]=artifact(path)
        csv_rows.append({'method':name,**{k:v for k,v in results[name].items() if not isinstance(v,(list,dict))}})
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    report={'schema':'action_effect_matched_comparison_v1','status':'COMPLETE','baseline':artifact(a.baseline),
        'methods':pins,'results':results,'selection_performed':False,'navtest_accessed':False,
        'statistical_unit':'paired full source-log resampling; scene-weighted mean within each draw',
        'scientific_conclusion':'must assess matched controls, independent seeds and severe regressions together'}
    report['identity']=digest(report);write_json(output/'COMPARISON.json',report)
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    atomic_bytes(output/'FINAL_SYSTEM_RESULTS.csv',stream.getvalue().encode())
    print(report['identity'])


if __name__=='__main__':main()
