"""Paired canonical scores, explicit oracle/camera tables and failure transfers.

The registry names actual completed scorer roots. No model or checkpoint is
selected here. It reuses the existing scene-weighted log-cluster bootstrap.
"""
import argparse
import csv
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.decision_local.common import atomic_json,digest,read,source_sha
from tools.local_interaction_mask_v2.compare_pdms import read as read_scores,compare,METRICS

COMPARISONS=[('E_FUTURE_GT','E_CV_GT'),('R_FUTURE_ABS','R_CURRENT_PRED'),
    ('R_FUTURE_DIFF','R_FUTURE_ABS'),('R_FUTURE_DIFF','E_NONE')]


def transitions(first,base):
    tokens=sorted(base);a=np.array([float(first[t].get('score',0)) if first[t]['status']=='ok' else 0 for t in tokens]);b=np.array([float(base[t].get('score',0)) if base[t]['status']=='ok' else 0 for t in tokens])
    out={'zero_to_nonzero':int(((b==0)&(a>0)).sum()),'nonzero_to_zero':int(((b>0)&(a==0)).sum()),
         'both_zero':int(((b==0)&(a==0)).sum()),'both_nonzero':int(((b>0)&(a>0)).sum())}
    both=(a>0)&(b>0);out['both_nonzero_mean_change']=float((a[both]-b[both]).mean()) if both.any() else None
    for name,m in [('original_high_ge_0p9',b>=.9),('original_low_le_0p5',b<=.5)]:
        out[name]={'scenes':int(m.sum()),'mean_delta':float((a[m]-b[m]).mean()) if m.any() else None,
            'new_zeros':int((m&(b>0)&(a==0)).sum())}
    progress=np.array([float(first[t].get('ego_progress') or 0)-float(base[t].get('ego_progress') or 0) for t in tokens])
    unsafe=np.zeros(len(tokens),bool)
    for metric in ('no_at_fault_collisions','time_to_collision_within_bound','drivable_area_compliance'):
        delta=np.array([float(first[t].get(metric) or 0)-float(base[t].get(metric) or 0) for t in tokens]);unsafe|=delta<0
        out[metric+'_worsened_scenes']=int((delta<0).sum())
    out['progress_up_with_any_NC_TTC_DAC_down']=int(((progress>1e-12)&unsafe).sum())
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registry',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bootstrap',type=int,default=10000);a=p.parse_args();registry=read(a.registry)
    expected={'debug':None,'dev':1696,'navtest':12146}[registry['population']]
    a.output.mkdir(parents=True,exist_ok=False);tables={'camera_only':[],'oracle_probe':[]};runs={};baseline=None;protocol=None
    for name,item in registry['runs'].items():
        path=Path(item['scores']);meta=read(path/'identity.json');done=read(path/'COMPLETE.json')
        if done['identity']!=meta['identity']:raise ValueError('Incomplete scorer result')
        common={k:meta[k] for k in ('population','index_sha256','navsim_python_tree_sha256','nuplan_python_tree_sha256','runtime_versions','protocol','failure_policy')}
        if protocol is not None and common!=protocol:raise ValueError('Different evaluation protocols/populations')
        protocol=common
        original,final=[read_scores(path/variant/'scenes.csv') for variant in ('q0','q_final')]
        if expected is not None and len(final)!=expected:raise ValueError('Incomplete registered population')
        if set(original)!=set(final):raise ValueError('Different initial/final scene populations')
        if baseline is not None:
            if original!=baseline:raise ValueError('Compared heads do not share identical S0 q0 scoring')
        else:baseline=original
        export=read(Path(item['predictions'])/'identity.json')
        if export['identity']!=meta['prediction_identity']:raise ValueError('Scorer not bound to declared export')
        mode=export['mode']
        if mode not in tables:raise ValueError('Unknown information regime')
        if (name.startswith(('E_CURRENT_GT','E_CV_GT','E_FUTURE_GT','E_MASK_ONLY')))!=(mode=='oracle_probe'):raise ValueError('Privileged probe cannot be reported camera-only')
        paired,_=compare(final,original,bootstrap=a.bootstrap)
        row={'run':name,'mode':mode,'scenes':len(final),'PDMS':paired['PDMS_first_percent'],
            'PDMS_q0':paired['PDMS_baseline_percent'],'delta_percentage_points':paired['delta_percentage_points'],
            'valid':paired['valid'],**transitions(final,original),
            'metrics':{k:v['first'] for k,v in paired['metrics'].items()},'source':str(path)}
        tables[mode].append(row);runs[name]=final
        atomic_json(a.output/(name+'_versus_q0.json'),{'paired':paired,'transitions':transitions(final,original)})
    comparisons={};pending=[]
    for first,base in COMPARISONS:
        if first not in runs or base not in runs:pending.append(first+' minus '+base);continue
        result,rows=compare(runs[first],runs[base],bootstrap=a.bootstrap)
        comparisons[first+'_minus_'+base]={'paired':result,'transitions':transitions(runs[first],runs[base])}
        with (a.output/(first+'_minus_'+base+'.csv')).open('w') as stream:
            w=csv.DictWriter(stream,list(rows[0]));w.writeheader();w.writerows(rows)
    atomic_json(a.output/'summary.json',{'registry_sha256':digest(a.registry),'source_sha':source_sha(ROOT),'tool_sha256':digest(__file__),
        'population':registry['population'],'scope':registry['scope'],'protocol':protocol,'tables':tables,'comparisons':comparisons,
        'not_yet_available':pending,'base_training_seeds':1,
        'interpretation':'Frozen mature S0 incremental probes; privileged rows are not deployable; no causal upper bound or from-scratch gain claim. Internal q0 is a shared frozen trained baseline here.',
        'group_thresholds_fixed_before_formal_training':{'baseline_high_PDMS_fraction':.9,'baseline_low_PDMS_fraction':.5},
        'uncertainty':'log-cluster resampling with original scene weights; sampling seeds and head seeds do not replicate the S0 foundation training seed'})


if __name__=='__main__':main()
