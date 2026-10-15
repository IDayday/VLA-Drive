"""Paired canonical scores, explicit oracle/camera tables and failure transfers.

The registry names actual completed scorer roots. No model or checkpoint is
selected here. It reuses the existing scene-weighted log-cluster bootstrap.
"""
import argparse
import csv
import gzip
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.decision_local.common import atomic_json,digest,read,source_sha
from tools.local_interaction_mask_v2.compare_pdms import read as read_scores,compare,METRICS

COMPARISONS=[('E_FUTURE_GT','E_CV_GT'),('R_FUTURE_ABS','R_CURRENT_PRED'),
    ('R_FUTURE_DIFF','R_FUTURE_ABS'),('R_FUTURE_DIFF','E_NONE')]
SECONDARY=[('E_CURRENT_GT','E_NONE'),('E_CV_GT','E_CURRENT_GT'),
    ('E_FUTURE_GT','E_MASK_ONLY'),('E_FUTURE_GT','E_NONE'),('E_FUTURE_GT','R_FUTURE_DIFF')]


def natural_input_groups(root, original, score_path, foundation_identity):
    """Freeze descriptive groups from actual seed42 q0, before head comparisons.

    The stored population summary already verified the exact scored trajectory.
    Recheck that score file and row identity here; other seeds and auxiliary
    plans are not extra independent observations in the canonical comparison.
    """
    root=Path(root);meta=read(root/'SUMMARY.json');rows_path=root/'plans.csv.gz'
    if (meta.get('schema')!='stored_natural_local_population_summary_v1' or
        meta.get('anchor_identity')!=foundation_identity or
        meta.get('canonical_scores_sha256')!=digest(score_path) or
        meta.get('row_file_sha256')!=digest(rows_path)):
        raise ValueError('Natural groups are not bound to this foundation and scored q0')
    with gzip.open(rows_path,'rt') as stream:
        rows=[r for r in csv.DictReader(stream) if r['distribution']=='NATURAL'
              and int(r['anchor_index'])==0 and int(r['plan_index'])==0]
    selected={r['token']:r for r in rows}
    if len(selected)!=len(rows) or len(rows)!=meta['scenes'] or set(selected)!=set(original):
        raise ValueError('Natural groups require exactly one primary q0 for every scene')
    for token,row in selected.items():
        if (row['log']!=original[token]['log'] or original[token]['status']!='ok' or
            float(row['canonical_seed42_score'])!=float(original[token]['score'])):
            raise ValueError('Natural group score/log differs from the common input')

    def flag(row,key):
        if row[key] not in ('True','False'):
            raise ValueError('Invalid stored population flag: '+key)
        return row[key]=='True'

    definitions={
        'all_scenes':'Every original primary-seed scene, including UNKNOWN and zero scores',
        'dynamic_information_common_oracle_support':meta['dynamic_definition'],
        'checker_road_or_kinematic':'Original input has a known road or kinematic violation',
        'checker_known_violation':'Any original strict-checker known violation, independent of target eligibility',
        'sampled_known_road_violation':'Actual q0 has a reliable road violation at a stored physical time',
        'sampled_known_overlap':'Actual q0 has a reliable annotated-object overlap at a stored physical time',
        'original_high_ge_0p9':'Original common q0 canonical PDMS >=0.9; fixed before head rankings',
        'original_low_le_0p5':'Original common q0 canonical PDMS <=0.5; fixed before head rankings',
        'incomplete_full_4s_support':'Strict checker lacks whole-plan annotation support; local facts may still be valid',
        'strict_KEEP':'Original strict teacher KEEP; scoped geometry, not a real-world safety certificate',
        'strict_REPAIR':'Original strict teacher REPAIR, before independent execution review',
        'without_strict_target':'No full-plan teacher target; not proof that the original plan has no error',
    }
    groups={name:set() for name in definitions}
    copied=('dynamic_information_common_oracle_support','checker_road_or_kinematic',
            'checker_known_violation','sampled_known_road_violation','sampled_known_overlap')
    for token,row in selected.items():
        groups['all_scenes'].add(token)
        for key in copied:
            if flag(row,key):groups[key].add(token)
        for key,yes in (
            ('original_high_ge_0p9',float(original[token]['score'])>=.9),
            ('original_low_le_0p5',float(original[token]['score'])<=.5),
            ('incomplete_full_4s_support',not flag(row,'checker_complete_4s')),
            ('strict_KEEP',row['status']=='KEEP'),('strict_REPAIR',row['status']=='REPAIR'),
            ('without_strict_target',not flag(row,'strict_target_eligible'))):
            if yes:groups[key].add(token)
    evidence=dict(population_summary_sha256=digest(root/'SUMMARY.json'),
        row_file_sha256=meta['row_file_sha256'],label_identity=meta['label_identity'],
        anchor_identity=meta['anchor_identity'],scope=meta['scope'],definitions=definitions,
        selection='Original primary-seed input only; no final scores or head errors define groups',
        limitations=['Overlapping descriptive strata are not independent populations or extra hypothesis tests',
            'CV/FUT share privileged support; a relationship change does not prove a necessary action change',
            'Stored point support is not dense or full-trajectory certification',
            'Strict REPAIR is a teacher status; independent execution admission remains a separate label'])
    return groups,evidence


def grouped_comparisons(first, base, groups, bootstrap):
    """Keep scene weights and the existing log-cluster uncertainty operator."""
    if set(first)!=set(base):raise ValueError('Different paired group populations')
    result={}
    for name,tokens in groups.items():
        if not tokens<=set(first):raise ValueError('Foreign group scene')
        if not tokens:
            result[name]={'scenes':0,'logs':0,'status':'EMPTY','paired':None,'transitions':None}
            continue
        a={t:first[t] for t in tokens};b={t:base[t] for t in tokens}
        paired,_=compare(a,b,bootstrap=bootstrap)
        result[name]={'scenes':len(tokens),'logs':paired['logs'],'paired':paired,'transitions':transitions(a,b)}
    return result


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
    p.add_argument('--local-population',type=Path,
        help='Completed stored LocalEvidence population summary for the same natural seed42 q0')
    p.add_argument('--bootstrap',type=int,default=10000);a=p.parse_args();registry=read(a.registry)
    natural=registry.get('research_protocol')=='natural_error_local_v1'
    if a.local_population is not None and not natural:raise ValueError('Local groups require the natural-error protocol')
    if registry.get('research_protocol','decision_local_v1') not in ('decision_local_v1','natural_error_local_v1'):
        raise ValueError('Unknown research comparison protocol')
    primary=([('N_FUT','N_CV'),('R_ABS_FUTURE','R_CURRENT'),('R_DIFF_FUTURE','R_ABS_FUTURE'),
              ('R_DIFF_FUTURE','N_NONE')] if natural else COMPARISONS)
    secondary=([('N_CV','N_NONE'),('N_FUT','N_NONE'),('N_FUT','R_DIFF_FUTURE')] if natural else SECONDARY)
    expected={'debug':None,'dev':1696,'navtest':12146}[registry['population']]
    a.output.mkdir(parents=True,exist_ok=False);tables={'camera_only':[],'oracle_probe':[]};runs={};baseline=None;protocol=None
    groups=None;group_evidence=None
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
        if a.local_population is not None:
            if export.get('distribution')!='NATURAL' or export.get('sampling_seed')!=42:
                raise ValueError('Natural groups cannot be reused for perturbed or differently sampled plans')
            if groups is None:
                groups,group_evidence=natural_input_groups(a.local_population,original,path/'q0/scenes.csv',registry.get('foundation_identity'))
        mode=export['mode']
        if mode not in tables:raise ValueError('Unknown information regime')
        privileged=export['kind'] in (('N_CV','N_FUT') if natural else ('E_CURRENT_GT','E_CV_GT','E_FUTURE_GT','E_MASK_ONLY'))
        if privileged!=(mode=='oracle_probe'):raise ValueError('Privileged probe cannot be reported camera-only')
        paired,_=compare(final,original,bootstrap=a.bootstrap)
        row={'run':name,'mode':mode,'scenes':len(final),'PDMS':paired['PDMS_first_percent'],
            'PDMS_q0':paired['PDMS_baseline_percent'],'delta_percentage_points':paired['delta_percentage_points'],
            'valid':paired['valid'],**transitions(final,original),
            'metrics':{k:v['first'] for k,v in paired['metrics'].items()},'source':str(path)}
        tables[mode].append(row);runs[name]=final
        detail={'paired':paired,'transitions':transitions(final,original)}
        if groups is not None:detail['fixed_input_groups']=grouped_comparisons(final,original,groups,a.bootstrap)
        atomic_json(a.output/(name+'_versus_q0.json'),detail)
    comparisons={};pending=[]
    for first,base in primary+secondary:
        if first not in runs or base not in runs:pending.append(first+' minus '+base);continue
        result,rows=compare(runs[first],runs[base],bootstrap=a.bootstrap)
        comparisons[first+'_minus_'+base]={'paired':result,'transitions':transitions(runs[first],runs[base]),
            'primary_preregistered_comparison':(first,base) in primary}
        if groups is not None:
            comparisons[first+'_minus_'+base]['fixed_input_groups']=grouped_comparisons(runs[first],runs[base],groups,a.bootstrap)
        with (a.output/(first+'_minus_'+base+'.csv')).open('w') as stream:
            w=csv.DictWriter(stream,list(rows[0]));w.writeheader();w.writerows(rows)
    atomic_json(a.output/'summary.json',{'registry_sha256':digest(a.registry),'source_sha':source_sha(ROOT),'tool_sha256':digest(__file__),
        'population':registry['population'],'scope':registry['scope'],'protocol':protocol,'tables':tables,'comparisons':comparisons,
        'research_protocol':registry.get('research_protocol','decision_local_v1'),
        'foundation_identity':registry.get('foundation_identity'),
        'natural_input_group_evidence':group_evidence,
        'not_yet_available':pending,'base_training_seeds':1,
        'interpretation':'Frozen mature S0 incremental probes; privileged rows are not deployable; no causal upper bound or from-scratch gain claim. Internal q0 is a shared frozen trained baseline here.',
        'group_thresholds_fixed_before_formal_training':{'baseline_high_PDMS_fraction':.9,'baseline_low_PDMS_fraction':.5},
        'uncertainty':'log-cluster resampling with original scene weights; sampling seeds and head seeds do not replicate the S0 foundation training seed'})


if __name__=='__main__':main()
