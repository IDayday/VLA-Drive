"""Aggregate coherent K-samples and paired current-relation condition ablations."""
import argparse
import json
from pathlib import Path
import numpy as np
from tools.joint_local_scene_v3.analyze_campaign import read_queries,bootstrap,csv_out
from tools.joint_local_scene_v3.budget import atomic_json


def analyze(root):
    summary=json.loads((root/'summary.json').read_text())
    if not summary['complete'] or summary['failed']:raise ValueError('Incomplete/failed endpoint diagnostic')
    rows=read_queries(root/'condition_queries.csv');result={'scope':summary['scope'],'expected_queries':summary['expected_queries'],'applicable':summary['applicable'],'not_applicable':summary['not_applicable'],'failed':summary['failed'],'conditions':{},'sampling':{}}
    for actor in ('ego','neighbor'):
        chosen=[r for r in rows if (r['slot']==0)==(actor=='ego')]
        result['conditions'][actor]={'all_queries':len(chosen),'not_applicable':sum(r['status']=='NOT_APPLICABLE' for r in chosen),'pairs':{}}
        good=[r for r in chosen if r['status']=='ok']
        result['conditions'][actor]['exact_removed_point_count_matches']=sum(r['strong_xy_points']==r['weak_xy_points'] for r in good)
        for left,right in [('correct','remove_strong'),('correct','remove_weak'),('remove_weak','remove_strong')]:
            paired=[{'log':r['log'],'all_sum':r[left+'_error_sum_m'],'mask_sum':r[right+'_error_sum_m'],'points':r['valid_xy_points']} for r in good]
            metric=bootstrap(paired)
            for old,new in [('J_ALL_ADE_m',left+'_ADE_m'),('J_MASK_ADE_m',right+'_ADE_m'),('MASK_minus_ALL_m',right+'_minus_'+left+'_m')]:
                if old in metric:metric[new]=metric.pop(old)
            result['conditions'][actor]['pairs'][right+'_vs_'+left]=metric
    samples=read_queries(root/'k_queries.csv');ks=sorted({r['k'] for r in samples})
    for actor in ('ego','neighbor'):
        per_seed=[]
        for k in ks:
            group=[r for r in samples if r['k']==k and (r['slot']==0)==(actor=='ego')];points=sum(r['valid_xy_points'] for r in group)
            per_seed.append({'k':k,'seed':group[0]['seed'],'ADE_m':sum(r['xy_error_sum_m'] for r in group)/points,'queries':len(group),'valid_points':points})
        result['sampling'][actor]={'single_sample_primary_k':0,'per_sampling_seed':per_seed,'mean_ADE_over_sampling_seeds_m':float(np.mean([r['ADE_m'] for r in per_seed])),'std_ADE_over_sampling_seeds_m':float(np.std([r['ADE_m'] for r in per_seed])),'not_training_seed_uncertainty':True}
    scenes=read_queries(root/'k_scenes.csv');consistency=[]
    for k in ks:
        group=[r for r in scenes if r['k']==k];n=sum(r['joint_relative_valid_points'] for r in group)
        consistency.append({'k':k,'relative_vector_error_m':sum(r['joint_relative_error_sum_m'] for r in group)/n if n else None,'valid_pair_time_points':n,'ego_execution_max_difference_m':max(r['execution_ego_xy_difference_m'] for r in group)})
    oracle=[json.loads(x) for x in (root/'joint_diversity_oracle.jsonl').read_text().splitlines()]
    result['joint_consistency']=consistency;result['whole_joint_oracle_and_diversity']={'scene_count':len(oracle),'mean_scene_joint_oracle_ADE_m':float(np.mean([r['joint_oracle_ADE_m'] for r in oracle])),'mean_scene_pair_sample_distance_m':float(np.mean([r['mean_pair_sample_distance_m'] for r in oracle])),'oracle_non_deployment':True,'no_per_actor_sample_mixing':True}
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args();atomic_json(a.output,analyze(Path(a.input)))


if __name__=='__main__':main()
