"""Summarize saved local facts without another geometry/solver/model pass.

The native plan is query zero. Other query trajectories are auxiliary evidence,
not extra natural failures or independent scenes. Sampled 0.5-second support is
reported separately from the strict checker's dense whole-plan qualification.
"""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.decision_local.common import atomic_json, check_host, digest, identity, read, source_sha
from starVLA.model.modules.decision_local.local_evidence import UnknownReason
from starVLA.model.modules.decision_local.natural_error_contract import SCHEMA


def plan_facts(arrays, case, times, distance_change_m=1.):
    """Use the actual plan only, with exactly the oracle's common support."""
    a, p = case['anchor_index'], case['plan_index']
    if not np.array_equal(arrays['poses'][a,p,0], arrays['plan_in_physical'][a,p]):
        raise ValueError('Query zero is not the actual input plan')
    value = arrays['relations'][a,p,:,0]
    valid = arrays['valid'][a,p,:,0]
    common = arrays['common_valid'][a,p,0]
    positive = arrays['positive_known'][a,p,2,0]
    reasons = arrays['unknown_reason'][a,p,2,0]
    local_pair = valid[1] & valid[2]
    distance_difference = np.abs(value[2,:,1]-value[1,:,1])
    changed_overlap = value[2,:,2] != value[1,:,2]
    dynamic_common = (common[:,1] & (distance_difference >= distance_change_m)) | (common[:,2] & changed_overlap)
    flags = dict(
        population=True,
        checker_known_violation=bool(case['input_check']['known_violation']),
        checker_road_or_kinematic=bool(case['input_check']['road_violation'] or not case['input_check']['kinematic_ok']),
        checker_complete_4s=bool(case['input_check']['complete_annotations']),
        sampled_known_road_violation=bool(positive[:,0].any()),
        sampled_known_overlap=bool(positive[:,2].any()),
        sampled_local_fact_without_strict_target=bool((positive[:,0].any() or positive[:,2].any()) and not case['target_valid']),
        dynamic_information_common_oracle_support=bool(dynamic_common.any()),
        cv_fut_distance_change_local_support=bool((local_pair[:,1] & (distance_difference>=distance_change_m)).any()),
        cv_fut_overlap_change_local_support=bool((local_pair[:,2] & changed_overlap).any()),
        strict_target_eligible=bool(case['target_valid']))
    for reason in UnknownReason:
        flags['sampled_unknown_'+reason.name] = bool((reasons & int(reason)).any())
    for i,t in enumerate(times):
        for channel in range(3):
            flags[f'at_{t:g}s_channel{channel}_supported'] = bool(valid[2,i,channel])
            flags[f'at_{t:g}s_channel{channel}_oracle_common'] = bool(common[i,channel])
    for t in (.5,1.,2.,3.,4.):
        if t > times[-1]:
            continue
        selected = np.asarray(times) <= t+1e-9
        for channel in range(3):
            flags[f'sampled_prefix_{t:g}s_channel{channel}_supported'] = bool(valid[2,selected,channel].all())
    return flags


def execution_eligibility(record, case, review):
    """Apply the existing label-only review without promoting UNKNOWN/KEEP."""
    eligible = bool(case['target_valid'])
    if review is None or case['status_name'] != 'REPAIR' or not eligible:
        return eligible
    key = f"{record['token']}:{case['anchor_index']}:{case['plan_index']}"
    row = review['cases'][key]
    if (row['label_sha256'] != record['sha256'] or type(row['admitted']) is not bool or
            row['admitted'] != all(row['checks'].values())):
        raise ValueError('Execution review does not match this target')
    return row['admitted']


def verify_scored_q0(score_path, row, plan):
    if row['variant']!='q0' or row['status']!='ok':
        raise ValueError('Require successful original-input q0 scoring')
    path=score_path.parent/'proposals'/(row['token']+'.npz')
    if digest(path)!=row['proposal_sha256']:
        raise ValueError('Canonical input proposal changed')
    with np.load(path,allow_pickle=False) as data:
        if not np.array_equal(data['trajectory'],plan):
            raise ValueError('Canonical score belongs to another actual input plan')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('labels','current-root','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--execution-review',type=Path)
    parser.add_argument('--natural-scores',type=Path,
        help='Complete, canonical, seed42 input-q0 scenes.csv; never used for selection or targets')
    a=parser.parse_args();resources=check_host();start=time.monotonic();cpu=time.process_time()
    if a.output.exists():raise FileExistsError('Preserve the previous population report')
    meta=read(a.labels/'identity.json');complete=read(a.labels/'COMPLETE.json')
    if (meta['schema']!=SCHEMA or meta.get('local_fact_identity_layout')!='per_query_channel_int32_lookup_v1' or
            complete['identity']!=meta['identity'] or complete['scenes']!=len(meta['index']) or complete.get('errors')):
        raise ValueError('A complete, versioned local-evidence population is required')
    review=read(a.execution_review) if a.execution_review else None
    if review is not None:
        if ((a.execution_review.parent/'CORRECTNESS_HOLD.json').exists() or
                review['identity']!=identity({k:v for k,v in review.items() if k!='identity'}) or
                review.get('schema')!='natural_repair_execution_review_v1' or review['labels_identity']!=meta['identity'] or
                not review.get('all_label_repairs_checked') or not review.get('all_admitted_repairs_independently_checked')):
            raise ValueError('The independent execution review is incomplete or held')
    scores={}
    if a.natural_scores:
        score_meta=read(a.natural_scores.parent.parent/'identity.json')
        score_summary=read(a.natural_scores.parent/'summary.json')
        if (score_summary['identity']!=score_meta['identity'] or not score_summary['valid'] or score_summary['failed']!=0 or
                score_meta['navsim_python_tree_sha256']!='1468974c9af4405597c9dd3e8db1b68e7d7ab337327a7595fb57c2b3dcd4a84f' or
                score_meta['nuplan_python_tree_sha256']!='0e83aab8d5b92bf5792d815a1f25686088d1f4b8db70ea604331cd98f1606849' or
                score_meta['runtime_versions']!={'numpy':'1.26.4','scipy':'1.13.1','shapely':'2.0.7'}):
            raise ValueError('Require the complete pinned canonical scorer result')
        with a.natural_scores.open() as stream:score_rows=list(csv.DictReader(stream))
        scores={x['token']:x for x in score_rows}
        expected={x['token']:x['log'] for x in meta['index']}
        if len(scores)!=len(score_rows) or set(scores)!=set(expected):raise ValueError('Canonical population mismatch')
        if any(x.get('log',x.get('log_name'))!=expected[token] for token,x in scores.items()):
            raise ValueError('Canonical log identities differ')
        if any(not np.isfinite(float(x['score'])) or not 0<=float(x['score'])<=1 for x in scores.values()):
            raise ValueError('Failed or unnormalized canonical score; no failed scene may disappear')
    counts=Counter();scenes=defaultdict(set);logs=defaultdict(set);parents=defaultdict(set)
    event_tracks=defaultdict(set);event_times=defaultdict(set);event_scenes=defaultdict(set)
    cross=Counter();reviewed=set();case_rows=[];feature_meta=None
    def add(group,key,row):
        k=group,key;counts[k]+=1;scenes[k].add(row['token']);logs[k].add(row['log'])
        parents[k].add(row['log'].rsplit('_',2)[0])
    for row in meta['index']:
        record=read(a.labels/'records'/(row['token']+'.json'))
        path=a.labels/'scenes'/(row['token']+'.npz')
        if record['identity']!=meta['identity'] or record['log']!=row['log'] or digest(path)!=record['sha256']:
            raise ValueError('Changed local evidence or source scene')
        if feature_meta is None:
            feature_meta=read(Path(record['source_feature_chunk']).parent.parent/'identity.json')
            if feature_meta['identity']!=meta['feature_identity'] or feature_meta['seeds'][0]!=42:
                raise ValueError('Actual primary q0 sampling seed or feature population differs')
        with np.load(path,allow_pickle=False) as stream:arrays={k:stream[k] for k in stream.files}
        if scores:verify_scored_q0(a.natural_scores,scores[row['token']],arrays['plan_in_physical'][0,0])
        times=record['physical_times_s'];current=read(a.current_root/'current'/(row['token']+'.json'))
        timestamp=current['timestamp'];parent=row['log'].rsplit('_',2)[0]
        for case in record['cases']:
            group='NATURAL' if case['plan_index']==0 else 'SYNTHETIC'
            flags=plan_facts(arrays,case,times)
            admitted=execution_eligibility(record,case,review)
            if review is not None:
                flags['execution_review_target_eligible']=admitted
                if case['status_name']=='REPAIR' and case['target_valid']:
                    reviewed.add(f"{row['token']}:{case['anchor_index']}:{case['plan_index']}")
            for key,yes in flags.items():
                if yes:add(group,key,row)
            output=dict(**row,anchor_index=case['anchor_index'],plan_index=case['plan_index'],
                distribution=group,status=case['status_name'],**flags)
            if scores and group=='NATURAL' and case['anchor_index']==0:
                score=float(scores[row['token']]['score'])
                category='zero' if score==0 else 'low_0_to_0.5' if score<=.5 else 'normal_above_0.5'
                cross[(category,str(flags['checker_known_violation']),str(flags['sampled_known_road_violation'] or
                    flags['sampled_known_overlap']),str(flags['dynamic_information_common_oracle_support']),
                    str(admitted),case['status_name'])]+=1
                output.update(canonical_seed42_score=score,canonical_group=category)
            case_rows.append(output)
        for event in record['local_events']:
            if event['source']!='FUTURE':continue
            group='NATURAL' if event['plan_index']==0 else 'SYNTHETIC'
            # The remaining six queries are readout coverage, not q0 errors.
            scope='actual_plan' if event['query_index']<len(times) else 'auxiliary_query'
            key=(group,scope,event['constraint_type'])
            event_tracks[key].add((parent,event['actor']))
            event_times[key].add((parent,event['actor'],int(timestamp+round(event['time']*1e6))))
            event_scenes[key].add(row['token'])
    if review is not None and reviewed!=set(review['cases']):raise ValueError('Foreign or omitted reviewed REPAIR')
    a.output.mkdir()
    keys=sorted(set().union(*(row.keys() for row in case_rows)))
    with gzip.open(a.output/'plans.csv.gz','wt') as stream:
        writer=csv.DictWriter(stream,keys);writer.writeheader();writer.writerows(case_rows)
    result=dict(schema='stored_natural_local_population_summary_v1',source_sha=source_sha(ROOT),
        label_identity=meta['identity'],source_complete_sha256=digest(a.labels/'COMPLETE.json'),
        role=meta['role'],scope=meta['scope'],anchor_identity=meta['anchor_identity'],
        q0_sampling_seeds=feature_meta['seeds'],
        scenes=len(meta['index']),logs=len({x['log'] for x in meta['index']}),
        counts={group:{key:dict(plans=count,unique_scenes=len(scenes[(g,key)]),
            unique_logs=len(logs[(g,key)]),parent_recordings=len(parents[(g,key)]))
            for (g,key),count in counts.items() if g==group} for group in ('NATURAL','SYNTHETIC')},
        events=[dict(distribution=key[0],query_scope=key[1],type=key[2],unique_scenes=len(event_scenes[key]),
            labeled_actor_or_map_ids=len(event_tracks[key]),actor_absolute_time_pairs=len(event_times[key]))
            for key in sorted(event_tracks)],
        canonical_seed42_cross=[dict(zip(('canonical_group','checker_violation','sampled_local_violation',
            'dynamic_common_support','target_eligible','strict_status'),key),scenes=count) for key,count in sorted(cross.items())],
        execution_review_sha256=digest(a.execution_review) if review else None,
        canonical_scores_sha256=digest(a.natural_scores) if scores else None,
        dynamic_definition='Actual query0 CV/FUT distance change >=1m or a supported overlap change, within the exact common oracle support; fixed before new rankings',
        support_definition='Only stored 0.5s samples/prefixes; not the old 0.05s dense audit, not a continuous-time or full-plan certificate',
        event_definition='FUTURE endpoint changes; actor/time/seed/query observations are not independent scenes',
        no_geometry_or_solver_or_model_calls=True,no_scene_selection=True,resources=resources,
        CPU_process_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-start,
        row_file_sha256=digest(a.output/'plans.csv.gz'))
    atomic_json(a.output/'SUMMARY.json',result)
    print(json.dumps({k:result[k] for k in ('scenes','logs','role','scope','CPU_process_seconds')}))


if __name__=='__main__':main()
