"""Pinned G0 physical diagnostics, without re-decoding native feedback banks.

Event timestamps describe observed consequences, never causal responsibility.
This report is a TRAIN-cohort opportunity analysis, not a policy evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import numpy as np

from .common import artifact, atomic_bytes, digest, read_json, verify_artifact, write_json
from .verified_data import build_and_validate_datasets


SAFETY = {'NC':'no_at_fault_collisions','DAC':'drivable_area_compliance',
          'TTC':'time_to_collision_within_bound'}
EVENT_TIMES = {'collision':'first_at_fault_time_seconds','road':'first_road_time_seconds',
               'ttc':'first_ttc_state_time_seconds'}
LOCAL = {'longitudinal','lateral','joint'}
EPS = 1e-8


def distribution(values):
    x=np.asarray(values,dtype=np.float64)
    if not np.isfinite(x).all():raise ValueError('Nonfinite valid diagnostic quantity')
    return {'n':int(x.size),'mean':float(x.mean()) if x.size else None,
            'min':float(x.min()) if x.size else None,'max':float(x.max()) if x.size else None,
            'p25':float(np.quantile(x,.25)) if x.size else None,
            'median':float(np.quantile(x,.5)) if x.size else None,
            'p75':float(np.quantile(x,.75)) if x.size else None}


def label(labels,name):
    index=labels['names'].index(name)
    if not labels['valid'][index]:return None
    value=float(labels['values'][index])
    if not math.isfinite(value):raise ValueError('Nonfinite valid compact label')
    return value


def base_failure(row):
    return row['metrics']['score']<.8 or any(row['metrics'][key]<1-EPS for key in SAFETY.values())


def score_bin(score):
    return ('zero' if score<=EPS else 'positive_below_0.8' if score<.8
            else '0.8_to_below_0.95' if score<.95 else 'at_least_0.95')


def summarize(scenes,pairs,repairs):
    """Pure analysis; caller must first validate/pin its real population.

    Returned numerator/denominator counts remain useful on synthetic unit cases,
    but only the checked CLI admits a real-data report with artifact identities.
    """
    pairmap={(r['scene_id'],r['candidate_id']):r for r in pairs}
    repairmap={r['scene_id']:r for r in repairs}
    if len(pairmap)!=len(pairs) or len(repairmap)!=len(repairs):raise ValueError('Duplicate compact scene/candidate join')
    cohorts={};roles={};kind=defaultdict(Counter);kind_delta=defaultdict(list)
    affected=defaultdict(lambda:defaultdict(set))
    temporal=[];selected=[];tradeoffs=[];joint=[];invalid=Counter();failures=[]
    for scene in scenes:
        rows=scene['candidates'];base=rows[0];sid=scene['scene_id'];role=scene['role']
        invalid.update(r['status'] for r in rows if r['status']!='VALID_SCORED')
        if sid not in repairmap:raise ValueError('Missing repair compact row')
        repair=repairmap[sid]
        if repair['source_log']!=scene['source_log'] or repair['role']!=role:raise ValueError('Repair/source join mismatch')
        roleinfo=roles.setdefault(role,Counter(requested_scenes=0,valid_base=0,fixed_failures=0,verified_local_repairs=0))
        roleinfo['requested_scenes']+=1
        if base['status']!='VALID_SCORED':continue
        roleinfo['valid_base']+=1;failure=base_failure(base);score=base['metrics']['score']
        roleinfo['fixed_failures']+=failure
        roleinfo['verified_local_repairs']+=scene['repair']['status']=='VERIFIED_LOCAL_REPAIR'
        if failure:failures.append(scene)
        b=cohorts.setdefault(score_bin(score),Counter(scenes=0,fixed_failures=0,local_repair=0,support_only=0))
        b['scenes']+=1;b['fixed_failures']+=failure
        b['local_repair']+=scene['repair']['status']=='VERIFIED_LOCAL_REPAIR'
        b['support_only']+=scene['repair']['status']=='REQUIRES_LARGER_SUPPORT_SEARCH'
        bycoeff={tuple(r['coefficients']):r for r in rows if r['status']=='VALID_SCORED' and r.get('coefficients') is not None}
        for row in rows[1:]:
            if row['status']!='VALID_SCORED':continue
            metrics=row['metrics'];delta=metrics['score']-score;k=kind[row['kind']]
            k['valid_pairs']+=1;k['executable_pairs']+=bool(row.get('executable'))
            affected[row['kind']]['valid_pair_scenes'].add(sid)
            k['score_improved']+=delta>EPS;k['score_worsened']+=delta<-EPS
            k['eligible_repair']+=bool(row['eligible_repair']);kind_delta[row['kind']].append(delta)
            safety_loss=any(metrics[key]<base['metrics'][key]-EPS for key in SAFETY.values())
            comfort_loss=metrics['comfort']<base['metrics']['comfort']-EPS
            progress_change=row['progress_raw_meters']-base['progress_raw_meters']
            k['progress_increase_with_safety_loss']+=progress_change>EPS and safety_loss
            k['progress_increase_with_comfort_loss']+=progress_change>EPS and comfort_loss
            k['safety_gain_with_progress_loss']+=any(metrics[key]>base['metrics'][key]+EPS for key in SAFETY.values()) and progress_change<-EPS
            if score>=.8:
                k['original_high_pairs']+=1;k['high_to_zero']+=metrics['score']<=EPS
                if metrics['score']<=EPS:affected[row['kind']]['high_to_zero_scenes'].add(sid)
                # All physical-distance bins are reported below; this fixed
                # diagnostic threshold does not select targets or change G0.
                if row['distance_m']<=.5:
                    k['high_light_pairs']+=1;k['high_light_score_worsened']+=delta<-EPS
                    k['high_light_to_zero']+=metrics['score']<=EPS
                    if metrics['score']<=EPS:affected[row['kind']]['high_light_to_zero_scenes'].add(sid)
            for short,key in SAFETY.items():
                k['new_'+short]+=base['metrics'][key]>=1-EPS and metrics[key]<1-EPS
                if base['metrics'][key]>=1-EPS and metrics[key]<1-EPS:
                    affected[row['kind']]['new_'+short+'_scenes'].add(sid)
            pair=pairmap.get((sid,row['candidate_id']))
            if row['kind']=='policy_resample':continue
            if pair is None:raise ValueError('Valid controlled intervention lacks pinned effect pair')
            if pair['source_log']!=scene['source_log'] or pair['role']!=role:raise ValueError('Pair/source join mismatch')
            onset=min((v[0] for v in row['active_intervals_seconds']),default=0.)
            end=max((v[1] for v in row['active_intervals_seconds']),default=0.)
            ramp_end=max((v[1] for v in row['ramp_intervals_seconds']),default=0.)
            event={}
            for event_name,name in EVENT_TIMES.items():
                first=label(pair['base_labels'],name);after=label(pair['modified_labels'],name)
                event[event_name]={'base_first_seconds':first,'modified_first_seconds':after,
                    'base_event_minus_onset_seconds':None if first is None else first-onset,
                    'base_event_minus_ramp_end_seconds':None if first is None else first-ramp_end,
                    'modified_minus_base_first_seconds':None if first is None or after is None else after-first,
                    'time_delta_valid':first is not None and after is not None,
                    'transition':pair['event_transition'][event_name]}
            temporal.append({'scene_id':sid,'source_log':scene['source_log'],'role':role,
                'candidate_id':row['candidate_id'],'kind':row['kind'],'executable':row.get('executable'),
                'eligible_repair':row['eligible_repair'],'distance_m':row['distance_m'],'score_delta':delta,
                'ramp_intervals_seconds':row['ramp_intervals_seconds'],
                'actual_active_intervals_seconds':row['active_intervals_seconds'],
                'earliest_active_seconds':onset,'latest_active_seconds':end,'events':event})
            if row['kind']=='joint':
                coeff=np.asarray(row['coefficients'],dtype=float);along=coeff.copy();across=coeff.copy()
                along[4:]=0;across[:4]=0
                left,right=bycoeff.get(tuple(along)),bycoeff.get(tuple(across))
                if left is not None and right is not None:
                    joint.append({'scene_id':sid,'source_log':scene['source_log'],'role':role,
                        'joint_id':row['candidate_id'],'longitudinal_id':left['candidate_id'],'lateral_id':right['candidate_id'],
                        'score_nonlinear_interaction':metrics['score']-left['metrics']['score']-right['metrics']['score']+score,
                        'progress_nonlinear_interaction_m':row['progress_raw_meters']-left['progress_raw_meters']-right['progress_raw_meters']+base['progress_raw_meters'],
                        'interpretation':'same-scene observed factorial contrast, not a global gradient'})
        eligible=[r for r in rows[1:] if r['kind'] in LOCAL and r.get('eligible_repair')]
        if scene['repair']['status']=='VERIFIED_LOCAL_REPAIR':
            chosen=next(r for r in eligible if r['candidate_id']==scene['repair']['selected'])
            best=max(eligible,key=lambda r:(r['metrics']['score'],-r['distance_m'],r['trajectory_sha256']))
            if chosen['distance_m']!=min(r['distance_m'] for r in eligible):raise ValueError('Selected repair is not distance minimal')
            selected.append({'scene_id':sid,'source_log':scene['source_log'],'role':role,
                'selected_candidate':chosen['candidate_id'],'distance_m':chosen['distance_m'],
                'score_delta':chosen['metrics']['score']-score,'progress_delta_m':chosen['progress_raw_meters']-base['progress_raw_meters'],
                'maximum_score_eligible_candidate':best['candidate_id'],'maximum_score_eligible_distance_m':best['distance_m'],
                'maximum_score_eligible_score':best['metrics']['score']})
        for row in rows[1:]:
            if row['status']=='VALID_SCORED':tradeoffs.append({'scene_id':sid,'role':role,'kind':row['kind'],
                'distance_m':row['distance_m'],'score_delta':row['metrics']['score']-score,
                'base_score':score,'introduced_safety':any(row['metrics'][key]<base['metrics'][key]-EPS for key in SAFETY.values())})
    curve=[]
    for budget in (1,5,9,17,21,24,26):
        local=support=0
        for scene in failures:
            eligible=[r for r in scene['candidates'][:budget] if r.get('eligible_repair')]
            local+=any(r['kind'] in LOCAL for r in eligible)
            support+=any(r['kind'] in {'larger_support','policy_resample'} for r in eligible)
        curve.append({'proposal_slots_including_identity_geometric_rejections_and_duplicates':budget,
            'fixed_failure_scenes':len(failures),'local_repair_scenes':local,'support_success_scenes':support})
    distances=[]
    for low,high in ((0,.1),(.1,.25),(.25,.5),(.5,1),(1,2),(2,None)):
        selected_rows=[r for r in tradeoffs if r['distance_m']>=low and (high is None or r['distance_m']<high)]
        distances.append({'distance_interval_m':[low,high],'valid_pairs':len(selected_rows),
            'score_delta':distribution([r['score_delta'] for r in selected_rows]),
            'original_high_pairs':sum(r['base_score']>=.8 for r in selected_rows),
            'introduced_safety':sum(r['introduced_safety'] for r in selected_rows)})
    temporal_summary=[]
    for key in sorted({(r['role'],r['kind'],r['earliest_active_seconds']) for r in temporal}):
        selected_rows=[r for r in temporal if (r['role'],r['kind'],r['earliest_active_seconds'])==key]
        event_summary={}
        for event_name in EVENT_TIMES:
            values=[r['events'][event_name] for r in selected_rows]
            event_summary[event_name]={
                'observed_base_event_minus_onset_seconds':distribution([v['base_event_minus_onset_seconds'] for v in values if v['base_event_minus_onset_seconds'] is not None]),
                'observed_modified_minus_base_first_seconds':distribution([v['modified_minus_base_first_seconds'] for v in values if v['time_delta_valid']]),
                'repaired_pairs':sum(v['transition']['repaired'] for v in values),
                'introduced_pairs':sum(v['transition']['introduced'] for v in values)}
        temporal_summary.append({'role':key[0],'kind':key[1],'earliest_active_seconds':key[2],
            'valid_pairs':len(selected_rows),'unique_scenes':len({r['scene_id'] for r in selected_rows}),
            'score_delta':distribution([r['score_delta'] for r in selected_rows]),'events':event_summary})
    return {'requested_scenes':len(scenes),'source_logs':len({s['source_log'] for s in scenes}),
        'roles':roles,'base_score_bins':cohorts,'fixed_failure_scenes':len(failures),
        'invalid_candidate_status_counts':dict(invalid),
        'by_kind':{k:{**v,'unique_scene_counts':{n:len(ids) for n,ids in affected[k].items()},
                     'score_delta':distribution(kind_delta[k])} for k,v in kind.items()},
        'distance_bins':distances,'proposal_budget_curve':curve,'selected_minimal_repairs':selected,
        'minimal_repair_distance_m':distribution([r['distance_m'] for r in selected]),
        'temporal_intervention_event_observations':temporal,'temporal_summary_by_role_kind_onset':temporal_summary,
        'joint_factorial_interactions':joint,
        'joint_score_interaction':distribution([r['score_nonlinear_interaction'] for r in joint]),
        'continuous_road_margin':{'status':'UNAVAILABLE','reason':'No audited native continuous road clearance field'},
        'causal_boundary':'Native first-event time is an observation, not responsibility; support includes held tail after the quintic ramp.',
        'generalization_boundary':'TRAIN internal fit/validation roles only. Few failures or zero held-out repairs do not establish learned repair generalization.',
        'no_repair_meaning':'No verified repair within this registered budget; not proof of irreparability.',
        'thresholds':{'original_high_score':.8,'fixed_failure_score_below':.8,'full_safety_tolerance':EPS,
                      'light_intervention_mean_xy_distance_m':.5},'score_units':'0..1'}


def analyze(root,*,workers=1):
    root=Path(root)
    validation=build_and_validate_datasets(root,workers=workers)
    pair=read_json(verify_artifact(validation['pair_dataset']))
    repair=read_json(verify_artifact(validation['repair_dataset']))
    receipt=read_json(verify_artifact(validation['validation']['pair']['receipt']))
    inventory={r['path']:r for r in receipt['inventory']}
    def pinned(pin):
        path=str(Path(pin['path']).resolve())
        if path not in inventory or inventory[path]['sha256']!=pin['sha256']:
            raise ValueError('Metadata is absent from verified native inventory')
        return read_json(verify_artifact(pin))
    reg=pinned(pair['registration']);complete=pinned(pair['population_completion']);scenes=[]
    for item,pin in zip(reg['population'],complete['scenes']):
        sr=pinned(pin);scene=pinned(sr['scene_record'])
        if scene['context_entry']!=item['context']:raise ValueError('Pinned scene/context join mismatch')
        scenes.append(scene)
    result=summarize(scenes,pair['entries'],repair['entries'])
    result.update(schema='action_effect_g0_physical_analysis_v1',status='COMPLETE',scope='pilot_train_opportunity_analysis',
        registration=pair['registration'],pair_dataset=validation['pair_dataset'],repair_dataset=validation['repair_dataset'],
        validation_receipts={k:v['receipt'] for k,v in validation['validation'].items()},analyzer_source=artifact(__file__),
        actual_simulation_budget=pinned_counter(root,reg,inventory,scenes))
    result['identity']=digest(result)
    output=root/'physical_analysis'/result['identity']
    write_json(output/'G0_PHYSICAL_ANALYSIS.json',result)
    text=['# G0 physical intervention audit','',
        f"TRAIN pilot: {result['requested_scenes']} scenes, {result['source_logs']} source logs; {result['fixed_failure_scenes']} fixed-rule original failures.",
        f"Roles: `{dict(result['roles'])}`.",
        f"Verified minimal local repairs: {len(result['selected_minimal_repairs'])}; distance summary: `{result['minimal_repair_distance_m']}`.",'',
        'The JSON records complete score/role denominators, rejected proposals, high-score damage, safety/progress/comfort conflicts, matched joint contrasts and native-event timing.',
        'Proposal-slot budget includes identity, geometric rejections and duplicates; actual charged simulator admissions are separate.',
        result['causal_boundary'],result['generalization_boundary'],
        'Continuous road margin: UNAVAILABLE. No simulator or DEV/navtest read is used by this analyzer.']
    atomic_bytes(output/'G0_PHYSICAL_ANALYSIS.md',('\n\n'.join(text)+'\n').encode())
    return result


def pinned_counter(root,reg,inventory,scenes):
    # COUNTER may include reserved/orphan calls and therefore is explicitly a
    # mutable operational snapshot, never a supervision or success artifact.
    charged=set();historical=set()
    for scene in scenes:
        for row in scene['candidates']:
            if 'canonical_receipt' not in row:continue
            path=Path(row['canonical_receipt']['path']).parent/'ADMITTED.json'
            key=str(path.resolve());pin=inventory.get(key)
            if pin is None:raise ValueError('Admission absent from validated inventory')
            admission=read_json(verify_artifact(pin))
            (charged if admission.get('ledger_identity')==reg['identity'] else historical).add(key)
    return {'charged_completed_admissions':len(charged),'historical_completed_admissions_reused':len(historical),
            'mutable_counter_snapshot':read_json(root/'calls'/'COUNTER.json'),
            'note':'Only unique completed admissions referenced by this population; COUNTER includes reserved/orphan slots.'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--g0-root',required=True);p.add_argument('--workers',type=int,choices=range(1,5),default=1)
    args=p.parse_args(argv)
    try:
        result=analyze(args.g0_root,workers=args.workers)
        print(json.dumps({'status':result['status'],'identity':result['identity'],
                          'requested_scenes':result['requested_scenes'],'fixed_failures':result['fixed_failure_scenes']}));return 0
    except Exception as exc:
        print(json.dumps({'status':'FAIL','error_type':type(exc).__name__,'error':str(exc)}));return 1


if __name__=='__main__':raise SystemExit(main())
