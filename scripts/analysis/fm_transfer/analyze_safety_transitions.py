"""Describe saved single-ODE safety transitions under an explicit comparison.

No inference, simulation, fitting or checkpoint selection. Outcome-conditioned
subsets are descriptive diagnostics, not a causal test or an unbiased estimate
of the full population. Official first TTC infraction is a physical simulation
timestamp, not predicted time-to-impact and not a Flow generation step.
The default remains RL versus its exact own SFT; comparing two SFT milestones
requires an opt-in and the same supervised run/configuration.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.analysis.fm_transfer.analyze_evaluation import match_transfer
from scripts.analysis.simwam_recipe.analyze_evaluation import read, METRICS
from starVLA.rl.fm_transfer.contracts import require_dispatch
from starVLA.rl.simwam_recipe.contracts import file_sha, fingerprint
from starVLA.rl.flow_grpo.transactions import atomic_json

TTC = METRICS.index('time_to_collision_within_bound')
EP = METRICS.index('ego_progress')


def comparison_contract(identity, reference, kind):
    if kind == 'rl_vs_own_sft':
        if identity['stage'] != 'RL' or reference['stage'] != 'SFT':
            raise ValueError('Requires RL vs exact own SFT')
        if identity['new_SFT'] != reference['new_SFT']:
            raise ValueError('RL must retain its exact supervised parent')
        return None
    if kind != 'sft_milestones':
        raise ValueError('Unknown safety-analysis comparison')
    if identity['stage'] != 'SFT' or reference['stage'] != 'SFT':
        raise ValueError('SFT milestone comparison cannot include RL')
    current, parent = identity['new_SFT'], reference['new_SFT']
    if (not current.get('identity_sha256') or
            current['identity_sha256'] != parent.get('identity_sha256') or
            not current.get('config') or current['config'] != parent.get('config')):
        raise ValueError('SFT milestones must share the same run identity and configuration')
    before, after = parent['updates'], current['updates']
    if (type(before) is not int or type(after) is not int or
            not 0 <= before < after <= current['config']['max_updates'] or
            before not in current['config']['milestones'] or after not in current['config']['milestones']):
        raise ValueError('Requires ordered, registered SFT milestones')
    return dict(kind=kind,reference_updates=before,evaluation_updates=after,
        is_RL_increment=False,reference_is_final_RL_parent=False,
        source_reward_readout='Counterfactual 10:5:2 readout only; SFT optimizes weighted FM')


def distribution(values):
    values = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if not np.isfinite(values).all(): raise ValueError('Nonfinite diagnostic, never silently drop it')
    if not len(values): return dict(defined=0, mean=None, median=None, minimum=None, maximum=None)
    return dict(defined=len(values), mean=float(values.mean()), median=float(np.median(values)),
                minimum=float(values.min()), maximum=float(values.max()),
                positive=int((values>0).sum()), negative=int((values<0).sum()))


def source_reward(values):
    # Reuse the locked dual-readout algebra on the already scored submetrics.
    nc, dac, ttc = values[:,1], values[:,2], values[:,3]
    ep, comfort = values[:,4], values[:,5]
    canonical = nc*dac*(5*ep+5*ttc+2*comfort)/12
    if not np.allclose(canonical, values[:,0], rtol=0, atol=5e-15):
        raise ValueError('Stored official PDMS does not match the locked readout')
    return nc*dac*(10*ep+5*ttc+2*comfort)/17


def subsets(base, current):
    old_safe = (base[:,1:4] == 1).all(1); new_safe = (current[:,1:4] == 1).all(1)
    old_ttc = base[:,TTC] < 1; new_ttc = current[:,TTC] < 1
    return dict(all_scenes=np.ones(len(base), dtype=bool),
        new_TTC=(~old_ttc & new_ttc), recovered_TTC=(old_ttc & ~new_ttc),
        retained_TTC_failure=(old_ttc & new_ttc),
        new_DAC=((base[:,2] == 1) & (current[:,2] < 1)),
        SFT_safe_new_joint_violation=(old_safe & ~new_safe),
        both_safe_practical_improvement=(old_safe & new_safe & (current[:,0]-base[:,0] >= .005)))


def summarize_subset(mask, base, current, old_motion, new_motion, logs):
    indices = np.flatnonzero(mask)
    delta = current[:,0]-base[:,0]; reward_delta = source_reward(current)-source_reward(base)
    movements = {}
    for key in old_motion[0]:
        pairs = [(old_motion[i][key],new_motion[i][key]) for i in indices
                 if old_motion[i][key] is not None and new_motion[i][key] is not None]
        movements[key] = dict(reference=distribution([a for a,b in pairs]),
            current=distribution([b for a,b in pairs]),delta=distribution([b-a for a,b in pairs]))
    clusters = Counter(logs[i] for i in indices)
    return dict(scenes=len(indices), logs=len(clusters), log_scene_counts_descending=sorted(clusters.values(),reverse=True),
        metrics={k:dict(reference=distribution(base[mask,j]),current=distribution(current[mask,j]),
                       delta=distribution(current[mask,j]-base[mask,j])) for j,k in enumerate(METRICS)},
        source_reward_delta=distribution(reward_delta[mask]),
        source_improved_metric_degraded=int(((reward_delta>1e-12)&(delta< -1e-12)&mask).sum()),
        source_improved_metric_degraded_both_ge005=int(((reward_delta>=.005)&(delta<=-.005)&mask).sum()),
        canonical_positive_gain_sum_raw=float(delta[mask & (delta>0)].sum()),
        canonical_negative_degradation_sum_raw=float(-delta[mask & (delta<0)].sum()),
        motion=movements)


def physical_event_pair(reference, current, timestamp):
    old = np.asarray(reference['simulated_states'],dtype=np.float64)
    new = np.asarray(current['simulated_states'],dtype=np.float64)
    if old.shape != (41,11) or new.shape != old.shape or not np.isfinite([old,new]).all():
        raise ValueError('Unexpected physical simulation grid/state contract')
    if not np.array_equal(old[0],new[0]):
        raise ValueError('Paired trajectories do not share the same initial simulated state')
    index = round(timestamp/.1)
    if not 0 <= index <= 40 or abs(index*.1-timestamp)>1e-7:
        raise ValueError('Official event must use the physical 0.1s grid')
    heading = old[0,2];forward = np.array([np.cos(heading),np.sin(heading)])
    lateral = np.array([-np.sin(heading),np.cos(heading)])
    a = old[index,:2]-old[0,:2]; b = new[index,:2]-new[0,:2]
    # Segment displacement gives the mean speed over the preceding physical 0.1s,
    # not instantaneous state velocity or a reconstructed official TTC value.
    speed = lambda x: float(np.linalg.norm(x[index,:2]-x[index-1,:2])/.1) if index else None
    va,vb = speed(old),speed(new)
    return dict(timestamp_s=float(timestamp),initial_states_exactly_equal=True,
        reference_forward_m=float(a@forward),current_forward_m=float(b@forward),
        forward_difference_m=float((b-a)@forward),
        reference_abs_lateral_m=float(abs(a@lateral)),current_abs_lateral_m=float(abs(b@lateral)),
        abs_lateral_difference_m=float(abs(b@lateral)-abs(a@lateral)),
        reference_preceding_segment_speed_mps=va,current_preceding_segment_speed_mps=vb,
        preceding_segment_speed_difference_mps=vb-va if va is not None else None)


def stored_record(root, token, mode):
    path = Path(root)/'scenes'/token[:2]/(token+'.pt')
    seal = json.loads(path.with_suffix('.json').read_text())
    if file_sha(path) != seal['payload_sha256']: raise ValueError('Changed saved trajectory payload')
    payload = torch.load(path,map_location='cpu',weights_only=False)
    record = payload['records'][mode]
    if payload['token'] != token or record['status'] != 'OK' or record['metrics'] != seal['metrics'][mode]:
        raise ValueError('Unpaired or invalid saved feedback')
    return record


def events(evaluation, reference, mode, tokens, masks):
    result = {}
    for name in ('new_TTC','recovered_TTC','retained_TTC_failure'):
        rows = []; hashes = []; collision_count = 0
        for index in np.flatnonzero(masks[name]):
            token = tokens[index]
            old = stored_record(reference,token,mode); new = stored_record(evaluation,token,mode)
            if old['cache'] != new['cache'] or old['protocol_sha256'] != new['protocol_sha256']:
                raise ValueError('Same scene does not share the same official scoring inputs')
            failing = old if name == 'recovered_TTC' else new
            timestamp = failing['official_first_ttc_infraction_seconds']
            if timestamp is None or not np.isfinite(timestamp): raise ValueError('Missing official failure timestamp')
            rows.append(physical_event_pair(old,new,timestamp))
            collision_count += int(new['metrics']['no_at_fault_collisions'] < 1)
            hashes.append(fingerprint(dict(token=token,old_cache=old['cache'],new_cache=new['cache'],
                old_metric=old['metrics'],new_metric=new['metrics'],event=rows[-1])))
        result[name] = dict(scenes=len(rows),paired_record_digest=fingerprint(hashes),
            event_from='reference' if name=='recovered_TTC' else 'current',
            event_is_first_official_infraction_physical_timestamp=True,
            current_NC_not_fully_pass=collision_count,
            first_time_counts_s=dict(sorted(Counter(f"{r['timestamp_s']:.1f}" for r in rows).items(),key=lambda x:float(x[0]))),
            initial_states_exactly_equal_for_all=all(r['initial_states_exactly_equal'] for r in rows),
            values={k:distribution([row[k] for row in rows]) for k in (
                'timestamp_s','reference_forward_m','current_forward_m','forward_difference_m',
                'reference_abs_lateral_m','current_abs_lateral_m','abs_lateral_difference_m',
                'reference_preceding_segment_speed_mps','current_preceding_segment_speed_mps',
                'preceding_segment_speed_difference_mps')})
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--evaluation',required=True); parser.add_argument('--reference',required=True)
    parser.add_argument('--comparison-kind',choices=('rl_vs_own_sft','sft_milestones'),
        default='rl_vs_own_sft')
    parser.add_argument('--output',required=True); args=parser.parse_args(); require_dispatch()
    output=Path(args.output)
    if output.exists(): raise FileExistsError('Preserve previous derived evidence')
    start=time.monotonic()
    identity,tokens,logs,current,new_motion,_=read(args.evaluation)
    parent,old_tokens,old_logs,baseline,old_motion,_=read(args.reference)
    comparison=comparison_contract(identity,parent,args.comparison_kind)
    compatibility=match_transfer(identity,parent)
    if tokens!=old_tokens or logs!=old_logs: raise ValueError('Unpaired complete scene/log populations')
    result=dict(schema='fm_transfer_saved_safety_transition_analysis_v1',status='COMPLETE_SAVED_POPULATION',
        analysis_sha256=file_sha(__file__),evaluation=args.evaluation,reference=args.reference,
        evaluation_manifest_sha256=file_sha(Path(args.evaluation)/'manifest.json'),
        reference_manifest_sha256=file_sha(Path(args.reference)/'manifest.json'),
        compatibility=compatibility,scenes=len(tokens),logs=len(set(logs)),
        outcome_conditioned_subsets_not_causal=True,score_units='raw0to1',
        new_inference=0,new_scoring=0,optimizer_updates=0,modes={})
    if comparison is not None:
        result.update(schema='fm_transfer_saved_SFT_milestone_safety_analysis_v1',comparison=comparison)
    selections={}
    for mode in identity['modes']:
        masks=subsets(baseline[mode],current[mode]);selections[mode]=masks
        result['modes'][mode]=dict(subsets={name:summarize_subset(mask,baseline[mode],current[mode],
            old_motion[mode],new_motion[mode],logs) for name,mask in masks.items()},
            events=events(args.evaluation,args.reference,mode,tokens,masks))
    a,b=selections['E_NATIVE']['new_TTC'],selections['E_SHIFT5']['new_TTC']
    current_a,current_b=current['E_NATIVE'][:,TTC]<1,current['E_SHIFT5'][:,TTC]<1
    result['cross_grid']=dict(new_TTC_in_both=int((a&b).sum()),new_TTC_native_only=int((a&~b).sum()),
        new_TTC_shift5_only=int((~a&b).sum()),current_TTC_fail_both=int((current_a&current_b).sum()),
        current_TTC_fail_native_only=int((current_a&~current_b).sum()),
        current_TTC_fail_shift5_only=int((~current_a&current_b).sum()),
        caveat='Each new-failure set uses its own matching-grid SFT reference; no best-of-two selection')
    if comparison is not None:
        result['cross_grid']['caveat']='Each failure set uses its matching-grid earlier SFT milestone; no RL increment or best-of-two selection'
    result['analysis_wall_seconds']=time.monotonic()-start
    output.parent.mkdir(parents=True,exist_ok=True);atomic_json(output,result)
    print(json.dumps(dict(status=result['status'],scenes=len(tokens),cross_grid=result['cross_grid'],output=str(output))))


if __name__=='__main__': main()
