"""Existing evidence only: discovery -> fixed teacher gate, by reference stratum.

This performs no proposal, actor forward, scoring, optimizer or threshold fit.
The five-reference discovery mean and candidate0 teacher reference are audited
separately so stochastic reference differences are not attributed to the .9 gate.
"""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np
import torch

from starVLA.rl.qwen_flow_exploration.feedback.process import SAFETY
from starVLA.rl.qwen_flow_exploration.protocol import read_protocol
from starVLA.rl.qwen_sourceflow.contracts import file_sha, fingerprint


def sealed(path):
    value = json.loads(path.read_text()); body = dict(value)
    if body.pop('sha256') != fingerprint(body):
        raise ValueError('Changed prior evidence: '+str(path))
    return value


def safe(row):
    return all(row['native_subscores'][name] == 1. for name in SAFETY)


def gain_type(candidate, base):
    if not safe(candidate):
        return 'still_has_official_safety_failure'
    if not safe(base):
        return 'official_NC_DAC_TTC_recovery'
    a, b = candidate['native_subscores'], base['native_subscores']
    others = set(a)-{'score', 'pdm_score', 'ego_progress'}
    if all(abs(a[key]-b[key]) <= 1e-10 for key in others):
        return 'fully_safe_EP_only'
    return 'fully_safe_other_native_quality_change'


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--reference-strata', required=True)
    parser.add_argument('--search-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    protocol = read_protocol(args.protocol)
    strata = sealed(Path(args.reference_strata))
    if strata['protocol_sha256'] != protocol['sha256']:
        raise ValueError('Other frozen reference population')
    tokens = protocol['splits']['TRAIN']
    if set(strata['by_split']['TRAIN']) != set(tokens):
        raise ValueError('Missing fixed TRAIN reference strata')
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    results = dict(schema='feedback_teacher_admission_mechanism_v1',
        protocol_sha256=protocol['sha256'],base_model_sha256=protocol['base_model_sha256'],
        reference_strata_sha256=file_sha(args.reference_strata),split='TRAIN',scenes=len(tokens),
        frozen_thresholds=dict(minimum_quality=.9,minimum_gain_raw=.005,maximum_targets=4,
            minimum_target_pairwise_ADE_m=.2),
        discovery_reference='mean of five independent common-base reference roots',
        teacher_gain_reference='same shared candidate0 for every search arm',
        no_new_search_scoring_actor_or_optimizer_calls=True,methods={})
    private_rows=[]
    for method in ('F2_PDMS','F3_FEEDBACK'):
        groups=defaultdict(Counter); types=Counter(); weighted_types=Counter(); manifest={}
        expected_rewards,expected_references,expected_deltas,expected_base0_deltas=[],[],[],[]
        stable_zero_repairs=[]
        for token in tokens:
            path=Path(args.search_root)/method/'TRAIN/scenes'/f'{token}.json'
            record=sealed(path)
            if (record['status']!='COMPLETE' or record['scene_token']!=token or record['split']!='TRAIN' or
                record['common_protocol_sha256']!=protocol['sha256'] or record['result']['method']!=method or
                record['base_model_sha256']!=protocol['base_model_sha256']):
                raise ValueError('Changed fixed teacher source')
            manifest[token]=record['sha256']
            candidates=record['result']['candidates'];base=candidates[0];ref=strata['by_split']['TRAIN'][token]
            assert len(candidates)==32
            feasible=[row for row in candidates if row.get('raw_reward') is not None and safe(row)]
            d5=[row for row in feasible if row['raw_reward']>=ref['reference_mean']+.005]
            d0=[row for row in feasible if row['raw_reward']>=base['raw_reward']+.005]
            quality=[row for row in d0 if row['raw_reward']>=.9]
            teachers=record['teacher_candidates']
            if bool(quality)!=bool(teachers) or any(row['raw_reward']<max(.9,base['raw_reward']+.005) or not safe(row) for row in teachers):
                raise ValueError('Teacher admission differs from registered first-target rule')
            for row in teachers:
                types[gain_type(row,base)]+=1
            expected_reward=None
            if teachers:
                # Match choose_teacher's FP32 softmax; multinomial treats its
                # finite FP32 output as relative mass, normalized here in FP64.
                weights=torch.softmax(torch.tensor([row['raw_reward'] for row in teachers],dtype=torch.float32)/.1,0).double().numpy()
                weights/=weights.sum()
                for row,weight in zip(teachers,weights):weighted_types[gain_type(row,base)]+=float(weight)
                expected_reward=float(sum(row['raw_reward']*weight for row,weight in zip(teachers,weights)))
                expected_rewards.append(expected_reward);expected_references.append(ref['reference_mean'])
                expected_deltas.append(expected_reward-ref['reference_mean'])
                expected_base0_deltas.append(expected_reward-base['raw_reward'])
            teacher_types={gain_type(row,base) for row in teachers}
            scene=dict(method=method,scene_token=token,log_id=record['log_id'],stratum=ref['stratum'],
                reference_mean=ref['reference_mean'],base0_reward=base['raw_reward'],base0_safe=safe(base),
                discovered_safe_gain_vs_refmean=bool(d5),discovered_safe_gain_vs_base0=bool(d0),
                quality_gate_blocks_candidate0_gain=bool(d0) and not quality,
                d5_no_teacher_quality_gate=bool(d5) and not teachers and bool(d0) and not quality,
                d5_no_teacher_candidate0_reference_gate=bool(d5) and not d0,
                teacher_admitted=bool(teachers),teacher_count=len(teachers),
                teacher_has_useful_gain_vs_refmean=any(row['raw_reward']>=ref['reference_mean']+.005 for row in teachers),
                expected_teacher_reward=expected_reward,
                expected_teacher_delta_vs_refmean=expected_reward-ref['reference_mean'] if teachers else None,
                expected_teacher_useful_vs_refmean=bool(teachers) and expected_reward>=ref['reference_mean']+.005,
                expected_teacher_negative_vs_refmean=bool(teachers) and expected_reward<ref['reference_mean'],
                best_safe_reward=max((row['raw_reward'] for row in feasible),default=None),
                best_teacher_reward=max((row['raw_reward'] for row in teachers),default=None),
                teacher_is_safety_recovery='official_NC_DAC_TTC_recovery' in teacher_types,
                teacher_EP_only='fully_safe_EP_only' in teacher_types,
                teacher_other_quality='fully_safe_other_native_quality_change' in teacher_types)
            for name in ('ALL',ref['stratum']):
                group=groups[name];group['scenes']+=1;group['feasible_discovered_scenes']+=bool(feasible)
                for key in ('discovered_safe_gain_vs_refmean','discovered_safe_gain_vs_base0',
                    'quality_gate_blocks_candidate0_gain','d5_no_teacher_quality_gate',
                    'd5_no_teacher_candidate0_reference_gate','teacher_admitted',
                    'teacher_has_useful_gain_vs_refmean','expected_teacher_useful_vs_refmean',
                    'expected_teacher_negative_vs_refmean','teacher_is_safety_recovery','teacher_EP_only','teacher_other_quality'):
                    group[key]+=bool(scene[key])
                group['teacher_candidates']+=len(teachers)
                group['d5_discovery_without_teacher']+=bool(d5) and not teachers
                group['admitted_without_useful_gain_vs_refmean']+=bool(teachers) and not scene['teacher_has_useful_gain_vs_refmean']
            if ref['stratum']=='stable_zero' and feasible:
                stable_zero_repairs.append(dict(scene_token=token,best_safe_reward=scene['best_safe_reward'],
                    base0_reward=base['raw_reward'],reference_mean=ref['reference_mean'],
                    teacher_admitted=bool(teachers),quality_gate_blocks_candidate0_gain=scene['quality_gate_blocks_candidate0_gain']))
            private_rows.append(scene)
        results['methods'][method]=dict(by_fixed_stratum={k:dict(v) for k,v in groups.items()},
            teacher_candidate_gain_types=dict(types),
            expected_teacher_scene_mass_by_gain_type=dict(weighted_types),
            expected_teacher_scene_mass_denominator='512 population scenes; admitted scene mass1, no-teacher mass0',
            repaired_stable_zero_scenes=len(stable_zero_repairs),
            repaired_stable_zero_admitted_scenes=sum(row['teacher_admitted'] for row in stable_zero_repairs),
            repaired_stable_zero_best_safe_scores=[row['best_safe_reward'] for row in stable_zero_repairs],
            softmax_target_quality=dict(temperature_raw_score=.1,
                probabilities='same FP32 softmax as choose_teacher, normalized as multinomial mass in FP64 for expectation',
                covered_scenes=len(expected_deltas),covered_expected_reward_raw=float(np.mean(expected_rewards)),
                covered_five_ref_reward_raw=float(np.mean(expected_references)),
                covered_expected_delta_vs_five_ref_raw=float(np.mean(expected_deltas)),
                covered_expected_delta_vs_five_ref_PDMS_points=100*float(np.mean(expected_deltas)),
                covered_expected_delta_vs_candidate0_raw=float(np.mean(expected_base0_deltas)),
                expected_delta_vs_candidate0_min_raw=float(min(expected_base0_deltas)),
                covered_expected_gain_ge005_scenes=sum(delta>=.005 for delta in expected_deltas),
                covered_expected_gain_below005_scenes=sum(delta<.005 for delta in expected_deltas),
                covered_expected_gain_negative_scenes=sum(delta<0 for delta in expected_deltas),
                all_scenes=len(tokens),missing_teacher_scenes=len(tokens)-len(expected_deltas),
                missing_teacher_semantics='zero DELTA for accounting, not reward0 and not actual GT loss targets',
                all_scene_expected_delta_vs_five_ref_raw=float(np.sum(expected_deltas)/len(tokens)),
                all_scene_expected_delta_vs_five_ref_PDMS_points=100*float(np.sum(expected_deltas)/len(tokens)),
                all_scene_reference_mean_raw=float(np.mean([row['reference_mean'] for row in strata['by_split']['TRAIN'].values()])),
                all_scene_reference_filled_target_reward_raw=float((sum(expected_deltas)+sum(row['reference_mean'] for row in strata['by_split']['TRAIN'].values()))/len(tokens)),
                all_scene_positive_contribution_PDMS_points=100*float(np.maximum(expected_deltas,0).sum()/len(tokens)),
                all_scene_negative_contribution_PDMS_points=100*float(-np.minimum(expected_deltas,0).sum()/len(tokens)),
                interpretation='Supervision target quality under admission/softmax only; not deployed reward or a guarantee, and no-teacher training is GT loss rather than reference retention'),
            source_scene_manifest_sha256=fingerprint(manifest))
    csv_path=output/'private_scene_admission.csv'
    with csv_path.open('x') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(private_rows[0]));writer.writeheader();writer.writerows(private_rows)
    results['private_scene_csv_sha256']=file_sha(csv_path)
    results['analysis_source_sha256']=file_sha(__file__);results['sha256']=fingerprint(results)
    (output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
