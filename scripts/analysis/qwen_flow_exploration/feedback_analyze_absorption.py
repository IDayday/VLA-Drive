"""Existing TRAIN outputs only: each single request's nearest teacher ADE.

The minimum is over supervision labels for a diagnostic, never over policy
outputs. All five independently generated single outputs remain in the metric
denominator. No labels are averaged and no actor/scorer is run.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np
import torch

from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
from starVLA.rl.qwen_flow_exploration.common_analysis import checked_seal,load_evaluation,deployment_contract,zero_summary
from starVLA.rl.qwen_flow_exploration.feedback.process import SAFETY
from starVLA.rl.qwen_flow_exploration.protocol import read_protocol
from starVLA.rl.qwen_sourceflow.contracts import file_sha,fingerprint
from starVLA.rl.safe_flow_improvement.diagnostics import log_bootstrap_difference


def trajectory_key(physical):
    return np.asarray(physical,dtype=np.float32).tobytes()


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--protocol',required=True)
    parser.add_argument('--reference-strata',required=True);parser.add_argument('--search-root',required=True)
    parser.add_argument('--admission-analysis',required=True);parser.add_argument('--evaluation',action='append',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    protocol=read_protocol(args.protocol);output=Path(args.output)
    if output.exists():raise FileExistsError('Keep previous analysis; use a new output version')
    paths=dict(value.split('=',1) for value in args.evaluation)
    if set(paths)!={'S0','S1','S2','S3'}:raise ValueError('Exactly original base/GT-only/scalar/feedback policies are required')
    admission_path=Path(args.admission_analysis)
    admission=checked_seal(json.loads((admission_path/'summary.json').read_text()),'fixed admission')
    csv_source=admission_path/'private_scene_admission.csv'
    if file_sha(csv_source)!=admission['private_scene_csv_sha256']:raise ValueError('Changed admitted TRAIN population')
    covered={method:set() for method in ('F2_PDMS','F3_FEEDBACK')}
    for row in csv.DictReader(csv_source.open()):
        if row['teacher_admitted']=='True':covered[row['method']].add(row['scene_token'])
    if covered['F2_PDMS']!=covered['F3_FEEDBACK'] or len(covered['F2_PDMS'])!=237:
        raise ValueError('Expected fixed common237 teacher scenes')
    tokens=sorted(covered['F2_PDMS'])
    strata=checked_seal(json.loads(Path(args.reference_strata).read_text()),'reference strata')
    if strata['protocol_sha256']!=protocol['sha256']:raise ValueError('Other fixed strata')
    teacher_sets={method:{} for method in ('F2_PDMS','F3_FEEDBACK','COMMON_EXACT')}
    repair_flags={};teacher_hashes={};teacher_roundtrip=0.
    for token in tokens:
        sets={}
        for method in ('F2_PDMS','F3_FEEDBACK'):
            path=Path(args.search_root)/method/'TRAIN/scenes'/f'{token}.json'
            record=checked_seal(json.loads(path.read_text()),str(path))
            if record['split']!='TRAIN' or record['scene_token']!=token or record['common_protocol_sha256']!=protocol['sha256']:
                raise ValueError('Other teacher scene/split')
            teachers=record['teacher_candidates'];base=record['result']['candidates'][0]
            if not teachers or len(teachers)>4:raise ValueError('Missing fixed teachers')
            repair_flags[token]=any(base['native_subscores'][metric]<1 for metric in SAFETY)
            for row in teachers:
                if (row['split']!='TRAIN' or row.get('on_policy_logprob') is not None or
                    row['raw_reward']<max(.9,base['raw_reward']+.005) or
                    not all(row['native_subscores'][metric]==1 for metric in SAFETY)):
                    raise ValueError('Unqualified teacher entered absorption audit')
                physical=torch.tensor(row['planned_trajectory'],dtype=torch.float32)
                normalized=torch.tensor(row['normalized_action'],dtype=torch.float32)
                error=float((ActionCodec.denorm_odo(normalized)-physical).abs().max())
                if error>1e-5:raise ValueError('Teacher codec/local-frame contract mismatch')
                teacher_roundtrip=max(teacher_roundtrip,error)
            teacher_sets[method][token]=np.asarray([row['planned_trajectory'] for row in teachers],dtype=np.float64)
            sets[method]={trajectory_key(row['planned_trajectory']):row['planned_trajectory'] for row in teachers}
            teacher_hashes[method+'/'+token]=record['sha256']
        intersection=sets['F2_PDMS'].keys()&sets['F3_FEEDBACK'].keys()
        if not intersection:raise ValueError('No exact common qualified label for scene')
        teacher_sets['COMMON_EXACT'][token]=np.asarray([sets['F2_PDMS'][key] for key in sorted(intersection)],dtype=np.float64)
    evaluations={arm:load_evaluation(path,protocol) for arm,path in paths.items()}
    base_eval=evaluations['S0'];contract=deployment_contract(base_eval.identity,protocol)
    for evaluation in evaluations.values():
        if (evaluation.identity['split']!='TRAIN' or evaluation.tokens!=base_eval.tokens or evaluation.seeds!=base_eval.seeds or
            evaluation.logs!=base_eval.logs or evaluation.metric_names!=base_eval.metric_names or
            deployment_contract(evaluation.identity,protocol)!=contract):
            raise ValueError('Unmatched primary single-output TRAIN evaluation')
    source_indices=[base_eval.tokens.index(token) for token in tokens]
    logs=np.asarray(base_eval.logs)[source_indices].tolist();metrics=base_eval.metric_names
    score_index=metrics.index('score');safety_indices=[metrics.index(name) for name in SAFETY]
    cohorts={'all_common_teacher237':np.ones(len(tokens),dtype=bool),
             'teacher_reference_safety_recovery':np.asarray([repair_flags[token] for token in tokens]),
             'teacher_reference_already_safe':np.asarray([not repair_flags[token] for token in tokens])}
    for label in ('stable_high','stable_zero','other_or_unstable','low_nonzero'):
        mask=np.asarray([strata['by_split']['TRAIN'][token]['stratum']==label for token in tokens])
        if mask.any():cohorts[label]=mask
    result=dict(schema='feedback_actual_target_absorption_v1',protocol_sha256=protocol['sha256'],
        population='same237 TRAIN scenes with admitted F2/F3 teachers; not DEV or test generalization',
        scenes=len(tokens),logs=len(set(logs)),evaluation_seeds=list(base_eval.seeds),
        metric_semantics='Each original single output retained; seeds averaged per scene, then scenes equally weighted',
        ADE_semantics='min over qualified supervision labels of mean XY distance at eight .5sec request points, metres; per generated output, never output selection or label averaging',
        coordinates='original source8x3 rear-axle local x-forward/y-left/heading; requested trajectories, not simulated labels',
        teacher_set_sizes={name:sum(len(rows) for rows in sets.values()) for name,sets in teacher_sets.items()},
        teacher_scene_manifest_sha256=fingerprint(teacher_hashes),teacher_codec_roundtrip_max_abs=teacher_roundtrip,
        common_qualified_set='Exact original float32 requested8x3 intersection per scene, all237 scenes retained',
        no_new_model_score_optimizer_or_GT_actor_calls=True,models={},paired={})
    ade={};values={};private_rows=[];policy_roundtrip=0.
    for arm,evaluation in evaluations.items():
        planned=[]
        for token in tokens:
            single=[]
            for seed in evaluation.seeds:
                file=Path(paths[arm])/'scenes'/f'{token}_seed{seed}.json'
                row=checked_seal(json.loads(file.read_text()),str(file))
                physical=np.asarray(row['planned_trajectory'],dtype=np.float64)
                normalized=torch.tensor(row['normalized_action'],dtype=torch.float32)
                error=float(np.abs(ActionCodec.denorm_odo(normalized).numpy()-physical).max())
                if error>1e-5:raise ValueError('Policy codec/local-frame contract mismatch')
                policy_roundtrip=max(policy_roundtrip,error);single.append(physical)
            planned.append(single)
        planned=np.asarray(planned);values[arm]=evaluation.values[source_indices]
        ade[arm]={}
        for name,sets in teacher_sets.items():
            ade[arm][name]=np.asarray([np.linalg.norm(planned[i,:,None,:,:2]-sets[token][None,:,:,:2],axis=-1).mean(-1).min(-1)
                                      for i,token in enumerate(tokens)])
        model=dict(evaluation_identity=evaluation.identity['sha256'],cohorts={})
        for name,mask in cohorts.items():
            v=values[arm][mask];score=v[:,:,score_index]
            model['cohorts'][name]=dict(scenes=int(mask.sum()),
                mean_percent={metric:float(100*v[:,:,index].mean()) for index,metric in enumerate(metrics)},
                joint_fully_safe_single_output_rate=float((v[:,:,safety_indices]==1).all(-1).mean()),
                any_seed_joint_violation_scenes=int((v[:,:,safety_indices]<1).any(-1).any(-1).sum()),
                zero=zero_summary(score),
                mean_nearest_teacher_ADE_m={key:float(distance[mask].mean()) for key,distance in ade[arm].items()})
        result['models'][arm]=model
        for index,token in enumerate(tokens):
            private_rows.append(dict(arm=arm,scene_token=token,log_id=logs[index],
                fixed_stratum=strata['by_split']['TRAIN'][token]['stratum'],
                teacher_reference_safety_recovery=repair_flags[token],
                mean_score=float(values[arm][index,:,score_index].mean()),
                joint_fully_safe_draw_fraction=float((values[arm][index,:,safety_indices]==1).all(0).mean()),
                **{name+'_nearest_ADE_m':float(distance[index].mean()) for name,distance in ade[arm].items()}))
    for left,right in (('S1','S0'),('S2','S0'),('S3','S0'),('S2','S1'),('S3','S1'),('S3','S2')):
        score_delta=values[left][:,:,score_index].mean(1)-values[right][:,:,score_index].mean(1)
        result['paired'][left+'-'+right]=dict(
            PDMS_difference_raw_log_cluster=log_bootstrap_difference(score_delta,logs,repeats=5000,seed=42),
            nearest_teacher_ADE_difference_log_cluster={name:log_bootstrap_difference(
                ade[left][name].mean(1)-ade[right][name].mean(1),logs,repeats=5000,seed=42) for name in teacher_sets},
            ADE_negative_difference_means_closer_not_a_reward_or_safety_guarantee=True)
    output.mkdir(parents=True)
    private=output/'private_per_scene_absorption.csv'
    with private.open('x') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(private_rows[0]));writer.writeheader();writer.writerows(private_rows)
    result['private_per_scene_csv_sha256']=file_sha(private)
    result['policy_codec_roundtrip_max_abs']=policy_roundtrip
    result['analysis_source_sha256']=file_sha(__file__);result['sha256']=fingerprint(result)
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({arm:record['cohorts']['all_common_teacher237'] for arm,record in result['models'].items()},indent=2))


if __name__=='__main__':main()
