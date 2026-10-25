"""Read completed four-scene TRAIN expression evidence; no policy/scorer calls."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.qwen_flow_exploration.checkpoint import load_state
from starVLA.rl.qwen_flow_exploration.feedback.expression import CONTRACT,sealed,read_registration,validate_evaluation_population
from starVLA.rl.qwen_flow_exploration.feedback.process import SAFETY
from starVLA.rl.qwen_flow_exploration.protocol import read_protocol
from starVLA.rl.qwen_sourceflow.contracts import fingerprint,file_sha
from starVLA.rl.safe_flow_improvement.contracts import seed_for


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--protocol',required=True);p.add_argument('--registration',required=True)
    p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    output=Path(a.output)
    if output.exists():raise FileExistsError('Keep older fit analysis; choose new output')
    protocol=read_protocol(a.protocol);reg=read_registration(a.registration,protocol,check_sources=False);run=Path(a.run)
    app=sealed(run/'COMPLETE.json');state,payload=load_state(run,protocol['sha256'],restore_rng=False)
    if (app['registration_sha256']!=reg['sha256'] or payload['registration_sha256']!=reg['sha256'] or
        state['phase']!='COMPLETE' or state['updates']!=512 or state['scene_cursor']!=2048 or
        {int(row['step']) for row in payload['optimizer']['state'].values()}!={512} or len(payload['optimizer']['state'])!=480):
        raise ValueError('Probe terminal optimizer/data identity is incomplete')
    actual=validate_evaluation_population(run/'evaluation',reg)
    per_case=[];aggregate={}
    for update in CONTRACT['evaluation_updates']:
        records=[];cases=[]
        folder=run/'evaluation'/f'update{update:06d}';identity=sealed(folder/'identity.json')
        for index,teacher in enumerate(reg['scenes']):
            outputs=[]
            for seed in CONTRACT['evaluation_seeds']:
                row=sealed(folder/'scenes'/f"{teacher['scene_token']}_seed{seed}.json")
                if row['evaluation_identity']!=identity['sha256'] or row['status']!='OK':raise ValueError('Changed single output')
                root=torch.randn((1,8,3),generator=torch.Generator().manual_seed(seed_for(CONTRACT['root_namespace'],seed,teacher['scene_token'])))[0]
                if not np.array_equal(np.asarray(row['latent_or_proposal_parameters']['root']),root.numpy()):raise ValueError('Different physical-evaluation root')
                physical=np.asarray(row['planned_trajectory']);normalized=torch.tensor(row['normalized_action'],dtype=torch.float32)
                if physical.shape!=(8,3) or np.max(np.abs(ActionCodec.denorm_odo(normalized).numpy()-physical))>1e-5:
                    raise ValueError('Changed request/codec contract')
                outputs.append(row);records.append(row)
            target=np.asarray(teacher['planned_trajectory'])
            metrics=sorted(outputs[0]['native_subscores'])
            case=dict(case_index=index,category=teacher['category'],updates=update,teacher_score=teacher['teacher_reward'],
                PDMS=100*float(np.mean([row['raw_reward'] for row in outputs])),
                original_native_means={key:float(np.mean([row['native_subscores'][key] for row in outputs])) for key in metrics},
                safe_outputs=sum(all(row['native_subscores'][key]==1 for key in SAFETY) for row in outputs),
                exact_zero_outputs=sum(row['raw_reward']==0 for row in outputs),
                within005_teacher_score_outputs=sum(row['raw_reward']>=teacher['teacher_reward']-.005 for row in outputs),
                mean_requested_ADE_to_fixed_single_teacher_m=float(np.mean([
                    np.linalg.norm(np.asarray(row['planned_trajectory'])[:,:2]-target[:,:2],axis=-1).mean() for row in outputs])))
            cases.append(case);per_case.append(case)
        aggregate[str(update)]=dict(PDMS=100*float(np.mean([row['raw_reward'] for row in records])),
            fully_safe_outputs=sum(all(row['native_subscores'][key]==1 for key in SAFETY) for row in records),
            exact_zero_outputs=sum(row['raw_reward']==0 for row in records),logical_outputs=20,
            mean_requested_ADE_to_fixed_single_teacher_m=float(np.mean([case['mean_requested_ADE_to_fixed_single_teacher_m'] for case in cases])))
    result=dict(schema='TRAIN_teacher_expression_probe_result_v1',qualification=True,TRAIN_fit_only=True,
        not_generalization_or_formal_method_arm=True,registration_sha256=reg['sha256'],actual_single_ODE_outputs=actual,
        optimizer_updates=512,teacher_exposures=2048,GT_or_anchor_loss=False,changed_hyperparameters_during_run=False,
        milestones=aggregate,case_results=per_case,
        final_minus_initial_PDMS=aggregate['512']['PDMS']-aggregate['0']['PDMS'],
        note='Purposively selected four TRAIN scenes, five fixed roots each. No oracle output selection, label averaging, independence/CI or Navtest claim.',
        logical_and_physical_cost=app.get('cumulative_physical_cost_including_failed_attempts'),
        analysis_source_sha256=file_sha(__file__))
    result['sha256']=fingerprint(result);atomic_json(output,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
