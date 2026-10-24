"""CPU-only terminal L1/L2/L3 analysis using the single shared paired analyzer.

Requires complete registered evaluation populations; never runs a model, scorer,
or hyperparameter search. The finite screening window is not a training-seed
confirmation. Private per-scene outputs stay outside git.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.qwen_sourceflow.contracts import fingerprint, file_sha
from starVLA.rl.qwen_flow_exploration.protocol import read_protocol, ROOT
from starVLA.rl.qwen_flow_exploration.common_analysis import analyze_pair, load_evaluation


def root_diagnostics(path, evaluation):
    """A scene contributes once, rather than once per repeated inference seed."""
    rows = {}
    for file in sorted((Path(path)/'scenes').glob('*.json')):
        row = json.loads(file.read_text())
        if row['seed'] == evaluation.seeds[0]:
            rows[row['scene_token']] = row['latent_or_proposal_parameters']
    if set(rows) != set(evaluation.tokens):
        raise ValueError('Latent diagnostics must cover the same complete scenes')
    means, stds, kls = [], [], []
    for token in evaluation.tokens:
        parameters = rows[token]
        mean = np.asarray(parameters.get('mean',np.zeros((8,3))),dtype=np.float64)
        std = (np.exp(np.asarray(parameters['log_std'],dtype=np.float64)) if 'log_std' in parameters
               else np.full((8,3),evaluation.identity.get('root_scale',1.),dtype=np.float64))
        means.append(mean);stds.append(std)
        kls.append(float(.5*np.sum(mean**2+std**2-1-2*np.log(std))))
    means, stds, kls = np.asarray(means),np.asarray(stds),np.asarray(kls)
    score = evaluation.values[:,:,evaluation.metric_names.index('score')]
    indices = [evaluation.metric_names.index(key) for key in
               ['no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound']]
    feasible = np.all(evaluation.values[:,:,indices] == 1,axis=-1)
    return dict(scenes=len(evaluation.tokens),inference_draws=int(score.size),
                root_joint_prior_KL_mean=float(kls.mean()),root_joint_prior_KL_quantiles=np.quantile(kls,[0,.5,.9,1]).tolist(),
                root_per_dimension_prior_KL_mean=float(kls.mean()/24),
                root_mean_RMS=float(np.sqrt(np.mean(means**2))),root_std_mean=float(stds.mean()),
                root_std_quantiles=np.quantile(stds,[0,.01,.5,.99,1]).tolist(),
                root_std_near_bound_fraction=float(np.mean((stds <= .5001)|(stds >= 1.9999))),
                per_draw_joint_safe_fraction=float(feasible.mean()),
                per_draw_reward_ge09_fraction=float((score >= .9).mean()),
                per_draw_joint_safe_reward_ge09_fraction=float((feasible&(score >= .9)).mean()),
                finite_seed_any_joint_safe_scenes=int(feasible.any(1).sum()),
                caveat='These are fixed deployment-seed probabilities and frozen-root diagnostics, not K32 discovery curves, joint trajectory KL, or a safety guarantee')


def write_pair(output, name, current, reference, protocol, strata):
    result, rows = analyze_pair(current,reference,protocol,strata)
    folder = output/name
    folder.mkdir(exist_ok=False)
    csv_path=folder/'paired_scenes.csv'
    with csv_path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result.pop('sha256')
    result.update(paired_scene_results_path=str(csv_path.resolve()),paired_scene_results_sha256=file_sha(csv_path))
    result['sha256']=fingerprint(result)
    atomic_json(folder/'summary.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--protocol',default=str(ROOT/'common_protocol.json'))
    parser.add_argument('--evaluation-root',default=str(ROOT/'evaluations'))
    parser.add_argument('--reference-strata',default=str(ROOT/'common_bank/reference_strata.json'))
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    protocol=read_protocol(args.protocol)
    population=Path(args.evaluation_root)
    comparisons={}
    for arm in ['L1','L2','L3']:
        comparisons.update({arm+'_DEV_SELECT_sample':(arm+'_DEV_SELECT_sample','L0_sample'),
                            arm+'_TRAIN_sample':(arm+'_TRAIN_sample','L0_TRAIN'),
                            arm+'_DEV_mean':(arm+'_DEV_mean','L0_mean')})
    comparisons.update(L1_minus_L2_DEV=('L1_DEV_SELECT_sample','L2_DEV_SELECT_sample'),
                       L1_minus_L3_DEV=('L1_DEV_SELECT_sample','L3_DEV_SELECT_sample'),
                       L0S_07=('L0S_07','L0_sample'),L0S_13=('L0S_13','L0_sample'))
    names=sorted({name for pair in comparisons.values() for name in pair})
    missing=[name for name in names if not (population/name/'COMPLETE.json').is_file()]
    if missing:
        print(json.dumps(dict(status='WAITING_FOR_SEALED_EVALUATIONS',missing=missing)))
        return 2
    # Fail before publishing if any population or endpoint is invalid.
    loaded={name:load_evaluation(population/name,protocol) for name in names}
    expected_updates=protocol['screen']['rollout_rounds']*protocol['screen']['inner_epochs']
    for name,value in loaded.items():
        if name.startswith(('L1_','L2_','L3_')):
            state=value.identity['checkpoint_state']
            if state['phase']!='COMPLETE' or state['updates']!=expected_updates:
                raise ValueError('Only registered matched terminal models enter terminal analysis: '+name)
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    pairs={name:write_pair(output,name,population/current,population/reference,protocol,args.reference_strata)
           for name,(current,reference) in comparisons.items()}
    summary=dict(schema='frozen_root_terminal_screen_analysis_v1',status='COMPLETE',
                 protocol_sha256=protocol['sha256'],source_sha256=file_sha(__file__),
                 required_optimizer_updates=expected_updates,training_seed=42,
                 pairs=pairs,root_diagnostics={name:root_diagnostics(population/name,value) for name,value in loaded.items()},
                 causal_caveats=[
                     'L1 versus L2 has matched learning rate and update budget, not necessarily matched root-policy displacement; compare joint KL and retain this alternative explanation',
                     'L1 versus L3 isolates the registered prior coefficient, but training trajectories diverge once policies differ',
                     'DEV mean-root readout is deterministic and secondary; compare only against the separate mean-root baseline',
                     'DEV_SELECT has been used for finite selection; these screening results are not DEV_AUDIT, Navtest, or multi-training-seed confirmation',
                     'Independent stable strata are fixed; five inference draws are averaged inside scenes before log bootstrap',
                     'Fixed-scale evaluation costs and prior checkpoint construction costs are not free; use recorded run and evaluation ledgers',
                     'No conclusion about K32 discovery curves is inferred from five deployment draws'],
                 route_judgment='Requires human evidence review; software checks and a single training seed do not automatically establish route value')
    summary['sha256']=fingerprint(summary)
    atomic_json(output/'summary.json',summary)
    print(json.dumps(dict(status='COMPLETE',comparisons=len(pairs),output=str(output))))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
