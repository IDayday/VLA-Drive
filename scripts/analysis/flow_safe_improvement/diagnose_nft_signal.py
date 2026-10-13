"""Measure the actual locked NFT advantage signal without another model update."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.analysis.flow_safe_improvement.diagnose_policy_projection import sealed
from scripts.analysis.flow_safe_improvement.prepare_target_bank import unpack
from scripts.analysis.flow_safe_improvement.run_original_evaluation_queue import authorized
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.contracts import SAFETY_METRICS, V1_COMPONENTS, fingerprint, file_sha
from starVLA.rl.safe_flow_improvement.iteration import require_dispatch


def distribution(values):
    x = np.asarray(values, dtype=np.float64)
    return dict(count=len(x), mean=float(x.mean()) if len(x) else None,
                quantiles=np.quantile(x, [0, .1, .25, .5, .75, .9, .99, 1]).tolist() if len(x) else None)


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ('targets', 'population', 'output'):
        p.add_argument('--'+name, required=True)
    a = p.parse_args(); authorized(require_dispatch())
    target = json.loads((Path(a.targets)/'manifest.json').read_text())
    pop = json.loads(Path(a.population).read_text()); bank = Path(target['candidate_bank_path'])
    manifest = json.loads((bank/'manifest.json').read_text())
    if (target['objective'] != 'nft' or target['qualification_only'] or len(target['tokens']) != 4096
            or target['population_identity'] != pop['identity_sha256'] or target['candidate_bank_sha256'] != manifest['sha256']
            or not set(target['tokens']) <= set(pop['train_tokens']) or not (Path(a.targets)/'COMPLETE.json').exists()):
        raise ValueError('Only complete locked TRAIN NFT labels can be diagnosed')
    groups, hashes = [], {}; all_adv=[]; all_rewards=[]; maximum_mismatch=0.
    for token in target['tokens']:
        label, label_sha = sealed(a.targets, token, target['sha256'])
        row, bank_sha = sealed(bank, token, manifest['sha256'])
        if label['candidate_payload_sha256'] != bank_sha:
            raise ValueError('Candidate-to-target identity changed')
        candidates = [c for c in unpack(row) if c.source == 'ode' and not c.is_reference]
        for c in candidates:
            c.require_nft_behavior(manifest['behavior_sha256'])
        reward=np.array([c.score for c in candidates], dtype=np.float64)
        advantage=label['advantages'].double().numpy(); detail=label['diagnostics']
        wanted=((reward-reward.mean())/(detail['global_reward_std']+detail['epsilon'])).clip(-5,5)
        maximum_mismatch=max(maximum_mismatch,float(np.max(np.abs(advantage-wanted))))
        varying=[metric for metric in V1_COMPONENTS if np.ptp([c.metrics[metric] for c in candidates]) > 1e-12]
        active=float(np.ptp(reward)) > 1e-12
        groups.append(dict(equal=not active, all_zero=bool(np.all(reward == 0)),
            progress_only=active and varying == ['ego_progress'],
            safety_variation=any(m in varying for m in SAFETY_METRICS),
            reward_std=float(reward.std()), reward_range=float(np.ptp(reward)),
            mean_abs_advantage=float(np.abs(advantage).mean()), max_abs_advantage=float(np.abs(advantage).max())))
        all_adv.extend(advantage);all_rewards.extend(reward);hashes[token]=dict(target=label_sha,bank=bank_sha)
    if maximum_mismatch > 1e-12:
        raise ValueError('Published NFT targets do not match their registered official reward transform')
    adv=np.asarray(all_adv);rewards=np.asarray(all_rewards)
    summary={key:sum(bool(x[key]) for x in groups) for key in ('equal','all_zero','progress_only','safety_variation')}
    report=dict(status='COMPLETE_ACTUAL_NFT_TRAIN_SIGNAL_DIAGNOSTIC', scenes=len(groups), candidates=len(adv),
        target_identity=target['sha256'], behavior_sha256=target['behavior_sha256'], bank_identity=manifest['sha256'],
        source_sha256=file_sha(__file__), input_payloads_sha256=fingerprint(hashes),
        group_counts=summary, global_reward_std=float(rewards.std()),
        group_reward_std=distribution([x['reward_std'] for x in groups]),
        group_reward_range=distribution([x['reward_range'] for x in groups]),
        absolute_advantage=distribution(np.abs(adv)), positive_fraction_distance_from_half=distribution(np.abs(adv)/10),
        exact_zero_advantage_fraction=float(np.mean(adv == 0)), clipped_at_five_fraction=float(np.mean(np.abs(adv) >= 5)),
        strata={label:dict(scenes=sum(x[key] for x in groups),
            group_mean_abs_advantage=distribution([x['mean_abs_advantage'] for x in groups if x[key]]))
            for label,key in [('progress_only','progress_only'),('safety_variation','safety_variation')]},
        advantage_reconstruction_max_abs_error=maximum_mismatch,
        initial_gradient_identity='At current_velocity == old_velocity, the positive and negative branches have opposite gradients with identical detached scales. The combined derivative equals clipped_advantage times the positive-branch derivative at beta=1. A positive scalar loss alone is not evidence of a strong policy-improvement gradient.',
        limitations='Offline signal measurement, not an actual full-head gradient decomposition. Does not prove replay/retention dominates, justify changing normalization, or establish NFT convergence or failure.',
        no_parameter_update_or_hyperparameter_selection=True)
    report['sha256']=fingerprint(report);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'report.json',report)
    print(json.dumps(dict(status=report['status'],group_counts=summary,absolute_advantage=report['absolute_advantage'])),flush=True)


if __name__ == '__main__':
    main()
