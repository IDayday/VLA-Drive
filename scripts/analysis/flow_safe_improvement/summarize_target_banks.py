"""Aggregate locked TRAIN target diagnostics without fitting new hyperparameters."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.contracts import file_sha, fingerprint
from starVLA.rl.safe_flow_improvement.iteration import require_dispatch


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--targets', nargs='+', required=True)
    p.add_argument('--population', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args(); require_dispatch()
    pop = json.loads(Path(a.population).read_text())
    result = {}
    for root_name in a.targets:
        root = Path(root_name)
        manifest = json.loads((root/'manifest.json').read_text())
        complete = json.loads((root/'COMPLETE.json').read_text())
        rows = json.loads((root/'diagnostics.json').read_text())
        if (manifest['objective'] not in ('reward', 'safe') or manifest['qualification_only']
                or manifest['population_identity'] != pop['identity_sha256']
                or complete['identity_sha256'] != manifest['sha256']
                or len(rows) != 4096 or {x['token'] for x in rows} != set(manifest['tokens'])
                or not set(manifest['tokens']) <= set(pop['train_tokens'])):
            raise ValueError('Require all 4096 locked TRAIN positive-FM targets')
        tables = {}
        for group in ('all', 'reference_stable_joint_safe', 'reference_not_stably_joint_safe'):
            chosen = [x for x in rows if group == 'all' or
                      (group == 'reference_stable_joint_safe') == bool(np.all(np.asarray(x['cost_reference']) == 0))]
            if not chosen:
                continue
            stats = {}
            for key in ('reward_reference', 'reward_weights', 'reward_target', 'gain_before_mixing', 'gain_after_mixing',
                        'alpha', 'ess_weights', 'ess_target', 'kl_to_original_u', 'fully_safe_mass_weights', 'fully_safe_mass_target'):
                values = np.array([x[key] for x in chosen], dtype=np.float64)
                if not np.isfinite(values).all():
                    raise ValueError('Nonfinite target diagnostics')
                stats[key] = dict(mean=float(values.mean()), quantiles=np.quantile(values, [0, .1, .5, .9, 1]).tolist())
            reference, weighted, mixed = [np.asarray([x[k] for x in chosen]) for k in ('cost_reference', 'cost_weights', 'cost_target')]
            tables[group] = dict(scenes=len(chosen), statistics=stats,
                no_reward_improvement_beyond_numeric_tolerance=sum(x['gain_before_mixing'] <= 1e-9 for x in chosen),
                alpha_below_001=sum(x['alpha'] < .01 for x in chosen), alpha_saturated=sum(x['alpha'] >= .5-1e-9 for x in chosen),
                mean_cost_reference=reference.mean(0).tolist(), mean_cost_weights=weighted.mean(0).tolist(),
                mean_cost_target=mixed.mean(0).tolist(), max_cost_increase=float(np.maximum(mixed-reference, 0).max()),
                max_primal_residual=max(x['primal_residual'] for x in chosen),
                max_target_primal_residual=max(x['target_primal_residual'] for x in chosen),
                solver_counts=dict(Counter(x['solver'] for x in chosen)))
        result[str(root)] = dict(identity_sha256=manifest['sha256'], objective=manifest['objective'],
            candidate_bank_sha256=manifest['candidate_bank_sha256'], behavior_sha256=manifest['behavior_sha256'],
            eta=manifest['eta'], alpha_max=manifest['alpha_max'], delta_useful=manifest['delta_useful'],
            diagnostics_sha256=file_sha(root/'diagnostics.json'), tables=tables)
    report = dict(status='COMPLETE_LOCKED_TARGET_SUMMARY', targets=result, source_sha256=file_sha(__file__),
                  units='raw_0_to_1', cost_order=['NC', 'DAC', 'TTC'], candidate_targets_are_not_policy_scores=True,
                  reference_not_stably_safe_does_not_mean_all_reference_samples_fail=True,
                  numeric_no_improvement_tolerance=1e-9, no_parameter_or_hyperparameter_fitting=True)
    report['sha256'] = fingerprint(report)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    atomic_json(out/'report.json', report)
    print(json.dumps(dict(status=report['status'], banks=len(result))), flush=True)


if __name__ == '__main__':
    main()
