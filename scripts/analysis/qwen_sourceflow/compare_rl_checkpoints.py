"""Read-only, same-parent comparison of two complete single-candidate RL evaluations.

This is an RL-to-RL checkpoint comparison, not improvement over an IL parent.
The existing scene seals, population checks and log bootstrap remain authoritative.
"""
import argparse
import json
from pathlib import Path


def require_same_contract(current, reference, tokens, reference_tokens):
    for key in ('schema', 'cohort', 'data_sha256', 'token_sha256', 'count',
                'metric', 'protocol', 'seed', 'inference', 'precision', 'source_files'):
        if current[key] != reference[key]:
            raise ValueError('Unmatched RL comparison: ' + key)
    if tokens != reference_tokens:
        raise ValueError('Unpaired scene population')
    if current['cohort'] != 'NAVTEST' or current['qualification'] or reference['qualification']:
        raise ValueError('Require complete Navtest evaluations')
    if not current['full_population'] or not reference['full_population']:
        raise ValueError('Subset scores are not full Navtest results')
    a, b = current['rl_export'], reference['rl_export']
    if not a or not b:
        raise ValueError('This comparison requires two RL exports')
    if (a['parent_model_sha256'] != b['parent_model_sha256'] or
            current['parent_model_sha256'] != reference['parent_model_sha256'] or
            a['parent_model_sha256'] != current['parent_model_sha256']):
        raise ValueError('RL models must share their exact IL parent')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--evaluation', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError('Preserve previous paired analysis: ' + str(destination))

    import numpy as np
    from scripts.analysis.qwen_sourceflow.analyze_paired import read
    from scripts.analysis.simwam_recipe.analyze_evaluation import METRICS, SAFETY
    from starVLA.rl.safe_flow_improvement.diagnostics import log_bootstrap_difference
    from starVLA.rl.qwen_sourceflow.contracts import file_sha, fingerprint
    from starVLA.rl.flow_grpo.transactions import atomic_json

    current, tokens, rows, values = read(args.evaluation)
    reference, reference_tokens, _, base = read(args.reference)
    require_same_contract(current, reference, tokens, reference_tokens)
    delta = values[:, 0] - base[:, 0]
    bootstrap = log_bootstrap_difference(delta, [row['log'] for row in rows])
    bootstrap['mean_PDMS_points'] = 100 * bootstrap['mean']
    bootstrap['CI95_PDMS_points'] = [100 * x for x in bootstrap['CI95']]
    old_safe = (base[:, 1:4] == 1).all(1)
    new_safe = (values[:, 1:4] == 1).all(1)
    result = dict(
        schema='same_parent_RL_checkpoints_paired_navtest_v1', status='COMPLETE',
        interpretation='RL-to-RL comparison; not RL gain over IL. Intermediate Test scan is exploratory.',
        evaluation=args.evaluation, reference=args.reference,
        evaluation_identity_sha256=current['sha256'], reference_identity_sha256=reference['sha256'],
        parent_model_sha256=current['parent_model_sha256'],
        analysis_sha256=file_sha(__file__), scenes=len(tokens), logs=len({r['log'] for r in rows}),
        seed=current['seed'], training_seed_repeats=1, inference_seeds=1,
        uncertainty='Scene-weighted paired log bootstrap; excludes seed and checkpoint-selection uncertainty',
        current_mean_percent={key: float(100 * values[:, i].mean()) for i, key in enumerate(METRICS)},
        reference_mean_percent={key: float(100 * base[:, i].mean()) for i, key in enumerate(METRICS)},
        mean_delta_percentage_points={key: float(100 * (values[:, i] - base[:, i]).mean())
                                      for i, key in enumerate(METRICS)},
        PDMS_log_cluster_bootstrap=bootstrap,
        G_plus_PDMS_points=float(100 * delta[delta > 0].sum() / len(delta)),
        G_minus_PDMS_points=float(-100 * delta[delta < 0].sum() / len(delta)),
        practical_improved_ge005=int((delta >= .005).sum()),
        practical_degraded_le_minus005=int((delta <= -.005).sum()),
        nonzero_to_zero=int(((base[:, 0] > 0) & (values[:, 0] == 0)).sum()),
        zero_to_nonzero=int(((base[:, 0] == 0) & (values[:, 0] > 0)).sum()),
        fixed_reference_safe_scenes=int(old_safe.sum()),
        fixed_reference_safe_new_joint_violation=int((old_safe & ~new_safe).sum()),
        fixed_reference_unsafe_joint_recovery=int((~old_safe & new_safe).sum()),
        safety_transitions={})
    for i, key in enumerate(SAFETY, 1):
        old, new = base[:, i] < 1, values[:, i] < 1
        result['safety_transitions'][key] = dict(
            reference_violation=int(old.sum()), current_violation=int(new.sum()),
            new_violation=int((~old & new).sum()), recovery=int((old & ~new).sum()))
    result['sha256'] = fingerprint(result)
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(destination, result)
    print(json.dumps({key: result[key] for key in ('status', 'scenes', 'mean_delta_percentage_points')}))


if __name__ == '__main__':
    main()
