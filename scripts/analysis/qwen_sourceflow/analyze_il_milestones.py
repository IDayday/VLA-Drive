"""Read-only paired comparison of two complete native IL evaluations."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.analysis.qwen_sourceflow.analyze_paired import read
from scripts.analysis.simwam_recipe.analyze_evaluation import METRICS, SAFETY, motion
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.qwen_sourceflow.contracts import file_sha, fingerprint
from starVLA.rl.safe_flow_improvement.diagnostics import log_bootstrap_difference


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--evaluation', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError('Preserve previous analyses: ' + str(destination))

    current, tokens, rows, values = read(args.evaluation)
    reference, ref_tokens, ref_rows, base = read(args.reference)
    for key in ['schema', 'cohort', 'data_sha256', 'token_sha256', 'count',
                'metric', 'protocol', 'seed', 'inference', 'precision', 'source_files']:
        if current[key] != reference[key]:
            raise ValueError('Unmatched native IL evaluation: ' + key)
    if tokens != ref_tokens or current['rl_export'] is not None or reference['rl_export'] is not None:
        raise ValueError('This analyzer requires paired IL milestones, not RL exports')
    if current['parent_model_sha256'] == reference['parent_model_sha256']:
        raise ValueError('Require two distinct, explicitly identified IL models')

    delta = values[:, 0] - base[:, 0]
    old_safe = (base[:, 1:4] == 1).all(1)
    new_safe = (values[:, 1:4] == 1).all(1)
    bootstrap = log_bootstrap_difference(delta, [r['log'] for r in rows])
    bootstrap['mean_PDMS_points'] = 100 * bootstrap['mean']
    bootstrap['CI95_PDMS_points'] = [100 * x for x in bootstrap['CI95']]
    result = dict(
        schema='native_IL_milestones_paired_v1', status='COMPLETE',
        interpretation='IL checkpoint learning only; not RL gain or IL parent selection',
        evaluation=args.evaluation, reference=args.reference,
        current_checkpoint=current['checkpoint'], reference_checkpoint=reference['checkpoint'],
        current_model_sha256=current['parent_model_sha256'],
        reference_model_sha256=reference['parent_model_sha256'],
        analysis_sha256=file_sha(__file__), scenes=len(tokens),
        logs=len({r['log'] for r in rows}), seed=current['seed'],
        inference=current['inference'], metric=current['metric'],
        training_seed_repeats=1, inference_seeds=1,
        uncertainty='Scene-weighted log-cluster bootstrap; excludes training/inference seed uncertainty',
        current_mean_percent={k: float(100 * values[:, i].mean()) for i, k in enumerate(METRICS)},
        reference_mean_percent={k: float(100 * base[:, i].mean()) for i, k in enumerate(METRICS)},
        mean_delta_percentage_points={k: float(100 * (values[:, i] - base[:, i]).mean())
                                      for i, k in enumerate(METRICS)},
        PDMS_log_cluster_bootstrap=bootstrap,
        G_plus_raw_score_sum=float(delta[delta > 0].sum()),
        G_minus_raw_score_sum=float(-delta[delta < 0].sum()),
        G_plus_PDMS_points=float(100 * delta[delta > 0].sum() / len(delta)),
        G_minus_PDMS_points=float(-100 * delta[delta < 0].sum() / len(delta)),
        any_improved=int((delta > 0).sum()), any_degraded=int((delta < 0).sum()),
        practical_improved_ge005=int((delta >= .005).sum()),
        practical_degraded_le_minus005=int((delta <= -.005).sum()),
        reference_zero_scenes=int((base[:, 0] == 0).sum()),
        current_zero_scenes=int((values[:, 0] == 0).sum()),
        nonzero_to_zero=int(((base[:, 0] > 0) & (values[:, 0] == 0)).sum()),
        zero_to_nonzero=int(((base[:, 0] == 0) & (values[:, 0] > 0)).sum()),
        reference_joint_safe_scenes=int(old_safe.sum()), current_joint_safe_scenes=int(new_safe.sum()),
        reference_safe_new_joint_violation=int((old_safe & ~new_safe).sum()),
        reference_unsafe_joint_recovery=int((~old_safe & new_safe).sum()),
        current_full_score_scenes=int((values[:, 0] == 1).sum()), safety_transitions={})
    for i, key in enumerate(SAFETY, 1):
        old, new = base[:, i] < 1, values[:, i] < 1
        result['safety_transitions'][key] = dict(
            reference_violation=int(old.sum()), current_violation=int(new.sum()),
            new_violation=int((~old & new).sum()), recovery=int((old & ~new).sum()))
    for label, population in [('current', rows), ('reference', ref_rows)]:
        motions = [motion(row['feedback']) for row in population]
        result[label + '_motion'] = {
            key: dict(mean=float(np.mean(v)) if v else None, defined_scenes=len(v))
            for key in motions[0] for v in [[m[key] for m in motions if m[key] is not None]]}
    result['sha256'] = fingerprint(result)
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(destination, result)
    print(json.dumps({k: result[k] for k in ['status', 'scenes', 'mean_delta_percentage_points']}))


if __name__ == '__main__':
    main()
