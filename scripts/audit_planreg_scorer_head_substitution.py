#!/usr/bin/env python3
"""Offline true-head substitution on immutable candidates, never deployment."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_planreg_candidate_tasks import _cluster_bootstrap, _metric_cache_logs, _sha256

NC = 'no_at_fault_collisions'
DAC = 'drivable_area_compliance'
TTC = 'time_to_collision_within_bound'
EP = 'ego_progress'
C = 'comfort'
DDC = 'driving_direction_compliance'


def substitution_log_score(probabilities, truth, replace, original_log_score=None):
    """Use unchanged sigmoid probabilities and the exact weighted formula.

    NC/DDC 0.5 -> 0 is the fixed loss label map. An invalid TTC (2) is
    unavailable oracle information, so that candidate keeps its prediction.
    DDC has zero weight and is deliberately not logged (0 * log(0) is NaN).
    """
    values = {name: np.asarray(value, dtype=np.float64).copy() for name, value in probabilities.items()}
    for name in replace:
        target = np.asarray(truth[name], dtype=np.float64).copy()
        if name in (NC, DDC):
            target[target == 0.5] = 0
        if name == TTC:
            target = np.where(target == 2, values[name], target)
        if not np.isfinite(target).all() or ((target < 0) | (target > 1)).any():
            raise ValueError(f'Invalid official component labels: {name}')
        values[name] = target
    def aggregate(v):
        with np.errstate(divide='ignore'):
            return np.log(v[NC]) + np.log(v[DAC]) + np.log(5*v[TTC] + 5*v[EP] + 2*v[C])
    result = aggregate(values)
    if original_log_score is not None:
        # Keep the original float32 log-score rounding when no component is
        # changed; offset only finite originals to avoid inf-inf subtraction.
        old = aggregate(probabilities)
        finite = np.isfinite(old)
        result[finite] += np.asarray(original_log_score)[finite] - old[finite]
    if np.isnan(result).any():
        raise FloatingPointError('NaN in offline substitution aggregation')
    return result


def audit(bank_path, metric_cache, output, bootstrap_samples=2000):
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite diagnostic: {output}')
    with np.load(bank_path, allow_pickle=False) as bank:
        tokens = bank['tokens']
        p = {str(n): bank['component_probabilities'][..., i] for i, n in enumerate(bank['component_names'])}
        truth = {str(n): bank['candidate_scores'][..., i] for i, n in enumerate(bank['official_component_names'])}
        selected = bank['selected_indices']
        log_score = bank['predicted_log_pdm']
    scores = truth['pdm_score']
    rows = np.arange(len(tokens))
    baseline = scores[rows, selected]
    logs = _metric_cache_logs(tokens, metric_cache)
    cases = {}
    for label, replacement in [(n, (n,)) for n in (NC, DAC, TTC, EP, C)] + [('EP+TTC', (EP, TTC))]:
        indices = substitution_log_score(p, truth, replacement, log_score).argmax(1)
        values = scores[rows, indices]
        delta = values - baseline
        cases[label] = {'selected_pdms': float(values.mean()), 'delta': float(delta.mean()),
                        'benefited_scene_count': int((delta > 1e-8).sum()),
                        'harmed_scene_count': int((delta < -1e-8).sum()),
                        'unchanged_scene_count': int((abs(delta) <= 1e-8).sum()),
                        'catastrophic_fraction': float(((scores.max(1) > .9) & (values < .5)).mean()),
                        'paired_log_cluster_ci': _cluster_bootstrap({'delta': delta}, logs, samples=bootstrap_samples, seed=42)['delta']}
    result = {'schema_version': 1, 'evidence_type': 'new_offline_analysis_of_existing_bank',
              'bank': str(bank_path), 'bank_sha256': _sha256(bank_path),
              'scene_count': len(tokens), 'log_count': len(set(logs)),
              'baseline_selected_from_stored_indices': float(baseline.mean()),
              'offline_oracle_at_64': float(scores.max(1).mean()),
              'invalid_ttc_count': int((truth[TTC] == 2).sum()), 'substitutions': cases,
              'deployable': False, 'training_labels_generated': False,
              'interpretation': 'Conditional offline head replacement bounds; not independent causal contributions or deployable PDMS.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-bank', type=Path, required=True)
    parser.add_argument('--metric-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    args = parser.parse_args()
    print(json.dumps(audit(args.candidate_bank, args.metric_cache, args.output, args.bootstrap_samples), indent=2))
