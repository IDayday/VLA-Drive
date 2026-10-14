#!/usr/bin/env python3
"""Measure existing D_S supervision geometry without generating or fitting data."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from iqe.contracts import require
from iqe.io import atomic_json, atomic_bytes, file_hash
from iqe.robust_scorer.config import load
from iqe.specialist_study.bank import load_bank
from iqe.scoring.reference import COMPONENTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = load(args.config)
    output = config.root/'qualification/boundary_coverage'
    rules = dict(role='scorer_fit', banks=['deploy', 'rich'], no_dev_or_test_access=True,
                 ade_thresholds_m=[.1, .25, .5, 1., 2.], score_gap_threshold=.5,
                 high_score=.8, zero_epsilon=1e-8,
                 distance='mean L2 xy distance over the same eight physical waypoints',
                 safety_boundary='one candidate NC and DAC full, the other NC or DAC not full',
                 interpretation='descriptive proximity, not physical collision clearance or causal proof',
                 new_candidates=0, new_optimizer_steps=0)
    atomic_json(output/'RULES.json', rules, immutable=True)
    results = {}
    for kind in rules['banks']:
        meta, bank = load_bank(config.source(), kind, 'scorer_fit')
        valid = np.asarray(bank['comparison_valid'], bool)
        scores, trajectories = bank['scores'], bank['trajectories']
        n, k = scores.shape
        nc = bank['components'][..., COMPONENTS.index('no_at_fault_collisions')]
        dac = bank['components'][..., COMPONENTS.index('drivable_area_compliance')]
        component_valid = np.asarray(bank['component_valid'], bool)
        known = (np.isfinite(nc) & np.isfinite(dac) & valid
                 & component_valid[..., COMPONENTS.index('no_at_fault_collisions')]
                 & component_valid[..., COMPONENTS.index('drivable_area_compliance')])
        safe = known & (nc >= 1-1e-8) & (dac >= 1-1e-8)
        fail = known & ~safe
        i, j = np.triu_indices(k, 1)
        pair_valid = valid[:, i] & valid[:, j]
        distance = np.linalg.norm(trajectories[:, i, :, :2]-trajectories[:, j, :, :2], axis=-1).mean(-1)
        safety_cross = ((safe[:, i] & fail[:, j]) | (fail[:, i] & safe[:, j])) & pair_valid
        big_gap = pair_valid & (np.abs(scores[:, i]-scores[:, j]) >= rules['score_gap_threshold'])
        require(np.isfinite(distance[pair_valid]).all(), 'invalid physical trajectory in an official bank')
        def count(mask):
            scenes = mask.any(1)
            return dict(pairs=int(mask.sum()), scenes=int(scenes.sum()),
                        logs=len(set(np.asarray(meta['source_logs'])[scenes])))
        thresholds = []
        for threshold in rules['ade_thresholds_m']:
            close = distance <= threshold
            thresholds.append(dict(ADE_max_m=threshold, safety_crossing=count(close & safety_cross),
                                   gap_at_least_0_5=count(close & big_gap), all_pairs=count(close & pair_valid)))
        base_distance = np.linalg.norm(trajectories[..., :2]-trajectories[:, :1, :, :2], axis=-1).mean(-1)
        # Rich rows preserve the original Base first; metadata identity is
        # additionally checked against the actual deploy row's physical values.
        if kind == 'deploy':
            base_physical = trajectories[:, 0].copy()
            ids = meta['scene_ids']
        else:
            require(meta['scene_ids'] == ids and np.array_equal(trajectories[:, 0], base_physical),
                    'rich/deploy Base or scene join changed')
        base_high = scores[:, 0] >= rules['high_score']
        base_near_failure = []
        for threshold in rules['ade_thresholds_m']:
            mask = (base_distance <= threshold) & fail & base_high[:, None]
            found = mask.any(1)
            base_near_failure.append(dict(ADE_max_m=threshold, candidates=int(mask.sum()), scenes=int(found.sum()),
                                          logs=len(set(np.asarray(meta['source_logs'])[found]))))
        results[kind] = dict(bank_hash=meta['hash'], scene_count=n, log_count=len(set(meta['source_logs'])),
                             valid_unique_candidates=int(valid.sum()), maximum_K=k,
                             zero_candidates=int((valid & (scores <= 1e-8)).sum()),
                             safety_failure_candidates=int(fail.sum()),
                             all_safety_crossings=count(safety_cross), all_large_gap_pairs=count(big_gap),
                             close_pairs=thresholds, base_high_denominator=int(base_high.sum()),
                             high_Base_near_safety_failure=base_near_failure)
    result = dict(rules_hash=file_hash(output/'RULES.json'), rows=results, no_training_or_selection=True)
    atomic_json(output/'REPORT.json', result, immutable=True)
    lines = ['# Existing Scorer supervision near trajectory boundaries', '',
             'Read-only D_S audit. No validation, dev, test, new trajectory, optimizer or threshold selection.',
             'XY ADE is geometric proximity, not a simulator clearance metric; close candidates can still differ dynamically.', '',
             '| Bank | ADE limit (m) | Safety-crossing pairs | Scenes | Logs | Large reward-gap pairs |',
             '|---|---:|---:|---:|---:|---:|']
    for kind, row in results.items():
        for entry in row['close_pairs']:
            s=entry['safety_crossing']
            lines.append(f"| {kind} | {entry['ADE_max_m']} | {s['pairs']} | {s['scenes']} | {s['logs']} | "
                         f"{entry['gap_at_least_0_5']['pairs']} |")
    lines += ['', 'Neither abundant total negatives nor a small ADE crossing count proves the cause of generalization failure.',
              'These counts describe supervision coverage and cannot be substituted for trained heldout performance.']
    atomic_bytes(output/'REPORT.md', ('\n'.join(lines)+'\n').encode(), immutable=True)
    print(output)


if __name__ == '__main__':
    main()
