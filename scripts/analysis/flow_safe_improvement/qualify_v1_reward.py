"""Real-scene single-candidate v1 scoring invariance and CPU cost probe."""
import argparse
import json
from pathlib import Path
import pickle
import time

import numpy as np

from starVLA.rl.safe_flow_improvement.v1_reward import V1RewardService, protocol_identity
from starVLA.rl.flow_grpo.transactions import atomic_json


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--requests', required=True)
    p.add_argument('--population', required=True)
    p.add_argument('--cache-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=2)
    a = p.parse_args()
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(Path(a.population).read_text())
    with np.load(a.requests, allow_pickle=False) as data:
        tokens, trajectories = data['tokens'].tolist(), data['requested_trajectories'].copy()
    assert set(tokens) <= set(manifest['train_tokens'])
    spec = {'repo': str(Path(__file__).resolve().parents[3]), 'cache_roots': [a.cache_root], 'maps_root': '/mnt/navsim/maps'}
    jobs, references = [], []
    for token, requested in zip(tokens, trajectories):
        log = manifest['token_logs'][token]
        different = np.zeros((8, 3), dtype=requested.dtype)
        start = len(jobs)
        jobs.extend([(token, log, [requested]), (token, log, [different, requested]),
                     (token, log, [requested, different]), (token, log, [requested])])
        references.append((start, [(start, 0), (start + 1, 1), (start + 2, 0), (start + 3, 0)]))
    started = time.monotonic()
    report = {'status': 'RUNNING', 'protocol': protocol_identity(spec['repo']),
              'logical_candidate_evaluations': sum(len(j[2]) for j in jobs), 'scene_count': len(tokens),
              'purpose': 'Canonical contract check, not an unbiased model evaluation'}
    atomic_json(output / 'report.json', report)
    try:
        with V1RewardService(spec, a.workers) as service:
            rows = service.score(jobs)
        invariance = []
        for first, indices in references:
            expected = rows[first][0]
            equal = all(rows[i][j]['metrics'] == expected['metrics']
                        and np.array_equal(rows[i][j]['simulated_states'], expected['simulated_states'])
                        for i, j in indices)
            invariance.append(equal)
        # Save all requests and simulated outcomes, including valid zero scores.
        with (output / 'feedback.pkl').open('xb') as stream:
            pickle.dump(rows, stream, protocol=pickle.HIGHEST_PROTOCOL)
        seconds = [row['scoring_seconds'] for batch in rows for row in batch]
        report.update(status='PASS' if all(invariance) else 'FAIL', invariant_scenes=sum(invariance),
                      wall_seconds=time.monotonic() - started, scoring_cpu_seconds=sum(seconds),
                      median_seconds_per_candidate=float(np.median(seconds)),
                      baseline_scene_scores=[rows[i][0]['metrics']['score'] for i, _ in references],
                      errors_hidden_from_denominator=0)
        atomic_json(output / 'report.json', report)
        print(json.dumps({k: v for k, v in report.items() if k != 'protocol'}, sort_keys=True))
        if report['status'] != 'PASS':
            raise RuntimeError('Single-candidate canonical invariance failed')
    except Exception as error:
        report.update(status='FAILED', error_type=type(error).__name__, error=str(error))
        atomic_json(output / 'report.json', report)
        raise


if __name__ == '__main__':
    main()
