"""Verify packaged files, scene aggregates, checkpoint selection and label archives."""
import argparse
import csv
import gzip
import hashlib
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--refresh-manifest', action='store_true')
    a = p.parse_args()
    root = a.root.resolve()
    manifest_path = root / 'MANIFEST.json'
    if a.refresh_manifest:
        files = {str(f.relative_to(root)): {'sha256': digest(f), 'bytes': f.stat().st_size}
                 for f in sorted(root.rglob('*')) if f.is_file() and f != manifest_path
                 and '__pycache__' not in f.parts}
        manifest_path.write_text(json.dumps({'schema': 'experiment_snapshot_sha256_v1', 'files': files}, indent=2) + '\n')
    manifest = json.loads(manifest_path.read_text())['files']
    for name, expected in manifest.items():
        path = root / name
        assert path.stat().st_size == expected['bytes'] and digest(path) == expected['sha256'], name
    jobs = {j['evaluation_id']: j for j in json.loads((root / 'data/evaluations.json').read_text())}
    best = list(csv.DictReader((root / 'data/best_checkpoints.csv').open()))
    best_ids = {}
    for row in best:
        matches = [key for key, j in jobs.items() if j['dataset'] == 'navtest'
                   and j['version'] == row['version'] and j['arm'] == row['arm']
                   and j['update'] == int(row['update'])]
        assert len(matches) == 1
        best_ids[row['name']] = matches[0]
    best_scenes = defaultdict(dict)
    counts, totals, logs, unique = defaultdict(int), defaultdict(float), defaultdict(set), defaultdict(set)
    for path in sorted((root / 'data/scenes').glob('*.csv.gz')):
        with gzip.open(path, 'rt', newline='') as f:
            for row in csv.DictReader(f):
                key = row['evaluation_id']
                assert key in jobs and row['status'] == 'ok'
                assert row['scene_id'] not in unique[key], (key, row['scene_id'])
                unique[key].add(row['scene_id'])
                counts[key] += 1
                totals[key] += float(row['PDMS'])
                logs[key].add(row['log_id'])
                if key in best_ids.values():
                    best_scenes[key][row['scene_id']] = row
        print('checked', path.name, flush=True)
    for key, job in jobs.items():
        assert counts[key] == job['expected_scenes'] and len(logs[key]) == job['expected_logs'], key
        assert abs(totals[key] / counts[key] - job['PDMS']) < 1e-8, key
    history = list(csv.DictReader((root / 'data/checkpoint_history.csv').open()))
    for row in best:
        candidates = [r for r in history if r['split'] == 'navtest' and r['version'] == row['version'] and r['arm'] == row['arm']]
        peak = max(candidates, key=lambda r: float(r['PDMS']))
        assert row['update'] == peak['update'] and row['checkpoint_sha256'] == peak['checkpoint_sha256']
        assert abs(float(row['PDMS']) - float(peak['PDMS'])) < 1e-8
    for version in ['r1', 'V12']:
        label_root = root / 'data/optimized_labels' / version
        identity = json.loads((label_root / 'identity.json').read_text())
        assert digest(label_root / 'labels.npz') == identity['labels_sha256']
        with zipfile.ZipFile(label_root / 'labels.npz') as z:
            assert z.testzip() is None
            assert set(z.namelist()) == {'tokens.npy', 'trajectories.npy', 'accepted.npy'}
        with np.load(label_root / 'labels.npz', allow_pickle=False) as arrays:
            assert arrays['trajectories'].shape == (identity['scenes'], 8, 3)
            assert arrays['accepted'].dtype == np.bool_ and arrays['accepted'].sum() == identity['accepted_scenes']
            assert len(set(arrays['tokens'].tolist())) == identity['scenes']
            assert np.isfinite(arrays['trajectories']).all()
    paired = json.loads((root / 'analysis/best_checkpoint_summary.json').read_text())['paired_comparisons']
    order = json.loads((root / 'data/bootstrap_log_order.json').read_text())['navtest']
    lookup = {name: i for i, name in enumerate(order)}
    weights = np.random.default_rng(20260928).multinomial(len(order), [1 / len(order)] * len(order), size=10000)
    metrics = ['PDMS', 'NC', 'DAC', 'TTC', 'EP']
    for name, expected in paired.items():
        first, baseline = name.split('-', 1)
        left, right = best_scenes[best_ids[first]], best_scenes[best_ids[baseline]]
        assert set(left) == set(right)
        tokens = sorted(left)
        indices = np.array([lookup[left[t]['log_id']] for t in tokens])
        counts_by_log = np.bincount(indices, minlength=len(order))
        delta = np.array([[float(left[t][k]) - float(right[t][k]) for k in metrics] for t in tokens])
        sums = np.zeros((len(order), len(metrics)))
        np.add.at(sums, indices, delta)
        samples = (weights @ sums) / (weights @ counts_by_log)[:, None]
        for i, metric in enumerate(metrics):
            target = expected['metrics'][metric]
            assert abs(float(delta[:, i].mean()) - target['delta_points']) < 1e-8, (name, metric)
            assert np.max(np.abs(np.quantile(samples[:, i], [.025, .975]) - target['paired_log_95CI'])) < 1e-8, (name, metric)
    snapshot = json.loads((root / 'provenance/snapshot.json').read_text())
    assert len(jobs) == snapshot['evaluation_count'] and sum(counts.values()) == snapshot['scene_rows']
    result = {'status': 'PASS', 'files': len(manifest), 'evaluations': len(jobs),
              'scene_rows': sum(counts.values()), 'best_checkpoints': len(best),
              'label_versions': ['r1', 'V12'], 'recomputed_paired_comparisons': len(paired)}
    print(json.dumps(result))


if __name__ == '__main__':
    main()
