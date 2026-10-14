"""Post-execution ego trajectory errors using native NAVSIM ground truth.

No labels are made available to deployment. Reads the completed, already locked
exports and the native Scene pose conversion, without map or obstacle rebuilding.
All scenes remain in the output; unavailable future timestamps are explicit masks.
"""
import argparse
import csv
import inspect
from collections import defaultdict
from pathlib import Path
import pickle
import sys
import time
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.decision_local.common import atomic_json, check_host, digest, read, source_sha
from tools.local_interaction_mask_v2.score_async import initialize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', type=Path, nargs='+', required=True)
    p.add_argument('--features', type=Path, required=True, help='Read only the immutable population identity, not feature tensors')
    p.add_argument('--log-root', type=Path, required=True)
    p.add_argument('--devkit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--verify-ego-root', type=Path)
    p.add_argument('--limit', type=int, default=0)
    a = p.parse_args(); resource = check_host(); started = time.monotonic()
    initialize(str(a.devkit))
    from navsim.common.dataclasses import Scene
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    metas = [read(path/'identity.json') for path in a.predictions]
    feature_meta = read(a.features/'identity.json')
    index = feature_meta['index']
    if any(meta['features'] != feature_meta['identity'] for meta in metas):
        raise ValueError('Different prediction populations')
    if any(read(path/'COMPLETE.json')['identity'] != meta['identity']
           for path, meta in zip(a.predictions, metas)):
        raise ValueError('Use completed immutable predictions')
    expected_tokens = {row['token'] for row in index}
    if any({f.stem for f in (bank/'records').glob('*.json')} != expected_tokens for bank in a.predictions):
        raise ValueError('Incomplete or foreign exported scene membership')
    selected = index[:a.limit or None]
    grouped = defaultdict(list)
    for row in selected:
        grouped[row['log']].append(row)
    result = [[] for _ in a.predictions]; hashes = {}; parity = 0.; parity_count = 0
    if a.verify_ego_root:
        import torch
        from starVLA.dataloader.foresight_dataset import decode_ego
        target_identity = read(a.verify_ego_root/'ego_identity.json')['identity']
    for log, rows in grouped.items():
        path = a.log_root/(log+'.pkl'); hashes[log] = digest(path)
        with path.open('rb') as stream:
            frames = pickle.load(stream)
        lookup = {frame['token']: i for i, frame in enumerate(frames)}
        for row in rows:
            token = row['token']; index = lookup[token]; current = frames[index]
            future, valid = [current], []
            for offset in range(1, 9):
                frame = frames[index+offset] if index+offset < len(frames) else None
                ok = frame is not None and abs(frame['timestamp']-current['timestamp']-offset*500000) <= 60000
                valid.append(ok); future.append(frame if ok else current)
            valid = np.asarray(valid, dtype=bool)
            scene = Scene(scene_metadata=SimpleNamespace(num_future_frames=8, num_history_frames=1),
                map_api=None, frames=[SimpleNamespace(ego_status=Scene._build_ego_status(f)) for f in future])
            gt = scene.get_future_trajectory().poses
            if a.verify_ego_root:
                target = torch.load(a.verify_ego_root/'ego'/(token+'.pt'), map_location='cpu', weights_only=True)
                if target['identity'] != target_identity or target['token'] != token:
                    raise ValueError('Wrong reference ego label')
                expected = decode_ego(target['ego']).numpy()
                np.testing.assert_allclose(gt, expected, rtol=1e-5, atol=1e-5)
                parity = max(parity, float(np.abs(gt-expected).max())); parity_count += 1
            shared_q0 = None
            for bank, meta, rows_out in zip(a.predictions, metas, result):
                file = bank/'predictions'/(token+'.npz'); receipt = read(bank/'records'/(token+'.json'))
                if receipt['identity'] != meta['identity'] or digest(file) != receipt['sha256']:
                    raise ValueError('Changed immutable trajectory')
                with np.load(file) as values:
                    if shared_q0 is None: shared_q0 = values['q0'].copy()
                    elif not np.array_equal(shared_q0, values['q0']): raise ValueError('Different S0 proposal')
                    record = {'token': token, 'log': log, 'valid_future_points': int(valid.sum()),
                              'complete_future': bool(valid.all()), 'endpoint_valid': bool(valid[-1])}
                    for name in ('q0', 'q_final'):
                        trajectory = values[name]
                        if trajectory.shape != (8, 3) or not np.isfinite(trajectory).all():
                            raise ValueError('Invalid prediction; no scene silently dropped')
                        distance = np.linalg.norm(trajectory[:, :2]-gt[:, :2], axis=-1)
                        angle = np.abs(np.arctan2(np.sin(trajectory[:, 2]-gt[:, 2]), np.cos(trajectory[:, 2]-gt[:, 2])))
                        record[name+'_ADE_m'] = float(distance[valid].mean()) if valid.any() else None
                        record[name+'_FDE_m'] = float(distance[-1]) if valid[-1] else None
                        record[name+'_heading_MAE_rad'] = float(angle[valid].mean()) if valid.any() else None
                    rows_out.append(record)
    summaries = []
    for bank, meta, rows in zip(a.predictions, metas, result):
        target = a.output/(bank.name+'.csv')
        with target.open('w') as stream:
            writer = csv.DictWriter(stream, list(rows[0])); writer.writeheader(); writer.writerows(rows)
        numeric = {}
        for key in rows[0]:
            if key.endswith(('_m', '_rad')):
                numbers = [r[key] for r in rows if r[key] is not None]
                numeric[key] = {'mean': float(np.mean(numbers)) if numbers else None, 'supported_scenes': len(numbers)}
        summaries.append({'predictions': str(bank), 'identity': meta['identity'], 'scenes': len(rows),
            'complete_future_scenes': sum(r['complete_future'] for r in rows), 'metrics': numeric,
            'csv_sha256': digest(target)})
    atomic_json(a.output/'COMPLETE.json', {'source_sha': source_sha(ROOT), 'tool_sha256': digest(__file__),
        'resources': resource, 'seconds': time.monotonic()-started, 'runs': summaries,
        'log_sha256': hashes, 'native_conversion': 'Scene._build_ego_status + Scene.get_future_trajectory',
        'native_scene_source_sha256': digest(inspect.getsourcefile(Scene)),
        'existing_ego_labels_checked': parity_count, 'native_versus_training_GT_max_abs': parity,
        'limitations': ['Descriptive open-loop errors; do not replace canonical PDMS or alter exported trajectories',
                       'Future masks only control this post-execution error statistic; every scene remains in CSV',
                       'No test map, future obstacle or additional model-input cache constructed']})


if __name__ == '__main__':
    main()
