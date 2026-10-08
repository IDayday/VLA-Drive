"""Privileged native GT export for canonical evaluator self-check, never a model."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.local_interaction_mask_v2.score_async import initialize, atomic_json, digest


def main():
    parser = argparse.ArgumentParser()
    for name in ('index', 'log-root', 'devkit', 'output'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--scenes', type=int, default=4)
    args = parser.parse_args()
    initialize(str(args.devkit))
    from navsim.common.dataclasses import Scene
    rows = json.loads(args.index.read_text())[:args.scenes]
    contract = {'schema': 'privileged_native_GT_evaluator_selfcheck_not_model_result',
        'index': rows, 'source_sha256': digest(Path(__file__)), 'deployable': False}
    identity = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ('records', 'predictions'): (args.output/name).mkdir()
    atomic_json(args.output/'identity.json', {'identity': identity, **contract})
    atomic_json(args.output/'index.json', rows)
    for row in rows:
        with (args.log_root/(row['log']+'.pkl')).open('rb') as stream: frames = pickle.load(stream)
        index = next(i for i, f in enumerate(frames) if f['token'] == row['token'])
        selected = frames[index:index+9]
        if len(selected) != 9: raise ValueError('Missing native GT horizon')
        scene = Scene(scene_metadata=SimpleNamespace(num_future_frames=8, num_history_frames=1), map_api=None,
            frames=[SimpleNamespace(ego_status=Scene._build_ego_status(frame)) for frame in selected])
        trajectory = scene.get_future_trajectory().poses
        file = args.output/'predictions'/(row['token']+'.npz')
        with file.open('wb') as stream: np.savez_compressed(stream, q0=trajectory, q_final=trajectory)
        atomic_json(args.output/'records'/(row['token']+'.json'), {'token': row['token'], 'identity': identity, 'sha256': digest(file)})
    atomic_json(args.output/'COMPLETE.json', {'identity': identity, 'scenes': len(rows), 'is_model_result': False})


if __name__ == '__main__': main()
