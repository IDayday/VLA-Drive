"""Rebuild current inputs and separate full supervision directly from licensed raw NAVSIM logs."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import pickle
import subprocess
import sys
import numpy as np
from pyquaternion import Quaternion
import torch
from starVLA.model.modules.structured_world.geometry import geometric_fov
from starVLA.model.modules.structured_world.targets import make_targets
from tools.local_interaction_mask_v2.train_foundation import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('index', 'raw-log-root', 'sensor-root', 'output'): parser.add_argument('--' + key, required=True)
    args = parser.parse_args(); output = Path(args.output)
    # This existing entry point reads ONLY <=t0 ego/current cameras. It is also
    # independently usable at deployment where future logs/annotations are absent.
    subprocess.run([sys.executable, '-m', 'tools.local_interaction_mask_v2.prepare_current_records',
        '--index', args.index, '--raw-log-root', args.raw_log_root, '--sensor-root', args.sensor_root,
        '--output', args.output], check=True)
    target_dir = output / 'labels' / 'targets'; target_dir.mkdir(parents=True)
    ego_dir = output / 'labels' / 'ego_meta'; ego_dir.mkdir()
    index = json.loads(Path(args.index).read_text())
    if isinstance(index, dict): index = index['records']
    tokens = [row['token'] for row in index]
    if len(set(tokens)) != len(tokens): raise ValueError('Duplicate training tokens')
    groups = defaultdict(list)
    for row in index: groups[row['log']].append(row['token'])
    records, failures = [], []
    for log, selected in groups.items():
        with (Path(args.raw_log_root) / (log + '.pkl')).open('rb') as handle: frames = pickle.load(handle)
        positions = {frame['token']: i for i, frame in enumerate(frames)}
        for token in selected:
            try:
                i = positions[token]; current = frames[i]
                if i < 3 or i + 8 >= len(frames): raise ValueError('Incomplete four-state history/eight-step ego labels')
                if not np.allclose(current['lidar2ego'], np.eye(4), atol=1e-6):
                    raise ValueError('Nonidentity current annotation frame requires explicit adaptation')
                future = frames[i+1:i+9]
                if any(abs((frame['timestamp']-current['timestamp'])/1e6 - .5*(j+1)) > .05 for j, frame in enumerate(future)):
                    raise ValueError('Ego future labels do not follow original .5s action timestamps')
                with np.load(output / 'observations' / (token + '.npz'), allow_pickle=False) as observation:
                    ks, es, ds = [observation[key] for key in ('intrinsics', 'extrinsics', 'distortion')]
                annotations = current.get('anns')
                boxes = np.asarray(annotations['gt_boxes']) if annotations is not None else np.empty((0, 7))
                supported = geometric_fov(boxes[:, :3], ks, es, ds)
                # Store every current ROI/support target. Model slot limits are
                # applied in supervision matching, never by truncating this cache.
                target = make_targets(current, future, capacity=max(1, len(boxes)), current_eligibility=supported)
                xx, yy = np.meshgrid(np.arange(1.5, 50.), np.arange(-19.5, 20.), indexing='ij')
                grid = np.zeros(xx.shape, dtype=bool)
                for height in (0., 1., 2.):
                    points = np.stack([xx, yy, np.full_like(xx, height)], -1).reshape(-1, 3)
                    grid |= geometric_fov(points, ks, es, ds).reshape(xx.shape)
                target.supervision_grid = torch.from_numpy(grid)
                torch.save(asdict(target), target_dir / (token + '.pt'))
                # Compatibility metadata lives under LABELS. Online prediction
                # uses current_records, never this future-containing pickle.
                sequence = frames[i-3:i+9]
                poses = np.asarray([[float(frame['ego2global_translation'][0]), float(frame['ego2global_translation'][1]),
                    float(Quaternion(frame['ego2global_rotation']).yaw_pitch_roll[0])] for frame in sequence], dtype=np.float64)
                status = {'global_poses': poses,
                    'commands': np.asarray([frame['driving_command'] for frame in sequence[:4]]),
                    'velocities': np.asarray([np.asarray(frame['ego_dynamic_state'])[:2] for frame in sequence[:4]])}
                with (ego_dir / (token + '.pkl')).open('wb') as handle: pickle.dump({'glo_status': status}, handle)
                records.append({'token': token, 'log': log, 'targets': len(target.track_ids), 'overflow': target.overflow,
                    'valid_future_points': int(target.future_valid_mask.sum()), 'annotation_valid': bool(target.annotation_valid_mask)})
            except Exception as error: failures.append({'token': token, 'log': log, 'error': repr(error)})
        print(json.dumps({'logs_done': len({row['log'] for row in records}), 'labels_done': len(records), 'failed': len(failures)}), flush=True)
    atomic_json(output / 'tokens.json', tokens)
    atomic_json(output / 'labels' / 'audit.json', {'requested': len(tokens), 'completed': len(records), 'records': records,
        'failures': failures, 'roi': [1, -20, 50, 20], 'future_steps': 8, 'interval_s': .5,
        'coordinates': 'ego_t0 full SE3 for neighbors; original ego action normalization applied at training load',
        'support_limitation': 'calibrated FOV, not occlusion or actual visibility', 'full_current_population_cached': True})
    atomic_json(output / 'current_dataset.json', {'tokens': str(output / 'tokens.json'),
        'observations': str(output / 'observations'), 'current_records': str(output / 'current_records')})
    atomic_json(output / 'training_dataset.json', {'tokens': str(output / 'tokens.json'),
        'observations': str(output / 'observations'), 'meta_root': str(ego_dir), 'targets': str(output / 'labels')})
    if failures: raise RuntimeError('Incomplete training labels; failures retained, never silently filtered')


if __name__ == '__main__': main()
