"""Reconstruct official current inputs from raw logs, never a driving-model cache.

Preserves the official train/val split except that the existing common development
logs cannot enter the optimizer. Validation trajectories remain the recorded GT.
"""
import argparse
from collections import Counter, defaultdict
import json
import pickle
from pathlib import Path
import sys

import numpy as np
import yaml

from .assets import atomic_json, check_official, digest, read


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ('official-source', 'official-revision', 'stage1', 'train-data', 'dev-data',
                 'optimized-labels', 'logs', 'sensors', 'output'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    source = check_official(a.official_source, a.official_revision)
    sys.path.insert(0, str(source))
    from navsim.common.dataclasses import AgentInput, SensorConfig
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import convert_absolute_to_relative_se2_array
    from nuplan.common.actor_state.state_representation import StateSE2
    from pyquaternion import Quaternion
    train = read(Path(a.train_data) / 'index.json')
    dev = read(Path(a.dev_data) / 'index.json')
    dev_logs = {r['log'] for r in dev}
    if dev_logs & {r['log'] for r in train}:
        raise ValueError('Shared training/development logs overlap')
    split_path = source / 'navsim/planning/script/config/training/default_train_val_test_log_split.yaml'
    splits = yaml.safe_load(split_path.read_text())
    train_logs, val_logs = set(splits['train_logs']), set(splits['val_logs'])
    if train_logs & val_logs:
        raise ValueError('Official train/val logs overlap')
    label_root = Path(a.optimized_labels)
    identity = read(label_root / 'identity.json')
    if digest(label_root / 'labels.npz') != identity['labels_sha256']:
        raise ValueError('Optimized label checksum mismatch')
    labels = np.load(label_root / 'labels.npz', allow_pickle=False)
    lookup = {token: i for i, token in enumerate(labels['tokens'].tolist())}
    if set(lookup) != {r['token'] for r in train}:
        raise ValueError('Optimized labels must match the same existing full training population')
    stage1 = Path(a.stage1)
    metadata = (stage1 / '.cache/huggingface/download/model.safetensors.metadata').read_text().splitlines()
    weight_hash = digest(stage1 / 'model.safetensors')
    if weight_hash != metadata[1]:
        raise ValueError('Stage1 file differs from its public download SHA256')
    groups = defaultdict(list)
    for row in train + dev:
        groups[row['log']].append(row)
    output = Path(a.output)
    if output.exists():
        raise FileExistsError('Manifest is immutable; choose a new output')
    rows, counts = [], Counter()
    for log, requested in sorted(groups.items()):
        with (Path(a.logs) / (log + '.pkl')).open('rb') as f:
            frames = pickle.load(f)
        indexes = {frame['token']: i for i, frame in enumerate(frames)}
        for row in requested:
            token = row['token']; index = indexes[token]
            history = frames[index - 3:index + 1]
            future = frames[index + 1:index + 9]
            if len(history) != 4 or len(future) != 8:
                raise ValueError('Incomplete official trajectory context ' + token)
            agent = AgentInput.from_scene_dict_list(history, Path(a.sensors), 4,
                                                    SensorConfig.build_no_sensors(), load_image_path=True)
            now = agent.ego_statuses[-1]
            pose_list = []
            for frame in frames[index:index + 9]:
                xy = frame['ego2global_translation']
                pose_list.append([xy[0], xy[1], Quaternion(frame['ego2global_rotation']).yaw_pitch_roll[0]])
            recorded = convert_absolute_to_relative_se2_array(StateSE2(*pose_list[0]), np.array(pose_list[1:]))
            split = 'reserved_dev' if log in dev_logs else 'train' if log in train_logs else 'val' if log in val_logs else None
            if split is None:
                raise ValueError('Requested scene is not in the official train/val logs')
            target = recorded.astype(np.float32)
            accepted = False
            if split == 'train':
                i = lookup[token]
                accepted = bool(labels['accepted'][i])
                if not accepted and not np.allclose(labels['trajectories'][i], target, atol=2e-5):
                    raise ValueError('Optimized fallback does not equal recorded physical trajectory')
                target = labels['trajectories'][i]
            image = Path(a.sensors) / frames[index]['cams']['CAM_F0']['data_path']
            if not image.is_file() or not np.isfinite(target).all():
                raise ValueError('Missing current image or illegal trajectory')
            rows.append(dict(token=token, log=log, split=split, image=str(image),
                history=[status.ego_pose.tolist() for status in agent.ego_statuses],
                command=np.asarray(now.driving_command).tolist(), velocity=now.ego_velocity.tolist(),
                acceleration=now.ego_acceleration.tolist(), trajectory=target.tolist(), optimized=accepted))
            counts[split] += 1
            counts[split + '_optimized' if accepted else split + '_recorded'] += 1
        if len(rows) % 2000 < len(requested):
            print('manifest scenes', len(rows), flush=True)
    identity = dict(schema='official_recogdrive_stage2_current_inputs_optimized_targets_v1',
        official_revision=a.official_revision, official_split_sha256=digest(split_path),
        stage1=dict(repo='owl10/ReCogDrive-VLM-2B', revision=metadata[0], sha256=weight_hash,
                    path=str(stage1), config_sha256=digest(stage1 / 'config.json')),
        train_index_sha256=digest(Path(a.train_data) / 'index.json'),
        dev_index_sha256=digest(Path(a.dev_data) / 'index.json'),
        optimized_labels=read(label_root / 'identity.json'),
        split_adjustment='Official split, excluding the 16 existing shared development logs from optimization; original GT for all validation',
        counts=dict(counts), retained_history=4, current_views=['cam_f0'], source_sensors=a.sensors,
        scene_count=len(rows), hidden_dtype='float32', hidden_definition='official final full sequence; padding preserved')
    atomic_json(output, dict(identity=identity, rows=rows))
    print(json.dumps(identity['counts']), flush=True)


if __name__ == '__main__':
    main()
