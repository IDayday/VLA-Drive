import copy

import numpy as np
from PIL import Image
import pytest
import torch

from tools.full_foresight.prepare_from_navsim import current_and_ego, validate_partition


def raw_scene(tmp_path):
    image_path = tmp_path / 'camera.jpg'
    Image.new('RGB', (1600, 900)).save(image_path)
    camera = {'data_path': 'camera.jpg', 'cam_intrinsic': np.eye(3),
              'sensor2lidar_rotation': np.eye(3), 'sensor2lidar_translation': np.zeros(3),
              'distortion': np.zeros(5)}
    frames = []
    for t in range(12):
        frames.append({'token': str(t), 'timestamp': t * 500000,
                       'ego2global_translation': [t * .5, 0., 0.],
                       'ego2global_rotation': [1., 0., 0., 0.],
                       'driving_command': [0, 1, 0, 0],
                       'ego_dynamic_state': [1., 0., 0., 0.],
                       'cams': {c: copy.deepcopy(camera) for c in ('CAM_F0', 'CAM_L0', 'CAM_R0')}})
    return frames, {'sensor_root': str(tmp_path), 'fallback_sensor_root': []}


def test_raw_current_whitelist_and_original_ego_encoding(tmp_path):
    frames, config = raw_scene(tmp_path)
    current, target, calibration = current_and_ego(frames, 3, 'train-log', config, 'current', 'ego')
    assert set(current) == {'identity', 'token', 'log', 'timestamp', 'image_paths',
                            'global_pose_history', 'navigation', 'ego_speed', 'current_calibration'}
    assert target['ego'].shape == (8, 4)
    assert torch.isfinite(target['ego']).all()
    assert len(current['image_paths']) == 3 and len(current['global_pose_history']) == 4
    assert calibration['intrinsics'].shape == (3, 3, 3)
    assert current['timestamp'] == 1500000


def test_future_changes_only_ego_target_not_current_input(tmp_path):
    frames, config = raw_scene(tmp_path)
    current, target, _ = current_and_ego(frames, 3, 'log', config, 'current', 'ego')
    changed = copy.deepcopy(frames)
    for frame in changed[4:]:
        frame['ego2global_translation'][1] += 10.
        frame['cams'] = {}  # Future cameras are not a dependency of this entry.
    second_current, second_target, _ = current_and_ego(changed, 3, 'log', config, 'current', 'ego')
    assert current == second_current
    assert not torch.equal(target['ego'], second_target['ego'])


def test_missing_ego_horizon_and_invalid_current_rejected(tmp_path):
    frames, config = raw_scene(tmp_path)
    with pytest.raises(ValueError, match='four observed'):
        current_and_ego(frames, 2, 'log', config, 'current', 'ego')
    with pytest.raises(ValueError, match='eight future'):
        current_and_ego(frames[:-1], 3, 'log', config, 'current', 'ego')
    frames[3]['ego_dynamic_state'][0] = float('nan')
    with pytest.raises(ValueError, match='ego state'):
        current_and_ego(frames, 3, 'log', config, 'current', 'ego')


def test_fixed_population_preserved_and_log_leakage_rejected():
    p = {'train_tokens': ['train'], 'dev_tokens': ['dev'],
         'token_logs': {'train': 'log1', 'dev': 'log2'}}
    assert validate_partition(p) == {'train': [{'token': 'train', 'log': 'log1'}],
                                     'dev': [{'token': 'dev', 'log': 'log2'}]}
    p['token_logs']['dev'] = 'log1'
    with pytest.raises(ValueError, match='log leakage'):
        validate_partition(p)
    p['train_tokens'] = ['train', 'train']
    with pytest.raises(ValueError, match='duplicate'):
        validate_partition(p)
