import copy
import json
import numpy as np
from PIL import Image
import pytest
import torch

from starVLA.dataloader.ddpolicy_current import current_example, CurrentCameraDataset
from starVLA.model.modules.vehicle_joint.initialization import identity_hash
from tools.ddpolicy_vehicle.checkpoints import scene_noise


def record(tmp_path):
    path = tmp_path/'current.png'
    Image.new('RGB', (48, 32), (10, 20, 30)).save(path)
    return dict(identity='test', token='scene', log='log', timestamp=123,
        image_paths=[str(path)]*3, global_pose_history=[[i, 0, 0] for i in range(4)],
        ego_speed=2., navigation=1,
        current_calibration=dict(intrinsics=np.eye(3)[None].repeat(3, 0).tolist(),
            extrinsics=np.eye(4)[None].repeat(3, 0).tolist(), distortion=np.zeros((3, 5)).tolist()))


def test_current_inputs_need_no_future_and_reject_label_fields(tmp_path):
    value = record(tmp_path)
    sample = current_example(value)
    assert sample['state'].shape == (1, 4) and all(x.size == (1024, 576) for x in sample['image'])
    for field in ('anns', 'future', 'vehicle_targets', 'qwen_feature_cache'):
        polluted = {**value, field: object()}
        with pytest.raises(ValueError, match='whitelist'): current_example(polluted)
    for field, invalid in (('navigation', -1), ('ego_speed', float('nan'))):
        with pytest.raises(ValueError): current_example({**value, field: invalid})


def test_current_index_integrity(tmp_path):
    index = [{'token':'scene', 'log':'log'}]
    (tmp_path/'identity.json').write_text(json.dumps(dict(schema='ddpolicy_current_cameras_v1', identity='test', index_sha256=identity_hash(index))))
    (tmp_path/'index.json').write_text(json.dumps([{'token':'another', 'log':'log'}]))
    with pytest.raises(ValueError, match='index changed'): CurrentCameraDataset(tmp_path)


def test_scene_noise_is_shard_independent_and_preserves_ego():
    a = scene_noise('scene', 42, 1, 'cpu')
    b = scene_noise('scene', 42, 9, 'cpu')
    assert torch.equal(a[:, 0], b[:, 0])
    assert torch.equal(b, scene_noise('scene', 42, 9, 'cpu'))
    assert not torch.equal(b, scene_noise('scene', 43, 9, 'cpu'))
