from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch
from navsim.agents.EpisodeDrive.layers.world_model.logged_future_pose import build_logged_future_pose_metadata
from navsim.agents.EpisodeDrive.layers.world_model.task_future_training import validate_lite_targets, INPUT_SCHEMA


def test_logged_future_pose_global_to_current_rear_axle():
    frames=[NS(ego_status=NS(ego_pose=np.array([10.,20.+i,np.pi/2]))) for i in range(10)]
    scene=NS(frames=frames,scene_metadata=NS(num_history_frames=4,map_name='test'))
    target=build_logged_future_pose_metadata(scene)
    torch.testing.assert_close(target['logged_future_poses'][:,0],torch.tensor([1.,3.,6.]))
    assert target['logged_future_poses'][:,1:].abs().max()<1e-6
    assert target['task_future_input_schema']==INPUT_SCHEMA


def test_stale_future_pose_cache_explicitly_rejected():
    with pytest.raises(RuntimeError,match='Stale Lite input cache'):
        validate_lite_targets({},2)
