import numpy as np
import torch

from infer import deal_action_1226
from starVLA.model.modules.action_model.multi_trajectory.trajectory_codec import (
    normalized_deltas_to_poses,
    poses_to_normalized_deltas,
)


def test_torch_codec_matches_existing_1226_decoder():
    generator = torch.Generator().manual_seed(20260820)
    actions = (torch.rand(2, 5, 8, 3, generator=generator) * 2.0 - 1.0).double()
    actual = normalized_deltas_to_poses(actions)
    expected = np.stack(
        [deal_action_1226(sample.numpy()) for sample in actions], axis=0
    )
    torch.testing.assert_close(
        actual.double(), torch.from_numpy(expected), rtol=1e-6, atol=1e-6
    )


def test_trajectory_codec_round_trip_across_heading_wrap():
    poses = torch.tensor(
        [
            [1.0, 0.0, 3.05],
            [0.1, 0.1, -3.02],
            [-0.7, -0.2, -2.75],
            [-1.1, -0.8, 2.95],
        ],
        dtype=torch.float64,
    )
    encoded = poses_to_normalized_deltas(poses)
    decoded = normalized_deltas_to_poses(encoded)
    torch.testing.assert_close(decoded[..., :2], poses[..., :2])
    torch.testing.assert_close(
        torch.cos(decoded[..., 2]), torch.cos(poses[..., 2])
    )
    torch.testing.assert_close(
        torch.sin(decoded[..., 2]), torch.sin(poses[..., 2])
    )


def test_codec_does_not_clip_static_vocabulary_poses():
    poses = torch.tensor([[[10.0, 4.0, 2.8], [20.0, -3.0, -2.9]]])
    encoded = poses_to_normalized_deltas(poses)
    assert (encoded.abs() > 1.0).any()
    decoded = normalized_deltas_to_poses(encoded)
    torch.testing.assert_close(decoded[..., :2], poses[..., :2], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        torch.sin(decoded[..., 2]), torch.sin(poses[..., 2]), rtol=1e-5, atol=1e-5
    )
