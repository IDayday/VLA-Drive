from __future__ import annotations

import numpy as np

from research.action_effect.trajectory_io import absolute_poses_to_current_ego


def test_absolute_to_current_ego_preserves_navsim_rear_axle_frame() -> None:
    poses = np.zeros((12, 3), dtype=np.float64)
    poses[:, 0] = 10.0
    poses[:, 1] = np.arange(12, dtype=np.float64)
    poses[:, 2] = np.pi / 2.0

    relative = absolute_poses_to_current_ego(poses)

    np.testing.assert_allclose(relative[:, 0], np.arange(1, 9), atol=1e-8)
    np.testing.assert_allclose(relative[:, 1], 0.0, atol=1e-8)
    np.testing.assert_allclose(relative[:, 2], 0.0, atol=1e-8)


def test_heading_wrap_is_continuous_across_pi() -> None:
    poses = np.zeros((12, 3), dtype=np.float64)
    poses[:, 0] = np.arange(12, dtype=np.float64)
    poses[:, 2] = np.pi - 0.01
    poses[4:, 2] = -np.pi + 0.01
    relative = absolute_poses_to_current_ego(poses)
    np.testing.assert_allclose(relative[:, 2], 0.02, atol=1e-8)
