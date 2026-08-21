from __future__ import annotations

import numpy as np

from research.action_effect.probe_data import (
    deterministic_scene_split,
    encode_trajectories,
    trajectory_normalization,
)


def test_scene_split_is_deterministic_and_disjoint() -> None:
    first = deterministic_scene_split([f"s{i}" for i in range(10)], fraction=0.8, seed=7)
    second = deterministic_scene_split(list(reversed([f"s{i}" for i in range(10)])), fraction=0.8, seed=7)
    assert first == second
    assert len(first[0]) == 8 and len(first[1]) == 2
    assert set(first[0]).isdisjoint(first[1])


def test_trajectory_encoding_is_finite_and_heading_periodic() -> None:
    trajectory = np.zeros((2, 8, 3), dtype=np.float32)
    trajectory[0, :, 0] = np.arange(8)
    trajectory[1, :, 0] = np.arange(8) + 1
    trajectory[1, :, 2] = 2 * np.pi
    stats = trajectory_normalization(trajectory)
    encoded = encode_trajectories(trajectory, stats)
    assert encoded.shape == (2, 8, 4)
    assert np.isfinite(encoded).all()
    np.testing.assert_allclose(encoded[0, :, 2:], encoded[1, :, 2:], atol=1e-6)
