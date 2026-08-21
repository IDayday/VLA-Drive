from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from shapely.geometry import box

from research.action_effect.structured_future import (
    FutureTubeConfig,
    candidate_aligned_grid,
    rasterize_future_slice,
)


def _object(x: float, y: float, vx: float = 0.0, vy: float = 0.0):
    return SimpleNamespace(
        center=SimpleNamespace(x=x, y=y),
        velocity=SimpleNamespace(x=vx, y=vy),
        box=SimpleNamespace(geometry=box(x - 1, y - 0.5, x + 1, y + 0.5)),
    )


def test_candidate_aligned_grid_rotates_with_heading() -> None:
    config = FutureTubeConfig(resolution=8)
    zero, x, y = candidate_aligned_grid(np.asarray([10, 20, 0, 0, 0]), config)
    rotated, _, _ = candidate_aligned_grid(np.asarray([10, 20, np.pi / 2, 0, 0]), config)
    np.testing.assert_allclose(zero[..., 0], 10 + x)
    np.testing.assert_allclose(zero[..., 1], 20 + y)
    np.testing.assert_allclose(rotated[..., 0], 10 - y, atol=1e-6)
    np.testing.assert_allclose(rotated[..., 1], 20 + x, atol=1e-6)


def test_future_slice_has_separate_static_dynamic_and_velocity_channels() -> None:
    config = FutureTubeConfig(
        resolution=16,
        longitudinal_min_m=-4,
        longitudinal_max_m=12,
        lateral_min_m=-8,
        lateral_max_m=8,
    )
    static = (box(-5, -10, 20, 10), box(-5, -2, 20, 2), box(0, -2, 20, 2))
    result = rasterize_future_slice(
        state=np.asarray([0, 0, 0, 2, 0], dtype=float),
        tracked_objects=[_object(5, 0, 6, 1)],
        static_unions=static,
        config=config,
    )
    assert result.shape == (7, 16, 16)
    assert result[0].mean() > result[1].mean() > 0
    assert result[3].sum() > 0
    assert result[4][result[3].astype(bool)].mean() > 0
    assert result[5][result[3].astype(bool)].mean() > 0
    assert np.all((0 <= result[6]) & (result[6] <= 1))
