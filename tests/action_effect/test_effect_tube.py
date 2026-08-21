from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
import shapely
import torch

from research.action_effect.effect_tube import (
    EFFECT_TUBE_CHANNELS,
    EffectTubeConfig,
    build_effect_tube,
    signed_distance_from_mask,
)
from research.action_effect.losses import EffectTubeLoss


def test_signed_distance_has_declared_sign() -> None:
    config = EffectTubeConfig(resolution=8)
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    sdf = signed_distance_from_mask(mask, config)
    assert sdf[3, 3] > 0
    assert sdf[0, 0] < 0
    assert np.max(np.abs(sdf)) <= 1.0


def test_effect_tube_shape_and_swept_footprint() -> None:
    config = EffectTubeConfig(resolution=8)
    states = np.zeros((41, 11), dtype=np.float64)
    states[:, 0] = np.linspace(0.0, 8.0, 41)
    states[:, 3] = 2.0
    empty_tracks = [
        SimpleNamespace(tracked_objects=SimpleNamespace(tracked_objects=[])) for _ in range(40)
    ]
    drivable = shapely.box(-100, -100, 100, 100)
    lane = shapely.box(-100, -4, 100, 4)
    route = shapely.box(-100, -2, 100, 2)
    target = build_effect_tube(
        simulated_states=states,
        future_tracks=empty_tracks,
        static_unions=(drivable, lane, route),
        vehicle_parameters=get_pacifica_parameters(),
        config=config,
    )
    assert target.shape == (3, len(EFFECT_TUBE_CHANNELS), 8, 8)
    assert np.all(target[:, 0] == 0)
    assert np.all(target[:, 6] == 1)
    assert np.any(target[:, 8] > 0)


def test_effect_tube_loss_has_per_sample_reduction() -> None:
    loss = EffectTubeLoss(torch.tensor([2.0, 3.0, 1.0]))
    prediction = torch.zeros(2, 3, 9, 8, 8)
    target = torch.zeros_like(prediction)
    target[:, :, 6] = 1.0
    target[:, :, 8, 3:5, 3:5] = 1.0
    per_sample = loss(prediction, target, reduction="none")
    assert set(per_sample) == {
        "total",
        "occupancy",
        "sdf",
        "velocity",
        "clearance",
        "collision",
        "footprint",
    }
    assert per_sample["total"].shape == (2,)
    assert loss(prediction, target)["total"].ndim == 0
