from __future__ import annotations

import numpy as np
import torch

from research.action_effect.gate2_5 import (
    calibrate_risk_threshold,
    trajectory_summary_target,
)
from research.action_effect.world_probe import ActionEffectWorldProbe


def test_trajectory_summary_uses_action_geometry() -> None:
    straight = np.stack(
        (np.arange(1, 9, dtype=np.float32), np.zeros(8), np.zeros(8)), axis=1
    )
    lateral = straight.copy()
    lateral[:, 1] = np.linspace(0.0, 1.0, 8)
    lateral[:, 2] = np.linspace(0.0, 0.2, 8)
    target = trajectory_summary_target(np.stack((straight, lateral)), interval_s=0.5)
    assert target.shape == (2, 5)
    assert target[0, 1] == 0.0
    assert target[1, 1] == 1.0
    assert target[1, 3] > target[0, 3]
    assert target[1, 4] > target[0, 4]


def test_calibrated_threshold_separates_known_risk() -> None:
    labels = np.asarray([False, False, True, True])
    scores = np.asarray([0.1, 0.2, 0.8, 0.9])
    threshold, metrics = calibrate_risk_threshold(labels, scores)
    assert 0.2 < threshold <= 0.8
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["false_safe_rate"] == 0.0


def test_action_path_has_nonzero_gradient_and_jacobian() -> None:
    torch.manual_seed(7)
    model = ActionEffectWorldProbe(
        scene_input_dim=12,
        consequence_dim=5,
        latent_dim=16,
        trajectory_input_dim=4,
        trajectory_token_dim=8,
        dropout=0.0,
    )
    scene = torch.randn(4, 8, 12)
    trajectory = torch.randn(4, 8, 4, requires_grad=True)
    output = model(scene, trajectory)["consequence_prediction"]
    assert isinstance(output, torch.Tensor)
    gradient = torch.autograd.grad(output.square().sum(), trajectory)[0]
    assert float(gradient.norm()) > 0.0
    encoded = model.encode_trajectory(trajectory.detach())
    assert float(encoded.var(dim=0, unbiased=False).mean()) > 0.0
