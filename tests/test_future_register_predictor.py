from __future__ import annotations

import torch
from torch import nn

from navsim.agents.EpisodeDrive.layers.world_model import FutureRegisterPredictor


HORIZONS = (0.5, 1.5, 3.0)


def test_predictor_supports_k1_and_k64() -> None:
    torch.manual_seed(31)
    predictor = FutureRegisterPredictor(hidden_dim=256, predictor_layers=2)
    current = torch.randn(1, 16, 256)
    for candidate_count in (1, 64):
        trajectories = torch.randn(1, candidate_count, 8, 3)
        output = predictor(current, trajectories, HORIZONS)
        assert output.shape == (1, candidate_count, 3, 16, 256)


def test_zero_initialized_residual_and_layernorm_output() -> None:
    predictor = FutureRegisterPredictor(hidden_dim=32, predictor_layers=2, num_heads=4)
    current = torch.randn(2, 16, 32)
    trajectories = torch.randn(2, 3, 8, 3)
    output = predictor(current, trajectories, HORIZONS)
    expected = predictor.output_norm(current)[:, None, None].expand_as(output)
    torch.testing.assert_close(output, expected, rtol=0.0, atol=0.0)
    assert torch.count_nonzero(predictor.residual_output.weight).item() == 0
    assert torch.count_nonzero(predictor.residual_output.bias).item() == 0


def test_no_action_condition_zeros_complete_trajectory_encoding() -> None:
    torch.manual_seed(33)
    predictor = FutureRegisterPredictor(hidden_dim=32, predictor_layers=2, num_heads=4)
    nn.init.normal_(predictor.residual_output.weight, std=0.01)
    current = torch.randn(2, 16, 32)
    first_trajectory = torch.randn(2, 1, 8, 3)
    second_trajectory = torch.randn(2, 1, 8, 3) * 100.0

    first_no_action = predictor(
        current, first_trajectory, HORIZONS, use_action_condition=False
    )
    second_no_action = predictor(
        current, second_trajectory, HORIZONS, use_action_condition=False
    )
    torch.testing.assert_close(first_no_action, second_no_action)

    first_correct = predictor(current, first_trajectory, HORIZONS)
    second_correct = predictor(current, second_trajectory, HORIZONS)
    assert not torch.allclose(first_correct, second_correct)


def test_trajectory_features_are_xy_sincos_speed_acceleration() -> None:
    predictor = FutureRegisterPredictor(hidden_dim=32, predictor_layers=2, num_heads=4)
    trajectory = torch.zeros(1, 1, 8, 3)
    trajectory[0, 0, :, 0] = torch.arange(1, 9) * 0.5
    features = predictor.trajectory_point_features(trajectory)
    assert features.shape == (1, 1, 8, 6)
    torch.testing.assert_close(features[..., 0], trajectory[..., 0])
    torch.testing.assert_close(features[..., 1], trajectory[..., 1])
    torch.testing.assert_close(features[..., 2], torch.zeros_like(features[..., 2]))
    torch.testing.assert_close(features[..., 3], torch.ones_like(features[..., 3]))
    torch.testing.assert_close(features[..., 4], torch.ones_like(features[..., 4]))
    assert features[0, 0, 0, 5].item() == 2.0
    torch.testing.assert_close(features[0, 0, 1:, 5], torch.zeros(7))
