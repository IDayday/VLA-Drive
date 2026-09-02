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


def test_zero_initialized_residual_equals_affine_free_normalized_current() -> None:
    predictor = FutureRegisterPredictor(hidden_dim=32, predictor_layers=2, num_heads=4)
    current = torch.randn(2, 16, 32)
    trajectories = torch.randn(2, 3, 8, 3)
    output = predictor(current, trajectories, HORIZONS)
    expected = predictor.normalize_register_state(current)[:, None, None].expand_as(output)
    torch.testing.assert_close(output, expected, rtol=0.0, atol=0.0)
    predicted_delta = output - expected
    assert torch.count_nonzero(predicted_delta).item() == 0
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


def test_trajectory_features_are_scaled_xy_sincos_speed_acceleration() -> None:
    predictor = FutureRegisterPredictor(hidden_dim=32, predictor_layers=2, num_heads=4)
    trajectory = torch.zeros(1, 1, 8, 3)
    trajectory[0, 0, :, 0] = torch.arange(1, 9) * 0.5
    features = predictor.trajectory_point_features(
        trajectory, current_speed=torch.ones(1)
    )
    assert features.shape == (1, 1, 8, 6)
    torch.testing.assert_close(features[..., 0], trajectory[..., 0] / 30.0)
    torch.testing.assert_close(features[..., 1], trajectory[..., 1] / 10.0)
    torch.testing.assert_close(features[..., 2], torch.zeros_like(features[..., 2]))
    torch.testing.assert_close(features[..., 3], torch.ones_like(features[..., 3]))
    torch.testing.assert_close(
        features[..., 4], torch.ones_like(features[..., 4]) / 15.0
    )
    torch.testing.assert_close(features[..., 5], torch.zeros_like(features[..., 5]))


def test_constant_speed_trajectory_uses_current_ego_speed_for_first_acceleration() -> None:
    predictor = FutureRegisterPredictor(hidden_dim=32, predictor_layers=2, num_heads=4)
    trajectory = torch.zeros(2, 3, 8, 3)
    trajectory[..., 0] = torch.arange(1, 9) * 0.5
    features = predictor.trajectory_point_features(
        trajectory, current_speed=torch.ones(2)
    )
    torch.testing.assert_close(
        features[..., 5], torch.zeros_like(features[..., 5]), atol=1e-7, rtol=0
    )


def test_normalized_state_cosine_is_invariant_to_target_scale() -> None:
    predictor = FutureRegisterPredictor(hidden_dim=32, predictor_layers=2, num_heads=4)
    target = torch.randn(2, 3, 16, 32)
    normalized = predictor.normalize_register_state(target)
    scaled = predictor.normalize_register_state(target * 17.0)
    cosine = torch.nn.functional.cosine_similarity(normalized, scaled, dim=-1)
    torch.testing.assert_close(cosine, torch.ones_like(cosine), atol=1e-6, rtol=0)


def test_predictor_gradients_are_finite() -> None:
    torch.manual_seed(34)
    predictor = FutureRegisterPredictor(hidden_dim=32, predictor_layers=2, num_heads=4)
    current = torch.randn(2, 16, 32, requires_grad=True)
    trajectory = torch.randn(2, 1, 8, 3)
    output = predictor(
        current,
        trajectory,
        HORIZONS,
        current_ego_motion=torch.tensor([[3.0, 4.0], [0.0, 1.0]]),
    )
    output.square().mean().backward()
    gradients = [
        parameter.grad
        for parameter in predictor.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert current.grad is not None
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert torch.isfinite(current.grad).all()
