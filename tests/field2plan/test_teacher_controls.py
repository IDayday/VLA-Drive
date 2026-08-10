import inspect

import pytest
import torch

from starVLA.model.modules.field2plan.controls import (
    GTMLPFieldControl,
    apply_dynamics_teacher_controls,
    apply_geometry_teacher_controls,
)


def test_random_teacher_is_frozen_deterministic_and_scene_independent() -> None:
    depth = torch.zeros(2, 3, 4, 5)
    confidence = torch.zeros_like(depth)
    tokens = ("token-a", "token-b")

    first = apply_geometry_teacher_controls(
        depth,
        confidence,
        tokens,
        seed=17,
        random_teacher=True,
        shuffle_teacher_across_batch=False,
    )
    second = apply_geometry_teacher_controls(
        depth + 99.0,
        confidence + 1.0,
        tokens,
        seed=17,
        random_teacher=True,
        shuffle_teacher_across_batch=False,
    )

    torch.testing.assert_close(first.depth_m, second.depth_m, rtol=0, atol=0)
    torch.testing.assert_close(first.confidence, torch.ones_like(confidence))
    assert not torch.equal(first.depth_m[0], first.depth_m[1])
    assert first.mode == "random"


def test_shuffled_teacher_is_deterministic_and_cross_sample() -> None:
    depth = torch.arange(3.0).reshape(3, 1, 1, 1).expand(3, 1, 2, 2)
    confidence = torch.ones_like(depth)

    output = apply_geometry_teacher_controls(
        depth,
        confidence,
        ("a", "b", "c"),
        seed=9,
        random_teacher=False,
        shuffle_teacher_across_batch=True,
    )

    assert output.permutation.shape == (3,)
    assert not torch.equal(output.permutation, torch.arange(3))
    torch.testing.assert_close(output.depth_m, depth[output.permutation])
    assert output.mode == "shuffled"


def test_teacher_controls_reject_ambiguous_or_singleton_shuffle() -> None:
    depth = torch.ones(1, 1, 2, 2)
    with pytest.raises(ValueError, match="mutually exclusive"):
        apply_geometry_teacher_controls(
            depth, depth, ("x",), random_teacher=True,
            shuffle_teacher_across_batch=True,
        )
    with pytest.raises(ValueError, match="batch size"):
        apply_geometry_teacher_controls(
            depth, depth, ("x",), shuffle_teacher_across_batch=True
        )


def test_gt_mlp_control_uses_current_state_only_and_has_expected_shape() -> None:
    module = GTMLPFieldControl(
        state_dim=4,
        output_channels=8,
        field_size=(4, 5),
        hidden_dim=16,
    )
    state = torch.randn(2, 1, 4, requires_grad=True)

    field = module(state)

    assert field.shape == (2, 8, 4, 5)
    assert "future_action" not in inspect.signature(module.forward).parameters
    field.square().mean().backward()
    assert state.grad is not None


def test_temporal_shuffled_dynamics_teacher_is_deterministic_and_misaligned() -> None:
    features = torch.arange(8.0).reshape(1, 8, 1, 1, 1, 1).expand(
        2, 8, 1, 2, 1, 1
    )
    confidence = torch.ones(2, 8, 1, 1, 1)
    first = apply_dynamics_teacher_controls(
        features,
        confidence,
        ("a", "b"),
        seed=31,
        temporal_shuffle=True,
    )
    second = apply_dynamics_teacher_controls(
        features,
        confidence,
        ("a", "b"),
        seed=31,
        temporal_shuffle=True,
    )

    assert first.mode == "temporal_shuffled"
    assert not torch.equal(first.temporal_permutation, torch.arange(8))
    torch.testing.assert_close(first.features, second.features, rtol=0, atol=0)
    torch.testing.assert_close(
        first.features, features[:, first.temporal_permutation], rtol=0, atol=0
    )
