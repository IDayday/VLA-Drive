import inspect

import pytest
import torch

from starVLA.model.modules.grounded_world.core import GroundedWorldCore


def _inputs(batch: int = 2):
    return {
        "finest_geometry": torch.randn(batch, 16, 16, 16),
        "history_current_from_ego": torch.eye(4).repeat(batch, 4, 1, 1),
        "history_valid_mask": torch.ones(batch, 4, dtype=torch.bool),
    }


def test_prior_stage_builds_current_memory_without_future_or_trajectory() -> None:
    core = GroundedWorldCore(
        geometry_input_channels=16,
        geometry_channels=(16, 24, 32),
        scale_factors=(1, 2, 4),
        dynamics_channels=12,
        history_length=4,
        horizon=8,
        future_enabled=False,
    )
    output = core(stage="prior", **_inputs())
    assert output.geometry.levels[2].shape == (2, 32, 4, 4)
    assert output.current_dynamics.field.shape == (2, 12, 16, 16)
    assert output.predictive is None
    forbidden = {"action", "draft", "trajectory", "future_action"}
    assert forbidden.isdisjoint(inspect.signature(core.forward).parameters)


def test_predictive_stage_is_action_free_and_requires_future_module() -> None:
    core = GroundedWorldCore(
        geometry_input_channels=16,
        geometry_channels=(16, 24, 32),
        scale_factors=(1, 2, 4),
        dynamics_channels=12,
        history_length=4,
        horizon=8,
        future_enabled=True,
    )
    output = core(stage="predictive", **_inputs())
    assert output.predictive is not None
    assert output.predictive.future.shape == (2, 8, 12, 16, 16)
    with pytest.raises(ValueError, match="stage"):
        core(stage="planning", **_inputs())


def test_no_teacher_control_keeps_identical_future_architecture() -> None:
    real = GroundedWorldCore(
        16, (16, 24, 32), (1, 2, 4), 12, 4, 8, future_enabled=True
    )
    no_teacher = GroundedWorldCore(
        16, (16, 24, 32), (1, 2, 4), 12, 4, 8, future_enabled=True
    )
    assert list(real.state_dict()) == list(no_teacher.state_dict())
    assert sum(p.numel() for p in real.parameters()) == sum(
        p.numel() for p in no_teacher.parameters()
    )


def test_history_visual_geometry_is_an_explicit_student_input() -> None:
    core = GroundedWorldCore(
        16, (16, 24, 32), (1, 2, 4), 12, 4, 8, future_enabled=False
    )
    inputs = _inputs(batch=1)
    history = torch.randn(1, 4, 16, 16, 16, requires_grad=True)
    output = core(stage="prior", history_geometry=history, **inputs)
    assert output.current_dynamics.field.shape == (1, 12, 16, 16)
    output.current_dynamics.field.sum().backward()
    assert history.grad is not None
