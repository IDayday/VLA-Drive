import inspect

import torch

from starVLA.model.modules.grounded_world.dynamics_memory import (
    CurrentDynamicsEncoder,
    PredictiveMemoryForecaster,
)
from starVLA.model.modules.grounded_world.geometry_memory import (
    MultiScaleGeometryMemoryWriter,
)


def test_multiscale_geometry_memory_shapes_and_gradient() -> None:
    writer = MultiScaleGeometryMemoryWriter(
        input_channels=16,
        output_channels=(16, 24, 32),
        scale_factors=(1, 2, 4),
    )
    field = torch.randn(2, 16, 32, 32, requires_grad=True)
    memory = writer(field).validate()

    assert [tuple(level.shape) for level in memory.levels] == [
        (2, 16, 32, 32),
        (2, 24, 16, 16),
        (2, 32, 8, 8),
    ]
    assert memory.scale_factors == (1, 2, 4)
    sum(level.mean() for level in memory.levels).backward()
    assert field.grad is not None
    assert torch.isfinite(field.grad).all()


def test_current_prior_and_future_prediction_are_separate_and_action_free() -> None:
    current_encoder = CurrentDynamicsEncoder(
        geometry_channels=16,
        output_channels=12,
        history_length=4,
        hidden_channels=24,
    )
    forecaster = PredictiveMemoryForecaster(
        geometry_channels=16,
        dynamics_channels=12,
        horizon=8,
        hidden_channels=24,
    )
    geometry = torch.randn(2, 16, 16, 16, requires_grad=True)
    transforms = torch.eye(4).reshape(1, 1, 4, 4).repeat(2, 4, 1, 1)
    transforms[:, :, 0, 3] = torch.arange(4, dtype=torch.float32)
    valid = torch.ones(2, 4, dtype=torch.bool)

    current = current_encoder(geometry, transforms, valid).validate()
    predictive = forecaster(geometry, current.field).validate()

    assert current.field.shape == (2, 12, 16, 16)
    assert predictive.current.field.data_ptr() == current.field.data_ptr()
    assert predictive.future.shape == (2, 8, 12, 16, 16)
    assert predictive.log_variance.shape == (2, 8, 1, 16, 16)
    prohibited = {"action", "actions", "draft", "trajectory", "future_action"}
    for module in (current_encoder, forecaster):
        names = set(inspect.signature(module.forward).parameters)
        assert not names.intersection(prohibited)

    predictive.future.mean().backward()
    assert geometry.grad is not None
    assert torch.isfinite(geometry.grad).all()

