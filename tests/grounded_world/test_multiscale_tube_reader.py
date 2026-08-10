import torch

from starVLA.model.modules.grounded_world.geometry_memory import (
    MultiScaleGeometryMemoryWriter,
)
from starVLA.model.modules.grounded_world.trajectory_tube_reader import (
    MultiScaleTrajectoryTubeReader,
)


def test_reader_uses_all_geometry_levels_and_future_slice() -> None:
    writer = MultiScaleGeometryMemoryWriter(8, (8, 12, 16), (1, 2, 4))
    memory = writer(torch.randn(2, 8, 16, 16))
    reader = MultiScaleTrajectoryTubeReader(
        geometry_channels=(8, 12, 16),
        dynamics_channels=10,
        output_dim=24,
        x_range_m=(-8.0, 56.0),
        y_range_m=(-32.0, 32.0),
    )
    trajectory = torch.zeros(2, 1, 8, 3)
    trajectory[..., 0] = torch.linspace(1.0, 20.0, 8)
    future = torch.randn(2, 8, 10, 16, 16)
    output = reader(memory, trajectory, future_dynamics=future)
    assert output.waypoint_context.shape == (2, 1, 8, 24)
    assert output.source_gates.shape == (2, 1, 8, 4)
    assert output.tube_valid_mask.ndim == 4
    output.waypoint_context.sum().backward()
    assert all(projection.weight.grad is not None for projection in reader.geometry_projections)
    assert reader.future_projection.weight.grad is not None


def test_disable_access_is_exact_zero_equal_capacity_path() -> None:
    writer = MultiScaleGeometryMemoryWriter(8, (8, 12, 16), (1, 2, 4))
    memory = writer(torch.randn(1, 8, 16, 16))
    reader = MultiScaleTrajectoryTubeReader(
        geometry_channels=(8, 12, 16),
        dynamics_channels=10,
        output_dim=24,
        x_range_m=(-8.0, 56.0),
        y_range_m=(-32.0, 32.0),
    )
    trajectory = torch.zeros(1, 2, 8, 3)
    future = torch.randn(1, 8, 10, 16, 16)
    output = reader(memory, trajectory, future_dynamics=future, disable_access=True)
    assert torch.count_nonzero(output.waypoint_context) == 0
    assert torch.count_nonzero(output.source_gates) == 0
