import torch

from starVLA.model.modules.field2plan.trajectory_tube_reader import (
    TrajectoryTubeReader,
)


def test_reader_supports_multiple_candidates_masks_boundaries_and_has_gradient() -> None:
    field = torch.ones(2, 4, 8, 8, requires_grad=True)
    draft = torch.zeros(2, 3, 8, 3)
    draft[..., 0] = 2.0
    draft[:, 1, :, 0] = 100.0
    reader = TrajectoryTubeReader(
        geometry_channels=4,
        output_dim=6,
        x_range_m=(0.0, 8.0),
        y_range_m=(-4.0, 4.0),
        lateral_offsets_m=(0.0,),
        longitudinal_offsets_m=(0.0,),
    )

    output = reader(field, draft)

    assert output.waypoint_context.shape == (2, 3, 8, 6)
    assert output.valid_mask.shape == (2, 3, 8, 1)
    assert output.source_gates.shape[:3] == (2, 3, 8)
    assert not output.valid_mask[:, 1].any()
    output.waypoint_context[:, 0].sum().backward()
    assert field.grad is not None
    assert torch.isfinite(field.grad).all()


def test_reader_accepts_m_equals_one_short_form_and_disable_access() -> None:
    reader = TrajectoryTubeReader(4, 4, (0.0, 8.0), (-4.0, 4.0))
    field = torch.randn(1, 4, 8, 8)
    draft = torch.zeros(1, 8, 3)
    enabled = reader(field, draft)
    disabled = reader(field, draft, disable_access=True)
    assert enabled.waypoint_context.shape == (1, 1, 8, 4)
    assert torch.count_nonzero(disabled.waypoint_context) == 0
    assert torch.count_nonzero(disabled.source_gates) == 0


def test_reader_samples_time_aligned_dynamics_and_preserves_legacy_without_it() -> None:
    field = torch.zeros(1, 4, 8, 8)
    draft = torch.zeros(1, 2, 8, 3)
    draft[..., 0] = 2.0
    legacy = TrajectoryTubeReader(
        4, 5, (0.0, 8.0), (-4.0, 4.0),
        lateral_offsets_m=(0.0,), longitudinal_offsets_m=(0.0,),
    )
    dynamic = TrajectoryTubeReader(
        4, 5, (0.0, 8.0), (-4.0, 4.0),
        lateral_offsets_m=(0.0,), longitudinal_offsets_m=(0.0,),
        dynamics_channels=3,
    )
    dynamic.geometry_projection.load_state_dict(legacy.geometry_projection.state_dict())
    no_dynamic = legacy(field, draft)

    dynamics = torch.zeros(1, 8, 3, 8, 8)
    dynamics[:, :, 0] = torch.arange(8).reshape(1, 8, 1, 1)
    with_dynamic = dynamic(
        field,
        draft,
        dynamics_field=dynamics,
        dynamics_times_s=torch.arange(1, 9, dtype=torch.float32) * 0.5,
        waypoint_times_s=torch.arange(1, 9, dtype=torch.float32) * 0.5,
    )

    assert no_dynamic.waypoint_context.shape == (1, 2, 8, 5)
    assert with_dynamic.waypoint_context.shape == (1, 2, 8, 5)
    assert with_dynamic.source_gates.shape == (1, 2, 8, 2)
    assert not torch.equal(with_dynamic.waypoint_context[:, :, 0], with_dynamic.waypoint_context[:, :, -1])
