import torch

from starVLA.model.modules.field2plan.geometry_supervision import (
    GeometrySupervisionHead,
    GeometryTargets,
    build_geometry_targets,
    geometry_supervision_losses,
)
from starVLA.model.modules.field2plan.types import CameraBatch


def _forward_camera(batch=1) -> CameraBatch:
    # camera_x=-ego_y, camera_y=-ego_z, camera_z=ego_x
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]
    )
    transform = torch.eye(4)
    transform[:3, :3] = rotation
    intrinsics = torch.tensor(
        [[4.0, 0.0, 7.5], [0.0, 4.0, 7.5], [0.0, 0.0, 1.0]]
    )
    return CameraBatch(
        intrinsics=intrinsics.reshape(1, 1, 3, 3).repeat(batch, 1, 1, 1),
        ego_to_camera=transform.reshape(1, 1, 4, 4).repeat(batch, 1, 1, 1),
        image_hw=torch.tensor([[[16.0, 16.0]]]).repeat(batch, 1, 1),
        view_names=("cam_f0",),
        frame_index=3,
    )


def test_metric_depth_targets_have_expected_shape_and_physical_ordering() -> None:
    depth = torch.full((1, 1, 16, 16), 4.5)
    confidence = torch.ones_like(depth)
    targets = build_geometry_targets(
        depth_m=depth,
        confidence=confidence,
        camera=_forward_camera(),
        field_size=(4, 3),
        x_range_m=(2.0, 6.0),
        y_range_m=(-1.5, 1.5),
        height_anchors_m=(0.0,),
        occupancy_threshold_m=0.4,
        free_space_margin_m=0.4,
        relative_depth_scale_m=4.0,
    )

    assert targets.depth_residual_m.shape == (1, 1, 1, 4, 3)
    assert targets.relative_geometry.shape == (1, 1, 1, 4, 3)
    assert targets.occupancy.shape == (1, 1, 1, 4, 3)
    assert targets.free_space.shape == (1, 1, 1, 4, 3)
    assert targets.weights.shape == (1, 1, 1, 4, 3)
    assert torch.all(targets.weights > 0)
    # BEV x centers are 2.5,3.5,4.5,5.5 and camera-z equals ego-x.
    torch.testing.assert_close(
        targets.depth_residual_m[0, 0, 0, :, 1],
        torch.tensor([2.0, 1.0, 0.0, -1.0]),
        atol=1e-5,
        rtol=0,
    )
    assert targets.free_space[0, 0, 0, 0].all()
    assert targets.free_space[0, 0, 0, 1].all()
    assert not targets.free_space[0, 0, 0, 2].any()
    assert targets.occupancy[0, 0, 0, 2].all()


def test_geometry_losses_are_confidence_masked_and_backpropagate() -> None:
    head = GeometrySupervisionHead(
        input_channels=8,
        num_views=1,
        num_heights=2,
        hidden_channels=12,
        max_depth_residual_m=20.0,
    )
    field = torch.randn(2, 8, 4, 3, requires_grad=True)
    prediction = head(field)
    shape = (2, 1, 2, 4, 3)
    targets = GeometryTargets(
        depth_residual_m=torch.randn(shape),
        relative_geometry=torch.randn(shape).clamp(-1, 1),
        occupancy=torch.randint(0, 2, shape).float(),
        free_space=torch.randint(0, 2, shape).float(),
        weights=torch.ones(shape),
        projected_depth_m=torch.ones(shape),
        sampled_depth_m=torch.ones(shape),
    )

    losses, metrics = geometry_supervision_losses(prediction, targets)

    assert set(losses) == {
        "geometry_depth",
        "geometry_occupancy",
        "geometry_free_space",
        "geometry_relative",
    }
    assert set(metrics) >= {
        "geometry_depth_mae_m",
        "geometry_valid_ratio",
        "geometry_occupancy_accuracy",
    }
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()
    assert field.grad is not None
    assert torch.isfinite(field.grad).all()
    assert any(parameter.grad is not None for parameter in head.parameters())


def test_zero_confidence_produces_linked_zero_losses() -> None:
    head = GeometrySupervisionHead(4, num_views=1, num_heights=1)
    prediction = head(torch.randn(1, 4, 2, 2, requires_grad=True))
    zeros = torch.zeros((1, 1, 1, 2, 2))
    targets = GeometryTargets(
        depth_residual_m=zeros,
        relative_geometry=zeros,
        occupancy=zeros,
        free_space=zeros,
        weights=zeros,
        projected_depth_m=zeros,
        sampled_depth_m=zeros,
    )

    losses, metrics = geometry_supervision_losses(prediction, targets)

    assert all(loss.item() == 0.0 for loss in losses.values())
    assert metrics["geometry_valid_ratio"].item() == 0.0
    sum(losses.values()).backward()


def test_geometry_target_shape_assertions() -> None:
    camera = _forward_camera()
    try:
        build_geometry_targets(
            depth_m=torch.ones(1, 16, 16),
            confidence=torch.ones(1, 1, 16, 16),
            camera=camera,
            field_size=(2, 2),
            x_range_m=(1.0, 3.0),
            y_range_m=(-1.0, 1.0),
            height_anchors_m=(0.0,),
        )
    except ValueError as error:
        assert "[B,V,Hd,Wd]" in str(error)
    else:
        raise AssertionError("invalid teacher depth shape was accepted")
