import torch

from starVLA.model.modules.field2plan.dynamics_supervision import (
    TemporalCameraBatch,
    build_dynamics_targets,
)


def _temporal_camera(batch: int, horizon: int, views: int) -> TemporalCameraBatch:
    intrinsics = torch.tensor(
        [[4.0, 0.0, 3.5], [0.0, 4.0, 3.5], [0.0, 0.0, 1.0]]
    ).reshape(1, 1, 1, 3, 3).repeat(batch, horizon, views, 1, 1)
    ego_to_camera = torch.eye(4).reshape(1, 1, 1, 4, 4).repeat(
        batch, horizon, views, 1, 1
    )
    # Planning x-forward, y-left, z-up -> camera x-right, y-down, z-forward.
    ego_to_camera[..., :3, :3] = torch.tensor(
        [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]
    )
    return TemporalCameraBatch(
        intrinsics=intrinsics,
        ego_to_camera=ego_to_camera,
        image_hw=torch.tensor([8.0, 8.0]).reshape(1, 1, 1, 2).repeat(
            batch, horizon, views, 1
        ),
        current_to_ego=torch.eye(4).reshape(1, 1, 4, 4).repeat(
            batch, horizon, 1, 1
        ),
        valid_mask=torch.ones(batch, horizon, views, dtype=torch.bool),
        view_names=tuple(f"v{index}" for index in range(views)),
        frame_indices=tuple(range(4, 4 + horizon)),
    ).validate()


def test_future_teacher_features_project_to_current_bev_and_have_gradient_free_targets() -> None:
    batch, horizon, views, channels = 2, 3, 1, 4
    teacher = torch.ones(batch, horizon, views, channels, 8, 8)
    teacher[:, 1] *= 2.0
    teacher[:, 2] *= 3.0
    confidence = torch.ones(batch, horizon, views, 8, 8)
    targets = build_dynamics_targets(
        teacher_features=teacher,
        confidence=confidence,
        camera=_temporal_camera(batch, horizon, views),
        field_size=(4, 4),
        x_range_m=(1.0, 5.0),
        y_range_m=(-2.0, 2.0),
        height_anchors_m=(0.0,),
        normalize_features=False,
    ).validate()

    assert targets.features.shape == (batch, horizon, channels, 4, 4)
    assert targets.weights.shape == (batch, horizon, 4, 4)
    assert targets.weights.max() <= 1.0
    valid = targets.weights > 0
    assert valid.any()
    for time in range(horizon):
        values = targets.features[:, time].permute(0, 2, 3, 1)[valid[:, time]]
        torch.testing.assert_close(values, torch.full_like(values, float(time + 1)))

