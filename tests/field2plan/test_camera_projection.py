import torch

from starVLA.model.modules.field2plan.camera_geometry import (
    project_ego_points,
    scale_intrinsics_for_crop_resize,
    sensor_to_lidar_to_ego_to_camera,
)


def test_synthetic_camera_projects_known_points() -> None:
    points = torch.tensor([[[0.0, 0.0, 10.0], [1.0, 2.0, -1.0]]])
    intrinsics = torch.tensor(
        [[[[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]]]
    )
    ego_to_camera = torch.eye(4).reshape(1, 1, 4, 4)
    image_hw = torch.tensor([[[80.0, 100.0]]])

    pixels, valid, depth = project_ego_points(
        points, intrinsics, ego_to_camera, image_hw
    )

    assert pixels.shape == (1, 1, 2, 2)
    torch.testing.assert_close(pixels[0, 0, 0], torch.tensor([50.0, 40.0]))
    assert valid[0, 0, 0]
    assert not valid[0, 0, 1]
    torch.testing.assert_close(depth[0, 0, 0], torch.tensor(10.0))


def test_crop_resize_updates_principal_point_and_focal_length() -> None:
    k = torch.tensor([[[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]])
    crop_xywh = torch.tensor([[320.0, 0.0, 1280.0, 1080.0]])
    output_hw = torch.tensor([[576.0, 1024.0]])

    scaled = scale_intrinsics_for_crop_resize(k, crop_xywh, output_hw)

    torch.testing.assert_close(scaled[0, 0, 0], torch.tensor(800.0))
    torch.testing.assert_close(scaled[0, 1, 1], torch.tensor(533.3333333))
    torch.testing.assert_close(scaled[0, 0, 2], torch.tensor(512.0))
    torch.testing.assert_close(scaled[0, 1, 2], torch.tensor(288.0))


def test_sensor_lidar_conversion_requires_explicit_lidar_to_ego() -> None:
    sensor_to_lidar = torch.eye(4).reshape(1, 1, 4, 4)
    lidar_to_ego = torch.eye(4).reshape(1, 4, 4)
    ego_to_camera = sensor_to_lidar_to_ego_to_camera(
        sensor_to_lidar, lidar_to_ego
    )
    torch.testing.assert_close(ego_to_camera, sensor_to_lidar)
