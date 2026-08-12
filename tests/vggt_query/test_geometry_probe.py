from __future__ import annotations

import math

import torch

from starVLA.model.modules.vggt_query.geometry_probe import (
    apply_slot_residualization,
    fit_slot_residualizer,
    project_lidar_to_depth_grid,
    regression_metrics,
    scale_aligned_depth_metrics,
)


def test_project_lidar_to_depth_grid_uses_navsim_sensor2lidar_contract():
    # sensor2lidar is camera -> lidar.  This setup translates camera points by
    # +1m along LiDAR x, so a LiDAR point at [1, 0, 2] is centered in camera.
    points = torch.tensor(
        [
            [1.0, 0.0, 2.0],
            [1.0, 0.0, 4.0],
            [2.0, 0.0, 2.0],
            [1.0, 2.0, 2.0],  # outside the image
            [1.0, 0.0, -1.0],  # behind the camera
        ]
    )
    rotation = torch.eye(3)
    translation = torch.tensor([1.0, 0.0, 0.0])
    intrinsics = torch.tensor(
        [[2.0, 0.0, 2.0], [0.0, 2.0, 2.0], [0.0, 0.0, 1.0]]
    )

    depth, valid, counts = project_lidar_to_depth_grid(
        points,
        sensor2lidar_rotation=rotation,
        sensor2lidar_translation=translation,
        intrinsics=intrinsics,
        image_size=(4, 4),
        grid_size=(2, 2),
        min_points=2,
    )

    assert depth.shape == valid.shape == counts.shape == (2, 2)
    assert counts.tolist() == [[0, 0], [0, 3]]
    assert valid.tolist() == [[False, False], [False, True]]
    assert depth[1, 1].item() == 2.0  # median of [2, 4, 2]


def test_slot_residualizer_removes_location_template_without_test_leakage():
    template_x = torch.tensor([[10.0, -2.0], [-3.0, 7.0]])
    template_y = torch.tensor([[4.0], [-5.0]])
    train_signal = torch.tensor([-1.0, 0.0, 1.0])
    train_x = template_x.unsqueeze(0) + train_signal[:, None, None] * torch.tensor(
        [1.0, 2.0]
    )
    train_y = template_y.unsqueeze(0) + 3.0 * train_signal[:, None, None]
    valid = torch.ones(3, 2, dtype=torch.bool)

    residualizer = fit_slot_residualizer(train_x, train_y, valid)
    x_residual, y_residual, flat_valid = apply_slot_residualization(
        train_x, train_y, valid, residualizer
    )

    assert flat_valid.all()
    assert torch.allclose(x_residual.reshape(3, 2, 2).mean(dim=0), torch.zeros(2, 2))
    assert torch.allclose(y_residual.reshape(3, 2, 1).mean(dim=0), torch.zeros(2, 1))

    # Evaluation must use training means rather than re-centering the test set.
    test_x = template_x.unsqueeze(0) + 5.0
    test_y = template_y.unsqueeze(0) + 6.0
    test_x_residual, test_y_residual, _ = apply_slot_residualization(
        test_x,
        test_y,
        torch.ones(1, 2, dtype=torch.bool),
        residualizer,
    )
    assert torch.allclose(test_x_residual, torch.full_like(test_x_residual, 5.0))
    assert torch.allclose(test_y_residual, torch.full_like(test_y_residual, 6.0))


def test_regression_metrics_compare_against_slot_template_baseline():
    target = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    prediction = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    baseline = torch.full_like(target, 2.5)
    metrics = regression_metrics(prediction, target, baseline)
    assert metrics["r2_vs_constant"] == 1.0
    assert metrics["sse_ratio_vs_slot_template"] == 0.0
    assert metrics["pearson_mean"] > 0.999


def test_scale_aligned_depth_metrics_recovers_global_scale_and_masks_bins():
    prediction = torch.tensor([[1.0, 2.0], [100.0, 4.0]])
    target = torch.tensor([[2.0, 4.0], [7.0, 8.0]])
    valid = torch.tensor([[True, True], [False, True]])
    metrics = scale_aligned_depth_metrics(prediction, target, valid)
    assert math.isclose(metrics["median_scale"], 2.0)
    assert metrics["abs_rel"] < 1e-7
    assert metrics["rmse"] < 1e-7
