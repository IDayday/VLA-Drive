"""CPU-friendly geometry diagnostics for VGGT teacher features.

The helpers in this module do not load VGGT or NAVSIM.  They define the
coordinate, residualization, and metric contracts used by the offline probe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass(frozen=True)
class SlotResidualizer:
    """Training-set slot templates for features and regression targets."""

    feature_mean: torch.Tensor  # [Q,D]
    target_mean: torch.Tensor  # [Q,T]
    slot_valid: torch.Tensor  # [Q]


def project_lidar_to_depth_grid(
    points_xyz: torch.Tensor,
    *,
    sensor2lidar_rotation: torch.Tensor,
    sensor2lidar_translation: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size: tuple[int, int],
    grid_size: tuple[int, int],
    min_points: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project LiDAR points into a robust camera-depth grid.

    Args:
        points_xyz: LiDAR-frame points ``[N,3]``.
        sensor2lidar_rotation: Camera-to-LiDAR rotation ``[3,3]``.
        sensor2lidar_translation: Camera origin in LiDAR ``[3]``.
        intrinsics: Pinhole camera matrix ``[3,3]``.
        image_size: Original camera image ``(height, width)``.
        grid_size: Output content grid ``(rows, cols)``.

    Returns:
        Median camera-z depth ``[R,C]``, validity ``[R,C]``, and point counts
        ``[R,C]``.  Invalid depth cells are zero.
    """

    assert points_xyz.ndim == 2 and points_xyz.shape[-1] == 3
    assert sensor2lidar_rotation.shape == (3, 3)
    assert sensor2lidar_translation.shape == (3,)
    assert intrinsics.shape == (3, 3)
    height, width = int(image_size[0]), int(image_size[1])
    rows, cols = int(grid_size[0]), int(grid_size[1])
    assert height > 0 and width > 0 and rows > 0 and cols > 0
    assert min_points > 0

    device = points_xyz.device
    compute_dtype = torch.float64 if points_xyz.dtype == torch.float64 else torch.float32
    points = points_xyz.to(dtype=compute_dtype)
    rotation = sensor2lidar_rotation.to(device=device, dtype=compute_dtype)
    translation = sensor2lidar_translation.to(device=device, dtype=compute_dtype)
    camera_matrix = intrinsics.to(device=device, dtype=compute_dtype)

    # For row vectors, p_camera = (p_lidar - t_camera_in_lidar) @ R_camera_to_lidar.
    camera_points = (points - translation) @ rotation
    depth = camera_points[:, 2]
    projected = camera_points @ camera_matrix.transpose(0, 1)
    denominator = projected[:, 2]
    finite = torch.isfinite(projected).all(dim=-1) & torch.isfinite(depth)
    positive = (depth > 1e-5) & (denominator.abs() > 1e-8)
    u = projected[:, 0] / denominator.clamp_min(1e-8)
    v = projected[:, 1] / denominator.clamp_min(1e-8)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    keep = finite & positive & inside
    u, v, depth = u[keep], v[keep], depth[keep]

    row_index = torch.floor(v * rows / height).long().clamp(0, rows - 1)
    col_index = torch.floor(u * cols / width).long().clamp(0, cols - 1)
    flat_index = row_index * cols + col_index
    counts = torch.bincount(flat_index, minlength=rows * cols).reshape(rows, cols)
    output = torch.zeros(rows * cols, device=device, dtype=compute_dtype)
    for cell_index in flat_index.unique(sorted=True).tolist():
        values = depth[flat_index == int(cell_index)]
        output[int(cell_index)] = values.median()
    output = output.reshape(rows, cols)
    valid = counts >= int(min_points)
    output = output.masked_fill(~valid, 0)
    return output.to(dtype=torch.float32), valid, counts


def fit_slot_residualizer(
    features: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> SlotResidualizer:
    """Fit per-location means from training tensors ``[N,Q,D/T]``."""

    assert features.ndim == 3 and targets.ndim == 3
    assert features.shape[:2] == targets.shape[:2] == valid_mask.shape
    assert valid_mask.dtype == torch.bool
    counts = valid_mask.float().sum(dim=0)
    slot_valid = counts > 0
    assert slot_valid.any(), "at least one slot needs a valid training target"
    weights = valid_mask.unsqueeze(-1).to(dtype=torch.float32)
    denominator = counts.clamp_min(1)[:, None]
    feature_mean = (features.float() * weights).sum(dim=0) / denominator
    target_mean = (targets.float() * weights).sum(dim=0) / denominator
    return SlotResidualizer(
        feature_mean=feature_mean,
        target_mean=target_mean,
        slot_valid=slot_valid,
    )


def apply_slot_residualization(
    features: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    residualizer: SlotResidualizer,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Subtract training slot templates and flatten to ``[N*Q,D/T]``."""

    assert features.ndim == 3 and targets.ndim == 3
    assert features.shape[:2] == targets.shape[:2] == valid_mask.shape
    assert residualizer.feature_mean.shape == features.shape[1:]
    assert residualizer.target_mean.shape == targets.shape[1:]
    assert residualizer.slot_valid.shape == features.shape[1:2]
    feature_residual = features.float() - residualizer.feature_mean.to(features.device)
    target_residual = targets.float() - residualizer.target_mean.to(targets.device)
    return (
        feature_residual.reshape(-1, features.shape[-1]),
        target_residual.reshape(-1, targets.shape[-1]),
        (valid_mask & residualizer.slot_valid.to(valid_mask.device).unsqueeze(0)).reshape(-1),
    )


def regression_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    slot_template_prediction: torch.Tensor,
) -> Dict[str, float]:
    """Regression metrics for matched two-dimensional arrays ``[N,T]``."""

    assert prediction.ndim == target.ndim == slot_template_prediction.ndim == 2
    assert prediction.shape == target.shape == slot_template_prediction.shape
    prediction = prediction.detach().float()
    target = target.detach().float()
    baseline = slot_template_prediction.detach().float()
    sse = (prediction - target).square().sum()
    baseline_sse = (baseline - target).square().sum()
    centered_sse = (target - target.mean(dim=0, keepdim=True)).square().sum()
    correlations = []
    for target_index in range(target.shape[1]):
        x = prediction[:, target_index] - prediction[:, target_index].mean()
        y = target[:, target_index] - target[:, target_index].mean()
        denominator = x.square().sum().sqrt() * y.square().sum().sqrt()
        correlations.append(float((x * y).sum() / denominator.clamp_min(1e-12)))
    return {
        "rmse": float(torch.sqrt((prediction - target).square().mean())),
        "r2_vs_constant": float(1.0 - sse / centered_sse.clamp_min(1e-12)),
        "sse_ratio_vs_slot_template": float(sse / baseline_sse.clamp_min(1e-12)),
        "pearson_mean": sum(correlations) / len(correlations),
    }


def scale_aligned_depth_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Dict[str, float]:
    """Median-scale-align positive depth maps and report valid-bin errors."""

    assert prediction.shape == target.shape == valid_mask.shape
    assert valid_mask.dtype == torch.bool
    valid = valid_mask & torch.isfinite(prediction) & torch.isfinite(target)
    valid &= (prediction > 1e-6) & (target > 1e-6)
    assert valid.any(), "depth comparison has no valid positive bins"
    predicted = prediction.detach().float()[valid]
    expected = target.detach().float()[valid]
    scale = torch.median(expected / predicted)
    aligned = predicted * scale
    difference = aligned - expected
    return {
        "valid_bins": int(valid.sum()),
        "median_scale": float(scale),
        "abs_rel": float((difference.abs() / expected).mean()),
        "rmse": float(torch.sqrt(difference.square().mean())),
        "log_rmse": float(
            torch.sqrt((torch.log(aligned) - torch.log(expected)).square().mean())
        ),
    }
