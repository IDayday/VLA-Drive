"""Metric geometry targets, auxiliary head and confidence-masked losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .camera_geometry import make_ego_bev_anchors, project_ego_points
from .types import CameraBatch


@dataclass(frozen=True)
class GeometryTargets:
    """Teacher targets with all tensors shaped ``[B,V,Z,Ny,Nx]``."""

    depth_residual_m: torch.Tensor
    relative_geometry: torch.Tensor
    occupancy: torch.Tensor
    free_space: torch.Tensor
    weights: torch.Tensor
    projected_depth_m: torch.Tensor
    sampled_depth_m: torch.Tensor

    def validate(self) -> "GeometryTargets":
        shape = self.depth_residual_m.shape
        if len(shape) != 5:
            raise ValueError("geometry targets must have shape [B,V,Z,Ny,Nx]")
        for name in (
            "relative_geometry",
            "occupancy",
            "free_space",
            "weights",
            "projected_depth_m",
            "sampled_depth_m",
        ):
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} must match depth_residual_m shape")
        return self


@dataclass(frozen=True)
class GeometryPrediction:
    """Student predictions, each shaped ``[B,V,Z,Ny,Nx]``."""

    depth_residual_m: torch.Tensor
    relative_geometry: torch.Tensor
    occupancy_logits: torch.Tensor
    free_space_logits: torch.Tensor

    def validate(self) -> "GeometryPrediction":
        shape = self.depth_residual_m.shape
        if len(shape) != 5:
            raise ValueError("geometry predictions must have shape [B,V,Z,Ny,Nx]")
        for name in (
            "relative_geometry",
            "occupancy_logits",
            "free_space_logits",
        ):
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} must match depth_residual_m shape")
        return self


def build_geometry_targets(
    *,
    depth_m: torch.Tensor,
    confidence: torch.Tensor,
    camera: CameraBatch,
    field_size: Sequence[int],
    x_range_m: Sequence[float],
    y_range_m: Sequence[float],
    height_anchors_m: Sequence[float],
    occupancy_threshold_m: float = 0.75,
    free_space_margin_m: float = 0.75,
    relative_depth_scale_m: float = 10.0,
    min_teacher_depth_m: float = 0.1,
    max_teacher_depth_m: float = 200.0,
) -> GeometryTargets:
    """Project ego anchors into metric-depth maps.

    Args:
        depth_m/confidence: offline teacher maps ``[B,V,Hd,Wd]``.
        camera: raw-image calibrated camera matching the teacher views.

    No action or future trajectory is an input to this function.
    """

    if depth_m.ndim != 4:
        raise ValueError("depth_m must have shape [B,V,Hd,Wd]")
    if confidence.shape != depth_m.shape:
        raise ValueError("confidence must match depth_m [B,V,Hd,Wd]")
    camera.validate()
    batch, views = depth_m.shape[:2]
    if camera.intrinsics.shape[:2] != (batch, views):
        raise ValueError("teacher camera B,V must match depth maps")
    if occupancy_threshold_m <= 0 or free_space_margin_m <= 0:
        raise ValueError("occupancy/free-space margins must be positive")
    if relative_depth_scale_m <= 0:
        raise ValueError("relative_depth_scale_m must be positive")
    if min_teacher_depth_m < 0 or max_teacher_depth_m <= min_teacher_depth_m:
        raise ValueError("teacher depth range is invalid")

    anchors = make_ego_bev_anchors(
        field_size,
        x_range_m,
        y_range_m,
        height_anchors_m,
        device=depth_m.device,
    )
    ny, nx, heights = anchors.shape[:3]
    points = anchors[None].expand(batch, -1, -1, -1, -1)
    pixels, projection_valid, projected_depth = project_ego_points(
        points,
        camera.intrinsics,
        camera.ego_to_camera,
        camera.image_hw,
    )
    image_hw = camera.image_hw.to(device=depth_m.device, dtype=torch.float32)
    width = image_hw[..., 1, None, None, None]
    height = image_hw[..., 0, None, None, None]
    grid = torch.stack(
        (
            2.0 * (pixels[..., 0] + 0.5) / width - 1.0,
            2.0 * (pixels[..., 1] + 0.5) / height - 1.0,
        ),
        dim=-1,
    ).reshape(batch * views, ny * nx * heights, 1, 2)

    def sample(maps: torch.Tensor) -> torch.Tensor:
        sampled = F.grid_sample(
            maps.reshape(batch * views, 1, *maps.shape[-2:]).float(),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )[..., 0]
        return sampled.reshape(batch, views, ny, nx, heights).permute(
            0, 1, 4, 2, 3
        )

    sampled_depth = sample(depth_m)
    sampled_confidence = sample(confidence).clamp(0.0, 1.0)
    projected_depth = projected_depth.permute(0, 1, 4, 2, 3)
    projection_valid = projection_valid.permute(0, 1, 4, 2, 3)
    teacher_valid = (
        torch.isfinite(sampled_depth)
        & (sampled_depth >= min_teacher_depth_m)
        & (sampled_depth <= max_teacher_depth_m)
    )
    weights = (
        projection_valid.to(torch.float32)
        * teacher_valid.to(torch.float32)
        * sampled_confidence
    )
    sampled_depth = torch.where(teacher_valid, sampled_depth, projected_depth)
    residual = sampled_depth - projected_depth
    occupancy = (residual.abs() <= occupancy_threshold_m).to(torch.float32)
    free_space = (residual > free_space_margin_m).to(torch.float32)
    relative = (residual / relative_depth_scale_m).clamp(-1.0, 1.0)
    return GeometryTargets(
        depth_residual_m=residual,
        relative_geometry=relative,
        occupancy=occupancy,
        free_space=free_space,
        weights=weights,
        projected_depth_m=projected_depth,
        sampled_depth_m=sampled_depth,
    ).validate()


class GeometrySupervisionHead(nn.Module):
    """Pointwise BEV head preserving view/height-specific physical targets."""

    def __init__(
        self,
        input_channels: int,
        num_views: int,
        num_heights: int,
        hidden_channels: int = 128,
        max_depth_residual_m: float = 50.0,
    ) -> None:
        super().__init__()
        if min(input_channels, num_views, num_heights, hidden_channels) <= 0:
            raise ValueError("geometry head dimensions must be positive")
        if max_depth_residual_m <= 0:
            raise ValueError("max_depth_residual_m must be positive")
        self.num_views = int(num_views)
        self.num_heights = int(num_heights)
        self.max_depth_residual_m = float(max_depth_residual_m)
        self.projection = nn.Sequential(
            nn.Linear(int(input_channels), int(hidden_channels)),
            nn.GELU(),
            nn.Linear(int(hidden_channels), self.num_views * self.num_heights * 4),
        )

    def forward(self, field: torch.Tensor) -> GeometryPrediction:
        """Predict geometry from field ``[B,C,Ny,Nx]``."""

        if field.ndim != 4:
            raise ValueError("field must have shape [B,C,Ny,Nx]")
        batch, _, ny, nx = field.shape
        raw = self.projection(field.permute(0, 2, 3, 1))
        raw = raw.reshape(
            batch, ny, nx, self.num_views, self.num_heights, 4
        ).permute(0, 3, 4, 1, 2, 5)
        return GeometryPrediction(
            depth_residual_m=torch.tanh(raw[..., 0])
            * self.max_depth_residual_m,
            relative_geometry=torch.tanh(raw[..., 1]),
            occupancy_logits=raw[..., 2],
            free_space_logits=raw[..., 3],
        ).validate()


def _weighted_mean(value: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (value * weights).sum() / weights.sum().clamp_min(1.0)


def geometry_supervision_losses(
    prediction: GeometryPrediction,
    targets: GeometryTargets,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Return confidence-masked scalar losses and interpretable probes."""

    prediction.validate()
    targets.validate()
    shape = prediction.depth_residual_m.shape
    if targets.depth_residual_m.shape != shape:
        raise ValueError("geometry prediction and target shapes differ")
    weights = targets.weights.to(
        device=prediction.depth_residual_m.device, dtype=torch.float32
    )
    depth_target = targets.depth_residual_m.to(
        device=prediction.depth_residual_m.device, dtype=torch.float32
    )
    relative_target = targets.relative_geometry.to(
        device=prediction.depth_residual_m.device, dtype=torch.float32
    )
    occupancy_target = targets.occupancy.to(
        device=prediction.depth_residual_m.device, dtype=torch.float32
    )
    free_target = targets.free_space.to(
        device=prediction.depth_residual_m.device, dtype=torch.float32
    )
    depth_error = F.smooth_l1_loss(
        prediction.depth_residual_m.float(), depth_target, reduction="none"
    )
    relative_error = F.smooth_l1_loss(
        prediction.relative_geometry.float(), relative_target, reduction="none"
    )
    occupancy_error = F.binary_cross_entropy_with_logits(
        prediction.occupancy_logits.float(), occupancy_target, reduction="none"
    )
    free_error = F.binary_cross_entropy_with_logits(
        prediction.free_space_logits.float(), free_target, reduction="none"
    )
    losses = {
        "geometry_depth": _weighted_mean(depth_error, weights),
        "geometry_occupancy": _weighted_mean(occupancy_error, weights),
        "geometry_free_space": _weighted_mean(free_error, weights),
        "geometry_relative": _weighted_mean(relative_error, weights),
    }
    occupancy_correct = (
        (prediction.occupancy_logits >= 0) == (occupancy_target >= 0.5)
    ).float()
    metrics = {
        "geometry_depth_mae_m": _weighted_mean(
            (prediction.depth_residual_m.float() - depth_target).abs(), weights
        ).detach(),
        "geometry_relative_mae": _weighted_mean(
            (prediction.relative_geometry.float() - relative_target).abs(), weights
        ).detach(),
        "geometry_occupancy_accuracy": _weighted_mean(
            occupancy_correct, weights
        ).detach(),
        "geometry_valid_ratio": (weights > 0).float().mean().detach(),
    }
    return losses, metrics

