"""Structured future-feature supervision for action-free dynamics fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .camera_geometry import make_ego_bev_anchors, project_ego_points
from .types import CameraBatch, TemporalCameraBatch


@dataclass(frozen=True)
class DynamicsTargets:
    """Aligned teacher field and weights.

    Shapes are ``features=[B,H,Ct,Ny,Nx]`` and
    ``weights=[B,H,Ny,Nx]``.
    """

    features: torch.Tensor
    weights: torch.Tensor

    def validate(self) -> "DynamicsTargets":
        if self.features.ndim != 5:
            raise ValueError("dynamics target features must be [B,H,Ct,Ny,Nx]")
        if self.weights.shape != (
            self.features.shape[0],
            self.features.shape[1],
            self.features.shape[3],
            self.features.shape[4],
        ):
            raise ValueError("dynamics target weights must be [B,H,Ny,Nx]")
        return self


@dataclass(frozen=True)
class DynamicsPrediction:
    """Predicted teacher-space features ``[B,H,Ct,Ny,Nx]``."""

    features: torch.Tensor

    def validate(self) -> "DynamicsPrediction":
        if self.features.ndim != 5:
            raise ValueError("dynamics prediction must have shape [B,H,Ct,Ny,Nx]")
        return self


class DynamicsSupervisionHead(nn.Module):
    """Pointwise student-to-teacher feature adapter."""

    def __init__(
        self,
        input_channels: int,
        teacher_channels: int,
        hidden_channels: int | None = None,
    ) -> None:
        super().__init__()
        if min(int(input_channels), int(teacher_channels)) <= 0:
            raise ValueError("dynamics supervision dimensions must be positive")
        hidden = int(hidden_channels or input_channels)
        if hidden <= 0:
            raise ValueError("dynamics supervision hidden_channels must be positive")
        self.input_channels = int(input_channels)
        self.teacher_channels = int(teacher_channels)
        self.projection = nn.Sequential(
            nn.Linear(self.input_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.teacher_channels),
        )

    def forward(self, field: torch.Tensor) -> DynamicsPrediction:
        if field.ndim != 5 or field.shape[2] != self.input_channels:
            raise ValueError("dynamics field must have shape [B,H,C,Ny,Nx]")
        features = self.projection(field.permute(0, 1, 3, 4, 2))
        return DynamicsPrediction(
            features.permute(0, 1, 4, 2, 3).contiguous()
        ).validate()


def build_dynamics_targets(
    *,
    teacher_features: torch.Tensor,
    confidence: torch.Tensor,
    camera: TemporalCameraBatch,
    field_size,
    x_range_m,
    y_range_m,
    height_anchors_m,
    normalize_features: bool = True,
) -> DynamicsTargets:
    """Project future per-view features into the current ego BEV.

    ``teacher_features`` is ``[B,H,V,Ct,Ht,Wt]`` and contains only offline
    demonstrated-future supervision.  The returned tensor is detached.  It is
    never provided to the action-free writer at inference time.
    """

    if teacher_features.ndim != 6:
        raise ValueError("teacher_features must have shape [B,H,V,Ct,Ht,Wt]")
    batch, horizon, views, channels, feature_h, feature_w = teacher_features.shape
    if confidence.shape != (batch, horizon, views, feature_h, feature_w):
        raise ValueError("confidence must have shape [B,H,V,Ht,Wt]")
    camera.validate()
    if camera.intrinsics.shape[:3] != (batch, horizon, views):
        raise ValueError("teacher features and temporal camera B,H,V differ")
    if len(field_size) != 2 or min(field_size) <= 0:
        raise ValueError("field_size must be positive [Ny,Nx]")
    if not height_anchors_m:
        raise ValueError("height_anchors_m cannot be empty")

    features = teacher_features.detach().to(dtype=torch.float32)
    confidence = confidence.detach().to(
        device=features.device, dtype=torch.float32
    ).clamp(0.0, 1.0)
    if normalize_features:
        features = F.normalize(features, dim=3, eps=1e-6)
    anchors = make_ego_bev_anchors(
        field_size,
        x_range_m,
        y_range_m,
        height_anchors_m,
        device=features.device,
    )
    ny, nx, num_heights = anchors.shape[:3]
    points = anchors.reshape(1, ny, nx, num_heights, 3).expand(
        batch * horizon, -1, -1, -1, -1
    )
    current_to_camera = camera.ego_to_camera.to(
        device=features.device, dtype=torch.float32
    ) @ camera.current_to_ego.to(
        device=features.device, dtype=torch.float32
    )[:, :, None]
    flattened_camera = CameraBatch(
        intrinsics=camera.intrinsics.to(
            device=features.device, dtype=torch.float32
        ).reshape(batch * horizon, views, 3, 3),
        ego_to_camera=current_to_camera.reshape(batch * horizon, views, 4, 4),
        image_hw=camera.image_hw.to(
            device=features.device, dtype=torch.float32
        ).reshape(batch * horizon, views, 2),
        view_names=camera.view_names,
        frame_index=int(camera.frame_indices[0]),
    ).validate()
    pixels, projection_valid, _ = project_ego_points(
        points,
        flattened_camera.intrinsics,
        flattened_camera.ego_to_camera,
        flattened_camera.image_hw,
    )
    image_hw = flattened_camera.image_hw
    width = image_hw[..., 1, None, None, None]
    height = image_hw[..., 0, None, None, None]
    grid = torch.stack(
        (
            2.0 * (pixels[..., 0] + 0.5) / width - 1.0,
            2.0 * (pixels[..., 1] + 0.5) / height - 1.0,
        ),
        dim=-1,
    ).reshape(batch * horizon * views, ny * nx * num_heights, 1, 2)

    sampled = F.grid_sample(
        features.reshape(batch * horizon * views, channels, feature_h, feature_w),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )[..., 0]
    sampled = sampled.reshape(
        batch, horizon, views, channels, ny, nx, num_heights
    )
    sampled_confidence = F.grid_sample(
        confidence.reshape(batch * horizon * views, 1, feature_h, feature_w),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )[..., 0].reshape(batch, horizon, views, ny, nx, num_heights)
    valid = projection_valid.reshape(
        batch, horizon, views, ny, nx, num_heights
    )
    valid = valid & camera.valid_mask.to(
        device=features.device, dtype=torch.bool
    )[..., None, None, None]
    weights = valid.to(torch.float32) * sampled_confidence
    denominator = weights.sum(dim=(2, 5)).clamp_min(1e-6)
    target = (sampled * weights[:, :, :, None]).sum(dim=(2, 6)) / denominator[
        :, :, None
    ]
    if normalize_features:
        target = F.normalize(target, dim=2, eps=1e-6)
    aggregate_weight = weights.mean(dim=(2, 5)).clamp(0.0, 1.0)
    return DynamicsTargets(target.detach(), aggregate_weight.detach()).validate()


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def dynamics_supervision_losses(
    prediction: DynamicsPrediction,
    targets: DynamicsTargets,
    *,
    log_variance: torch.Tensor | None = None,
    temporal_contrast_margin: float = 0.05,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Compute feature, temporal-control and uncertainty losses.

    The shuffled comparison rolls teacher time only.  It is used as a probe
    and a margin loss, never as a candidate-specific counterfactual future.
    """

    prediction.validate()
    targets.validate()
    if prediction.features.shape != targets.features.shape:
        raise ValueError("dynamics prediction and target feature shapes differ")
    if temporal_contrast_margin < 0:
        raise ValueError("temporal_contrast_margin cannot be negative")
    predicted = prediction.features.float()
    target = targets.features.to(device=predicted.device, dtype=torch.float32)
    weights = targets.weights.to(device=predicted.device, dtype=torch.float32)
    finite = torch.isfinite(target).all(dim=2)
    weights = weights * finite.to(dtype=weights.dtype)
    target = torch.where(finite[:, :, None], target, torch.zeros_like(target))

    predicted_normalized = F.normalize(predicted, dim=2, eps=1e-6)
    target_normalized = F.normalize(target, dim=2, eps=1e-6)
    positive_similarity = (predicted_normalized * target_normalized).sum(dim=2)
    cosine_loss = _weighted_mean(1.0 - positive_similarity, weights)
    pointwise_smooth_l1 = F.smooth_l1_loss(
        predicted, target, reduction="none"
    ).mean(dim=2)
    smooth_l1_loss = _weighted_mean(pointwise_smooth_l1, weights)

    if predicted.shape[1] > 1:
        shuffled_target = torch.roll(target_normalized, shifts=1, dims=1)
        shuffled_similarity = (predicted_normalized * shuffled_target).sum(dim=2)
        temporal_contrast = _weighted_mean(
            F.relu(
                float(temporal_contrast_margin)
                + shuffled_similarity
                - positive_similarity
            ),
            weights,
        )
        shuffled_metric = _weighted_mean(shuffled_similarity, weights)
    else:
        linked_zero = predicted.sum() * 0.0
        temporal_contrast = linked_zero
        shuffled_metric = linked_zero.detach()

    if log_variance is None:
        uncertainty_loss = predicted.sum() * 0.0
    else:
        expected_shape = (
            predicted.shape[0],
            predicted.shape[1],
            1,
            predicted.shape[3],
            predicted.shape[4],
        )
        if log_variance.shape != expected_shape:
            raise ValueError("log_variance must have shape [B,H,1,Ny,Nx]")
        log_var = log_variance.float().squeeze(2).clamp(-8.0, 8.0)
        heteroscedastic = torch.exp(-log_var) * pointwise_smooth_l1 + 0.5 * log_var
        uncertainty_loss = _weighted_mean(heteroscedastic, weights)

    losses = {
        "dynamics_cosine": cosine_loss,
        "dynamics_smooth_l1": smooth_l1_loss,
        "dynamics_temporal_contrast": temporal_contrast,
        "dynamics_uncertainty": uncertainty_loss,
    }
    metrics = {
        "dynamics_cosine_similarity": _weighted_mean(
            positive_similarity, weights
        ).detach(),
        "dynamics_shuffled_cosine_similarity": shuffled_metric.detach(),
        "dynamics_probe_margin": (
            _weighted_mean(positive_similarity, weights) - shuffled_metric
        ).detach(),
        "dynamics_valid_ratio": (weights > 0).float().mean().detach(),
        "dynamics_feature_mae": _weighted_mean(
            (predicted - target).abs().mean(dim=2), weights
        ).detach(),
    }
    return losses, metrics
