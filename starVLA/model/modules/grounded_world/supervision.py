"""Confound-separated current-prior and future-memory supervision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class FeatureAlignmentTarget:
    """Detached spatial target with explicit identity.

    Features are current ``[B,C,H,W]`` or future ``[B,T,C,H,W]``. Weights
    have the corresponding shape with the channel dimension removed.
    """

    features: torch.Tensor
    weights: torch.Tensor
    target_id: str

    def validate(self) -> "FeatureAlignmentTarget":
        if self.features.ndim not in (4, 5):
            raise ValueError("alignment features must be current or future fields")
        expected = (
            (self.features.shape[0], *self.features.shape[2:])
            if self.features.ndim == 4
            else (
                self.features.shape[0],
                self.features.shape[1],
                *self.features.shape[3:],
            )
        )
        if self.weights.shape != expected:
            raise ValueError("alignment weights shape does not match features")
        if not isinstance(self.target_id, str) or not self.target_id:
            raise ValueError("alignment target_id must be non-empty")
        if not torch.isfinite(self.features).all() or not torch.isfinite(self.weights).all():
            raise ValueError("alignment target contains non-finite values")
        if (self.weights < 0).any():
            raise ValueError("alignment weights cannot be negative")
        return self


@dataclass(frozen=True)
class FutureTargetContract:
    """Require a teacher-control-independent student/EMA future target."""

    source: str
    target_id: str
    shared_across_teacher_controls: bool

    def validate(self) -> "FutureTargetContract":
        if self.source != "student_ema":
            raise ValueError("future target source must be student_ema")
        if not self.target_id:
            raise ValueError("future target_id must be non-empty")
        if not self.shared_across_teacher_controls:
            raise ValueError("future target must be shared across teacher controls")
        return self


def _channel_dim(value: torch.Tensor) -> int:
    return 1 if value.ndim == 4 else 2


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def alignment_losses(
    prediction: torch.Tensor,
    target: FeatureAlignmentTarget,
    *,
    prefix: str,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Align one current or future field and report scene-shuffle margin."""

    target.validate()
    if prediction.shape != target.features.shape:
        raise ValueError("prediction and alignment target shapes differ")
    if not prefix:
        raise ValueError("alignment prefix must be non-empty")
    channel_dim = _channel_dim(prediction)
    predicted = prediction.float()
    expected = target.features.detach().to(device=predicted.device, dtype=torch.float32)
    weights = target.weights.detach().to(device=predicted.device, dtype=torch.float32)
    pred_norm = F.normalize(predicted, dim=channel_dim, eps=1e-6)
    target_norm = F.normalize(expected, dim=channel_dim, eps=1e-6)
    positive = (pred_norm * target_norm).sum(dim=channel_dim)
    cosine = _weighted_mean(1.0 - positive, weights)
    smooth_l1 = _weighted_mean(
        F.smooth_l1_loss(predicted, expected, reduction="none").mean(
            dim=channel_dim
        ),
        weights,
    )
    if predicted.shape[0] > 1:
        shuffled = torch.roll(target_norm, shifts=1, dims=0)
        shuffled_similarity = (pred_norm * shuffled).sum(dim=channel_dim)
        shuffled_mean = _weighted_mean(shuffled_similarity, weights)
    else:
        shuffled_mean = predicted.sum() * 0.0
    positive_mean = _weighted_mean(positive, weights)
    return (
        {
            f"{prefix}_cosine": cosine,
            f"{prefix}_smooth_l1": smooth_l1,
        },
        {
            f"{prefix}_cosine_similarity": positive_mean.detach(),
            f"{prefix}_scene_shuffled_similarity": shuffled_mean.detach(),
            f"{prefix}_scene_shuffle_margin": (
                positive_mean - shuffled_mean
            ).detach(),
            f"{prefix}_valid_ratio": (weights > 0).float().mean().detach(),
        },
    )


def global_alignment_losses(
    prediction: torch.Tensor,
    *,
    target: torch.Tensor,
    weights: torch.Tensor,
    target_id: str,
    prefix: str,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Align global current/history priors without inventing BEV coordinates.

    ``prediction`` and ``target`` are ``[B,C]`` and ``weights`` is ``[B]``.
    The target is detached. The scene-shuffle margin tests whether the learned
    representation is scene-specific instead of matching only teacher stats.
    """

    if prediction.ndim != 2 or target.shape != prediction.shape:
        raise ValueError("global alignment prediction/target must share [B,C]")
    if weights.shape != (prediction.shape[0],):
        raise ValueError("global alignment weights must have shape [B]")
    if not target_id or not prefix:
        raise ValueError("global alignment target_id and prefix must be non-empty")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("global alignment features contain non-finite values")
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("global alignment weights must be finite and non-negative")

    predicted = prediction.float()
    expected = target.detach().to(device=predicted.device, dtype=torch.float32)
    scene_weights = weights.detach().to(device=predicted.device, dtype=torch.float32)
    pred_norm = F.normalize(predicted, dim=1, eps=1e-6)
    target_norm = F.normalize(expected, dim=1, eps=1e-6)
    positive = (pred_norm * target_norm).sum(dim=1)
    cosine = _weighted_mean(1.0 - positive, scene_weights)
    smooth_l1 = _weighted_mean(
        F.smooth_l1_loss(predicted, expected, reduction="none").mean(dim=1),
        scene_weights,
    )
    if predicted.shape[0] > 1:
        shuffled_mean = _weighted_mean(
            (pred_norm * torch.roll(target_norm, shifts=1, dims=0)).sum(dim=1),
            scene_weights,
        )
    else:
        shuffled_mean = predicted.sum() * 0.0
    positive_mean = _weighted_mean(positive, scene_weights)
    return (
        {
            f"{prefix}_cosine": cosine,
            f"{prefix}_smooth_l1": smooth_l1,
        },
        {
            f"{prefix}_cosine_similarity": positive_mean.detach(),
            f"{prefix}_scene_shuffled_similarity": shuffled_mean.detach(),
            f"{prefix}_scene_shuffle_margin": (
                positive_mean - shuffled_mean
            ).detach(),
            f"{prefix}_valid_ratio": (scene_weights > 0).float().mean().detach(),
            f"{prefix}_alignment_is_global": prediction.new_tensor(1.0).detach(),
        },
    )


def future_prediction_losses(
    prediction: torch.Tensor,
    target: FeatureAlignmentTarget,
    *,
    contract: FutureTargetContract,
    temporal_margin: float = 0.05,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Supervise future prediction with the common student/EMA target."""

    contract.validate()
    target.validate()
    if target.target_id != contract.target_id:
        raise ValueError("future target identity differs from configured contract")
    if prediction.ndim != 5 or prediction.shape != target.features.shape:
        raise ValueError("future prediction and target must share [B,H,C,Ny,Nx]")
    if temporal_margin < 0:
        raise ValueError("temporal_margin cannot be negative")
    base_losses, metrics = alignment_losses(
        prediction,
        target,
        prefix="future",
    )
    predicted = F.normalize(prediction.float(), dim=2, eps=1e-6)
    expected = F.normalize(
        target.features.detach().to(device=prediction.device, dtype=torch.float32),
        dim=2,
        eps=1e-6,
    )
    weights = target.weights.detach().to(device=prediction.device, dtype=torch.float32)
    positive = (predicted * expected).sum(dim=2)
    if prediction.shape[1] > 1:
        temporal = (predicted * torch.roll(expected, shifts=1, dims=1)).sum(dim=2)
        temporal_contrast = _weighted_mean(
            F.relu(float(temporal_margin) + temporal - positive),
            weights,
        )
        temporal_mean = _weighted_mean(temporal, weights)
    else:
        temporal_contrast = prediction.sum() * 0.0
        temporal_mean = temporal_contrast.detach()
    positive_mean = _weighted_mean(positive, weights)
    losses = {
        "future_cosine": base_losses["future_cosine"],
        "future_smooth_l1": base_losses["future_smooth_l1"],
        "future_temporal_contrast": temporal_contrast,
    }
    metrics.update(
        {
            "future_temporal_shuffled_similarity": temporal_mean.detach(),
            "future_temporal_margin": (positive_mean - temporal_mean).detach(),
        }
    )
    return losses, metrics
