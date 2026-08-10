"""Training-only physical consequence grounding heads and losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F


CONSEQUENCE_NAMES = (
    "clearance",
    "ttc",
    "collision",
    "lane_distance",
    "progress",
    "comfort",
)


@dataclass(frozen=True)
class ConsequencePrediction:
    """Physical component values/log variances ``[B,K,6]``."""

    values: torch.Tensor
    log_variance: torch.Tensor

    def validate(self) -> "ConsequencePrediction":
        if self.values.ndim != 3 or self.values.shape[-1] != len(CONSEQUENCE_NAMES):
            raise ValueError("consequence values must have shape [B,K,6]")
        if self.log_variance.shape != self.values.shape:
            raise ValueError("consequence log_variance shape differs from values")
        return self


@dataclass(frozen=True)
class ConsequenceTargets:
    """Physical labels and availability mask ``[B,K,6]``."""

    values: torch.Tensor
    valid_mask: torch.Tensor

    def validate(self) -> "ConsequenceTargets":
        if self.values.ndim != 3 or self.values.shape[-1] != len(CONSEQUENCE_NAMES):
            raise ValueError("consequence targets must have shape [B,K,6]")
        if self.valid_mask.shape != self.values.shape:
            raise ValueError("consequence valid_mask shape differs from targets")
        if self.valid_mask.dtype != torch.bool:
            raise ValueError("consequence valid_mask must be bool")
        return self


class PlanningConsequenceHead(nn.Module):
    """Decode physical outcomes from perturbed-trajectory readout.

    The head is a representation-training auxiliary only. It exposes no
    inference-time ranking, selection, or aggregate metric API.
    """

    def __init__(self, context_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if min(context_dim, hidden_dim) <= 0:
            raise ValueError("consequence head dimensions must be positive")
        self.context_dim = int(context_dim)
        self.backbone = nn.Sequential(
            nn.LayerNorm(self.context_dim),
            nn.Linear(self.context_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
        )
        self.value_head = nn.Linear(int(hidden_dim), len(CONSEQUENCE_NAMES))
        self.uncertainty_head = nn.Linear(int(hidden_dim), len(CONSEQUENCE_NAMES))

    def forward(
        self,
        trajectory_context: torch.Tensor,
        waypoint_valid_mask: torch.Tensor,
    ) -> ConsequencePrediction:
        """Aggregate ``[B,K,H,C]`` readout into component predictions."""

        if trajectory_context.ndim != 4 or trajectory_context.shape[-1] != self.context_dim:
            raise ValueError("trajectory_context must have shape [B,K,H,C]")
        if waypoint_valid_mask.shape != trajectory_context.shape[:3]:
            raise ValueError("waypoint_valid_mask must have shape [B,K,H]")
        weights = waypoint_valid_mask.to(
            device=trajectory_context.device, dtype=trajectory_context.dtype
        )
        pooled = (trajectory_context * weights[..., None]).sum(dim=2) / weights.sum(
            dim=2, keepdim=True
        ).clamp_min(1.0)
        hidden = self.backbone(pooled)
        return ConsequencePrediction(
            self.value_head(hidden),
            self.uncertainty_head(hidden).clamp(-8.0, 8.0),
        ).validate()


def consequence_losses(
    prediction: ConsequencePrediction,
    targets: ConsequenceTargets,
    scales: Optional[torch.Tensor | Sequence[float]] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Return scaled losses and physical-unit diagnostics for ``[B,K,6]``."""

    prediction.validate()
    targets.validate()
    if prediction.values.shape != targets.values.shape:
        raise ValueError("consequence prediction and target shapes differ")
    expected = targets.values.to(device=prediction.values.device, dtype=torch.float32)
    valid = targets.valid_mask.to(device=prediction.values.device)
    predicted = prediction.values.float()
    log_variance = prediction.log_variance.float()
    if scales is None:
        scale_tensor = torch.ones(
            len(CONSEQUENCE_NAMES), device=predicted.device, dtype=torch.float32
        )
    else:
        scale_tensor = torch.as_tensor(
            scales, device=predicted.device, dtype=torch.float32
        )
    if scale_tensor.shape != (len(CONSEQUENCE_NAMES),):
        raise ValueError("consequence scales must have shape [6]")
    if not torch.isfinite(scale_tensor).all() or (scale_tensor <= 0).any():
        raise ValueError("consequence scales must be finite and positive")
    losses: Dict[str, torch.Tensor] = {}
    metrics: Dict[str, torch.Tensor] = {
        "consequence_valid_ratio": valid.float().mean().detach(),
    }
    for index, name in enumerate(CONSEQUENCE_NAMES):
        mask = valid[..., index]
        if name == "collision":
            pointwise = F.binary_cross_entropy_with_logits(
                predicted[..., index],
                expected[..., index],
                reduction="none",
            )
            physical_prediction = predicted[..., index].sigmoid()
        else:
            normalized_residual = (
                predicted[..., index] - expected[..., index]
            ) / scale_tensor[index]
            regression = F.smooth_l1_loss(
                normalized_residual,
                torch.zeros_like(normalized_residual),
                reduction="none",
            )
            pointwise = (
                torch.exp(-log_variance[..., index]) * regression
                + 0.5 * log_variance[..., index]
            )
            physical_prediction = predicted[..., index]
        weight = mask.to(dtype=pointwise.dtype)
        losses[f"consequence_{name}"] = (pointwise * weight).sum() / weight.sum().clamp_min(1.0)
        denominator = weight.sum().clamp_min(1.0)
        target_value = expected[..., index]
        prediction_mean = (physical_prediction * weight).sum() / denominator
        target_mean = (target_value * weight).sum() / denominator
        prediction_variance = (
            (physical_prediction - prediction_mean).square() * weight
        ).sum() / denominator
        target_variance = (
            (target_value - target_mean).square() * weight
        ).sum() / denominator
        metrics[f"consequence_{name}_valid_ratio"] = weight.mean().detach()
        metrics[f"consequence_{name}_mae"] = (
            (physical_prediction - target_value).abs() * weight
        ).sum().div(denominator).detach()
        metrics[f"consequence_{name}_prediction_std"] = (
            prediction_variance.clamp_min(0.0).sqrt().detach()
        )
        metrics[f"consequence_{name}_target_std"] = (
            target_variance.clamp_min(0.0).sqrt().detach()
        )
        if name == "collision":
            correct = ((physical_prediction >= 0.5) == (target_value >= 0.5)).to(
                dtype=weight.dtype
            )
            metrics["consequence_collision_accuracy"] = (
                (correct * weight).sum() / denominator
            ).detach()
    return losses, metrics
