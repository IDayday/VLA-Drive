"""Losses used by the staged action-effect research probes."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConsequencePredictionLoss(nn.Module):
    """Balanced hard-safety classification plus robust soft regression."""

    def __init__(
        self,
        hard_dim: int,
        *,
        hard_weight: float = 1.0,
        soft_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if hard_dim < 1:
            raise ValueError("hard_dim must be positive")
        self.hard_dim = hard_dim
        self.hard_weight = float(hard_weight)
        self.soft_weight = float(soft_weight)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        soft_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if prediction.shape != target.shape:
            raise ValueError(f"prediction/target mismatch: {prediction.shape} vs {target.shape}")
        hard = F.binary_cross_entropy_with_logits(
            prediction[:, : self.hard_dim], target[:, : self.hard_dim]
        )
        soft_prediction = prediction[:, self.hard_dim :]
        soft_target = target[:, self.hard_dim :]
        if soft_prediction.numel() == 0:
            soft = prediction.new_zeros(())
        elif soft_mask is None:
            soft = F.smooth_l1_loss(soft_prediction, soft_target)
        else:
            expanded = soft_mask.to(dtype=torch.bool, device=prediction.device)
            if expanded.ndim == 1:
                expanded = expanded.unsqueeze(0).expand_as(soft_prediction)
            soft = (
                F.smooth_l1_loss(soft_prediction[expanded], soft_target[expanded])
                if expanded.any()
                else prediction.new_zeros(())
            )
        total = self.hard_weight * hard + self.soft_weight * soft
        return {"total": total, "hard": hard, "soft": soft}


class StructuredFutureLoss(nn.Module):
    """Raster loss separating binary fields, occupied velocity, and clearance."""

    def __init__(self, binary_positive_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        if binary_positive_weight is not None:
            self.register_buffer(
                "binary_positive_weight",
                binary_positive_weight.detach().float().reshape(1, 1, 4, 1, 1),
            )
        else:
            self.binary_positive_weight = None

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        if prediction.shape != target.shape or prediction.ndim != 5 or prediction.shape[2] != 7:
            raise ValueError("structured prediction/target must match [B,H,7,R,R]")
        binary = F.binary_cross_entropy_with_logits(
            prediction[:, :, :4],
            target[:, :, :4],
            pos_weight=self.binary_positive_weight,
        )
        occupancy = target[:, :, 3:4] > 0.5
        velocity_mask = occupancy.expand(-1, -1, 2, -1, -1)
        velocity = (
            F.smooth_l1_loss(prediction[:, :, 4:6][velocity_mask], target[:, :, 4:6][velocity_mask])
            if velocity_mask.any()
            else prediction.new_zeros(())
        )
        clearance = F.smooth_l1_loss(prediction[:, :, 6], target[:, :, 6])
        total = binary + velocity + clearance
        return {
            "total": total,
            "binary": binary,
            "velocity": velocity,
            "clearance": clearance,
        }
