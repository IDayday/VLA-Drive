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
        reduction: str = "mean",
    ) -> dict[str, torch.Tensor]:
        if prediction.shape != target.shape:
            raise ValueError(f"prediction/target mismatch: {prediction.shape} vs {target.shape}")
        hard = F.binary_cross_entropy_with_logits(
            prediction[:, : self.hard_dim], target[:, : self.hard_dim], reduction="none"
        ).mean(dim=1)
        soft_prediction = prediction[:, self.hard_dim :]
        soft_target = target[:, self.hard_dim :]
        if soft_prediction.numel() == 0:
            soft = prediction.new_zeros(len(prediction))
        elif soft_mask is None:
            soft = F.smooth_l1_loss(
                soft_prediction, soft_target, reduction="none"
            ).mean(dim=1)
        else:
            expanded = soft_mask.to(dtype=torch.bool, device=prediction.device)
            if expanded.ndim == 1:
                expanded = expanded.unsqueeze(0).expand_as(soft_prediction)
            element = F.smooth_l1_loss(soft_prediction, soft_target, reduction="none")
            soft = (element * expanded).sum(dim=1) / expanded.sum(dim=1).clamp_min(1)
        total = self.hard_weight * hard + self.soft_weight * soft
        values = {"total": total, "hard": hard, "soft": soft}
        if reduction == "none":
            return values
        if reduction != "mean":
            raise ValueError(f"unsupported reduction: {reduction}")
        return {name: value.mean() for name, value in values.items()}


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


class EffectTubeLoss(nn.Module):
    """Channel-aware loss for the nine-channel trajectory-aligned effect tube."""

    def __init__(
        self,
        binary_positive_weight: torch.Tensor | None = None,
        *,
        occupancy_weight: float = 1.0,
        sdf_weight: float = 1.0,
        velocity_weight: float = 1.0,
        clearance_weight: float = 1.0,
        collision_weight: float = 1.0,
        footprint_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if binary_positive_weight is None:
            binary_positive_weight = torch.ones(3)
        if binary_positive_weight.numel() != 3:
            raise ValueError("effect-tube binary positive weight must have three values")
        self.register_buffer("binary_positive_weight", binary_positive_weight.detach().float())
        self.weights = {
            "occupancy": float(occupancy_weight),
            "sdf": float(sdf_weight),
            "velocity": float(velocity_weight),
            "clearance": float(clearance_weight),
            "collision": float(collision_weight),
            "footprint": float(footprint_weight),
        }

    @staticmethod
    def _reduce(value: torch.Tensor, reduction: str) -> torch.Tensor:
        if reduction == "none":
            return value
        if reduction == "mean":
            return value.mean()
        raise ValueError(f"unsupported reduction: {reduction}")

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        reduction: str = "mean",
    ) -> dict[str, torch.Tensor]:
        if prediction.shape != target.shape or prediction.ndim != 5 or prediction.shape[2] != 9:
            raise ValueError("effect prediction/target must match [B,H,9,R,R]")

        def binary_loss(channel: int, weight_index: int) -> torch.Tensor:
            loss = F.binary_cross_entropy_with_logits(
                prediction[:, :, channel],
                target[:, :, channel],
                pos_weight=self.binary_positive_weight[weight_index],
                reduction="none",
            )
            return loss.mean(dim=(1, 2, 3))

        occupancy = binary_loss(0, 0)
        sdf = F.smooth_l1_loss(
            prediction[:, :, 1:4], target[:, :, 1:4], reduction="none"
        ).mean(dim=(1, 2, 3, 4))
        velocity_error = F.smooth_l1_loss(
            prediction[:, :, 4:6], target[:, :, 4:6], reduction="none"
        )
        velocity_mask = (target[:, :, 0:1] > 0.5).expand_as(velocity_error)
        velocity = (
            (velocity_error * velocity_mask).sum(dim=(1, 2, 3, 4))
            / velocity_mask.sum(dim=(1, 2, 3, 4)).clamp_min(1)
        )
        clearance = F.smooth_l1_loss(
            prediction[:, :, 6], target[:, :, 6], reduction="none"
        ).mean(dim=(1, 2, 3))
        collision = binary_loss(7, 1)
        footprint_bce = binary_loss(8, 2)
        footprint_probability = torch.sigmoid(prediction[:, :, 8])
        intersection = (footprint_probability * target[:, :, 8]).sum(dim=(1, 2, 3))
        denominator = (
            footprint_probability.sum(dim=(1, 2, 3))
            + target[:, :, 8].sum(dim=(1, 2, 3))
        )
        footprint_dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
        footprint = footprint_bce + footprint_dice
        total = (
            self.weights["occupancy"] * occupancy
            + self.weights["sdf"] * sdf
            + self.weights["velocity"] * velocity
            + self.weights["clearance"] * clearance
            + self.weights["collision"] * collision
            + self.weights["footprint"] * footprint
        )
        values = {
            "total": total,
            "occupancy": occupancy,
            "sdf": sdf,
            "velocity": velocity,
            "clearance": clearance,
            "collision": collision,
            "footprint": footprint,
        }
        return {name: self._reduce(value, reduction) for name, value in values.items()}


def normalized_pair_loss(
    left_latent: torch.Tensor,
    right_latent: torch.Tensor,
    *,
    equivalent: torch.Tensor,
    divergent: torch.Tensor,
    consequence_distance: torch.Tensor,
    confidence_weight: torch.Tensor | None = None,
    base_margin: float = 0.5,
    margin_scale: float = 0.2,
    maximum_margin: float = 1.5,
    global_separation: bool = False,
) -> dict[str, torch.Tensor]:
    """Pair loss on L2-normalized effect latents.

    The function returns a per-pair loss so the caller can aggregate within
    scene before averaging across scenes. Ambiguous pairs have both masks false
    and therefore receive no AEE constraint.
    """

    if left_latent.shape != right_latent.shape or left_latent.ndim != 2:
        raise ValueError("pair latents must be aligned rank-2 tensors")
    count = len(left_latent)
    for value, name in (
        (equivalent, "equivalent"),
        (divergent, "divergent"),
        (consequence_distance, "consequence_distance"),
    ):
        if value.shape != (count,):
            raise ValueError(f"{name} must have shape [{count}]")
    left = F.normalize(left_latent, dim=-1)
    right = F.normalize(right_latent, dim=-1)
    distance = torch.linalg.vector_norm(left - right, dim=-1)
    margin = torch.clamp(
        base_margin + margin_scale * consequence_distance,
        min=base_margin,
        max=maximum_margin,
    )
    if global_separation:
        active = torch.ones_like(distance, dtype=torch.bool)
        loss = torch.relu(margin - distance).square()
    else:
        equivalent = equivalent.to(dtype=torch.bool)
        divergent = divergent.to(dtype=torch.bool)
        active = equivalent | divergent
        loss = torch.where(
            equivalent,
            distance.square(),
            torch.where(divergent, torch.relu(margin - distance).square(), torch.zeros_like(distance)),
        )
    if confidence_weight is not None:
        if confidence_weight.shape != (count,):
            raise ValueError(f"confidence weight must have shape [{count}]")
        loss = loss * confidence_weight
    return {
        "loss": loss,
        "distance": distance,
        "margin": margin,
        "active": active,
        "left_norm": torch.linalg.vector_norm(left_latent, dim=-1),
        "right_norm": torch.linalg.vector_norm(right_latent, dim=-1),
    }


def equal_scene_mean(
    values: torch.Tensor,
    scene_index: torch.Tensor,
    *,
    scene_count: int,
) -> torch.Tensor:
    """Average samples within scene, then give every represented scene equal weight."""

    if values.ndim != 1 or scene_index.shape != values.shape:
        raise ValueError("values and scene_index must be aligned vectors")
    sums = values.new_zeros(scene_count)
    counts = values.new_zeros(scene_count)
    sums.scatter_add_(0, scene_index, values)
    counts.scatter_add_(0, scene_index, torch.ones_like(values))
    represented = counts > 0
    if not represented.any():
        return values.new_zeros(())
    return (sums[represented] / counts[represented]).mean()
