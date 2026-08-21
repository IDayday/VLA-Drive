# SPDX-License-Identifier: Apache-2.0
# DrivoR portions adapted from valeoai/DrivoR commit
# f02665403df799c1b4ddd8b0d34e073f0555c13a, file
# navsim/agents/drivoR/layers/losses/drivor_loss.py.
# DriveSuprim portions adapted from William-Yao-2000/DriveSuprim commit
# 80fe792d7654a596d92e20d030d1650f6f605c02, file
# navsim/agents/drivesuprim/drivesuprim_loss_fn.py.
# Compatibility changes: operates on externally supplied detached candidates,
# removes donor generator/diversity terms, supports batch-specific joint
# candidates, and intentionally adds no static/dynamic source weighting.

"""Official sub-score and imitation losses used by DDP-DRS."""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .trajectory_resampler import STATIC_SAMPLE_INDICES


DRIVOR_SCORE_NAMES = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "time_to_collision_within_bound",
    "ego_progress",
    "driving_direction_compliance",
    "comfort",
)

SUPRIM_SCORE_NAMES = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "time_to_collision_within_bound",
    "ego_progress",
    "driving_direction_compliance",
    "lane_keeping",
    "traffic_light_compliance",
    "history_comfort",
)

SUPRIM_PDM_WEIGHTS = {
    "no_at_fault_collisions": 3.0,
    "drivable_area_compliance": 3.0,
    "time_to_collision_within_bound": 4.0,
    "ego_progress": 2.0,
    "driving_direction_compliance": 1.0,
    "lane_keeping": 2.0,
    "traffic_light_compliance": 3.0,
    "history_comfort": 1.0,
}


def three_to_two_classes(target: Tensor) -> Tensor:
    """Official 0.5 -> 0 conversion without mutating the cache tensor."""

    return torch.where(target == 0.5, torch.zeros_like(target), target)


class DrivoRSubScoreLoss(nn.Module):
    """Six official DrivoR BCE losses, including masked TTC supervision."""

    def forward(
        self,
        predictions: Mapping[str, Tensor],
        targets: Mapping[str, Tensor],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        missing_predictions = set(DRIVOR_SCORE_NAMES).difference(predictions)
        missing_targets = set(DRIVOR_SCORE_NAMES).difference(targets)
        if missing_predictions or missing_targets:
            raise KeyError(
                "DrivoR loss keys missing: "
                f"predictions={sorted(missing_predictions)}, "
                f"targets={sorted(missing_targets)}"
            )
        dtype = predictions["comfort"].dtype
        device = predictions["comfort"].device

        def target(name: str) -> Tensor:
            value = targets[name].to(device=device, dtype=dtype)
            if value.shape != predictions[name].shape:
                raise ValueError(
                    f"DrivoR target {name} shape {tuple(value.shape)} does not "
                    f"match prediction {tuple(predictions[name].shape)}"
                )
            return value

        dac = F.binary_cross_entropy_with_logits(
            predictions["drivable_area_compliance"],
            target("drivable_area_compliance"),
        )
        ttc_target = target("time_to_collision_within_bound")
        valid_ttc = (ttc_target != 2.0).to(dtype)
        ttc_elementwise = F.binary_cross_entropy_with_logits(
            predictions["time_to_collision_within_bound"],
            ttc_target,
            reduction="none",
        )
        ttc = (ttc_elementwise * valid_ttc).sum() / valid_ttc.sum().clamp_min(1.0)
        noc = F.binary_cross_entropy_with_logits(
            predictions["no_at_fault_collisions"],
            three_to_two_classes(target("no_at_fault_collisions")),
        )
        progress = F.binary_cross_entropy_with_logits(
            predictions["ego_progress"], target("ego_progress")
        )
        ddc = F.binary_cross_entropy_with_logits(
            predictions["driving_direction_compliance"],
            three_to_two_classes(target("driving_direction_compliance")),
        )
        comfort = F.binary_cross_entropy_with_logits(
            predictions["comfort"], target("comfort")
        )
        components = {
            "dac_loss": dac,
            "ttc_loss": ttc,
            "noc_loss": noc,
            "progress_loss": progress,
            "ddc_loss": ddc,
            "comfort_loss": comfort,
        }
        # DrivoR's official scorer loss assigns unit weight to all six terms.
        total = torch.stack(tuple(components.values())).sum()
        return total, components


def _suprim_sub_score_losses(
    predictions: Mapping[str, Tensor],
    targets: Mapping[str, Tensor],
    weights: Mapping[str, float],
) -> Tuple[Tensor, Dict[str, Tensor]]:
    missing_predictions = set(SUPRIM_SCORE_NAMES).difference(predictions)
    missing_targets = set(SUPRIM_SCORE_NAMES).difference(targets)
    if missing_predictions or missing_targets:
        raise KeyError(
            "DriveSuprim loss keys missing: "
            f"predictions={sorted(missing_predictions)}, "
            f"targets={sorted(missing_targets)}"
        )
    reference = predictions["drivable_area_compliance"]
    components: Dict[str, Tensor] = {}
    for name in SUPRIM_SCORE_NAMES:
        target = targets[name].to(device=reference.device, dtype=reference.dtype)
        if target.shape != predictions[name].shape:
            raise ValueError(
                f"DriveSuprim target {name} shape {tuple(target.shape)} does not "
                f"match prediction {tuple(predictions[name].shape)}"
            )
        if name in {"no_at_fault_collisions", "driving_direction_compliance"}:
            target = three_to_two_classes(target)
        components[f"pdm_{name}_loss"] = weights[name] * F.binary_cross_entropy_with_logits(
            predictions[name], target
        )
    return torch.stack(tuple(components.values())).sum(), components


def _batch_candidate_trajectories(candidate_trajectories: Tensor, batch_size: int) -> Tensor:
    if candidate_trajectories.ndim == 3:
        return candidate_trajectories.unsqueeze(0).expand(batch_size, -1, -1, -1)
    if candidate_trajectories.ndim == 4 and candidate_trajectories.shape[0] == batch_size:
        return candidate_trajectories
    raise ValueError("candidate_trajectories must have shape [N,40,3] or [B,N,40,3]")


def drivesuprim_imitation_distribution_loss(
    imitation_logits: Tensor,
    candidate_trajectories: Tensor,
    target_trajectory_8: Tensor,
    sigma: float = 0.5,
) -> Tensor:
    """Official DriveSuprim soft imitation-distribution cross entropy."""

    if sigma <= 0:
        raise ValueError("DriveSuprim imitation sigma must be positive")
    if target_trajectory_8.ndim != 3 or target_trajectory_8.shape[-2:] != (8, 3):
        raise ValueError("target_trajectory_8 must have shape [B,8,3]")
    batch_size = target_trajectory_8.shape[0]
    candidates = _batch_candidate_trajectories(candidate_trajectories, batch_size)
    if candidates.shape[1] != imitation_logits.shape[1]:
        raise ValueError("imitation logits and candidate counts differ")
    sampled = candidates[..., list(STATIC_SAMPLE_INDICES), :]
    target = target_trajectory_8[:, None].to(device=sampled.device, dtype=sampled.dtype)
    l2_distance = -((sampled - target) ** 2) / sigma
    target_distribution = l2_distance.sum(dim=(-2, -1)).softmax(dim=1)
    return F.cross_entropy(imitation_logits, target_distribution)


class DriveSuprimLoss(nn.Module):
    """Official coarse and intermediate single-stage DriveSuprim losses."""

    def __init__(
        self,
        sigma: float = 0.5,
        imitation_weight: float = 1.0,
        pdm_weights: Optional[Mapping[str, float]] = None,
    ) -> None:
        super().__init__()
        self.sigma = sigma
        self.imitation_weight = imitation_weight
        self.pdm_weights = dict(SUPRIM_PDM_WEIGHTS if pdm_weights is None else pdm_weights)
        if set(self.pdm_weights) != set(SUPRIM_SCORE_NAMES):
            raise ValueError("DriveSuprim PDM weights must cover every official sub-score")

    def coarse_loss(
        self,
        predictions: Mapping[str, Tensor],
        score_targets: Mapping[str, Tensor],
        candidate_trajectories: Tensor,
        target_trajectory_8: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        pdm_loss, components = _suprim_sub_score_losses(
            predictions, score_targets, self.pdm_weights
        )
        if "imi" not in predictions:
            raise KeyError("DriveSuprim coarse predictions are missing imitation logits")
        imitation = self.imitation_weight * drivesuprim_imitation_distribution_loss(
            predictions["imi"],
            candidate_trajectories,
            target_trajectory_8,
            sigma=self.sigma,
        )
        components = dict(components)
        components["imi_loss"] = imitation
        return pdm_loss + imitation, components

    def refinement_loss(
        self,
        layer_predictions: Sequence[Mapping[str, Tensor]],
        layer_score_targets: Mapping[str, Tensor],
        top_trajectories: Tensor,
        target_trajectory_8: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        if not layer_predictions:
            raise ValueError("DriveSuprim refinement requires intermediate layer outputs")
        total = target_trajectory_8.new_zeros(())
        losses: Dict[str, Tensor] = {}
        for index, predictions in enumerate(layer_predictions, start=1):
            layer_loss, _ = _suprim_sub_score_losses(
                predictions, layer_score_targets, self.pdm_weights
            )
            if "imi" in predictions:
                layer_loss = layer_loss + self.imitation_weight * (
                    drivesuprim_imitation_distribution_loss(
                        predictions["imi"],
                        top_trajectories,
                        target_trajectory_8,
                        sigma=self.sigma,
                    )
                )
            total = total + layer_loss
            losses[f"layer_{index}"] = layer_loss
        return total, losses
