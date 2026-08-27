# SPDX-License-Identifier: Apache-2.0
"""CLOVER set-coverage and conservative self-distillation objectives.

The implementation follows the equations in CLOVER (arXiv:2605.15120v2),
not the incomplete public preview loss.  In particular, it keeps the scalar
Top-k set term, vector-Pareto set term, and register-aligned stability term as
three separate objectives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn

from starVLA.model.modules.trajectory_scorer.losses import DRIVOR_METRICS


def _validate_trajectories(value: Tensor, name: str) -> None:
    if value.ndim != 4 or tuple(value.shape[-2:]) != (8, 3):
        raise ValueError(f"{name} must have shape [B,K,8,3]")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or Inf")


def set_coverage_l1(
    proposals: Tensor,
    targets: Tensor,
    target_mask: Tensor | None = None,
) -> Tensor:
    """Mean target-to-set L1 distance from CLOVER equations (4) and (6)."""

    _validate_trajectories(proposals, "proposals")
    _validate_trajectories(targets, "targets")
    if proposals.shape[0] != targets.shape[0]:
        raise ValueError("proposal and target batch sizes differ")
    if target_mask is None:
        target_mask = torch.ones(
            targets.shape[:2], dtype=torch.bool, device=targets.device
        )
    if target_mask.dtype is not torch.bool or tuple(target_mask.shape) != tuple(
        targets.shape[:2]
    ):
        raise ValueError("target_mask must be boolean [B,M]")
    # [B,K,M,8,3] -> [B,K,M], then match every target to its nearest proposal.
    distance = torch.linalg.vector_norm(
        proposals[:, :, None] - targets[:, None], ord=1, dim=-1
    ).mean(dim=-1)
    target_min = distance.amin(dim=1)
    valid = target_mask.to(dtype=target_min.dtype)
    per_scene = (target_min * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
    # A scene with no valid targets contributes zero instead of an arbitrary
    # padded-trajectory loss.
    per_scene = torch.where(target_mask.any(dim=1), per_scene, torch.zeros_like(per_scene))
    return per_scene.mean()


def winner_take_all_l1(proposals: Tensor, ground_truth: Tensor) -> Tensor:
    _validate_trajectories(proposals, "proposals")
    if ground_truth.ndim != 3 or tuple(ground_truth.shape[-2:]) != (8, 3):
        raise ValueError("ground_truth must have shape [B,8,3]")
    error = torch.linalg.vector_norm(
        proposals - ground_truth[:, None], ord=1, dim=-1
    ).mean(dim=-1)
    return error.amin(dim=1).mean()


@dataclass
class CloverStage1LossOutput:
    loss: Tensor
    gt_loss: Tensor
    pseudo_expert_loss: Tensor


class CloverStage1TrajectoryLoss(nn.Module):
    """Logged-trajectory prior plus evaluator-filtered pseudo-expert coverage."""

    def __init__(self, *, gt_weight: float = 1.0, pseudo_expert_weight: float = 0.5):
        super().__init__()
        if gt_weight < 0 or pseudo_expert_weight < 0:
            raise ValueError("CLOVER Stage-1 loss weights must be non-negative")
        self.gt_weight = float(gt_weight)
        self.pseudo_expert_weight = float(pseudo_expert_weight)

    def forward(
        self,
        proposals: Tensor,
        ground_truth: Tensor,
        pseudo_experts: Tensor,
        pseudo_expert_mask: Tensor,
    ) -> CloverStage1LossOutput:
        gt_loss = winner_take_all_l1(proposals, ground_truth)
        pseudo_expert_loss = set_coverage_l1(
            proposals, pseudo_experts, pseudo_expert_mask
        )
        return CloverStage1LossOutput(
            loss=self.gt_weight * gt_loss
            + self.pseudo_expert_weight * pseudo_expert_loss,
            gt_loss=gt_loss,
            pseudo_expert_loss=pseudo_expert_loss,
        )


@dataclass
class TeacherTargetSets:
    topk_trajectories: Tensor
    topk_mask: Tensor
    topk_indices: Tensor
    pareto_trajectories: Tensor
    pareto_mask: Tensor
    pareto_indices: Tensor
    predicted_rewards: Tensor
    predicted_aggregate: Tensor


def _pareto_mask(rewards: Tensor) -> Tensor:
    """Return non-dominated candidates for maximization in every component."""

    if rewards.ndim != 3:
        raise ValueError("reward vectors must have shape [B,K,M]")
    # dominates[b,j,i] iff j is no worse than i in every metric and strictly
    # better in at least one. Equal vectors therefore remain non-dominated.
    lhs = rewards[:, :, None, :]
    rhs = rewards[:, None, :, :]
    dominates = (lhs >= rhs).all(dim=-1) & (lhs > rhs).any(dim=-1)
    return ~dominates.any(dim=1)


def build_teacher_target_sets(
    teacher_proposals: Tensor,
    metric_logits: Mapping[str, Tensor],
    predicted_aggregate: Tensor,
    *,
    topk: int = 8,
    pareto_max_size: int = 8,
    pareto_min_size: int = 2,
    reward_threshold: float = 0.4,
) -> TeacherTargetSets:
    """Construct scalar Top-k and vector-Pareto teacher sets.

    Pareto candidates are ordered by the composed predicted score. Low mean
    reward candidates are removed only when at least ``pareto_min_size`` valid
    alternatives remain; the minimum is then filled from scalar ranking. This
    matches CLOVER's conservative fallback instead of returning an empty set.
    """

    _validate_trajectories(teacher_proposals, "teacher_proposals")
    batch_size, candidate_count = teacher_proposals.shape[:2]
    if predicted_aggregate.shape != (batch_size, candidate_count):
        raise ValueError("predicted_aggregate must have shape [B,K]")
    missing = set(DRIVOR_METRICS).difference(metric_logits)
    if missing:
        raise KeyError(f"teacher logits are missing {sorted(missing)}")
    if not 1 <= topk <= candidate_count:
        raise ValueError("topk must lie inside the teacher candidate count")
    if not 1 <= pareto_min_size <= pareto_max_size <= candidate_count:
        raise ValueError("invalid Pareto set size bounds")

    rewards = torch.stack(
        [torch.sigmoid(metric_logits[name].float()) for name in DRIVOR_METRICS],
        dim=-1,
    )
    nondominated = _pareto_mask(rewards)
    scalar_order = predicted_aggregate.float().argsort(dim=1, descending=True)
    topk_indices = scalar_order[:, :topk]
    topk_trajectories = torch.gather(
        teacher_proposals,
        1,
        topk_indices[:, :, None, None].expand(-1, -1, 8, 3),
    )
    topk_mask = torch.ones(
        (batch_size, topk), dtype=torch.bool, device=teacher_proposals.device
    )

    pareto_indices = torch.zeros(
        (batch_size, pareto_max_size),
        dtype=torch.long,
        device=teacher_proposals.device,
    )
    pareto_valid = torch.zeros(
        (batch_size, pareto_max_size),
        dtype=torch.bool,
        device=teacher_proposals.device,
    )
    reward_mean = rewards.mean(dim=-1)
    for batch_index in range(batch_size):
        ordered = scalar_order[batch_index]
        selected = [
            int(index)
            for index in ordered.tolist()
            if bool(nondominated[batch_index, index])
            and float(reward_mean[batch_index, index]) >= reward_threshold
        ]
        if len(selected) < pareto_min_size:
            # Keep unthresholded non-dominated modes before scalar fallbacks.
            selected = [
                int(index)
                for index in ordered.tolist()
                if bool(nondominated[batch_index, index])
            ]
        used = set(selected)
        if len(selected) < pareto_min_size:
            for index in ordered.tolist():
                index = int(index)
                if index not in used:
                    selected.append(index)
                    used.add(index)
                if len(selected) >= pareto_min_size:
                    break
        selected = selected[:pareto_max_size]
        count = len(selected)
        pareto_indices[batch_index, :count] = torch.tensor(
            selected, dtype=torch.long, device=teacher_proposals.device
        )
        pareto_valid[batch_index, :count] = True

    pareto_trajectories = torch.gather(
        teacher_proposals,
        1,
        pareto_indices[:, :, None, None].expand(-1, -1, 8, 3),
    )
    return TeacherTargetSets(
        topk_trajectories=topk_trajectories,
        topk_mask=topk_mask,
        topk_indices=topk_indices,
        pareto_trajectories=pareto_trajectories,
        pareto_mask=pareto_valid,
        pareto_indices=pareto_indices,
        predicted_rewards=rewards,
        predicted_aggregate=predicted_aggregate,
    )


@dataclass
class CloverStage2LossOutput:
    loss: Tensor
    trajectory_loss: Tensor
    gt_loss: Tensor
    diversity_loss: Tensor
    topk_loss: Tensor
    pareto_loss: Tensor
    stability_loss: Tensor


def clover_inter_trajectory_loss(proposals: Tensor) -> Tensor:
    """CLOVER/DrivoR's released closest-pair diversity objective.

    The donor names this term ``inter_loss``.  It is a negative closest-pair
    L1 distance over complete trajectories, not an intermediate-decoder loss.
    The exact zero replacement is retained for checkpoint/recipe fidelity.
    """

    _validate_trajectories(proposals, "proposals")
    distance = torch.linalg.vector_norm(
        proposals[:, :, None] - proposals[:, None], ord=1, dim=-1
    ).mean(dim=-1)
    distance = distance + (distance == 0).to(distance.dtype)
    return -distance.amin(dim=1).amin(dim=1).mean()


class CloverStage2GeneratorLoss(nn.Module):
    """CLOVER equation (7), including index-aligned teacher stability."""

    def __init__(
        self,
        *,
        trajectory_weight: float = 0.1,
        diversity_weight: float = 0.02,
        topk_weight: float = 1.0,
        pareto_weight: float = 1.0,
        stability_weight: float = 0.05,
    ) -> None:
        super().__init__()
        values = (
            trajectory_weight,
            diversity_weight,
            topk_weight,
            pareto_weight,
            stability_weight,
        )
        if any(value < 0 for value in values):
            raise ValueError("CLOVER Stage-2 weights must be non-negative")
        self.trajectory_weight = float(trajectory_weight)
        self.diversity_weight = float(diversity_weight)
        self.topk_weight = float(topk_weight)
        self.pareto_weight = float(pareto_weight)
        self.stability_weight = float(stability_weight)

    def forward(
        self,
        proposal_list: Sequence[Tensor],
        ground_truth: Tensor,
        teacher_proposals: Tensor,
        target_sets: TeacherTargetSets,
    ) -> CloverStage2LossOutput:
        if not proposal_list:
            raise ValueError("proposal_list cannot be empty")
        final = proposal_list[-1]
        _validate_trajectories(final, "student proposals")
        _validate_trajectories(teacher_proposals, "teacher proposals")
        if final.shape != teacher_proposals.shape:
            raise ValueError("teacher and student register layouts must match")
        gt_loss = winner_take_all_l1(final, ground_truth)
        diversity_loss = clover_inter_trajectory_loss(final)
        trajectory_loss = gt_loss + self.diversity_weight * diversity_loss
        topk_loss = set_coverage_l1(
            final, target_sets.topk_trajectories, target_sets.topk_mask
        )
        pareto_loss = set_coverage_l1(
            final, target_sets.pareto_trajectories, target_sets.pareto_mask
        )
        # Equation (7) is a mean over corresponding learned registers. Using
        # amin(K) here (as in the public preview) removes the trust region for
        # 63/64 registers and permits proposal collapse.
        stability_loss = torch.linalg.vector_norm(
            final - teacher_proposals.detach(), ord=1, dim=-1
        ).mean(dim=(-1, -2)).mean()
        total = (
            self.trajectory_weight * trajectory_loss
            + self.topk_weight * topk_loss
            + self.pareto_weight * pareto_loss
            + self.stability_weight * stability_loss
        )
        return CloverStage2LossOutput(
            loss=total,
            trajectory_loss=trajectory_loss,
            gt_loss=gt_loss,
            diversity_loss=diversity_loss,
            topk_loss=topk_loss,
            pareto_loss=pareto_loss,
            stability_loss=stability_loss,
        )


def selected_set_enrichment(
    target_sets: TeacherTargetSets,
    true_aggregate: Tensor,
) -> dict[str, Tensor]:
    """Measure the condition required by CLOVER's conservative-update proof."""

    if true_aggregate.shape != target_sets.predicted_aggregate.shape:
        raise ValueError("true aggregate scores must match teacher candidates")
    topk_true = torch.gather(true_aggregate, 1, target_sets.topk_indices)
    pareto_true = torch.gather(true_aggregate, 1, target_sets.pareto_indices)
    pareto_valid = target_sets.pareto_mask.to(pareto_true.dtype)
    pareto_mean = (pareto_true * pareto_valid).sum(dim=1) / pareto_valid.sum(
        dim=1
    ).clamp_min(1.0)
    pool_mean = true_aggregate.mean(dim=1)
    return {
        "pool_true_mean": pool_mean.mean(),
        "topk_true_mean": topk_true.mean(),
        "pareto_true_mean": pareto_mean.mean(),
        "topk_enrichment": (topk_true.mean(dim=1) - pool_mean).mean(),
        "pareto_enrichment": (pareto_mean - pool_mean).mean(),
        "topk_enriched_scene_ratio": (
            topk_true.mean(dim=1) > pool_mean
        ).float().mean(),
        "pareto_enriched_scene_ratio": (pareto_mean > pool_mean).float().mean(),
    }
