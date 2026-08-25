"""Winner-take-all Register generator loss and diversity diagnostics."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
from torch import Tensor, nn

from .outputs import RegisterLossOutput


def _validate_proposal(proposal: Tensor, ground_truth: Tensor) -> None:
    if proposal.ndim != 4 or tuple(proposal.shape[-2:]) != (8, 3):
        raise ValueError("each proposal stage must have shape [B,K,8,3]")
    if ground_truth.ndim != 3 or tuple(ground_truth.shape[-2:]) != (8, 3):
        raise ValueError("gt_trajectory must have shape [B,8,3]")
    if proposal.shape[0] != ground_truth.shape[0]:
        raise ValueError("proposal and ground truth batch sizes differ")


def _wta_l1(proposal: Tensor, ground_truth: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    per_pose_error = torch.linalg.norm(
        proposal - ground_truth[:, None], ord=1, dim=-1
    )
    candidate_error = per_pose_error.mean(dim=-1)
    min_error, winner_index = candidate_error.min(dim=1)
    return min_error.mean(), winner_index, candidate_error


def _pairwise_geometry(proposals: Tensor) -> tuple[Tensor, Tensor]:
    candidate_count = proposals.shape[1]
    if candidate_count < 2:
        zero = proposals.new_zeros(())
        return zero, zero
    xy = proposals[..., :2]
    pairwise = torch.linalg.vector_norm(
        xy[:, :, None] - xy[:, None, :], ord=2, dim=-1
    )
    rows, cols = torch.triu_indices(
        candidate_count, candidate_count, offset=1, device=proposals.device
    )
    return (
        pairwise[:, rows, cols].mean(dim=(-1, -2)).mean(),
        pairwise[:, rows, cols, -1].mean(),
    )


def _usage_metrics(winner_index: Tensor, candidate_count: int) -> dict[str, Tensor]:
    histogram = torch.bincount(winner_index, minlength=candidate_count).to(
        dtype=torch.float32, device=winner_index.device
    )
    probabilities = histogram / histogram.sum().clamp_min(1.0)
    nonzero = probabilities > 0
    entropy = -(probabilities[nonzero] * probabilities[nonzero].log()).sum()
    normalized_entropy = entropy / math.log(candidate_count) if candidate_count > 1 else entropy
    return {
        "register_usage_histogram": histogram,
        "register_usage_entropy": normalized_entropy,
        "active_register_ratio": nonzero.float().mean(),
        "top1_register_fraction": probabilities.max(),
    }


class RegisterTrajectoryLoss(nn.Module):
    """Donor-style final-stage min-L1 objective.

    Diversity values are diagnostics only and never enter the optimization
    objective.  A non-zero diversity weight is rejected instead of silently
    changing the production loss.
    """

    def __init__(
        self,
        *,
        stage_loss_mode: str = "final_only",
        stage_loss_weights: Optional[Sequence[float]] = None,
        diversity_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if stage_loss_mode not in {"final_only", "all_layers"}:
            raise ValueError("stage_loss_mode must be 'final_only' or 'all_layers'")
        if float(diversity_weight) != 0.0:
            raise ValueError("diversity regularization is not part of Register64 v1")
        self.stage_loss_mode = stage_loss_mode
        self.stage_loss_weights = (
            tuple(float(value) for value in stage_loss_weights)
            if stage_loss_weights is not None
            else (0.1, 0.2, 0.3, 0.5, 1.0)
        )
        self.diversity_weight = 0.0

    def forward(
        self, proposal_list: Sequence[Tensor], gt_trajectory: Tensor
    ) -> RegisterLossOutput:
        if not proposal_list:
            raise ValueError("proposal_list must contain at least one stage")
        proposals = list(proposal_list)
        for proposal in proposals:
            _validate_proposal(proposal, gt_trajectory)
        if any(proposal.shape != proposals[-1].shape for proposal in proposals):
            raise ValueError("all proposal stages must share one shape")

        final_loss, winner_index, _ = _wta_l1(proposals[-1], gt_trajectory)
        if self.stage_loss_mode == "final_only":
            stage_losses = [final_loss]
            total_loss = final_loss
        else:
            if len(self.stage_loss_weights) != len(proposals):
                raise ValueError(
                    "all_layers stage_loss_weights must match proposal_list length"
                )
            if any(weight < 0.0 for weight in self.stage_loss_weights):
                raise ValueError("stage loss weights must be non-negative")
            if sum(self.stage_loss_weights) <= 0.0:
                raise ValueError("at least one stage loss weight must be positive")
            stage_losses = [_wta_l1(stage, gt_trajectory)[0] for stage in proposals]
            total_loss = sum(
                weight * loss
                for weight, loss in zip(self.stage_loss_weights, stage_losses)
            )

        final = proposals[-1]
        xy_errors = torch.linalg.vector_norm(
            final[..., :2] - gt_trajectory[:, None, :, :2], ord=2, dim=-1
        )
        ade = xy_errors.mean(dim=-1)
        fde = xy_errors[..., -1]
        min_ade, min_ade_index = ade.min(dim=1)
        min_fde = fde.min(dim=1).values
        pairwise_ade, pairwise_fde = _pairwise_geometry(final)
        endpoints = final[:, :, -1, :2]
        endpoint_std = endpoints.std(dim=1, unbiased=False).mean()
        centered = endpoints - endpoints.mean(dim=1, keepdim=True)
        covariance = torch.einsum("bki,bkj->bij", centered, centered) / max(
            final.shape[1] - 1, 1
        )
        endpoint_covariance = covariance.diagonal(dim1=-2, dim2=-1).sum(-1).mean()
        rows = torch.arange(final.shape[0], device=final.device)
        metrics = {
            "min_ade_1": ade[:, 0].mean(),
            "min_ade_64": min_ade.mean(),
            "min_fde_1": fde[:, 0].mean(),
            "min_fde_64": min_fde.mean(),
            "winner_ade": ade[rows, winner_index].mean(),
            "ade_winner_matches_l1_winner": (min_ade_index == winner_index).float().mean(),
            "pairwise_ade": pairwise_ade,
            "pairwise_fde": pairwise_fde,
            "endpoint_std": endpoint_std,
            "endpoint_covariance": endpoint_covariance,
            **_usage_metrics(winner_index, final.shape[1]),
        }
        return RegisterLossOutput(
            loss=total_loss,
            winner_index=winner_index,
            metrics=metrics,
            stage_losses=stage_losses,
        )
