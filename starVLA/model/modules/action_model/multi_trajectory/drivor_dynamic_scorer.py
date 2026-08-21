# SPDX-License-Identifier: Apache-2.0
# Adapted from valeoai/DrivoR, commit f02665403df799c1b4ddd8b0d34e073f0555c13a:
#   navsim/agents/drivoR/transformer_decoder.py
#   navsim/agents/drivoR/score_module/scorer.py
#   navsim/agents/drivoR/drivor_model.py
# Compatibility changes: accepts detached external DDP proposals, replaces the
# donor's same-width scene attention with layer-local 256-query/2048-memory
# asymmetric attention, and exposes typed Top-M outputs.  The six heads and
# official aggregate-score formula are unchanged.

"""DrivoR dynamic scorer over Qwen-derived global scene memory."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .asymmetric_decoder import AsymmetricTransformerDecoder
from .candidate_types import DynamicScorerOutput
from .config import DrivoRConfig, PlanningConfig


class Scorer(nn.Module):
    """The six official DrivoR sub-score heads."""

    SCORE_NAMES = (
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "time_to_collision_within_bound",
        "ego_progress",
        "driving_direction_compliance",
        "comfort",
    )

    def __init__(self, planning_dim: int, ffn_dim: int) -> None:
        super().__init__()
        self.pred_score = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(planning_dim, ffn_dim),
                    nn.ReLU(),
                    nn.Linear(ffn_dim, 1),
                )
                for name in self.SCORE_NAMES
            }
        )

    def forward(self, score_states: Tensor) -> Dict[str, Tensor]:
        return {
            name: head(score_states).squeeze(-1)
            for name, head in self.pred_score.items()
        }


class DrivoRDynamicScorer(nn.Module):
    """Score K detached DDP proposals and return the global DrivoR Top-M."""

    def __init__(
        self,
        config: DrivoRConfig,
        ego_status_dim: int,
        *,
        planning_config: Optional[PlanningConfig] = None,
        scene_dim: int = 2048,
    ) -> None:
        super().__init__()
        if ego_status_dim <= 0:
            raise ValueError("ego_status_dim must be positive for DrivoR scoring")
        planning = planning_config or PlanningConfig()
        if scene_dim <= 0:
            raise ValueError("scene_dim must be positive")
        self.config = config
        self.planning_config = planning
        self.ego_status_dim = int(ego_status_dim)
        self.scene_dim = int(scene_dim)
        planning_dim = planning.planning_dim

        self.ego_encoder = nn.Linear(self.ego_status_dim, planning_dim)
        self.trajectory_pos_embed = nn.Sequential(
            nn.Linear(8 * 3, planning.ffn_dim),
            nn.ReLU(),
            nn.Linear(planning.ffn_dim, planning_dim),
        )
        self.scorer_decoder = AsymmetricTransformerDecoder(
            num_layers=config.scorer_layers,
            planning_dim=planning_dim,
            memory_dim=scene_dim,
            num_heads=planning.num_heads,
            ffn_dim=planning.ffn_dim,
            dropout=planning.dropout,
            return_intermediate=False,
        )
        self.metric_heads = Scorer(planning_dim, planning.ffn_dim)

    @staticmethod
    def aggregate_sub_scores(
        sub_scores: Dict[str, Tensor], config: DrivoRConfig
    ) -> Tensor:
        """Numerically stable form of the unchanged official DrivoR formula."""

        missing = set(Scorer.SCORE_NAMES).difference(sub_scores)
        if missing:
            raise KeyError(f"DrivoR sub-scores missing keys: {sorted(missing)}")
        weighted_additive = (
            config.ttc_weight
            * torch.sigmoid(sub_scores["time_to_collision_within_bound"])
            + config.ep_weight * torch.sigmoid(sub_scores["ego_progress"])
            + config.comfort_weight * torch.sigmoid(sub_scores["comfort"])
        )
        aggregate = (
            config.noc_weight * F.logsigmoid(sub_scores["no_at_fault_collisions"])
            + config.dac_weight
            * F.logsigmoid(sub_scores["drivable_area_compliance"])
            + torch.log(
                weighted_additive.clamp_min(
                    torch.finfo(weighted_additive.dtype).tiny
                )
            )
        )
        # The official default DDC coefficient is exactly zero.  Omitting the
        # multiplication avoids the undefined 0 * -inf case at saturated logits.
        if config.ddc_weight != 0.0:
            aggregate = aggregate + config.ddc_weight * F.logsigmoid(
                sub_scores["driving_direction_compliance"]
            )
        return aggregate

    def _last_ego_status(self, ego_status: Tensor) -> Tensor:
        if ego_status.ndim == 2:
            current = ego_status
        elif ego_status.ndim >= 3:
            current = ego_status[:, -1].flatten(start_dim=1)
        else:
            raise ValueError("ego_status must have shape [B, D] or [B, T, ...]")
        if current.shape[-1] != self.ego_status_dim:
            raise ValueError(
                f"ego_status has dimension {current.shape[-1]}, expected "
                f"{self.ego_status_dim}"
            )
        return current

    def forward(
        self,
        proposals: Tensor,
        global_scene_tokens: Tensor,
        ego_status: Tensor,
        topk: Optional[int] = None,
    ) -> DynamicScorerOutput:
        if proposals.ndim != 4 or proposals.shape[-2:] != (8, 3):
            raise ValueError("DrivoR proposals must have shape [B, K, 8, 3]")
        if (
            global_scene_tokens.ndim != 3
            or global_scene_tokens.shape[0] != proposals.shape[0]
            or global_scene_tokens.shape[-1] != self.scene_dim
        ):
            raise ValueError(
                f"global_scene_tokens must have shape [B, S, {self.scene_dim}]"
            )
        if ego_status is None:
            raise ValueError("DrivoR scoring requires the existing baseline ego state")
        if ego_status.shape[0] != proposals.shape[0]:
            raise ValueError("ego_status batch dimension does not match proposals")

        batch_size, candidate_count = proposals.shape[:2]
        requested_topk = self.config.dynamic_topk if topk is None else int(topk)
        if requested_topk <= 0 or requested_topk > candidate_count:
            raise ValueError(
                f"DrivoR Top-k {requested_topk} exceeds candidate count "
                f"{candidate_count}"
            )

        # This is the explicit boundary that prohibits scorer-guided generator
        # backpropagation while retaining gradients into scene memory.
        embedded_traj = self.trajectory_pos_embed(
            proposals.reshape(batch_size, candidate_count, 8 * 3).detach()
        )
        score_states = self.scorer_decoder(
            tgt=embedded_traj,
            memory=global_scene_tokens,
            memory_key_padding_mask=None,
        )
        ego_token = self.ego_encoder(
            self._last_ego_status(ego_status).to(
                device=score_states.device, dtype=score_states.dtype
            )
        )[:, None, :]
        score_states = score_states + ego_token
        if self.config.debug_validate_finite and not torch.isfinite(score_states).all():
            raise ValueError("DrivoR score states contain NaN or Inf")
        sub_scores = self.metric_heads(score_states)
        aggregate_score = self.aggregate_sub_scores(sub_scores, self.config)
        if not torch.isfinite(aggregate_score).all():
            raise ValueError("DrivoR aggregate score contains NaN or Inf")

        topk_indices = torch.topk(
            aggregate_score,
            k=requested_topk,
            dim=1,
            largest=True,
            sorted=True,
        ).indices
        gather_indices = topk_indices[..., None, None].expand(-1, -1, 8, 3)
        topk_trajectories = torch.gather(
            proposals.detach(), dim=1, index=gather_indices
        )
        return DynamicScorerOutput(
            sub_scores=sub_scores,
            aggregate_score=aggregate_score,
            topk_indices=topk_indices,
            topk_trajectories=topk_trajectories,
            score_states=score_states,
        )
