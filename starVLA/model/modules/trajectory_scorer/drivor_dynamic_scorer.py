# SPDX-License-Identifier: Apache-2.0
# Adapted from valeoai/DrivoR commit
# f02665403df799c1b4ddd8b0d34e073f0555c13a, files
# navsim/agents/drivoR/transformer_decoder.py,
# navsim/agents/drivoR/score_module/scorer.py, and drivor_model.py.
# Project adaptations: accept Flow-DiT proposals, detach proposal geometry at
# the scorer boundary, and use asymmetric 256-query/scene-memory attention.

"""DrivoR dynamic-proposal pre-scorer without its trajectory generator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor, nn

from .attention import AsymmetricDecoder
from .losses import DRIVOR_METRICS, aggregate_drivor_score


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


@dataclass
class DynamicScorerOutput:
    """DrivoR scores and Top-M tensors aligned to original dynamic IDs."""

    metric_logits: Dict[str, Tensor]
    aggregate_score: Tensor
    topm_indices: Tensor
    topm_trajectories_8: Tensor
    topm_candidate_states: Tensor


class DrivoRDynamicScorer(nn.Module):
    """Score detached physical ``[B,K,8,3]`` Flow proposals."""

    def __init__(
        self,
        *,
        scene_dim: int = 2048,
        ego_state_dim: int = 4,
        model_dim: int = 256,
        ffn_dim: int = 1024,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.0,
        noc: float = 1.0,
        dac: float = 1.0,
        ddc: float = 0.0,
        ttc: float = 5.0,
        ep: float = 5.0,
        comfort: float = 2.0,
        debug_validate_finite: bool = False,
    ) -> None:
        super().__init__()
        if ego_state_dim <= 0 or model_dim <= 0:
            raise ValueError("ego_state_dim and model_dim must be positive")
        self.scene_dim = scene_dim
        self.ego_state_dim = ego_state_dim
        self.model_dim = model_dim
        self.debug_validate_finite = debug_validate_finite
        self.aggregate_weights = {
            "noc": noc,
            "dac": dac,
            "ddc": ddc,
            "ttc": ttc,
            "ep": ep,
            "comfort": comfort,
        }
        self.trajectory_embedding = _mlp(8 * 3, ffn_dim, model_dim)
        self.ego_encoder = _mlp(ego_state_dim, ffn_dim, model_dim)
        self.scorer_decoder = AsymmetricDecoder(
            num_layers=num_layers,
            query_dim=model_dim,
            memory_dim=scene_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            return_intermediate=False,
        )
        self.metric_heads = nn.ModuleDict(
            {name: _mlp(model_dim, ffn_dim, 1) for name in DRIVOR_METRICS}
        )

    def _encode_ego(self, ego_state: Tensor, batch_size: int, reference: Tensor) -> Tensor:
        if ego_state is None:
            raise ValueError("DrivoR scorer requires ego_state")
        if ego_state.ndim == 3 and ego_state.shape[1] == 1:
            ego_state = ego_state[:, 0]
        if ego_state.ndim != 2 or tuple(ego_state.shape) != (
            batch_size,
            self.ego_state_dim,
        ):
            raise ValueError(
                f"ego_state must have shape [B,{self.ego_state_dim}] or "
                f"[B,1,{self.ego_state_dim}], got {tuple(ego_state.shape)}"
            )
        return self.ego_encoder(
            ego_state.to(device=reference.device, dtype=reference.dtype)
        )[:, None]

    def forward(
        self,
        proposals_navsim: Tensor,
        global_scene_tokens: Tensor,
        ego_state: Tensor,
        *,
        topm: int = 32,
    ) -> DynamicScorerOutput:
        """Score proposals; gradients never propagate to proposal geometry."""

        if proposals_navsim.ndim != 4 or tuple(proposals_navsim.shape[-2:]) != (8, 3):
            raise ValueError("proposals_navsim must have shape [B,K,8,3]")
        batch_size, candidate_count = proposals_navsim.shape[:2]
        if topm <= 0 or topm > candidate_count:
            raise ValueError(
                f"topm must be in [1,{candidate_count}], got {topm}"
            )
        if global_scene_tokens.ndim != 3 or (
            global_scene_tokens.shape[0] != batch_size
            or global_scene_tokens.shape[-1] != self.scene_dim
        ):
            raise ValueError(
                f"global_scene_tokens must have shape [B,S,{self.scene_dim}]"
            )
        if proposals_navsim.device != global_scene_tokens.device:
            raise ValueError("proposals and global scene tokens must share a device")

        detached_proposals = proposals_navsim.detach().to(
            dtype=global_scene_tokens.dtype
        )
        candidate_states = self.trajectory_embedding(
            detached_proposals.reshape(batch_size, candidate_count, 8 * 3)
        )
        candidate_states = self.scorer_decoder(
            candidate_states, global_scene_tokens
        )
        candidate_states = candidate_states + self._encode_ego(
            ego_state, batch_size, candidate_states
        )
        metric_logits = {
            name: head(candidate_states).squeeze(-1)
            for name, head in self.metric_heads.items()
        }
        aggregate_score = aggregate_drivor_score(
            metric_logits, **self.aggregate_weights
        )
        if self.debug_validate_finite and (
            not torch.isfinite(candidate_states).all()
            or not torch.isfinite(aggregate_score).all()
        ):
            raise ValueError("DrivoR scorer produced NaN or Inf")
        _, topm_indices = torch.topk(aggregate_score, k=topm, dim=1)
        trajectory_index = topm_indices[..., None, None].expand(-1, -1, 8, 3)
        state_index = topm_indices[..., None].expand(-1, -1, self.model_dim)
        return DynamicScorerOutput(
            metric_logits=metric_logits,
            aggregate_score=aggregate_score,
            topm_indices=topm_indices,
            topm_trajectories_8=torch.gather(
                detached_proposals, 1, trajectory_index
            ),
            topm_candidate_states=torch.gather(
                candidate_states, 1, state_index
            ),
        )
