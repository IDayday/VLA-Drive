"""M0-owned scorer-private representation with a Base-calibrated residual.

The frozen M0 proposal generator and released scorer provide deployable
current-inference tensors (proposals, factor logits, and aggregate scores).
An independent M0 visual-token encoder learns candidate-specific corrections.
No future annotation, evaluator value, or external-model representation enters
the forward path.  Zero-initialized correction heads preserve Base selection
exactly before training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from navsim.agents.EpisodeDrive.score_module.independent_ranker import (
    FACTOR_KEYS,
    IndependentProposalRanker,
    IndependentRankerConfig,
    pdms_factor_log_utility,
)


@dataclass(frozen=True)
class M0PrivateResidualConfig:
    hidden_dim: int = 256
    num_layers: int = 2
    num_heads: int = 8
    dropout: float = 0.1
    top_k: int = 64
    max_residual: float = 0.5
    inference_scale: float = 1.0
    score_mode: str = "hybrid"

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0 or self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.max_residual <= 0 or self.inference_scale < 0:
            raise ValueError("residual scale parameters are invalid")
        if self.score_mode not in {"direct", "factor", "hybrid"}:
            raise ValueError("score_mode must be direct, factor, or hybrid")


def _zero_last(module: nn.Sequential) -> None:
    last = module[-1]
    if not isinstance(last, nn.Linear):
        raise TypeError("expected a final Linear layer")
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)


def base_anchored_topk_indices(
    base_scores: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Return K candidates while making the deployed Base choice explicit."""

    if base_scores.ndim != 2:
        raise ValueError("base_scores must have shape [B,K]")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    count = min(top_k, base_scores.shape[1])
    base = base_scores.argmax(dim=1, keepdim=True)
    if count == 1:
        return base
    other = base_scores.clone()
    other.scatter_(1, base, float("-inf"))
    return torch.cat((base, other.topk(count - 1, dim=1).indices), dim=1)


class M0PrivateResidualRanker(nn.Module):
    """Joint scorer-private visual encoder and Base-calibrated value head."""

    def __init__(
        self,
        private_config: IndependentRankerConfig,
        residual_config: M0PrivateResidualConfig = M0PrivateResidualConfig(),
    ) -> None:
        super().__init__()
        if private_config.model_dim != residual_config.hidden_dim:
            raise ValueError("private and residual hidden dimensions must match")
        self.private_config = private_config
        self.residual_config = residual_config
        self.private_ranker = IndependentProposalRanker(private_config)
        factor_width = len(FACTOR_KEYS) * 2 + 1
        self.base_factor_encoder = nn.Sequential(
            nn.LayerNorm(factor_width),
            nn.Linear(factor_width, residual_config.hidden_dim),
            nn.GELU(),
            nn.Linear(residual_config.hidden_dim, residual_config.hidden_dim),
        )
        self.input_fusion = nn.Sequential(
            nn.LayerNorm(residual_config.hidden_dim * 2),
            nn.Linear(
                residual_config.hidden_dim * 2,
                residual_config.hidden_dim,
            ),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=residual_config.hidden_dim,
            nhead=residual_config.num_heads,
            dim_feedforward=4 * residual_config.hidden_dim,
            dropout=residual_config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.candidate_context = nn.TransformerEncoder(
            layer,
            num_layers=residual_config.num_layers,
            norm=nn.LayerNorm(residual_config.hidden_dim),
        )
        self.utility_delta_head = nn.Sequential(
            nn.LayerNorm(residual_config.hidden_dim),
            nn.Linear(residual_config.hidden_dim, residual_config.hidden_dim),
            nn.GELU(),
            nn.Linear(residual_config.hidden_dim, 1),
        )
        self.factor_delta_head = nn.Sequential(
            nn.LayerNorm(residual_config.hidden_dim),
            nn.Linear(residual_config.hidden_dim, residual_config.hidden_dim),
            nn.GELU(),
            nn.Linear(residual_config.hidden_dim, len(FACTOR_KEYS)),
        )
        self.relative_safety_head = nn.Sequential(
            nn.LayerNorm(residual_config.hidden_dim * 4),
            nn.Linear(
                residual_config.hidden_dim * 4,
                residual_config.hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(residual_config.hidden_dim, 3),
        )
        _zero_last(self.utility_delta_head)
        _zero_last(self.factor_delta_head)
        _zero_last(self.relative_safety_head)

    def forward(
        self,
        observation_tokens: torch.Tensor,
        status_feature: torch.Tensor,
        proposals: torch.Tensor,
        base_factor_logits: torch.Tensor,
        base_scores: torch.Tensor,
        observation_valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if base_factor_logits.shape != (*base_scores.shape, len(FACTOR_KEYS)):
            raise ValueError("base_factor_logits must have shape [B,K,6]")
        if proposals.shape[:2] != base_scores.shape:
            raise ValueError("proposal and Base-score candidate dimensions disagree")
        private = self.private_ranker(
            observation_tokens,
            status_feature,
            proposals,
            observation_valid_mask=observation_valid_mask,
        )
        candidate_features = private["candidate_features"]
        clipped = base_factor_logits.clamp(-12.0, 12.0)
        factor_context = self.base_factor_encoder(
            torch.cat(
                (clipped, clipped.sigmoid(), base_scores.unsqueeze(-1)),
                dim=-1,
            )
        )
        hidden = self.candidate_context(
            self.input_fusion(torch.cat((candidate_features, factor_context), dim=-1))
        )

        maximum = self.residual_config.max_residual
        utility_delta = maximum * torch.tanh(
            self.utility_delta_head(hidden).squeeze(-1)
        )
        refined_factor_logits = base_factor_logits + self.factor_delta_head(hidden)
        raw_factor_delta = (
            pdms_factor_log_utility(refined_factor_logits)
            - pdms_factor_log_utility(base_factor_logits)
        )
        factor_score_delta = maximum * torch.tanh(raw_factor_delta / maximum)
        if self.residual_config.score_mode == "direct":
            score_delta = utility_delta
        elif self.residual_config.score_mode == "factor":
            score_delta = factor_score_delta
        else:
            score_delta = utility_delta + factor_score_delta
        refined_scores = (
            base_scores + self.residual_config.inference_scale * score_delta
        )

        shortlist = base_anchored_topk_indices(
            base_scores,
            self.residual_config.top_k,
        )
        shortlist_mask = torch.zeros_like(base_scores, dtype=torch.bool)
        shortlist_mask.scatter_(1, shortlist, True)
        selection_scores = torch.where(
            shortlist_mask,
            refined_scores,
            base_scores - 100.0,
        )

        base_indices = base_scores.argmax(dim=1, keepdim=True)
        base_hidden = hidden.gather(
            1,
            base_indices[..., None].expand(-1, 1, hidden.shape[-1]),
        ).expand_as(hidden)
        relative_safety_logits = self.relative_safety_head(
            torch.cat(
                (
                    hidden,
                    base_hidden,
                    hidden - base_hidden,
                    hidden * base_hidden,
                ),
                dim=-1,
            )
        )
        return {
            "selection_scores": selection_scores,
            "refined_scores": refined_scores,
            "residual": score_delta,
            "utility_delta": utility_delta,
            "factor_score_delta": factor_score_delta,
            "refined_factor_logits": refined_factor_logits,
            "relative_safety_logits": relative_safety_logits,
            "shortlist_mask": shortlist_mask,
            "private_factor_logits": private["factor_logits"],
            "private_candidate_features": candidate_features,
            "private_scene_tokens": private["private_scene_tokens"],
        }


__all__ = (
    "M0PrivateResidualConfig",
    "M0PrivateResidualRanker",
    "base_anchored_topk_indices",
)
