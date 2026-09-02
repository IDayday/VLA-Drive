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
    ConservativeReferenceConfig,
    ConservativeReferenceHead,
    FACTOR_KEYS,
    IndependentProposalRanker,
    IndependentRankerConfig,
    conservative_reference_selection_scores,
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
    switch_penalty: float = 0.0
    safety_floor: float = 0.0
    safety_relative_tolerance: float = 1.0
    preserve_ddc: bool = False
    safety_gate_mode: str = "none"
    # Fuse the released M0 Q-Former scene/ego representation with the
    # scorer-private raw visual stream instead of replacing it.  This remains
    # an M0-only, current-observation input and is disabled for old artifacts.
    m0_context_fusion: bool = False
    m0_scene_dim: int = 256
    m0_ego_dim: int = 256
    # Preserve and refine the candidate-conditioned hidden state produced by
    # the same frozen M0 scorer_attention path.  This is current-observation
    # M0 state, not an evaluator target or external-model representation.
    m0_candidate_fusion: bool = False
    m0_candidate_dim: int = 256
    # Low-capacity ablation: rank from the frozen M0 scorer-attention
    # candidate token plus Base factor context. The scorer-private scene
    # encoder can still receive auxiliary supervision, but its candidate
    # feature cannot influence ranking in this mode.
    m0_candidate_only: bool = False
    # Optional M0-owned policy-improvement head.  It predicts uncertainty and
    # safety *relative to the Base-selected proposal* instead of adding an
    # unconstrained absolute residual to every candidate.
    conservative_reference: bool = False
    reference_hidden_dim: int = 512
    reference_layers: int = 2
    gain_quantile_index: int = 1
    minimum_lcb_gain: float = 0.0
    maximum_safety_worse_probability: float = 0.1
    minimum_safe_improvement_probability: float = 0.7

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0 or self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.max_residual <= 0 or self.inference_scale < 0:
            raise ValueError("residual scale parameters are invalid")
        if self.switch_penalty < 0:
            raise ValueError("switch_penalty must be nonnegative")
        if not 0.0 <= self.safety_floor <= 1.0:
            raise ValueError("safety_floor must be in [0,1]")
        if self.safety_relative_tolerance < 0:
            raise ValueError("safety_relative_tolerance must be nonnegative")
        if self.m0_context_fusion and (
            self.m0_scene_dim <= 0 or self.m0_ego_dim <= 0
        ):
            raise ValueError("M0 context dimensions must be positive")
        if self.m0_candidate_fusion and self.m0_candidate_dim <= 0:
            raise ValueError("M0 candidate dimension must be positive")
        if self.m0_candidate_only and not self.m0_candidate_fusion:
            raise ValueError("M0 candidate-only mode requires candidate fusion")
        if self.reference_hidden_dim <= 0 or self.reference_layers <= 0:
            raise ValueError("reference-head dimensions must be positive")
        if self.gain_quantile_index not in (0, 1, 2):
            raise ValueError("gain_quantile_index must identify q10, q50, or q90")
        if not 0.0 <= self.maximum_safety_worse_probability <= 1.0:
            raise ValueError("maximum safety-worse probability must be in [0,1]")
        if not 0.0 <= self.minimum_safe_improvement_probability <= 1.0:
            raise ValueError("minimum safe-improvement probability must be in [0,1]")
        if self.score_mode not in {"direct", "factor", "hybrid"}:
            raise ValueError("score_mode must be direct, factor, or hybrid")
        if self.safety_gate_mode not in {
            "none",
            "relative_factor",
            "factor_all",
        }:
            raise ValueError(
                "safety_gate_mode must be none, relative_factor, or factor_all"
            )


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
        if residual_config.m0_candidate_fusion:
            self.m0_candidate_projection = nn.Sequential(
                nn.LayerNorm(residual_config.m0_candidate_dim),
                nn.Linear(
                    residual_config.m0_candidate_dim,
                    residual_config.hidden_dim,
                ),
            )
            self.m0_candidate_fusion = nn.Sequential(
                nn.LayerNorm(residual_config.hidden_dim * 2),
                nn.Linear(
                    residual_config.hidden_dim * 2,
                    residual_config.hidden_dim,
                ),
                nn.GELU(),
            )
        else:
            self.m0_candidate_projection = None
            self.m0_candidate_fusion = None
        if residual_config.m0_context_fusion:
            self.m0_scene_projection = nn.Sequential(
                nn.LayerNorm(residual_config.m0_scene_dim),
                nn.Linear(
                    residual_config.m0_scene_dim,
                    residual_config.hidden_dim,
                ),
            )
            self.m0_ego_projection = nn.Sequential(
                nn.LayerNorm(residual_config.m0_ego_dim),
                nn.Linear(
                    residual_config.m0_ego_dim,
                    residual_config.hidden_dim,
                ),
            )
            self.m0_scene_position_embedding = nn.Parameter(
                torch.empty(1, 16, residual_config.hidden_dim)
            )
            self.m0_context_type_embedding = nn.Parameter(
                torch.empty(1, 2, residual_config.hidden_dim)
            )
            self.m0_context_attention = nn.MultiheadAttention(
                residual_config.hidden_dim,
                residual_config.num_heads,
                dropout=residual_config.dropout,
                batch_first=True,
            )
            self.m0_context_norm = nn.LayerNorm(residual_config.hidden_dim)
            # The experiment starts from the exact private-only hidden path;
            # the released M0 context is admitted only as this gate learns.
            self.m0_context_gate = nn.Parameter(torch.zeros(()))
            nn.init.trunc_normal_(self.m0_scene_position_embedding, std=0.01)
            nn.init.trunc_normal_(self.m0_context_type_embedding, std=0.01)
        else:
            self.m0_scene_projection = None
            self.m0_ego_projection = None
            self.register_parameter("m0_scene_position_embedding", None)
            self.register_parameter("m0_context_type_embedding", None)
            self.m0_context_attention = None
            self.m0_context_norm = None
            self.register_parameter("m0_context_gate", None)
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
        if residual_config.conservative_reference:
            self.conservative_reference_head = ConservativeReferenceHead(
                ConservativeReferenceConfig(
                    model_dim=residual_config.hidden_dim,
                    hidden_dim=residual_config.reference_hidden_dim,
                    num_heads=residual_config.num_heads,
                    num_layers=residual_config.reference_layers,
                    dropout=residual_config.dropout,
                )
            )
        else:
            self.conservative_reference_head = None

    def forward(
        self,
        observation_tokens: torch.Tensor,
        status_feature: torch.Tensor,
        proposals: torch.Tensor,
        base_factor_logits: torch.Tensor,
        base_scores: torch.Tensor,
        observation_valid_mask: Optional[torch.Tensor] = None,
        m0_scene_features: Optional[torch.Tensor] = None,
        m0_ego_features: Optional[torch.Tensor] = None,
        m0_candidate_features: Optional[torch.Tensor] = None,
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
        fused = self.input_fusion(
            torch.cat((candidate_features, factor_context), dim=-1)
        )
        if self.residual_config.m0_candidate_fusion:
            if m0_candidate_features is None:
                raise ValueError(
                    "M0 candidate fusion requires released candidate features"
                )
            expected = (
                fused.shape[0],
                fused.shape[1],
                self.residual_config.m0_candidate_dim,
            )
            if m0_candidate_features.shape != expected:
                raise ValueError(
                    f"m0_candidate_features must have shape {expected}"
                )
            released_candidate = self.m0_candidate_projection(
                m0_candidate_features.float()
            )
            private_or_factor = (
                factor_context
                if self.residual_config.m0_candidate_only
                else fused
            )
            fused = self.m0_candidate_fusion(
                torch.cat((private_or_factor, released_candidate), dim=-1)
            )
        context_gate = fused.new_zeros(())
        if self.residual_config.m0_context_fusion:
            if m0_scene_features is None or m0_ego_features is None:
                raise ValueError(
                    "M0 context fusion requires released scene and ego features"
                )
            if m0_scene_features.shape != (
                fused.shape[0],
                16,
                self.residual_config.m0_scene_dim,
            ):
                raise ValueError("m0_scene_features must have shape [B,16,D]")
            if m0_ego_features.shape != (
                fused.shape[0],
                1,
                self.residual_config.m0_ego_dim,
            ):
                raise ValueError("m0_ego_features must have shape [B,1,D]")
            scene_context = self.m0_scene_projection(m0_scene_features.float())
            scene_context = scene_context + self.m0_scene_position_embedding
            scene_context = (
                scene_context + self.m0_context_type_embedding[:, :1]
            )
            ego_context = self.m0_ego_projection(m0_ego_features.float())
            ego_context = ego_context + self.m0_context_type_embedding[:, 1:]
            memory = torch.cat((scene_context, ego_context), dim=1)
            attended, _ = self.m0_context_attention(
                fused,
                memory,
                memory,
                need_weights=False,
            )
            context_gate = torch.tanh(self.m0_context_gate)
            fused = fused + context_gate * self.m0_context_norm(attended)
        hidden = self.candidate_context(fused)

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
        base_indices = base_scores.argmax(dim=1, keepdim=True)
        base_mask = torch.zeros_like(base_scores, dtype=torch.bool)
        base_mask.scatter_(1, base_indices, True)
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

        reference_output: Dict[str, torch.Tensor] = {}
        if self.conservative_reference_head is not None:
            reference_output = self.conservative_reference_head(
                hidden,
                base_indices.squeeze(1),
            )
            selection_scores = conservative_reference_selection_scores(
                reference_output["gain_quantiles"],
                reference_output["safety_worse_logits"],
                reference_output["safe_improvement_logit"],
                base_indices.squeeze(1),
                gain_quantile_index=self.residual_config.gain_quantile_index,
                minimum_lcb_gain=self.residual_config.minimum_lcb_gain,
                maximum_safety_worse_probability=(
                    self.residual_config.maximum_safety_worse_probability
                ),
                minimum_safe_improvement_probability=(
                    self.residual_config.minimum_safe_improvement_probability
                ),
                allowed_candidate_mask=shortlist_mask,
            )
            safety_mask = (
                reference_output["safety_worse_logits"].sigmoid().amax(dim=-1)
                <= self.residual_config.maximum_safety_worse_probability
            )
            eligible = torch.isfinite(selection_scores)
        else:
            gate_mode = self.residual_config.safety_gate_mode
            if gate_mode == "none":
                safety_mask = torch.ones_like(shortlist_mask)
            elif gate_mode == "relative_factor":
                safety_mask = (
                    relative_safety_logits.sigmoid()
                    >= self.residual_config.safety_floor
                ).all(dim=-1)
            else:
                safety_indices = [0, 1, 3]
                if self.residual_config.preserve_ddc:
                    safety_indices.append(2)
                probabilities = refined_factor_logits.sigmoid()[..., safety_indices]
                base_probabilities = probabilities.gather(
                    1,
                    base_indices[..., None].expand(
                        -1,
                        1,
                        len(safety_indices),
                    ),
                )
                safety_mask = (
                    probabilities >= self.residual_config.safety_floor
                ).all(dim=-1)
                safety_mask &= (
                    probabilities
                    >= base_probabilities
                    - self.residual_config.safety_relative_tolerance
                ).all(dim=-1)
            eligible = (shortlist_mask & safety_mask) | base_mask
            adjusted_scores = refined_scores - (
                (~base_mask).to(refined_scores.dtype)
                * self.residual_config.switch_penalty
            )
            selection_scores = torch.where(
                eligible,
                adjusted_scores,
                base_scores - 100.0,
            )

        result = {
            "selection_scores": selection_scores,
            "refined_scores": refined_scores,
            "residual": score_delta,
            "utility_delta": utility_delta,
            "factor_score_delta": factor_score_delta,
            "refined_factor_logits": refined_factor_logits,
            "relative_safety_logits": relative_safety_logits,
            "shortlist_mask": shortlist_mask,
            "safety_mask": safety_mask,
            "eligible_mask": eligible,
            "private_factor_logits": private["factor_logits"],
            "private_candidate_features": candidate_features,
            "private_scene_tokens": private["private_scene_tokens"],
            "m0_context_fusion_gate": context_gate,
        }
        result.update(reference_output)
        # Auxiliary targets supervise the scorer-owned current-observation
        # representation during training.  Their predictions are exposed to
        # the loss without adding any future tensor to this forward signature.
        for key in (
            "current_actor_presence_logits",
            "current_actor_type_logits",
            "current_actor_state",
            "shared_future_presence_logits",
            "shared_future_type_logits",
            "shared_future_actor_state",
            "candidate_relative_consequence",
            "candidate_relative_consequence_token",
            "shared_future_fusion_gate",
        ):
            if key in private:
                result[key] = private[key]
        return result


__all__ = (
    "M0PrivateResidualConfig",
    "M0PrivateResidualRanker",
    "base_anchored_topk_indices",
)
