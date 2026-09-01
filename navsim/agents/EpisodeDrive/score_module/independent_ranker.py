"""Base-score-independent proposal ranker for EpisodeDrive research.

The module consumes only current-observation tokens, current ego status and
decoded proposal geometry.  It deliberately has no argument for the released
scorer logits/features/scores or any future/evaluator field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


FACTOR_KEYS: Tuple[str, ...] = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "time_to_collision_within_bound",
    "ego_progress",
    "comfort",
)

FORBIDDEN_INFERENCE_FIELDS = frozenset(
    {
        "future_image",
        "future_images",
        "future_annotation",
        "future_annotations",
        "future_trajectory",
        "future_traffic_lights",
        "official_score",
        "official_scores",
        "pdm_score",
        "pdm_scores",
        "metric_cache",
    }
)


def assert_current_observation_only(features: Dict[str, object]) -> None:
    """Reject fields that cannot exist in deployable scorer inference."""

    leaked = sorted(FORBIDDEN_INFERENCE_FIELDS.intersection(features))
    if leaked:
        raise RuntimeError(f"Future/evaluator input leaked into scorer: {leaked}")


@dataclass(frozen=True)
class IndependentRankerConfig:
    observation_dim: int = 1536
    model_dim: int = 256
    status_dim: int = 8
    num_poses: int = 8
    num_heads: int = 8
    num_private_layers: int = 2
    num_trajectory_layers: int = 2
    num_candidate_layers: int = 1
    num_fine_layers: int = 2
    dynamic_queries: int = 12
    static_queries: int = 8
    signal_queries: int = 4
    global_queries: int = 4
    max_observation_tokens: int = 4096
    fine_top_k: int = 12
    consequence_dim: int = 4
    interval_seconds: float = 0.5
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.model_dim % self.num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if self.num_poses <= 0 or self.fine_top_k <= 0:
            raise ValueError("num_poses and fine_top_k must be positive")
        if self.max_observation_tokens <= 0:
            raise ValueError("max_observation_tokens must be positive")
        if self.interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        for count in (
            self.dynamic_queries,
            self.static_queries,
            self.signal_queries,
            self.global_queries,
        ):
            if count <= 0:
                raise ValueError("all scorer-private query banks must be non-empty")


def _make_decoder_layer(config: IndependentRankerConfig) -> nn.TransformerDecoderLayer:
    return nn.TransformerDecoderLayer(
        d_model=config.model_dim,
        nhead=config.num_heads,
        dim_feedforward=4 * config.model_dim,
        dropout=config.dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )


def _make_encoder_layer(config: IndependentRankerConfig) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d_model=config.model_dim,
        nhead=config.num_heads,
        dim_feedforward=4 * config.model_dim,
        dropout=config.dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )


class ScorerPrivateSceneEncoder(nn.Module):
    """Compress current-observation tokens with scorer-owned query banks."""

    _BANK_NAMES = ("dynamic", "static", "signal", "global")

    def __init__(self, config: IndependentRankerConfig):
        super().__init__()
        self.config = config
        self.observation_projection = nn.Sequential(
            nn.LayerNorm(config.observation_dim),
            nn.Linear(config.observation_dim, config.model_dim),
        )
        self.observation_position_embedding = nn.Parameter(
            torch.empty(1, config.max_observation_tokens, config.model_dim)
        )
        counts = {
            "dynamic": config.dynamic_queries,
            "static": config.static_queries,
            "signal": config.signal_queries,
            "global": config.global_queries,
        }
        self.query_banks = nn.ParameterDict(
            {
                name: nn.Parameter(torch.empty(1, count, config.model_dim))
                for name, count in counts.items()
            }
        )
        self.decoders = nn.ModuleDict(
            {
                name: nn.TransformerDecoder(
                    _make_decoder_layer(config),
                    num_layers=config.num_private_layers,
                    norm=nn.LayerNorm(config.model_dim),
                )
                for name in self._BANK_NAMES
            }
        )
        for bank in self.query_banks.values():
            nn.init.trunc_normal_(bank, std=0.02)
        nn.init.trunc_normal_(self.observation_position_embedding, std=0.01)

    def forward(
        self,
        observation_tokens: torch.Tensor,
        observation_valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if observation_tokens.ndim != 3:
            raise ValueError("observation_tokens must have shape [B, L, D]")
        if observation_tokens.shape[-1] != self.config.observation_dim:
            raise ValueError(
                "observation token width mismatch: "
                f"{observation_tokens.shape[-1]} != {self.config.observation_dim}"
            )
        if observation_tokens.shape[1] > self.config.max_observation_tokens:
            raise ValueError(
                "observation sequence is longer than max_observation_tokens: "
                f"{observation_tokens.shape[1]} > {self.config.max_observation_tokens}"
            )
        padding_mask = None
        if observation_valid_mask is not None:
            if observation_valid_mask.shape != observation_tokens.shape[:2]:
                raise ValueError("observation_valid_mask must have shape [B, L]")
            padding_mask = ~observation_valid_mask.bool()

        memory = self.observation_projection(observation_tokens)
        memory = memory + self.observation_position_embedding[
            :, : observation_tokens.shape[1]
        ]
        batch_size = observation_tokens.shape[0]
        return {
            name: self.decoders[name](
                self.query_banks[name].expand(batch_size, -1, -1),
                memory,
                memory_key_padding_mask=padding_mask,
            )
            for name in self._BANK_NAMES
        }


class ProposalTrajectoryEncoder(nn.Module):
    """Encode proposal geometry without candidate-index/type embeddings."""

    _POINT_FEATURE_DIM = 11

    def __init__(self, config: IndependentRankerConfig):
        super().__init__()
        self.config = config
        self.point_encoder = nn.Sequential(
            nn.Linear(self._POINT_FEATURE_DIM, config.model_dim),
            nn.GELU(),
            nn.Linear(config.model_dim, config.model_dim),
        )
        self.time_embedding = nn.Parameter(
            torch.empty(1, 1, config.num_poses, config.model_dim)
        )
        self.temporal_encoder = nn.TransformerEncoder(
            _make_encoder_layer(config),
            num_layers=config.num_trajectory_layers,
            norm=nn.LayerNorm(config.model_dim),
        )
        nn.init.trunc_normal_(self.time_embedding, std=0.02)

    @staticmethod
    def _geometry_features(
        proposals: torch.Tensor,
        interval_seconds: float = 0.5,
    ) -> torch.Tensor:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        xy = proposals[..., :2]
        heading = proposals[..., 2]
        previous_xy = torch.cat((torch.zeros_like(xy[..., :1, :]), xy[..., :-1, :]), dim=-2)
        delta_xy = xy - previous_xy
        velocity_xy = delta_xy / interval_seconds
        speed = torch.linalg.vector_norm(velocity_xy, dim=-1)
        previous_speed = torch.cat((torch.zeros_like(speed[..., :1]), speed[..., :-1]), dim=-1)
        acceleration = (speed - previous_speed) / interval_seconds
        previous_heading = torch.cat(
            (torch.zeros_like(heading[..., :1]), heading[..., :-1]), dim=-1
        )
        heading_delta = torch.atan2(
            torch.sin(heading - previous_heading),
            torch.cos(heading - previous_heading),
        )
        yaw_rate = heading_delta / interval_seconds
        curvature = yaw_rate / speed.clamp_min(0.5)
        return torch.cat(
            (
                xy / xy.new_tensor((30.0, 15.0)),
                torch.sin(heading).unsqueeze(-1),
                torch.cos(heading).unsqueeze(-1),
                velocity_xy / 20.0,
                (speed / 20.0).unsqueeze(-1),
                (acceleration.clamp(-10.0, 10.0) / 10.0).unsqueeze(-1),
                (yaw_rate.clamp(-2.0, 2.0) / 2.0).unsqueeze(-1),
                (curvature.clamp(-2.0, 2.0) / 2.0).unsqueeze(-1),
                torch.linspace(
                    0.0,
                    1.0,
                    proposals.shape[-2],
                    device=proposals.device,
                    dtype=proposals.dtype,
                )
                .view(*([1] * (proposals.ndim - 2)), proposals.shape[-2], 1)
                .expand(*proposals.shape[:-1], 1),
            ),
            dim=-1,
        )

    def forward(self, proposals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if proposals.ndim != 4 or proposals.shape[-1] != 3:
            raise ValueError("proposals must have shape [B, K, T, 3]")
        if proposals.shape[-2] != self.config.num_poses:
            raise ValueError(
                f"expected {self.config.num_poses} poses, got {proposals.shape[-2]}"
            )
        point_features = self._geometry_features(
            proposals,
            interval_seconds=self.config.interval_seconds,
        )
        point_tokens = self.point_encoder(point_features) + self.time_embedding
        batch_size, candidate_count, pose_count, width = point_tokens.shape
        encoded = self.temporal_encoder(
            point_tokens.reshape(batch_size * candidate_count, pose_count, width)
        ).reshape(batch_size, candidate_count, pose_count, width)
        pooled = 0.5 * (encoded.mean(dim=-2) + encoded[..., -1, :])
        return pooled, encoded


class IndependentProposalRanker(nn.Module):
    """Independent all-candidate coarse-to-fine scorer.

    Candidate self-attention has no positional/index embedding, so the scorer
    is permutation equivariant with respect to candidate order (apart from
    mathematically tied top-k scores).
    """

    def __init__(self, config: IndependentRankerConfig):
        super().__init__()
        self.config = config
        self.scene_encoder = ScorerPrivateSceneEncoder(config)
        self.trajectory_encoder = ProposalTrajectoryEncoder(config)
        self.status_encoder = nn.Sequential(
            nn.Linear(config.status_dim, config.model_dim),
            nn.GELU(),
            nn.Linear(config.model_dim, config.model_dim),
        )
        self.stream_attention = nn.ModuleDict(
            {
                name: nn.MultiheadAttention(
                    config.model_dim,
                    config.num_heads,
                    dropout=config.dropout,
                    batch_first=True,
                )
                for name in ScorerPrivateSceneEncoder._BANK_NAMES
            }
        )
        self.stream_gate = nn.Sequential(
            nn.Linear(config.model_dim * len(ScorerPrivateSceneEncoder._BANK_NAMES), config.model_dim),
            nn.GELU(),
            nn.Linear(config.model_dim, len(ScorerPrivateSceneEncoder._BANK_NAMES)),
        )
        self.coarse_candidate_encoder = nn.TransformerEncoder(
            _make_encoder_layer(config),
            num_layers=config.num_candidate_layers,
            norm=nn.LayerNorm(config.model_dim),
        )
        self.coarse_factor_heads = nn.ModuleDict(
            {key: nn.Linear(config.model_dim, 1) for key in FACTOR_KEYS}
        )
        self.coarse_utility_head = nn.Linear(config.model_dim, 1)
        self.consequence_head = nn.Linear(config.model_dim, config.consequence_dim)
        self.confidence_head = nn.Linear(config.model_dim, 1)

        self.fine_encoder = nn.TransformerEncoder(
            _make_encoder_layer(config),
            num_layers=config.num_fine_layers,
            norm=nn.LayerNorm(config.model_dim),
        )
        self.fine_delta_head = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim),
            nn.GELU(),
            nn.Linear(config.model_dim, 1),
        )
        # The fine stage begins as an exact identity over the coarse ranker.
        nn.init.zeros_(self.fine_delta_head[-1].weight)
        nn.init.zeros_(self.fine_delta_head[-1].bias)

    @staticmethod
    def _gather_candidates(features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return torch.gather(
            features,
            dim=1,
            index=indices.unsqueeze(-1).expand(-1, -1, features.shape[-1]),
        )

    def forward(
        self,
        observation_tokens: torch.Tensor,
        status_feature: torch.Tensor,
        proposals: torch.Tensor,
        observation_valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if status_feature.ndim != 2 or status_feature.shape[-1] != self.config.status_dim:
            raise ValueError(
                f"status_feature must have shape [B, {self.config.status_dim}]"
            )
        if proposals.shape[0] != observation_tokens.shape[0] or proposals.shape[0] != status_feature.shape[0]:
            raise ValueError("batch dimensions do not match")

        # This is intentionally executed once and shared by every proposal.
        scene_streams = self.scene_encoder(observation_tokens, observation_valid_mask)
        proposal_tokens, temporal_tokens = self.trajectory_encoder(proposals)
        proposal_tokens = proposal_tokens + self.status_encoder(status_feature).unsqueeze(1)

        attended = []
        for name in ScorerPrivateSceneEncoder._BANK_NAMES:
            stream, _ = self.stream_attention[name](
                proposal_tokens,
                scene_streams[name],
                scene_streams[name],
                need_weights=False,
            )
            attended.append(stream)
        gates = torch.softmax(self.stream_gate(torch.cat(attended, dim=-1)), dim=-1)
        fused = proposal_tokens + sum(
            gates[..., index : index + 1] * stream
            for index, stream in enumerate(attended)
        )
        coarse_features = self.coarse_candidate_encoder(fused)
        factor_logits = torch.stack(
            [self.coarse_factor_heads[key](coarse_features).squeeze(-1) for key in FACTOR_KEYS],
            dim=-1,
        )
        coarse_utility = self.coarse_utility_head(coarse_features).squeeze(-1)

        fine_count = min(self.config.fine_top_k, proposals.shape[1])
        fine_indices = torch.topk(coarse_utility, k=fine_count, dim=1).indices
        fine_features = self.fine_encoder(
            self._gather_candidates(coarse_features, fine_indices)
        )
        fine_delta = self.fine_delta_head(fine_features).squeeze(-1)
        refined_utility = self._gather_candidates(
            coarse_utility.unsqueeze(-1), fine_indices
        ).squeeze(-1) + fine_delta

        # Coarse-to-fine means that the coarse stage defines a shortlist and
        # the final decision is made *inside* that shortlist.  Scattering a
        # possibly negative refinement delta back into the full coarse vector
        # can otherwise let an unrefined rank-(K+1) candidate win the final
        # argmax, defeating the purpose of fine scoring.
        selection_utility = torch.full_like(coarse_utility, -1.0e4)
        selection_utility.scatter_(1, fine_indices, refined_utility)

        fine_mask = torch.zeros_like(coarse_utility, dtype=torch.bool)
        fine_mask.scatter_(1, fine_indices, True)
        return {
            "utility": selection_utility,
            "selection_utility": selection_utility,
            "coarse_utility": coarse_utility,
            "refined_utility": refined_utility,
            "factor_logits": factor_logits,
            "factor_keys": FACTOR_KEYS,
            "predicted_consequence": self.consequence_head(coarse_features),
            "confidence_logit": self.confidence_head(coarse_features).squeeze(-1),
            "fine_indices": fine_indices,
            "fine_mask": fine_mask,
            "private_scene_tokens": torch.cat(
                [scene_streams[name] for name in ScorerPrivateSceneEncoder._BANK_NAMES],
                dim=1,
            ),
            "trajectory_tokens": temporal_tokens,
            # Exposed for scorer-owned policy-improvement heads.  This tensor
            # is computed solely from current observation, ego status, and
            # proposal geometry; it contains no released score or evaluator
            # target.
            "candidate_features": coarse_features,
        }


@dataclass(frozen=True)
class ConservativeReferenceConfig:
    """Configuration for reference-relative, uncertainty-aware selection."""

    model_dim: int = 256
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 1
    safety_factor_count: int = 3
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.model_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("model dimensions must be positive")
        if self.model_dim % self.num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if self.num_layers <= 0 or self.safety_factor_count <= 0:
            raise ValueError("layer and safety-factor counts must be positive")


class ConservativeReferenceHead(nn.Module):
    """Predict candidate gain and safety regressions relative to one candidate.

    The caller supplies only the *index* of the reference candidate.  Numeric
    Base scores are deliberately absent from the interface.  Consequently the
    same head can use the released Base choice, its own coarse choice, or any
    other deployable current-observation policy as the conservative fallback.
    """

    def __init__(self, config: ConservativeReferenceConfig):
        super().__init__()
        self.config = config
        pair_width = 4 * config.model_dim
        self.pair_projection = nn.Sequential(
            nn.LayerNorm(pair_width),
            nn.Linear(pair_width, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.model_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.num_heads,
            dim_feedforward=4 * config.model_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.candidate_context = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.model_dim),
        )
        # Median plus positive lower/upper gaps enforces q10 <= q50 <= q90.
        self.gain_quantile_head = nn.Linear(config.model_dim, 3)
        self.safety_worse_head = nn.Linear(
            config.model_dim, config.safety_factor_count
        )
        self.safe_improvement_head = nn.Linear(config.model_dim, 1)

    @staticmethod
    def _reference_features(
        candidate_features: torch.Tensor,
        reference_indices: torch.Tensor,
    ) -> torch.Tensor:
        if candidate_features.ndim != 3:
            raise ValueError("candidate_features must have shape [B,K,D]")
        if reference_indices.shape != (candidate_features.shape[0],):
            raise ValueError("reference_indices must have shape [B]")
        if reference_indices.dtype not in (torch.int32, torch.int64):
            raise TypeError("reference_indices must be an integer tensor")
        if bool((reference_indices < 0).any()) or bool(
            (reference_indices >= candidate_features.shape[1]).any()
        ):
            raise IndexError("reference candidate index is out of range")
        gather_index = reference_indices[:, None, None].expand(
            -1, 1, candidate_features.shape[-1]
        )
        return candidate_features.gather(1, gather_index).expand_as(
            candidate_features
        )

    def forward(
        self,
        candidate_features: torch.Tensor,
        reference_indices: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        reference = self._reference_features(candidate_features, reference_indices)
        pair = torch.cat(
            (
                candidate_features,
                reference,
                candidate_features - reference,
                candidate_features * reference,
            ),
            dim=-1,
        )
        features = self.candidate_context(self.pair_projection(pair))
        raw_quantiles = self.gain_quantile_head(features)
        median = raw_quantiles[..., 1]
        lower = median - F.softplus(raw_quantiles[..., 0])
        upper = median + F.softplus(raw_quantiles[..., 2])
        quantiles = torch.stack((lower, median, upper), dim=-1)
        safety_logits = self.safety_worse_head(features)
        safe_improvement_logit = self.safe_improvement_head(features).squeeze(-1)

        # The reference is the zero-gain fallback by definition.  Making this
        # identity exact prevents numerical drift from accidentally replacing
        # the fallback with itself under a different score.
        reference_mask = torch.zeros(
            candidate_features.shape[:2],
            dtype=torch.bool,
            device=candidate_features.device,
        )
        reference_mask.scatter_(1, reference_indices[:, None], True)
        quantiles = quantiles.masked_fill(reference_mask.unsqueeze(-1), 0.0)
        safety_logits = safety_logits.masked_fill(
            reference_mask.unsqueeze(-1), -20.0
        )
        safe_improvement_logit = safe_improvement_logit.masked_fill(
            reference_mask, -20.0
        )
        return {
            "gain_quantiles": quantiles,
            "safety_worse_logits": safety_logits,
            "safe_improvement_logit": safe_improvement_logit,
            "reference_mask": reference_mask,
            "reference_indices": reference_indices,
            "reference_relative_features": features,
        }


class IndependentConservativeReferenceRanker(nn.Module):
    """Independent scorer with a conservative policy-improvement head.

    ``reference_indices`` identify a deployable fallback policy's choice.  No
    numeric score from that policy is consumed by either the proposal ranker
    or the reference head.
    """

    def __init__(
        self,
        ranker_config: IndependentRankerConfig,
        reference_config: Optional[ConservativeReferenceConfig] = None,
    ) -> None:
        super().__init__()
        self.ranker_config = ranker_config
        self.reference_config = reference_config or ConservativeReferenceConfig(
            model_dim=ranker_config.model_dim,
            num_heads=ranker_config.num_heads,
            dropout=ranker_config.dropout,
        )
        if self.reference_config.model_dim != ranker_config.model_dim:
            raise ValueError("reference and proposal-ranker widths must match")
        self.ranker = IndependentProposalRanker(ranker_config)
        self.reference_head = ConservativeReferenceHead(self.reference_config)

    def forward(
        self,
        observation_tokens: torch.Tensor,
        status_feature: torch.Tensor,
        proposals: torch.Tensor,
        reference_indices: torch.Tensor,
        observation_valid_mask: Optional[torch.Tensor] = None,
        *,
        minimum_lcb_gain: float = 0.0,
        maximum_safety_worse_probability: float = 0.25,
        minimum_safe_improvement_probability: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        output = self.ranker(
            observation_tokens,
            status_feature,
            proposals,
            observation_valid_mask=observation_valid_mask,
        )
        relative = self.reference_head(
            output["candidate_features"], reference_indices
        )
        relative["reference_selection_scores"] = (
            conservative_reference_selection_scores(
                relative["gain_quantiles"],
                relative["safety_worse_logits"],
                relative["safe_improvement_logit"],
                reference_indices,
                minimum_lcb_gain=minimum_lcb_gain,
                maximum_safety_worse_probability=(
                    maximum_safety_worse_probability
                ),
                minimum_safe_improvement_probability=(
                    minimum_safe_improvement_probability
                ),
            )
        )
        return output | relative


def conservative_reference_selection_scores(
    gain_quantiles: torch.Tensor,
    safety_worse_logits: torch.Tensor,
    safe_improvement_logit: torch.Tensor,
    reference_indices: torch.Tensor,
    *,
    gain_quantile_index: int = 0,
    minimum_lcb_gain: float = 0.0,
    maximum_safety_worse_probability: float = 0.25,
    minimum_safe_improvement_probability: float = 0.5,
    allowed_candidate_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return switch scores with an exact zero-valued fallback candidate.

    The lower gain quantile is a conservative utility estimate.  Candidates
    failing either learned safety gate receive ``-inf``; the reference always
    remains eligible with score zero.  Thresholds must be fixed on training or
    held-out validation logs and never tuned on Navtest.
    """

    if gain_quantiles.ndim != 3 or gain_quantiles.shape[-1] != 3:
        raise ValueError("gain_quantiles must have shape [B,K,3]")
    if safety_worse_logits.shape[:2] != gain_quantiles.shape[:2]:
        raise ValueError("safety logits must share [B,K]")
    if safe_improvement_logit.shape != gain_quantiles.shape[:2]:
        raise ValueError("safe_improvement_logit must have shape [B,K]")
    if reference_indices.shape != (gain_quantiles.shape[0],):
        raise ValueError("reference_indices must have shape [B]")
    if not 0.0 <= maximum_safety_worse_probability <= 1.0:
        raise ValueError("maximum safety probability must lie in [0,1]")
    if not 0.0 <= minimum_safe_improvement_probability <= 1.0:
        raise ValueError("minimum improvement probability must lie in [0,1]")
    if allowed_candidate_mask is not None:
        if allowed_candidate_mask.shape != gain_quantiles.shape[:2]:
            raise ValueError("allowed_candidate_mask must have shape [B,K]")
        if allowed_candidate_mask.dtype != torch.bool:
            raise TypeError("allowed_candidate_mask must be a boolean tensor")

    if not 0 <= gain_quantile_index < gain_quantiles.shape[-1]:
        raise ValueError("gain_quantile_index is out of range")
    lower_gain = gain_quantiles[..., gain_quantile_index]
    safety_probability = safety_worse_logits.sigmoid().amax(dim=-1)
    improvement_probability = safe_improvement_logit.sigmoid()
    eligible = (
        (lower_gain > minimum_lcb_gain)
        & (safety_probability <= maximum_safety_worse_probability)
        & (improvement_probability >= minimum_safe_improvement_probability)
    )
    if allowed_candidate_mask is not None:
        eligible = eligible & allowed_candidate_mask
    scores = lower_gain.masked_fill(~eligible, -torch.inf)
    scores.scatter_(1, reference_indices[:, None], 0.0)
    return scores


def masked_pinball_quantile_loss(
    predicted_quantiles: torch.Tensor,
    target: torch.Tensor,
    quantile_levels: Tuple[float, ...] = (0.1, 0.5, 0.9),
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pinball loss for ordered reference-relative gain quantiles."""

    if predicted_quantiles.ndim != 3:
        raise ValueError("predicted_quantiles must have shape [B,K,Q]")
    if target.shape != predicted_quantiles.shape[:2]:
        raise ValueError("target must have shape [B,K]")
    if predicted_quantiles.shape[-1] != len(quantile_levels):
        raise ValueError("quantile level count does not match prediction width")
    if any(level <= 0.0 or level >= 1.0 for level in quantile_levels):
        raise ValueError("quantile levels must lie strictly inside (0,1)")
    levels = predicted_quantiles.new_tensor(quantile_levels)
    error = target.unsqueeze(-1) - predicted_quantiles
    losses = torch.maximum((levels - 1.0) * error, levels * error)
    if valid_mask is None:
        return losses.mean()
    if valid_mask.shape != target.shape:
        raise ValueError("valid_mask must have shape [B,K]")
    weights = valid_mask.unsqueeze(-1).expand_as(losses).to(losses.dtype)
    return (losses * weights).sum() / weights.sum().clamp_min(1.0)


def pdms_factor_log_utility(
    factor_logits: torch.Tensor,
    weights: Tuple[float, float, float, float, float, float] = (
        1.0,
        1.0,
        0.0,
        5.0,
        5.0,
        2.0,
    ),
) -> torch.Tensor:
    """Reproduce the released six-factor M0 selection formula."""

    if factor_logits.shape[-1] != len(FACTOR_KEYS):
        raise ValueError(f"factor_logits must end in {len(FACTOR_KEYS)} factors")
    probabilities = factor_logits.sigmoid().clamp(1e-6, 1.0 - 1e-6)
    nc, dac, ddc, ttc, progress, comfort = probabilities.unbind(dim=-1)
    nc_weight, dac_weight, ddc_weight, ttc_weight, progress_weight, comfort_weight = weights
    return (
        nc_weight * nc.log()
        + dac_weight * dac.log()
        + ddc_weight * ddc.log()
        + (
            ttc_weight * ttc
            + progress_weight * progress
            + comfort_weight * comfort
        ).log()
    )


def weighted_pairwise_rank_loss(
    predicted_utility: torch.Tensor,
    target_utility: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    minimum_target_delta: float = 0.02,
) -> torch.Tensor:
    """PDMS-difference-weighted RankNet loss within each scene."""

    if predicted_utility.shape != target_utility.shape or predicted_utility.ndim != 2:
        raise ValueError("predicted_utility and target_utility must both be [B, K]")
    target_delta = target_utility[:, :, None] - target_utility[:, None, :]
    predicted_delta = predicted_utility[:, :, None] - predicted_utility[:, None, :]
    candidate_count = predicted_utility.shape[1]
    pair_mask = torch.triu(
        torch.ones(candidate_count, candidate_count, dtype=torch.bool, device=predicted_utility.device),
        diagonal=1,
    ).unsqueeze(0)
    pair_mask = pair_mask & (target_delta.abs() >= minimum_target_delta)
    if valid_mask is not None:
        if valid_mask.shape != predicted_utility.shape:
            raise ValueError("valid_mask must have shape [B, K]")
        pair_mask = pair_mask & valid_mask[:, :, None].bool() & valid_mask[:, None, :].bool()
    if not pair_mask.any():
        return predicted_utility.sum() * 0.0
    signs = target_delta.sign()
    weights = target_delta.abs().clamp_max(1.0)
    losses = F.softplus(-signs * predicted_delta) * weights
    return losses[pair_mask].sum() / weights[pair_mask].sum().clamp_min(1e-6)


def top_heavy_listwise_loss(
    predicted_utility: torch.Tensor,
    target_utility: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    """ListNet-style auxiliary objective emphasizing the best candidates."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    target_distribution = torch.softmax(target_utility / temperature, dim=1)
    return -(target_distribution * torch.log_softmax(predicted_utility, dim=1)).sum(dim=1).mean()


def top_regret_rank_loss(
    predicted_utility: torch.Tensor,
    target_utility: torch.Tensor,
    minimum_target_delta: float = 0.01,
) -> torch.Tensor:
    """Directly rank each scene's oracle candidate above every worse option.

    An all-pairs objective gives only ``K-1`` of ``K(K-1)/2`` comparisons to
    the candidate that controls top-1 regret.  This loss gives those pairs
    their own normalized objective and weights them by the regret incurred by
    choosing the competing candidate.
    """

    if predicted_utility.shape != target_utility.shape or predicted_utility.ndim != 2:
        raise ValueError("predicted_utility and target_utility must both be [B, K]")
    best_indices = target_utility.argmax(dim=1, keepdim=True)
    best_target = target_utility.gather(1, best_indices)
    best_prediction = predicted_utility.gather(1, best_indices)
    regret = best_target - target_utility
    valid = regret >= minimum_target_delta
    if not valid.any():
        return predicted_utility.sum() * 0.0
    element = F.softplus(-(best_prediction - predicted_utility))
    weights = regret.clamp_max(1.0)
    return (element * weights)[valid].sum() / weights[valid].sum().clamp_min(1e-6)


def factor_prediction_loss(
    factor_logits: torch.Tensor,
    factor_targets: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    positive_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Masked factor supervision; labels are training-only PDM outputs."""

    if factor_logits.shape != factor_targets.shape:
        raise ValueError("factor logits/targets must have identical shape")
    losses = F.binary_cross_entropy_with_logits(
        factor_logits,
        factor_targets.to(factor_logits.dtype),
        reduction="none",
        pos_weight=positive_weights,
    )
    if valid_mask is None:
        return losses.mean()
    if valid_mask.shape != factor_logits.shape[:-1]:
        raise ValueError("valid_mask must have shape [B, K]")
    expanded = valid_mask.unsqueeze(-1).expand_as(losses).to(losses.dtype)
    return (losses * expanded).sum() / expanded.sum().clamp_min(1.0)
