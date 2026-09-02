"""Base-score-independent proposal ranker for EpisodeDrive research.

The module consumes only current-observation tokens, current ego status and
decoded proposal geometry.  It deliberately has no argument for the released
scorer logits/features/scores or any future/evaluator field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

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

CURRENT_ACTOR_STATE_FIELDS: Tuple[str, ...] = (
    "x_over_50m",
    "y_over_50m",
    "vx_over_20mps",
    "vy_over_20mps",
    "sin_heading",
    "cos_heading",
    "length_over_10m",
    "width_over_5m",
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

# Local NAVSIM evaluates a Pacifica footprint constructed from the trajectory
# rear axle.  These are the exact values returned by the deployed
# ``get_pacifica_parameters()`` implementation.  Keeping the geometry in this
# torch-only module avoids importing nuPlan objects into the deployable scorer.
EGO_FRONT_LENGTH_FROM_REAR_AXLE_M = 4.049
EGO_REAR_LENGTH_FROM_REAR_AXLE_M = 1.127
EGO_HALF_LENGTH_M = 0.5 * (
    EGO_FRONT_LENGTH_FROM_REAR_AXLE_M + EGO_REAR_LENGTH_FROM_REAR_AXLE_M
)
EGO_REAR_AXLE_TO_CENTER_M = 0.5 * (
    EGO_FRONT_LENGTH_FROM_REAR_AXLE_M - EGO_REAR_LENGTH_FROM_REAR_AXLE_M
)
EGO_HALF_WIDTH_M = 0.5 * 2.297


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
    current_actor_auxiliary: bool = False
    current_actor_type_count: int = 3
    shared_future_auxiliary: bool = False
    shared_future_horizons: int = 8
    shared_future_relabeling: bool = False
    shared_future_constant_velocity_residual: bool = False
    # Let every trajectory point query the uncompressed current-observation
    # token grid directly.  The existing query-bank path remains intact and
    # this option is off for all legacy artifacts.  It is specifically meant
    # to preserve path-local obstacle evidence that can be lost when the
    # visual stream is first compressed into a small candidate-independent
    # query bank.
    trajectory_observation_attention: bool = False

    def __post_init__(self) -> None:
        if self.model_dim % self.num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if self.num_poses <= 0 or self.fine_top_k <= 0:
            raise ValueError("num_poses and fine_top_k must be positive")
        if self.max_observation_tokens <= 0:
            raise ValueError("max_observation_tokens must be positive")
        if self.interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        if self.current_actor_type_count <= 0:
            raise ValueError("current_actor_type_count must be positive")
        if self.shared_future_horizons <= 0:
            raise ValueError("shared_future_horizons must be positive")
        if self.shared_future_auxiliary and not self.dynamic_queries:
            raise ValueError(
                "shared-future supervision requires dynamic actor queries"
            )
        if self.shared_future_relabeling and not self.shared_future_auxiliary:
            raise ValueError(
                "candidate relabeling requires the shared-future prediction head"
            )
        if self.shared_future_relabeling and (
            self.shared_future_horizons != self.num_poses
        ):
            raise ValueError(
                "candidate relabeling requires one shared-future horizon per pose"
            )
        if self.shared_future_constant_velocity_residual and not (
            self.shared_future_auxiliary and self.current_actor_auxiliary
        ):
            raise ValueError(
                "constant-velocity future residuals require current-actor and "
                "shared-future prediction heads"
            )
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

    def project_observation(
        self,
        observation_tokens: torch.Tensor,
        observation_valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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
        return memory, padding_mask

    def decode_streams(
        self,
        memory: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if memory.ndim != 3 or memory.shape[-1] != self.config.model_dim:
            raise ValueError("memory must have shape [B, L, model_dim]")
        if padding_mask is not None and padding_mask.shape != memory.shape[:2]:
            raise ValueError("padding_mask must have shape [B, L]")
        batch_size = memory.shape[0]
        return {
            name: self.decoders[name](
                self.query_banks[name].expand(batch_size, -1, -1),
                memory,
                memory_key_padding_mask=padding_mask,
            )
            for name in self._BANK_NAMES
        }

    def forward(
        self,
        observation_tokens: torch.Tensor,
        observation_valid_mask: Optional[torch.Tensor] = None,
        return_memory: bool = False,
    ) -> Union[
        Dict[str, torch.Tensor],
        Tuple[
            Dict[str, torch.Tensor],
            torch.Tensor,
            Optional[torch.Tensor],
        ],
    ]:
        memory, padding_mask = self.project_observation(
            observation_tokens,
            observation_valid_mask,
        )
        streams = self.decode_streams(memory, padding_mask)
        if return_memory:
            return streams, memory, padding_mask
        return streams


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


class SharedFutureCandidateRelabeler(nn.Module):
    """Turn one predicted actor future into candidate-relative risk features.

    Actor states are predicted once in the current-ego frame.  This module is
    proposal conditioned only through deterministic, differentiable geometry;
    it never creates a separate future world for each candidate.
    """

    FEATURE_NAMES: Tuple[str, ...] = (
        "soft_min_box_clearance_over_20m",
        "soft_collision_probability",
        "soft_min_ttc_over_10s",
        "candidate_corridor_occupancy_over_4",
        "nearest_actor_relative_x_over_50m",
        "nearest_actor_relative_y_over_20m",
        "nearest_actor_relative_vx_over_20mps",
        "nearest_actor_relative_vy_over_20mps",
    )

    def __init__(
        self,
        model_dim: int,
        horizons: int,
        num_heads: int,
        dropout: float,
        interval_seconds: float,
    ) -> None:
        super().__init__()
        self.horizons = horizons
        self.interval_seconds = interval_seconds
        self.feature_encoder = nn.Sequential(
            nn.LayerNorm(len(self.FEATURE_NAMES)),
            nn.Linear(len(self.FEATURE_NAMES), model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.time_embedding = nn.Parameter(
            torch.empty(1, 1, horizons, model_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=2 * model_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            layer,
            num_layers=1,
            norm=nn.LayerNorm(model_dim),
        )
        nn.init.trunc_normal_(self.time_embedding, std=0.02)

    @staticmethod
    def _candidate_center_and_velocity(
        proposals: torch.Tensor,
        interval_seconds: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        heading = proposals[..., 2]
        center = proposals[..., :2] + EGO_REAR_AXLE_TO_CENTER_M * torch.stack(
            (torch.cos(heading), torch.sin(heading)), dim=-1
        )
        current_center = torch.zeros_like(center[..., :1, :])
        current_center[..., 0] = EGO_REAR_AXLE_TO_CENTER_M
        previous = torch.cat((current_center, center[..., :-1, :]), dim=-2)
        velocity = (center - previous) / interval_seconds
        return center, velocity

    def consequence_only(
        self,
        presence_logits: torch.Tensor,
        normalized_actor_state: torch.Tensor,
        proposals: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mask-aware candidate-relative physical features.

        The actor slot axis is padded.  Presence therefore participates in
        both the soft nearest-actor weights and the explicit no-actor
        fallback; otherwise a horizon with no valid actor would spuriously
        treat zero-filled slots as obstacles at the ego origin.
        """
        if presence_logits.ndim != 3:
            raise ValueError("presence_logits must have shape [B,H,N]")
        if normalized_actor_state.shape != (*presence_logits.shape, 8):
            raise ValueError("normalized_actor_state must have shape [B,H,N,8]")
        if proposals.ndim != 4 or proposals.shape[-2:] != (self.horizons, 3):
            raise ValueError("proposals must have shape [B,K,H,3]")
        if proposals.shape[0] != presence_logits.shape[0]:
            raise ValueError("shared future and proposals have different batches")
        if presence_logits.shape[1] != self.horizons:
            raise ValueError("shared future has an unexpected horizon count")

        # Decode the bounded training representation into metric current-ego
        # actor states.  Heading is represented periodically as sin/cos.
        actor_x = normalized_actor_state[..., 0] * 50.0
        actor_y = normalized_actor_state[..., 1] * 50.0
        actor_vx = normalized_actor_state[..., 2] * 20.0
        actor_vy = normalized_actor_state[..., 3] * 20.0
        actor_heading = torch.atan2(
            normalized_actor_state[..., 4],
            normalized_actor_state[..., 5],
        )
        actor_length = normalized_actor_state[..., 6].abs() * 10.0
        actor_width = normalized_actor_state[..., 7].abs() * 5.0
        presence = presence_logits.sigmoid()[:, None]

        candidate_center, candidate_velocity = self._candidate_center_and_velocity(
            proposals, self.interval_seconds
        )
        candidate_x = candidate_center[..., 0, None]
        candidate_y = candidate_center[..., 1, None]
        candidate_heading = proposals[..., 2, None]
        cosine = torch.cos(candidate_heading)
        sine = torch.sin(candidate_heading)
        delta_x = actor_x[:, None] - candidate_x
        delta_y = actor_y[:, None] - candidate_y
        relative_x = cosine * delta_x + sine * delta_y
        relative_y = -sine * delta_x + cosine * delta_y

        delta_vx = actor_vx[:, None] - candidate_velocity[..., 0, None]
        delta_vy = actor_vy[:, None] - candidate_velocity[..., 1, None]
        relative_vx = cosine * delta_vx + sine * delta_vy
        relative_vy = -sine * delta_vx + cosine * delta_vy

        relative_heading = actor_heading[:, None] - candidate_heading
        heading_cosine = torch.cos(relative_heading).abs()
        heading_sine = torch.sin(relative_heading).abs()
        projected_half_length = 0.5 * (
            heading_cosine * actor_length[:, None]
            + heading_sine * actor_width[:, None]
        )
        projected_half_width = 0.5 * (
            heading_sine * actor_length[:, None]
            + heading_cosine * actor_width[:, None]
        )

        # Signed distance of the actor rectangle to an ego-aligned rectangle.
        # It is exact for the projected axis-aligned approximation: negative
        # values indicate overlap and positive values indicate clearance.
        longitudinal = relative_x.abs() - (
            EGO_HALF_LENGTH_M + projected_half_length
        )
        lateral = relative_y.abs() - (
            EGO_HALF_WIDTH_M + projected_half_width
        )
        outside = torch.sqrt(
            torch.relu(longitudinal).square()
            + torch.relu(lateral).square()
            + 1.0e-6
        )
        inside = torch.minimum(
            torch.maximum(longitudinal, lateral),
            torch.zeros_like(longitudinal),
        )
        clearance = outside + inside

        any_actor = 1.0 - torch.prod(1.0 - presence, dim=-1)
        nearest_logits = -clearance / 2.0 + torch.log(
            presence.clamp_min(1.0e-8)
        )
        nearest_weight = torch.softmax(nearest_logits, dim=-1)
        nearest_clearance = (nearest_weight * clearance).sum(dim=-1)
        soft_min_clearance = (
            any_actor * nearest_clearance + (1.0 - any_actor) * 40.0
        )
        collision_per_actor = presence * torch.sigmoid(-clearance / 0.75)
        soft_collision = 1.0 - torch.prod(
            (1.0 - collision_per_actor).clamp(1.0e-5, 1.0), dim=-1
        )

        center_distance = torch.sqrt(
            relative_x.square() + relative_y.square() + 1.0e-6
        )
        closing_speed = -(
            relative_x * relative_vx + relative_y * relative_vy
        ) / center_distance
        closing_probability = torch.sigmoid((closing_speed - 0.2) / 0.25)
        ttc = (
            clearance.clamp_min(0.0) / closing_speed.clamp_min(0.1)
        ).clamp(0.0, 10.0)
        ttc_relevance = presence * closing_probability
        any_closing_actor = 1.0 - torch.prod(1.0 - ttc_relevance, dim=-1)
        ttc_logits = -ttc + torch.log(ttc_relevance.clamp_min(1.0e-8))
        ttc_weight = torch.softmax(ttc_logits, dim=-1)
        nearest_ttc = (ttc_weight * ttc).sum(dim=-1)
        soft_min_ttc = (
            any_closing_actor * nearest_ttc
            + (1.0 - any_closing_actor) * 10.0
        )

        ahead = torch.sigmoid(
            (relative_x + EGO_HALF_LENGTH_M) / 0.75
        ) * torch.sigmoid(
            (12.0 - relative_x) / 1.5
        )
        in_width = torch.sigmoid(
            (1.0 + projected_half_width - relative_y.abs()) / 0.4
        )
        corridor_occupancy = (presence * ahead * in_width).sum(dim=-1)
        nearest_relative_x = any_actor * (nearest_weight * relative_x).sum(dim=-1)
        nearest_relative_y = any_actor * (nearest_weight * relative_y).sum(dim=-1)
        nearest_relative_vx = any_actor * (nearest_weight * relative_vx).sum(dim=-1)
        nearest_relative_vy = any_actor * (nearest_weight * relative_vy).sum(dim=-1)

        return torch.stack(
            (
                soft_min_clearance.clamp(-5.0, 40.0) / 20.0,
                soft_collision,
                soft_min_ttc / 10.0,
                corridor_occupancy.clamp(0.0, 16.0) / 4.0,
                nearest_relative_x.clamp(-100.0, 100.0) / 50.0,
                nearest_relative_y.clamp(-50.0, 50.0) / 20.0,
                nearest_relative_vx.clamp(-40.0, 40.0) / 20.0,
                nearest_relative_vy.clamp(-40.0, 40.0) / 20.0,
            ),
            dim=-1,
        )

    def forward(
        self,
        presence_logits: torch.Tensor,
        normalized_actor_state: torch.Tensor,
        proposals: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        consequence = self.consequence_only(
            presence_logits,
            normalized_actor_state,
            proposals,
        )
        batch_size, candidate_count, horizons, _ = consequence.shape
        temporal = self.feature_encoder(consequence) + self.time_embedding
        temporal = self.temporal_encoder(
            temporal.reshape(batch_size * candidate_count, horizons, -1)
        ).reshape(batch_size, candidate_count, horizons, -1)
        token = 0.5 * (temporal.mean(dim=-2) + temporal[..., -1, :])
        return consequence, token


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
        if config.trajectory_observation_attention:
            self.trajectory_observation_attention = nn.MultiheadAttention(
                config.model_dim,
                config.num_heads,
                dropout=config.dropout,
                batch_first=True,
            )
            self.trajectory_observation_temporal_encoder = nn.TransformerEncoder(
                _make_encoder_layer(config),
                num_layers=1,
                norm=nn.LayerNorm(config.model_dim),
            )
            self.trajectory_observation_norm = nn.LayerNorm(config.model_dim)
            # The direct spatial path begins as an exact no-op.  Main scorer
            # heads and auxiliary losses first learn whether the additional
            # path-local evidence is useful, after which this scalar gate can
            # admit it without changing legacy/default behavior.
            self.trajectory_observation_gate = nn.Parameter(torch.zeros(()))
        else:
            self.trajectory_observation_attention = None
            self.trajectory_observation_temporal_encoder = None
            self.trajectory_observation_norm = None
            self.register_parameter("trajectory_observation_gate", None)
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
        if config.current_actor_auxiliary:
            self.current_actor_presence_head = nn.Linear(config.model_dim, 1)
            self.current_actor_type_head = nn.Linear(
                config.model_dim, config.current_actor_type_count
            )
            self.current_actor_state_head = nn.Linear(
                config.model_dim, len(CURRENT_ACTOR_STATE_FIELDS)
            )
        else:
            self.current_actor_presence_head = None
            self.current_actor_type_head = None
            self.current_actor_state_head = None
        if config.shared_future_auxiliary:
            horizons = config.shared_future_horizons
            self.shared_future_presence_head = nn.Linear(
                config.model_dim, horizons
            )
            self.shared_future_type_head = nn.Linear(
                config.model_dim,
                horizons * config.current_actor_type_count,
            )
            self.shared_future_state_head = nn.Linear(
                config.model_dim,
                horizons * len(CURRENT_ACTOR_STATE_FIELDS),
            )
            if config.shared_future_constant_velocity_residual:
                for head in (
                    self.shared_future_presence_head,
                    self.shared_future_type_head,
                    self.shared_future_state_head,
                ):
                    nn.init.zeros_(head.weight)
                    nn.init.zeros_(head.bias)
        else:
            self.shared_future_presence_head = None
            self.shared_future_type_head = None
            self.shared_future_state_head = None
        if config.shared_future_relabeling:
            self.shared_future_relabeler = SharedFutureCandidateRelabeler(
                model_dim=config.model_dim,
                horizons=config.shared_future_horizons,
                num_heads=config.num_heads,
                dropout=config.dropout,
                interval_seconds=config.interval_seconds,
            )
            self.shared_future_fusion_gate = nn.Parameter(torch.zeros(()))
        else:
            self.shared_future_relabeler = None
            self.register_parameter("shared_future_fusion_gate", None)

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

    def _constant_velocity_actor_future(
        self,
        current_actor_state: torch.Tensor,
    ) -> torch.Tensor:
        """Extrapolate normalized current actor slots in current-ego frame."""

        if current_actor_state.ndim != 3 or current_actor_state.shape[-1] != 8:
            raise ValueError("current_actor_state must have shape [B,N,8]")
        horizons = self.config.shared_future_horizons
        seconds = torch.arange(
            1,
            horizons + 1,
            dtype=current_actor_state.dtype,
            device=current_actor_state.device,
        ) * self.config.interval_seconds
        baseline = current_actor_state[:, None].expand(
            -1, horizons, -1, -1
        ).clone()
        # Position targets are divided by 50 m and velocity targets by 20 m/s.
        displacement_scale = seconds[None, :, None] * (20.0 / 50.0)
        baseline[..., 0] = (
            baseline[..., 0]
            + current_actor_state[:, None, :, 2] * displacement_scale
        )
        baseline[..., 1] = (
            baseline[..., 1]
            + current_actor_state[:, None, :, 3] * displacement_scale
        )
        return baseline

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

        # The projected current-observation memory is intentionally computed
        # once and shared by every proposal.  Both the candidate-independent
        # semantic banks and the optional path-local queries read this same
        # tensor; no candidate can alter the underlying scene memory.
        scene_streams, observation_memory, observation_padding_mask = (
            self.scene_encoder(
                observation_tokens,
                observation_valid_mask,
                return_memory=True,
            )
        )
        dynamic_tokens = scene_streams["dynamic"]
        current_actor: Dict[str, torch.Tensor] = {}
        if self.current_actor_presence_head is not None:
            current_actor = {
                "current_actor_presence_logits": (
                    self.current_actor_presence_head(dynamic_tokens).squeeze(-1)
                ),
                "current_actor_type_logits": self.current_actor_type_head(
                    dynamic_tokens
                ),
                "current_actor_state": self.current_actor_state_head(
                    dynamic_tokens
                ),
            }
        shared_future: Dict[str, torch.Tensor] = {}
        if self.shared_future_presence_head is not None:
            batch_size, actor_slots, _ = dynamic_tokens.shape
            horizons = self.config.shared_future_horizons
            type_count = self.config.current_actor_type_count
            state_width = len(CURRENT_ACTOR_STATE_FIELDS)
            future_presence = (
                self.shared_future_presence_head(dynamic_tokens)
                .permute(0, 2, 1)
                .contiguous()
            )
            future_type = (
                self.shared_future_type_head(dynamic_tokens)
                .reshape(batch_size, actor_slots, horizons, type_count)
                .permute(0, 2, 1, 3)
                .contiguous()
            )
            future_state = (
                self.shared_future_state_head(dynamic_tokens)
                .reshape(batch_size, actor_slots, horizons, state_width)
                .permute(0, 2, 1, 3)
                .contiguous()
            )
            if self.config.shared_future_constant_velocity_residual:
                future_presence = (
                    future_presence
                    + current_actor["current_actor_presence_logits"][:, None]
                )
                future_type = (
                    future_type
                    + current_actor["current_actor_type_logits"][:, None]
                )
                future_state = (
                    future_state
                    + self._constant_velocity_actor_future(
                        current_actor["current_actor_state"]
                    )
                )
            shared_future = {
                "shared_future_presence_logits": future_presence,
                "shared_future_type_logits": future_type,
                "shared_future_actor_state": future_state,
            }
        proposal_tokens, temporal_tokens = self.trajectory_encoder(proposals)
        status_token = self.status_encoder(status_feature)
        proposal_tokens = proposal_tokens + status_token.unsqueeze(1)
        trajectory_observation_token = torch.zeros_like(proposal_tokens)
        trajectory_observation_gate = proposal_tokens.new_zeros(())
        if self.trajectory_observation_attention is not None:
            batch_size, candidate_count, pose_count, width = temporal_tokens.shape
            point_queries = temporal_tokens + status_token[:, None, None, :]
            attended_points, _ = self.trajectory_observation_attention(
                point_queries.reshape(
                    batch_size,
                    candidate_count * pose_count,
                    width,
                ),
                observation_memory,
                observation_memory,
                key_padding_mask=observation_padding_mask,
                need_weights=False,
            )
            attended_points = attended_points.reshape(
                batch_size * candidate_count,
                pose_count,
                width,
            )
            attended_points = self.trajectory_observation_temporal_encoder(
                attended_points
            ).reshape(batch_size, candidate_count, pose_count, width)
            trajectory_observation_token = self.trajectory_observation_norm(
                0.5
                * (
                    attended_points.mean(dim=-2)
                    + attended_points[..., -1, :]
                )
            )
            trajectory_observation_gate = torch.tanh(
                self.trajectory_observation_gate
            )
            proposal_tokens = (
                proposal_tokens
                + trajectory_observation_gate * trajectory_observation_token
            )

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
        candidate_consequence = None
        candidate_consequence_token = None
        if self.shared_future_relabeler is not None:
            candidate_consequence, candidate_consequence_token = (
                self.shared_future_relabeler(
                    shared_future["shared_future_presence_logits"],
                    shared_future["shared_future_actor_state"],
                    proposals,
                )
            )
            coarse_features = coarse_features + torch.tanh(
                self.shared_future_fusion_gate
            ) * candidate_consequence_token
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
        result = {
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
            "trajectory_observation_token": trajectory_observation_token,
            "trajectory_observation_gate": trajectory_observation_gate,
            # Exposed for scorer-owned policy-improvement heads.  This tensor
            # is computed solely from current observation, ego status, and
            # proposal geometry; it contains no released score or evaluator
            # target.
            "candidate_features": coarse_features,
        }
        result.update(current_actor)
        result.update(shared_future)
        if candidate_consequence is not None:
            result.update(
                {
                    "candidate_relative_consequence": candidate_consequence,
                    "candidate_relative_consequence_token": (
                        candidate_consequence_token
                    ),
                    "shared_future_fusion_gate": torch.tanh(
                        self.shared_future_fusion_gate
                    ),
                }
            )
        return result


def normalize_current_actor_targets(
    target_actor_state: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert current-ego actor labels into bounded auxiliary targets.

    The raw layout is ``type, x, y, vx, vy, heading, length, width``.  It is
    a training target only: inference still receives current visual tokens,
    current ego/navigation state, and proposal geometry.
    """

    if target_actor_state.ndim != 3 or target_actor_state.shape[-1] != 8:
        raise ValueError("target_actor_state must have shape [B, N, 8]")
    actor_type = target_actor_state[..., 0].round().long()
    continuous = torch.stack(
        (
            target_actor_state[..., 1] / 50.0,
            target_actor_state[..., 2] / 50.0,
            target_actor_state[..., 3] / 20.0,
            target_actor_state[..., 4] / 20.0,
            torch.sin(target_actor_state[..., 5]),
            torch.cos(target_actor_state[..., 5]),
            target_actor_state[..., 6] / 10.0,
            target_actor_state[..., 7] / 5.0,
        ),
        dim=-1,
    )
    return actor_type, continuous


def current_actor_auxiliary_loss(
    output: Dict[str, torch.Tensor],
    target_actor_state: torch.Tensor,
    target_actor_mask: torch.Tensor,
    supervision_valid: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Supervise scorer-private dynamic queries with current actor labels."""

    required = {
        "current_actor_presence_logits",
        "current_actor_type_logits",
        "current_actor_state",
    }
    missing = sorted(required.difference(output))
    if missing:
        raise RuntimeError(f"current-actor auxiliary heads are disabled: {missing}")
    presence_logits = output["current_actor_presence_logits"]
    type_logits = output["current_actor_type_logits"]
    predicted_state = output["current_actor_state"]
    if target_actor_mask.shape != presence_logits.shape:
        raise ValueError("target_actor_mask must match dynamic query slots")
    if supervision_valid.shape != (presence_logits.shape[0],):
        raise ValueError("supervision_valid must have shape [B]")
    if target_actor_state.shape[:2] != presence_logits.shape:
        raise ValueError("target_actor_state slots must match dynamic queries")
    actor_type, continuous = normalize_current_actor_targets(target_actor_state)
    if predicted_state.shape != continuous.shape:
        raise ValueError("predicted current actor state has an unexpected shape")
    if type_logits.shape[:2] != presence_logits.shape:
        raise ValueError("current actor type logits have an unexpected shape")

    supervised_slots = supervision_valid.bool().unsqueeze(1).expand_as(
        target_actor_mask
    )
    presence_element = F.binary_cross_entropy_with_logits(
        presence_logits,
        target_actor_mask.to(presence_logits.dtype),
        reduction="none",
    )
    presence_weights = supervised_slots.to(presence_element.dtype)
    presence = (presence_element * presence_weights).sum() / (
        presence_weights.sum().clamp_min(1.0)
    )

    actor_valid = supervised_slots & target_actor_mask.bool()
    if bool(actor_valid.any()):
        maximum_type = type_logits.shape[-1] - 1
        valid_types = actor_type[actor_valid]
        if bool((valid_types < 0).any()) or bool((valid_types > maximum_type).any()):
            raise ValueError("current actor type is outside the configured classes")
        actor_type_loss = F.cross_entropy(type_logits[actor_valid], valid_types)
        state = F.smooth_l1_loss(
            predicted_state[actor_valid], continuous[actor_valid]
        )
    else:
        zero = presence_logits.sum() * 0.0
        actor_type_loss = zero
        state = zero
    total = presence + 0.25 * actor_type_loss + state
    return {
        "total": total,
        "presence": presence,
        "type": actor_type_loss,
        "state": state,
    }


def shared_future_auxiliary_loss(
    output: Dict[str, torch.Tensor],
    target_actor_future: torch.Tensor,
    target_actor_mask: torch.Tensor,
    supervision_valid: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Supervise candidate-independent actor futures in the current ego frame."""

    required = {
        "shared_future_presence_logits",
        "shared_future_type_logits",
        "shared_future_actor_state",
    }
    missing = sorted(required.difference(output))
    if missing:
        raise RuntimeError(f"shared-future auxiliary heads are disabled: {missing}")
    presence_logits = output["shared_future_presence_logits"]
    type_logits = output["shared_future_type_logits"]
    predicted_state = output["shared_future_actor_state"]
    if target_actor_future.ndim != 4 or target_actor_future.shape[-1] != 8:
        raise ValueError("target_actor_future must have shape [B,H,N,8]")
    if target_actor_mask.shape != target_actor_future.shape[:3]:
        raise ValueError("target_actor_mask must have shape [B,H,N]")
    if presence_logits.shape != target_actor_mask.shape:
        raise ValueError("shared-future presence shape does not match target")
    if predicted_state.shape != (*target_actor_mask.shape, 8):
        raise ValueError("shared-future state shape does not match target")
    if type_logits.shape[:3] != target_actor_mask.shape:
        raise ValueError("shared-future type shape does not match target")
    if supervision_valid.shape != (target_actor_future.shape[0],):
        raise ValueError("supervision_valid must have shape [B]")

    batch_size, horizons, actor_slots, _ = target_actor_future.shape
    actor_type, continuous = normalize_current_actor_targets(
        target_actor_future.reshape(batch_size * horizons, actor_slots, 8)
    )
    actor_type = actor_type.reshape(batch_size, horizons, actor_slots)
    continuous = continuous.reshape(batch_size, horizons, actor_slots, -1)
    supervised_slots = supervision_valid.bool()[:, None, None].expand_as(
        target_actor_mask
    )
    presence_element = F.binary_cross_entropy_with_logits(
        presence_logits,
        target_actor_mask.to(presence_logits.dtype),
        reduction="none",
    )
    presence_weights = supervised_slots.to(presence_element.dtype)
    presence = (presence_element * presence_weights).sum() / (
        presence_weights.sum().clamp_min(1.0)
    )

    actor_valid = supervised_slots & target_actor_mask.bool()
    if bool(actor_valid.any()):
        valid_types = actor_type[actor_valid]
        maximum_type = type_logits.shape[-1] - 1
        if bool((valid_types < 0).any()) or bool((valid_types > maximum_type).any()):
            raise ValueError("shared-future actor type is outside configured classes")
        actor_type_loss = F.cross_entropy(type_logits[actor_valid], valid_types)
        state = F.smooth_l1_loss(
            predicted_state[actor_valid], continuous[actor_valid]
        )
    else:
        zero = presence_logits.sum() * 0.0
        actor_type_loss = zero
        state = zero
    total = presence + 0.25 * actor_type_loss + state
    return {
        "total": total,
        "presence": presence,
        "type": actor_type_loss,
        "state": state,
    }


def candidate_relative_consequence_loss(
    output: Dict[str, torch.Tensor],
    relabeler: SharedFutureCandidateRelabeler,
    proposals: torch.Tensor,
    target_actor_future: torch.Tensor,
    target_actor_mask: torch.Tensor,
    supervision_valid: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Supervise predicted candidate consequences derived from logged future.

    The target is not an official score or a candidate-specific world.  One
    shared logged actor future is transformed by the same deterministic
    geometry for every candidate.  It is used only in the training loss; the
    model forward path continues to receive current observations and proposal
    geometry exclusively.
    """

    key = "candidate_relative_consequence"
    if key not in output:
        raise RuntimeError("candidate-relative relabeling is disabled")
    predicted = output[key]
    if target_actor_future.ndim != 4 or target_actor_future.shape[-1] != 8:
        raise ValueError("target_actor_future must have shape [B,H,N,8]")
    if target_actor_mask.shape != target_actor_future.shape[:3]:
        raise ValueError("target_actor_mask must have shape [B,H,N]")
    if supervision_valid.shape != (target_actor_future.shape[0],):
        raise ValueError("supervision_valid must have shape [B]")
    if proposals.shape[0] != target_actor_future.shape[0]:
        raise ValueError("proposal and future-target batches disagree")

    batch_size, horizons, actor_slots, _ = target_actor_future.shape
    _, normalized = normalize_current_actor_targets(
        target_actor_future.reshape(batch_size * horizons, actor_slots, 8)
    )
    normalized = normalized.reshape(batch_size, horizons, actor_slots, 8)
    with torch.no_grad():
        target_presence_logits = torch.where(
            target_actor_mask,
            torch.full_like(target_actor_mask, 20.0, dtype=predicted.dtype),
            torch.full_like(target_actor_mask, -20.0, dtype=predicted.dtype),
        )
        target = relabeler.consequence_only(
            target_presence_logits,
            normalized.to(predicted.dtype),
            proposals,
        )
    if predicted.shape != target.shape:
        raise ValueError("predicted and target consequences have different shapes")

    valid = supervision_valid.bool()[:, None, None, None].expand_as(predicted)
    element = F.smooth_l1_loss(predicted, target, reduction="none")
    # Collision, TTC, and clearance are the planning-critical dynamic fields.
    # Relative actor state remains supervised but cannot dominate merely by
    # contributing four channels.
    field_weights = predicted.new_tensor(
        (2.0, 4.0, 2.0, 1.0, 0.25, 0.25, 0.25, 0.25)
    )
    weights = valid.to(element.dtype) * field_weights
    total = (element * weights).sum() / weights.sum().clamp_min(1.0)

    def field_mean(indices: Tuple[int, ...]) -> torch.Tensor:
        selected = element[..., list(indices)]
        selected_valid = valid[..., list(indices)].to(selected.dtype)
        return (selected * selected_valid).sum() / selected_valid.sum().clamp_min(1.0)

    return {
        "total": total,
        "clearance": field_mean((0,)),
        "collision": field_mean((1,)),
        "ttc": field_mean((2,)),
        "occupancy": field_mean((3,)),
        "relative_state": field_mean((4, 5, 6, 7)),
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


def episode_drive_factor_loss(
    factor_logits: torch.Tensor,
    factor_targets: torch.Tensor,
    safety_negative_weight: float = 1.0,
) -> torch.Tensor:
    """Match the released EpisodeDrive six-head scorer loss.

    The factor order is NOC, DAC, DDC, TTC, progress, comfort.  Released M0
    maps partial NOC/DDC credit to binary failure, then applies BCE to all six
    heads, including the continuous progress target.  A non-unit safety weight
    is retained only as an explicit ablation; ``1.0`` is source-equivalent.
    """

    if factor_logits.shape != factor_targets.shape:
        raise ValueError("factor logits/targets must have identical shape")
    if factor_logits.shape[-1] != len(FACTOR_KEYS):
        raise ValueError("EpisodeDrive factor tensors must contain six fields")
    if safety_negative_weight <= 0:
        raise ValueError("safety_negative_weight must be positive")
    target = factor_targets.to(factor_logits.dtype).clone()
    target[..., 0] = (target[..., 0] == 1.0).to(target.dtype)
    target[..., 2] = (target[..., 2] == 1.0).to(target.dtype)
    element = F.binary_cross_entropy_with_logits(
        factor_logits, target, reduction="none"
    )
    if safety_negative_weight == 1.0:
        return element.mean()
    weights = torch.ones_like(element)
    # NOC, DAC, DDC and TTC failures are all rare harmful outcomes.  Weighting
    # mode is an explicit ablation; the source-equivalent 1.0 path above is
    # unchanged bit for bit.
    for index in (0, 1, 2, 3):
        weights[..., index] = torch.where(
            target[..., index] < 0.5,
            safety_negative_weight,
            1.0,
        )
    return (element * weights).sum() / weights.sum().clamp_min(1.0)
