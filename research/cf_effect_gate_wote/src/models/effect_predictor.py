"""A small candidate-conditioned forward predictor for legal replay effects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


EGO_EFFECT_DIM = 16
MAP_EFFECT_DIM = 8
ACTOR_EFFECT_DIM = 13
ACTOR_SLOTS = 16
HORIZON = 8


def _pad_ego_status(value: Tensor, width: int = 16) -> Tensor:
    if value.ndim != 2:
        raise ValueError(f"ego status must be [scene,feature], got {tuple(value.shape)}")
    if value.shape[-1] > width:
        raise ValueError(f"ego status width {value.shape[-1]} exceeds fixed limit {width}")
    return F.pad(value, (0, width - value.shape[-1]))


class EffectDecoderLayer(nn.Module):
    """Temporal self-attention followed by current-BEV cross-attention."""

    def __init__(self, hidden_dim: int, heads: int):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=0.0, batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=0.0, batch_first=True
        )
        self.norm_self = nn.LayerNorm(hidden_dim)
        self.norm_cross = nn.LayerNorm(hidden_dim)
        self.norm_ff = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, temporal: Tensor, memory: Tensor) -> Tensor:
        attended, _ = self.self_attention(temporal, temporal, temporal, need_weights=False)
        temporal = self.norm_self(temporal + attended)
        attended, _ = self.cross_attention(temporal, memory, memory, need_weights=False)
        temporal = self.norm_cross(temporal + attended)
        return self.norm_ff(temporal + self.feed_forward(temporal))


class CandidateEffectPredictor(nn.Module):
    """Predict structured replay effects without RGB or future-rollout inputs."""

    def __init__(
        self,
        hidden_dim: int = 256,
        decoder_layers: int = 2,
        attention_heads: int = 8,
        actor_slots: int = ACTOR_SLOTS,
    ):
        super().__init__()
        if hidden_dim <= 0 or decoder_layers not in {2, 3, 4}:
            raise ValueError("hidden_dim must be positive and decoder_layers must be 2-4")
        if hidden_dim % attention_heads != 0 or actor_slots != ACTOR_SLOTS:
            raise ValueError("attention heads must divide hidden dim; Gate fixes 16 actor slots")
        self.hidden_dim = hidden_dim
        self.actor_slots = actor_slots
        self.bev_projection = nn.Linear(256, hidden_dim)
        self.trajectory_projection = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.ego_projection = nn.Sequential(
            nn.Linear(16, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.temporal_embedding = nn.Parameter(torch.zeros(HORIZON, hidden_dim))
        self.actor_embedding = nn.Parameter(torch.zeros(actor_slots, hidden_dim))
        self.layers = nn.ModuleList(
            [EffectDecoderLayer(hidden_dim, attention_heads) for _ in range(decoder_layers)]
        )
        self.ego_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, EGO_EFFECT_DIM)
        )
        self.map_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, MAP_EFFECT_DIM)
        )
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, ACTOR_EFFECT_DIM),
        )
        self.actor_presence_head = nn.Linear(hidden_dim, 1)
        self.interaction_head = nn.Linear(hidden_dim, 1)
        nn.init.normal_(self.temporal_embedding, std=0.02)
        nn.init.normal_(self.actor_embedding, std=0.02)

    def forward(
        self,
        current_bev_tokens: Tensor,
        ego_status: Tensor,
        candidate_trajectory: Tensor,
    ) -> Mapping[str, Tensor]:
        if current_bev_tokens.ndim != 3 or current_bev_tokens.shape[1:] != (64, 256):
            raise ValueError(
                f"current BEV tokens must be [scene,64,256], got {tuple(current_bev_tokens.shape)}"
            )
        if candidate_trajectory.ndim != 4 or candidate_trajectory.shape[2:] != (HORIZON, 3):
            raise ValueError(
                "candidate trajectory must be [scene,candidate,8,3], got "
                f"{tuple(candidate_trajectory.shape)}"
            )
        scenes, candidates = candidate_trajectory.shape[:2]
        if current_bev_tokens.shape[0] != scenes or ego_status.shape[0] != scenes:
            raise ValueError("effect predictor scene axes do not align")
        if not all(
            torch.isfinite(value).all()
            for value in (current_bev_tokens, ego_status, candidate_trajectory)
        ):
            raise ValueError("effect predictor input contains NaN/Inf")
        memory = self.bev_projection(current_bev_tokens.float())
        memory = (
            memory[:, None]
            .expand(scenes, candidates, 64, self.hidden_dim)
            .reshape(scenes * candidates, 64, self.hidden_dim)
        )
        trajectory = self.trajectory_projection(candidate_trajectory.float())
        ego = self.ego_projection(_pad_ego_status(ego_status.float()))
        temporal = trajectory + ego[:, None, None] + self.temporal_embedding[None, None]
        temporal = temporal.reshape(scenes * candidates, HORIZON, self.hidden_dim)
        for layer in self.layers:
            temporal = layer(temporal, memory)
        ego_effect = self.ego_head(temporal).reshape(
            scenes, candidates, HORIZON, EGO_EFFECT_DIM
        )
        map_effect = self.map_head(temporal).reshape(
            scenes, candidates, HORIZON, MAP_EFFECT_DIM
        )
        actor_latent = temporal[:, :, None] + self.actor_embedding[None, None]
        actor_effect = self.actor_head(actor_latent).reshape(
            scenes, candidates, HORIZON, self.actor_slots, ACTOR_EFFECT_DIM
        )
        actor_presence_logits = self.actor_presence_head(actor_latent).squeeze(-1).reshape(
            scenes, candidates, HORIZON, self.actor_slots
        )
        interaction_logits = self.interaction_head(actor_latent).squeeze(-1).reshape(
            scenes, candidates, HORIZON, self.actor_slots
        )
        return {
            "ego_effect_transformed": ego_effect,
            "map_effect_transformed": map_effect,
            "actor_effect_transformed": actor_effect,
            "actor_presence_logits": actor_presence_logits,
            "interaction_logits": interaction_logits,
        }


def decode_transformed_effect(value: Tensor, limit: float = 12.0) -> Tensor:
    """Invert the stable asinh target transform with a finite safety bound."""

    return torch.sinh(value.clamp(min=-limit, max=limit))


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(value.dtype)
    while weights.ndim < value.ndim:
        weights = weights.unsqueeze(-1)
    denominator = weights.expand_as(value).sum().clamp_min(1.0)
    return (value * weights).sum() / denominator


@dataclass(frozen=True)
class EffectLossWeights:
    ego: float = 1.0
    map: float = 1.0
    actor: float = 1.0
    actor_presence: float = 0.25
    interaction: float = 0.5
    temporal_consistency: float = 0.1

    def validate(self) -> None:
        if any(value < 0 for value in vars(self).values()):
            raise ValueError("effect loss weights must be non-negative")


def effect_prediction_loss(
    prediction: Mapping[str, Tensor],
    target: Mapping[str, Tensor],
    weights: EffectLossWeights = EffectLossWeights(),
) -> tuple[Tensor, Mapping[str, Tensor]]:
    """Loss against only G2 replay tensors; no planning factors are accepted."""

    weights.validate()
    required = {
        "ego_effect",
        "map_effect",
        "actor_effect",
        "actor_mask",
        "interaction_mask",
    }
    missing = sorted(required - target.keys())
    if missing:
        raise ValueError(f"effect targets missing {missing}")
    unexpected = sorted(target.keys() - required)
    if unexpected:
        raise ValueError(f"effect targets contain forbidden/unknown fields {unexpected}")
    ego_target = torch.asinh(target["ego_effect"].float())
    map_target = torch.asinh(target["map_effect"].float())
    actor_target = torch.asinh(target["actor_effect"].float())
    actor_mask = target["actor_mask"].bool()
    interaction = target["interaction_mask"].float()
    if prediction["ego_effect_transformed"].shape != ego_target.shape:
        raise ValueError("predicted and target ego effect shapes differ")
    if prediction["map_effect_transformed"].shape != map_target.shape:
        raise ValueError("predicted and target map effect shapes differ")
    if prediction["actor_effect_transformed"].shape != actor_target.shape:
        raise ValueError("predicted and target actor effect shapes differ")
    ego_loss = F.smooth_l1_loss(prediction["ego_effect_transformed"], ego_target)
    map_loss = F.smooth_l1_loss(prediction["map_effect_transformed"], map_target)
    actor_loss = _masked_mean(
        F.smooth_l1_loss(
            prediction["actor_effect_transformed"], actor_target, reduction="none"
        ),
        actor_mask,
    )
    presence_loss = F.binary_cross_entropy_with_logits(
        prediction["actor_presence_logits"], actor_mask.float()
    )
    interaction_loss = _masked_mean(
        F.binary_cross_entropy_with_logits(
            prediction["interaction_logits"], interaction, reduction="none"
        ),
        actor_mask,
    )
    temporal_ego = F.smooth_l1_loss(
        prediction["ego_effect_transformed"][..., 1:, :]
        - prediction["ego_effect_transformed"][..., :-1, :],
        ego_target[..., 1:, :] - ego_target[..., :-1, :],
    )
    temporal_map = F.smooth_l1_loss(
        prediction["map_effect_transformed"][..., 1:, :]
        - prediction["map_effect_transformed"][..., :-1, :],
        map_target[..., 1:, :] - map_target[..., :-1, :],
    )
    consecutive_actor_mask = actor_mask[..., 1:, :] & actor_mask[..., :-1, :]
    temporal_actor = _masked_mean(
        F.smooth_l1_loss(
            prediction["actor_effect_transformed"][..., 1:, :, :]
            - prediction["actor_effect_transformed"][..., :-1, :, :],
            actor_target[..., 1:, :, :] - actor_target[..., :-1, :, :],
            reduction="none",
        ),
        consecutive_actor_mask,
    )
    temporal = (temporal_ego + temporal_map + temporal_actor) / 3.0
    components = {
        "ego": ego_loss,
        "map": map_loss,
        "actor": actor_loss,
        "actor_presence": presence_loss,
        "interaction": interaction_loss,
        "temporal_consistency": temporal,
    }
    total = (
        weights.ego * ego_loss
        + weights.map * map_loss
        + weights.actor * actor_loss
        + weights.actor_presence * presence_loss
        + weights.interaction * interaction_loss
        + weights.temporal_consistency * temporal
    )
    if not torch.isfinite(total):
        raise FloatingPointError("effect predictor produced non-finite loss")
    return total, components


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
