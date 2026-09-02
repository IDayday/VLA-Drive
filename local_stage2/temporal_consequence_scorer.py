"""Deployable temporal-consequence scorer for frozen EpisodeDrive proposals.

The model learns a scorer-specific hidden state from the current scene and a
candidate trajectory.  It predicts collision/TTC timing, candidate-relative
key-actor geometry, and road-area violations before producing a zero-initialized
residual over the released public-Base score.  Logged-future/PDM tensors are
training targets only and are never arguments to :meth:`forward`.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from local_stage2.public_base_residual_scorer import (
    FACTOR_KEYS,
    base_anchored_topk_indices,
    pdm_log_aggregate,
)
from navsim.agents.EpisodeDrive.episodedrive_agent import EpisodeDriveAgent


HORIZON_COUNT = 8
RISK_KINDS = 2
AREA_KINDS = 2
ACTOR_STATE_DIM = 6


@dataclass(frozen=True)
class TemporalConsequenceConfig:
    hidden_dim: int = 256
    trajectory_dim: int = 128
    temporal_layers: int = 2
    temporal_heads: int = 8
    scene_layers: int = 2
    scene_heads: int = 8
    dropout: float = 0.1
    top_k: int = 16
    max_residual: float = 0.5
    inference_scale: float = 1.0
    switch_penalty: float = 0.0
    safety_floor: float = 0.0
    safety_relative_tolerance: float = 1.0
    use_base_candidate_features: bool = False
    score_mode: str = "residual"
    use_relative_safety_head: bool = False
    safety_gate_mode: str = "absolute"
    utility_head_mode: str = "independent"

    def __post_init__(self) -> None:
        if self.hidden_dim % self.temporal_heads:
            raise ValueError("hidden_dim must be divisible by temporal_heads")
        if self.hidden_dim % self.scene_heads:
            raise ValueError("hidden_dim must be divisible by scene_heads")
        if self.temporal_layers <= 0 or self.scene_layers <= 0:
            raise ValueError("temporal_layers and scene_layers must be positive")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.max_residual <= 0:
            raise ValueError("max_residual must be positive")
        if self.inference_scale < 0 or self.switch_penalty < 0:
            raise ValueError("inference scale and switch penalty must be non-negative")
        if not 0.0 <= self.safety_floor <= 1.0:
            raise ValueError("safety_floor must be in [0, 1]")
        if self.safety_relative_tolerance < 0:
            raise ValueError("safety_relative_tolerance must be non-negative")
        if self.score_mode not in {"residual", "factor_aggregate", "hybrid"}:
            raise ValueError(
                "score_mode must be residual, factor_aggregate, or hybrid"
            )
        if self.safety_gate_mode not in {"absolute", "relative"}:
            raise ValueError("safety_gate_mode must be absolute or relative")
        if self.safety_gate_mode == "relative" and not self.use_relative_safety_head:
            raise ValueError("relative safety gating requires the relative safety head")
        if self.utility_head_mode not in {"independent", "base_relative"}:
            raise ValueError("utility_head_mode must be independent or base_relative")


def _wrap_angle(value: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(value), torch.cos(value))


def temporal_trajectory_features(
    proposals: torch.Tensor,
    interval_seconds: float = 0.5,
) -> torch.Tensor:
    """Return normalized per-horizon kinematics with shape ``[B,K,8,8]``."""

    if proposals.ndim != 4 or proposals.shape[-2:] != (HORIZON_COUNT, 3):
        raise ValueError("proposals must have shape [B, K, 8, 3]")
    origin = torch.zeros_like(proposals[:, :, :1])
    states = torch.cat((origin, proposals), dim=2)
    delta_xy = states[:, :, 1:, :2] - states[:, :, :-1, :2]
    speed = torch.linalg.vector_norm(delta_xy, dim=-1) / interval_seconds
    previous_speed = torch.cat((torch.zeros_like(speed[:, :, :1]), speed[:, :, :-1]), dim=2)
    acceleration = (speed - previous_speed) / interval_seconds
    heading_delta = _wrap_angle(states[:, :, 1:, 2] - states[:, :, :-1, 2])
    yaw_rate = heading_delta / interval_seconds
    curvature = yaw_rate / speed.clamp_min(0.5)
    return torch.stack(
        (
            proposals[..., 0] / 30.0,
            proposals[..., 1] / 15.0,
            torch.sin(proposals[..., 2]),
            torch.cos(proposals[..., 2]),
            speed / 20.0,
            acceleration.clamp(-10.0, 10.0) / 10.0,
            yaw_rate.clamp(-2.0, 2.0) / 2.0,
            curvature.clamp(-2.0, 2.0) / 2.0,
        ),
        dim=-1,
    )


class _CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        normalized_memory = self.memory_norm(memory)
        attended, _ = self.attention(
            self.query_norm(query),
            normalized_memory,
            normalized_memory,
            need_weights=False,
        )
        hidden = query + self.dropout(attended)
        return hidden + self.ffn(hidden)


def _zero_last(module: nn.Sequential) -> None:
    last = module[-1]
    if not isinstance(last, nn.Linear):
        raise TypeError("Expected final Linear layer")
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)


class TemporalConsequenceRanker(nn.Module):
    """Current-observation consequence predictor and Base-anchored fine-ranker."""

    def __init__(
        self,
        config: TemporalConsequenceConfig = TemporalConsequenceConfig(),
    ) -> None:
        super().__init__()
        self.config = config
        self.point_encoder = nn.Sequential(
            nn.Linear(8, config.trajectory_dim),
            nn.GELU(),
            nn.Linear(config.trajectory_dim, config.hidden_dim),
        )
        self.time_embedding = nn.Parameter(
            torch.randn(1, 1, HORIZON_COUNT, config.hidden_dim) * 0.01
        )
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.temporal_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer,
            num_layers=config.temporal_layers,
        )
        self.scene_projection = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, config.hidden_dim),
        )
        self.ego_projection = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, config.hidden_dim),
        )
        self.scene_attention = nn.ModuleList(
            [
                _CrossAttentionBlock(
                    config.hidden_dim,
                    config.scene_heads,
                    config.dropout,
                )
                for _ in range(config.scene_layers)
            ]
        )
        if config.use_base_candidate_features:
            self.base_feature_projection: Optional[nn.Module] = nn.Sequential(
                nn.LayerNorm(256),
                nn.Linear(256, config.hidden_dim),
            )
        else:
            self.base_feature_projection = None
        self.horizon_fusion = nn.Sequential(
            nn.LayerNorm(config.hidden_dim * 2),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
        )

        self.risk_head = nn.Linear(config.hidden_dim, RISK_KINDS)
        self.area_head = nn.Linear(config.hidden_dim, AREA_KINDS)
        self.actor_valid_head = nn.Linear(config.hidden_dim, RISK_KINDS)
        self.actor_state_head = nn.Linear(
            config.hidden_dim,
            RISK_KINDS * ACTOR_STATE_DIM,
        )

        consequence_input = HORIZON_COUNT * (
            RISK_KINDS + AREA_KINDS + RISK_KINDS + RISK_KINDS * ACTOR_STATE_DIM
        )
        self.consequence_encoder = nn.Sequential(
            nn.LayerNorm(consequence_input),
            nn.Linear(consequence_input, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.utility_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim * 2),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )
        if config.utility_head_mode == "base_relative":
            scorer_width = config.hidden_dim * 2
            self.base_relative_utility_head: Optional[nn.Module] = nn.Sequential(
                nn.LayerNorm(scorer_width * 4),
                nn.Linear(scorer_width * 4, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, 1),
            )
        else:
            self.base_relative_utility_head = None
        self.factor_delta_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim * 2),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, len(FACTOR_KEYS)),
        )
        if config.use_relative_safety_head:
            scorer_width = config.hidden_dim * 2
            self.relative_safety_head: Optional[nn.Module] = nn.Sequential(
                nn.LayerNorm(scorer_width * 4),
                nn.Linear(scorer_width * 4, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, 3),
            )
        else:
            self.relative_safety_head = None
        _zero_last(self.utility_head)
        if self.base_relative_utility_head is not None:
            _zero_last(self.base_relative_utility_head)
        _zero_last(self.factor_delta_head)
        if self.relative_safety_head is not None:
            _zero_last(self.relative_safety_head)

    @staticmethod
    def _normalized_actor_state(
        actor_state: torch.Tensor,
        actor_probability: torch.Tensor,
    ) -> torch.Tensor:
        scale = actor_state.new_tensor((30.0, 15.0, 10.0, 5.0, 1.0, 1.0))
        normalized = actor_state / scale
        return normalized * actor_probability.unsqueeze(-1)

    def forward(
        self,
        candidate_features: torch.Tensor,
        proposals: torch.Tensor,
        base_factor_logits: torch.Tensor,
        base_scores: torch.Tensor,
        scene_features: torch.Tensor,
        ego_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if candidate_features.shape[:2] != base_scores.shape:
            raise ValueError("candidate feature/base score dimensions disagree")
        if proposals.shape != (*base_scores.shape, HORIZON_COUNT, 3):
            raise ValueError("proposals must have shape [B, K, 8, 3]")
        if base_factor_logits.shape != (*base_scores.shape, len(FACTOR_KEYS)):
            raise ValueError("base_factor_logits must have shape [B, K, 6]")
        batch_size, candidate_count = base_scores.shape
        if scene_features.ndim != 3 or scene_features.shape[0] != batch_size:
            raise ValueError("scene_features must have shape [B, S, 256]")
        if ego_features.shape != (batch_size, 1, 256):
            raise ValueError("ego_features must have shape [B, 1, 256]")

        point_hidden = self.point_encoder(temporal_trajectory_features(proposals))
        point_hidden = point_hidden + self.time_embedding
        point_hidden = self.temporal_encoder(
            point_hidden.reshape(batch_size * candidate_count, HORIZON_COUNT, -1)
        ).reshape(batch_size, candidate_count, HORIZON_COUNT, -1)
        candidate_hidden = point_hidden.mean(dim=2)
        memory = torch.cat(
            (
                self.scene_projection(scene_features.float()),
                self.ego_projection(ego_features.float()),
            ),
            dim=1,
        )
        for block in self.scene_attention:
            candidate_hidden = block(candidate_hidden, memory)
        if self.base_feature_projection is not None:
            candidate_hidden = candidate_hidden + self.base_feature_projection(
                candidate_features.float()
            )

        horizon_hidden = self.horizon_fusion(
            torch.cat(
                (
                    point_hidden,
                    candidate_hidden.unsqueeze(2).expand_as(point_hidden),
                ),
                dim=-1,
            )
        )
        risk_logits = self.risk_head(horizon_hidden)
        area_logits = self.area_head(horizon_hidden)
        actor_valid_logits = self.actor_valid_head(horizon_hidden)
        actor_state = self.actor_state_head(horizon_hidden).reshape(
            batch_size,
            candidate_count,
            HORIZON_COUNT,
            RISK_KINDS,
            ACTOR_STATE_DIM,
        )
        actor_probability = actor_valid_logits.sigmoid()
        consequence_values = torch.cat(
            (
                risk_logits.sigmoid(),
                area_logits.sigmoid(),
                actor_probability,
                self._normalized_actor_state(actor_state, actor_probability).flatten(3),
            ),
            dim=-1,
        ).flatten(2)
        consequence_token = self.consequence_encoder(consequence_values)
        scorer_hidden = torch.cat((candidate_hidden, consequence_token), dim=-1)
        base_indices = base_scores.argmax(dim=1, keepdim=True)
        relation_hidden: Optional[torch.Tensor] = None
        if (
            self.relative_safety_head is not None
            or self.base_relative_utility_head is not None
        ):
            base_hidden = scorer_hidden.gather(
                1,
                base_indices[..., None].expand(-1, 1, scorer_hidden.shape[-1]),
            ).expand_as(scorer_hidden)
            relation_hidden = torch.cat(
                (
                    scorer_hidden,
                    base_hidden,
                    scorer_hidden - base_hidden,
                    scorer_hidden * base_hidden,
                ),
                dim=-1,
            )
        if self.relative_safety_head is not None:
            assert relation_hidden is not None
            relative_safety_logits = self.relative_safety_head(relation_hidden)
        else:
            relative_safety_logits = scorer_hidden.new_zeros(
                batch_size,
                candidate_count,
                3,
            )
        if self.base_relative_utility_head is not None:
            assert relation_hidden is not None
            utility_delta = self.config.max_residual * torch.tanh(
                self.base_relative_utility_head(relation_hidden).squeeze(-1)
            )
            # The public Base candidate is the reference action and must not
            # receive an arbitrary learned offset.
            utility_delta = utility_delta - utility_delta.gather(1, base_indices)
        else:
            utility_delta = self.config.max_residual * torch.tanh(
                self.utility_head(scorer_hidden).squeeze(-1)
            )
        refined_factor_logits = base_factor_logits + self.factor_delta_head(scorer_hidden)
        raw_factor_delta = pdm_log_aggregate(
            refined_factor_logits
        ) - pdm_log_aggregate(base_factor_logits)
        factor_score_delta = self.config.max_residual * torch.tanh(
            raw_factor_delta / self.config.max_residual
        )
        if self.config.score_mode == "residual":
            score_delta = utility_delta
        elif self.config.score_mode == "factor_aggregate":
            score_delta = factor_score_delta
        else:
            score_delta = utility_delta + factor_score_delta
        refined_scores = base_scores + self.config.inference_scale * score_delta

        top_indices = base_anchored_topk_indices(base_scores, self.config.top_k)
        top_k_mask = torch.zeros_like(base_scores, dtype=torch.bool)
        top_k_mask.scatter_(1, top_indices, True)
        base_selected = torch.zeros_like(top_k_mask)
        base_selected.scatter_(1, base_indices, True)

        # risk logits represent P(event by horizon); negating the final logit
        # is exactly the logit of P(no event by 4 s).
        absolute_predicted_safety = torch.sigmoid(-risk_logits[..., -1, :])
        if self.config.safety_gate_mode == "relative":
            predicted_safety = relative_safety_logits.sigmoid()
        else:
            predicted_safety = absolute_predicted_safety
        safety_width = predicted_safety.shape[-1]
        base_safety = predicted_safety.gather(
            1,
            base_indices[..., None].expand(-1, 1, safety_width),
        )
        safe = (predicted_safety >= self.config.safety_floor).all(dim=-1)
        safe &= (
            predicted_safety
            >= base_safety - self.config.safety_relative_tolerance
        ).all(dim=-1)
        eligible = top_k_mask & safe
        eligible |= base_selected
        adjusted_scores = refined_scores - (
            (~base_selected).to(refined_scores.dtype) * self.config.switch_penalty
        )
        selection_scores = torch.where(
            eligible,
            adjusted_scores,
            base_scores - 100.0,
        )
        return {
            "selection_scores": selection_scores,
            "refined_scores": refined_scores,
            "residual": score_delta,
            "utility_delta": utility_delta,
            "factor_score_delta": factor_score_delta,
            "refined_factor_logits": refined_factor_logits,
            "relative_safety_logits": relative_safety_logits,
            "risk_logits": risk_logits,
            "area_logits": area_logits,
            "actor_valid_logits": actor_valid_logits,
            "actor_state": actor_state,
            "predicted_safety": predicted_safety,
            "absolute_predicted_safety": absolute_predicted_safety,
            "consequence_token": consequence_token,
            "candidate_hidden": candidate_hidden,
            "top_k_mask": top_k_mask,
            "eligible_mask": eligible,
        }


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


class TemporalConsequenceScorerAgent(EpisodeDriveAgent):
    """Online EpisodeDrive wrapper; future labels are never loaded here."""

    ARTIFACT_TYPE = "episode_drive_temporal_consequence_scorer_v1"

    def __init__(self, *args, temporal_consequence_config=None, **kwargs) -> None:
        action_config = kwargs.get("action_head_config")
        if action_config is None:
            raise ValueError("action_head_config is required")
        try:
            from omegaconf import OmegaConf

            OmegaConf.update(action_config, "return_scorer_features", True, force_add=True)
            OmegaConf.update(action_config, "return_memory_fields", True, force_add=True)
        except Exception:
            setattr(action_config, "return_scorer_features", True)
            setattr(action_config, "return_memory_fields", True)
        super().__init__(*args, **kwargs)
        self._configured_consequence = dict(temporal_consequence_config or {})
        self.consequence_scorer: Optional[TemporalConsequenceRanker] = None
        self._consequence_artifact: Optional[Dict[str, object]] = None

    def _install(self, config_values: Mapping[str, object]) -> None:
        if self.consequence_scorer is None:
            self.consequence_scorer = TemporalConsequenceRanker(
                TemporalConsequenceConfig(**dict(config_values))
            )

    def initialize(self) -> None:
        if self._initialized and self.consequence_scorer is not None:
            return
        requested = Path(self.checkpoint_path) if self.checkpoint_path else None
        artifact = None
        if requested is not None and requested.is_file():
            probe = torch.load(requested, map_location="cpu")
            if probe.get("artifact_type") == self.ARTIFACT_TYPE:
                artifact = dict(probe)
            del probe
        if artifact is None:
            super().initialize()
            self._install(self._configured_consequence)
            return

        base_path = Path(str(artifact["base_checkpoint_path"]))
        if not base_path.is_file():
            raise FileNotFoundError(base_path)
        requested_path = self.checkpoint_path
        self.checkpoint_path = str(base_path)
        try:
            super().initialize()
        finally:
            self.checkpoint_path = requested_path
        self._install(artifact["model_config"])
        assert self.consequence_scorer is not None
        self.consequence_scorer.load_state_dict(artifact["model_state_dict"], strict=True)
        self._consequence_artifact = artifact
        print(f"✅ Temporal consequence scorer artifact loaded: {requested}")

    def forward(self, features, targets=None, tokens_list=None):
        prediction = super().forward(features, targets, tokens_list)
        if self.consequence_scorer is None:
            raise RuntimeError("Temporal consequence scorer was not initialized")
        base_factor_logits = torch.stack(
            [prediction["pred_logit"][key] for key in FACTOR_KEYS], dim=-1
        )
        result = self.consequence_scorer(
            prediction["scorer_candidate_features"],
            prediction["proposals"],
            base_factor_logits,
            prediction["pdm_score"],
            prediction["language_feature"],
            prediction["ego_feature"],
        )
        selected = result["selection_scores"].argmax(dim=1)
        prediction["base_pdm_score"] = prediction["pdm_score"]
        prediction["predicted_consequence"] = {
            key: result[key]
            for key in (
                "risk_logits",
                "area_logits",
                "actor_valid_logits",
                "actor_state",
            )
        }
        prediction["pdm_score"] = result["selection_scores"]
        prediction["trajectory"] = prediction["proposals"][
            torch.arange(len(selected), device=selected.device), selected
        ]
        return prediction


def build_temporal_consequence_artifact(
    model: TemporalConsequenceRanker,
    base_checkpoint_path: Path,
    *,
    metadata: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    return {
        "artifact_type": TemporalConsequenceScorerAgent.ARTIFACT_TYPE,
        "artifact_version": 1,
        "base_checkpoint_path": str(base_checkpoint_path.resolve()),
        "base_checkpoint_sha256": _sha256(base_checkpoint_path),
        "model_config": asdict(model.config),
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "metadata": dict(metadata or {}),
    }


__all__ = [
    "TemporalConsequenceConfig",
    "TemporalConsequenceRanker",
    "TemporalConsequenceScorerAgent",
    "build_temporal_consequence_artifact",
    "temporal_trajectory_features",
]
