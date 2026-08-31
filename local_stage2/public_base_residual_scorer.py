"""Deployable residual fine-ranker for the released EpisodeDrive proposal bank.

The public Base model remains frozen.  This module consumes only tensors that
already exist in its inference path: proposals, factor logits, the released
trajectory-conditioned scorer token, and the released aggregate score.  The
last layers are zero initialized, so installing an untrained model preserves
the public selection exactly.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn

from navsim.agents.EpisodeDrive.episodedrive_agent import EpisodeDriveAgent


FACTOR_KEYS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "time_to_collision_within_bound",
    "ego_progress",
    "comfort",
)


@dataclass(frozen=True)
class ResidualScorerConfig:
    hidden_dim: int = 256
    trajectory_dim: int = 128
    factor_dim: int = 64
    num_blocks: int = 2
    dropout: float = 0.1
    mode: str = "local"
    score_mode: str = "residual"
    set_layers: int = 1
    set_heads: int = 8
    scene_layers: int = 2
    scene_heads: int = 8
    top_k: int = 16
    max_residual: float = 0.5
    inference_scale: float = 1.0
    switch_penalty: float = 0.0
    safety_floor: float = 0.0
    safety_relative_tolerance: float = 1.0
    preserve_ddc: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {
            "local",
            "set_aware",
            "scene_cross_attention",
            "scene_cross_attention_set",
        }:
            raise ValueError(
                "mode must be local, set_aware, scene_cross_attention, or "
                "scene_cross_attention_set"
            )
        if self.score_mode not in {"residual", "factor_aggregate", "hybrid"}:
            raise ValueError(
                "score_mode must be residual, factor_aggregate, or hybrid"
            )
        if self.hidden_dim % self.set_heads:
            raise ValueError("hidden_dim must be divisible by set_heads")
        if self.hidden_dim % self.scene_heads:
            raise ValueError("hidden_dim must be divisible by scene_heads")
        if self.scene_layers <= 0:
            raise ValueError("scene_layers must be positive")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.max_residual <= 0:
            raise ValueError("max_residual must be positive")
        if self.inference_scale < 0:
            raise ValueError("inference_scale must be non-negative")
        if self.switch_penalty < 0:
            raise ValueError("switch_penalty must be non-negative")
        if not 0.0 <= self.safety_floor <= 1.0:
            raise ValueError("safety_floor must be in [0, 1]")
        if self.safety_relative_tolerance < 0:
            raise ValueError("safety_relative_tolerance must be non-negative")


def _wrap_angle(value: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(value), torch.cos(value))


def proposal_kinematic_features(
    proposals: torch.Tensor, interval_seconds: float = 0.5
) -> torch.Tensor:
    """Build normalized, candidate-local kinematics without future labels."""

    if proposals.ndim != 4 or proposals.shape[-1] != 3:
        raise ValueError("proposals must have shape [B, K, H, 3]")
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    origin = torch.zeros_like(proposals[:, :, :1])
    states = torch.cat([origin, proposals], dim=2)
    delta_xy = states[:, :, 1:, :2] - states[:, :, :-1, :2]
    speed = torch.linalg.vector_norm(delta_xy, dim=-1) / interval_seconds
    acceleration = torch.diff(
        torch.cat([torch.zeros_like(speed[:, :, :1]), speed], dim=2), dim=2
    ) / interval_seconds
    heading_delta = _wrap_angle(states[:, :, 1:, 2] - states[:, :, :-1, 2])
    yaw_rate = heading_delta / interval_seconds
    curvature = yaw_rate / speed.clamp_min(0.5)

    features = torch.stack(
        (
            proposals[..., 0] / 30.0,
            proposals[..., 1] / 15.0,
            torch.sin(proposals[..., 2]),
            torch.cos(proposals[..., 2]),
            speed / 20.0,
            acceleration / 10.0,
            yaw_rate.clamp(-2.0, 2.0) / 2.0,
            curvature.clamp(-2.0, 2.0) / 2.0,
        ),
        dim=-1,
    )
    return features.flatten(start_dim=2)


class _ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.net(value)


class _SceneCrossAttentionBlock(nn.Module):
    """Candidate queries attend to frozen current-scene tokens independently."""

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
        self.attention_dropout = nn.Dropout(dropout)
        self.feed_forward = _ResidualBlock(hidden_dim, dropout)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(
            self.query_norm(query),
            self.memory_norm(memory),
            self.memory_norm(memory),
            need_weights=False,
        )
        return self.feed_forward(query + self.attention_dropout(attended))


def _zero_last(module: nn.Sequential) -> None:
    last = module[-1]
    if not isinstance(last, nn.Linear):
        raise TypeError("Expected a final Linear layer")
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)


def pdm_log_aggregate(factor_logits: torch.Tensor) -> torch.Tensor:
    """Match EpisodeDrive's deployment-time factor aggregation up to a constant."""

    if factor_logits.shape[-1] != len(FACTOR_KEYS):
        raise ValueError("factor_logits must end with the six EpisodeDrive factors")
    probabilities = factor_logits.sigmoid()
    additive = (
        5.0 * probabilities[..., 3]
        + 5.0 * probabilities[..., 4]
        + 2.0 * probabilities[..., 5]
    )
    return (
        torch.nn.functional.logsigmoid(factor_logits[..., 0])
        + torch.nn.functional.logsigmoid(factor_logits[..., 1])
        + additive.clamp_min(1e-8).log()
    )


class PublicBaseResidualRanker(nn.Module):
    """Candidate-local primary ranker with an optional set-aware control."""

    def __init__(self, config: ResidualScorerConfig = ResidualScorerConfig()) -> None:
        super().__init__()
        self.config = config
        trajectory_input_dim = 8 * 8
        self.trajectory_encoder = nn.Sequential(
            nn.LayerNorm(trajectory_input_dim),
            nn.Linear(trajectory_input_dim, config.trajectory_dim),
            nn.GELU(),
            nn.Linear(config.trajectory_dim, config.trajectory_dim),
        )
        # Six logits, six probabilities, and the released aggregate score.
        self.factor_encoder = nn.Sequential(
            nn.LayerNorm(13),
            nn.Linear(13, config.factor_dim),
            nn.GELU(),
            nn.Linear(config.factor_dim, config.factor_dim),
        )
        self.input_projection = nn.Linear(
            256 + config.trajectory_dim + config.factor_dim,
            config.hidden_dim,
        )
        self.local_blocks = nn.ModuleList(
            [_ResidualBlock(config.hidden_dim, config.dropout) for _ in range(config.num_blocks)]
        )
        if config.mode.startswith("scene_cross_attention"):
            self.scene_projection = nn.Sequential(
                nn.LayerNorm(256),
                nn.Linear(256, config.hidden_dim),
            )
            self.ego_projection = nn.Sequential(
                nn.LayerNorm(256),
                nn.Linear(256, config.hidden_dim),
            )
            self.scene_cross_attention = nn.ModuleList(
                [
                    _SceneCrossAttentionBlock(
                        config.hidden_dim,
                        config.scene_heads,
                        config.dropout,
                    )
                    for _ in range(config.scene_layers)
                ]
            )
        else:
            self.scene_projection = None
            self.ego_projection = None
            self.scene_cross_attention = None

        if config.mode in {"set_aware", "scene_cross_attention_set"}:
            layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.set_heads,
                dim_feedforward=config.hidden_dim * 4,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.set_context = nn.TransformerEncoder(layer, num_layers=config.set_layers)
        else:
            self.set_context = None

        self.utility_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.factor_delta_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, len(FACTOR_KEYS)),
        )
        _zero_last(self.utility_head)
        _zero_last(self.factor_delta_head)

    def forward(
        self,
        candidate_features: torch.Tensor,
        proposals: torch.Tensor,
        base_factor_logits: torch.Tensor,
        base_scores: torch.Tensor,
        scene_features: Optional[torch.Tensor] = None,
        ego_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if candidate_features.shape[:2] != base_scores.shape:
            raise ValueError("candidate feature/base score dimensions disagree")
        if base_factor_logits.shape != (*base_scores.shape, len(FACTOR_KEYS)):
            raise ValueError("base_factor_logits must have shape [B, K, 6]")
        if proposals.shape[:2] != base_scores.shape:
            raise ValueError("proposal/base score dimensions disagree")

        trajectory = self.trajectory_encoder(proposal_kinematic_features(proposals))
        clipped_logits = base_factor_logits.clamp(-12.0, 12.0)
        factor_input = torch.cat(
            [clipped_logits, clipped_logits.sigmoid(), base_scores.unsqueeze(-1)], dim=-1
        )
        factor = self.factor_encoder(factor_input)
        hidden = self.input_projection(
            torch.cat([candidate_features.float(), trajectory, factor], dim=-1)
        )
        for block in self.local_blocks:
            hidden = block(hidden)
        if self.scene_cross_attention is not None:
            if scene_features is None or ego_features is None:
                raise ValueError(
                    "scene_features and ego_features are required for "
                    f"mode={self.config.mode}"
                )
            if scene_features.ndim != 3 or scene_features.shape[0] != hidden.shape[0]:
                raise ValueError("scene_features must have shape [B, S, 256]")
            if ego_features.shape != (hidden.shape[0], 1, 256):
                raise ValueError("ego_features must have shape [B, 1, 256]")
            assert self.scene_projection is not None
            assert self.ego_projection is not None
            memory = torch.cat(
                [
                    self.scene_projection(scene_features.float()),
                    self.ego_projection(ego_features.float()),
                ],
                dim=1,
            )
            for block in self.scene_cross_attention:
                hidden = block(hidden, memory)
        if self.set_context is not None:
            hidden = self.set_context(hidden)

        utility_delta = self.config.max_residual * torch.tanh(
            self.utility_head(hidden).squeeze(-1)
        )
        refined_factor_logits = base_factor_logits + self.factor_delta_head(hidden)
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
        top_k = min(self.config.top_k, base_scores.shape[1])
        top_indices = base_scores.topk(top_k, dim=1).indices
        top_k_mask = torch.zeros_like(base_scores, dtype=torch.bool)
        top_k_mask.scatter_(1, top_indices, True)
        eligible = top_k_mask.clone()
        base_selected = torch.zeros_like(base_scores, dtype=torch.bool)
        base_indices = base_scores.argmax(dim=1, keepdim=True)
        base_selected.scatter_(1, base_indices, True)
        safety_indices = [0, 1, 3]
        if self.config.preserve_ddc:
            safety_indices.append(2)
        safety_probabilities = refined_factor_logits.sigmoid()[..., safety_indices]
        base_safety = safety_probabilities.gather(
            1,
            base_indices[..., None].expand(-1, 1, len(safety_indices)),
        )
        safe = (safety_probabilities >= self.config.safety_floor).all(dim=-1)
        safe &= (
            safety_probabilities
            >= base_safety - self.config.safety_relative_tolerance
        ).all(dim=-1)
        eligible &= safe
        eligible |= base_selected
        adjusted_scores = refined_scores - (
            (~base_selected).to(refined_scores.dtype) * self.config.switch_penalty
        )
        # Keep scores finite for audit serialization while making candidates
        # outside the frozen Base top-K ineligible for final selection.
        selection_scores = torch.where(eligible, adjusted_scores, base_scores - 100.0)
        return {
            "residual": score_delta,
            "utility_delta": utility_delta,
            "factor_score_delta": factor_score_delta,
            "refined_scores": refined_scores,
            "selection_scores": selection_scores,
            "refined_factor_logits": refined_factor_logits,
            "top_k_mask": top_k_mask,
            "eligible_mask": eligible,
            "hidden": hidden,
        }


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


class PublicBaseResidualScorerAgent(EpisodeDriveAgent):
    """EpisodeDrive inference wrapper for a trained small residual artifact."""

    ARTIFACT_TYPE = "episode_drive_public_base_residual_scorer_v1"

    def __init__(self, *args, residual_scorer_config=None, **kwargs) -> None:
        action_config = kwargs.get("action_head_config")
        if action_config is None:
            raise ValueError("action_head_config is required")
        # DictConfig permits force-added fields only via OmegaConf.update.
        try:
            from omegaconf import OmegaConf

            OmegaConf.update(
                action_config, "return_scorer_features", True, force_add=True
            )
            OmegaConf.update(
                action_config, "return_memory_fields", True, force_add=True
            )
        except Exception:
            setattr(action_config, "return_scorer_features", True)
            setattr(action_config, "return_memory_fields", True)
        super().__init__(*args, **kwargs)
        self._configured_residual = dict(residual_scorer_config or {})
        self.residual_scorer: Optional[PublicBaseResidualRanker] = None
        self._residual_artifact: Optional[Dict[str, object]] = None

    @staticmethod
    def _load_payload(path: Path) -> Mapping[str, object]:
        return torch.load(path, map_location="cpu")

    def _install_residual(self, config_values: Mapping[str, object]) -> None:
        if self.residual_scorer is not None:
            return
        config = ResidualScorerConfig(**dict(config_values))
        self.residual_scorer = PublicBaseResidualRanker(config)

    def initialize(self) -> None:
        if self._initialized and self.residual_scorer is not None:
            return
        requested = Path(self.checkpoint_path) if self.checkpoint_path else None
        artifact = None
        if requested is not None and requested.is_file():
            probe = self._load_payload(requested)
            if probe.get("artifact_type") == self.ARTIFACT_TYPE:
                artifact = dict(probe)
            del probe

        if artifact is None:
            # Public full checkpoint: strict base restore first, then install a
            # zero-output residual whose initial selection is exactly public.
            super().initialize()
            self._install_residual(self._configured_residual)
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
        self._install_residual(artifact["model_config"])
        assert self.residual_scorer is not None
        self.residual_scorer.load_state_dict(artifact["model_state_dict"], strict=True)
        self._residual_artifact = artifact
        print(f"✅ Residual scorer artifact loaded: {requested}")

    def forward(self, features, targets=None, tokens_list=None):
        prediction = super().forward(features, targets, tokens_list)
        if self.residual_scorer is None:
            raise RuntimeError("Residual scorer agent was not initialized")
        base_factor_logits = torch.stack(
            [prediction["pred_logit"][key] for key in FACTOR_KEYS], dim=-1
        )
        base_scores = prediction["pdm_score"]
        residual = self.residual_scorer(
            prediction["scorer_candidate_features"],
            prediction["proposals"],
            base_factor_logits,
            base_scores,
            prediction.get("language_feature"),
            prediction.get("ego_feature"),
        )
        selected = residual["selection_scores"].argmax(dim=1)
        prediction["base_pdm_score"] = base_scores
        prediction["residual_score"] = residual["residual"]
        prediction["residual_eligible_mask"] = residual["eligible_mask"]
        prediction["pdm_score"] = residual["selection_scores"]
        prediction["trajectory"] = prediction["proposals"][
            torch.arange(len(selected), device=selected.device), selected
        ]
        return prediction


def build_residual_artifact(
    model: PublicBaseResidualRanker,
    base_checkpoint_path: Path,
    *,
    metadata: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    return {
        "artifact_type": PublicBaseResidualScorerAgent.ARTIFACT_TYPE,
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
    "FACTOR_KEYS",
    "PublicBaseResidualRanker",
    "PublicBaseResidualScorerAgent",
    "ResidualScorerConfig",
    "build_residual_artifact",
    "proposal_kinematic_features",
]
