"""Capacity-matched Oracle Effect probe with the rehabilitated Direct backbone.

All A--L variants instantiate this exact module.  The variant is expressed only
through zeroed or packed inputs.  A zero-initialized residual effect adapter
makes initialization exactly reproduce the independently validated hybrid
current-only Direct scorer before any Oracle Effect optimization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .structured_six_factor_probe import SIX_FACTOR_ORDER, pdms_from_six_factors
from .top_aware_direct_scorer import (
    TopAwareDirectScorerConfig,
    TopAwareDirectScorerV3,
    load_v3_checkpoint,
)


CHECKPOINT_SCHEMA = "matched_hybrid_oracle_effect_probe.v3"


def _sinusoidal_positions(length: int, width: int) -> Tensor:
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    half = max(width // 2, 1)
    scale = torch.exp(
        torch.arange(half, dtype=torch.float32)
        * (-torch.log(torch.tensor(10_000.0)) / max(half - 1, 1))
    )
    values = positions * scale.unsqueeze(0)
    embedding = torch.cat([values.sin(), values.cos()], dim=1)
    return F.pad(embedding[:, :width], (0, max(0, width - embedding.shape[1])))


@dataclass(frozen=True)
class MatchedHybridProbeConfig:
    d_model: int = 128
    num_heads: int = 4
    auxiliary_layers: int = 2
    auxiliary_tokens: int = 32
    auxiliary_width: int = 64
    candidate_chunk: int = 64
    dropout: float = 0.0

    def validate(self) -> None:
        expected = (128, 4, 2, 32, 64, 0.0)
        actual = (
            self.d_model,
            self.num_heads,
            self.auxiliary_layers,
            self.auxiliary_tokens,
            self.auxiliary_width,
            self.dropout,
        )
        if actual != expected:
            raise ValueError(
                "registered matched hybrid probe requires "
                "d_model=128, heads=4, auxiliary_layers=2, shape=[32,64], dropout=0"
            )
        if self.candidate_chunk <= 0:
            raise ValueError("candidate_chunk must be positive")


class MatchedHybridOracleEffectProbe(nn.Module):
    """Hybrid current scorer plus a shared residual auxiliary interaction path."""

    def __init__(
        self,
        config: MatchedHybridProbeConfig = MatchedHybridProbeConfig(),
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        width = config.d_model
        self.direct = TopAwareDirectScorerV3(
            TopAwareDirectScorerConfig(representation="hybrid_current")
        )
        self.auxiliary_projection = nn.Linear(config.auxiliary_width, width)
        auxiliary_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=config.num_heads,
            dim_feedforward=4 * width,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.auxiliary_encoder = nn.TransformerEncoder(
            auxiliary_layer,
            num_layers=config.auxiliary_layers,
            enable_nested_tensor=False,
        )
        self.effect_interaction = nn.Sequential(
            nn.LayerNorm(2 * width),
            nn.Linear(2 * width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, width),
        )
        # Exact inheritance contract: before training, any auxiliary input has
        # zero residual and the model is bitwise-equivalent to Direct V3.
        nn.init.zeros_(self.effect_interaction[-1].weight)
        nn.init.zeros_(self.effect_interaction[-1].bias)
        self.register_buffer(
            "auxiliary_position_embedding",
            _sinusoidal_positions(config.auxiliary_tokens, width),
            persistent=True,
        )
        self._initialized_from_direct = False

    def initialize_from_direct_checkpoint(
        self,
        checkpoint: Path,
    ) -> Mapping[str, Any]:
        source, payload = load_v3_checkpoint(checkpoint, map_location="cpu")
        if source.config.representation != "hybrid_current":
            raise ValueError("matched Oracle probe requires a hybrid_current Direct checkpoint")
        if tuple(payload.get("factor_order", ())) != SIX_FACTOR_ORDER:
            raise ValueError("Direct initialization is not the independent six-factor model")
        self.direct.load_state_dict(source.state_dict(), strict=True)
        self._initialized_from_direct = True
        return payload

    def _validate_inputs(
        self,
        trajectory: Tensor,
        ego_status: Tensor,
        current_bev_tokens: Tensor,
        candidate_current_feature: Tensor,
        auxiliary_tokens: Tensor,
    ) -> tuple[int, int]:
        if trajectory.ndim != 4 or trajectory.shape[2:] != (8, 3):
            raise ValueError("trajectory must be [B,K,8,3]")
        batch, candidates = trajectory.shape[:2]
        expected = {
            "ego_status": (batch, 8),
            "current_bev_tokens": (batch, 64, 256),
            "candidate_current_feature": (batch, candidates, 256),
            "auxiliary_tokens": (batch, candidates, 32, 64),
        }
        actual = {
            "ego_status": tuple(ego_status.shape),
            "current_bev_tokens": tuple(current_bev_tokens.shape),
            "candidate_current_feature": tuple(candidate_current_feature.shape),
            "auxiliary_tokens": tuple(auxiliary_tokens.shape),
        }
        for name, shape in expected.items():
            if actual[name] != shape:
                raise ValueError(f"{name} expected {shape}, got {actual[name]}")
        values = (
            trajectory,
            ego_status,
            current_bev_tokens,
            candidate_current_feature,
            auxiliary_tokens,
        )
        if not all(torch.isfinite(value).all() for value in values):
            raise ValueError("matched Oracle input contains NaN/Inf")
        return batch, candidates

    def _effect_residual(self, direct_hidden: Tensor, auxiliary_tokens: Tensor) -> Tensor:
        batch, candidates = direct_hidden.shape[:2]
        width = self.config.d_model
        outputs: list[Tensor] = []
        for start in range(0, candidates, self.config.candidate_chunk):
            stop = min(start + self.config.candidate_chunk, candidates)
            count = stop - start
            auxiliary = self.auxiliary_projection(
                auxiliary_tokens[:, start:stop].reshape(
                    batch * count,
                    self.config.auxiliary_tokens,
                    self.config.auxiliary_width,
                )
            )
            auxiliary = (
                auxiliary
                + self.auxiliary_position_embedding.to(auxiliary.dtype)[None]
            )
            auxiliary = self.auxiliary_encoder(auxiliary).mean(dim=1)
            auxiliary = auxiliary.reshape(batch, count, width)
            outputs.append(
                self.effect_interaction(
                    torch.cat([direct_hidden[:, start:stop], auxiliary], dim=-1)
                )
            )
        return torch.cat(outputs, dim=1)

    def forward(
        self,
        trajectory: Tensor,
        ego_status: Tensor,
        current_bev_tokens: Tensor,
        candidate_current_feature: Tensor,
        auxiliary_tokens: Tensor,
    ) -> Mapping[str, Tensor]:
        self._validate_inputs(
            trajectory,
            ego_status,
            current_bev_tokens,
            candidate_current_feature,
            auxiliary_tokens,
        )
        direct = self.direct(
            trajectory,
            ego_status,
            current_bev_tokens,
            candidate_current_feature,
            candidate_chunk=self.config.candidate_chunk,
        )
        hidden = direct["candidate_embedding"]
        hidden = hidden + self._effect_residual(hidden, auxiliary_tokens)
        factor_logits = self.direct.factor_head(hidden)
        factors = factor_logits.sigmoid()
        utility_logit = self.direct.utility_head(hidden).squeeze(-1)
        hard_safety_logit = self.direct.hard_safety_head(hidden).squeeze(-1)
        score = pdms_from_six_factors(factors)
        return {
            "candidate_embedding": hidden,
            "logits": factor_logits,
            "factor_logits": factor_logits,
            "factors": factors,
            "score": score,
            "factor_score": score,
            "utility_logit": utility_logit,
            "utility_score": utility_logit.sigmoid(),
            "hard_safety_logit": hard_safety_logit,
        }


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def checkpoint_payload(
    model: MatchedHybridOracleEffectProbe,
    *,
    model_type: str,
    seed: int,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    if not model._initialized_from_direct:
        raise ValueError("refusing checkpoint not initialized from validated Direct V3")
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "factor_order": list(SIX_FACTOR_ORDER),
        "model_type": str(model_type),
        "seed": int(seed),
        "architecture": asdict(model.config),
        "trainable_parameter_count": trainable_parameter_count(model),
        "direct_initialization_required": True,
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "metadata": dict(metadata),
    }


def load_matched_v3_checkpoint(
    path: Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[MatchedHybridOracleEffectProbe, Mapping[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"matched checkpoint is not a mapping: {path}")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError(
            f"matched loader refuses schema {payload.get('schema_version')!r}; "
            f"required {CHECKPOINT_SCHEMA!r}"
        )
    if tuple(payload.get("factor_order", ())) != SIX_FACTOR_ORDER:
        raise ValueError("matched checkpoint factor order changed")
    model = MatchedHybridOracleEffectProbe(
        MatchedHybridProbeConfig(**dict(payload["architecture"]))
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model._initialized_from_direct = True
    if trainable_parameter_count(model) != int(payload["trainable_parameter_count"]):
        raise ValueError("matched checkpoint parameter-count audit failed")
    return model, payload
