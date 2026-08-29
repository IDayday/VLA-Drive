"""Capacity-matched six-factor probe for the Oracle Primitive Effect Gate.

Every experimental variant instantiates this exact module.  Variants differ
only in which already-shaped inputs are replaced by zeros; no variant-specific
projection head or trainable adapter is permitted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


CHECKPOINT_SCHEMA = "structured_six_factor_probe.v2"
SIX_FACTOR_ORDER = ("NC", "DAC", "DDC", "EP", "TTC", "Comfort")
BINARY_FACTOR_INDICES = (0, 1, 2, 4, 5)


def pdms_from_six_factors(factors: Tensor) -> Tensor:
    """Reassemble the registered six-factor score without synthesizing DDC."""

    if factors.ndim == 0 or factors.shape[-1] != 6:
        raise ValueError(
            "six-factor score expects final dimension 6 in order "
            f"{SIX_FACTOR_ORDER}, got {tuple(factors.shape)}"
        )
    if not torch.isfinite(factors).all():
        raise ValueError("six-factor score input contains NaN/Inf")
    nc, dac, ddc, ep, ttc, comfort = factors.unbind(dim=-1)
    return nc * dac * ddc * ((5.0 * ep + 5.0 * ttc + 2.0 * comfort) / 12.0)


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


class CrossAttentionBlock(nn.Module):
    """Pre-norm candidate-query to current-BEV cross attention."""

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.ff_norm = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, query: Tensor, memory: Tensor) -> Tensor:
        normalized_query = self.query_norm(query)
        attended, _ = self.attention(
            normalized_query,
            self.memory_norm(memory),
            self.memory_norm(memory),
            need_weights=False,
        )
        value = query + attended
        return value + self.feed_forward(self.ff_norm(value))


@dataclass(frozen=True)
class StructuredProbeConfig:
    d_model: int = 128
    num_heads: int = 4
    trajectory_layers: int = 2
    cross_attention_layers: int = 2
    auxiliary_layers: int = 2
    dropout: float = 0.0
    trajectory_steps: int = 8
    bev_tokens: int = 64
    bev_width: int = 256
    ego_status_width: int = 8
    auxiliary_tokens: int = 32
    auxiliary_width: int = 64
    factor_outputs: int = 6

    def validate(self) -> None:
        if self.d_model != 128 or self.num_heads != 4:
            raise ValueError("registered architecture requires d_model=128, num_heads=4")
        if (
            self.trajectory_layers != 2
            or self.cross_attention_layers != 2
            or self.auxiliary_layers != 2
        ):
            raise ValueError("registered architecture requires 2/2/2 encoder layers")
        if self.dropout != 0.0 or self.factor_outputs != 6:
            raise ValueError("registered architecture requires dropout=0 and six outputs")
        if (self.trajectory_steps, self.bev_tokens, self.bev_width) != (8, 64, 256):
            raise ValueError("registered trajectory/BEV shape is [8,3] and [64,256]")
        if (self.auxiliary_tokens, self.auxiliary_width) != (32, 64):
            raise ValueError("registered auxiliary shape is [32,64]")


class StructuredSixFactorProbe(nn.Module):
    """Full-spatial-BEV, trajectory-token, matched-capacity six-factor scorer."""

    def __init__(self, config: StructuredProbeConfig = StructuredProbeConfig()) -> None:
        super().__init__()
        config.validate()
        self.config = config
        d_model = config.d_model

        self.trajectory_projection = nn.Linear(3, d_model)
        self.ego_status_projection = nn.Linear(config.ego_status_width, d_model)
        trajectory_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.num_heads,
            dim_feedforward=4 * d_model,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.trajectory_encoder = nn.TransformerEncoder(
            trajectory_layer,
            num_layers=config.trajectory_layers,
            enable_nested_tensor=False,
        )

        self.current_bev_projection = nn.Linear(config.bev_width, d_model)
        self.cross_attention = nn.ModuleList(
            CrossAttentionBlock(d_model, config.num_heads, config.dropout)
            for _ in range(config.cross_attention_layers)
        )

        self.auxiliary_projection = nn.Linear(config.auxiliary_width, d_model)
        auxiliary_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.num_heads,
            dim_feedforward=4 * d_model,
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
        self.fusion = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
        )
        self.factor_head = nn.Linear(d_model, config.factor_outputs)

        self.register_buffer(
            "trajectory_time_embedding",
            _sinusoidal_positions(config.trajectory_steps, d_model),
            persistent=True,
        )
        self.register_buffer(
            "bev_position_embedding",
            _sinusoidal_positions(config.bev_tokens, d_model),
            persistent=True,
        )
        self.register_buffer(
            "auxiliary_position_embedding",
            _sinusoidal_positions(config.auxiliary_tokens, d_model),
            persistent=True,
        )

    def _validate_inputs(
        self,
        trajectory: Tensor,
        ego_status: Tensor,
        current_bev_tokens: Tensor,
        auxiliary_tokens: Tensor,
    ) -> tuple[int, int]:
        if trajectory.ndim != 4 or trajectory.shape[2:] != (8, 3):
            raise ValueError(f"trajectory expected [B,K,8,3], got {tuple(trajectory.shape)}")
        batch, candidates = trajectory.shape[:2]
        if ego_status.shape != (batch, self.config.ego_status_width):
            raise ValueError(
                f"ego_status expected [{batch},{self.config.ego_status_width}], "
                f"got {tuple(ego_status.shape)}"
            )
        if current_bev_tokens.shape != (batch, 64, 256):
            raise ValueError(
                f"current_bev_tokens expected [{batch},64,256], "
                f"got {tuple(current_bev_tokens.shape)}"
            )
        if auxiliary_tokens.shape != (batch, candidates, 32, 64):
            raise ValueError(
                f"auxiliary_tokens expected [{batch},{candidates},32,64], "
                f"got {tuple(auxiliary_tokens.shape)}"
            )
        for name, value in (
            ("trajectory", trajectory),
            ("ego_status", ego_status),
            ("current_bev_tokens", current_bev_tokens),
            ("auxiliary_tokens", auxiliary_tokens),
        ):
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains NaN/Inf")
        return batch, candidates

    def forward(
        self,
        trajectory: Tensor,
        ego_status: Tensor,
        current_bev_tokens: Tensor,
        auxiliary_tokens: Tensor,
    ) -> Mapping[str, Tensor]:
        batch, candidates = self._validate_inputs(
            trajectory, ego_status, current_bev_tokens, auxiliary_tokens
        )
        merged = batch * candidates

        trajectory_tokens = self.trajectory_projection(trajectory.reshape(merged, 8, 3))
        status = self.ego_status_projection(ego_status)
        status = status[:, None, :].expand(batch, candidates, -1).reshape(merged, 1, -1)
        trajectory_tokens = (
            trajectory_tokens
            + status
            + self.trajectory_time_embedding.to(trajectory_tokens.dtype)[None]
        )
        trajectory_tokens = self.trajectory_encoder(trajectory_tokens)

        bev = self.current_bev_projection(current_bev_tokens)
        bev = bev + self.bev_position_embedding.to(bev.dtype)[None]
        bev = (
            bev[:, None]
            .expand(batch, candidates, self.config.bev_tokens, self.config.d_model)
            .reshape(merged, self.config.bev_tokens, self.config.d_model)
        )
        for block in self.cross_attention:
            trajectory_tokens = block(trajectory_tokens, bev)

        auxiliary = self.auxiliary_projection(
            auxiliary_tokens.reshape(merged, 32, 64)
        )
        auxiliary = auxiliary + self.auxiliary_position_embedding.to(auxiliary.dtype)[None]
        auxiliary = self.auxiliary_encoder(auxiliary)

        fused = torch.cat(
            [trajectory_tokens.mean(dim=1), auxiliary.mean(dim=1)], dim=-1
        )
        logits = self.factor_head(self.fusion(fused)).reshape(batch, candidates, 6)
        factors = logits.sigmoid()
        return {
            "logits": logits,
            "factors": factors,
            "score": pdms_from_six_factors(factors),
        }


def six_factor_probe_loss(
    logits: Tensor,
    factor_labels: Tensor,
    predicted_score: Tensor,
    score_labels: Tensor,
    pair_indices: Tensor,
    pairwise_weight: float,
) -> Mapping[str, Tensor]:
    """Registered factor plus deterministic pairwise ranking objective."""

    if logits.shape != factor_labels.shape or logits.shape[-1] != 6:
        raise ValueError(
            f"six-factor logits/labels must share [B,K,6], got "
            f"{tuple(logits.shape)}/{tuple(factor_labels.shape)}"
        )
    if predicted_score.shape != score_labels.shape or predicted_score.ndim != 2:
        raise ValueError("predicted and stored score labels must share [B,K]")
    binary = torch.as_tensor(BINARY_FACTOR_INDICES, device=logits.device)
    factor_loss = F.binary_cross_entropy_with_logits(
        logits.index_select(-1, binary), factor_labels.index_select(-1, binary)
    ) + F.smooth_l1_loss(logits[..., 3].sigmoid(), factor_labels[..., 3])

    if pair_indices.ndim != 3 or pair_indices.shape[0] != logits.shape[0] or pair_indices.shape[-1] != 2:
        raise ValueError("pair_indices must be [B,P,2]")
    left, right = pair_indices[..., 0], pair_indices[..., 1]
    predicted_delta = predicted_score.gather(1, left) - predicted_score.gather(1, right)
    true_delta = score_labels.gather(1, left) - score_labels.gather(1, right)
    non_ties = true_delta.abs() > 1.0e-8
    if non_ties.any():
        targets = (true_delta[non_ties] > 0).to(predicted_delta.dtype)
        ranking_loss = F.binary_cross_entropy_with_logits(
            predicted_delta[non_ties], targets
        )
    else:
        ranking_loss = predicted_delta.sum() * 0.0
    total = factor_loss + float(pairwise_weight) * ranking_loss
    return {"total": total, "factor": factor_loss, "pairwise": ranking_loss}


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def checkpoint_payload(
    model: StructuredSixFactorProbe,
    *,
    model_type: str,
    seed: int,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "factor_order": list(SIX_FACTOR_ORDER),
        "model_type": model_type,
        "seed": int(seed),
        "architecture": asdict(model.config),
        "trainable_parameter_count": trainable_parameter_count(model),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "metadata": dict(metadata),
    }


def load_v2_checkpoint(
    path: Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[StructuredSixFactorProbe, Mapping[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"checkpoint is not a mapping: {path}")
    schema = payload.get("schema_version")
    if schema != CHECKPOINT_SCHEMA:
        raise ValueError(
            f"v2 evaluator refuses checkpoint schema {schema!r}; "
            f"required {CHECKPOINT_SCHEMA!r}"
        )
    if tuple(payload.get("factor_order", ())) != SIX_FACTOR_ORDER:
        raise ValueError("checkpoint factor order is not the registered six-factor order")
    config = StructuredProbeConfig(**dict(payload["architecture"]))
    model = StructuredSixFactorProbe(config)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, payload

