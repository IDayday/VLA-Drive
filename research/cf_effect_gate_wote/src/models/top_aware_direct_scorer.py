"""Top-aware, current-only Direct Scorer with an unbounded ranking utility."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .structured_six_factor_probe import CrossAttentionBlock, pdms_from_six_factors


CHECKPOINT_SCHEMA = "top_aware_direct_scorer.v3"
SIX_FACTOR_ORDER = ("NC", "DAC", "DDC", "EP", "TTC", "Comfort")
REPRESENTATIONS = (
    "trajectory_only",
    "old_spatial_xattn",
    "pretrained_candidate_query",
    "path_aligned_current",
    "hybrid_current",
    "wote_current_only_rollout",
)


def _sinusoidal_positions(values: Tensor, width: int) -> Tensor:
    values = values.to(dtype=torch.float32).reshape(-1, 1)
    half = max(width // 2, 1)
    scale = torch.exp(
        torch.arange(half, dtype=torch.float32, device=values.device)
        * (-torch.log(torch.tensor(10_000.0, device=values.device)) / max(half - 1, 1))
    )
    angles = values * scale.unsqueeze(0)
    embedding = torch.cat([angles.sin(), angles.cos()], dim=1)
    return F.pad(embedding[:, :width], (0, max(0, width - embedding.shape[1])))


def fixed_2d_bev_position_embedding(
    height: int,
    width: int,
    channels: int,
) -> Tensor:
    """Return row/column-aware fixed positions in row-major token order."""

    if height <= 0 or width <= 0 or channels <= 0:
        raise ValueError("2D position dimensions must be positive")
    row_width = channels // 2
    column_width = channels - row_width
    rows = torch.arange(height, dtype=torch.float32)
    columns = torch.arange(width, dtype=torch.float32)
    row_embedding = _sinusoidal_positions(rows, row_width)
    column_embedding = _sinusoidal_positions(columns, column_width)
    grid = torch.cat(
        [
            row_embedding[:, None, :].expand(height, width, row_width),
            column_embedding[None, :, :].expand(height, width, column_width),
        ],
        dim=-1,
    )
    return grid.reshape(height * width, channels)


class CandidateToBEVGrid(nn.Module):
    """Exact 8x8 coordinate contract used by WoTE ego-feature injection.

    WoTE maps local x to the BEV row with ``row=x*H/32`` and local y to
    the column with ``column=y*W/64+W/2``.  This class centralizes that
    mapping for both bilinear path sampling and masked 3x3 tube pooling.
    """

    def __init__(self, height: int = 8, width: int = 8) -> None:
        super().__init__()
        if (height, width) != (8, 8):
            raise ValueError("registered WoTE current BEV grid is exactly 8x8")
        self.height = int(height)
        self.width = int(width)

    def continuous_indices(self, xy: Tensor) -> tuple[Tensor, Tensor]:
        if xy.shape[-1] != 2:
            raise ValueError(f"candidate coordinates require [...,2], got {xy.shape}")
        row = xy[..., 0] * (self.height / 32.0)
        column = xy[..., 1] * (self.width / 64.0) + (self.width / 2.0)
        return row, column

    def bilinear_weights_indices(
        self, xy: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        row, column = self.continuous_indices(xy)
        row0 = torch.floor(row).to(torch.long)
        column0 = torch.floor(column).to(torch.long)
        row1 = row0 + 1
        column1 = column0 + 1
        dr = row - row0.to(row.dtype)
        dc = column - column0.to(column.dtype)
        rows = torch.stack([row0, row0, row1, row1], dim=-1)
        columns = torch.stack([column0, column1, column0, column1], dim=-1)
        weights = torch.stack(
            [(1.0 - dr) * (1.0 - dc), (1.0 - dr) * dc, dr * (1.0 - dc), dr * dc],
            dim=-1,
        )
        valid = (
            (rows >= 0)
            & (rows < self.height)
            & (columns >= 0)
            & (columns < self.width)
        )
        return rows, columns, weights, valid

    def sample(self, bev_tokens: Tensor, xy: Tensor) -> tuple[Tensor, Tensor]:
        """Bilinearly sample ``[B,64,C]`` for ``xy=[B,K,T,2]``."""

        if bev_tokens.ndim != 3 or bev_tokens.shape[1] != self.height * self.width:
            raise ValueError(
                f"current BEV expected [B,{self.height * self.width},C], got {bev_tokens.shape}"
            )
        if xy.ndim != 4 or xy.shape[0] != bev_tokens.shape[0] or xy.shape[-1] != 2:
            raise ValueError("candidate xy must be [B,K,T,2] and share batch with BEV")
        rows, columns, weights, valid = self.bilinear_weights_indices(xy)
        linear = (rows * self.width + columns).clamp(
            min=0, max=self.height * self.width - 1
        )
        batch, candidates, steps, neighbors = linear.shape
        channels = bev_tokens.shape[-1]
        expanded = bev_tokens[:, None, None, :, :].expand(
            batch, candidates, steps, self.height * self.width, channels
        )
        sampled = torch.gather(
            expanded,
            3,
            linear[..., None].expand(batch, candidates, steps, neighbors, channels),
        )
        masked_weights = weights * valid.to(weights.dtype)
        output = (sampled * masked_weights[..., None]).sum(dim=3)
        has_support = valid.any(dim=-1)
        return output, has_support

    def tube_pool(self, bev_tokens: Tensor, xy: Tensor) -> tuple[Tensor, Tensor]:
        """Masked local 3x3 mean around the nearest injection-contract cell."""

        if bev_tokens.ndim != 3 or bev_tokens.shape[1] != 64:
            raise ValueError("current BEV must be [B,64,C]")
        row, column = self.continuous_indices(xy)
        center_row = torch.floor(row + 0.5).to(torch.long)
        center_column = torch.floor(column + 0.5).to(torch.long)
        offsets = torch.tensor(
            [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)],
            device=xy.device,
            dtype=torch.long,
        )
        rows = center_row[..., None] + offsets[:, 0]
        columns = center_column[..., None] + offsets[:, 1]
        valid = (
            (rows >= 0)
            & (rows < self.height)
            & (columns >= 0)
            & (columns < self.width)
        )
        linear = (rows * self.width + columns).clamp(min=0, max=63)
        batch, candidates, steps, neighbors = linear.shape
        channels = bev_tokens.shape[-1]
        expanded = bev_tokens[:, None, None, :, :].expand(
            batch, candidates, steps, 64, channels
        )
        values = torch.gather(
            expanded,
            3,
            linear[..., None].expand(batch, candidates, steps, neighbors, channels),
        )
        mask = valid.to(values.dtype)[..., None]
        count = mask.sum(dim=3).clamp_min(1.0)
        pooled = (values * mask).sum(dim=3) / count
        return pooled, valid.any(dim=-1)


def deterministic_trajectory_features(trajectory: Tensor, interval: float = 0.5) -> Tensor:
    """Compute only candidate-determined local kinematics, never future state."""

    if trajectory.ndim != 4 or trajectory.shape[2:] != (8, 3):
        raise ValueError(f"trajectory expected [B,K,8,3], got {trajectory.shape}")
    if interval <= 0:
        raise ValueError("trajectory interval must be positive")
    xy = trajectory[..., :2]
    heading = trajectory[..., 2]
    origin = torch.zeros_like(xy[..., :1, :])
    delta_xy = torch.diff(torch.cat([origin, xy], dim=-2), dim=-2)
    speed = torch.linalg.vector_norm(delta_xy, dim=-1) / interval
    initial_speed = torch.zeros_like(speed[..., :1])
    acceleration = torch.diff(torch.cat([initial_speed, speed], dim=-1), dim=-1) / interval
    initial_heading = torch.zeros_like(heading[..., :1])
    delta_heading = torch.atan2(
        torch.sin(torch.diff(torch.cat([initial_heading, heading], dim=-1), dim=-1)),
        torch.cos(torch.diff(torch.cat([initial_heading, heading], dim=-1), dim=-1)),
    )
    curvature = delta_heading / torch.linalg.vector_norm(delta_xy, dim=-1).clamp_min(1.0e-3)
    return torch.stack(
        [
            xy[..., 0],
            xy[..., 1],
            heading.sin(),
            heading.cos(),
            speed,
            acceleration,
            curvature,
        ],
        dim=-1,
    )


@dataclass(frozen=True)
class TopAwareDirectScorerConfig:
    representation: str = "hybrid_current"
    d_model: int = 128
    num_heads: int = 4
    trajectory_layers: int = 2
    cross_attention_layers: int = 2
    dropout: float = 0.0
    candidate_feature_width: int = 256
    bev_width: int = 256
    ego_status_width: int = 8
    trajectory_steps: int = 8

    def validate(self) -> None:
        if self.representation not in REPRESENTATIONS:
            raise ValueError(f"unknown Direct representation: {self.representation}")
        if self.d_model != 128 or self.num_heads != 4:
            raise ValueError("registered V3 requires d_model=128 and num_heads=4")
        if self.trajectory_layers != 2 or self.cross_attention_layers != 2:
            raise ValueError("registered V3 requires two trajectory/cross-attention layers")
        if self.dropout != 0.0:
            raise ValueError("registered V3 requires dropout=0 for chunk invariance")
        if (self.bev_width, self.candidate_feature_width) != (256, 256):
            raise ValueError("registered frozen WoTE features have width 256")
        if (self.ego_status_width, self.trajectory_steps) != (8, 8):
            raise ValueError("registered ego/trajectory shapes are 8 and 8")


class TopAwareDirectScorerV3(nn.Module):
    """Current-only scorer whose ranking head is not bounded to [0,1]."""

    def __init__(
        self,
        config: TopAwareDirectScorerConfig = TopAwareDirectScorerConfig(),
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        width = config.d_model
        self.grid = CandidateToBEVGrid()

        raw_width = 3 if config.representation == "old_spatial_xattn" else 7
        self.trajectory_projection = nn.Linear(raw_width, width)
        self.ego_projection = nn.Linear(config.ego_status_width, width)
        trajectory_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=config.num_heads,
            dim_feedforward=4 * width,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.trajectory_encoder = nn.TransformerEncoder(
            trajectory_layer,
            num_layers=config.trajectory_layers,
            enable_nested_tensor=False,
        )

        self.uses_global_bev = config.representation in {
            "old_spatial_xattn",
            "pretrained_candidate_query",
            "hybrid_current",
        }
        self.uses_candidate_query = config.representation in {
            "pretrained_candidate_query",
            "hybrid_current",
        }
        self.uses_path = config.representation in {
            "path_aligned_current",
            "hybrid_current",
            "wote_current_only_rollout",
        }
        self.uses_current = config.representation != "trajectory_only"

        if self.uses_global_bev:
            self.current_bev_projection = nn.Linear(config.bev_width, width)
            self.cross_attention = nn.ModuleList(
                CrossAttentionBlock(width, config.num_heads, 0.0)
                for _ in range(config.cross_attention_layers)
            )
        else:
            self.current_bev_projection = None
            self.cross_attention = nn.ModuleList()

        self.candidate_projection = (
            nn.Linear(config.candidate_feature_width, width)
            if self.uses_candidate_query
            else None
        )
        self.path_projection = (
            nn.Linear(2 * config.bev_width, width) if self.uses_path else None
        )
        self.current_only_step_encoder = (
            nn.GRU(width, width, batch_first=True)
            if config.representation == "wote_current_only_rollout"
            else None
        )

        # R0 retains the old constant-zero auxiliary branch so D1 changes the
        # objective/heads while preserving the legacy representation capacity.
        if config.representation == "old_spatial_xattn":
            self.legacy_auxiliary_projection = nn.Linear(64, width)
            auxiliary_layer = nn.TransformerEncoderLayer(
                d_model=width,
                nhead=config.num_heads,
                dim_feedforward=4 * width,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.legacy_auxiliary_encoder = nn.TransformerEncoder(
                auxiliary_layer, num_layers=2, enable_nested_tensor=False
            )
            fusion_parts = 2
        else:
            self.legacy_auxiliary_projection = None
            self.legacy_auxiliary_encoder = None
            fusion_parts = 1 + int(self.uses_candidate_query) + int(self.uses_path)
            if self.uses_global_bev:
                fusion_parts += 1

        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_parts * width),
            nn.Linear(fusion_parts * width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, width),
            nn.GELU(),
        )
        self.factor_head = nn.Linear(width, 6)
        self.utility_head = nn.Linear(width, 1)
        self.hard_safety_head = nn.Linear(width, 1)

        if config.representation == "old_spatial_xattn":
            bev_position = _sinusoidal_positions(torch.arange(64), width)
        else:
            bev_position = fixed_2d_bev_position_embedding(8, 8, width)
        self.register_buffer("bev_position_embedding", bev_position, persistent=True)
        self.register_buffer(
            "trajectory_time_embedding",
            _sinusoidal_positions(torch.arange(8), width),
            persistent=True,
        )
        self.register_buffer(
            "legacy_auxiliary_position_embedding",
            _sinusoidal_positions(torch.arange(32), width),
            persistent=config.representation == "old_spatial_xattn",
        )

        if trainable_parameter_count(self) > 3_000_000:
            raise ValueError(
                f"V3 exceeds 3M trainable parameters: {trainable_parameter_count(self)}"
            )

    def _validate_inputs(
        self,
        trajectory: Tensor,
        ego_status: Tensor,
        current_bev_tokens: Tensor,
        candidate_current_feature: Tensor | None,
    ) -> tuple[int, int]:
        if trajectory.ndim != 4 or trajectory.shape[2:] != (8, 3):
            raise ValueError("trajectory must be [B,K,8,3]")
        batch, candidates = trajectory.shape[:2]
        if ego_status.shape != (batch, 8):
            raise ValueError(f"ego status must be [{batch},8]")
        if current_bev_tokens.shape != (batch, 64, 256):
            raise ValueError(f"current BEV must be [{batch},64,256]")
        if self.uses_candidate_query and (
            candidate_current_feature is None
            or candidate_current_feature.shape != (batch, candidates, 256)
        ):
            raise ValueError(
                f"{self.config.representation} requires candidate_current_feature "
                f"[{batch},{candidates},256]"
            )
        values = [trajectory, ego_status, current_bev_tokens]
        if candidate_current_feature is not None:
            values.append(candidate_current_feature)
        if not all(torch.isfinite(value).all() for value in values):
            raise ValueError("Direct V3 input contains NaN/Inf")
        return batch, candidates

    def encode_current_bev(self, current_bev_tokens: Tensor) -> Tensor | None:
        """Encode current BEV once per scene, before candidate chunking."""

        if not self.uses_current:
            return None
        if self.current_bev_projection is None:
            return current_bev_tokens
        value = self.current_bev_projection(current_bev_tokens)
        return value + self.bev_position_embedding.to(value.dtype)[None]

    def _trajectory_summary(self, trajectory: Tensor, ego_status: Tensor) -> tuple[Tensor, Tensor]:
        batch, candidates = trajectory.shape[:2]
        merged = batch * candidates
        raw = (
            trajectory
            if self.config.representation == "old_spatial_xattn"
            else deterministic_trajectory_features(trajectory)
        )
        tokens = self.trajectory_projection(raw.reshape(merged, 8, -1))
        status = self.ego_projection(ego_status)[:, None, :].expand(
            batch, candidates, -1
        )
        tokens = (
            tokens
            + status.reshape(merged, 1, -1)
            + self.trajectory_time_embedding.to(tokens.dtype)[None]
        )
        tokens = self.trajectory_encoder(tokens)
        return tokens, tokens.mean(dim=1).reshape(batch, candidates, -1)

    def _forward_candidate_chunk(
        self,
        trajectory: Tensor,
        ego_status: Tensor,
        current_bev_tokens: Tensor,
        encoded_bev: Tensor | None,
        candidate_current_feature: Tensor | None,
    ) -> Mapping[str, Tensor]:
        batch, candidates = trajectory.shape[:2]
        width = self.config.d_model
        trajectory_tokens, trajectory_summary = self._trajectory_summary(
            trajectory, ego_status
        )
        parts: list[Tensor] = [trajectory_summary]

        if self.config.representation == "old_spatial_xattn":
            if encoded_bev is None:
                raise ValueError("R0 requires current BEV")
            memory = encoded_bev[:, None].expand(batch, candidates, 64, width).reshape(
                batch * candidates, 64, width
            )
            attended = trajectory_tokens
            for block in self.cross_attention:
                attended = block(attended, memory)
            zeros = torch.zeros(
                batch * candidates,
                32,
                64,
                device=trajectory.device,
                dtype=trajectory.dtype,
            )
            auxiliary = self.legacy_auxiliary_projection(zeros)
            auxiliary = (
                auxiliary
                + self.legacy_auxiliary_position_embedding.to(auxiliary.dtype)[None]
            )
            auxiliary = self.legacy_auxiliary_encoder(auxiliary)
            parts = [
                attended.mean(dim=1).reshape(batch, candidates, width),
                auxiliary.mean(dim=1).reshape(batch, candidates, width),
            ]
        else:
            if self.uses_candidate_query:
                assert self.candidate_projection is not None
                assert candidate_current_feature is not None
                candidate = self.candidate_projection(candidate_current_feature)
                parts.append(candidate)
            if self.uses_global_bev:
                if encoded_bev is None:
                    raise ValueError("global current attention requires encoded BEV")
                query = (
                    parts[-1] if self.uses_candidate_query else trajectory_summary
                ).reshape(batch * candidates, 1, width)
                memory = encoded_bev[:, None].expand(
                    batch, candidates, 64, width
                ).reshape(batch * candidates, 64, width)
                for block in self.cross_attention:
                    query = block(query, memory)
                parts.append(query.reshape(batch, candidates, width))
            if self.uses_path:
                assert self.path_projection is not None
                path, _ = self.grid.sample(current_bev_tokens, trajectory[..., :2])
                tube, _ = self.grid.tube_pool(current_bev_tokens, trajectory[..., :2])
                path_tokens = self.path_projection(torch.cat([path, tube], dim=-1))
                if self.current_only_step_encoder is not None:
                    flattened = path_tokens.reshape(batch * candidates, 8, width)
                    path_tokens, _ = self.current_only_step_encoder(flattened)
                    path_tokens = path_tokens.reshape(batch, candidates, 8, width)
                parts.append(path_tokens.mean(dim=2))

        fused = self.fusion(torch.cat(parts, dim=-1))
        factor_logits = self.factor_head(fused)
        factors = factor_logits.sigmoid()
        utility_logit = self.utility_head(fused).squeeze(-1)
        hard_safety_logit = self.hard_safety_head(fused).squeeze(-1)
        return {
            "candidate_embedding": fused,
            "factor_logits": factor_logits,
            "factors": factors,
            "factor_score": pdms_from_six_factors(factors),
            "utility_logit": utility_logit,
            "utility_score": utility_logit.sigmoid(),
            "hard_safety_logit": hard_safety_logit,
        }

    def forward(
        self,
        trajectory: Tensor,
        ego_status: Tensor,
        current_bev_tokens: Tensor,
        candidate_current_feature: Tensor | None = None,
        *,
        candidate_chunk: int | None = None,
    ) -> Mapping[str, Tensor]:
        batch, candidates = self._validate_inputs(
            trajectory, ego_status, current_bev_tokens, candidate_current_feature
        )
        chunk = candidates if candidate_chunk is None else int(candidate_chunk)
        if chunk <= 0:
            raise ValueError("candidate_chunk must be positive")
        encoded_bev = self.encode_current_bev(current_bev_tokens)
        outputs: dict[str, list[Tensor]] = {}
        for start in range(0, candidates, chunk):
            stop = min(start + chunk, candidates)
            result = self._forward_candidate_chunk(
                trajectory[:, start:stop],
                ego_status,
                current_bev_tokens,
                encoded_bev,
                (
                    None
                    if candidate_current_feature is None
                    else candidate_current_feature[:, start:stop]
                ),
            )
            for key, value in result.items():
                outputs.setdefault(key, []).append(value)
        return {key: torch.cat(values, dim=1) for key, values in outputs.items()}


def selection_logit(outputs: Mapping[str, Tensor], lambda_safe: float) -> Tensor:
    if float(lambda_safe) < 0:
        raise ValueError("lambda_safe must be non-negative")
    utility = outputs["utility_logit"]
    safety = outputs["hard_safety_logit"]
    if utility.shape != safety.shape:
        raise ValueError("utility and hard-safety logits must share [B,K]")
    return utility + float(lambda_safe) * F.logsigmoid(safety)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def checkpoint_payload(
    model: TopAwareDirectScorerV3,
    *,
    seed: int,
    objective: Mapping[str, Any],
    selection: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "factor_order": list(SIX_FACTOR_ORDER),
        "architecture": asdict(model.config),
        "objective": dict(objective),
        "selection": dict(selection),
        "seed": int(seed),
        "trainable_parameter_count": trainable_parameter_count(model),
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "metadata": dict(metadata),
    }


def load_v3_checkpoint(
    path: Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[TopAwareDirectScorerV3, Mapping[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"V3 checkpoint is not a mapping: {path}")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError(
            f"V3 loader refuses schema {payload.get('schema_version')!r}; "
            f"required {CHECKPOINT_SCHEMA!r}"
        )
    if tuple(payload.get("factor_order", ())) != SIX_FACTOR_ORDER:
        raise ValueError("V3 checkpoint six-factor order changed")
    model = TopAwareDirectScorerV3(
        TopAwareDirectScorerConfig(**dict(payload["architecture"]))
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    if trainable_parameter_count(model) != int(payload["trainable_parameter_count"]):
        raise ValueError("V3 checkpoint parameter-count audit failed")
    return model, payload
