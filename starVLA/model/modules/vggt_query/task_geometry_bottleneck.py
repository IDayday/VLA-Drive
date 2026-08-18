"""Planning-conditioned bottleneck over dense offline VGGT patch features."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn


class PlanningConditionedDenseVGGTBottleneck(nn.Module):
    """Read dense VGGT memory into 32 task-aware tokens and eight readouts.

    Qwen planning states construct the attention queries, but the returned task
    geometry tokens are the cross-attention output itself.  There is
    intentionally no ``queries + attention_output`` residual that could let a
    downstream auxiliary head bypass the VGGT key/value memory.
    """

    def __init__(
        self,
        *,
        planning_dim: int,
        source_dim: int = 2048,
        bottleneck_dim: int = 512,
        expected_horizons: int = 8,
        slots_per_horizon: int = 4,
        num_heads: int = 8,
        ffn_expansion: int = 2,
        detach_planning_queries: bool = True,
        attention_dropout: float = 0.0,
        return_attention_diagnostics: bool = False,
        view_count: int = 3,
    ) -> None:
        super().__init__()
        if planning_dim <= 0 or source_dim <= 0 or bottleneck_dim <= 0:
            raise ValueError("planning/source/bottleneck dimensions must be positive")
        if expected_horizons <= 0 or slots_per_horizon <= 0:
            raise ValueError("horizon and subslot counts must be positive")
        if bottleneck_dim % num_heads:
            raise ValueError("bottleneck_dim must be divisible by num_heads")
        if view_count != 3:
            raise ValueError("the dense VGGT source contract requires three views")

        self.planning_dim = int(planning_dim)
        self.source_dim = int(source_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.expected_horizons = int(expected_horizons)
        self.slots_per_horizon = int(slots_per_horizon)
        self.query_count = self.expected_horizons * self.slots_per_horizon
        self.detach_planning_queries = bool(detach_planning_queries)
        self.return_attention_diagnostics = bool(return_attention_diagnostics)
        self.view_count = int(view_count)

        self.source_norm = nn.LayerNorm(self.source_dim)
        self.source_projection = nn.Linear(
            self.source_dim, self.bottleneck_dim, bias=False
        )
        self.view_embedding = nn.Embedding(self.view_count, self.bottleneck_dim)
        self.uv_mlp = nn.Sequential(
            nn.Linear(2, 128),
            nn.SiLU(),
            nn.Linear(128, self.bottleneck_dim),
        )
        self.ray_mlp = nn.Sequential(
            nn.Linear(6, 256),
            nn.SiLU(),
            nn.Linear(256, self.bottleneck_dim),
        )

        self.planning_norm = nn.LayerNorm(self.planning_dim)
        self.planning_projection = nn.Linear(
            self.planning_dim, self.bottleneck_dim, bias=False
        )
        self.slot_embeddings = nn.Parameter(
            torch.empty(self.slots_per_horizon, self.bottleneck_dim)
        )
        nn.init.normal_(self.slot_embeddings, mean=0.0, std=0.02)
        self.query_norm = nn.LayerNorm(self.bottleneck_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.bottleneck_dim,
            num_heads=int(num_heads),
            dropout=float(attention_dropout),
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(self.bottleneck_dim)
        hidden_dim = self.bottleneck_dim * int(ffn_expansion)
        self.ffn = nn.Sequential(
            nn.Linear(self.bottleneck_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.bottleneck_dim),
        )
        self.readout_projection = nn.Linear(
            self.slots_per_horizon * self.bottleneck_dim,
            self.bottleneck_dim,
        )
        self.readout_norm = nn.LayerNorm(self.bottleneck_dim)
        self.up_projection = nn.Linear(self.bottleneck_dim, self.planning_dim)
        nn.init.zeros_(self.up_projection.weight)
        nn.init.zeros_(self.up_projection.bias)

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.to(dtype=values.dtype)
        return (values * weight).sum() / weight.sum().clamp_min(1)

    def _base_diagnostics(
        self,
        *,
        planning_tokens: torch.Tensor,
        source_features: torch.Tensor,
        source_valid_mask: torch.Tensor,
        task_tokens: torch.Tensor,
        readout: torch.Tensor,
        delta: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        source_norm = source_features.float().norm(dim=-1)
        planning_norm = planning_tokens.float().norm(dim=-1).mean().clamp_min(1e-8)
        slots = task_tokens.float().reshape(
            task_tokens.shape[0],
            self.expected_horizons,
            self.slots_per_horizon,
            self.bottleneck_dim,
        )
        unit_slots = F.normalize(slots, dim=-1, eps=1e-6)
        similarity = unit_slots @ unit_slots.transpose(-1, -2)
        off_diagonal = ~torch.eye(
            self.slots_per_horizon,
            dtype=torch.bool,
            device=similarity.device,
        )
        slot_pairwise = similarity[..., off_diagonal].mean()
        return {
            "source_token_count_mean": source_valid_mask.sum(dim=1).float().mean().detach(),
            "source_feature_norm": self._masked_mean(
                source_norm, source_valid_mask
            ).detach(),
            "task_geometry_norm": task_tokens.float().norm(dim=-1).mean().detach(),
            "horizon_readout_norm": readout.float().norm(dim=-1).mean().detach(),
            "planning_delta_norm": delta.float().norm(dim=-1).mean().detach(),
            "planning_delta_ratio": (
                delta.float().norm(dim=-1).mean() / planning_norm
            ).detach(),
            "slot_pairwise_cosine": slot_pairwise.detach(),
        }

    def _attention_diagnostics(
        self,
        attention: torch.Tensor,
        view_ids: torch.Tensor,
        source_valid_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # MultiheadAttention returns [B,heads,Q,N] when weights are not averaged.
        probability = attention.float().mean(dim=1).clamp_min(1e-8)
        probability = probability * source_valid_mask[:, None, :]
        probability = probability / probability.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        entropy = -(
            probability * probability.clamp_min(1e-8).log()
        ).sum(dim=-1)
        diagnostics: Dict[str, torch.Tensor] = {
            "attention_entropy": entropy.mean().detach(),
            "attention_max": probability.max(dim=-1).values.mean().detach(),
        }
        horizon_entropy = entropy.reshape(
            entropy.shape[0], self.expected_horizons, self.slots_per_horizon
        ).mean(dim=-1)
        for horizon in range(self.expected_horizons):
            diagnostics[f"attention_entropy_horizon_{horizon}"] = (
                horizon_entropy[:, horizon].mean().detach()
            )
        for view in range(self.view_count):
            view_mask = view_ids.eq(view) & source_valid_mask
            mass = (probability * view_mask[:, None, :]).sum(dim=-1).mean()
            diagnostics[f"attention_view_{view}_mass"] = mass.detach()
        diagnostics["attention_weights"] = attention.detach()
        return diagnostics

    def forward(
        self,
        planning_tokens: torch.Tensor,
        source_features: torch.Tensor,
        source_valid_mask: torch.Tensor,
        view_ids: torch.Tensor,
        uv_coords: torch.Tensor,
        ray_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Return enhanced planning states, eight readouts and 32 task tokens."""

        if planning_tokens.ndim != 3 or planning_tokens.shape[1:] != (
            self.expected_horizons,
            self.planning_dim,
        ):
            raise ValueError(
                "planning_tokens must be "
                f"[B,{self.expected_horizons},{self.planning_dim}]"
            )
        if source_features.ndim != 3 or source_features.shape[-1] != self.source_dim:
            raise ValueError(f"source_features must be [B,N,{self.source_dim}]")
        batch, source_count = source_features.shape[:2]
        expected_2d = (batch, source_count)
        if source_valid_mask.shape != expected_2d or source_valid_mask.dtype != torch.bool:
            raise ValueError("source_valid_mask must be BoolTensor[B,N]")
        if view_ids.shape != expected_2d:
            raise ValueError("view_ids must be [B,N]")
        if uv_coords.shape != (*expected_2d, 2):
            raise ValueError("uv_coords must be [B,N,2]")
        if ray_features.shape != (*expected_2d, 6):
            raise ValueError("ray_features must be [B,N,6]")
        if not source_valid_mask.any(dim=1).all():
            raise ValueError("every sample needs at least one valid dense VGGT token")
        valid_view_ids = view_ids[source_valid_mask]
        if valid_view_ids.numel() and (
            valid_view_ids.min() < 0 or valid_view_ids.max() >= self.view_count
        ):
            raise ValueError("valid dense VGGT view IDs must be in [0,2]")

        compute_dtype = self.source_norm.weight.dtype
        source_input = source_features.to(dtype=compute_dtype)
        source = self.source_projection(self.source_norm(source_input))
        source = source + self.view_embedding(view_ids.long()).to(dtype=compute_dtype)
        source = source + self.uv_mlp(uv_coords.to(dtype=compute_dtype))
        source = source + self.ray_mlp(ray_features.to(dtype=compute_dtype))

        planning_base_input = (
            planning_tokens.detach()
            if self.detach_planning_queries
            else planning_tokens
        )
        planning_base = self.planning_projection(
            self.planning_norm(planning_base_input.to(dtype=compute_dtype))
        )
        queries = planning_base[:, :, None, :] + self.slot_embeddings[None, None]
        queries = queries.reshape(batch, self.query_count, self.bottleneck_dim)

        need_weights = self.return_attention_diagnostics
        attention_output, attention = self.cross_attention(
            query=self.query_norm(queries),
            key=source,
            value=source,
            key_padding_mask=~source_valid_mask,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        # Deliberately no residual from ``queries`` here.
        task_geometry_tokens = attention_output
        task_geometry_tokens = task_geometry_tokens + self.ffn(
            self.ffn_norm(task_geometry_tokens)
        )
        grouped = task_geometry_tokens.reshape(
            batch,
            self.expected_horizons,
            self.slots_per_horizon * self.bottleneck_dim,
        )
        readout = self.readout_projection(grouped)
        delta = self.up_projection(self.readout_norm(readout))
        delta = delta.to(dtype=planning_tokens.dtype)
        enhanced = planning_tokens + delta
        diagnostics = self._base_diagnostics(
            planning_tokens=planning_tokens,
            source_features=source_features,
            source_valid_mask=source_valid_mask,
            task_tokens=task_geometry_tokens,
            readout=readout,
            delta=delta,
        )
        if need_weights:
            assert attention is not None
            diagnostics.update(
                self._attention_diagnostics(attention, view_ids, source_valid_mask)
            )
        return (
            enhanced,
            readout,
            task_geometry_tokens,
            diagnostics,
        )
