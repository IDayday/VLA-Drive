"""Interpretable geometry-to-planning heads for the VGGT V2 planner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from starVLA.model.modules.vggt_query.types import VGGTQueryLayout


@dataclass
class SupervisedHeadOutput:
    """A scalar auxiliary loss, prediction and detached metrics."""

    loss: torch.Tensor
    prediction: torch.Tensor
    metrics: Dict[str, torch.Tensor]


class WaypointGeometryReader(nn.Module):
    """Read 195 geometry slots into exactly one token per action waypoint."""

    def __init__(
        self,
        *,
        action_dim: int,
        memory_dim: int,
        num_heads: int,
        layout: VGGTQueryLayout,
    ) -> None:
        super().__init__()
        assert action_dim % num_heads == 0
        self.action_dim = int(action_dim)
        self.memory_dim = int(memory_dim)
        self.layout = layout
        self.action_norm = nn.LayerNorm(self.action_dim)
        self.memory_norm = nn.LayerNorm(self.memory_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.action_dim,
            num_heads=int(num_heads),
            kdim=self.memory_dim,
            vdim=self.memory_dim,
            batch_first=True,
        )
        nn.init.normal_(self.cross_attention.out_proj.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.cross_attention.out_proj.bias)

    @staticmethod
    def _js_divergence(probability: torch.Tensor) -> torch.Tensor:
        if probability.shape[1] < 2:
            return probability.new_zeros(())
        first = probability[:, :-1].clamp_min(1e-8)
        second = probability[:, 1:].clamp_min(1e-8)
        mixture = 0.5 * (first + second)
        divergence = 0.5 * (
            (first * (first.log() - mixture.log())).sum(-1)
            + (second * (second.log() - mixture.log())).sum(-1)
        )
        return divergence.mean()

    def forward(
        self,
        action_queries: torch.Tensor,
        geometry_memory: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Return readout ``[B,A,Ha]`` and scalar/attention diagnostics."""

        assert action_queries.ndim == geometry_memory.ndim == 3
        assert action_queries.shape[0] == geometry_memory.shape[0]
        assert action_queries.shape[-1] == self.action_dim
        assert geometry_memory.shape[1:] == (
            self.layout.query_count,
            self.memory_dim,
        )
        assert valid_mask.shape == geometry_memory.shape[:2]
        assert valid_mask.dtype == torch.bool
        assert valid_mask.any(dim=1).all(), "every sample needs valid geometry memory"
        query = self.action_norm(
            action_queries.to(dtype=self.action_norm.weight.dtype)
        )
        memory = self.memory_norm(
            geometry_memory.to(dtype=self.memory_norm.weight.dtype)
        )
        readout, attention = self.cross_attention(
            query,
            memory,
            memory,
            key_padding_mask=~valid_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        probability = attention.float().clamp_min(1e-8)
        entropy = -(probability * probability.log()).sum(-1)
        special_count = self.layout.special_query_count
        per_view_spatial = self.layout.spatial_rows * self.layout.spatial_cols
        group_mass = {}
        for view_index, name in enumerate(("front", "left", "right")):
            special_start = view_index * self.layout.special_per_view
            special_end = special_start + self.layout.special_per_view
            spatial_start = special_count + view_index * per_view_spatial
            spatial_end = spatial_start + per_view_spatial
            group_mass[name] = (
                probability[..., special_start:special_end].sum(-1)
                + probability[..., spatial_start:spatial_end].sum(-1)
            ).mean()
        diagnostics = {
            "planner_attention": attention.detach(),
            "attention_entropy": entropy.mean().detach(),
            "attention_max": probability.max(dim=-1).values.mean().detach(),
            "attention_special_mass": probability[..., :special_count].sum(-1).mean().detach(),
            "attention_spatial_mass": probability[..., special_count:].sum(-1).mean().detach(),
            "attention_front_view_mass": group_mass["front"].detach(),
            "attention_left_view_mass": group_mass["left"].detach(),
            "attention_right_view_mass": group_mass["right"].detach(),
            "attention_waypoint_js_divergence": self._js_divergence(probability).detach(),
            "geometry_readout_norm": readout.float().norm(dim=-1).mean().detach(),
        }
        diagnostics.update(
            {
                f"attention_entropy_waypoint_{index}": entropy[:, index].mean().detach()
                for index in range(entropy.shape[1])
            }
        )
        return readout.to(dtype=action_queries.dtype), diagnostics


class PhysicalGeometryHead(nn.Module):
    """Predict ``x/z, y/z, log(z/median_z)`` for 180 spatial slots."""

    def __init__(self, memory_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.memory_dim = int(memory_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(self.memory_dim),
            nn.Linear(self.memory_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 3),
        )

    def forward(
        self,
        spatial_memory: torch.Tensor,
        target: torch.Tensor,
        confidence: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> SupervisedHeadOutput:
        """Return confidence-weighted Huber geometry supervision."""

        assert spatial_memory.ndim == 3 and spatial_memory.shape[-1] == self.memory_dim
        assert target.shape == (*spatial_memory.shape[:2], 3)
        assert confidence.shape == valid_mask.shape == spatial_memory.shape[:2]
        assert valid_mask.dtype == torch.bool
        prediction = self.head(
            spatial_memory.to(dtype=self.head[0].weight.dtype)
        )
        finite = torch.isfinite(target).all(-1) & torch.isfinite(confidence)
        mask = valid_mask & finite & confidence.gt(0)
        assert mask.any(), "physical geometry batch has no valid target"
        weight = confidence.float().clamp_min(0) * mask.float()
        error = F.smooth_l1_loss(prediction, target.float(), reduction="none").mean(-1)
        loss = (error * weight).sum() / weight.sum().clamp_min(1e-6)
        absolute = (prediction.detach() - target.float()).abs()
        denominator = mask.sum().clamp_min(1)
        metrics = {
            "geometry_valid_ratio": mask.float().mean().detach(),
            "geometry_x_over_z_mae": (absolute[..., 0] * mask).sum() / denominator,
            "geometry_y_over_z_mae": (absolute[..., 1] * mask).sum() / denominator,
            "geometry_log_depth_mae": (absolute[..., 2] * mask).sum() / denominator,
        }
        return SupervisedHeadOutput(loss=loss, prediction=prediction, metrics=metrics)


class AuxiliaryTrajectoryHead(nn.Module):
    """Training-only deep supervision from 8 waypoint readouts to actions."""

    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int = 4) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.action_dim = int(action_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.action_dim),
        )

    def forward(
        self,
        waypoint_readout: torch.Tensor,
        target_action: torch.Tensor,
    ) -> SupervisedHeadOutput:
        """Return Huber loss against normalized ``[B,8,4]`` actions."""

        assert waypoint_readout.ndim == 3 and waypoint_readout.shape[-1] == self.input_dim
        assert target_action.shape == (*waypoint_readout.shape[:2], self.action_dim)
        prediction = self.head(
            waypoint_readout.to(dtype=self.head[0].weight.dtype)
        )
        prediction_float = prediction.float()
        target_float = target_action.float()
        loss = F.smooth_l1_loss(prediction_float, target_float)
        xy_error = (prediction_float.detach()[..., :2] - target_float[..., :2]).norm(dim=-1)
        heading_cosine = F.cosine_similarity(
            prediction_float.detach()[..., 2:4], target_float[..., 2:4], dim=-1
        )
        metrics = {
            "aux_plan_ade": xy_error.mean().detach(),
            "aux_plan_fde": xy_error[:, -1].mean().detach(),
            "aux_plan_heading_cosine": heading_cosine.mean().detach(),
        }
        return SupervisedHeadOutput(loss=loss, prediction=prediction, metrics=metrics)
