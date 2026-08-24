"""Bounded scene-conditioned retention gate for geometry memory."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class SceneConditionedGeometryGate(nn.Module):
    def __init__(
        self,
        scene_dim: int = 2048,
        geometry_dim: int = 512,
        minimum_retention: float = 0.05,
        maximum_retention: float = 0.50,
        initial_retention: float = 0.10,
    ) -> None:
        super().__init__()
        if not minimum_retention < initial_retention < maximum_retention:
            raise ValueError("initial retention must be strictly inside the bounds")
        self.minimum_retention = float(minimum_retention)
        self.maximum_retention = float(maximum_retention)
        self.initial_retention = float(initial_retention)
        self.scene_norm = nn.LayerNorm(scene_dim, elementwise_affine=False)
        self.scene_projection = nn.Linear(scene_dim, geometry_dim)
        self.gate_projection = nn.Linear(geometry_dim * 2, geometry_dim)
        nn.init.zeros_(self.gate_projection.weight)
        probability = (initial_retention - minimum_retention) / (
            maximum_retention - minimum_retention
        )
        nn.init.constant_(
            self.gate_projection.bias, math.log(probability / (1.0 - probability))
        )

    def forward(
        self, scene_summary: Tensor, geometry_memory: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if scene_summary.ndim != 3 or scene_summary.shape[1] != 1:
            raise ValueError("scene_summary must be [B,1,Ds]")
        if geometry_memory.ndim != 3 or geometry_memory.shape[0] != scene_summary.shape[0]:
            raise ValueError("geometry_memory must be [B,N,Dg]")
        scene = self.scene_projection(self.scene_norm(scene_summary))
        scene = scene.expand(-1, geometry_memory.shape[1], -1)
        logits = self.gate_projection(torch.cat((scene, geometry_memory), dim=-1))
        retention = self.minimum_retention + (
            self.maximum_retention - self.minimum_retention
        ) * torch.sigmoid(logits)
        gated = retention * geometry_memory
        span = self.maximum_retention - self.minimum_retention
        tolerance = span * 0.01
        detached = retention.detach()
        diagnostics = {
            "gp_sq3dmix/retention_mean": detached.mean(),
            "gp_sq3dmix/retention_std": detached.std(unbiased=False),
            "gp_sq3dmix/retention_min": detached.min(),
            "gp_sq3dmix/retention_max": detached.max(),
            "gp_sq3dmix/retention_near_lower_fraction": (
                detached <= self.minimum_retention + tolerance
            ).float().mean(),
            "gp_sq3dmix/retention_near_upper_fraction": (
                detached >= self.maximum_retention - tolerance
            ).float().mean(),
            "gp_sq3dmix/gated_geometry_norm": gated.detach().float().norm(dim=-1).mean(),
            "gp_sq3dmix/ungated_geometry_norm": geometry_memory.detach().float().norm(dim=-1).mean(),
        }
        return gated, diagnostics
