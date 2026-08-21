"""Scene-conditioned position-wise and channel-wise VGGT fusion."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn


class SceneConditionedGatedFusion(nn.Module):
    """Fuse scene semantics and projected geometry with exactly four linears."""

    def __init__(self, scene_dim: int, vggt_dim: int) -> None:
        super().__init__()
        if scene_dim <= 0 or vggt_dim <= 0:
            raise ValueError("scene_dim and vggt_dim must be positive")
        self.scene_dim = int(scene_dim)
        self.vggt_dim = int(vggt_dim)
        self.vggt_projection = nn.Linear(self.vggt_dim, self.scene_dim)
        self.gate_projection = nn.Linear(2 * self.scene_dim, self.scene_dim)
        self.semantic_projection = nn.Linear(self.scene_dim, self.scene_dim)
        self.geometry_projection = nn.Linear(self.scene_dim, self.scene_dim)

    def _validate_vggt_tokens(self, vggt_tokens: Tensor) -> None:
        if vggt_tokens.ndim != 3:
            raise ValueError("vggt_tokens must be [B,N,Dv]")
        if vggt_tokens.shape[-1] != self.vggt_dim:
            raise ValueError(
                f"VGGT dimension mismatch: expected {self.vggt_dim}, "
                f"found {vggt_tokens.shape[-1]}"
            )

    def project_geometry(self, vggt_tokens: Tensor) -> Tensor:
        self._validate_vggt_tokens(vggt_tokens)
        return self.vggt_projection(vggt_tokens)

    def forward(
        self,
        scene_tokens: Tensor,
        vggt_tokens: Tensor,
    ) -> tuple[Tensor, Dict[str, Tensor]]:
        if scene_tokens.ndim != 3:
            raise ValueError("scene_tokens must be [B,Q,D]")
        if scene_tokens.shape[-1] != self.scene_dim:
            raise ValueError(
                f"scene dimension mismatch: expected {self.scene_dim}, "
                f"found {scene_tokens.shape[-1]}"
            )
        self._validate_vggt_tokens(vggt_tokens)
        if scene_tokens.shape[0] != vggt_tokens.shape[0]:
            raise ValueError("scene and VGGT batch dimensions must match")

        scene_summary = scene_tokens.mean(dim=1, keepdim=True)
        geometry = self.vggt_projection(vggt_tokens)
        semantic = scene_summary.expand(-1, geometry.shape[1], -1)
        gate = torch.sigmoid(
            self.gate_projection(torch.cat([semantic, geometry], dim=-1))
        )
        semantic_branch = self.semantic_projection(semantic)
        geometry_branch = self.geometry_projection(geometry)
        fused = gate * semantic_branch + (1.0 - gate) * geometry_branch

        detached_gate = gate.detach().float()
        diagnostics = {
            "sq3dmix/gate_mean": detached_gate.mean(),
            "sq3dmix/gate_std": detached_gate.std(unbiased=False),
            "sq3dmix/gate_below_005_ratio": (detached_gate < 0.05).float().mean(),
            "sq3dmix/gate_above_095_ratio": (detached_gate > 0.95).float().mean(),
            "sq3dmix/scene_summary_norm": scene_summary.detach()
            .float()
            .norm(dim=-1)
            .mean(),
            "sq3dmix/projected_geometry_norm": geometry.detach()
            .float()
            .norm(dim=-1)
            .mean(),
            "sq3dmix/fused_geometry_norm": fused.detach()
            .float()
            .norm(dim=-1)
            .mean(),
            "sq3dmix/semantic_branch_norm": semantic_branch.detach()
            .float()
            .norm(dim=-1)
            .mean(),
            "sq3dmix/geometry_branch_norm": geometry_branch.detach()
            .float()
            .norm(dim=-1)
            .mean(),
        }
        return fused, diagnostics
