"""Action-conditioned access to distilled VGGT query representations."""

from __future__ import annotations

import math
from typing import Dict

import torch
from torch import nn


class PlanningQueryBridge(nn.Module):
    """Let action queries retrieve geometry before diffusion planning.

    Inputs are action queries ``[B,A,H]``, VGGT queries ``[B,Q,H]`` and a
    boolean target validity mask ``[B,Q]``. The returned query tensor remains
    ``[B,A,H]`` and is used together with the unpooled VGGT context by the
    action head.
    """

    def __init__(self, hidden_dim: int, num_heads: int, initial_gate: float = 0.5) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0
        assert 0.0 < initial_gate < 1.0
        self.action_norm = nn.LayerNorm(hidden_dim)
        self.geometry_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.gate_logit = nn.Parameter(
            torch.tensor(math.log(initial_gate / (1.0 - initial_gate)), dtype=torch.float32)
        )
        # A small residual starts stably while retaining a non-zero planning path.
        nn.init.normal_(self.cross_attention.out_proj.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.cross_attention.out_proj.bias)

    def forward(
        self,
        action_queries: torch.Tensor,
        geometry_queries: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert action_queries.ndim == 3, "action_queries must be [B,A,H]"
        assert geometry_queries.ndim == 3, "geometry_queries must be [B,Q,H]"
        assert action_queries.shape[0] == geometry_queries.shape[0], "batch mismatch"
        assert action_queries.shape[2] == geometry_queries.shape[2], "hidden dim mismatch"
        assert valid_mask.shape == geometry_queries.shape[:2]
        assert valid_mask.dtype == torch.bool
        assert valid_mask.any(dim=1).all(), "every sample needs at least one valid VGGT query"

        query = self.action_norm(action_queries)
        context = self.geometry_norm(geometry_queries).to(dtype=query.dtype)
        delta, attention = self.cross_attention(
            query,
            context,
            context,
            key_padding_mask=~valid_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        gate = torch.sigmoid(self.gate_logit).to(dtype=delta.dtype)
        enhanced = action_queries + gate * delta
        probability = attention.float().clamp_min(1e-8)
        entropy = -(probability * probability.log()).sum(dim=-1)
        diagnostics = {
            "planner_bridge_attention": attention.detach(),
            "planner_bridge_gate": gate.detach(),
            "planner_bridge_delta_norm": delta.float().norm(dim=-1).mean().detach(),
            "planner_bridge_attention_entropy": entropy.mean().detach(),
            "planner_bridge_attention_max": probability.max(dim=-1).values.mean().detach(),
        }
        return enhanced, diagnostics
