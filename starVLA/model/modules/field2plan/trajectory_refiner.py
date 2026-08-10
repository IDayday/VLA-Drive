"""Zero-initialized physical-space trajectory refiner."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .trajectory_codec import TrajectoryCodec
from .types import RefinerOutput


class TrajectoryRefiner(nn.Module):
    """Predict bounded physical deltas from draft and tube context.

    Input/output action shapes are ``[B,M,H,4]``. The last projection is zero
    initialized, and ``TrajectoryCodec.compose_delta`` preserves the draft
    bitwise when deltas are zero.
    """

    def __init__(
        self,
        context_dim: int,
        hidden_dim: int,
        num_layers: int,
        max_delta_xy_m: float = 4.0,
        max_delta_heading_rad: float = 0.5,
        action_query_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if min(context_dim, hidden_dim, num_layers) <= 0:
            raise ValueError("refiner dimensions/layers must be positive")
        self.context_dim = int(context_dim)
        self.max_delta_xy_m = float(max_delta_xy_m)
        self.max_delta_heading_rad = float(max_delta_heading_rad)
        self.action_query_projection = (
            nn.Linear(int(action_query_dim), hidden_dim)
            if action_query_dim is not None and action_query_dim > 0
            else None
        )
        input_dim = 4 + self.context_dim
        blocks = []
        for index in range(int(num_layers)):
            blocks.extend(
                [
                    nn.Linear(input_dim if index == 0 else hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                ]
            )
        self.backbone = nn.Sequential(*blocks)
        self.output_projection = nn.Linear(hidden_dim, 3)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self.codec = TrajectoryCodec()

    def forward(
        self,
        draft_action: torch.Tensor,
        waypoint_context: torch.Tensor,
        action_queries: Optional[torch.Tensor] = None,
    ) -> RefinerOutput:
        if draft_action.ndim == 3:
            draft_action = draft_action[:, None]
        if draft_action.ndim != 4 or draft_action.shape[-1] != 4:
            raise ValueError("draft_action must have shape [B,M,H,4]")
        if waypoint_context.shape[:-1] != draft_action.shape[:-1]:
            raise ValueError("waypoint_context leading shape must match draft")
        if waypoint_context.shape[-1] != self.context_dim:
            raise ValueError("waypoint_context channel mismatch")
        hidden = self.backbone(torch.cat((draft_action, waypoint_context), dim=-1))
        if action_queries is not None:
            if self.action_query_projection is None:
                raise ValueError("refiner was not configured with action_query_dim")
            if action_queries.ndim == 3:
                action_queries = action_queries[:, None].expand(
                    -1, draft_action.shape[1], -1, -1
                )
            hidden = hidden + self.action_query_projection(action_queries)
        raw_delta = self.output_projection(hidden)
        delta = torch.cat(
            (
                torch.tanh(raw_delta[..., :2]) * self.max_delta_xy_m,
                torch.tanh(raw_delta[..., 2:3]) * self.max_delta_heading_rad,
            ),
            dim=-1,
        )
        final = self.codec.compose_delta(draft_action, delta)
        if not isinstance(final, torch.Tensor):
            raise TypeError("refiner expects torch tensors")
        delta_norm = torch.linalg.vector_norm(delta, dim=-1).mean()
        return RefinerOutput(final, delta, delta_norm)
