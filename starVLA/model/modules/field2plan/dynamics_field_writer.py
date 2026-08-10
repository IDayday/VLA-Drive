"""Action-free future dynamics field predicted from present context."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .types import DynamicsFieldOutput


class ActionFreeDynamicsFieldWriter(nn.Module):
    """Predict ``[B,H,Cd,Ny,Nx]`` from current field and past ego motion.

    The interface intentionally accepts neither a draft nor any future action.
    ``history_current_from_ego`` contains only transforms at or before the
    current frame and has shape ``[B,Th,4,4]``.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        horizon: int,
        history_length: int,
        hidden_channels: int = 256,
    ) -> None:
        super().__init__()
        if min(
            int(input_channels),
            int(output_channels),
            int(horizon),
            int(history_length),
            int(hidden_channels),
        ) <= 0:
            raise ValueError("dynamics writer dimensions must be positive")
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.horizon = int(horizon)
        self.history_length = int(history_length)
        hidden_channels = int(hidden_channels)
        self.spatial_context_mode = "avg_pool3x3_linear_fusion"
        self.current_projection = nn.Sequential(
            nn.Linear(self.input_channels * 2, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, self.output_channels),
        )
        self.motion_encoder = nn.Sequential(
            nn.Linear(self.history_length * 5, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, self.output_channels),
        )
        self.temporal_embedding = nn.Parameter(
            torch.zeros(self.horizon, self.output_channels)
        )
        nn.init.normal_(self.temporal_embedding, mean=0.0, std=0.02)
        self.modulation = nn.Sequential(
            nn.Linear(self.output_channels * 2, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, self.output_channels * 2),
        )
        self.output_norm = nn.LayerNorm(self.output_channels)
        self.uncertainty_head = nn.Linear(self.output_channels, 1)

    def forward(
        self,
        current_field: torch.Tensor,
        history_current_from_ego: torch.Tensor,
        history_valid_mask: Optional[torch.Tensor] = None,
    ) -> DynamicsFieldOutput:
        """Predict future dynamics without future action conditioning."""

        if current_field.ndim != 4 or current_field.shape[1] != self.input_channels:
            raise ValueError("current_field must have shape [B,C,Ny,Nx]")
        batch, _, ny, nx = current_field.shape
        if history_current_from_ego.shape != (
            batch,
            self.history_length,
            4,
            4,
        ):
            raise ValueError(
                "history_current_from_ego must have shape [B,Th,4,4]"
            )
        transforms = history_current_from_ego.to(
            device=current_field.device, dtype=torch.float32
        )
        if not torch.isfinite(transforms).all():
            raise ValueError("history transforms contain non-finite values")
        rotation = transforms[..., :2, :2]
        yaw = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])
        motion = torch.stack(
            (
                transforms[..., 0, 3],
                transforms[..., 1, 3],
                torch.sin(yaw),
                torch.cos(yaw),
            ),
            dim=-1,
        )
        if history_valid_mask is None:
            mask = torch.ones(
                batch,
                self.history_length,
                device=current_field.device,
                dtype=torch.float32,
            )
        else:
            if history_valid_mask.shape != (batch, self.history_length):
                raise ValueError("history_valid_mask must have shape [B,Th]")
            mask = history_valid_mask.to(
                device=current_field.device, dtype=torch.float32
            )
        ordered_motion = torch.cat(
            (motion * mask[..., None], mask[..., None]), dim=-1
        ).reshape(batch, self.history_length * 5)
        motion_summary = self.motion_encoder(ordered_motion)

        local_context = F.avg_pool2d(
            current_field, kernel_size=3, stride=1, padding=1
        )
        base = self.current_projection(
            torch.cat((current_field, local_context), dim=1).permute(0, 2, 3, 1)
        )
        temporal = self.temporal_embedding.to(
            device=current_field.device, dtype=base.dtype
        )[None].expand(batch, -1, -1)
        conditioning = torch.cat(
            (motion_summary[:, None].expand(-1, self.horizon, -1), temporal),
            dim=-1,
        )
        scale, shift = self.modulation(conditioning).chunk(2, dim=-1)
        future = base[:, None] * (
            1.0 + 0.1 * torch.tanh(scale[:, :, None, None])
        ) + shift[:, :, None, None]
        future = self.output_norm(future)
        log_variance = self.uncertainty_head(future).clamp(-8.0, 8.0)
        field = future.permute(0, 1, 4, 2, 3).contiguous()
        log_variance = log_variance.permute(0, 1, 4, 2, 3).contiguous()
        return DynamicsFieldOutput(field, log_variance).validate()
