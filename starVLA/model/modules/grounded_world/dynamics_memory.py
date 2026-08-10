"""Separated current/history prior encoder and future memory predictor."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .types import CurrentDynamicsMemory, PredictiveWorldMemory


class CurrentDynamicsEncoder(nn.Module):
    """Encode current geometry and past ego motion into ``[B,Cd,Ny,Nx]``.

    This interface is deliberately action-free. External Driving-JEPA features
    supervise the returned *current/history* representation, not future slices.
    """

    def __init__(
        self,
        geometry_channels: int,
        output_channels: int,
        history_length: int,
        hidden_channels: int,
        x_range_m: Sequence[float] = (-8.0, 56.0),
        y_range_m: Sequence[float] = (-32.0, 32.0),
    ) -> None:
        super().__init__()
        if min(geometry_channels, output_channels, history_length, hidden_channels) <= 0:
            raise ValueError("current dynamics dimensions must be positive")
        self.geometry_channels = int(geometry_channels)
        self.output_channels = int(output_channels)
        self.history_length = int(history_length)
        self.x_range_m = (float(x_range_m[0]), float(x_range_m[1]))
        self.y_range_m = (float(y_range_m[0]), float(y_range_m[1]))
        if self.x_range_m[1] <= self.x_range_m[0] or self.y_range_m[1] <= self.y_range_m[0]:
            raise ValueError("dynamics field ranges must increase")
        self.spatial_projection = nn.Sequential(
            nn.Linear(self.geometry_channels * 2, int(hidden_channels)),
            nn.GELU(),
            nn.Linear(int(hidden_channels), self.output_channels),
        )
        self.motion_encoder = nn.Sequential(
            nn.Linear(self.history_length * 5, int(hidden_channels)),
            nn.GELU(),
            nn.Linear(int(hidden_channels), self.output_channels),
        )
        self.output_norm = nn.LayerNorm(self.output_channels)

    def forward(
        self,
        current_geometry: torch.Tensor,
        history_current_from_ego: torch.Tensor,
        history_valid_mask: Optional[torch.Tensor] = None,
        history_geometry: Optional[torch.Tensor] = None,
    ) -> CurrentDynamicsMemory:
        """Encode history transforms ``[B,Th,4,4]`` without future inputs."""

        if current_geometry.ndim != 4 or current_geometry.shape[1] != self.geometry_channels:
            raise ValueError("current_geometry must have shape [B,Cg,Ny,Nx]")
        batch = current_geometry.shape[0]
        if history_current_from_ego.shape != (batch, self.history_length, 4, 4):
            raise ValueError("history_current_from_ego must have shape [B,Th,4,4]")
        transforms = history_current_from_ego.to(
            device=current_geometry.device, dtype=torch.float32
        )
        if not torch.isfinite(transforms).all():
            raise ValueError("history transforms contain non-finite values")
        if history_valid_mask is None:
            mask = torch.ones(
                batch,
                self.history_length,
                device=current_geometry.device,
                dtype=torch.float32,
            )
        else:
            if history_valid_mask.shape != (batch, self.history_length):
                raise ValueError("history_valid_mask must have shape [B,Th]")
            mask = history_valid_mask.to(
                device=current_geometry.device, dtype=torch.float32
            )
        rotation = transforms[..., :2, :2]
        yaw = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])
        motion = torch.stack(
            (
                transforms[..., 0, 3],
                transforms[..., 1, 3],
                torch.sin(yaw),
                torch.cos(yaw),
                mask,
            ),
            dim=-1,
        )
        motion[..., :4] = motion[..., :4] * mask[..., None]
        motion_context = self.motion_encoder(motion.reshape(batch, -1))
        local = (
            self._warp_history_to_current(
                history_geometry,
                transforms,
                mask,
                current_geometry.shape[-2:],
            )
            if history_geometry is not None
            else F.avg_pool2d(current_geometry, 3, stride=1, padding=1)
        )
        spatial = self.spatial_projection(
            torch.cat((current_geometry, local), dim=1).permute(0, 2, 3, 1)
        )
        field = self.output_norm(spatial + motion_context[:, None, None])
        return CurrentDynamicsMemory(
            field.permute(0, 3, 1, 2).contiguous()
        ).validate()

    def _warp_history_to_current(
        self,
        history_geometry: torch.Tensor,
        current_from_ego: torch.Tensor,
        valid_mask: torch.Tensor,
        output_hw: tuple[int, int],
    ) -> torch.Tensor:
        """Ego-motion align ``[B,Th,C,Ny,Nx]`` into the current frame."""

        batch = current_from_ego.shape[0]
        ny, nx = output_hw
        if history_geometry.shape != (
            batch,
            self.history_length,
            self.geometry_channels,
            ny,
            nx,
        ):
            raise ValueError("history_geometry must have shape [B,Th,Cg,Ny,Nx]")
        source = history_geometry.to(
            device=current_from_ego.device, dtype=torch.float32
        )
        x_step = (self.x_range_m[1] - self.x_range_m[0]) / ny
        y_step = (self.y_range_m[1] - self.y_range_m[0]) / nx
        x = torch.linspace(
            self.x_range_m[0] + 0.5 * x_step,
            self.x_range_m[1] - 0.5 * x_step,
            ny,
            device=source.device,
            dtype=torch.float32,
        )
        y = torch.linspace(
            self.y_range_m[0] + 0.5 * y_step,
            self.y_range_m[1] - 0.5 * y_step,
            nx,
            device=source.device,
            dtype=torch.float32,
        )
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
        points = torch.stack(
            (grid_x, grid_y, torch.zeros_like(grid_x), torch.ones_like(grid_x)),
            dim=-1,
        )
        ego_from_current = torch.linalg.inv(current_from_ego)
        source_points = torch.einsum(
            "btij,hwj->bthwi", ego_from_current, points
        )
        source_x = source_points[..., 0]
        source_y = source_points[..., 1]
        normalized = torch.stack(
            (
                2.0 * (source_y - self.y_range_m[0])
                / (self.y_range_m[1] - self.y_range_m[0])
                - 1.0,
                2.0 * (source_x - self.x_range_m[0])
                / (self.x_range_m[1] - self.x_range_m[0])
                - 1.0,
            ),
            dim=-1,
        )
        warped = F.grid_sample(
            source.reshape(
                batch * self.history_length,
                self.geometry_channels,
                ny,
                nx,
            ),
            normalized.reshape(batch * self.history_length, ny, nx, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).reshape(batch, self.history_length, self.geometry_channels, ny, nx)
        spatial_valid = (
            (normalized[..., 0].abs() <= 1.0)
            & (normalized[..., 1].abs() <= 1.0)
            & valid_mask[..., None, None].bool()
        )
        weights = spatial_valid[:, :, None].to(dtype=warped.dtype)
        return (warped * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class PredictiveMemoryForecaster(nn.Module):
    """Predict action-free future memory from current geometry/dynamics."""

    def __init__(
        self,
        geometry_channels: int,
        dynamics_channels: int,
        horizon: int,
        hidden_channels: int,
    ) -> None:
        super().__init__()
        if min(geometry_channels, dynamics_channels, horizon, hidden_channels) <= 0:
            raise ValueError("predictive memory dimensions must be positive")
        self.geometry_channels = int(geometry_channels)
        self.dynamics_channels = int(dynamics_channels)
        self.horizon = int(horizon)
        self.input_projection = nn.Sequential(
            nn.Linear(
                self.geometry_channels + self.dynamics_channels,
                int(hidden_channels),
            ),
            nn.GELU(),
            nn.Linear(int(hidden_channels), self.dynamics_channels),
        )
        self.temporal_embedding = nn.Parameter(
            torch.empty(self.horizon, self.dynamics_channels)
        )
        nn.init.normal_(self.temporal_embedding, mean=0.0, std=0.02)
        self.modulation = nn.Sequential(
            nn.Linear(self.dynamics_channels, int(hidden_channels)),
            nn.GELU(),
            nn.Linear(int(hidden_channels), self.dynamics_channels * 2),
        )
        self.output_norm = nn.LayerNorm(self.dynamics_channels)
        self.uncertainty_head = nn.Linear(self.dynamics_channels, 1)

    def forward(
        self,
        current_geometry: torch.Tensor,
        current_dynamics: torch.Tensor,
    ) -> PredictiveWorldMemory:
        """Return future ``[B,H,Cd,Ny,Nx]`` without action conditioning."""

        if current_geometry.ndim != 4 or current_geometry.shape[1] != self.geometry_channels:
            raise ValueError("current_geometry must have shape [B,Cg,Ny,Nx]")
        if current_dynamics.ndim != 4 or current_dynamics.shape[1] != self.dynamics_channels:
            raise ValueError("current_dynamics must have shape [B,Cd,Ny,Nx]")
        if current_geometry.shape[0] != current_dynamics.shape[0] or current_geometry.shape[-2:] != current_dynamics.shape[-2:]:
            raise ValueError("current geometry and dynamics dimensions differ")
        base = self.input_projection(
            torch.cat((current_geometry, current_dynamics), dim=1).permute(0, 2, 3, 1)
        )
        temporal = self.temporal_embedding.to(device=base.device, dtype=base.dtype)
        scale, shift = self.modulation(temporal).chunk(2, dim=-1)
        future = base[:, None] * (
            1.0 + 0.1 * torch.tanh(scale)[None, :, None, None]
        ) + shift[None, :, None, None]
        future = self.output_norm(future)
        log_variance = self.uncertainty_head(future).clamp(-8.0, 8.0)
        future = future.permute(0, 1, 4, 2, 3).contiguous()
        log_variance = log_variance.permute(0, 1, 4, 2, 3).contiguous()
        return PredictiveWorldMemory(
            current=CurrentDynamicsMemory(current_dynamics),
            future=future,
            log_variance=log_variance,
        ).validate()
