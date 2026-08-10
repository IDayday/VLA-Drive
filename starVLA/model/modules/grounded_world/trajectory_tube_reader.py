"""Multi-scale swept-tube readout for GroundedWorld planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from starVLA.model.modules.field2plan.trajectory_codec import TrajectoryCodec

from .types import MultiScaleGeometryMemory


@dataclass(frozen=True)
class GroundedTubeReadout:
    """Per-waypoint context ``[B,M,H,C]`` and tube diagnostics."""

    waypoint_context: torch.Tensor
    tube_valid_mask: torch.Tensor
    source_gates: torch.Tensor
    tube_points: torch.Tensor


class MultiScaleTrajectoryTubeReader(nn.Module):
    """Read every geometry scale plus one current/future dynamics source.

    Physical trajectories are ``[B,M,H,3]`` with ``(x,y,heading)``. Geometry
    levels are ego-aligned ``[B,C_i,Ny_i,Nx_i]``. Dynamics may be current
    ``[B,Cd,Ny,Nx]`` or predicted future ``[B,H,Cd,Ny,Nx]``.
    """

    def __init__(
        self,
        geometry_channels: Sequence[int],
        dynamics_channels: int,
        output_dim: int,
        x_range_m: Sequence[float],
        y_range_m: Sequence[float],
        lateral_offsets_m: Sequence[float] = (-1.0, 0.0, 1.0),
        longitudinal_offsets_m: Sequence[float] = (0.0, 2.5),
    ) -> None:
        super().__init__()
        channels = tuple(int(value) for value in geometry_channels)
        if not channels or min(channels) <= 0:
            raise ValueError("geometry_channels must be positive")
        if min(int(dynamics_channels), int(output_dim)) <= 0:
            raise ValueError("reader dimensions must be positive")
        self.geometry_channels = channels
        self.dynamics_channels = int(dynamics_channels)
        self.output_dim = int(output_dim)
        self.x_range_m = (float(x_range_m[0]), float(x_range_m[1]))
        self.y_range_m = (float(y_range_m[0]), float(y_range_m[1]))
        if self.x_range_m[1] <= self.x_range_m[0] or self.y_range_m[1] <= self.y_range_m[0]:
            raise ValueError("field ranges must increase")
        self.lateral_offsets_m = tuple(float(value) for value in lateral_offsets_m)
        self.longitudinal_offsets_m = tuple(
            float(value) for value in longitudinal_offsets_m
        )
        self.codec = TrajectoryCodec()
        self.geometry_projections = nn.ModuleList(
            [nn.Linear(channel, self.output_dim) for channel in channels]
        )
        self.future_projection = nn.Linear(self.dynamics_channels, self.output_dim)
        self.source_gate_logits = nn.Parameter(torch.zeros(len(channels) + 1))

    def _grid_and_valid(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = points[..., 0], points[..., 1]
        valid = (
            (x >= self.x_range_m[0])
            & (x <= self.x_range_m[1])
            & (y >= self.y_range_m[0])
            & (y <= self.y_range_m[1])
        )
        grid_x = 2.0 * (y - self.y_range_m[0]) / (
            self.y_range_m[1] - self.y_range_m[0]
        ) - 1.0
        grid_y = 2.0 * (x - self.x_range_m[0]) / (
            self.x_range_m[1] - self.x_range_m[0]
        ) - 1.0
        return torch.stack((grid_x, grid_y), dim=-1), valid

    @staticmethod
    def _aggregate_static(
        field: torch.Tensor,
        grid: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        batch, candidates, horizon, points_per_waypoint, _ = grid.shape
        sampled = F.grid_sample(
            field.float(),
            grid.reshape(batch, -1, 1, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )[..., 0]
        sampled = sampled.transpose(1, 2).reshape(
            batch,
            candidates,
            horizon,
            points_per_waypoint,
            field.shape[1],
        )
        weights = valid[..., None].to(dtype=sampled.dtype)
        return (sampled * weights).sum(dim=-2) / weights.sum(dim=-2).clamp_min(1.0)

    @staticmethod
    def _aggregate_temporal(
        field: torch.Tensor,
        grid: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        batch, candidates, horizon, points_per_waypoint, _ = grid.shape
        if field.ndim != 5 or field.shape[:2] != (batch, horizon):
            raise ValueError("future_dynamics must have shape [B,H,Cd,Ny,Nx]")
        temporal_grid = grid.permute(0, 2, 1, 3, 4).reshape(
            batch * horizon, candidates * points_per_waypoint, 1, 2
        )
        sampled = F.grid_sample(
            field.reshape(batch * horizon, field.shape[2], *field.shape[-2:]).float(),
            temporal_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )[..., 0]
        sampled = sampled.reshape(
            batch,
            horizon,
            field.shape[2],
            candidates,
            points_per_waypoint,
        ).permute(0, 3, 1, 4, 2)
        weights = valid[..., None].to(dtype=sampled.dtype)
        return (sampled * weights).sum(dim=-2) / weights.sum(dim=-2).clamp_min(1.0)

    def forward(
        self,
        geometry: MultiScaleGeometryMemory,
        physical_trajectory: torch.Tensor,
        current_dynamics: Optional[torch.Tensor] = None,
        future_dynamics: Optional[torch.Tensor] = None,
        disable_access: bool = False,
    ) -> GroundedTubeReadout:
        """Read a swept tube without modifying or selecting the trajectory."""

        geometry.validate()
        if tuple(level.shape[1] for level in geometry.levels) != self.geometry_channels:
            raise ValueError("geometry memory channels differ from reader config")
        if physical_trajectory.ndim == 3:
            physical_trajectory = physical_trajectory[:, None]
        if physical_trajectory.ndim != 4 or physical_trajectory.shape[-1] != 3:
            raise ValueError("physical_trajectory must have shape [B,M,H,3]")
        batch, candidates, horizon, _ = physical_trajectory.shape
        if geometry.finest.shape[0] != batch:
            raise ValueError("trajectory and geometry batch dimensions differ")
        if current_dynamics is not None and future_dynamics is not None:
            raise ValueError("provide current_dynamics or future_dynamics, not both")
        dynamics = future_dynamics if future_dynamics is not None else current_dynamics
        if dynamics is None:
            raise ValueError("one current/future dynamics source is required")
        if dynamics.shape[0] != batch or dynamics.shape[-3] != self.dynamics_channels:
            raise ValueError("dynamics channel or batch dimension differs")

        tube = self.codec.tube_points(
            physical_trajectory,
            self.lateral_offsets_m,
            self.longitudinal_offsets_m,
        )
        if not isinstance(tube, torch.Tensor):
            raise TypeError("reader requires torch trajectory tensors")
        points = tube.to(device=geometry.finest.device, dtype=torch.float32)
        grid, valid = self._grid_and_valid(points)
        contexts = []
        for level, projection in zip(geometry.levels, self.geometry_projections):
            contexts.append(projection(self._aggregate_static(level, grid, valid)))
        if dynamics.ndim == 4:
            dynamic_context = self._aggregate_static(dynamics, grid, valid)
        else:
            dynamic_context = self._aggregate_temporal(dynamics, grid, valid)
        contexts.append(self.future_projection(dynamic_context))

        gate_logits = self.source_gate_logits.to(
            device=geometry.finest.device, dtype=contexts[0].dtype
        )
        gates = torch.softmax(gate_logits, dim=0).reshape(1, 1, 1, -1).expand(
            batch, candidates, horizon, -1
        )
        context = (torch.stack(contexts, dim=-2) * gates[..., None]).sum(dim=-2)
        if disable_access:
            gates = torch.zeros_like(gates)
            context = torch.zeros_like(context)
        return GroundedTubeReadout(context, valid, gates, tube)
