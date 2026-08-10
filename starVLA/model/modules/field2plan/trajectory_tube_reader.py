"""Draft-conditioned sparse readout from ego fields."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .temporal_alignment import interpolate_temporal_features
from .trajectory_codec import TrajectoryCodec
from .types import TubeReadoutOutput


class TrajectoryTubeReader(nn.Module):
    """Read center/lateral/longitudinal points along physical drafts.

    Geometry field convention is ``[B,C,Ny(x-forward),Nx(y-left)]``.
    Drafts may be ``[B,H,3]`` or ``[B,M,H,3]``. The output always retains M.
    """

    def __init__(
        self,
        geometry_channels: int,
        output_dim: int,
        x_range_m: Sequence[float],
        y_range_m: Sequence[float],
        lateral_offsets_m: Sequence[float] = (-1.0, 0.0, 1.0),
        longitudinal_offsets_m: Sequence[float] = (0.0, 2.5),
        semantic_channels: Optional[int] = None,
        dynamics_channels: Optional[int] = None,
        temporal_interpolation: str = "linear",
        source_dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.geometry_channels = int(geometry_channels)
        self.output_dim = int(output_dim)
        self.x_range_m = (float(x_range_m[0]), float(x_range_m[1]))
        self.y_range_m = (float(y_range_m[0]), float(y_range_m[1]))
        self.lateral_offsets_m = tuple(float(x) for x in lateral_offsets_m)
        self.longitudinal_offsets_m = tuple(float(x) for x in longitudinal_offsets_m)
        if temporal_interpolation not in {"nearest", "linear"}:
            raise ValueError("temporal_interpolation must be nearest or linear")
        if not 0.0 <= float(source_dropout_p) < 1.0:
            raise ValueError("source_dropout_p must be within [0,1)")
        self.temporal_interpolation = str(temporal_interpolation)
        self.source_dropout_p = float(source_dropout_p)
        self.codec = TrajectoryCodec()
        self.geometry_projection = nn.Linear(self.geometry_channels, self.output_dim)
        self.dynamics_channels = (
            int(dynamics_channels)
            if dynamics_channels is not None and int(dynamics_channels) > 0
            else None
        )
        self.dynamics_projection = (
            nn.Linear(self.dynamics_channels, self.output_dim)
            if self.dynamics_channels is not None
            else None
        )
        self.semantic_projection = (
            nn.Linear(int(semantic_channels), self.output_dim)
            if semantic_channels is not None and semantic_channels > 0
            else None
        )
        source_count = 1
        source_count += int(self.dynamics_projection is not None)
        source_count += int(self.semantic_projection is not None)
        self.source_gate_logits = nn.Parameter(torch.zeros(source_count))

    def forward(
        self,
        geometry_field: torch.Tensor,
        draft_physical: torch.Tensor,
        semantic_tokens: Optional[torch.Tensor] = None,
        disable_access: bool = False,
        dynamics_field: Optional[torch.Tensor] = None,
        dynamics_times_s: Optional[torch.Tensor] = None,
        waypoint_times_s: Optional[torch.Tensor] = None,
        disable_geometry_access: bool = False,
        disable_dynamics_access: bool = False,
        disable_semantic_access: bool = False,
    ) -> TubeReadoutOutput:
        if geometry_field.ndim != 4 or geometry_field.shape[1] != self.geometry_channels:
            raise ValueError("geometry_field must have shape [B,C,Ny,Nx]")
        if draft_physical.ndim == 3:
            draft_physical = draft_physical[:, None]
        if draft_physical.ndim != 4 or draft_physical.shape[-1] != 3:
            raise ValueError("draft_physical must have shape [B,M,H,3]")
        batch, candidates, horizon, _ = draft_physical.shape
        if batch != geometry_field.shape[0]:
            raise ValueError("draft and field batch dimensions differ")

        tube = self.codec.tube_points(
            draft_physical,
            self.lateral_offsets_m,
            self.longitudinal_offsets_m,
        )
        if not isinstance(tube, torch.Tensor):
            raise TypeError("reader expects torch tensors")
        points = tube.to(device=geometry_field.device, dtype=torch.float32)
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
        structured_grid = torch.stack((grid_x, grid_y), dim=-1)
        grid = structured_grid.reshape(batch, -1, 1, 2)
        sampled = F.grid_sample(
            geometry_field.float(),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )[..., 0]
        points_per_waypoint = tube.shape[-2]
        sampled = sampled.transpose(1, 2).reshape(
            batch, candidates, horizon, points_per_waypoint, self.geometry_channels
        )
        valid_float = valid[..., None].to(sampled.dtype)
        geometry_context = (sampled * valid_float).sum(dim=-2) / valid_float.sum(
            dim=-2
        ).clamp_min(1.0)
        geometry_context = self.geometry_projection(geometry_context)

        contexts = [geometry_context]
        if self.dynamics_projection is not None:
            if dynamics_field is None:
                raise ValueError(
                    "dynamics_field is required when dynamics_channels are configured"
                )
            if (
                dynamics_field.ndim != 5
                or dynamics_field.shape[0] != batch
                or dynamics_field.shape[2] != self.dynamics_channels
            ):
                raise ValueError(
                    "dynamics_field must have shape [B,T,Cd,Ny,Nx]"
                )
            if dynamics_times_s is None or waypoint_times_s is None:
                raise ValueError(
                    "dynamics_times_s and waypoint_times_s are required for dynamics readout"
                )
            query_times = torch.as_tensor(waypoint_times_s)
            if query_times.ndim != 1 or query_times.shape[0] != horizon:
                raise ValueError("waypoint_times_s must have shape [H]")
            temporal_field = interpolate_temporal_features(
                dynamics_field,
                torch.as_tensor(dynamics_times_s),
                query_times,
                mode=self.temporal_interpolation,
            )
            dynamic_grid = structured_grid.permute(0, 2, 1, 3, 4).reshape(
                batch * horizon, candidates * points_per_waypoint, 1, 2
            )
            dynamic_sampled = F.grid_sample(
                temporal_field.reshape(
                    batch * horizon,
                    self.dynamics_channels,
                    *temporal_field.shape[-2:],
                ).float(),
                dynamic_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )[..., 0]
            dynamic_sampled = dynamic_sampled.reshape(
                batch,
                horizon,
                self.dynamics_channels,
                candidates,
                points_per_waypoint,
            ).permute(0, 3, 1, 4, 2)
            dynamic_valid = valid[..., None].to(dynamic_sampled.dtype)
            dynamic_context = (
                dynamic_sampled * dynamic_valid
            ).sum(dim=-2) / dynamic_valid.sum(dim=-2).clamp_min(1.0)
            contexts.append(self.dynamics_projection(dynamic_context))
        if self.semantic_projection is not None:
            if semantic_tokens is None:
                semantic_context = torch.zeros_like(geometry_context)
            else:
                if semantic_tokens.ndim != 3:
                    raise ValueError("semantic_tokens must have shape [B,K,C]")
                semantic_context = self.semantic_projection(semantic_tokens.mean(dim=1))
                semantic_context = semantic_context[:, None, None].expand(
                    -1, candidates, horizon, -1
                )
            contexts.append(semantic_context)
        source_access = [not disable_geometry_access]
        if self.dynamics_projection is not None:
            source_access.append(not disable_dynamics_access)
        if self.semantic_projection is not None:
            source_access.append(not disable_semantic_access)
        has_access = (not disable_access) and any(source_access)
        access_mask = torch.tensor(
            source_access, device=geometry_field.device, dtype=torch.bool
        )
        if disable_access:
            access_mask = torch.zeros_like(access_mask)
        gate_logits = self.source_gate_logits.reshape(1, 1, 1, -1).expand(
            batch, candidates, horizon, -1
        )
        gate_logits = gate_logits.masked_fill(
            ~access_mask.reshape(1, 1, 1, -1), -1e4
        )
        if self.training and self.source_dropout_p > 0.0 and gate_logits.shape[-1] > 1:
            optional_keep = torch.rand(
                batch,
                candidates,
                horizon,
                gate_logits.shape[-1] - 1,
                device=gate_logits.device,
            ) >= self.source_dropout_p
            keep = torch.cat(
                (
                    torch.ones_like(optional_keep[..., :1], dtype=torch.bool),
                    optional_keep,
                ),
                dim=-1,
            )
            gate_logits = gate_logits.masked_fill(~keep, -1e4)
        if has_access:
            gates = torch.softmax(gate_logits, dim=-1)
            gates = gates * access_mask.reshape(1, 1, 1, -1).to(gates.dtype)
            gates = gates / gates.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        else:
            gates = torch.zeros_like(gate_logits)
        context_stack = torch.stack(contexts, dim=-2)
        waypoint_context = (context_stack * gates[..., None]).sum(dim=-2)
        if not has_access:
            waypoint_context = torch.zeros_like(waypoint_context)
        return TubeReadoutOutput(waypoint_context, valid, gates, tube)
