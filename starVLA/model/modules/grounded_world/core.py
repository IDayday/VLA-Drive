"""Action-free physical-memory core shared by all GroundedWorld stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import nn

from .dynamics_memory import CurrentDynamicsEncoder, PredictiveMemoryForecaster
from .geometry_memory import MultiScaleGeometryMemoryWriter
from .types import (
    CurrentDynamicsMemory,
    MultiScaleGeometryMemory,
    PredictiveWorldMemory,
)


@dataclass(frozen=True)
class GroundedWorldMemoryOutput:
    """Physical memories produced without an action or future trajectory input."""

    geometry: MultiScaleGeometryMemory
    current_dynamics: CurrentDynamicsMemory
    predictive: Optional[PredictiveWorldMemory]


class GroundedWorldCore(nn.Module):
    """Build current/history and optional future physical memory.

    ``finest_geometry`` is a camera-grounded ego field ``[B,Cg,Ny,Nx]``;
    historical transforms are ``[B,Th,4,4]``. The signature intentionally
    contains no action, draft, trajectory, or demonstrated future action.
    """

    def __init__(
        self,
        geometry_input_channels: int,
        geometry_channels: Sequence[int],
        scale_factors: Sequence[int],
        dynamics_channels: int,
        history_length: int,
        horizon: int,
        future_enabled: bool,
        hidden_channels: int = 256,
        x_range_m: Sequence[float] = (-8.0, 56.0),
        y_range_m: Sequence[float] = (-32.0, 32.0),
    ) -> None:
        super().__init__()
        channels = tuple(int(value) for value in geometry_channels)
        if not channels:
            raise ValueError("geometry_channels must not be empty")
        self.future_enabled = bool(future_enabled)
        self.geometry_memory = MultiScaleGeometryMemoryWriter(
            int(geometry_input_channels), channels, scale_factors
        )
        self.current_dynamics_encoder = CurrentDynamicsEncoder(
            geometry_channels=channels[0],
            output_channels=int(dynamics_channels),
            history_length=int(history_length),
            hidden_channels=int(hidden_channels),
            x_range_m=x_range_m,
            y_range_m=y_range_m,
        )
        self.predictive_memory = (
            PredictiveMemoryForecaster(
                geometry_channels=channels[0],
                dynamics_channels=int(dynamics_channels),
                horizon=int(horizon),
                hidden_channels=int(hidden_channels),
            )
            if self.future_enabled
            else None
        )

    def forward(
        self,
        finest_geometry: torch.Tensor,
        history_current_from_ego: torch.Tensor,
        history_valid_mask: Optional[torch.Tensor] = None,
        history_geometry: Optional[torch.Tensor] = None,
        stage: str = "prior",
    ) -> GroundedWorldMemoryOutput:
        """Return current memory, and future memory only in predictive mode."""

        if stage not in {"prior", "predictive"}:
            raise ValueError("GroundedWorldCore stage must be prior or predictive")
        if stage == "predictive" and self.predictive_memory is None:
            raise ValueError("predictive stage requires future_enabled=true")
        geometry = self.geometry_memory(finest_geometry)
        projected_history = None
        if history_geometry is not None:
            if (
                history_geometry.ndim != 5
                or history_geometry.shape[0] != finest_geometry.shape[0]
                or history_geometry.shape[2] != self.geometry_memory.input_channels
                or history_geometry.shape[-2:] != finest_geometry.shape[-2:]
            ):
                raise ValueError("history_geometry must have shape [B,Th,Cg,Ny,Nx]")
            projected_history = self.geometry_memory.projections[0](
                history_geometry.permute(0, 1, 3, 4, 2)
            ).permute(0, 1, 4, 2, 3).contiguous()
        current = self.current_dynamics_encoder(
            geometry.finest,
            history_current_from_ego,
            history_valid_mask=history_valid_mask,
            history_geometry=projected_history,
        )
        predictive = (
            self.predictive_memory(geometry.finest, current.field)
            if stage == "predictive"
            else None
        )
        return GroundedWorldMemoryOutput(geometry, current, predictive)
