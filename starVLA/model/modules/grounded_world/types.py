"""Typed tensor contracts for GroundedWorld predictive memory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass(frozen=True)
class MultiScaleGeometryMemory:
    """Ego-aligned geometry pyramid.

    ``levels[i]`` is ``[B,C_i,Ny_i,Nx_i]`` and ``valid_masks[i]`` is
    optionally ``[B,Ny_i,Nx_i]``.  Scale factors are relative to level zero.
    """

    levels: Tuple[torch.Tensor, ...]
    scale_factors: Tuple[int, ...]
    valid_masks: Optional[Tuple[torch.Tensor, ...]] = None

    def validate(self) -> "MultiScaleGeometryMemory":
        if not self.levels or len(self.levels) != len(self.scale_factors):
            raise ValueError("geometry levels and scale_factors must be non-empty")
        if self.scale_factors[0] != 1 or any(
            right <= left
            for left, right in zip(self.scale_factors, self.scale_factors[1:])
        ):
            raise ValueError("geometry scale_factors must start at 1 and increase")
        batch = self.levels[0].shape[0] if self.levels[0].ndim == 4 else -1
        base_h, base_w = self.levels[0].shape[-2:]
        for level, factor in zip(self.levels, self.scale_factors):
            if level.ndim != 4 or level.shape[0] != batch:
                raise ValueError("geometry levels must have shape [B,C,Ny,Nx]")
            if (base_h // factor, base_w // factor) != level.shape[-2:]:
                raise ValueError("geometry level resolution differs from scale factor")
        if self.valid_masks is not None:
            if len(self.valid_masks) != len(self.levels):
                raise ValueError("geometry valid mask count differs from levels")
            for level, mask in zip(self.levels, self.valid_masks):
                if mask.shape != (batch, *level.shape[-2:]):
                    raise ValueError("geometry valid masks must have shape [B,Ny,Nx]")
        return self

    @property
    def finest(self) -> torch.Tensor:
        return self.levels[0]


@dataclass(frozen=True)
class CurrentDynamicsMemory:
    """Action-free current/history dynamics field ``[B,Cd,Ny,Nx]``."""

    field: torch.Tensor
    valid_mask: Optional[torch.Tensor] = None

    def validate(self) -> "CurrentDynamicsMemory":
        if self.field.ndim != 4:
            raise ValueError("current dynamics field must have shape [B,C,Ny,Nx]")
        if self.valid_mask is not None and self.valid_mask.shape != (
            self.field.shape[0],
            self.field.shape[2],
            self.field.shape[3],
        ):
            raise ValueError("current dynamics valid_mask must have shape [B,Ny,Nx]")
        return self


@dataclass(frozen=True)
class PredictiveWorldMemory:
    """Current dynamics and action-free future prediction.

    Shapes are ``future=[B,H,Cd,Ny,Nx]`` and
    ``log_variance=[B,H,1,Ny,Nx]``.
    """

    current: CurrentDynamicsMemory
    future: torch.Tensor
    log_variance: torch.Tensor

    def validate(self) -> "PredictiveWorldMemory":
        self.current.validate()
        if self.future.ndim != 5:
            raise ValueError("future memory must have shape [B,H,C,Ny,Nx]")
        batch, horizon, _, ny, nx = self.future.shape
        if self.current.field.shape[0] != batch or self.current.field.shape[-2:] != (
            ny,
            nx,
        ):
            raise ValueError("current and future memory dimensions differ")
        if self.log_variance.shape != (batch, horizon, 1, ny, nx):
            raise ValueError("future log_variance must have shape [B,H,1,Ny,Nx]")
        return self
