"""Multi-scale ego geometry memory built from a grounded finest field."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .types import MultiScaleGeometryMemory


class MultiScaleGeometryMemoryWriter(nn.Module):
    """Create an explicit geometry pyramid from ``[B,C,Ny,Nx]``.

    The operation contains no trajectory or future input.  Channel projections
    are declared in ``__init__`` and applied pointwise in NHWC layout.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: Sequence[int],
        scale_factors: Sequence[int] = (1, 2, 4),
    ) -> None:
        super().__init__()
        channels = tuple(int(value) for value in output_channels)
        factors = tuple(int(value) for value in scale_factors)
        if input_channels <= 0 or not channels or min(channels) <= 0:
            raise ValueError("geometry memory channels must be positive")
        if len(channels) != len(factors):
            raise ValueError("output_channels and scale_factors length must match")
        if factors[0] != 1 or any(
            right <= left for left, right in zip(factors, factors[1:])
        ):
            raise ValueError("scale_factors must start at 1 and increase")
        self.input_channels = int(input_channels)
        self.output_channels = channels
        self.scale_factors = factors
        self.projections = nn.ModuleList(
            [nn.Linear(self.input_channels, channel) for channel in channels]
        )

    def forward(
        self,
        finest_field: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> MultiScaleGeometryMemory:
        """Return levels ``[B,C_i,Ny/f_i,Nx/f_i]``."""

        if finest_field.ndim != 4 or finest_field.shape[1] != self.input_channels:
            raise ValueError("finest_field must have shape [B,C,Ny,Nx]")
        batch, _, ny, nx = finest_field.shape
        if any(ny % factor or nx % factor for factor in self.scale_factors):
            raise ValueError("field resolution must be divisible by every scale factor")
        if valid_mask is not None and valid_mask.shape != (batch, ny, nx):
            raise ValueError("valid_mask must have shape [B,Ny,Nx]")

        levels = []
        masks = [] if valid_mask is not None else None
        for factor, projection in zip(self.scale_factors, self.projections):
            pooled = (
                finest_field
                if factor == 1
                else F.avg_pool2d(finest_field.float(), factor, factor).to(
                    dtype=finest_field.dtype
                )
            )
            level = projection(pooled.permute(0, 2, 3, 1))
            levels.append(level.permute(0, 3, 1, 2).contiguous())
            if masks is not None:
                pooled_mask = (
                    valid_mask
                    if factor == 1
                    else F.max_pool2d(
                        valid_mask[:, None].float(), factor, factor
                    )[:, 0]
                    > 0
                )
                masks.append(pooled_mask.to(dtype=torch.bool))
        return MultiScaleGeometryMemory(
            tuple(levels),
            self.scale_factors,
            tuple(masks) if masks is not None else None,
        ).validate()
