"""Adapters from offline current/history teacher features to student memory."""

from __future__ import annotations

import torch
from torch import nn


class ExternalPriorAdapter(nn.Module):
    """Aggregate ``[B,Th,V,Ct,Ht,Wt]`` into global ``[B,Cd]``.

    Confidence is ``[B,Th,V,Ht,Wt]``. A cached teacher patch grid is not an
    ego-aligned BEV unless its preprocessing and camera transform explicitly
    establish that correspondence. The revised MVP therefore transfers a
    global current/history dynamics prior and leaves local BEV structure to
    the calibrated geometry/history path. This declared adapter is retained
    for no-teacher controls so parameter capacity remains identical.
    """

    def __init__(self, teacher_channels: int, output_channels: int) -> None:
        super().__init__()
        if min(int(teacher_channels), int(output_channels)) <= 0:
            raise ValueError("prior adapter channels must be positive")
        self.teacher_channels = int(teacher_channels)
        self.output_channels = int(output_channels)
        self.projection = nn.Sequential(
            nn.LayerNorm(self.teacher_channels),
            nn.Linear(self.teacher_channels, self.output_channels),
        )

    def forward(
        self,
        features: torch.Tensor,
        confidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return global target ``[B,Cd]`` and scene weights ``[B]``."""

        if features.ndim != 6 or features.shape[3] != self.teacher_channels:
            raise ValueError("prior features must have shape [B,Th,V,Ct,Ht,Wt]")
        if confidence.shape != (
            features.shape[0],
            features.shape[1],
            features.shape[2],
            features.shape[4],
            features.shape[5],
        ):
            raise ValueError("prior confidence shape differs from features")
        projected = self.projection(features.float().movedim(3, -1))
        token_weights = confidence.float().unsqueeze(-1)
        denominator = token_weights.sum(dim=(1, 2, 3, 4)).clamp_min(1e-6)
        target = (projected * token_weights).sum(dim=(1, 2, 3, 4)) / denominator
        weights = confidence.float().mean(dim=(1, 2, 3, 4)).clamp(0.0, 1.0)
        return target, weights
