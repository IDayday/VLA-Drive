"""Shared numerical guard for Register trajectory proposals.

The fixed 100 m XY bound and heading canonicalization follow the released
CLOVER proposal-sanitization contract.  Keeping the operation at the generator
boundary guarantees that training, metric evaluation, candidate-bank export,
and inference consume identical physical trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


REGISTER_XY_LIMIT_METERS = 100.0


@dataclass(frozen=True)
class TrajectorySanitizationStats:
    """Detached counters describing one sanitization call."""

    nonfinite_count: Tensor
    xy_clamped_count: Tensor
    heading_wrapped_count: Tensor
    coordinate_count: Tensor

    def rates(self) -> dict[str, Tensor]:
        denominator = self.coordinate_count.clamp_min(1).to(torch.float32)
        return {
            "proposal_nonfinite_rate": self.nonfinite_count.to(torch.float32)
            / denominator,
            "proposal_xy_clamped_rate": self.xy_clamped_count.to(torch.float32)
            / denominator,
            "proposal_heading_wrapped_rate": self.heading_wrapped_count.to(
                torch.float32
            )
            / denominator,
        }


def sanitize_register_trajectories(
    trajectories: Tensor,
    *,
    xy_limit: float = REGISTER_XY_LIMIT_METERS,
) -> tuple[Tensor, TrajectorySanitizationStats]:
    """Return finite ``[..., 8, 3]`` trajectories in canonical NAVSIM space."""

    if trajectories.ndim < 3 or tuple(trajectories.shape[-2:]) != (8, 3):
        raise ValueError("Register trajectories must end in [8,3]")
    if not xy_limit > 0:
        raise ValueError("xy_limit must be positive")

    original = trajectories
    finite = torch.isfinite(original)
    cleaned = torch.nan_to_num(
        original,
        nan=0.0,
        posinf=float(xy_limit),
        neginf=-float(xy_limit),
    )
    xy_before = cleaned[..., :2]
    xy = xy_before.clamp(min=-float(xy_limit), max=float(xy_limit))
    heading_before = cleaned[..., 2]
    heading = torch.atan2(torch.sin(heading_before), torch.cos(heading_before))
    sanitized = torch.cat((xy, heading.unsqueeze(-1)), dim=-1)

    stats = TrajectorySanitizationStats(
        nonfinite_count=(~finite).sum().detach(),
        xy_clamped_count=(xy != xy_before).sum().detach(),
        heading_wrapped_count=(
            (heading - heading_before).abs() > 1.0e-6
        ).sum().detach(),
        coordinate_count=torch.as_tensor(
            original.numel(), device=original.device, dtype=torch.long
        ),
    )
    return sanitized, stats
