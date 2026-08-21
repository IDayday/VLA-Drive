"""Deterministic NAVSIM 8-point to DriveSuprim 40-point resampling."""

from __future__ import annotations

import math

import torch


STATIC_SAMPLE_INDICES = (4, 9, 14, 19, 24, 29, 34, 39)


def _wrap_heading(heading: torch.Tensor) -> torch.Tensor:
    return torch.remainder(heading + math.pi, 2.0 * math.pi) - math.pi


def trajectory_8_to_40(trajectory_8: torch.Tensor) -> torch.Tensor:
    """Linearly interpolate x/y and unwrapped heading on the fixed time grid.

    ``trajectory_8`` represents t=0.5, 1.0, ..., 4.0 seconds.  The implicit
    pose at t=0 is exactly (0, 0, 0); the returned samples represent
    t=0.1, 0.2, ..., 4.0 seconds.
    """

    if not torch.is_tensor(trajectory_8):
        raise TypeError("trajectory_8 must be a torch.Tensor")
    if trajectory_8.ndim < 2 or trajectory_8.shape[-2:] != (8, 3):
        raise ValueError(
            f"trajectory_8 must end in [8, 3], got {tuple(trajectory_8.shape)}"
        )
    if not trajectory_8.is_floating_point():
        raise TypeError("trajectory_8 must use a floating-point dtype")

    origin = torch.zeros_like(trajectory_8[..., :1, :])
    trajectory_9 = torch.cat((origin, trajectory_8), dim=-2)

    # Unwrap heading by accumulating shortest signed angular increments.
    heading = trajectory_9[..., 2]
    heading_delta = _wrap_heading(heading[..., 1:] - heading[..., :-1])
    unwrapped_heading = torch.cat(
        (heading[..., :1], heading[..., :1] + torch.cumsum(heading_delta, dim=-1)),
        dim=-1,
    )

    target_times = torch.arange(
        1, 41, device=trajectory_8.device, dtype=torch.float64
    ) / 10.0
    source_times = torch.arange(
        0, 9, device=trajectory_8.device, dtype=torch.float64
    ) / 2.0
    lower = torch.searchsorted(source_times, target_times, right=False) - 1
    lower = lower.clamp_(0, 7)
    upper = lower + 1
    alpha = ((target_times - source_times[lower]) / 0.5).to(trajectory_8.dtype)

    xy_lower = trajectory_9[..., lower, :2]
    xy_upper = trajectory_9[..., upper, :2]
    xy = torch.lerp(xy_lower, xy_upper, alpha[..., None])

    heading_lower = unwrapped_heading[..., lower]
    heading_upper = unwrapped_heading[..., upper]
    interpolated_heading = torch.lerp(heading_lower, heading_upper, alpha)
    interpolated_heading = _wrap_heading(interpolated_heading)

    trajectory_40 = torch.cat((xy, interpolated_heading[..., None]), dim=-1)
    if not torch.isfinite(trajectory_40).all():
        raise ValueError("trajectory interpolation produced NaN or Inf")
    return trajectory_40
