"""Lossless tensor codec between Flow actions and NAVSIM pose trajectories."""

from __future__ import annotations

import torch
from torch import Tensor


# Shared with navsim_dataset.py.  Do not duplicate these values in the loader.
FLOW_X_MEAN = 10.172484
FLOW_X_STD = 8.805105
FLOW_Y_MEAN = 0.360762
FLOW_Y_STD = 2.277741

TRAJECTORY_40_TO_8_INDICES = (4, 9, 14, 19, 24, 29, 34, 39)


def wrap_to_pi(angle: Tensor) -> Tensor:
    """Wrap an angle tensor to ``[-pi, pi)`` on its existing device/dtype."""

    return torch.remainder(angle + torch.pi, 2.0 * torch.pi) - torch.pi


class TrajectoryCodec:
    """Convert normalized ``[x,y,sin(h),cos(h)]`` actions and metric poses.

    All methods accept arbitrary leading batch/candidate dimensions and retain
    the input tensor's device and floating dtype.
    """

    action_dim = 4
    num_poses_8 = 8
    num_poses_40 = 40

    @staticmethod
    def _validate(value: Tensor, points: int, width: int, name: str) -> None:
        if not torch.is_tensor(value) or not value.is_floating_point():
            raise TypeError(f"{name} must be a floating-point tensor")
        if value.ndim < 2 or tuple(value.shape[-2:]) != (points, width):
            raise ValueError(
                f"{name} must end with [{points},{width}], got {tuple(value.shape)}"
            )
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains NaN or Inf")

    def flow_to_navsim(self, action_8x4: Tensor) -> Tensor:
        """Decode normalized Flow actions ``[...,8,4]`` to metric ``[...,8,3]``."""

        self._validate(action_8x4, 8, 4, "Flow action")
        x = action_8x4[..., 0] * FLOW_X_STD + FLOW_X_MEAN
        y = action_8x4[..., 1] * FLOW_Y_STD + FLOW_Y_MEAN
        heading = torch.atan2(action_8x4[..., 2], action_8x4[..., 3])
        return torch.stack((x, y, heading), dim=-1)

    def navsim_to_flow(self, trajectory_8x3: Tensor) -> Tensor:
        """Encode metric poses ``[...,8,3]`` into normalized Flow actions."""

        self._validate(trajectory_8x3, 8, 3, "NAVSIM trajectory")
        x = (trajectory_8x3[..., 0] - FLOW_X_MEAN) / FLOW_X_STD
        y = (trajectory_8x3[..., 1] - FLOW_Y_MEAN) / FLOW_Y_STD
        heading = trajectory_8x3[..., 2]
        return torch.stack((x, y, torch.sin(heading), torch.cos(heading)), dim=-1)

    def upsample_8_to_40(self, trajectory_8x3: Tensor) -> Tensor:
        """Interpolate 0.5-second poses to 0.1-second poses without learning."""

        self._validate(trajectory_8x3, 8, 3, "NAVSIM trajectory")
        # Do the angular interpolation in fp32 when AMP supplies fp16/bf16.
        # Besides improving wrap precision, this avoids backend-dependent type
        # promotion from scalar pi constants.  The public result is cast back
        # to the caller's dtype before exact anchors are restored.
        work_dtype = (
            torch.float32
            if trajectory_8x3.dtype in (torch.float16, torch.bfloat16)
            else trajectory_8x3.dtype
        )
        work_trajectory = trajectory_8x3.to(dtype=work_dtype)
        origin = torch.zeros_like(work_trajectory[..., :1, :])
        points = torch.cat((origin, work_trajectory), dim=-2)

        raw_heading = points[..., 2]
        heading_delta = wrap_to_pi(raw_heading[..., 1:] - raw_heading[..., :-1])
        unwrapped_heading = torch.cat(
            (
                raw_heading[..., :1],
                raw_heading[..., :1] + torch.cumsum(heading_delta, dim=-1),
            ),
            dim=-1,
        )

        target_times = torch.arange(
            1, 41, device=trajectory_8x3.device, dtype=work_dtype
        ) * 0.1
        scaled = target_times / 0.5
        left = torch.floor(scaled).to(torch.long).clamp(max=7)
        right = left + 1
        weight = (scaled - left.to(scaled.dtype)).view(
            *((1,) * (trajectory_8x3.ndim - 2)), 40, 1
        )

        xy_left = points[..., left, :2]
        xy_right = points[..., right, :2]
        xy = xy_left + weight * (xy_right - xy_left)
        heading_left = unwrapped_heading[..., left]
        heading_right = unwrapped_heading[..., right]
        heading_weight = weight.squeeze(-1)
        heading = wrap_to_pi(
            heading_left + heading_weight * (heading_right - heading_left)
        )
        result = torch.cat((xy, heading.unsqueeze(-1)), dim=-1).to(
            dtype=trajectory_8x3.dtype
        )
        # Preserve the source anchors bit-for-bit.  This also treats the
        # equivalent +pi/-pi boundary consistently with the public round-trip
        # contract instead of changing an anchor by exactly 2*pi.
        result[..., list(TRAJECTORY_40_TO_8_INDICES), :] = trajectory_8x3
        # This is also a runtime assertion of the 0.5s/0.1s convention.
        torch.testing.assert_close(
            result[..., list(TRAJECTORY_40_TO_8_INDICES), :],
            trajectory_8x3,
            rtol=1e-4,
            atol=1e-5,
        )
        return result

    def downsample_40_to_8(self, trajectory_40x3: Tensor) -> Tensor:
        """Select the eight 0.5-second anchors from ``[...,40,3]``."""

        self._validate(trajectory_40x3, 40, 3, "NAVSIM 40-point trajectory")
        return trajectory_40x3[..., list(TRAJECTORY_40_TO_8_INDICES), :]
