"""Deterministic codec between the unchanged DDP action and planner spaces.

The cfg_yaw_1226 action head predicts normalized body-frame SE(2) deltas.
DrivoR and DriveSuprim consume trajectories in metric ego-frame poses.  These
pure tensor functions keep that conversion at the DDP-DRS boundary and return
the selected pose trajectory to the existing action decoder in its original
normalized-delta representation.
"""

from __future__ import annotations

import torch
from torch import Tensor


ACTION_Q01 = (
    -0.01789146974507183,
    -0.19088272509455573,
    -0.1892357842470911,
)
ACTION_Q99 = (
    6.199554522088146,
    0.24262804072441968,
    0.1804889553518122,
)


def wrap_to_pi(angle: Tensor) -> Tensor:
    """Wrap radians to ``[-pi, pi)`` without leaving the tensor device."""

    return torch.remainder(angle + torch.pi, 2.0 * torch.pi) - torch.pi


def _statistics(reference: Tensor) -> tuple[Tensor, Tensor]:
    q01 = reference.new_tensor(ACTION_Q01)
    q99 = reference.new_tensor(ACTION_Q99)
    return q01, q99


def normalized_deltas_to_poses(action_norm: Tensor) -> Tensor:
    """Decode cfg_yaw_1226 normalized body deltas to metric SE(2) poses.

    Any leading dimensions are supported; only the final ``[T, 3]`` contract
    is fixed.  The origin before the first point is ``(0, 0, 0)``.
    """

    if not torch.is_tensor(action_norm) or action_norm.ndim < 2:
        raise ValueError("normalized actions must have shape [..., T, 3]")
    if action_norm.shape[-1] != 3:
        raise ValueError("normalized actions must contain [dx, dy, d_heading]")
    if not action_norm.is_floating_point():
        raise TypeError("normalized actions must be floating-point tensors")
    if not torch.isfinite(action_norm).all():
        raise ValueError("normalized actions contain NaN or Inf")

    q01, q99 = _statistics(action_norm)
    deltas = (action_norm + 1.0) * 0.5 * (q99 - q01) + q01
    delta_heading = wrap_to_pi(deltas[..., 2])

    # Heading before step t is the cumulative heading through step t-1.
    heading = wrap_to_pi(torch.cumsum(delta_heading, dim=-1))
    zero_heading = torch.zeros_like(heading[..., :1])
    previous_heading = torch.cat((zero_heading, heading[..., :-1]), dim=-1)
    cosine = torch.cos(previous_heading)
    sine = torch.sin(previous_heading)
    world_dx = cosine * deltas[..., 0] - sine * deltas[..., 1]
    world_dy = sine * deltas[..., 0] + cosine * deltas[..., 1]
    x = torch.cumsum(world_dx, dim=-1)
    y = torch.cumsum(world_dy, dim=-1)
    return torch.stack((x, y, heading), dim=-1)


def poses_to_normalized_deltas(poses: Tensor) -> Tensor:
    """Inverse of :func:`normalized_deltas_to_poses` for an origin at zero.

    Values are deliberately not clipped to ``[-1, 1]``: clipping would make
    the conversion lossy for a selected static-vocabulary trajectory and the
    unchanged downstream decoder accepts the affine inverse for all values.
    """

    if not torch.is_tensor(poses) or poses.ndim < 2:
        raise ValueError("poses must have shape [..., T, 3]")
    if poses.shape[-1] != 3:
        raise ValueError("poses must contain [x, y, heading]")
    if not poses.is_floating_point():
        raise TypeError("poses must be floating-point tensors")
    if not torch.isfinite(poses).all():
        raise ValueError("poses contain NaN or Inf")

    zeros = torch.zeros_like(poses[..., :1, :])
    previous = torch.cat((zeros, poses[..., :-1, :]), dim=-2)
    world_dx = poses[..., 0] - previous[..., 0]
    world_dy = poses[..., 1] - previous[..., 1]
    previous_heading = previous[..., 2]
    cosine = torch.cos(previous_heading)
    sine = torch.sin(previous_heading)
    body_dx = cosine * world_dx + sine * world_dy
    body_dy = -sine * world_dx + cosine * world_dy
    delta_heading = wrap_to_pi(poses[..., 2] - previous_heading)
    deltas = torch.stack((body_dx, body_dy, delta_heading), dim=-1)

    q01, q99 = _statistics(poses)
    return 2.0 * (deltas - q01) / (q99 - q01) - 1.0
