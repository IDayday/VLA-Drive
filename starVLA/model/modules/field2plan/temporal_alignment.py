"""Explicit NAVSIM temporal and ego-motion alignment primitives."""

from __future__ import annotations

from typing import Sequence

import torch

from .types import TemporalAlignment


def se2_poses_to_transforms(global_poses: torch.Tensor) -> torch.Tensor:
    """Convert ``[...,T,3]`` ``(x,y,yaw)`` poses to ``[...,T,4,4]``.

    The returned matrix maps a point in a frame's ego coordinates into the
    global coordinate system.  Construction is differentiable and inherits
    the input device; transform arithmetic is performed in float32.
    """

    poses = torch.as_tensor(global_poses)
    if poses.ndim not in (2, 3) or poses.shape[-1] != 3:
        raise ValueError("global_poses must have shape [T,3] or [B,T,3]")
    poses = poses.to(dtype=torch.float32)
    x, y, yaw = poses.unbind(dim=-1)
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    transforms = torch.zeros(
        *poses.shape[:-1], 4, 4, device=poses.device, dtype=torch.float32
    )
    transforms[..., 0, 0] = cosine
    transforms[..., 0, 1] = -sine
    transforms[..., 1, 0] = sine
    transforms[..., 1, 1] = cosine
    transforms[..., 2, 2] = 1.0
    transforms[..., 3, 3] = 1.0
    transforms[..., 0, 3] = x
    transforms[..., 1, 3] = y
    return transforms


def build_temporal_alignment(
    global_poses: torch.Tensor,
    *,
    current_index: int,
    history_indices: Sequence[int],
    future_indices: Sequence[int],
    frame_interval_s: float,
    valid_mask: torch.Tensor | None = None,
) -> TemporalAlignment:
    """Build transforms without exposing future motion to the dynamics writer.

    This function computes both directions for alignment and supervision.  A
    caller constructing an action-free writer must pass only the selected
    ``history_indices`` slice of ``current_from_ego`` to that writer.
    """

    transforms = se2_poses_to_transforms(global_poses)
    time_count = transforms.shape[-3]
    current_index = int(current_index)
    history = tuple(int(index) for index in history_indices)
    future = tuple(int(index) for index in future_indices)
    if frame_interval_s <= 0:
        raise ValueError("frame_interval_s must be positive")
    if not 0 <= current_index < time_count:
        raise ValueError("current_index is outside the temporal sequence")
    if not history or any(index < 0 or index >= time_count for index in history):
        raise ValueError("history indices are outside the temporal sequence")
    if any(index > current_index for index in history):
        raise ValueError("history indices cannot occur after current_index")
    if not future or any(index <= current_index or index >= time_count for index in future):
        raise ValueError("future indices must occur after current_index")

    current_global_from_ego = transforms[..., current_index, :, :]
    global_to_current = torch.linalg.inv(current_global_from_ego)
    current_from_ego = global_to_current[..., None, :, :] @ transforms
    ego_from_current = torch.linalg.inv(current_from_ego)
    if valid_mask is None:
        validity = torch.ones(
            transforms.shape[:-2], device=transforms.device, dtype=torch.bool
        )
    else:
        validity = torch.as_tensor(valid_mask, device=transforms.device)
        if validity.shape != transforms.shape[:-2]:
            raise ValueError("valid_mask must have shape [T] or [B,T]")
        validity = validity.to(dtype=torch.bool)
    frame_times_s = torch.arange(
        time_count, device=transforms.device, dtype=torch.float32
    ) * float(frame_interval_s)
    return TemporalAlignment(
        global_from_ego=transforms,
        current_from_ego=current_from_ego,
        ego_from_current=ego_from_current,
        frame_times_s=frame_times_s,
        valid_mask=validity,
        current_index=current_index,
        history_indices=history,
        future_indices=future,
    ).validate()


def interpolate_temporal_features(
    features: torch.Tensor,
    source_times_s: torch.Tensor,
    query_times_s: torch.Tensor,
    *,
    mode: str = "linear",
) -> torch.Tensor:
    """Resample ``[B,T,C,Ny,Nx]`` features at query times ``[Q]``."""

    if features.ndim != 5:
        raise ValueError("features must have shape [B,T,C,Ny,Nx]")
    source = torch.as_tensor(
        source_times_s, device=features.device, dtype=torch.float32
    )
    query = torch.as_tensor(
        query_times_s, device=features.device, dtype=torch.float32
    )
    if source.ndim != 1 or source.shape[0] != features.shape[1]:
        raise ValueError("source_times_s must have shape [T]")
    if query.ndim != 1 or query.numel() == 0:
        raise ValueError("query_times_s must be a non-empty [Q] tensor")
    if source.numel() < 1 or (source[1:] <= source[:-1]).any():
        raise ValueError("source_times_s must be strictly increasing")
    if mode not in {"nearest", "linear"}:
        raise ValueError("temporal interpolation mode must be nearest or linear")

    if mode == "nearest" or source.numel() == 1:
        distances = (query[:, None] - source[None]).abs()
        indices = distances.argmin(dim=1)
        return features.index_select(1, indices)

    upper = torch.searchsorted(source, query, right=False).clamp(1, source.numel() - 1)
    lower = upper - 1
    lower_time = source.index_select(0, lower)
    upper_time = source.index_select(0, upper)
    alpha = ((query - lower_time) / (upper_time - lower_time).clamp_min(1e-8)).clamp(
        0.0, 1.0
    )
    lower_features = features.index_select(1, lower)
    upper_features = features.index_select(1, upper)
    weight = alpha.to(dtype=features.dtype).reshape(1, -1, 1, 1, 1)
    return lower_features + (upper_features - lower_features) * weight
