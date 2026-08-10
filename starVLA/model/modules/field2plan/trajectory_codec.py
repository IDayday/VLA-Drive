"""Single-source trajectory normalization and geometry utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple, Union

import numpy as np
import torch


ArrayLike = Union[np.ndarray, torch.Tensor]


@dataclass(frozen=True)
class TrajectoryStats:
    """Statistics used by the verified ``ver_1225=1, act_norm=1`` baseline."""

    x_mean: float = 10.172484
    x_std: float = 8.805105
    y_mean: float = 0.360762
    y_std: float = 2.277741


class TrajectoryCodec:
    """Encode/decode baseline trajectories using a torch implementation.

    Accepted leading dimensions are arbitrary.  The last two dimensions are
    ``[H,3]`` for physical ``(x,y,theta)`` or ``[H,4]`` for normalized
    ``(x,y,sin(theta),cos(theta))``. NumPy inputs are wrapped through the same
    torch formulas and returned as NumPy arrays.
    """

    def __init__(self, stats: TrajectoryStats = TrajectoryStats(), horizon: int = 8):
        self.stats = stats
        self.horizon = int(horizon)

    @staticmethod
    def _to_tensor(value: ArrayLike) -> Tuple[torch.Tensor, bool]:
        if isinstance(value, torch.Tensor):
            if not value.is_floating_point():
                raise TypeError("trajectory tensors must be floating point")
            return value, False
        array = np.asarray(value)
        if not np.issubdtype(array.dtype, np.floating):
            raise TypeError("trajectory arrays must be floating point")
        return torch.from_numpy(array), True

    @staticmethod
    def _restore(value: torch.Tensor, was_numpy: bool) -> ArrayLike:
        return value.detach().cpu().numpy() if was_numpy else value

    def _check(self, value: torch.Tensor, width: int) -> None:
        if value.ndim < 2 or value.shape[-1] != width:
            raise ValueError(f"expected [...,H,{width}], got {tuple(value.shape)}")
        if value.shape[-2] != self.horizon:
            raise ValueError(f"expected horizon {self.horizon}, got {value.shape[-2]}")

    @staticmethod
    def wrap_heading(theta: ArrayLike) -> ArrayLike:
        tensor, was_numpy = TrajectoryCodec._to_tensor(theta)
        wrapped = torch.remainder(tensor + torch.pi, 2 * torch.pi) - torch.pi
        return TrajectoryCodec._restore(wrapped, was_numpy)

    def encode_trajectory(self, trajectory: ArrayLike) -> ArrayLike:
        """Encode physical ``[...,H,3]`` into normalized ``[...,H,4]``."""

        value, was_numpy = self._to_tensor(trajectory)
        self._check(value, 3)
        theta = torch.remainder(value[..., 2] + torch.pi, 2 * torch.pi) - torch.pi
        encoded = torch.stack(
            (
                (value[..., 0] - self.stats.x_mean) / self.stats.x_std,
                (value[..., 1] - self.stats.y_mean) / self.stats.y_std,
                torch.sin(theta),
                torch.cos(theta),
            ),
            dim=-1,
        )
        return self._restore(encoded, was_numpy)

    def decode_action(self, action: ArrayLike) -> ArrayLike:
        """Decode normalized ``[...,H,4]`` into physical ``[...,H,3]``."""

        value, was_numpy = self._to_tensor(action)
        self._check(value, 4)
        decoded = torch.stack(
            (
                value[..., 0] * self.stats.x_std + self.stats.x_mean,
                value[..., 1] * self.stats.y_std + self.stats.y_mean,
                torch.remainder(
                    torch.atan2(value[..., 2], value[..., 3]) + torch.pi,
                    2 * torch.pi,
                )
                - torch.pi,
            ),
            dim=-1,
        )
        return self._restore(decoded, was_numpy)

    def normalize_heading_pair(self, pair: ArrayLike, eps: float = 1e-8) -> ArrayLike:
        """Normalize ``[...,2]`` sin/cos pairs; zero pairs map to ``[0,1]``."""

        value, was_numpy = self._to_tensor(pair)
        if value.ndim < 1 or value.shape[-1] != 2:
            raise ValueError("heading pair must have shape [...,2]")
        norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
        fallback = torch.zeros_like(value)
        fallback[..., 1] = 1
        normalized = torch.where(norm > eps, value / norm.clamp_min(eps), fallback)
        return self._restore(normalized, was_numpy)

    def compose_delta(self, draft_action: ArrayLike, delta_physical: ArrayLike) -> ArrayLike:
        """Apply physical ``(dx,dy,dtheta)`` to a normalized draft.

        Heading composition uses a complex rotation, so a bitwise-zero delta
        preserves even a non-unit baseline sin/cos pair exactly.
        """

        draft, draft_numpy = self._to_tensor(draft_action)
        delta, delta_numpy = self._to_tensor(delta_physical)
        self._check(draft, 4)
        self._check(delta, 3)
        if draft.shape[:-1] != delta.shape[:-1]:
            raise ValueError("draft and delta leading shapes must match")
        if draft_numpy != delta_numpy:
            raise TypeError("draft and delta must both be torch or both be NumPy")
        delta = delta.to(device=draft.device, dtype=draft.dtype)
        sin_delta, cos_delta = torch.sin(delta[..., 2]), torch.cos(delta[..., 2])
        sin_old, cos_old = draft[..., 2], draft[..., 3]
        result = torch.stack(
            (
                draft[..., 0] + delta[..., 0] / self.stats.x_std,
                draft[..., 1] + delta[..., 1] / self.stats.y_std,
                sin_old * cos_delta + cos_old * sin_delta,
                cos_old * cos_delta - sin_old * sin_delta,
            ),
            dim=-1,
        )
        return self._restore(result, draft_numpy)

    def tube_points(
        self,
        physical_trajectory: ArrayLike,
        lateral_offsets_m: Sequence[float],
        longitudinal_offsets_m: Sequence[float],
    ) -> ArrayLike:
        """Create swept tube points ``[...,H,P,2]`` from physical poses."""

        trajectory, was_numpy = self._to_tensor(physical_trajectory)
        if trajectory.ndim < 2 or trajectory.shape[-1] != 3:
            raise ValueError("physical trajectory must have shape [...,H,3]")
        lateral = trajectory.new_tensor(tuple(lateral_offsets_m))
        longitudinal = trajectory.new_tensor(tuple(longitudinal_offsets_m))
        if lateral.numel() == 0 or longitudinal.numel() == 0:
            raise ValueError("tube offsets cannot be empty")
        lat, lon = torch.meshgrid(lateral, longitudinal, indexing="ij")
        lat, lon = lat.flatten(), lon.flatten()
        theta = trajectory[..., 2, None]
        x = trajectory[..., 0, None] + lon * torch.cos(theta) - lat * torch.sin(theta)
        y = trajectory[..., 1, None] + lon * torch.sin(theta) + lat * torch.cos(theta)
        points = torch.stack((x, y), dim=-1)
        return self._restore(points, was_numpy)
