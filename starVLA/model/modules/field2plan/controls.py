"""Teacher-specific scientific controls for Field2Plan ablations."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
from torch import nn


@dataclass(frozen=True)
class ControlledGeometryTeacher:
    """Controlled depth/confidence maps ``[B,V,Hd,Wd]`` and provenance."""

    depth_m: torch.Tensor
    confidence: torch.Tensor
    permutation: torch.Tensor
    mode: str


@dataclass(frozen=True)
class ControlledDynamicsTeacher:
    """Controlled future features and their explicit permutations.

    Shapes are features ``[B,H,V,C,Ht,Wt]``, confidence
    ``[B,H,V,Ht,Wt]``, batch permutation ``[B]``, and temporal permutation
    ``[H]``.
    """

    features: torch.Tensor
    confidence: torch.Tensor
    batch_permutation: torch.Tensor
    temporal_permutation: torch.Tensor
    mode: str


def _token_phase(token: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{token}".encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return (integer / float(2**64 - 1)) * (2.0 * math.pi)


def apply_geometry_teacher_controls(
    depth_m: torch.Tensor,
    confidence: torch.Tensor,
    tokens: Sequence[str],
    *,
    seed: int = 0,
    random_teacher: bool = False,
    shuffle_teacher_across_batch: bool = False,
    random_depth_range_m: Tuple[float, float] = (1.0, 80.0),
) -> ControlledGeometryTeacher:
    """Apply real/random/shuffled teacher controls deterministically.

    Random targets are a frozen token-seeded spatial field independent of the
    real scene values. Shuffling permutes complete samples across the batch.
    The two controls are intentionally mutually exclusive.
    """

    if depth_m.ndim != 4 or confidence.shape != depth_m.shape:
        raise ValueError("depth_m/confidence must share shape [B,V,Hd,Wd]")
    if len(tokens) != depth_m.shape[0]:
        raise ValueError("tokens length must match teacher batch size")
    if random_teacher and shuffle_teacher_across_batch:
        raise ValueError("random and shuffled teacher controls are mutually exclusive")
    if random_depth_range_m[0] <= 0 or random_depth_range_m[0] >= random_depth_range_m[1]:
        raise ValueError("random_depth_range_m must be positive and increasing")
    batch = depth_m.shape[0]
    identity = torch.arange(batch, device=depth_m.device, dtype=torch.long)

    if random_teacher:
        points = depth_m[0].numel()
        index = torch.arange(
            points, device=depth_m.device, dtype=torch.float32
        ).reshape(depth_m.shape[1:])
        generated = []
        low, high = map(float, random_depth_range_m)
        for token in tokens:
            phase = _token_phase(str(token), int(seed))
            unit = torch.remainder(
                torch.sin(index * 12.9898 + phase) * 43758.5453, 1.0
            )
            generated.append(low + unit * (high - low))
        random_depth = torch.stack(generated).to(dtype=torch.float32)
        return ControlledGeometryTeacher(
            depth_m=random_depth,
            confidence=torch.ones_like(random_depth),
            permutation=identity,
            mode="random",
        )

    if shuffle_teacher_across_batch:
        if batch < 2:
            raise ValueError("shuffled teacher requires batch size >= 2")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        permutation = torch.randperm(batch, generator=generator)
        if torch.equal(permutation, torch.arange(batch)):
            permutation = torch.roll(permutation, shifts=1)
        permutation = permutation.to(device=depth_m.device)
        return ControlledGeometryTeacher(
            depth_m=depth_m[permutation],
            confidence=confidence[permutation],
            permutation=permutation,
            mode="shuffled",
        )

    return ControlledGeometryTeacher(
        depth_m=depth_m,
        confidence=confidence,
        permutation=identity,
        mode="real",
    )


def apply_dynamics_teacher_controls(
    features: torch.Tensor,
    confidence: torch.Tensor,
    tokens: Sequence[str],
    *,
    seed: int = 0,
    random_teacher: bool = False,
    shuffle_teacher_across_batch: bool = False,
    temporal_shuffle: bool = False,
) -> ControlledDynamicsTeacher:
    """Apply deterministic controls to ``[B,H,V,C,Ht,Wt]`` features."""

    if features.ndim != 6:
        raise ValueError("dynamics features must have shape [B,H,V,C,Ht,Wt]")
    expected_confidence = (
        features.shape[0],
        features.shape[1],
        features.shape[2],
        features.shape[4],
        features.shape[5],
    )
    if confidence.shape != expected_confidence:
        raise ValueError("dynamics confidence must have shape [B,H,V,Ht,Wt]")
    if len(tokens) != features.shape[0]:
        raise ValueError("tokens length must match dynamics batch size")
    control_count = sum(
        int(value)
        for value in (
            random_teacher,
            shuffle_teacher_across_batch,
            temporal_shuffle,
        )
    )
    if control_count > 1:
        raise ValueError("dynamics teacher controls are mutually exclusive")
    batch, horizon = features.shape[:2]
    batch_identity = torch.arange(batch, device=features.device, dtype=torch.long)
    temporal_identity = torch.arange(
        horizon, device=features.device, dtype=torch.long
    )

    if random_teacher:
        flat_shape = features.shape[1:]
        point_count = math.prod(flat_shape)
        indices = torch.arange(
            point_count, device=features.device, dtype=torch.float32
        ).reshape(flat_shape)
        generated = []
        for token in tokens:
            phase = _token_phase(str(token), int(seed))
            generated.append(torch.sin(indices * 0.0137 + phase))
        random_features = torch.stack(generated).to(dtype=features.dtype)
        random_features = torch.nn.functional.normalize(
            random_features, dim=3, eps=1e-6
        )
        return ControlledDynamicsTeacher(
            features=random_features,
            confidence=torch.ones_like(confidence),
            batch_permutation=batch_identity,
            temporal_permutation=temporal_identity,
            mode="random",
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    if shuffle_teacher_across_batch:
        if batch < 2:
            raise ValueError("shuffled dynamics teacher requires batch size >= 2")
        permutation = torch.randperm(batch, generator=generator)
        if torch.equal(permutation, torch.arange(batch)):
            permutation = torch.roll(permutation, shifts=1)
        permutation = permutation.to(device=features.device)
        return ControlledDynamicsTeacher(
            features=features[permutation],
            confidence=confidence[permutation],
            batch_permutation=permutation,
            temporal_permutation=temporal_identity,
            mode="shuffled",
        )

    if temporal_shuffle:
        if horizon < 2:
            raise ValueError("temporal shuffle requires horizon >= 2")
        permutation = torch.randperm(horizon, generator=generator)
        if torch.equal(permutation, torch.arange(horizon)):
            permutation = torch.roll(permutation, shifts=1)
        permutation = permutation.to(device=features.device)
        return ControlledDynamicsTeacher(
            features=features[:, permutation],
            confidence=confidence[:, permutation],
            batch_permutation=batch_identity,
            temporal_permutation=permutation,
            mode="temporal_shuffled",
        )

    return ControlledDynamicsTeacher(
        features=features,
        confidence=confidence,
        batch_permutation=batch_identity,
        temporal_permutation=temporal_identity,
        mode="real",
    )


class GTMLPFieldControl(nn.Module):
    """Learned current-state-only field used as a teacher-specific control.

    Input is the existing normalized current ego-motion state ``[B,1,4]``
    (or ``[B,4]``). It has no route to images, demonstrated future trajectory,
    future poses, proposal drafts, or evaluator outcomes.
    """

    def __init__(
        self,
        state_dim: int,
        output_channels: int,
        field_size: Sequence[int],
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if min(state_dim, output_channels, hidden_dim) <= 0:
            raise ValueError("GT-MLP dimensions must be positive")
        if len(field_size) != 2 or min(field_size) <= 0:
            raise ValueError("GT-MLP field_size must be positive [Ny,Nx]")
        self.state_dim = int(state_dim)
        self.output_channels = int(output_channels)
        self.field_size = (int(field_size[0]), int(field_size[1]))
        self.state_projection = nn.Linear(self.state_dim, int(hidden_dim))
        self.spatial_embedding = nn.Parameter(
            torch.zeros(1, *self.field_size, int(hidden_dim))
        )
        nn.init.normal_(self.spatial_embedding, mean=0.0, std=0.02)
        self.output_projection = nn.Sequential(
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.output_channels),
        )

    def forward(self, current_state: torch.Tensor) -> torch.Tensor:
        """Return a control field ``[B,C,Ny,Nx]`` from current state only."""

        if current_state.ndim == 3:
            if current_state.shape[1] != 1:
                raise ValueError("current_state temporal dimension must be 1")
            current_state = current_state[:, 0]
        if current_state.ndim != 2 or current_state.shape[-1] != self.state_dim:
            raise ValueError(f"current_state must have shape [B,{self.state_dim}]")
        state = self.state_projection(current_state.float())[:, None, None]
        hidden = state + self.spatial_embedding.to(
            device=state.device, dtype=state.dtype
        )
        field = self.output_projection(hidden)
        return field.permute(0, 3, 1, 2).contiguous()
