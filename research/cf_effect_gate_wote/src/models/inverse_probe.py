"""Leakage-controlled same-scene inverse retrieval and delta probes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn
from torch.nn import functional as F


INVERSE_EFFECT_DIM = 768
TRAJECTORY_DIM = 24
DELTA_DIM = 6
INVERSE_MODES = ("ego_only", "environment_only", "full_effect")
EGO_KINEMATIC_COLUMNS = (3, 4, 5, 6, 7)
DELTA_SCALES = np.asarray([10.0, 40.0, math.pi, 10.0, 20.0, 20.0], dtype=np.float32)


def _pad(value: npt.NDArray[np.float32], width: int) -> npt.NDArray[np.float32]:
    if value.shape[-1] > width:
        raise ValueError(f"inverse feature width {value.shape[-1]} exceeds {width}")
    output = np.zeros(value.shape[:-1] + (width,), dtype=np.float32)
    output[..., : value.shape[-1]] = value
    return output


def _masked_actor_summary(
    actor: npt.NDArray[np.float32], mask: npt.NDArray[np.bool_]
) -> npt.NDArray[np.float32]:
    if actor.ndim != 4 or mask.shape != actor.shape[:-1]:
        raise ValueError(f"inverse actor/mask shape mismatch: {actor.shape}/{mask.shape}")
    counts = mask.sum(axis=2, keepdims=True)
    denominator = np.maximum(counts, 1)
    mean = (actor * mask[..., None]).sum(axis=2) / denominator
    minimum = np.where(mask[..., None], actor, np.inf).min(axis=2)
    maximum = np.where(mask[..., None], actor, -np.inf).max(axis=2)
    empty = counts[..., 0] == 0
    minimum[empty] = 0.0
    maximum[empty] = 0.0
    return np.concatenate([mean, minimum, maximum], axis=-1).reshape(len(actor), -1)


def pack_inverse_effect(
    effects: Mapping[str, npt.NDArray[object]],
    mode: str,
    *,
    time_permutation: npt.NDArray[np.int64] | None = None,
) -> npt.NDArray[np.float32]:
    """Pack an effect without trajectory/index/score or future absolute ego pose."""

    if mode not in INVERSE_MODES:
        raise ValueError(f"unsupported inverse mode: {mode}")
    ego = np.asarray(effects["ego_effect"], dtype=np.float32)
    map_effect = np.asarray(effects["map_effect"], dtype=np.float32)
    actor = np.asarray(effects["actor_effect"], dtype=np.float32)
    actor_mask = np.asarray(effects["actor_mask"], dtype=bool)
    interaction = np.asarray(effects["interaction_mask"], dtype=bool)
    if ego.ndim != 3 or ego.shape[1:] != (8, 16):
        raise ValueError(f"inverse ego effect must be [K,8,16], got {ego.shape}")
    if map_effect.shape != (len(ego), 8, 8):
        raise ValueError("inverse map effect shape mismatch")
    if actor.shape != (len(ego), 8, 16, 13):
        raise ValueError("inverse actor effect shape mismatch")
    if actor_mask.shape != actor.shape[:-1] or interaction.shape != actor_mask.shape:
        raise ValueError("inverse masks do not align with actor effect")
    if time_permutation is not None:
        permutation = np.asarray(time_permutation, dtype=np.int64)
        if sorted(permutation.tolist()) != list(range(8)):
            raise ValueError("time permutation must be a bijection of eight steps")
        ego = ego[:, permutation]
        map_effect = map_effect[:, permutation]
        actor = actor[:, permutation]
        actor_mask = actor_mask[:, permutation]
        interaction = interaction[:, permutation]
    pieces: list[npt.NDArray[np.float32]] = []
    if mode in {"ego_only", "full_effect"}:
        # Deliberately omit x/y/heading and swept-corner columns.
        pieces.append(ego[..., list(EGO_KINEMATIC_COLUMNS)].reshape(len(ego), -1))
    if mode in {"environment_only", "full_effect"}:
        actor_summary = _masked_actor_summary(actor, actor_mask)
        valid_counts = np.maximum(actor_mask.sum(axis=2), 1)
        pieces.extend(
            [
                map_effect.reshape(len(ego), -1),
                actor_summary,
                actor_mask.mean(axis=2),
                ((interaction & actor_mask).sum(axis=2) / valid_counts).astype(np.float32),
            ]
        )
    packed = np.concatenate(pieces, axis=-1).astype(np.float32)
    if not np.isfinite(packed).all():
        raise ValueError("inverse effect contains NaN/Inf")
    return _pad(np.arcsinh(np.clip(packed, -1.0e4, 1.0e4)), INVERSE_EFFECT_DIM)


def pack_trajectory(trajectory: npt.ArrayLike) -> npt.NDArray[np.float32]:
    value = np.asarray(trajectory, dtype=np.float32)
    if value.ndim != 3 or value.shape[1:] != (8, 3):
        raise ValueError(f"trajectory must be [K,8,3], got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("trajectory contains NaN/Inf")
    return np.arcsinh(np.clip(value.reshape(len(value), -1), -1.0e4, 1.0e4)).astype(
        np.float32
    )


def farthest_point_candidates(trajectory: npt.ArrayLike, count: int = 16) -> npt.NDArray[np.int64]:
    """Geometry-only deterministic FPS; no labels participate."""

    value = np.asarray(trajectory, dtype=np.float32)
    if value.ndim != 3 or value.shape[1:] != (8, 3):
        raise ValueError(f"FPS trajectory must be [K,8,3], got {value.shape}")
    if not 1 <= count <= len(value):
        raise ValueError(f"FPS count {count} is invalid for {len(value)} candidates")
    geometry = np.concatenate(
        [
            value[..., :2].reshape(len(value), -1),
            np.sin(value[..., 2]),
            np.cos(value[..., 2]),
        ],
        axis=-1,
    )
    scale = geometry.std(axis=0)
    scale[scale < 1.0e-6] = 1.0
    geometry = (geometry - geometry.mean(axis=0)) / scale
    centroid_distance = np.square(geometry - geometry.mean(axis=0)).sum(axis=1)
    selected = [int(np.argmax(centroid_distance))]
    minimum_distance = np.square(geometry - geometry[selected[0]]).sum(axis=1)
    minimum_distance[selected[0]] = -np.inf
    while len(selected) < count:
        next_index = int(np.argmax(minimum_distance))
        selected.append(next_index)
        distance = np.square(geometry - geometry[next_index]).sum(axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
        minimum_distance[selected] = -np.inf
    return np.asarray(selected, dtype=np.int64)


def trajectory_delta_descriptors(
    trajectory: npt.ArrayLike,
    first: npt.NDArray[np.int64],
    second: npt.NDArray[np.int64],
    interval_seconds: float = 0.5,
) -> npt.NDArray[np.float32]:
    value = np.asarray(trajectory, dtype=np.float32)
    first_value = value[first]
    second_value = value[second]

    def descriptors(item: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        previous = np.concatenate(
            [np.zeros((len(item), 1, 2), dtype=np.float32), item[:, :-1, :2]], axis=1
        )
        distances = np.linalg.norm(item[:, :, :2] - previous, axis=-1)
        speed = distances / interval_seconds
        return np.stack(
            [
                item[:, -1, 1],
                item[:, -1, 0],
                item[:, -1, 2],
                speed.mean(axis=1),
                distances[:, :4].sum(axis=1),
                distances[:, 4:].sum(axis=1),
            ],
            axis=-1,
        )

    delta = descriptors(first_value) - descriptors(second_value)
    delta[:, 2] = (delta[:, 2] + np.pi) % (2.0 * np.pi) - np.pi
    return (delta / DELTA_SCALES[None]).astype(np.float32)


class InverseProbe(nn.Module):
    """Dual tower plus a delta head sharing the leakage-controlled effect tower."""

    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding dimension must be positive")
        self.effect_tower = nn.Sequential(
            nn.Linear(INVERSE_EFFECT_DIM, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, embedding_dim),
        )
        self.trajectory_tower = nn.Sequential(
            nn.Linear(TRAJECTORY_DIM, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, embedding_dim),
        )
        self.delta_head = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.GELU(),
            nn.Linear(256, DELTA_DIM),
        )

    def encode_effect(self, effect: Tensor) -> Tensor:
        if effect.shape[-1] != INVERSE_EFFECT_DIM:
            raise ValueError(f"effect tower expected 768D, got {tuple(effect.shape)}")
        return F.normalize(self.effect_tower(effect.float()), dim=-1)

    def encode_trajectory(self, trajectory: Tensor) -> Tensor:
        if trajectory.shape[-1] != TRAJECTORY_DIM:
            raise ValueError(f"trajectory tower expected 24D, got {tuple(trajectory.shape)}")
        return F.normalize(self.trajectory_tower(trajectory.float()), dim=-1)

    def retrieval_logits(self, effect: Tensor, trajectory: Tensor, temperature: float) -> Tensor:
        if temperature <= 0:
            raise ValueError("inverse temperature must be positive")
        effect_embedding = self.encode_effect(effect)
        trajectory_embedding = self.encode_trajectory(trajectory)
        return effect_embedding @ trajectory_embedding.transpose(-1, -2) / temperature

    def predict_delta(self, first_effect: Tensor, second_effect: Tensor) -> Tensor:
        return self.delta_head(
            self.encode_effect(first_effect) - self.encode_effect(second_effect)
        )


@dataclass(frozen=True)
class InverseLoss:
    total: Tensor
    retrieval: Tensor
    delta: Tensor


def inverse_training_loss(
    model: InverseProbe,
    effects: Tensor,
    trajectories: Tensor,
    pair_first: Tensor,
    pair_second: Tensor,
    delta_targets: Tensor,
    temperature: float,
    delta_weight: float = 1.0,
) -> InverseLoss:
    if effects.ndim != 3 or trajectories.ndim != 3 or effects.shape[:2] != trajectories.shape[:2]:
        raise ValueError("inverse training requires matching [scene,K,*] towers")
    logits = model.retrieval_logits(effects, trajectories, temperature)
    batch, candidates = effects.shape[:2]
    labels = torch.arange(candidates, device=effects.device)[None].expand(batch, -1)
    retrieval = (
        F.cross_entropy(logits.reshape(batch * candidates, candidates), labels.reshape(-1))
        + F.cross_entropy(
            logits.transpose(1, 2).reshape(batch * candidates, candidates),
            labels.reshape(-1),
        )
    ) / 2.0
    rows = torch.arange(batch, device=effects.device)[:, None]
    predicted_delta = model.predict_delta(
        effects[rows, pair_first], effects[rows, pair_second]
    )
    if predicted_delta.shape != delta_targets.shape:
        raise ValueError("inverse delta prediction/target shapes differ")
    delta = F.smooth_l1_loss(predicted_delta, delta_targets)
    total = retrieval + delta_weight * delta
    if not torch.isfinite(total):
        raise FloatingPointError("inverse loss is NaN/Inf")
    return InverseLoss(total=total, retrieval=retrieval, delta=delta)
