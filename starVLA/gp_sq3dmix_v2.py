"""Deterministic contracts shared by GP-SQ3D-Mix Stage-A-v2.

This module intentionally has no accelerator or NAVSIM runtime dependency.  It
is imported by the CPU asset builders, the dataset, the framework, and tests so
that the intervention semantics cannot silently diverge between those paths.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


DESCRIPTOR_DIMENSION = 128
DESCRIPTOR_PROJECTION_SEED = 20260824
DESCRIPTOR_PROJECTION_SHAPE = (2048, DESCRIPTOR_DIMENSION)
SPATIAL_SLOTS_PER_VIEW = 60
SPATIAL_VIEW_COUNT = 3
VIEW_ORDER = ("cam_f0", "cam_l0", "cam_r0")
COMMAND_LABELS = ("turn left", "keep straight", "turn right", "unknown")

HARD_NEGATIVE_CONTRACT = {
    "schema_version": 2,
    "method": "same-command_moderate-action_geometry-far",
    "action_top_k": 256,
    "level_0_rank_range": [9, 128],
    "level_1_rank_range": [5, 192],
    "level_0_reuse_capacity": 16,
    "level_1_reuse_capacity": 16,
    "level_2_reuse_capacity": 32,
    "minimum_temporal_distance_seconds": 5.0,
    "same_log_allowed": False,
    "same_episode_allowed": False,
    "different_command_allowed": False,
    "random_donor_allowed": False,
    "batch_local_donor_allowed": False,
    "action_shape": [8, 4],
    "action_normalization": "training-channel-mean-std",
    "geometry_metric": "cosine_distance",
    "tie_break": "donor_token_lexical",
}


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def token_order_sha256(tokens: Sequence[str]) -> str:
    return sha256_bytes(canonical_json_bytes(list(tokens)))


def hard_negative_contract_sha256() -> str:
    return sha256_bytes(canonical_json_bytes(HARD_NEGATIVE_CONTRACT))


def descriptor_projection(
    seed: int = DESCRIPTOR_PROJECTION_SEED,
    shape: tuple[int, int] = DESCRIPTOR_PROJECTION_SHAPE,
) -> Tensor:
    """Return the fixed, untrained Gaussian projection on CPU."""

    if tuple(shape) != DESCRIPTOR_PROJECTION_SHAPE:
        raise ValueError(
            f"descriptor projection must have shape {DESCRIPTOR_PROJECTION_SHAPE}"
        )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    projection = torch.randn(shape, generator=generator, dtype=torch.float32)
    return projection.div_(math.sqrt(shape[1])).contiguous()


def tensor_sha256(tensor: Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return sha256_bytes(value.numpy().tobytes(order="C"))


def pooled_scene_descriptor(
    pooled_features: Tensor, projection: Tensor | None = None
) -> Tensor:
    """Compute one deterministic normalized 128-D scene descriptor.

    ``pooled_features`` may be ``[180,2048]`` or ``[B,180,2048]``.  LayerNorm
    is applied independently to every slot without affine parameters, exactly
    matching the Stage-A-v2 asset contract.
    """

    if pooled_features.ndim not in (2, 3) or pooled_features.shape[-2:] != (
        180,
        2048,
    ):
        raise ValueError("pooled_features must be [180,2048] or [B,180,2048]")
    projection = descriptor_projection() if projection is None else projection
    if projection.shape != DESCRIPTOR_PROJECTION_SHAPE:
        raise ValueError("descriptor projection has the wrong shape")
    normalized = F.layer_norm(pooled_features.float(), (2048,))
    descriptor = normalized.mean(dim=-2) @ projection.to(normalized.device)
    return F.normalize(descriptor, dim=-1, eps=1e-12)


def spatial_derangement_indices(
    seed: int,
    scene_token: str,
    view_index: int,
    slots_per_view: int = SPATIAL_SLOTS_PER_VIEW,
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    """Build a topology-independent cyclic derangement for one camera view."""

    if not 0 <= int(view_index) < SPATIAL_VIEW_COUNT:
        raise ValueError("view_index must be 0, 1, or 2")
    if int(slots_per_view) < 2:
        raise ValueError("a derangement requires at least two slots")
    payload = canonical_json_bytes(
        [int(seed), str(scene_token), int(view_index)]
    )
    shift = 1 + int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (
        int(slots_per_view) - 1
    )
    base = torch.arange(int(slots_per_view), device=device, dtype=torch.long)
    permutation = (base + shift) % int(slots_per_view)
    if torch.any(permutation == base):
        raise RuntimeError("deterministic spatial permutation contains a fixed point")
    return permutation


def spatial_shuffle_pooled_geometry(
    pooled: Mapping[str, Tensor], tokens: Sequence[str], seed: int
) -> tuple[dict[str, Tensor], Tensor]:
    """Shuffle only features within each view while preserving all metadata."""

    required = ("features", "view_ids", "uv_coords", "ray_features")
    if any(key not in pooled for key in required):
        raise KeyError("pooled geometry is missing required fields")
    features = pooled["features"]
    if features.ndim != 3 or features.shape[1] != 180:
        raise ValueError("pooled features must be [B,180,D]")
    if len(tokens) != features.shape[0]:
        raise ValueError("token count must match pooled batch size")
    shuffled = features.clone()
    fixed_points = torch.zeros((), dtype=torch.long, device=features.device)
    for batch_index, token in enumerate(tokens):
        for view_index in range(SPATIAL_VIEW_COUNT):
            start = view_index * SPATIAL_SLOTS_PER_VIEW
            stop = start + SPATIAL_SLOTS_PER_VIEW
            permutation = spatial_derangement_indices(
                seed, token, view_index, device=features.device
            )
            local = torch.arange(
                SPATIAL_SLOTS_PER_VIEW, device=features.device, dtype=torch.long
            )
            fixed_points += (permutation == local).sum()
            shuffled[batch_index, start:stop] = features[
                batch_index, start:stop
            ].index_select(0, permutation)
    result = {key: pooled[key] for key in required}
    result["features"] = shuffled
    return result, fixed_points


def wrap_to_pi_numpy(value: np.ndarray) -> np.ndarray:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def navsim_training_action(
    raw: Mapping[str, object], *, act_norm: bool = True
) -> np.ndarray:
    """Extract the exact ver_1225=1 NAVSIM 8x4 training target from a raw pkl."""

    try:
        poses = np.asarray(raw["glo_status"]["global_poses"], dtype=np.float64)
    except (KeyError, TypeError) as error:
        raise ValueError("processed sample has no glo_status.global_poses") from error
    if poses.ndim != 2 or poses.shape[0] < 12 or poses.shape[1] < 3:
        raise ValueError("global_poses must contain at least 12 SE2 poses")
    origin = poses[3, :3]
    delta = poses[:12, :2] - origin[:2]
    cosine, sine = np.cos(origin[2]), np.sin(origin[2])
    relative_xy = np.empty_like(delta)
    relative_xy[:, 0] = cosine * delta[:, 0] + sine * delta[:, 1]
    relative_xy[:, 1] = -sine * delta[:, 0] + cosine * delta[:, 1]
    relative_heading = wrap_to_pi_numpy(poses[:12, 2] - origin[2])
    dxy = relative_xy[4:12].astype(np.float32, copy=True)
    if act_norm:
        dxy[:, 0] = (dxy[:, 0] - 10.172484) / 8.805105
        dxy[:, 1] = (dxy[:, 1] - 0.360762) / 2.277741
    else:
        dxy[:, 0] /= 4.5912
    heading = wrap_to_pi_numpy(relative_heading[4:12] - relative_heading[3])
    heading_sc = np.stack((np.sin(heading), np.cos(heading)), axis=-1).astype(
        np.float32
    )
    action = np.concatenate((dxy, heading_sc), axis=-1).astype(np.float32)
    if action.shape != (8, 4) or not np.isfinite(action).all():
        raise ValueError("processed sample produced an invalid 8x4 action target")
    return action


def navsim_navigation_command(raw: Mapping[str, object]) -> str:
    try:
        command = np.asarray(
            raw["glo_status"]["commands"][3], dtype=np.float64
        ).reshape(-1)
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("processed sample has no timestep-3 navigation command") from error
    if command.size != len(COMMAND_LABELS) or not np.isfinite(command).all():
        raise ValueError("navigation command must be a finite four-way vector")
    return COMMAND_LABELS[int(command.argmax())]


def navsim_log_id(raw: Mapping[str, object]) -> str:
    """Read log identity from processed image metadata, never from scene token."""

    try:
        image_path = Path(raw["glo_images"]["cam_f0"]["image_paths"][3])
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("processed sample has no cam_f0 timestep-3 image path") from error
    if image_path.parent.name.lower() != "cam_f0":
        raise ValueError(
            "cannot identify log: cam_f0 image path does not contain a CAM_F0 parent"
        )
    log_id = image_path.parent.parent.name
    if not log_id:
        raise ValueError("processed sample yielded an empty log id")
    return log_id
