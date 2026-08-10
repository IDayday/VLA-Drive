#!/usr/bin/env python3
"""Generate an offline, manifest-bound V-JEPA 2.1 dynamics cache.

The tool launches one independent process per visible accelerator and never
initializes a training process group.  V-JEPA is lazy-loaded from an explicit
local repository/checkpoint; network downloads are deliberately unsupported.
Each entry stores future-aligned per-view features as ``[H,V,C,Ht,Wt]``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from starVLA.dataloader.field2plan_cache import (
    DynamicsCacheReader,
    atomic_write_json,
    atomic_write_npz,
    hash_tokens,
    sha256_file,
)
from starVLA.model.modules.field2plan.dynamics_teachers import (
    OfficialVJEPA2Adapter,
    seeded_orthogonal_projection,
)


DEFAULT_DYNAMICS_VIEWS = ("cam_f0", "cam_l0", "cam_r0")
DEFAULT_INPUT_FRAMES = tuple(range(12))
DEFAULT_HISTORY_FRAMES = (0, 1, 2, 3)
DEFAULT_FUTURE_FRAMES = tuple(range(4, 12))


@dataclass(frozen=True)
class VJEPACacheInputs:
    """Ordered NAVSIM clip paths and sizes.

    ``image_paths`` is nested as ``[V][T]`` and ``source_image_hw`` is
    ``[T,V,2]`` in height/width order.
    """

    token: str
    view_names: Tuple[str, ...]
    frame_indices: Tuple[int, ...]
    image_paths: Tuple[Tuple[Path, ...], ...]
    source_image_hw: np.ndarray

    def load_rgb(self) -> np.ndarray:
        """Read RGB pixels as uint8 ``[V,T,H,W,3]`` without fallback."""

        videos = []
        for view_paths in self.image_paths:
            frames = []
            for path in view_paths:
                try:
                    with Image.open(path) as image:
                        frames.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
                except (OSError, UnidentifiedImageError) as error:
                    raise ValueError(f"invalid NAVSIM image: {path}") from error
            try:
                videos.append(np.stack(frames, axis=0))
            except ValueError as error:
                raise ValueError(
                    f"image dimensions changed within clip token={self.token}"
                ) from error
        try:
            video = np.stack(videos, axis=0)
        except ValueError as error:
            raise ValueError(
                f"image dimensions differ across views token={self.token}"
            ) from error
        if video.ndim != 5 or video.shape[-1] != 3:
            raise ValueError("loaded V-JEPA video must have shape [V,T,H,W,3]")
        return video


def _git_commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _load_tokens(datalist_path: Path, max_samples: int = 0) -> list[str]:
    try:
        payload = json.loads(datalist_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid dynamics datalist JSON: {datalist_path}") from error
    if not isinstance(payload, list) or not all(
        isinstance(token, str) and token for token in payload
    ):
        raise ValueError("dynamics datalist must be a list of non-empty tokens")
    tokens = payload[:max_samples] if max_samples > 0 else payload
    if not tokens:
        raise ValueError("dynamics datalist selected zero tokens")
    if len(tokens) != len(set(tokens)):
        raise ValueError("dynamics datalist contains duplicate tokens")
    return tokens


def _resolve_navsim_image_path(
    embedded_path: str | os.PathLike,
    *,
    runtime_raw_root: Path,
    trainval_sensor_root: Path | None,
) -> Path:
    path = Path(embedded_path)
    if path.is_file():
        return path
    marker = f"{os.sep}navsim_dataset_raw{os.sep}"
    encoded = os.fspath(path)
    candidates: list[Path] = []
    if marker in encoded:
        relative = encoded.split(marker, 1)[1]
        trainval_prefix = f"sensor_blobs{os.sep}trainval{os.sep}"
        if trainval_sensor_root is not None and relative.startswith(trainval_prefix):
            candidates.append(
                trainval_sensor_root / relative[len(trainval_prefix) :]
            )
        candidates.append(runtime_raw_root / relative)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "NAVSIM image not found: "
        f"embedded={path}; candidates={[str(value) for value in candidates]}"
    )


def load_navsim_vjepa_inputs(
    *,
    token: str,
    meta_path: Path,
    view_names: Sequence[str],
    input_frame_indices: Sequence[int],
    runtime_raw_root: Path,
    trainval_sensor_root: Path | None,
) -> VJEPACacheInputs:
    """Resolve one ordered multi-view clip from NAVSIM metadata."""

    if not meta_path.is_file():
        raise FileNotFoundError(f"NAVSIM metadata not found: {meta_path}")
    try:
        with meta_path.open("rb") as stream:
            metadata = pickle.load(stream)
    except (OSError, pickle.UnpicklingError, EOFError) as error:
        raise ValueError(f"corrupt NAVSIM metadata: {meta_path}") from error
    if not isinstance(metadata, dict):
        raise ValueError(f"NAVSIM metadata must be a mapping: {meta_path}")
    views = tuple(str(value) for value in view_names)
    frames = tuple(int(value) for value in input_frame_indices)
    if not views or len(set(views)) != len(views):
        raise ValueError("V-JEPA view_names must be non-empty and unique")
    if not frames or any(index < 0 for index in frames):
        raise ValueError("V-JEPA input frame indices must be non-negative")
    if any(right <= left for left, right in zip(frames, frames[1:])):
        raise ValueError("V-JEPA input frame indices must increase")

    paths: list[Tuple[Path, ...]] = []
    sizes_by_view: list[list[tuple[int, int]]] = []
    try:
        cameras = metadata["glo_images"]
        for view in views:
            image_paths = cameras[view]["image_paths"]
            view_paths = []
            view_sizes = []
            for frame in frames:
                path = _resolve_navsim_image_path(
                    image_paths[frame],
                    runtime_raw_root=runtime_raw_root,
                    trainval_sensor_root=trainval_sensor_root,
                )
                try:
                    with Image.open(path) as image:
                        width, height = image.size
                except (OSError, UnidentifiedImageError) as error:
                    raise ValueError(f"invalid NAVSIM image: {path}") from error
                if min(height, width) <= 0:
                    raise ValueError(f"invalid NAVSIM image size: {path}")
                view_paths.append(path)
                view_sizes.append((height, width))
            paths.append(tuple(view_paths))
            sizes_by_view.append(view_sizes)
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(
            f"invalid NAVSIM camera contract for token={token!r}"
        ) from error
    source_hw = np.asarray(sizes_by_view, dtype=np.int64).transpose(1, 0, 2)
    return VJEPACacheInputs(
        token=str(token),
        view_names=views,
        frame_indices=frames,
        image_paths=tuple(paths),
        source_image_hw=source_hw,
    )


def resample_vjepa_tokens(
    tokens: torch.Tensor,
    *,
    token_center_indices: torch.Tensor,
    target_frame_indices: torch.Tensor,
) -> torch.Tensor:
    """Linearly map ``[V,Tt,Ht,Wt,C]`` tubelets to ``[H,V,C,Ht,Wt]``."""

    if tokens.ndim != 5:
        raise ValueError("V-JEPA tokens must have shape [V,Tt,Ht,Wt,C]")
    centers = torch.as_tensor(
        token_center_indices, device=tokens.device, dtype=torch.float32
    )
    targets = torch.as_tensor(
        target_frame_indices, device=tokens.device, dtype=torch.float32
    )
    if centers.ndim != 1 or centers.numel() != tokens.shape[1]:
        raise ValueError("token_center_indices must have shape [Tt]")
    if targets.ndim != 1 or targets.numel() < 1:
        raise ValueError("target_frame_indices must have shape [H]")
    if centers.numel() < 2 or not torch.all(centers[1:] > centers[:-1]):
        raise ValueError("V-JEPA token centers must strictly increase")
    right = torch.searchsorted(centers, targets, right=False)
    right = right.clamp(1, centers.numel() - 1)
    left = right - 1
    below = targets <= centers[0]
    above = targets >= centers[-1]
    left = torch.where(below, torch.zeros_like(left), left)
    right = torch.where(below, torch.zeros_like(right), right)
    last = torch.full_like(left, centers.numel() - 1)
    left = torch.where(above, last, left)
    right = torch.where(above, last, right)
    denominator = (centers[right] - centers[left]).clamp_min(1e-6)
    alpha = torch.where(
        left == right,
        torch.zeros_like(targets),
        (targets - centers[left]) / denominator,
    )
    left_values = tokens.index_select(1, left).permute(1, 0, 2, 3, 4)
    right_values = tokens.index_select(1, right).permute(1, 0, 2, 3, 4)
    interpolated = left_values + alpha[:, None, None, None, None].to(
        dtype=tokens.dtype
    ) * (right_values - left_values)
    return interpolated.permute(0, 1, 4, 2, 3).contiguous()


def project_and_resize_features(
    features: torch.Tensor,
    projection: torch.Tensor,
    output_hw: Sequence[int],
) -> torch.Tensor:
    """Project ``[H,V,C,Ht,Wt]`` to normalized ``[H,V,Co,Ho,Wo]``."""

    if features.ndim != 5 or projection.ndim != 2:
        raise ValueError("features/projection ranks must be 5 and 2")
    if features.shape[2] != projection.shape[0]:
        raise ValueError("V-JEPA feature and projection channel dimensions differ")
    output_hw = tuple(int(value) for value in output_hw)
    if len(output_hw) != 2 or min(output_hw) <= 0:
        raise ValueError("output_hw must contain two positive dimensions")
    value = features.float().permute(0, 1, 3, 4, 2)
    value = value @ projection.to(device=value.device, dtype=torch.float32)
    value = F.normalize(value, dim=-1, eps=1e-6).permute(0, 1, 4, 2, 3)
    horizon, views, channels, height, width = value.shape
    if (height, width) != output_hw:
        value = F.interpolate(
            value.reshape(horizon * views, channels, height, width),
            size=output_hw,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).reshape(horizon, views, channels, *output_hw)
        value = F.normalize(value, dim=2, eps=1e-6)
    return value


def _source_index_sha256(
    tokens: Iterable[str], metadata_checksums: Dict[str, str]
) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        encoded = str(token).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big", signed=False))
        digest.update(encoded)
        digest.update(bytes.fromhex(metadata_checksums[str(token)]))
    return digest.hexdigest()


def _preprocessing_payload(input_image_hw: Sequence[int]) -> dict:
    height, width = (int(value) for value in input_image_hw)
    if height != width or height <= 0:
        raise ValueError("V-JEPA input_image_hw must be a positive square")
    return {
        "input_image_hw": [height, width],
        "resize_policy": "center_crop_square_then_bilinear",
        "interpolation": "bilinear_antialias_align_corners_false",
        "color_space": "RGB",
        "value_range": "0_1",
        "normalization_mean": [0.485, 0.456, 0.406],
        "normalization_std": [0.229, 0.224, 0.225],
    }


def build_dynamics_manifest(
    *,
    datalist_path: Path,
    split: str,
    tokens: Sequence[str],
    view_names: Sequence[str],
    feature_channels: int,
    output_hw: Sequence[int],
    git_commit: str,
    vjepa_repo: Path,
    vjepa_repo_commit: str,
    checkpoint: Path,
    checkpoint_revision: str,
    metadata_checksums: Dict[str, str],
    model_variant: str,
    projection_seed: int,
    current_frame_index: int,
    history_frame_indices: Sequence[int],
    future_frame_indices: Sequence[int],
    frame_interval_s: float,
    input_image_hw: Sequence[int],
) -> dict:
    """Build a content-bound V-JEPA dynamics manifest."""

    token_list = [str(token) for token in tokens]
    views = tuple(str(value) for value in view_names)
    output_hw = tuple(int(value) for value in output_hw)
    history = tuple(int(value) for value in history_frame_indices)
    future = tuple(int(value) for value in future_frame_indices)
    if not token_list or set(metadata_checksums) != set(token_list):
        raise ValueError("metadata checksums must cover the selected token set")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"V-JEPA checkpoint not found: {checkpoint}")
    if not vjepa_repo.is_dir():
        raise FileNotFoundError(f"V-JEPA repository not found: {vjepa_repo}")
    if len(vjepa_repo_commit) != 40:
        raise ValueError("V-JEPA repository commit must be a full SHA1")
    if min(int(feature_channels), *output_hw) <= 0:
        raise ValueError("dynamics feature dimensions must be positive")
    if not history or any(index > current_frame_index for index in history):
        raise ValueError("history_frame_indices must be at or before current")
    if not future or any(index <= current_frame_index for index in future):
        raise ValueError("future_frame_indices must be after current")
    if any(right <= left for left, right in zip(future, future[1:])):
        raise ValueError("future_frame_indices must strictly increase")
    if frame_interval_s <= 0:
        raise ValueError("frame_interval_s must be positive")
    preprocess = _preprocessing_payload(input_image_hw)
    preprocess["sha256"] = hashlib.sha256(
        json.dumps(preprocess, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    horizon = len(future)
    height, width = output_hw
    input_frames = tuple(range(12))
    token_centers = [float(index) + 0.5 for index in range(0, 12, 2)]
    return {
        "schema_version": 1,
        "cache_type": "dynamics_teacher",
        "status": "complete",
        "teacher": {
            "name": OfficialVJEPA2Adapter.name,
            "version": "vjepa2_1_vitl16_384_official_local_v1",
            "model_variant": str(model_variant),
            "selected_layers": [-1],
            "repo": str(vjepa_repo.resolve()),
            "repo_commit": str(vjepa_repo_commit),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_revision": str(checkpoint_revision),
            "checkpoint_sha256": sha256_file(checkpoint),
            "source_index_sha256": _source_index_sha256(
                token_list, metadata_checksums
            ),
            "runtime_compatibility": [
                "rope_position_cast_to_qk_dtype_for_torch2.4_strict_sdpa"
            ],
        },
        "generator": {
            "git_commit": str(git_commit),
            "tool": "tools/field2plan/cache_dynamics_vjepa.py",
        },
        "splits": {
            str(split): {
                "entry_count": len(token_list),
                "tokens_sha256": hash_tokens(token_list),
                "datalist_sha256": sha256_file(datalist_path),
            }
        },
        "tensor_schema": {
            "view_names": list(views),
            "features": {
                "dtype": "float16",
                "shape": [horizon, len(views), int(feature_channels), height, width],
            },
            "confidence": {
                "dtype": "float32",
                "shape": [horizon, len(views), height, width],
            },
            "valid_mask": {
                "dtype": "bool",
                "shape": [horizon, len(views), height, width],
            },
            "frame_indices": {"dtype": "int64", "shape": [horizon]},
            "frame_times_s": {"dtype": "float32", "shape": [horizon]},
            "source_image_hw": {
                "dtype": "int64",
                "shape": [horizon, len(views), 2],
            },
            "feature_hw": {
                "dtype": "int64",
                "shape": [horizon, len(views), 2],
            },
        },
        "temporal": {
            "input_frame_indices": list(input_frames),
            "current_frame_index": int(current_frame_index),
            "history_frame_indices": list(history),
            "future_frame_indices": list(future),
            "frame_interval_s": float(frame_interval_s),
            "teacher_temporal_stride": 2,
            "teacher_token_center_indices": token_centers,
            "resampling": "linear_with_endpoint_clamp",
            "ego_motion_alignment": "dataset_current_ego_projection",
        },
        "features": {
            "spatial_layout": "per_view_patch_grid",
            "normalization": "l2",
            "confidence_source": "finite_token_validity",
            "projection": {
                "algorithm": "seeded_orthogonal",
                "seed": int(projection_seed),
                "input_dim": 1024,
                "output_dim": int(feature_channels),
            },
        },
        "preprocessing": preprocess,
    }


def existing_entry_is_valid(
    path: Path,
    *,
    token: str,
    expected_feature_shape: tuple[int, int, int, int, int],
    cache_fingerprint: str,
) -> bool:
    """Return whether a resumable entry matches this exact generation run."""

    if not path.is_file():
        return False
    horizon, views, _, height, width = expected_feature_shape
    try:
        with np.load(path, allow_pickle=False) as payload:
            required = {
                "token",
                "cache_fingerprint",
                "features",
                "confidence",
                "valid_mask",
                "frame_indices",
                "frame_times_s",
                "source_image_hw",
                "feature_hw",
            }
            return (
                required.issubset(payload.files)
                and str(payload["token"].item()) == token
                and str(payload["cache_fingerprint"].item()) == cache_fingerprint
                and payload["features"].shape == expected_feature_shape
                and payload["features"].dtype == np.float16
                and payload["confidence"].shape == (horizon, views, height, width)
                and payload["confidence"].dtype == np.float32
                and payload["valid_mask"].shape == (horizon, views, height, width)
                and payload["valid_mask"].dtype == np.bool_
            )
    except (OSError, ValueError):
        return False


def _cache_fingerprint(args: argparse.Namespace, repo_commit: str) -> str:
    stat = args.checkpoint.stat()
    payload = {
        "datalist_sha256": sha256_file(args.datalist),
        "checkpoint_revision": args.checkpoint_revision,
        "checkpoint_expected_sha256": args.checkpoint_sha256,
        "checkpoint_size": stat.st_size,
        "repo_commit": repo_commit,
        "model_variant": args.model_variant,
        "input_frames": list(args.input_frame_indices),
        "future_frames": list(args.future_frame_indices),
        "input_image_hw": [args.image_size, args.image_size],
        "feature_channels": args.feature_channels,
        "output_hw": [args.output_height, args.output_width],
        "projection_seed": args.projection_seed,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_dynamics_cache(
    *,
    cache_root: Path,
    split: str,
    tokens: Sequence[str],
    datalist: Path,
    cache_fingerprint: str,
    rank: int,
    world_size: int,
) -> dict:
    """Validate one deterministic shard using the runtime reader."""

    reader = DynamicsCacheReader(str(cache_root), split)
    reader.validate_dataset_binding(tokens, str(datalist))
    checked = 0
    for token in tokens[rank::world_size]:
        path = cache_root / split / f"{token}.npz"
        expected_shape = reader.array_schema["features"][1]
        if not existing_entry_is_valid(
            path,
            token=token,
            expected_feature_shape=expected_shape,
            cache_fingerprint=cache_fingerprint,
        ):
            raise ValueError(f"invalid dynamics cache entry: {path}")
        reader.load(token)
        checked += 1
    return {"rank": rank, "world_size": world_size, "validated": checked}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datalist", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--meta-root", type=Path, required=True)
    parser.add_argument("--runtime-raw-root", type=Path, required=True)
    parser.add_argument("--trainval-sensor-root", type=Path)
    parser.add_argument("--vjepa-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument(
        "--model-variant", default="vjepa2_1_vit_large_384"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--feature-channels", type=int, default=96)
    parser.add_argument("--output-height", type=int, default=16)
    parser.add_argument("--output-width", type=int, default=16)
    parser.add_argument("--projection-seed", type=int, default=20260809)
    parser.add_argument("--current-frame-index", type=int, default=3)
    parser.add_argument(
        "--input-frame-indices", type=int, nargs="+", default=list(range(12))
    )
    parser.add_argument(
        "--history-frame-indices", type=int, nargs="+", default=[0, 1, 2, 3]
    )
    parser.add_argument(
        "--future-frame-indices", type=int, nargs="+", default=list(range(4, 12))
    )
    parser.add_argument("--frame-interval-s", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--view-batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--coordination-timeout-s", type=int, default=3600)
    parser.add_argument("--compiler-cache-root", type=Path, default=Path("/tmp/field2plan-vjepa"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid rank/world-size environment")
    if min(
        args.image_size,
        args.feature_channels,
        args.output_height,
        args.output_width,
        args.view_batch_size,
    ) <= 0:
        raise ValueError("V-JEPA cache dimensions must be positive")
    if tuple(args.input_frame_indices) != tuple(range(12)):
        raise ValueError(
            "the pinned Phase-3 V-JEPA cache requires input frames [0,...,11]"
        )
    if tuple(args.history_frame_indices) != DEFAULT_HISTORY_FRAMES:
        raise ValueError("Phase-3 history frames must be [0,1,2,3]")
    if tuple(args.future_frame_indices) != DEFAULT_FUTURE_FRAMES:
        raise ValueError("Phase-3 future frames must be [4,...,11]")
    if len(args.checkpoint_sha256) != 64:
        raise ValueError("checkpoint-sha256 must be a full SHA256")
    repo_commit = _git_commit(args.vjepa_repo)
    tokens = _load_tokens(args.datalist, args.max_samples)
    fingerprint = _cache_fingerprint(args, repo_commit)

    if args.validate_only:
        result = validate_dynamics_cache(
            cache_root=args.output_dir,
            split=args.split,
            tokens=tokens,
            datalist=args.datalist,
            cache_fingerprint=fingerprint,
            rank=rank,
            world_size=world_size,
        )
        print(json.dumps(result, sort_keys=True), flush=True)
        return

    if args.device == "cuda":
        device = f"cuda:{local_rank}" if world_size > 1 else "cuda"
    else:
        device = args.device
    rank_cache = args.compiler_cache_root / f"rank-{rank:02d}"
    rank_cache.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(rank_cache / "triton")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(rank_cache / "inductor")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    adapter = OfficialVJEPA2Adapter(
        local_repo=args.vjepa_repo,
        checkpoint=args.checkpoint,
        model_variant=args.model_variant,
        device=device,
        dtype=dtype,
        num_frames=len(args.input_frame_indices),
        image_size=args.image_size,
    )
    projection = seeded_orthogonal_projection(
        1024,
        args.feature_channels,
        seed=args.projection_seed,
        device=device,
    )
    token_centers = torch.arange(
        0.5,
        len(args.input_frame_indices),
        2.0,
        device=device,
        dtype=torch.float32,
    )
    future_frames = torch.tensor(
        args.future_frame_indices, device=device, dtype=torch.float32
    )
    future_positions = [
        args.input_frame_indices.index(index) for index in args.future_frame_indices
    ]
    output_split = args.output_dir / args.split
    output_split.mkdir(parents=True, exist_ok=True)
    expected_shape = (
        len(args.future_frame_indices),
        len(DEFAULT_DYNAMICS_VIEWS),
        args.feature_channels,
        args.output_height,
        args.output_width,
    )
    shard_tokens = tokens[rank::world_size]
    metadata_checksums: Dict[str, str] = {}
    generated = resumed = 0
    for local_index, token in enumerate(shard_tokens, start=1):
        meta_path = args.meta_root / f"{token}.pkl"
        metadata_checksums[token] = sha256_file(meta_path)
        output_path = output_split / f"{token}.npz"
        if not args.overwrite and existing_entry_is_valid(
            output_path,
            token=token,
            expected_feature_shape=expected_shape,
            cache_fingerprint=fingerprint,
        ):
            resumed += 1
        else:
            inputs = load_navsim_vjepa_inputs(
                token=token,
                meta_path=meta_path,
                view_names=DEFAULT_DYNAMICS_VIEWS,
                input_frame_indices=args.input_frame_indices,
                runtime_raw_root=args.runtime_raw_root,
                trainval_sensor_root=args.trainval_sensor_root,
            )
            preprocessed = adapter.preprocess_video(inputs.load_rgb())
            encoded_chunks = []
            for start in range(0, preprocessed.shape[0], args.view_batch_size):
                encoded_chunks.append(
                    adapter.encode_video(
                        preprocessed[start : start + args.view_batch_size]
                    )
                )
            encoded = torch.cat(encoded_chunks, dim=0)
            aligned = resample_vjepa_tokens(
                encoded,
                token_center_indices=token_centers,
                target_frame_indices=future_frames,
            )
            features = project_and_resize_features(
                aligned,
                projection,
                (args.output_height, args.output_width),
            )
            finite = torch.isfinite(features).all(dim=2)
            confidence = finite.to(torch.float32)
            safe_features = torch.where(
                finite[:, :, None], features, torch.zeros_like(features)
            )
            source_hw = inputs.source_image_hw[future_positions]
            feature_hw = np.broadcast_to(
                np.asarray([args.output_height, args.output_width], dtype=np.int64),
                (len(args.future_frame_indices), len(DEFAULT_DYNAMICS_VIEWS), 2),
            ).copy()
            frame_indices = np.asarray(args.future_frame_indices, dtype=np.int64)
            frame_times = (
                frame_indices.astype(np.float32) - np.float32(args.current_frame_index)
            ) * np.float32(args.frame_interval_s)
            atomic_write_npz(
                output_path,
                token=np.asarray(token),
                cache_fingerprint=np.asarray(fingerprint),
                features=safe_features.cpu().numpy().astype(np.float16),
                confidence=confidence.cpu().numpy().astype(np.float32),
                valid_mask=finite.cpu().numpy().astype(np.bool_),
                frame_indices=frame_indices,
                frame_times_s=frame_times,
                source_image_hw=source_hw.astype(np.int64),
                feature_hw=feature_hw,
            )
            generated += 1
        if local_index % 20 == 0 or local_index == len(shard_tokens):
            print(
                f"[vjepa-cache rank={rank}] {local_index}/{len(shard_tokens)} "
                f"generated={generated} resumed={resumed}",
                flush=True,
            )

    shard_dir = args.output_dir / ".shards"
    shard_path = shard_dir / f"{fingerprint}-rank{rank:05d}.json"
    atomic_write_json(
        shard_path,
        {
            "fingerprint": fingerprint,
            "rank": rank,
            "world_size": world_size,
            "generated": generated,
            "resumed": resumed,
            "metadata_checksums": metadata_checksums,
        },
    )
    if rank != 0:
        return
    shard_paths = [
        shard_dir / f"{fingerprint}-rank{index:05d}.json"
        for index in range(world_size)
    ]
    deadline = time.monotonic() + args.coordination_timeout_s
    while not all(path.is_file() for path in shard_paths):
        if time.monotonic() >= deadline:
            missing = [str(path) for path in shard_paths if not path.is_file()]
            raise TimeoutError(f"timed out waiting for V-JEPA cache shards: {missing}")
        time.sleep(5)
    combined: Dict[str, str] = {}
    total_generated = total_resumed = 0
    for path in shard_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("fingerprint") != fingerprint:
            raise ValueError(f"V-JEPA shard fingerprint mismatch: {path}")
        combined.update(payload["metadata_checksums"])
        total_generated += int(payload["generated"])
        total_resumed += int(payload["resumed"])
    if set(combined) != set(tokens):
        raise ValueError("V-JEPA shard summaries do not cover selected tokens")
    actual_checkpoint_sha256 = sha256_file(args.checkpoint)
    if actual_checkpoint_sha256 != args.checkpoint_sha256:
        raise ValueError(
            "V-JEPA checkpoint checksum mismatch: "
            f"expected={args.checkpoint_sha256} actual={actual_checkpoint_sha256}"
        )
    manifest = build_dynamics_manifest(
        datalist_path=args.datalist,
        split=args.split,
        tokens=tokens,
        view_names=DEFAULT_DYNAMICS_VIEWS,
        feature_channels=args.feature_channels,
        output_hw=(args.output_height, args.output_width),
        git_commit=_git_commit(PROJECT_ROOT),
        vjepa_repo=args.vjepa_repo,
        vjepa_repo_commit=repo_commit,
        checkpoint=args.checkpoint,
        checkpoint_revision=args.checkpoint_revision,
        metadata_checksums=combined,
        model_variant=args.model_variant,
        projection_seed=args.projection_seed,
        current_frame_index=args.current_frame_index,
        history_frame_indices=args.history_frame_indices,
        future_frame_indices=args.future_frame_indices,
        frame_interval_s=args.frame_interval_s,
        input_image_hw=(args.image_size, args.image_size),
    )
    manifest["generator"].update(
        {"world_size": world_size, "cache_fingerprint": fingerprint}
    )
    atomic_write_json(args.output_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "cache": str(args.output_dir),
                "entries": len(tokens),
                "generated": total_generated,
                "resumed": total_resumed,
                "manifest_sha256": sha256_file(args.output_dir / "manifest.json"),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
