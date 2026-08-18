#!/usr/bin/env python3
"""Build a variable-length dense final-layer VGGT cache for NAVSIM.

VGGT is imported only after the caller selects this offline capability and
supplies a local repository/checkpoint.  No weights are downloaded.  The
cache keeps every final-layer patch token in view-major/row-major/column-major
order and excludes all camera/register tokens.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import os
import pickle
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.cache.navsim_feature_cache import (  # noqa: E402
    CACHE_SCHEMA_VERSION,
    NavsimFeatureCacheReader,
    RankCacheWriter,
    component_dir,
    write_manifest,
    write_rank_completion,
)


COMPONENT = "vggt_dense"
DEFAULT_VIEWS = ("cam_f0", "cam_l0", "cam_r0")
PATCH_SIZE = 14
TARGET_WIDTH = 518
PATCH_START_INDEX = 5
TEACHER_LAYER_INDEX = 23
DEFAULT_FEATURE_DIM = 2048
RAY_FRAME = "navsim_current_ego_planning_frame"


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"


def distributed_context() -> tuple[int, int, torch.device]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        if world_size > 1 and not dist.is_initialized():
            dist.init_process_group("nccl")
    else:
        device = torch.device("cpu")
        if world_size > 1 and not dist.is_initialized():
            dist.init_process_group("gloo")
    return rank, world_size, device


def load_tokens(path: Path, max_samples: int | None) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing NAVSIM datalist: {path}")
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise TypeError("NAVSIM datalist must be a JSON list of scene-token strings")
    return values if max_samples is None else values[: int(max_samples)]


def metadata_path(data_root: Path, split: str, token: str) -> Path:
    path = data_root / "meta" / split / f"{token}.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing NAVSIM metadata: {path}")
    return path


def resolve_sensor_path(path_value: str, sensor_root: Path | None) -> Path:
    path = Path(path_value)
    if path.is_file():
        return path
    if sensor_root is None:
        raise FileNotFoundError(
            f"Image path from metadata is unavailable: {path}. Configure --sensor-root."
        )
    normalized = path_value.replace("\\", "/")
    for marker in (
        "/sensor_blobs/trainval/",
        "/trainval_sensor_blobs/trainval/",
        "/sensor_blobs/train/",
        "/sensor_blobs/test/",
    ):
        if marker in normalized:
            candidate = sensor_root / normalized.split(marker, 1)[1]
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        f"Cannot remap metadata image {path}; checked sensor root {sensor_root}"
    )


def scene_image_paths(
    raw: Mapping[str, object],
    views: tuple[str, ...],
    frame_index: int,
    sensor_root: Path | None,
) -> list[Path]:
    images = raw.get("glo_images")
    if not isinstance(images, Mapping):
        raise KeyError("NAVSIM metadata has no glo_images mapping")
    paths = []
    for view in views:
        camera = images.get(view)
        if not isinstance(camera, Mapping):
            raise KeyError(f"NAVSIM metadata has no requested view {view!r}")
        image_paths = camera.get("image_paths")
        if not isinstance(image_paths, Sequence) or frame_index >= len(image_paths):
            raise IndexError(f"View {view} has no frame index {frame_index}")
        paths.append(resolve_sensor_path(str(image_paths[frame_index]), sensor_root))
    return paths


def vggt_crop_geometry(
    width: int,
    height: int,
    *,
    target_width: int = TARGET_WIDTH,
    patch_size: int = PATCH_SIZE,
) -> dict[str, int | float]:
    """Mirror official VGGT ``mode='crop'`` resize and center-crop geometry."""

    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    resized_width = int(target_width)
    resized_height = round(height * (resized_width / width) / patch_size) * patch_size
    if resized_height <= 0:
        raise ValueError("official VGGT crop preprocessing rounded image height to zero")
    crop_top = max(0, (resized_height - target_width) // 2)
    output_height = min(resized_height, target_width)
    return {
        "resized_width": resized_width,
        "resized_height": resized_height,
        "output_width": resized_width,
        "output_height": output_height,
        "crop_top": crop_top,
        "scale_x": resized_width / width,
        "scale_y": resized_height / height,
    }


def mirror_vggt_crop_preprocess(
    image_paths: Sequence[Path | str],
) -> torch.Tensor:
    """Small dependency-free mirror used by estimate mode and CPU tests.

    Full cache extraction calls VGGT's own ``load_and_preprocess_images``.  This
    mirror intentionally supports only equal output shapes, matching the
    builder's shape-grouping contract rather than silently padding a batch.
    """

    images = []
    shapes = set()
    for path in image_paths:
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            geometry = vggt_crop_geometry(*image.size)
            image = image.resize(
                (
                    int(geometry["resized_width"]),
                    int(geometry["resized_height"]),
                ),
                Image.Resampling.BICUBIC,
            )
            crop_top = int(geometry["crop_top"])
            output_height = int(geometry["output_height"])
            if crop_top:
                image = image.crop((0, crop_top, image.width, crop_top + output_height))
            array = np.asarray(image, dtype=np.float32).copy() / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        images.append(tensor)
        shapes.add(tuple(tensor.shape[-2:]))
    if not images:
        raise ValueError("At least one image is required")
    if len(shapes) != 1:
        raise ValueError(
            "VGGT crop images have different output shapes; group them by shape "
            "instead of padding or stretching"
        )
    return torch.stack(images)


def _frame_value(camera: Mapping[str, object], name: str, frame_index: int) -> np.ndarray:
    value = camera.get(name)
    if value is None:
        raise KeyError(f"NAVSIM camera metadata is missing {name!r}")
    array = np.asarray(value)
    if array.ndim == 0 or frame_index >= array.shape[0]:
        raise IndexError(f"NAVSIM camera {name!r} has no frame index {frame_index}")
    return np.asarray(array[frame_index], dtype=np.float64)


def updated_crop_intrinsic(
    intrinsic: np.ndarray,
    *,
    original_width: int,
    original_height: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Apply official VGGT crop resize/crop to a camera intrinsic matrix."""

    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError("camera intrinsic must be a finite 3x3 matrix")
    geometry = vggt_crop_geometry(original_width, original_height)
    transform = np.array(
        [
            [geometry["scale_x"], 0.0, 0.0],
            [0.0, geometry["scale_y"], -geometry["crop_top"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    processed = transform @ intrinsic
    output_hw = (int(geometry["output_height"]), int(geometry["output_width"]))
    return processed, output_hw


def _quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if q.shape != (4,):
        raise ValueError("lidar2ego quaternion must contain four wxyz values")
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("lidar2ego quaternion is invalid")
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rotation_matrix(value: object, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (4,):
        array = _quaternion_wxyz_to_matrix(array)
    if array.shape != (3, 3) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite 3x3 matrix or wxyz quaternion")
    if not np.allclose(array.T @ array, np.eye(3), atol=1e-4):
        raise ValueError(f"{label} is not orthonormal")
    if not np.isclose(np.linalg.det(array), 1.0, atol=1e-4):
        raise ValueError(f"{label} determinant is not +1")
    return array


def resolve_lidar_to_ego(
    raw: Mapping[str, object], frame_index: int
) -> tuple[np.ndarray, np.ndarray, str]:
    """Resolve LiDAR→planning-ego, or assert the processed NAVSIM identity contract.

    Processed metadata produced by ``navsim_data_process/make_data.py`` stores
    camera-to-merged-LiDAR extrinsics and action targets in the current vehicle
    frame.  In this repository that merged LiDAR frame is the current planning
    ego frame (also called ``ego/lidar`` in the projection utilities).  The
    explicit identity below is therefore a checked data contract, not a
    missing-transform fallback.
    """

    containers: list[Mapping[str, object]] = [raw]
    status = raw.get("glo_status")
    if isinstance(status, Mapping):
        containers.append(status)
    rotation_value = translation_value = None
    for container in containers:
        for key in ("lidar2ego_rotations", "lidar2ego_rotation"):
            if key in container:
                value = np.asarray(container[key])
                rotation_value = (
                    value
                    if value.shape in ((3, 3), (4,))
                    else value[frame_index]
                )
                break
        for key in ("lidar2ego_translations", "lidar2ego_translation"):
            if key in container:
                value = np.asarray(container[key])
                translation_value = value if value.shape == (3,) else value[frame_index]
                break
        if rotation_value is not None or translation_value is not None:
            break
    if (rotation_value is None) != (translation_value is None):
        raise RuntimeError("NAVSIM metadata contains only half of lidar2ego transform")
    if rotation_value is not None:
        rotation = _rotation_matrix(rotation_value, "lidar2ego_rotation")
        translation = np.asarray(translation_value, dtype=np.float64).reshape(-1)
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise ValueError("lidar2ego_translation must be a finite xyz vector")
        return rotation, translation, RAY_FRAME

    images = raw.get("glo_images")
    if not isinstance(images, Mapping) or not isinstance(status, Mapping):
        raise RuntimeError(
            "Cannot establish LiDAR→ego: metadata is neither explicit nor the "
            "processed NAVSIM glo_images/glo_status contract"
        )
    global_poses = np.asarray(status.get("global_poses"))
    if (
        global_poses.ndim != 2
        or global_poses.shape[0] <= frame_index
        or global_poses.shape[1] < 3
        or not np.isfinite(global_poses[frame_index, :3]).all()
    ):
        raise RuntimeError(
            "Cannot assert the processed NAVSIM LiDAR=current-planning-ego "
            "contract without a finite current global pose"
        )
    return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64), RAY_FRAME


def build_patch_geometry(
    *,
    patch_h: int,
    patch_w: int,
    intrinsic: np.ndarray,
    distortion: np.ndarray,
    camera_to_ego_rotation: np.ndarray,
    camera_origin_ego: np.ndarray,
) -> dict[str, torch.Tensor]:
    """Return normalized UV and calibrated ego rays in row-major order."""

    if patch_h <= 0 or patch_w <= 0:
        raise ValueError("patch grid dimensions must be positive")
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    distortion = np.asarray(distortion, dtype=np.float64).reshape(-1)
    rotation = _rotation_matrix(camera_to_ego_rotation, "camera_to_ego_rotation")
    origin = np.asarray(camera_origin_ego, dtype=np.float64).reshape(-1)
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError("processed camera intrinsic must be finite 3x3")
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("camera origin must be a finite xyz vector")
    if distortion.size not in (0, 4, 5, 8, 12, 14) or not np.isfinite(distortion).all():
        raise ValueError("unsupported or non-finite OpenCV distortion coefficients")

    rows, cols = np.meshgrid(
        np.arange(patch_h, dtype=np.float64),
        np.arange(patch_w, dtype=np.float64),
        indexing="ij",
    )
    pixels = np.stack(
        ((cols + 0.5) * PATCH_SIZE, (rows + 0.5) * PATCH_SIZE), axis=-1
    ).reshape(-1, 2)
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "Calibrated VGGT rays require the project's OpenCV dependency"
        ) from error
    undistorted = cv2.undistortPoints(
        pixels.reshape(-1, 1, 2),
        intrinsic,
        distortion if distortion.size else None,
    ).reshape(-1, 2)
    camera_directions = np.concatenate(
        (undistorted, np.ones((undistorted.shape[0], 1), dtype=np.float64)), axis=1
    )
    camera_directions /= np.linalg.norm(camera_directions, axis=1, keepdims=True)
    ego_directions = (rotation @ camera_directions.T).T
    ego_directions /= np.linalg.norm(ego_directions, axis=1, keepdims=True)
    origins = np.broadcast_to(origin, ego_directions.shape).copy()
    uv = np.stack(
        (
            2.0 * (cols + 0.5) / patch_w - 1.0,
            2.0 * (rows + 0.5) / patch_h - 1.0,
        ),
        axis=-1,
    ).reshape(-1, 2)
    return {
        "uv_coords": torch.from_numpy(uv.astype(np.float32)),
        "ray_features": torch.from_numpy(
            np.concatenate((origins, ego_directions), axis=1).astype(np.float32)
        ),
    }


def scene_patch_geometry(
    raw: Mapping[str, object],
    paths: Sequence[Path],
    views: tuple[str, ...],
    frame_index: int,
    processed_hw: tuple[int, int],
) -> tuple[list[dict[str, torch.Tensor]], str]:
    images = raw.get("glo_images")
    if not isinstance(images, Mapping):
        raise KeyError("NAVSIM metadata has no glo_images mapping")
    lidar_rotation, lidar_translation, frame_contract = resolve_lidar_to_ego(
        raw, frame_index
    )
    patch_h, patch_w = processed_hw[0] // PATCH_SIZE, processed_hw[1] // PATCH_SIZE
    geometry = []
    for path, view in zip(paths, views):
        camera = images.get(view)
        if not isinstance(camera, Mapping):
            raise KeyError(f"NAVSIM metadata has no camera calibration for {view}")
        intrinsic = _frame_value(camera, "intrinsics", frame_index)
        distortion = _frame_value(camera, "distortions", frame_index)
        camera_to_lidar = _rotation_matrix(
            _frame_value(camera, "sensor2lidar_rotations", frame_index),
            f"{view}.sensor2lidar_rotation",
        )
        camera_origin_lidar = _frame_value(
            camera, "sensor2lidar_translations", frame_index
        ).reshape(-1)
        if camera_origin_lidar.shape != (3,):
            raise ValueError(f"{view}.sensor2lidar_translation must be xyz")
        with Image.open(path) as image:
            processed_intrinsic, expected_hw = updated_crop_intrinsic(
                intrinsic,
                original_width=image.width,
                original_height=image.height,
            )
        if expected_hw != processed_hw:
            raise RuntimeError(
                f"Official VGGT crop shape {processed_hw} disagrees with calibrated "
                f"intrinsic shape {expected_hw} for {path}"
            )
        camera_to_ego = lidar_rotation @ camera_to_lidar
        camera_origin_ego = (
            lidar_rotation @ camera_origin_lidar + lidar_translation
        )
        geometry.append(
            build_patch_geometry(
                patch_h=patch_h,
                patch_w=patch_w,
                intrinsic=processed_intrinsic,
                distortion=distortion,
                camera_to_ego_rotation=camera_to_ego,
                camera_origin_ego=camera_origin_ego,
            )
        )
    return geometry, frame_contract


def build_dense_payload(
    last_tokens: torch.Tensor,
    *,
    patch_start_idx: int,
    patch_grid_hw: torch.Tensor,
    patch_geometry: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Package one scene without special tokens or spatial pooling."""

    if last_tokens.ndim != 3:
        raise ValueError("last_tokens must be [V,5+P,D]")
    views, token_count, feature_dim = last_tokens.shape
    if int(patch_start_idx) != PATCH_START_INDEX:
        raise RuntimeError(
            f"VGGT patch_start_idx changed: expected {PATCH_START_INDEX}, "
            f"found {patch_start_idx}"
        )
    grids = torch.as_tensor(patch_grid_hw, dtype=torch.int16).cpu()
    if grids.shape != (views, 2) or len(patch_geometry) != views:
        raise ValueError("patch grids/geometry must contain one entry per view")
    features = []
    view_ids = []
    uv_coords = []
    ray_features = []
    for view_index in range(views):
        patch_h, patch_w = (int(value) for value in grids[view_index])
        count = patch_h * patch_w
        if token_count != patch_start_idx + count:
            raise ValueError(
                "VGGT aggregator uses one shared patch grid per scene; found "
                f"tokens={token_count}, view={view_index}, grid={patch_h}x{patch_w}"
            )
        geometry = patch_geometry[view_index]
        uv = geometry["uv_coords"]
        rays = geometry["ray_features"]
        if uv.shape != (count, 2) or rays.shape != (count, 6):
            raise ValueError("UV/ray lengths do not match dense VGGT patch count")
        features.append(last_tokens[view_index, patch_start_idx:])
        view_ids.append(torch.full((count,), view_index, dtype=torch.int16))
        uv_coords.append(uv)
        ray_features.append(rays)
    dense_features = torch.cat(features, dim=0)
    expected_count = sum(int(h) * int(w) for h, w in grids.tolist())
    if dense_features.shape != (expected_count, feature_dim):
        raise AssertionError("dense VGGT flattening changed token order or count")
    return {
        "features": dense_features.detach().to(torch.bfloat16).cpu().contiguous(),
        "valid_mask": torch.ones(expected_count, dtype=torch.bool),
        "view_ids": torch.cat(view_ids).contiguous(),
        "uv_coords": torch.cat(uv_coords).to(torch.float16).contiguous(),
        "ray_features": torch.cat(ray_features).to(torch.float32).contiguous(),
        "patch_grid_hw": grids.contiguous(),
    }


def load_official_preprocess(repo_path: Path) -> Callable:
    if not repo_path.is_dir():
        raise FileNotFoundError(
            f"Missing local VGGT repository: {repo_path}. Set VGGT_REPO; no download is attempted."
        )
    sys.path.insert(0, str(repo_path))
    try:
        from vggt.utils.load_fn import load_and_preprocess_images
    except ImportError as error:
        raise RuntimeError(f"Failed to import VGGT preprocessing from {repo_path}: {error}") from error
    return load_and_preprocess_images


def load_local_vggt(
    repo_path: Path, checkpoint_path: Path, device: torch.device
) -> tuple[torch.nn.Module, Callable]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Missing local VGGT checkpoint: {checkpoint_path}. Set VGGT_CHECKPOINT; no download is attempted."
        )
    preprocess = load_official_preprocess(repo_path)
    try:
        from vggt.models.vggt import VGGT
    except ImportError as error:
        raise RuntimeError(f"Failed to import VGGT model from {repo_path}: {error}") from error
    model = VGGT(
        enable_camera=False,
        enable_point=False,
        enable_depth=False,
        enable_track=False,
    )
    if checkpoint_path.suffix == ".safetensors":
        try:
            from safetensors import safe_open
        except ImportError as error:
            raise RuntimeError("Loading VGGT safetensors requires safetensors") from error
        state_dict = {}
        with safe_open(str(checkpoint_path), framework="pt", device="cpu") as stream:
            for key in stream.keys():
                if key.startswith("aggregator."):
                    state_dict[key] = stream.get_tensor(key)
    else:
        full_state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = {
            key: value for key, value in full_state.items() if key.startswith("aggregator.")
        }
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "VGGT checkpoint architecture mismatch: "
            f"missing={incompatible.missing_keys[:5]} "
            f"unexpected={incompatible.unexpected_keys[:5]}"
        )
    model.eval().requires_grad_(False).to(device=device, dtype=torch.float32)
    return model, preprocess


def _load_scene(
    *,
    token: str,
    data_root: Path,
    split: str,
    sensor_root: Path | None,
    views: tuple[str, ...],
    frame_index: int,
) -> tuple[Mapping[str, object], list[Path], tuple[int, int]]:
    with metadata_path(data_root, split, token).open("rb") as stream:
        raw = pickle.load(stream)
    paths = scene_image_paths(raw, views, frame_index, sensor_root)
    shapes = []
    for path in paths:
        with Image.open(path) as image:
            geometry = vggt_crop_geometry(image.width, image.height)
        shapes.append((int(geometry["output_height"]), int(geometry["output_width"])))
    if len(set(shapes)) != 1:
        raise RuntimeError(
            f"Scene {token} has different crop shapes across views {shapes}; "
            "the dense multi-view aggregator will not stretch or pad them"
        )
    return raw, paths, shapes[0]


def _iter_batches(values: Sequence[int], batch_size: int) -> Iterable[list[int]]:
    for start in range(0, len(values), batch_size):
        yield list(values[start : start + batch_size])


def estimate_cache(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root).expanduser().resolve()
    sensor_root = Path(args.sensor_root).expanduser().resolve() if args.sensor_root else None
    repo_path = Path(args.vggt_repo).expanduser().resolve()
    preprocess = load_official_preprocess(repo_path)
    tokens = load_tokens(Path(args.datalist_path), args.max_samples)
    if not tokens:
        raise ValueError("Cannot estimate an empty datalist")
    sample_limit = min(len(tokens), max(1, min(4, int(args.batch_size))))
    views = tuple(item.strip() for item in args.views.split(",") if item.strip())
    token_counts = []
    grids = []
    for token in tokens[:sample_limit]:
        _, paths, expected_hw = _load_scene(
            token=token,
            data_root=data_root,
            split=args.split,
            sensor_root=sensor_root,
            views=views,
            frame_index=args.frame_index,
        )
        images = preprocess([str(path) for path in paths], mode="crop")
        actual_hw = tuple(int(value) for value in images.shape[-2:])
        if actual_hw != expected_hw:
            raise RuntimeError(
                f"Official VGGT crop returned {actual_hw}, expected {expected_hw}"
            )
        patch_h, patch_w = actual_hw[0] // PATCH_SIZE, actual_hw[1] // PATCH_SIZE
        count = len(views) * patch_h * patch_w
        token_counts.append(count)
        grids.append([patch_h, patch_w])
    typical_tokens = int(np.median(token_counts))
    feature_dim = DEFAULT_FEATURE_DIM
    per_token_bytes = feature_dim * 2 + 1 + 2 + 2 * 2 + 6 * 4
    record_bytes = typical_tokens * per_token_bytes + len(views) * 2 * 2
    full_samples = len(json.loads(Path(args.datalist_path).read_text(encoding="utf-8")))
    report = {
        "mode": "estimate-only",
        "sampled_scenes": sample_limit,
        "sample_patch_grids": grids,
        "sample_token_counts": token_counts,
        "typical_N": typical_tokens,
        "Dg": feature_dim,
        "estimated_bytes_per_record": record_bytes,
        "estimated_full_cache_gib": record_bytes * full_samples / 1024**3,
        "full_sample_count": full_samples,
        "does_not_run_vggt_model": True,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def validate_payload(payload: Mapping[str, torch.Tensor], feature_dim: int) -> None:
    required = {
        "features",
        "valid_mask",
        "view_ids",
        "uv_coords",
        "ray_features",
        "patch_grid_hw",
    }
    missing = required - set(payload)
    if missing:
        raise RuntimeError(f"Dense VGGT payload is missing keys: {sorted(missing)}")
    features = payload["features"]
    if features.ndim != 2 or features.shape[1] != feature_dim:
        raise RuntimeError("Dense VGGT features have an invalid shape")
    count = features.shape[0]
    if features.dtype != torch.bfloat16:
        raise RuntimeError("Dense VGGT features must be BF16")
    if payload["valid_mask"].shape != (count,) or payload["valid_mask"].dtype != torch.bool:
        raise RuntimeError("Dense VGGT valid_mask contract mismatch")
    if payload["view_ids"].shape != (count,):
        raise RuntimeError("Dense VGGT view_ids contract mismatch")
    if payload["uv_coords"].shape != (count, 2):
        raise RuntimeError("Dense VGGT uv_coords contract mismatch")
    if payload["ray_features"].shape != (count, 6):
        raise RuntimeError("Dense VGGT ray_features contract mismatch")
    grids = payload["patch_grid_hw"]
    if grids.shape != (3, 2):
        raise RuntimeError("Dense VGGT patch_grid_hw contract mismatch")
    if sum(int(h) * int(w) for h, w in grids.tolist()) != count:
        raise RuntimeError("Dense VGGT patch grids do not sum to the feature count")
    if not payload["valid_mask"].all():
        raise RuntimeError("Unpadded cache records must mark every dense token valid")
    directions = payload["ray_features"][:, 3:].float()
    if not torch.allclose(
        directions.norm(dim=-1), torch.ones(count), atol=2e-4, rtol=2e-4
    ):
        raise RuntimeError("Dense VGGT ray directions are not unit normalized")


def validate_cache(args: argparse.Namespace) -> None:
    tokens = load_tokens(Path(args.datalist_path), args.max_samples)
    reader = NavsimFeatureCacheReader(
        args.cache_root, components=(COMPONENT,), strict=True
    )
    manifest = reader.manifests[COMPONENT]
    expected = {
        "component": COMPONENT,
        "view_order": [item.strip() for item in args.views.split(",") if item.strip()],
        "frame_index": int(args.frame_index),
        "teacher_layer_index": TEACHER_LAYER_INDEX,
        "teacher_layer": "aggregator[-1]",
        "teacher_attention_branch": "full_aggregated_feature",
        "include_special_tokens": False,
        "spatial_pooling": None,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(
                f"Dense VGGT manifest {key} mismatch: {manifest.get(key)!r} != {value!r}"
            )
    preprocess = manifest.get("preprocess", {})
    if preprocess.get("mode") != "crop" or not preprocess.get(
        "preserve_aspect_ratio", False
    ):
        raise RuntimeError("Dense VGGT cache was not produced with crop preprocessing")
    feature_dim = int(manifest.get("feature_dim", -1))
    if feature_dim <= 0:
        raise RuntimeError("Dense VGGT manifest has no valid feature_dim")
    for index, token in enumerate(tqdm(tokens, desc="validate dense VGGT cache")):
        payload = reader.get(COMPONENT, index, token)
        if payload is None:
            raise RuntimeError(f"Missing dense VGGT payload for {token}")
        validate_payload(payload, feature_dim)
    print(f"[vggt-dense-cache] VALID samples={len(tokens)} root={args.cache_root}")


def build_cache(args: argparse.Namespace) -> None:
    rank, world_size, device = distributed_context()
    repo_path = Path(args.vggt_repo).expanduser().resolve()
    checkpoint_path = Path(args.vggt_checkpoint).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    sensor_root = Path(args.sensor_root).expanduser().resolve() if args.sensor_root else None
    views = tuple(item.strip() for item in args.views.split(",") if item.strip())
    if views != DEFAULT_VIEWS:
        if len(views) != 3:
            raise ValueError("Dense VGGT currently requires exactly three configured views")
    tokens = load_tokens(Path(args.datalist_path), args.max_samples)
    model, preprocess = load_local_vggt(repo_path, checkpoint_path, device)
    owned = list(range(rank, len(tokens), world_size))
    component_path = component_dir(args.cache_root, COMPONENT)
    component_path.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    feature_dim: int | None = None
    # Every ray is expressed in the same target frame whether metadata supplies
    # an explicit lidar2ego transform or uses the checked processed-NAVSIM
    # LiDAR=current-ego contract. Seed this for resume-only ranks too.
    ray_frames: set[str] = {RAY_FRAME}
    token_counts: list[int] = []
    patch_grids: set[tuple[int, int]] = set()
    started = time.time()

    with RankCacheWriter(
        args.cache_root,
        COMPONENT,
        rank,
        int(args.map_size_gb * 1024**3),
        commit_interval=args.commit_interval,
    ) as writer:
        iterator: Iterable[list[int]] = _iter_batches(owned, args.batch_size)
        if rank == 0:
            iterator = tqdm(list(iterator), desc="dense VGGT cache")
        for index_batch in iterator:
            scenes = []
            for sample_index in index_batch:
                token = tokens[sample_index]
                existing = writer.transaction.get(token.encode("utf-8"))
                if existing is not None and not args.overwrite:
                    payload = torch.load(
                        io.BytesIO(bytes(existing)), map_location="cpu", weights_only=True
                    )
                    existing_dim = int(payload["features"].shape[1])
                    validate_payload(payload, existing_dim)
                    feature_dim = feature_dim or existing_dim
                    if feature_dim != existing_dim:
                        raise RuntimeError("Existing dense cache mixes feature dimensions")
                    token_counts.append(int(payload["features"].shape[0]))
                    for grid in payload["patch_grid_hw"].tolist():
                        patch_grids.add((int(grid[0]), int(grid[1])))
                    writer.skipped += 1
                    continue
                raw, paths, output_hw = _load_scene(
                    token=token,
                    data_root=data_root,
                    split=args.split,
                    sensor_root=sensor_root,
                    views=views,
                    frame_index=args.frame_index,
                )
                scenes.append((sample_index, token, raw, paths, output_hw))

            grouped: dict[tuple[int, int], list[tuple]] = defaultdict(list)
            for scene in scenes:
                grouped[scene[-1]].append(scene)
            for output_hw, shape_scenes in grouped.items():
                image_batches = []
                geometry_batches = []
                for _, _, raw, paths, _ in shape_scenes:
                    images = preprocess([str(path) for path in paths], mode="crop")
                    if tuple(images.shape) != (len(views), 3, *output_hw):
                        raise RuntimeError(
                            "Official VGGT preprocessing changed the grouped crop shape: "
                            f"expected {(len(views), 3, *output_hw)}, found {tuple(images.shape)}"
                        )
                    image_batches.append(images)
                    geometry, ray_frame = scene_patch_geometry(
                        raw, paths, views, args.frame_index, output_hw
                    )
                    geometry_batches.append(geometry)
                    ray_frames.add(ray_frame)
                images = torch.stack(image_batches).to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                with torch.inference_mode(), torch.autocast(
                    device_type=device.type,
                    dtype=dtype,
                    enabled=device.type == "cuda",
                ):
                    aggregated_tokens_list, patch_start_idx = model.aggregator(images)
                if int(patch_start_idx) != PATCH_START_INDEX:
                    raise RuntimeError(
                        f"VGGT patch_start_idx changed: expected {PATCH_START_INDEX}, "
                        f"found {patch_start_idx}"
                    )
                if (
                    len(aggregated_tokens_list) != TEACHER_LAYER_INDEX + 1
                    or aggregated_tokens_list[TEACHER_LAYER_INDEX] is None
                ):
                    raise RuntimeError(
                        "VGGT aggregator[-1] is no longer the configured layer 23"
                    )
                last_tokens = aggregated_tokens_list[-1]
                if last_tokens is None or last_tokens.ndim != 4:
                    raise RuntimeError("VGGT final aggregator did not return [B,V,N,D]")
                patch_h, patch_w = output_hw[0] // PATCH_SIZE, output_hw[1] // PATCH_SIZE
                expected_tokens = PATCH_START_INDEX + patch_h * patch_w
                if last_tokens.shape[1:3] != (len(views), expected_tokens):
                    raise RuntimeError(
                        "VGGT final token grid mismatch: "
                        f"expected views/tokens={(len(views), expected_tokens)}, "
                        f"found={tuple(last_tokens.shape[1:3])}"
                    )
                current_dim = int(last_tokens.shape[-1])
                if feature_dim is None:
                    feature_dim = current_dim
                if current_dim != feature_dim:
                    raise RuntimeError("VGGT final feature dimension changed within cache")
                grid_tensor = torch.tensor(
                    [[patch_h, patch_w]] * len(views), dtype=torch.int16
                )
                for scene_index, (_, token, _, _, _) in enumerate(shape_scenes):
                    payload = build_dense_payload(
                        last_tokens[scene_index],
                        patch_start_idx=int(patch_start_idx),
                        patch_grid_hw=grid_tensor,
                        patch_geometry=geometry_batches[scene_index],
                    )
                    validate_payload(payload, feature_dim)
                    writer.put(token, payload, overwrite=args.overwrite)
                    token_counts.append(int(payload["features"].shape[0]))
                    patch_grids.add((patch_h, patch_w))
                del images, last_tokens, aggregated_tokens_list

        completion = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "component": COMPONENT,
            "rank": rank,
            "world_size": world_size,
            "sample_count": len(tokens),
            "owned_samples": len(owned),
            "written": writer.written,
            "skipped": writer.skipped,
            "feature_dim": feature_dim,
            "ray_frames": sorted(ray_frames),
            "token_count_min": min(token_counts) if token_counts else None,
            "token_count_max": max(token_counts) if token_counts else None,
            "patch_grids": sorted([list(value) for value in patch_grids]),
            "elapsed_seconds": time.time() - started,
        }
    write_rank_completion(args.cache_root, COMPONENT, rank, completion)
    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        completions = []
        dimensions = set()
        frames = set()
        all_grids = set()
        token_min = None
        token_max = None
        for owner in range(world_size):
            path = component_path / f"rank_{owner:05d}.complete.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing dense cache completion: {path}")
            completion = json.loads(path.read_text(encoding="utf-8"))
            completions.append(completion)
            if completion.get("feature_dim") is not None:
                dimensions.add(int(completion["feature_dim"]))
            frames.update(completion.get("ray_frames", []))
            all_grids.update(tuple(value) for value in completion.get("patch_grids", []))
            value_min = completion.get("token_count_min")
            value_max = completion.get("token_count_max")
            if value_min is not None:
                token_min = int(value_min) if token_min is None else min(token_min, int(value_min))
            if value_max is not None:
                token_max = int(value_max) if token_max is None else max(token_max, int(value_max))
        if len(dimensions) != 1:
            raise RuntimeError(f"Dense cache ranks disagree on feature_dim: {dimensions}")
        if len(frames) != 1:
            raise RuntimeError(f"Dense cache ranks disagree on ray frame: {frames}")
        final_dim = dimensions.pop()
        manifest = {
            "component": COMPONENT,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "world_size": world_size,
            "sample_count": len(tokens),
            "view_order": list(views),
            "flatten_order": "view-major,row-major,col-major",
            "frame_index": int(args.frame_index),
            "teacher_layer_index": TEACHER_LAYER_INDEX,
            "teacher_layer": "aggregator[-1]",
            "teacher_attention_branch": "full_aggregated_feature",
            "include_special_tokens": False,
            "patch_start_idx": PATCH_START_INDEX,
            "spatial_pooling": None,
            "preprocess": {
                "mode": "crop",
                "target_long_side": TARGET_WIDTH,
                "patch_size": PATCH_SIZE,
                "preserve_aspect_ratio": True,
                "shape_policy": "group_scenes_by_official_output_hw",
            },
            "feature_dim": final_dim,
            "ray_frame": frames.pop(),
            "patch_grids_observed": [list(value) for value in sorted(all_grids)],
            "token_count_min": token_min,
            "token_count_max": token_max,
            "datalist_sha256": sha256_file(Path(args.datalist_path)),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "vggt_repo_commit": git_revision(repo_path),
            "project_commit": git_revision(REPO_ROOT),
            "extractor_sha256": sha256_file(Path(__file__).resolve()),
            "payload_contract": {
                "features": "bfloat16[N,Dg]",
                "valid_mask": "bool[N]",
                "view_ids": "int16[N]",
                "uv_coords": "float16[N,2] normalized to [-1,1]",
                "ray_features": "float32[N,6] ego origin_xyz + unit direction_xyz",
                "patch_grid_hw": "int16[3,2]",
            },
            "rank_completions": completions,
            "diagnostic_paths": {
                "vggt_repo": str(repo_path),
                "vggt_checkpoint": str(checkpoint_path),
                "data_root": str(data_root),
                "sensor_root": str(sensor_root) if sensor_root else None,
            },
        }
        write_manifest(args.cache_root, COMPONENT, manifest)
        print(
            f"[vggt-dense-cache] COMPLETE samples={len(tokens)} "
            f"N={token_min}..{token_max} Dg={final_dim} "
            f"manifest={component_path / 'manifest.json'}"
        )

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def main(args: argparse.Namespace) -> None:
    if args.estimate_only:
        estimate_cache(args)
        return
    if args.validate_only:
        validate_cache(args)
        return
    build_cache(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datalist-path", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--sensor-root", default=None)
    parser.add_argument(
        "--cache-root",
        default=os.environ.get("NAVSIM_VGGT_DENSE_CACHE_ROOT", ""),
    )
    parser.add_argument("--vggt-repo", default=os.environ.get("VGGT_REPO", ""))
    parser.add_argument(
        "--vggt-checkpoint", default=os.environ.get("VGGT_CHECKPOINT", "")
    )
    parser.add_argument("--views", default=",".join(DEFAULT_VIEWS))
    parser.add_argument("--frame-index", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--map-size-gb", type=int, default=64)
    parser.add_argument("--commit-interval", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--estimate-only", action="store_true")
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    if not str(parsed.cache_root).strip() and not parsed.estimate_only:
        raise ValueError(
            "Set --cache-root or NAVSIM_VGGT_DENSE_CACHE_ROOT for dense VGGT cache"
        )
    if not str(parsed.vggt_repo).strip() and not parsed.validate_only:
        raise ValueError("Set --vggt-repo or VGGT_REPO; no repository is downloaded")
    if (
        not parsed.estimate_only
        and not parsed.validate_only
        and not str(parsed.vggt_checkpoint).strip()
    ):
        raise ValueError(
            "Set --vggt-checkpoint or VGGT_CHECKPOINT; no weights are downloaded"
        )
    main(parsed)
