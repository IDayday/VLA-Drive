#!/usr/bin/env python3
"""Probe whether VGGT layer-11 global tokens retain usable scene geometry.

This is a read-only teacher diagnostic.  It trains linear ridge probes on
canonical three-view samples, then evaluates the same probes with canonical,
left/right-order-swapped, and cross-scene side-view inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pickle
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import Ridge
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.model.modules.vggt_query.geometry_probe import (  # noqa: E402
    SlotResidualizer,
    apply_slot_residualization,
    fit_slot_residualizer,
    project_lidar_to_depth_grid,
    regression_metrics,
    scale_aligned_depth_metrics,
)
from starVLA.model.modules.vggt_query.resolution_probe import (  # noqa: E402
    crop_and_pool_valid_patches,
)
from tools.precompute_vggt_query_cache import (  # noqa: E402
    git_revision,
    metadata_path,
    patch_validity_for_image,
    resolve_sensor_path,
    scene_image_paths,
    sha256_file,
)


SCHEMA_VERSION = 1
VIEWS = ("cam_f0", "cam_l0", "cam_r0")
PATCH_START_INDEX = 5
PATCH_GRID_SIZE = 37
GLOBAL_LAYER_INDEX = 11


@dataclass(frozen=True)
class SampleRef:
    token: str
    scene_name: str


@dataclass
class ProbeBatch:
    feature: torch.Tensor  # [N,Q,1024]
    teacher_target: torch.Tensor  # [N,Q,4]: log-depth, asinh(world xyz / 10m)
    teacher_valid: torch.Tensor  # [N,Q]
    lidar_target: torch.Tensor  # [N,Q,1]: log-depth
    lidar_valid: torch.Tensor  # [N,Q]
    direct_lidar_metrics: list[dict[str, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datalist-path", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--sensor-root", required=True)
    parser.add_argument("--vggt-repo", required=True)
    parser.add_argument("--vggt-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--frame-index", type=int, default=3)
    parser.add_argument("--train-samples", type=int, default=96)
    parser.add_argument("--val-samples", type=int, default=32)
    parser.add_argument("--grid-rows", type=int, default=6)
    parser.add_argument("--grid-cols", type=int, default=10)
    parser.add_argument("--lidar-min-points", type=int, default=3)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--skip-checkpoint-hash", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {value}, but the CUDA-compatible runtime is unavailable")
    return device


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def load_metadata(data_root: Path, split: str, token: str) -> Mapping[str, object]:
    with metadata_path(data_root, split, token).open("rb") as stream:
        raw = pickle.load(stream)
    if not isinstance(raw, Mapping):
        raise TypeError(f"NAVSIM metadata for {token} is not a mapping")
    return raw


def scene_name_from_metadata(raw: Mapping[str, object], frame_index: int) -> str:
    images = raw.get("glo_images")
    if not isinstance(images, Mapping) or "cam_f0" not in images:
        raise KeyError("NAVSIM metadata has no cam_f0 image contract")
    path = Path(images["cam_f0"]["image_paths"][frame_index])
    return path.parent.parent.name


def select_distinct_scene_samples(
    tokens: Sequence[str],
    *,
    data_root: Path,
    split: str,
    requested: int,
    frame_index: int,
    seed: int,
) -> list[SampleRef]:
    """Select one token per log so train/validation cannot share a log."""

    if requested <= 1:
        raise ValueError("geometry probe needs at least two distinct scenes")
    candidate_count = min(len(tokens), max(requested * 32, requested))
    indices = torch.linspace(0, len(tokens) - 1, steps=candidate_count).round().long().tolist()
    selected: dict[str, SampleRef] = {}
    for index in indices:
        token = tokens[index]
        raw = load_metadata(data_root, split, token)
        scene_name = scene_name_from_metadata(raw, frame_index)
        selected.setdefault(scene_name, SampleRef(token=token, scene_name=scene_name))
        if len(selected) >= requested:
            break
    if len(selected) < requested:
        raise RuntimeError(
            f"Only found {len(selected)} distinct NAVSIM logs; requested {requested}"
        )
    values = list(selected.values())
    random.Random(seed).shuffle(values)
    return values


def load_local_vggt_with_geometry_heads(
    repo_path: Path,
    checkpoint_path: Path,
    device: torch.device,
):
    """Load local VGGT depth/point heads without track or network access."""

    if not repo_path.is_dir():
        raise FileNotFoundError(f"Missing local VGGT repository: {repo_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing local VGGT checkpoint: {checkpoint_path}")
    sys.path.insert(0, str(repo_path))
    try:
        from safetensors import safe_open
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
    except ImportError as error:
        raise RuntimeError(
            f"Failed to import the local VGGT geometry stack from {repo_path}: {error}"
        ) from error

    model = VGGT(enable_camera=False, enable_point=True, enable_depth=True, enable_track=False)
    allowed_prefixes = ("aggregator.", "depth_head.", "point_head.")
    state_dict: dict[str, torch.Tensor] = {}
    if checkpoint_path.suffix == ".safetensors":
        with safe_open(str(checkpoint_path), framework="pt", device="cpu") as stream:
            for key in stream.keys():
                if key.startswith(allowed_prefixes):
                    state_dict[key] = stream.get_tensor(key)
    else:
        full_state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = {
            key: value for key, value in full_state.items() if key.startswith(allowed_prefixes)
        }
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "VGGT geometry checkpoint mismatch: "
            f"missing={incompatible.missing_keys[:5]} "
            f"unexpected={incompatible.unexpected_keys[:5]}"
        )
    # Keep DPT heads in float32.  The official VGGT forward disables autocast
    # around these heads, while the aggregator can safely run under bf16
    # autocast on the CUDA-compatible PPU runtime.
    model.eval().requires_grad_(False).to(device=device, dtype=torch.float32)
    return model, load_and_preprocess_images


def content_pixel_bounds(path: Path, target_size: int = 518, patch_size: int = 14) -> tuple[int, int, int, int]:
    """Return ``top,left,height,width`` for official VGGT pad preprocessing."""

    with Image.open(path) as image:
        width, height = image.size
    if width >= height:
        new_width = target_size
        new_height = round(height * target_size / width / patch_size) * patch_size
    else:
        new_height = target_size
        new_width = round(width * target_size / height / patch_size) * patch_size
    new_width = max(patch_size, min(target_size, new_width))
    new_height = max(patch_size, min(target_size, new_height))
    return (
        (target_size - new_height) // 2,
        (target_size - new_width) // 2,
        new_height,
        new_width,
    )


def pool_dense_front_map(
    value: torch.Tensor,
    *,
    front_path: Path,
    grid_size: tuple[int, int],
) -> torch.Tensor:
    """Crop VGGT padding and pool front map ``[518,518,C]`` to ``[Q,C]``."""

    assert value.ndim == 3 and value.shape[:2] == (518, 518)
    top, left, height, width = content_pixel_bounds(front_path)
    crop = value[top : top + height, left : left + width].permute(2, 0, 1).float()
    pooled = F.adaptive_avg_pool2d(crop, grid_size).permute(1, 2, 0)
    return pooled.reshape(grid_size[0] * grid_size[1], value.shape[-1])


def load_lidar_target(
    raw: Mapping[str, object],
    *,
    token: str,
    front_path: Path,
    sensor_root: Path,
    frame_index: int,
    grid_size: tuple[int, int],
    min_points: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load token-matched PCD and return front-view log-depth ``[Q,1]``."""

    try:
        from nuplan.database.utils.pointclouds.lidar import LidarPointCloud
    except ImportError as error:
        raise RuntimeError("LiDAR geometry probe requires the installed nuplan package") from error
    scene_name = scene_name_from_metadata(raw, frame_index)
    pcd_path = sensor_root / scene_name / "MergedPointCloud" / f"{token}.pcd"
    if not pcd_path.is_file():
        raise FileNotFoundError(f"Missing token-matched NAVSIM point cloud: {pcd_path}")
    with pcd_path.open("rb") as stream:
        point_cloud = LidarPointCloud.from_buffer(io.BytesIO(stream.read()), "pcd")
    points = torch.from_numpy(np.asarray(point_cloud.points[:3]).T.copy()).float()
    front = raw["glo_images"]["cam_f0"]
    rotation = torch.from_numpy(np.asarray(front["sensor2lidar_rotations"][frame_index])).float()
    translation = torch.from_numpy(np.asarray(front["sensor2lidar_translations"][frame_index])).float()
    intrinsics = torch.from_numpy(np.asarray(front["intrinsics"][frame_index])).float()
    with Image.open(front_path) as image:
        image_size = (image.height, image.width)
    depth, valid, _ = project_lidar_to_depth_grid(
        points,
        sensor2lidar_rotation=rotation,
        sensor2lidar_translation=translation,
        intrinsics=intrinsics,
        image_size=image_size,
        grid_size=grid_size,
        min_points=min_points,
    )
    log_depth = torch.log(depth.clamp_min(1e-6)).reshape(-1, 1)
    return log_depth, valid.reshape(-1)


class VGGTExtractor:
    """Capture pure global-block-11 tokens while running official heads once."""

    def __init__(self, model, preprocess, device: torch.device, grid_size: tuple[int, int]):
        self.model = model
        self.preprocess = preprocess
        self.device = device
        self.grid_size = grid_size
        self.captured: list[torch.Tensor] = []
        block = self.model.aggregator.global_blocks[GLOBAL_LAYER_INDEX]
        self.handle = block.register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output) -> None:
        value = output[0] if isinstance(output, tuple) else output
        if not isinstance(value, torch.Tensor):
            raise TypeError("VGGT global block hook did not return a tensor")
        self.captured.append(value.detach())

    def close(self) -> None:
        self.handle.remove()

    def extract(self, paths: Sequence[Path]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return front feature ``[Q,1024]``, depth ``[Q,1]``, points ``[Q,3]``."""

        if len(paths) != 3:
            raise ValueError("VGGT geometry probe requires front/left/right images")
        images = self.preprocess([str(path) for path in paths], mode="pad")
        assert images.shape == (3, 3, 518, 518)
        images = images.unsqueeze(0).to(device=self.device, dtype=torch.float32)
        self.captured.clear()
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            aggregated, patch_start_index = self.model.aggregator(images)
        if int(patch_start_index) != PATCH_START_INDEX:
            raise RuntimeError(
                f"VGGT patch_start_idx changed: expected {PATCH_START_INDEX}, "
                f"found {patch_start_index}"
            )
        head_inputs = [value.float() if value is not None else None for value in aggregated]
        with torch.inference_mode():
            depth, _depth_conf = self.model.depth_head(
                head_inputs, images=images, patch_start_idx=patch_start_index
            )
            points, _point_conf = self.model.point_head(
                head_inputs, images=images, patch_start_idx=patch_start_index
            )
        if len(self.captured) != 1:
            raise RuntimeError(
                f"Expected one global-layer-{GLOBAL_LAYER_INDEX} activation, got {len(self.captured)}"
            )
        global_tokens = self.captured[0]
        batch, flattened_tokens, feature_dim = global_tokens.shape
        per_view_tokens = PATCH_START_INDEX + PATCH_GRID_SIZE**2
        assert batch == 1 and flattened_tokens == 3 * per_view_tokens and feature_dim == 1024
        global_tokens = global_tokens.reshape(1, 3, per_view_tokens, feature_dim)
        patches = global_tokens[:, :, PATCH_START_INDEX:].reshape(
            1, 3, PATCH_GRID_SIZE, PATCH_GRID_SIZE, feature_dim
        )
        validity = torch.stack([patch_validity_for_image(path) for path in paths]).unsqueeze(0)
        validity = validity.to(device=patches.device)
        pooled, pooled_valid = crop_and_pool_valid_patches(
            patches,
            validity,
            output_size=self.grid_size,
            minimum_coverage=0.25,
        )
        front_feature = pooled[0, 0].reshape(-1, feature_dim).float().cpu()
        if not pooled_valid[0, 0].all():
            raise RuntimeError("Cropped front-view VGGT feature grid unexpectedly contains invalid cells")

        assert depth.shape[:4] == points.shape[:4] == (1, 3, 518, 518)
        front_depth = pool_dense_front_map(
            depth[0, 0].float().cpu(), front_path=paths[0], grid_size=self.grid_size
        )
        front_points = pool_dense_front_map(
            points[0, 0].float().cpu(), front_path=paths[0], grid_size=self.grid_size
        )
        assert front_depth.shape == (self.grid_size[0] * self.grid_size[1], 1)
        assert front_points.shape == (self.grid_size[0] * self.grid_size[1], 3)
        return front_feature, front_depth, front_points


def normalized_feature(features: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(features.float(), (features.shape[-1],))


def build_teacher_target(depth: torch.Tensor, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert depth.ndim == points.ndim == 2 and depth.shape[0] == points.shape[0]
    valid = torch.isfinite(depth).all(dim=-1) & torch.isfinite(points).all(dim=-1)
    valid &= depth[:, 0] > 1e-6
    target = torch.cat((torch.log(depth.clamp_min(1e-6)), torch.asinh(points / 10.0)), dim=-1)
    return target, valid


def collect_canonical(
    refs: Sequence[SampleRef],
    *,
    extractor: VGGTExtractor,
    data_root: Path,
    sensor_root: Path,
    split: str,
    frame_index: int,
    grid_size: tuple[int, int],
    lidar_min_points: int,
    description: str,
) -> ProbeBatch:
    features, teacher_targets, teacher_masks = [], [], []
    lidar_targets, lidar_masks, lidar_metrics = [], [], []
    for ref in tqdm(refs, desc=description):
        raw = load_metadata(data_root, split, ref.token)
        paths = scene_image_paths(raw, VIEWS, frame_index, sensor_root)
        feature, depth, points = extractor.extract(paths)
        teacher_target, teacher_valid = build_teacher_target(depth, points)
        lidar_target, lidar_valid = load_lidar_target(
            raw,
            token=ref.token,
            front_path=paths[0],
            sensor_root=sensor_root,
            frame_index=frame_index,
            grid_size=grid_size,
            min_points=lidar_min_points,
        )
        lidar_metrics.append(
            scale_aligned_depth_metrics(
                depth[:, 0].reshape(grid_size),
                torch.exp(lidar_target[:, 0]).reshape(grid_size),
                lidar_valid.reshape(grid_size),
            )
        )
        features.append(normalized_feature(feature))
        teacher_targets.append(teacher_target)
        teacher_masks.append(teacher_valid)
        lidar_targets.append(lidar_target)
        lidar_masks.append(lidar_valid)
    return ProbeBatch(
        feature=torch.stack(features),
        teacher_target=torch.stack(teacher_targets),
        teacher_valid=torch.stack(teacher_masks),
        lidar_target=torch.stack(lidar_targets),
        lidar_valid=torch.stack(lidar_masks),
        direct_lidar_metrics=lidar_metrics,
    )


def collect_variant(
    refs: Sequence[SampleRef],
    *,
    partner_refs: Sequence[SampleRef],
    mode: str,
    canonical: ProbeBatch,
    extractor: VGGTExtractor,
    data_root: Path,
    sensor_root: Path,
    split: str,
    frame_index: int,
    grid_size: tuple[int, int],
) -> ProbeBatch:
    if mode not in {"order_swap", "cross_scene_sides"}:
        raise ValueError(f"Unknown view corruption mode: {mode}")
    features, teacher_targets, teacher_masks = [], [], []
    lidar_metrics = []
    for index, ref in enumerate(tqdm(refs, desc=f"Validation {mode}")):
        raw = load_metadata(data_root, split, ref.token)
        paths = scene_image_paths(raw, VIEWS, frame_index, sensor_root)
        if mode == "order_swap":
            variant_paths = [paths[0], paths[2], paths[1]]
        else:
            partner = partner_refs[index]
            partner_raw = load_metadata(data_root, split, partner.token)
            partner_paths = scene_image_paths(partner_raw, VIEWS, frame_index, sensor_root)
            variant_paths = [paths[0], partner_paths[1], partner_paths[2]]
        feature, depth, points = extractor.extract(variant_paths)
        teacher_target, teacher_valid = build_teacher_target(depth, points)
        lidar_metrics.append(
            scale_aligned_depth_metrics(
                depth[:, 0].reshape(grid_size),
                torch.exp(canonical.lidar_target[index, :, 0]).reshape(grid_size),
                canonical.lidar_valid[index].reshape(grid_size),
            )
        )
        features.append(normalized_feature(feature))
        teacher_targets.append(teacher_target)
        teacher_masks.append(teacher_valid)
    return ProbeBatch(
        feature=torch.stack(features),
        teacher_target=torch.stack(teacher_targets),
        teacher_valid=torch.stack(teacher_masks),
        lidar_target=canonical.lidar_target,
        lidar_valid=canonical.lidar_valid,
        direct_lidar_metrics=lidar_metrics,
    )


def fit_ridge_probe(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    train_valid: torch.Tensor,
    *,
    alpha: float,
) -> tuple[SlotResidualizer, Ridge]:
    residualizer = fit_slot_residualizer(train_features, train_targets, train_valid)
    x, y, valid = apply_slot_residualization(
        train_features, train_targets, train_valid, residualizer
    )
    model = Ridge(alpha=alpha, fit_intercept=False, solver="lsqr")
    model.fit(x[valid].numpy(), y[valid].numpy())
    return residualizer, model


def evaluate_ridge_probe(
    features: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    residualizer: SlotResidualizer,
    model: Ridge,
) -> tuple[dict[str, float], list[float]]:
    x, _y, valid = apply_slot_residualization(features, targets, valid_mask, residualizer)
    residual_prediction = torch.from_numpy(model.predict(x.numpy())).float()
    baseline = residualizer.target_mean.unsqueeze(0).expand(targets.shape[0], -1, -1)
    prediction = baseline.reshape(-1, targets.shape[-1]) + residual_prediction
    target_flat = targets.reshape(-1, targets.shape[-1])
    baseline_flat = baseline.reshape(-1, targets.shape[-1])
    metrics = regression_metrics(prediction[valid], target_flat[valid], baseline_flat[valid])
    sample_errors = []
    prediction = prediction.reshape_as(targets)
    for index in range(targets.shape[0]):
        sample_valid = valid.reshape(targets.shape[:2])[index]
        sample_errors.append(
            float(torch.sqrt((prediction[index, sample_valid] - targets[index, sample_valid]).square().mean()))
        )
    return metrics, sample_errors


def feature_change(canonical: torch.Tensor, variant: torch.Tensor) -> dict[str, float]:
    assert canonical.shape == variant.shape
    canonical_unit = F.normalize(canonical.float(), dim=-1, eps=1e-6)
    variant_unit = F.normalize(variant.float(), dim=-1, eps=1e-6)
    return {
        "token_cosine_mean": float((canonical_unit * variant_unit).sum(dim=-1).mean()),
        "rms_change": float((canonical_unit - variant_unit).square().mean().sqrt()),
        "scene_descriptor_cosine_mean": float(
            F.cosine_similarity(canonical_unit.mean(dim=1), variant_unit.mean(dim=1), dim=-1).mean()
        ),
    }


def mean_metric(rows: Sequence[Mapping[str, float]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def paired_bootstrap_delta(
    canonical: Sequence[float],
    variant: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, float | bool]:
    """Bootstrap paired ``variant - canonical`` mean degradation."""

    if len(canonical) != len(variant) or not canonical:
        raise ValueError("paired bootstrap requires non-empty equal-length sequences")
    delta = torch.tensor(variant, dtype=torch.float64) - torch.tensor(canonical, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(len(delta), (samples, len(delta)), generator=generator)
    means = delta[indices].mean(dim=1)
    lower = float(torch.quantile(means, 0.025))
    upper = float(torch.quantile(means, 0.975))
    return {
        "mean_delta": float(delta.mean()),
        "ci95_lower": lower,
        "ci95_upper": upper,
        "probability_delta_positive": float((means > 0).double().mean()),
        "significant_degradation_95pct": lower > 0,
    }


def geometry_head_summary(batch: ProbeBatch) -> dict[str, float]:
    return {
        "scene_count": len(batch.direct_lidar_metrics),
        "mean_scale_aligned_abs_rel": mean_metric(batch.direct_lidar_metrics, "abs_rel"),
        "mean_scale_aligned_rmse_m": mean_metric(batch.direct_lidar_metrics, "rmse"),
        "mean_valid_lidar_bins": mean_metric(batch.direct_lidar_metrics, "valid_bins"),
    }


def main(args: argparse.Namespace) -> None:
    datalist_path = Path(args.datalist_path).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    sensor_root = Path(args.sensor_root).expanduser().resolve()
    repo_path = Path(args.vggt_repo).expanduser().resolve()
    checkpoint_path = Path(args.vggt_checkpoint).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    for path, kind in ((datalist_path, "datalist"), (checkpoint_path, "VGGT checkpoint")):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {kind}: {path}")
    for path, kind in ((data_root, "processed NAVSIM root"), (sensor_root, "sensor root"), (repo_path, "VGGT repo")):
        if not path.is_dir():
            raise FileNotFoundError(f"Missing {kind}: {path}")
    tokens = json.loads(datalist_path.read_text(encoding="utf-8"))
    if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
        raise TypeError("NAVSIM datalist must be a JSON list of token strings")
    total_samples = int(args.train_samples) + int(args.val_samples)
    refs = select_distinct_scene_samples(
        tokens,
        data_root=data_root,
        split=args.split,
        requested=total_samples,
        frame_index=int(args.frame_index),
        seed=int(args.seed),
    )
    train_refs = refs[: int(args.train_samples)]
    val_refs = refs[int(args.train_samples) :]
    if {ref.scene_name for ref in train_refs} & {ref.scene_name for ref in val_refs}:
        raise RuntimeError("Train and validation logs unexpectedly overlap")
    partner_refs = val_refs[1:] + val_refs[:1]
    if any(left.scene_name == right.scene_name for left, right in zip(val_refs, partner_refs)):
        raise RuntimeError("Cross-scene view corruption received a same-scene partner")

    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    grid_size = (int(args.grid_rows), int(args.grid_cols))
    started = time.time()
    model, preprocess = load_local_vggt_with_geometry_heads(repo_path, checkpoint_path, device)
    extractor = VGGTExtractor(model, preprocess, device, grid_size)
    try:
        train = collect_canonical(
            train_refs,
            extractor=extractor,
            data_root=data_root,
            sensor_root=sensor_root,
            split=args.split,
            frame_index=int(args.frame_index),
            grid_size=grid_size,
            lidar_min_points=int(args.lidar_min_points),
            description="Train canonical",
        )
        canonical = collect_canonical(
            val_refs,
            extractor=extractor,
            data_root=data_root,
            sensor_root=sensor_root,
            split=args.split,
            frame_index=int(args.frame_index),
            grid_size=grid_size,
            lidar_min_points=int(args.lidar_min_points),
            description="Validation canonical",
        )
        order_swap = collect_variant(
            val_refs,
            partner_refs=partner_refs,
            mode="order_swap",
            canonical=canonical,
            extractor=extractor,
            data_root=data_root,
            sensor_root=sensor_root,
            split=args.split,
            frame_index=int(args.frame_index),
            grid_size=grid_size,
        )
        cross_scene = collect_variant(
            val_refs,
            partner_refs=partner_refs,
            mode="cross_scene_sides",
            canonical=canonical,
            extractor=extractor,
            data_root=data_root,
            sensor_root=sensor_root,
            split=args.split,
            frame_index=int(args.frame_index),
            grid_size=grid_size,
        )
    finally:
        extractor.close()

    teacher_target_specs = {
        "depth": (slice(0, 1), "log(VGGT depth)"),
        "point_map": (slice(1, 4), "asinh(VGGT world xyz / 10m)"),
    }
    teacher_probes = {}
    for target_name, (target_slice, _description) in teacher_target_specs.items():
        teacher_probes[target_name] = fit_ridge_probe(
            train.feature,
            train.teacher_target[..., target_slice],
            train.teacher_valid,
            alpha=float(args.ridge_alpha),
        )
    lidar_residualizer, lidar_probe = fit_ridge_probe(
        train.feature,
        train.lidar_target,
        train.lidar_valid,
        alpha=float(args.ridge_alpha),
    )
    batches = {
        "canonical": canonical,
        "order_swap": order_swap,
        "cross_scene_sides": cross_scene,
    }
    teacher_results = {
        target_name: {"target": description}
        for target_name, (_target_slice, description) in teacher_target_specs.items()
    }
    teacher_errors: dict[str, dict[str, list[float]]] = {
        target_name: {} for target_name in teacher_target_specs
    }
    lidar_results, lidar_errors = {}, {}
    for target_name, (target_slice, _description) in teacher_target_specs.items():
        residualizer, probe = teacher_probes[target_name]
        train_metrics, _train_errors = evaluate_ridge_probe(
            train.feature,
            train.teacher_target[..., target_slice],
            train.teacher_valid,
            residualizer,
            probe,
        )
        teacher_results[target_name]["train"] = train_metrics
    for name, batch in batches.items():
        for target_name, (target_slice, _description) in teacher_target_specs.items():
            residualizer, probe = teacher_probes[target_name]
            metrics, errors = evaluate_ridge_probe(
                batch.feature,
                canonical.teacher_target[..., target_slice],
                canonical.teacher_valid,
                residualizer,
                probe,
            )
            teacher_results[target_name][name] = metrics
            teacher_errors[target_name][name] = errors
        metrics, errors = evaluate_ridge_probe(
            batch.feature,
            canonical.lidar_target,
            canonical.lidar_valid,
            lidar_residualizer,
            lidar_probe,
        )
        lidar_results[name] = metrics
        lidar_errors[name] = errors

    direct_geometry = {name: geometry_head_summary(batch) for name, batch in batches.items()}
    for variant_name in ("order_swap", "cross_scene_sides"):
        direct_geometry[variant_name]["paired_abs_rel_degradation"] = paired_bootstrap_delta(
            [row["abs_rel"] for row in canonical.direct_lidar_metrics],
            [row["abs_rel"] for row in batches[variant_name].direct_lidar_metrics],
            samples=int(args.bootstrap_samples),
            seed=int(args.seed),
        )
        for target_index, target_name in enumerate(teacher_target_specs):
            teacher_results[target_name][variant_name]["paired_rmse_degradation"] = (
                paired_bootstrap_delta(
                    teacher_errors[target_name]["canonical"],
                    teacher_errors[target_name][variant_name],
                    samples=int(args.bootstrap_samples),
                    seed=int(args.seed) + 1 + target_index,
                )
            )
        lidar_results[variant_name]["paired_rmse_degradation"] = paired_bootstrap_delta(
            lidar_errors["canonical"],
            lidar_errors[variant_name],
            samples=int(args.bootstrap_samples),
            seed=int(args.seed) + 2,
        )

    report = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_commit": git_revision(REPO_ROOT),
        "teacher_commit": git_revision(repo_path),
        "checkpoint_sha256": (
            "SKIPPED" if args.skip_checkpoint_hash else sha256_file(checkpoint_path)
        ),
        "datalist_sha256": file_sha256(datalist_path),
        "configuration": {
            "global_layer_index_zero_based": GLOBAL_LAYER_INDEX,
            "global_layer_human_name": "layer 11 global (zero-based block index 11)",
            "feature_dim": 1024,
            "grid": list(grid_size),
            "tokens_per_front_view": grid_size[0] * grid_size[1],
            "train_samples": len(train_refs),
            "val_samples": len(val_refs),
            "sampling": "one token per unique log; disjoint train/validation logs",
            "views": list(VIEWS),
            "frame_index": int(args.frame_index),
            "ridge_alpha": float(args.ridge_alpha),
            "lidar_min_points_per_bin": int(args.lidar_min_points),
            "seed": int(args.seed),
        },
        "controls": {
            "canonical": "front_i,left_i,right_i",
            "order_swap": "front_i,right_i,left_i; tests view-order equivariance",
            "cross_scene_sides": "front_i,left_j,right_j; tests cross-view scene use",
            "target_for_all_controls": "canonical scene i",
            "slot_template_control": "training-only per-grid-location target mean",
        },
        "feature_change_from_canonical": {
            "order_swap": feature_change(canonical.feature, order_swap.feature),
            "cross_scene_sides": feature_change(canonical.feature, cross_scene.feature),
        },
        "linear_probe_vggt_depth_and_point": teacher_results,
        "linear_probe_lidar_depth": lidar_results,
        "official_vggt_depth_head_vs_lidar": direct_geometry,
        "interpretation_contract": {
            "linear_probe_success": "R2 > 0 and SSE ratio < 1 versus slot-template baseline",
            "order_swap_expectation": "small/no degradation is expected from view-order equivariance",
            "cross_scene_expectation": "significant degradation supports use of side-view geometry",
            "causal_limit": "probe evidence is diagnostic, not proof of downstream planning gain",
        },
        "elapsed_seconds": time.time() - started,
        "diagnostic_paths": {
            "data_root": str(data_root),
            "sensor_root": str(sensor_root),
            "vggt_repo": str(repo_path),
            "vggt_checkpoint": str(checkpoint_path),
        },
    }
    write_json_atomic(output_path, report)
    print(f"[vggt-geometry-probe] COMPLETE output={output_path}")


if __name__ == "__main__":
    main(parse_args())
