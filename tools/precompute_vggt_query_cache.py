#!/usr/bin/env python3
"""Precompute compact DPT-pre VGGT query targets for NAVSIM.

VGGT and safetensors are imported only after local paths have been validated.
No model is downloaded. Each distributed rank writes an independent LMDB and
rank statistics; rank 0 atomically publishes a manifest only after all ranks
finish.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.cache.navsim_feature_cache import (  # noqa: E402
    CACHE_SCHEMA_VERSION,
    RankCacheWriter,
    component_dir,
    write_manifest,
    write_rank_completion,
)
from starVLA.model.modules.vggt_query.targets import (  # noqa: E402
    extract_vggt_layer11_memory_targets,
    select_vggt_global_teacher_layer,
)
from starVLA.model.modules.vggt_query.types import VGGTQueryLayout  # noqa: E402


COMPONENT = "vggt_query"
DEFAULT_VIEWS = ("cam_f0", "cam_l0", "cam_r0")
VGGT_PATCH_START_INDEX = 5
VGGT_TEACHER_LAYER_INDEX = 11
VGGT_TEACHER_BRANCH_DIM = 1024


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
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
    return values if max_samples is None else values[:max_samples]


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
    markers = (
        "/sensor_blobs/trainval/",
        "/trainval_sensor_blobs/trainval/",
        "/sensor_blobs/train/",
    )
    for marker in markers:
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
        if view not in images:
            raise KeyError(f"NAVSIM metadata has no requested view {view!r}")
        image_paths = images[view].get("image_paths")
        if frame_index >= len(image_paths):
            raise IndexError(f"View {view} has no frame index {frame_index}")
        paths.append(resolve_sensor_path(image_paths[frame_index], sensor_root))
    return paths


def patch_validity_for_image(path: Path, target_size: int = 518, patch_size: int = 14) -> torch.Tensor:
    """Return content coverage for official VGGT ``mode='pad'`` patches."""

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
    left = (target_size - new_width) // 2
    top = (target_size - new_height) // 2
    grid = target_size // patch_size
    validity = torch.zeros(grid, grid, dtype=torch.float32)
    for row in range(grid):
        y0, y1 = row * patch_size, (row + 1) * patch_size
        overlap_y = max(0, min(y1, top + new_height) - max(y0, top))
        for col in range(grid):
            x0, x1 = col * patch_size, (col + 1) * patch_size
            overlap_x = max(0, min(x1, left + new_width) - max(x0, left))
            validity[row, col] = overlap_x * overlap_y / float(patch_size**2)
    return validity


def load_local_vggt(repo_path: Path, checkpoint_path: Path, device: torch.device):
    if not repo_path.is_dir():
        raise FileNotFoundError(
            f"Missing local VGGT repository: {repo_path}. Set VGGT_REPO; no download is attempted."
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Missing local VGGT checkpoint: {checkpoint_path}. Set VGGT_CHECKPOINT; no download is attempted."
        )
    sys.path.insert(0, str(repo_path))
    try:
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
    except ImportError as error:
        raise RuntimeError(
            f"Failed to import VGGT from {repo_path}; install its local dependencies: {error}"
        ) from error

    model = VGGT(
        enable_camera=False,
        enable_point=True,
        enable_depth=True,
        enable_track=False,
    )
    allowed_prefixes = ("aggregator.", "depth_head.", "point_head.")
    if checkpoint_path.suffix == ".safetensors":
        try:
            from safetensors import safe_open
        except ImportError as error:
            raise RuntimeError("Loading a .safetensors VGGT checkpoint requires safetensors") from error
        state_dict = {}
        with safe_open(str(checkpoint_path), framework="pt", device="cpu") as stream:
            for key in stream.keys():
                if key.startswith(allowed_prefixes):
                    state_dict[key] = stream.get_tensor(key)
    else:
        full_state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = {
            key: value
            for key, value in full_state.items()
            if key.startswith(allowed_prefixes)
        }
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "VGGT checkpoint architecture mismatch: "
            f"missing={incompatible.missing_keys[:5]} unexpected={incompatible.unexpected_keys[:5]}"
        )
    model.eval().requires_grad_(False)
    # DPT heads follow the official float32 path. The aggregator is autocast
    # separately when the cache is generated.
    model.to(device=device, dtype=torch.float32)
    return model, load_and_preprocess_images


def content_pixel_bounds(
    path: Path,
    target_size: int = 518,
    patch_size: int = 14,
) -> tuple[int, int, int, int]:
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


def pool_dense_map(
    value: torch.Tensor,
    *,
    path: Path,
    output_size: tuple[int, int],
) -> torch.Tensor:
    """Crop padding and pool one dense VGGT map ``[518,518,C]``."""

    if value.ndim == 2:
        value = value.unsqueeze(-1)
    assert value.ndim == 3 and value.shape[:2] == (518, 518)
    top, left, height, width = content_pixel_bounds(path)
    crop = value[top : top + height, left : left + width].permute(2, 0, 1).float()
    pooled = F.adaptive_avg_pool2d(crop, output_size).permute(1, 2, 0)
    return pooled.reshape(output_size[0] * output_size[1], value.shape[-1])


def build_physical_geometry_targets(
    depth: torch.Tensor,
    depth_confidence: torch.Tensor,
    points: torch.Tensor,
    point_confidence: torch.Tensor,
    *,
    path_batches: list[list[Path]],
    output_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return relative geometry ``[B,180,3]`` and confidence/validity."""

    assert depth.ndim == points.ndim == 5
    assert depth.shape[:4] == points.shape[:4]
    assert depth.shape[-1] == 1 and points.shape[-1] == 3
    batch, views = depth.shape[:2]
    assert len(path_batches) == batch
    all_depth, all_points, all_confidence = [], [], []
    for sample_index in range(batch):
        assert len(path_batches[sample_index]) == views
        sample_depth, sample_points, sample_confidence = [], [], []
        for view_index in range(views):
            path = path_batches[sample_index][view_index]
            sample_depth.append(
                pool_dense_map(depth[sample_index, view_index], path=path, output_size=output_size)
            )
            sample_points.append(
                pool_dense_map(points[sample_index, view_index], path=path, output_size=output_size)
            )
            depth_conf = pool_dense_map(
                depth_confidence[sample_index, view_index], path=path, output_size=output_size
            )
            point_conf = pool_dense_map(
                point_confidence[sample_index, view_index], path=path, output_size=output_size
            )
            sample_confidence.append(torch.minimum(depth_conf, point_conf))
        all_depth.append(torch.cat(sample_depth, dim=0))
        all_points.append(torch.cat(sample_points, dim=0))
        all_confidence.append(torch.cat(sample_confidence, dim=0))
    pooled_depth = torch.stack(all_depth)
    pooled_points = torch.stack(all_points)
    raw_confidence = torch.stack(all_confidence).squeeze(-1)
    point_z = pooled_points[..., 2]
    valid = (
        pooled_depth[..., 0].gt(1e-5)
        & point_z.abs().gt(1e-5)
        & torch.isfinite(pooled_depth[..., 0])
        & torch.isfinite(pooled_points).all(-1)
        & torch.isfinite(raw_confidence)
    )
    median_depth = []
    for sample_index in range(batch):
        values = pooled_depth[sample_index, :, 0][valid[sample_index]]
        if values.numel() == 0:
            raise RuntimeError("VGGT geometry heads returned no valid positive depth")
        median_depth.append(values.median())
    median_depth = torch.stack(median_depth).unsqueeze(1)
    safe_z = torch.where(point_z.abs().gt(1e-5), point_z, torch.ones_like(point_z))
    target = torch.stack(
        (
            pooled_points[..., 0] / safe_z,
            pooled_points[..., 1] / safe_z,
            torch.log(pooled_depth[..., 0].clamp_min(1e-5) / median_depth),
        ),
        dim=-1,
    )
    confidence_scale = raw_confidence.abs().median(dim=1, keepdim=True).values.clamp_min(1e-6)
    confidence = (raw_confidence.clamp_min(0) / confidence_scale).clamp(max=1.0)
    confidence = confidence * valid.float()
    if not (confidence > 0).any(dim=1).all():
        raise RuntimeError("VGGT geometry confidence has no positive valid weight")
    target = target.masked_fill(~valid.unsqueeze(-1), 0.0)
    expected_slots = views * output_size[0] * output_size[1]
    assert target.shape == (batch, expected_slots, 3)
    assert confidence.shape == valid.shape == target.shape[:2]
    return target, confidence, valid


class SlotStatistics:
    def __init__(self, query_count: int, feature_dim: int) -> None:
        self.count = torch.zeros(query_count, dtype=torch.float64)
        self.total = torch.zeros(query_count, feature_dim, dtype=torch.float64)
        self.square_total = torch.zeros(query_count, feature_dim, dtype=torch.float64)

    def update(self, features: torch.Tensor, mask: torch.Tensor) -> None:
        values = F.layer_norm(
            features.detach().float(), (features.shape[-1],)
        ).cpu().double()
        valid = mask.detach().cpu().bool()
        assert values.shape[:1] == valid.shape
        self.count += valid.double()
        self.total += values * valid.unsqueeze(-1)
        self.square_total += values.square() * valid.unsqueeze(-1)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"count": self.count, "total": self.total, "square_total": self.square_total}

    def merge(self, state: Mapping[str, torch.Tensor]) -> None:
        self.count += state["count"].double()
        self.total += state["total"].double()
        self.square_total += state["square_total"].double()

    def variance(self) -> torch.Tensor:
        denominator = self.count.clamp_min(1).unsqueeze(-1)
        mean = self.total / denominator
        variance = (self.square_total / denominator - mean.square()).clamp_min(0)
        return variance.mean(dim=-1)

    def mean_and_scale(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized per-slot mean ``[Q,D]`` and RMS scale ``[Q]``."""

        denominator = self.count.clamp_min(1).unsqueeze(-1)
        mean = self.total / denominator
        variance = (self.square_total / denominator - mean.square()).clamp_min(0)
        return mean.float(), variance.mean(dim=-1).sqrt().float()


def write_torch_atomic(path: Path, payload: Mapping[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def deserialize_record(value: bytes) -> dict[str, torch.Tensor]:
    return torch.load(io.BytesIO(value), map_location="cpu", weights_only=True)


def validate_cache(args: argparse.Namespace) -> None:
    from starVLA.cache.navsim_feature_cache import NavsimFeatureCacheReader

    tokens = load_tokens(Path(args.datalist_path), args.max_samples)
    reader = NavsimFeatureCacheReader(
        args.cache_root, components=(COMPONENT,), strict=True
    )
    manifest = reader.manifests[COMPONENT]
    layout = VGGTQueryLayout()
    expected_manifest = {
        "query_count": layout.query_count,
        "feature_dim": layout.teacher_dim,
        "teacher_layer_index": VGGT_TEACHER_LAYER_INDEX,
        "teacher_attention_branch": "global",
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise RuntimeError(
                f"VGGT cache manifest {key} mismatch: "
                f"expected {expected!r}, found {manifest.get(key)!r}"
            )
    statistics_name = manifest.get("slot_statistics_file")
    if not statistics_name:
        raise RuntimeError("VGGT cache manifest has no slot_statistics_file")
    statistics_path = component_dir(args.cache_root, COMPONENT) / str(statistics_name)
    if not statistics_path.is_file():
        raise FileNotFoundError(f"Missing VGGT slot statistics: {statistics_path}")
    if sha256_file(statistics_path) != manifest.get("slot_statistics_sha256"):
        raise RuntimeError("VGGT slot statistics checksum mismatch")
    statistics = torch.load(statistics_path, map_location="cpu", weights_only=True)
    if statistics.get("slot_mean", torch.empty(0)).shape != (
        layout.query_count,
        layout.teacher_dim,
    ):
        raise RuntimeError("VGGT slot mean shape mismatch")
    if statistics.get("slot_scale", torch.empty(0)).shape != (layout.query_count,):
        raise RuntimeError("VGGT slot scale shape mismatch")
    expected_q = int(manifest["query_count"])
    expected_d = int(manifest["feature_dim"])
    for index, token in enumerate(tqdm(tokens, desc="validate VGGT query cache")):
        payload = reader.get(COMPONENT, index, token)
        if payload["features"].shape != (expected_q, expected_d):
            raise RuntimeError(f"Invalid feature shape for {token}: {payload['features'].shape}")
        if payload["valid_mask"].shape != (expected_q,):
            raise RuntimeError(f"Invalid validity shape for {token}")
        expected_spatial = layout.spatial_query_count
        if payload.get("geometry_target", torch.empty(0)).shape != (expected_spatial, 3):
            raise RuntimeError(f"Invalid geometry target shape for {token}")
        if payload.get("geometry_confidence", torch.empty(0)).shape != (expected_spatial,):
            raise RuntimeError(f"Invalid geometry confidence shape for {token}")
        if payload.get("geometry_valid_mask", torch.empty(0)).shape != (expected_spatial,):
            raise RuntimeError(f"Invalid geometry validity shape for {token}")
    print(f"[vggt-query-cache] VALID samples={len(tokens)} manifest={args.cache_root}")


def batches(values: list[int], batch_size: int) -> Iterable[list[int]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def main(args: argparse.Namespace) -> None:
    if args.validate_only:
        validate_cache(args)
        return
    rank, world_size, device = distributed_context()
    if not str(args.vggt_repo).strip():
        raise ValueError("Set --vggt-repo or VGGT_REPO; no repository is downloaded")
    if not str(args.vggt_checkpoint).strip():
        raise ValueError("Set --vggt-checkpoint or VGGT_CHECKPOINT; no weights are downloaded")
    repo_path = Path(args.vggt_repo).expanduser().resolve()
    checkpoint_path = Path(args.vggt_checkpoint).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    sensor_root = Path(args.sensor_root).expanduser().resolve() if args.sensor_root else None
    views = tuple(item.strip() for item in args.views.split(",") if item.strip())
    if len(views) != 3:
        raise ValueError("The current VGGT query contract requires exactly three views")
    tokens = load_tokens(Path(args.datalist_path), args.max_samples)
    model, preprocess = load_local_vggt(repo_path, checkpoint_path, device)
    layout = VGGTQueryLayout(view_count=len(views))
    owned = list(range(rank, len(tokens), world_size))
    statistics = SlotStatistics(layout.query_count, layout.teacher_dim)
    component_path = component_dir(args.cache_root, COMPONENT)
    component_path.mkdir(parents=True, exist_ok=True)
    started = time.time()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    with RankCacheWriter(
        args.cache_root,
        COMPONENT,
        rank,
        int(args.map_size_gb * 1024**3),
        commit_interval=args.commit_interval,
    ) as writer:
        iterator = list(batches(owned, args.batch_size))
        if rank == 0:
            iterator = tqdm(iterator, desc="VGGT query cache")
        for index_batch in iterator:
            pending = []
            image_batches = []
            path_batches = []
            validity_batches = []
            for sample_index in index_batch:
                token = tokens[sample_index]
                existing = writer.transaction.get(token.encode("utf-8"))
                if existing is not None and not args.overwrite:
                    payload = deserialize_record(bytes(existing))
                    if payload["features"].shape != (
                        layout.query_count,
                        layout.teacher_dim,
                    ):
                        raise RuntimeError(
                            "Existing VGGT cache record uses a different teacher contract. "
                            "Select a new NAVSIM_VGGT_CACHE_ROOT for layer11-global V2."
                        )
                    required_geometry = (
                        "geometry_target",
                        "geometry_confidence",
                        "geometry_valid_mask",
                    )
                    if any(key not in payload for key in required_geometry):
                        raise RuntimeError(
                            "Existing record predates V2 physical geometry targets. "
                            "Select a new NAVSIM_VGGT_CACHE_ROOT."
                        )
                    statistics.update(payload["features"], payload["valid_mask"])
                    writer.skipped += 1
                    continue
                with metadata_path(data_root, args.split, token).open("rb") as stream:
                    raw = pickle.load(stream)
                paths = scene_image_paths(raw, views, args.frame_index, sensor_root)
                images = preprocess([str(path) for path in paths], mode="pad")
                assert images.shape == (len(views), 3, 518, 518)
                image_batches.append(images)
                path_batches.append(paths)
                validity_batches.append(
                    torch.stack([patch_validity_for_image(path) for path in paths])
                )
                pending.append((sample_index, token))
            if not pending:
                continue
            images = torch.stack(image_batches).to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            validity = torch.stack(validity_batches).to(device=device, non_blocking=True)
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=device.type == "cuda",
            ):
                aggregated, patch_start_idx = model.aggregator(images)
            if int(patch_start_idx) != VGGT_PATCH_START_INDEX:
                raise RuntimeError(
                    f"VGGT patch_start_idx changed: expected {VGGT_PATCH_START_INDEX}, "
                    f"found {patch_start_idx}"
                )
            layer_tokens = select_vggt_global_teacher_layer(
                aggregated,
                layer_index=VGGT_TEACHER_LAYER_INDEX,
                branch_dim=VGGT_TEACHER_BRANCH_DIM,
            )
            targets, masks = extract_vggt_layer11_memory_targets(
                layer_tokens,
                spatial_validity=validity,
                patch_start_idx=int(patch_start_idx),
                patch_grid_size=37,
                output_size=(layout.spatial_rows, layout.spatial_cols),
                minimum_valid_ratio=args.minimum_valid_ratio,
            )
            assert targets.shape == (len(pending), layout.query_count, layout.teacher_dim)
            head_inputs = [value.float() if value is not None else None for value in aggregated]
            with torch.inference_mode():
                depth, depth_confidence = model.depth_head(
                    head_inputs, images=images, patch_start_idx=patch_start_idx
                )
                points, point_confidence = model.point_head(
                    head_inputs, images=images, patch_start_idx=patch_start_idx
                )
            geometry_target, geometry_confidence, geometry_valid = (
                build_physical_geometry_targets(
                    depth,
                    depth_confidence,
                    points,
                    point_confidence,
                    path_batches=path_batches,
                    output_size=(layout.spatial_rows, layout.spatial_cols),
                )
            )
            for batch_index, ((_, token), features, valid_mask) in enumerate(
                zip(pending, targets, masks)
            ):
                payload = {
                    "features": features.detach().to(dtype=torch.bfloat16).cpu().contiguous(),
                    "valid_mask": valid_mask.detach().cpu().bool().contiguous(),
                    "geometry_target": geometry_target[batch_index].detach().cpu().float().contiguous(),
                    "geometry_confidence": geometry_confidence[batch_index].detach().cpu().float().contiguous(),
                    "geometry_valid_mask": geometry_valid[batch_index].detach().cpu().bool().contiguous(),
                }
                writer.put(token, payload, overwrite=args.overwrite)
                statistics.update(payload["features"], payload["valid_mask"])

        completion = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "component": COMPONENT,
            "rank": rank,
            "world_size": world_size,
            "sample_count": len(tokens),
            "owned_samples": len(owned),
            "written": writer.written,
            "skipped": writer.skipped,
            "elapsed_seconds": time.time() - started,
        }

    stats_path = component_path / f"rank_{rank:05d}.stats.pt"
    write_torch_atomic(stats_path, statistics.state_dict())
    write_rank_completion(args.cache_root, COMPONENT, rank, completion)
    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        merged = SlotStatistics(layout.query_count, layout.teacher_dim)
        completions = []
        for owner in range(world_size):
            completion_path = component_path / f"rank_{owner:05d}.complete.json"
            statistics_path = component_path / f"rank_{owner:05d}.stats.pt"
            if not completion_path.is_file() or not statistics_path.is_file():
                raise FileNotFoundError(f"Missing completion/statistics for rank {owner}")
            completions.append(json.loads(completion_path.read_text(encoding="utf-8")))
            merged.merge(torch.load(statistics_path, map_location="cpu", weights_only=True))
        variance = merged.variance()
        slot_mean, slot_scale = merged.mean_and_scale()
        active = (merged.count > 0) & (variance >= args.minimum_slot_variance)
        if not active.any():
            raise RuntimeError(
                "Every VGGT target slot was filtered as low variance; lower "
                "--minimum-slot-variance or inspect the teacher cache."
            )
        extraction_config = {
            "view_order": list(views),
            "frame_index": args.frame_index,
            "teacher_layer": "aggregator.global_blocks[11]_before_dpt",
            "teacher_layer_index": VGGT_TEACHER_LAYER_INDEX,
            "teacher_attention_branch": "global",
            "patch_start_idx": VGGT_PATCH_START_INDEX,
            "source_patch_grid": [37, 37],
            "pooled_spatial_grid": [layout.spatial_rows, layout.spatial_cols],
            "include_special_tokens": True,
            "padding_policy": "crop_content_before_pool",
            "preprocess": {"mode": "pad", "target_size": 518},
            "minimum_valid_ratio": args.minimum_valid_ratio,
            "minimum_slot_variance": args.minimum_slot_variance,
            "geometry_target": ["x_over_z", "y_over_z", "log_depth_over_scene_median"],
            "geometry_confidence": "min(depth_head,point_head)_per_scene_normalized",
        }
        extraction_config_sha256 = hashlib.sha256(
            json.dumps(extraction_config, sort_keys=True).encode("utf-8")
        ).hexdigest()
        slot_statistics_path = component_path / "slot_statistics.pt"
        write_torch_atomic(
            slot_statistics_path,
            {
                "slot_mean": slot_mean,
                "slot_scale": slot_scale,
                "slot_variance": variance.float(),
                "slot_valid_counts": merged.count.long(),
            },
        )
        manifest = {
            "component": COMPONENT,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "world_size": world_size,
            "sample_count": len(tokens),
            "query_count": layout.query_count,
            "feature_dim": layout.teacher_dim,
            **extraction_config,
            "extraction_config_sha256": extraction_config_sha256,
            "slot_variance": variance.tolist(),
            "slot_valid_counts": merged.count.tolist(),
            "active_slot_mask": active.tolist(),
            "slot_statistics_file": slot_statistics_path.name,
            "slot_statistics_sha256": sha256_file(slot_statistics_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "vggt_repo_commit": git_revision(repo_path),
            "project_commit": git_revision(REPO_ROOT),
            "extractor_sha256": sha256_file(Path(__file__).resolve()),
            "datalist_sha256": sha256_file(Path(args.datalist_path)),
            # Local paths are diagnostics, never cache identity.
            "diagnostic_paths": {
                "vggt_repo": str(repo_path),
                "vggt_checkpoint": str(checkpoint_path),
                "data_root": str(data_root),
                "sensor_root": str(sensor_root) if sensor_root else None,
            },
            "payload_contract": {
                "features": (
                    f"bfloat16[{layout.query_count},{layout.teacher_dim}]"
                ),
                "valid_mask": f"bool[{layout.query_count}]",
                "geometry_target": f"float32[{layout.spatial_query_count},3]",
                "geometry_confidence": f"float32[{layout.spatial_query_count}]",
                "geometry_valid_mask": f"bool[{layout.spatial_query_count}]",
            },
            "rank_completions": completions,
        }
        write_manifest(args.cache_root, COMPONENT, manifest)
        print(
            f"[vggt-query-cache] COMPLETE samples={len(tokens)} active_slots="
            f"{int(active.sum())}/{layout.query_count} manifest={component_path / 'manifest.json'}"
        )

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datalist-path", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--sensor-root", default=None)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--vggt-repo", default=os.environ.get("VGGT_REPO", ""))
    parser.add_argument("--vggt-checkpoint", default=os.environ.get("VGGT_CHECKPOINT", ""))
    parser.add_argument("--views", default=",".join(DEFAULT_VIEWS))
    parser.add_argument("--frame-index", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--map-size-gb", type=int, default=64)
    parser.add_argument("--commit-interval", type=int, default=8)
    parser.add_argument("--minimum-valid-ratio", type=float, default=0.25)
    parser.add_argument("--minimum-slot-variance", type=float, default=1e-6)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
