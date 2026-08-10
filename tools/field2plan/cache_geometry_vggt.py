#!/usr/bin/env python3
"""Generate a strict metricized VGGT geometry cache offline.

The official VGGT model predicts scale-ambiguous depth.  For each simultaneous
NAVSIM camera triplet, this tool estimates metric scale from calibrated physical
camera baselines.  It never downloads assets and never runs inside training.
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
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from starVLA.dataloader.field2plan_cache import (
    GeometryCacheReader,
    atomic_write_json,
    atomic_write_npz_compressed,
    hash_tokens,
    sha256_file,
)
from starVLA.model.modules.field2plan.geometry_teachers import (
    DA3LegacyDepthAdapter,
    DEFAULT_GEOMETRY_VIEWS,
    OfficialVGGTMetricDepthAdapter,
)


@dataclass(frozen=True)
class VGGTCacheInputs:
    """One cache request with images ``V`` and rig centers ``[V,3]``."""

    token: str
    image_paths: Tuple[Path, ...]
    source_image_hw: np.ndarray
    known_camera_centers_m: np.ndarray


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
        raise ValueError(f"invalid datalist JSON: {datalist_path}") from error
    if not isinstance(payload, list) or not all(
        isinstance(token, str) and token for token in payload
    ):
        raise ValueError("VGGT datalist must be a JSON list of non-empty tokens")
    tokens = payload[:max_samples] if max_samples > 0 else payload
    if not tokens:
        raise ValueError("VGGT datalist selected zero tokens")
    if len(tokens) != len(set(tokens)):
        raise ValueError("VGGT datalist contains duplicate tokens")
    return tokens


def _resolve_navsim_image_path(
    embedded_path: str | os.PathLike,
    *,
    runtime_raw_root: Path,
    trainval_sensor_root: Path | None,
) -> Path:
    """Resolve preprocessing-time absolute paths without a silent fallback."""

    path = Path(embedded_path)
    if path.is_file():
        return path
    marker = f"{os.sep}navsim_dataset_raw{os.sep}"
    encoded = os.fspath(path)
    candidates = []
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
    rendered = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"NAVSIM image not found: embedded={path}; resolved_candidates=[{rendered}]"
    )


def load_navsim_vggt_inputs(
    *,
    token: str,
    meta_path: Path,
    view_names: Sequence[str],
    frame_index: int,
    runtime_raw_root: Path,
    trainval_sensor_root: Path | None,
) -> VGGTCacheInputs:
    """Load ordered current-view paths and calibrated centers for one token."""

    if not meta_path.is_file():
        raise FileNotFoundError(f"NAVSIM metadata not found: {meta_path}")
    try:
        with meta_path.open("rb") as stream:
            metadata = pickle.load(stream)
    except (OSError, pickle.UnpicklingError, EOFError) as error:
        raise ValueError(f"corrupt NAVSIM metadata: {meta_path}") from error
    if not isinstance(metadata, dict):
        raise ValueError(f"NAVSIM metadata must be a mapping: {meta_path}")

    paths = []
    centers = []
    source_hw = []
    try:
        cameras = metadata["glo_images"]
        for view in view_names:
            camera = cameras[view]
            path = _resolve_navsim_image_path(
                camera["image_paths"][frame_index],
                runtime_raw_root=runtime_raw_root,
                trainval_sensor_root=trainval_sensor_root,
            )
            center = np.asarray(
                camera["sensor2lidar_translations"][frame_index],
                dtype=np.float32,
            )
            if center.shape != (3,) or not np.isfinite(center).all():
                raise ValueError(f"invalid camera center for {view}")
            try:
                with Image.open(path) as image:
                    width, height = image.size
            except (OSError, UnidentifiedImageError) as error:
                raise ValueError(f"invalid NAVSIM image: {path}") from error
            if min(height, width) <= 0:
                raise ValueError(f"invalid NAVSIM image dimensions: {path}")
            paths.append(path)
            centers.append(center)
            source_hw.append((height, width))
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(
            f"invalid NAVSIM camera contract for token {token!r} frame {frame_index}"
        ) from error
    return VGGTCacheInputs(
        token=str(token),
        image_paths=tuple(paths),
        source_image_hw=np.asarray(source_hw, dtype=np.int64),
        known_camera_centers_m=np.asarray(centers, dtype=np.float32),
    )


def _hash_source_index(
    tokens: Iterable[str], metadata_checksums: Dict[str, str]
) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        checksum = metadata_checksums[token]
        encoded = token.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big", signed=False))
        digest.update(encoded)
        digest.update(bytes.fromhex(checksum))
    return digest.hexdigest()


def build_vggt_manifest(
    *,
    datalist_path: Path,
    split: str,
    tokens: Sequence[str],
    view_names: Sequence[str],
    output_hw: Sequence[int],
    git_commit: str,
    vggt_repo: Path,
    vggt_repo_commit: str,
    checkpoint: Path,
    checkpoint_revision: str,
    metadata_checksums: Dict[str, str],
    frame_index: int,
    metricization: str,
    scale_anchor_root: Path | None,
) -> dict:
    """Build a complete manifest bound to data, code, repository and weights."""

    output_hw = tuple(int(value) for value in output_hw)
    if len(output_hw) != 2 or min(output_hw) <= 0:
        raise ValueError("output_hw must be positive [height,width]")
    if set(metadata_checksums) != set(tokens):
        raise ValueError("metadata checksums must cover exactly the selected tokens")
    checkpoint_path = (
        checkpoint / "model.safetensors" if checkpoint.is_dir() else checkpoint
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"VGGT checkpoint not found: {checkpoint_path}")
    views = len(view_names)
    depth_shape = [views, *output_hw]
    view_hw_shape = [views, 2]
    if metricization not in {"da3_scale_anchor", "camera_rig"}:
        raise ValueError("unsupported VGGT metricization")
    if metricization == "da3_scale_anchor" and scale_anchor_root is None:
        raise ValueError("da3_scale_anchor requires scale_anchor_root")
    teacher_version = OfficialVGGTMetricDepthAdapter.version
    teacher = {
        "name": OfficialVGGTMetricDepthAdapter.name,
        "version": teacher_version,
        "source_index_sha256": _hash_source_index(tokens, metadata_checksums),
        "repo": str(vggt_repo.resolve()),
        "repo_commit": str(vggt_repo_commit),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_revision": str(checkpoint_revision),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    if metricization == "da3_scale_anchor":
        teacher["scale_anchor"] = {
            "name": DA3LegacyDepthAdapter.name,
            "version": DA3LegacyDepthAdapter.version,
            "root": str(scale_anchor_root.resolve()),
        }
        metricization_name = "robust_log_median_da3_metric_depth_ratio"
        metric_reference = "current_frame_da3_metric_depth_m"
    else:
        teacher["version"] = "facebook_vggt_1b_r860abec_rig_metric_experimental_v1"
        metricization_name = "median_known_to_predicted_camera_baseline_ratio"
        metric_reference = "sensor2lidar_translation_pairwise_baselines_m"
    return {
        "schema_version": 1,
        "cache_type": "geometry_teacher",
        "status": "complete",
        "teacher": teacher,
        "generator": {
            "git_commit": str(git_commit),
            "tool": "tools/field2plan/cache_geometry_vggt.py",
        },
        "splits": {
            split: {
                "entry_count": len(tokens),
                "tokens_sha256": hash_tokens(tokens),
                "datalist_sha256": sha256_file(datalist_path),
            }
        },
        "tensor_schema": {
            "view_names": list(view_names),
            "depth_m": {"dtype": "float32", "shape": depth_shape},
            "confidence": {"dtype": "float32", "shape": depth_shape},
            "valid_mask": {"dtype": "bool", "shape": depth_shape},
            "source_image_hw": {"dtype": "int64", "shape": view_hw_shape},
            "depth_hw": {"dtype": "int64", "shape": view_hw_shape},
            "resize_scale_xy": {"dtype": "float32", "shape": view_hw_shape},
        },
        "coordinates": {
            "frame": "camera_optical_z_depth_m",
            "frame_index": int(frame_index),
            "metricization": metricization_name,
            "metric_reference": metric_reference,
            "confidence_source": "teacher_confidence",
            "confidence_transform": "max(raw,0)/(1+max(raw,0))",
            "resize_policy": "official_crop_then_bilinear_output_hw",
        },
    }


def _cache_fingerprint(
    *,
    datalist: Path,
    checkpoint: Path,
    checkpoint_revision: str,
    repo_commit: str,
    output_hw: Sequence[int],
    frame_index: int,
    metricization: str,
    scale_anchor_root: Path | None,
) -> str:
    checkpoint_path = (
        checkpoint / "model.safetensors" if checkpoint.is_dir() else checkpoint
    )
    stat = checkpoint_path.stat()
    payload = {
        "datalist_sha256": sha256_file(datalist),
        "checkpoint_revision": checkpoint_revision,
        "checkpoint_size": stat.st_size,
        "checkpoint_mtime_ns": stat.st_mtime_ns,
        "repo_commit": repo_commit,
        "output_hw": list(output_hw),
        "frame_index": frame_index,
        "metricization": metricization,
        "scale_anchor_root": (
            str(scale_anchor_root.resolve()) if scale_anchor_root is not None else None
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _existing_entry_is_valid(
    path: Path,
    *,
    token: str,
    expected_shape: tuple[int, int, int],
    cache_fingerprint: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as payload:
            required = {
                "token",
                "cache_fingerprint",
                "depth_m",
                "confidence",
                "valid_mask",
                "source_image_hw",
                "depth_hw",
                "resize_scale_xy",
            }
            return (
                required.issubset(payload.files)
                and str(payload["token"].item()) == token
                and str(payload["cache_fingerprint"].item()) == cache_fingerprint
                and payload["depth_m"].shape == expected_shape
                and payload["depth_m"].dtype == np.float32
                and payload["confidence"].shape == expected_shape
                and payload["confidence"].dtype == np.float32
                and payload["valid_mask"].shape == expected_shape
                and payload["valid_mask"].dtype == np.bool_
            )
    except (OSError, ValueError):
        return False


def validate_vggt_cache(
    *,
    cache_root: Path,
    split: str,
    tokens: Sequence[str],
    datalist: Path,
    cache_fingerprint: str | None = None,
    rank: int = 0,
    world_size: int = 1,
) -> dict:
    """Validate one deterministic cache shard; all ranks cover the full cache."""

    reader = GeometryCacheReader(cache_root, split)
    reader.validate_dataset_binding(tokens, os.fspath(datalist))
    validated = 0
    for token in tokens[rank::world_size]:
        reader.load(token)
        if cache_fingerprint is not None:
            path = cache_root / split / f"{token}.npz"
            with np.load(path, allow_pickle=False) as payload:
                if "cache_fingerprint" not in payload.files or str(
                    payload["cache_fingerprint"].item()
                ) != cache_fingerprint:
                    raise ValueError(f"VGGT cache fingerprint mismatch: {path}")
        validated += 1
    return {"split": split, "rank": rank, "validated_entries": validated}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datalist", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--meta-root", type=Path, required=True)
    parser.add_argument("--runtime-raw-root", type=Path, required=True)
    parser.add_argument("--trainval-sensor-root", type=Path)
    parser.add_argument("--vggt-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument(
        "--metricization",
        choices=("da3_scale_anchor", "camera_rig"),
        default="da3_scale_anchor",
    )
    parser.add_argument("--da3-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-height", type=int, default=144)
    parser.add_argument("--output-width", type=int, default=256)
    parser.add_argument("--frame-index", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--coordination-timeout-s", type=int, default=7200)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if rank < 0 or world_size < 1 or rank >= world_size:
        raise ValueError("invalid distributed rank/world size")
    if args.max_samples < 0:
        raise ValueError("--max-samples cannot be negative")
    if args.coordination_timeout_s < 1:
        raise ValueError("--coordination-timeout-s must be positive")
    for path, name in (
        (args.datalist, "datalist"),
        (args.meta_root, "metadata root"),
        (args.runtime_raw_root, "runtime raw root"),
        (args.vggt_repo, "VGGT repository"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")
    if args.metricization == "da3_scale_anchor":
        if args.da3_root is None:
            raise ValueError("--da3-root is required for da3_scale_anchor")
        if not args.da3_root.is_dir():
            raise FileNotFoundError(f"DA3 scale-anchor root not found: {args.da3_root}")
    checkpoint = (
        args.checkpoint / "model.safetensors"
        if args.checkpoint.is_dir()
        else args.checkpoint
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"VGGT checkpoint not found: {checkpoint}")
    tokens = _load_tokens(args.datalist, args.max_samples)
    output_hw = (args.output_height, args.output_width)
    if min(output_hw) <= 0:
        raise ValueError("VGGT output dimensions must be positive")
    repo_commit = _git_commit(args.vggt_repo)
    fingerprint = _cache_fingerprint(
        datalist=args.datalist,
        checkpoint=checkpoint,
        checkpoint_revision=args.checkpoint_revision,
        repo_commit=repo_commit,
        output_hw=output_hw,
        frame_index=args.frame_index,
        metricization=args.metricization,
        scale_anchor_root=args.da3_root,
    )

    if args.validate_only:
        result = validate_vggt_cache(
            cache_root=args.output_dir,
            split=args.split,
            tokens=tokens,
            datalist=args.datalist,
            cache_fingerprint=fingerprint,
            rank=rank,
            world_size=world_size,
        )
        print(json.dumps(result, sort_keys=True))
        return

    output_split = args.output_dir / args.split
    output_split.mkdir(parents=True, exist_ok=True)
    expected_shape = (len(DEFAULT_GEOMETRY_VIEWS), *output_hw)
    device = args.device
    if device == "cuda" and world_size > 1:
        device = f"cuda:{local_rank}"
    adapter = OfficialVGGTMetricDepthAdapter(
        local_repo=args.vggt_repo,
        checkpoint=checkpoint,
        device=device,
        output_hw=output_hw,
        frame_index=args.frame_index,
        metricization=args.metricization,
    )
    da3_adapter = (
        DA3LegacyDepthAdapter(
            args.da3_root,
            view_names=DEFAULT_GEOMETRY_VIEWS,
            frame_index=args.frame_index,
        )
        if args.metricization == "da3_scale_anchor"
        else None
    )
    metadata_checksums: Dict[str, str] = {}
    generated = 0
    resumed = 0
    scales = []
    shard_tokens = tokens[rank::world_size]
    for local_index, token in enumerate(shard_tokens, start=1):
        meta_path = args.meta_root / f"{token}.pkl"
        metadata_checksum = sha256_file(meta_path)
        cache_inputs = load_navsim_vggt_inputs(
            token=token,
            meta_path=meta_path,
            view_names=DEFAULT_GEOMETRY_VIEWS,
            frame_index=args.frame_index,
            runtime_raw_root=args.runtime_raw_root,
            trainval_sensor_root=args.trainval_sensor_root,
        )
        output_path = output_split / f"{token}.npz"
        metric_depth_reference_m = None
        if da3_adapter is not None:
            da3_path = da3_adapter.cache_path(token)
            da3_checksum = sha256_file(da3_path)
            checksum_digest = hashlib.sha256()
            checksum_digest.update(bytes.fromhex(metadata_checksum))
            checksum_digest.update(bytes.fromhex(da3_checksum))
            metadata_checksums[token] = checksum_digest.hexdigest()
            metric_depth_reference_m = da3_adapter.load_cached(
                token, cache_inputs.source_image_hw
            ).depth_m
        else:
            metadata_checksums[token] = metadata_checksum
        if not args.overwrite and _existing_entry_is_valid(
            output_path,
            token=token,
            expected_shape=expected_shape,
            cache_fingerprint=fingerprint,
        ):
            resumed += 1
        else:
            sample = adapter.infer(
                token=token,
                image_paths=cache_inputs.image_paths,
                source_image_hw=cache_inputs.source_image_hw,
                known_camera_centers_m=cache_inputs.known_camera_centers_m,
                metric_depth_reference_m=metric_depth_reference_m,
            )
            atomic_write_npz_compressed(
                output_path,
                token=np.asarray(token),
                cache_fingerprint=np.asarray(fingerprint),
                depth_m=sample.depth_m,
                confidence=sample.confidence,
                valid_mask=sample.valid_mask,
                source_image_hw=sample.source_image_hw,
                depth_hw=sample.depth_hw,
                resize_scale_xy=sample.resize_scale_xy,
            )
            generated += 1
            scales.append(float(sample.metadata["metric_scale"]))
        if local_index % 20 == 0 or local_index == len(shard_tokens):
            print(
                f"[vggt-cache rank={rank}] {local_index}/{len(shard_tokens)} "
                f"generated={generated} resumed={resumed}",
                flush=True,
            )

    shard_payload = {
        "fingerprint": fingerprint,
        "rank": rank,
        "world_size": world_size,
        "generated": generated,
        "resumed": resumed,
        "metadata_checksums": metadata_checksums,
        "metric_scale_count": len(scales),
        "metric_scale_sum": float(sum(scales)),
        "metric_scale_min": float(min(scales)) if scales else None,
        "metric_scale_max": float(max(scales)) if scales else None,
    }
    shard_dir = args.output_dir / ".shards"
    shard_path = shard_dir / f"{fingerprint}-rank{rank:05d}.json"
    atomic_write_json(shard_path, shard_payload)
    if rank != 0:
        return

    deadline = time.monotonic() + args.coordination_timeout_s
    shard_paths = [
        shard_dir / f"{fingerprint}-rank{index:05d}.json"
        for index in range(world_size)
    ]
    while not all(path.is_file() for path in shard_paths):
        if time.monotonic() >= deadline:
            missing = [str(path) for path in shard_paths if not path.is_file()]
            raise TimeoutError(f"timed out waiting for VGGT cache shards: {missing}")
        time.sleep(5)
    combined_checksums: Dict[str, str] = {}
    total_generated = 0
    total_resumed = 0
    for path in shard_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("fingerprint") != fingerprint:
            raise ValueError(f"VGGT shard fingerprint mismatch: {path}")
        combined_checksums.update(payload["metadata_checksums"])
        total_generated += int(payload["generated"])
        total_resumed += int(payload["resumed"])
    if set(combined_checksums) != set(tokens):
        raise ValueError("VGGT shard summaries do not cover the selected token set")
    manifest = build_vggt_manifest(
        datalist_path=args.datalist,
        split=args.split,
        tokens=tokens,
        view_names=DEFAULT_GEOMETRY_VIEWS,
        output_hw=output_hw,
        git_commit=_git_commit(PROJECT_ROOT),
        vggt_repo=args.vggt_repo,
        vggt_repo_commit=repo_commit,
        checkpoint=checkpoint,
        checkpoint_revision=args.checkpoint_revision,
        metadata_checksums=combined_checksums,
        frame_index=args.frame_index,
        metricization=args.metricization,
        scale_anchor_root=args.da3_root,
    )
    manifest["generator"]["world_size"] = world_size
    manifest["generator"]["cache_fingerprint"] = fingerprint
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
        )
    )


if __name__ == "__main__":
    main()
