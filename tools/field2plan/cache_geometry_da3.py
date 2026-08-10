#!/usr/bin/env python3
"""Convert existing DA3 metric-depth files into a strict geometry cache.

This is a CPU-only, offline conversion task. It never imports or executes the
vendored Depth-Anything-3 model and is safe to run before formal training.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np

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
)


def _git_commit(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
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
        raise ValueError("geometry datalist must be a JSON list of non-empty tokens")
    tokens = payload[:max_samples] if max_samples > 0 else payload
    if not tokens:
        raise ValueError("geometry datalist selected zero tokens")
    if len(set(tokens)) != len(tokens):
        raise ValueError("geometry datalist contains duplicate tokens")
    return tokens


def _hash_source_index(
    source_root: Path,
    tokens: Iterable[str],
    source_checksums: Dict[str, str] | None = None,
) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        checksum = (
            source_checksums[token]
            if source_checksums is not None
            else sha256_file(source_root / f"{token}.pkl-depth.pkl")
        )
        encoded = token.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big", signed=False))
        digest.update(encoded)
        digest.update(bytes.fromhex(checksum))
    return digest.hexdigest()


def build_geometry_manifest(
    *,
    source_root: Path,
    datalist_path: Path,
    split: str,
    tokens: Sequence[str],
    view_names: Sequence[str],
    depth_shape: Sequence[int],
    git_commit: str,
    source_checksums: Dict[str, str] | None = None,
    source_image_hw: Tuple[int, int] = (1080, 1920),
    frame_index: int = 3,
) -> dict:
    """Build the complete, content-bound DA3 geometry manifest."""

    depth_shape = tuple(int(value) for value in depth_shape)
    if len(depth_shape) != 3 or depth_shape[0] != len(view_names):
        raise ValueError("depth_shape must be [V,Hd,Wd] and match view_names")
    views = len(view_names)
    source_hw = [views, 2]
    return {
        "schema_version": 1,
        "cache_type": "geometry_teacher",
        "status": "complete",
        "teacher": {
            "name": DA3LegacyDepthAdapter.name,
            "version": DA3LegacyDepthAdapter.version,
            "source_index_sha256": _hash_source_index(
                source_root, tokens, source_checksums
            ),
        },
        "generator": {
            "git_commit": git_commit,
            "tool": "tools/field2plan/cache_geometry_da3.py",
        },
        "source": {
            "legacy_meta_root": str(source_root.resolve()),
            "source_image_hw": list(source_image_hw),
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
            "depth_m": {"dtype": "float32", "shape": list(depth_shape)},
            "confidence": {"dtype": "float32", "shape": list(depth_shape)},
            "valid_mask": {"dtype": "bool", "shape": list(depth_shape)},
            "source_image_hw": {"dtype": "int64", "shape": source_hw},
            "depth_hw": {"dtype": "int64", "shape": source_hw},
            "resize_scale_xy": {"dtype": "float32", "shape": source_hw},
        },
        "coordinates": {
            "frame": "camera_optical_z_depth_m",
            "frame_index": int(frame_index),
            "confidence_source": "finite_positive_validity",
            "resize_policy": "legacy_da3_process_res_252",
        },
    }


def _existing_entry_is_valid(
    path: Path,
    token: str,
    view_count: int,
) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as payload:
            required = {
                "token",
                "depth_m",
                "confidence",
                "valid_mask",
                "source_image_hw",
                "depth_hw",
                "resize_scale_xy",
            }
            if not required.issubset(payload.files):
                return False
            if str(payload["token"].item()) != token:
                return False
            depth = payload["depth_m"]
            return (
                depth.ndim == 3
                and depth.shape[0] == view_count
                and depth.dtype == np.float32
                and payload["confidence"].shape == depth.shape
                and payload["confidence"].dtype == np.float32
                and payload["valid_mask"].shape == depth.shape
                and payload["valid_mask"].dtype == np.bool_
            )
    except (OSError, ValueError):
        return False


def _convert_one(spec: tuple) -> tuple[str, str, tuple[int, int, int], str]:
    (
        token,
        source_root,
        output_root,
        split,
        view_names,
        frame_index,
        source_image_hw,
        overwrite,
    ) = spec
    source_root = Path(source_root)
    output_path = Path(output_root) / split / f"{token}.npz"
    source_path = source_root / f"{token}.pkl-depth.pkl"
    if not source_path.is_file():
        raise FileNotFoundError(f"DA3 depth cache not found: {source_path}")
    source_checksum = sha256_file(source_path)
    if not overwrite and _existing_entry_is_valid(output_path, token, len(view_names)):
        with np.load(output_path, allow_pickle=False) as payload:
            shape = tuple(int(value) for value in payload["depth_m"].shape)
        return token, source_checksum, shape, "resumed"

    adapter = DA3LegacyDepthAdapter(
        source_root, view_names=view_names, frame_index=frame_index
    )
    source_hw = np.repeat(
        np.asarray(source_image_hw, dtype=np.int64)[None], len(view_names), axis=0
    )
    sample = adapter.load_cached(token, source_hw)
    atomic_write_npz_compressed(
        output_path,
        token=np.asarray(token),
        depth_m=sample.depth_m,
        confidence=sample.confidence,
        valid_mask=sample.valid_mask,
        source_image_hw=sample.source_image_hw,
        depth_hw=sample.depth_hw,
        resize_scale_xy=sample.resize_scale_xy,
    )
    return token, source_checksum, sample.depth_m.shape, "generated"


def validate_geometry_cache(
    cache_root: Path,
    split: str,
    tokens: Sequence[str],
) -> dict:
    """Validate the manifest binding and every geometry entry."""

    reader = GeometryCacheReader(str(cache_root), split)
    split_metadata = reader.manifest["splits"][split]
    if split_metadata["entry_count"] != len(tokens):
        raise ValueError("geometry entry_count does not match requested token count")
    if split_metadata["tokens_sha256"] != hash_tokens(tokens):
        raise ValueError("geometry token list checksum mismatch")
    for token in tokens:
        reader.load(token)
    return {"split": split, "validated_entries": len(tokens)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--datalist", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, min(112, (os.cpu_count() or 1) - 8)))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--source-height", type=int, default=1080)
    parser.add_argument("--source-width", type=int, default=1920)
    parser.add_argument("--frame-index", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.max_samples < 0:
        raise ValueError("--max-samples cannot be negative")
    if args.source_height < 1 or args.source_width < 1:
        raise ValueError("source image dimensions must be positive")
    if not args.source_root.is_dir():
        raise FileNotFoundError(f"DA3 source root not found: {args.source_root}")
    if not args.datalist.is_file():
        raise FileNotFoundError(f"datalist not found: {args.datalist}")
    tokens = _load_tokens(args.datalist, args.max_samples)
    if args.validate_only:
        print(json.dumps(validate_geometry_cache(args.output_dir, args.split, tokens)))
        return

    output_split = args.output_dir / args.split
    output_split.mkdir(parents=True, exist_ok=True)
    specs = [
        (
            token,
            str(args.source_root),
            str(args.output_dir),
            args.split,
            DEFAULT_GEOMETRY_VIEWS,
            args.frame_index,
            (args.source_height, args.source_width),
            args.overwrite,
        )
        for token in tokens
    ]
    checksums: Dict[str, str] = {}
    shapes = set()
    counts = {"generated": 0, "resumed": 0}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, (token, checksum, shape, status) in enumerate(
            executor.map(_convert_one, specs, chunksize=8), start=1
        ):
            checksums[token] = checksum
            shapes.add(shape)
            counts[status] += 1
            if index == 1 or index % 100 == 0 or index == len(tokens):
                print(
                    f"[geometry-cache] {index}/{len(tokens)} "
                    f"generated={counts['generated']} resumed={counts['resumed']}",
                    flush=True,
                )
    if len(shapes) != 1:
        raise ValueError(f"DA3 geometry cache contains inconsistent shapes: {shapes}")
    manifest = build_geometry_manifest(
        source_root=args.source_root,
        datalist_path=args.datalist,
        split=args.split,
        tokens=tokens,
        view_names=DEFAULT_GEOMETRY_VIEWS,
        depth_shape=next(iter(shapes)),
        git_commit=_git_commit(PROJECT_ROOT),
        source_checksums=checksums,
        source_image_hw=(args.source_height, args.source_width),
        frame_index=args.frame_index,
    )
    atomic_write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(validate_geometry_cache(args.output_dir, args.split, tokens)))


if __name__ == "__main__":
    main()
