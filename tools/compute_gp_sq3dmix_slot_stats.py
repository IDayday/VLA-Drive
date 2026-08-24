#!/usr/bin/env python3
"""Compute stable train-set pooled VGGT slot means for GP-SQ3D-Mix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import torch

from starVLA.cache.navsim_feature_cache import NavsimFeatureCacheReader
from starVLA.model.modules.vggt_query.gp_slot_stats import (
    POOLING_LAYOUT,
    VIEW_ORDER,
    sha256_file,
)
from starVLA.model.modules.vggt_query.vggt_patch_pool import (
    pool_dense_vggt_geometry_per_view,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--datalist", required=True)
    parser.add_argument("--stats-root", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    cache_root = Path(args.cache_root).resolve()
    datalist = Path(args.datalist).resolve()
    stats_root = Path(args.stats_root).resolve()
    cache_manifest = cache_root / "vggt_dense" / "manifest.json"
    for path in (datalist, cache_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    tokens = json.loads(datalist.read_text(encoding="utf-8"))
    if not isinstance(tokens, list) or not tokens or len(tokens) != len(set(tokens)):
        raise ValueError("datalist must be a non-empty list of unique tokens")
    cache_metadata = json.loads(cache_manifest.read_text(encoding="utf-8"))
    if cache_metadata.get("datalist_sha256") != sha256_file(datalist):
        raise RuntimeError("Dense cache manifest does not match the training datalist")
    if cache_metadata.get("view_order") != VIEW_ORDER:
        raise RuntimeError("Dense cache view order is not front/left/right")
    if int(cache_metadata.get("sample_count", -1)) != len(tokens):
        raise RuntimeError("Dense cache sample count does not match the datalist")
    resolved = {
        "cache_root": str(cache_root),
        "datalist": str(datalist),
        "stats_root": str(stats_root),
        "sample_count": len(tokens),
        "source_cache_manifest_sha256": sha256_file(cache_manifest),
        "datalist_sha256": sha256_file(datalist),
    }
    if args.dry_run:
        print(json.dumps(resolved, indent=2, sort_keys=True))
        return
    if stats_root.exists() and any(stats_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty stats root: {stats_root}")
    stats_root.mkdir(parents=True, exist_ok=False)
    reader = NavsimFeatureCacheReader(cache_root, components=("vggt_dense",), strict=True)
    mean = torch.zeros((180, 2048), dtype=torch.float64)
    count = 0
    for index, token in enumerate(tokens):
        payload = reader.get("vggt_dense", index, token)
        pooled = pool_dense_vggt_geometry_per_view(
            [payload], device="cpu", dtype=torch.float32
        )["features"][0].double()
        count += 1
        mean.add_((pooled - mean) / count)
        if count % 1000 == 0:
            print(f"[gp-slot-stats] {count}/{len(tokens)}", flush=True)
    stats_path = stats_root / "gp_sq3dmix_pooled_stats.pt"
    temporary = stats_path.with_name(stats_path.name + f".tmp-{os.getpid()}")
    torch.save({"pooled_feature_slot_mean": mean.float()}, temporary)
    os.replace(temporary, stats_path)
    manifest = {
        "schema_version": 1,
        "complete": True,
        "source_cache_manifest_sha256": sha256_file(cache_manifest),
        "datalist_sha256": sha256_file(datalist),
        "sample_count": count,
        "view_order": VIEW_ORDER,
        "pooling_layout": POOLING_LAYOUT,
        "feature_dimension": 2048,
        "code_commit": git_commit(),
        "stats_file_sha256": sha256_file(stats_path),
    }
    atomic_json(stats_root / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
