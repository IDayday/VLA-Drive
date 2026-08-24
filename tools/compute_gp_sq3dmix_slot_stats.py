#!/usr/bin/env python3
"""Build resumable full-train GP-SQ3D-Mix slot and descriptor statistics."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from starVLA.cache.navsim_feature_cache import NavsimFeatureCacheReader
from starVLA.gp_sq3dmix_v2 import (
    DESCRIPTOR_DIMENSION,
    DESCRIPTOR_PROJECTION_SEED,
    DESCRIPTOR_PROJECTION_SHAPE,
    descriptor_projection,
    pooled_scene_descriptor,
    tensor_sha256,
    token_order_sha256,
)
from starVLA.model.modules.vggt_query.gp_slot_stats import (
    POOLING_LAYOUT,
    VIEW_ORDER,
    sha256_file,
)
from starVLA.model.modules.vggt_query.vggt_patch_pool import (
    pool_dense_vggt_geometry_per_view,
)


SCHEMA_VERSION = 2
CONTRACT_FILE = "stats_contract.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--datalist", required=True)
    parser.add_argument("--stats-root", required=True)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--num-shards", type=int, default=16)
    parser.add_argument("--shard-id", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def git_commit() -> str:
    root = Path(__file__).resolve().parents[1]
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: object) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _read_tokens(path: Path) -> list[str]:
    tokens = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(tokens, list)
        or not tokens
        or not all(isinstance(token, str) and token for token in tokens)
        or len(tokens) != len(set(tokens))
    ):
        raise ValueError("datalist must be a non-empty list of unique string tokens")
    return tokens


def _build_contract(
    cache_root: Path,
    datalist: Path,
    stats_root: Path,
    num_shards: int,
    tokens: list[str],
) -> tuple[dict, Path]:
    cache_manifest = cache_root / "vggt_dense" / "manifest.json"
    if not cache_manifest.is_file():
        raise FileNotFoundError(cache_manifest)
    cache_metadata = json.loads(cache_manifest.read_text(encoding="utf-8"))
    datalist_sha = sha256_file(datalist)
    if cache_metadata.get("datalist_sha256") != datalist_sha:
        raise RuntimeError("Dense cache manifest does not match the training datalist")
    if cache_metadata.get("view_order") != VIEW_ORDER:
        raise RuntimeError("Dense cache view order is not front/left/right")
    if int(cache_metadata.get("sample_count", -1)) != len(tokens):
        raise RuntimeError("Dense cache sample count does not match the datalist")
    projection = descriptor_projection()
    contract = {
        "schema_version": SCHEMA_VERSION,
        "source_cache_manifest_sha256": sha256_file(cache_manifest),
        "datalist_sha256": datalist_sha,
        "sample_count": len(tokens),
        "token_order_sha256": token_order_sha256(tokens),
        "view_order": VIEW_ORDER,
        "pooling_layout": POOLING_LAYOUT,
        "feature_dimension": 2048,
        "descriptor_dimension": DESCRIPTOR_DIMENSION,
        "descriptor_projection_seed": DESCRIPTOR_PROJECTION_SEED,
        "descriptor_projection_shape": list(DESCRIPTOR_PROJECTION_SHAPE),
        "descriptor_projection_sha256": tensor_sha256(projection),
        "shard_count": int(num_shards),
        "cache_root": str(cache_root),
        "datalist": str(datalist),
        "stats_root": str(stats_root),
    }
    return contract, cache_manifest


def _contract_sha(contract: dict) -> str:
    encoded = json.dumps(
        contract, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    import hashlib

    return hashlib.sha256(encoded).hexdigest()


def _validate_or_create_root(
    stats_root: Path, contract: dict, *, resume: bool, merge_only: bool, dry_run: bool
) -> None:
    contract_path = stats_root / CONTRACT_FILE
    if not stats_root.exists():
        if merge_only:
            raise FileNotFoundError(
                f"--merge-only requires an existing stats root: {stats_root}"
            )
        if not dry_run:
            stats_root.mkdir(parents=True, exist_ok=False)
            atomic_json(contract_path, contract)
        return
    if not stats_root.is_dir():
        raise NotADirectoryError(stats_root)
    entries = list(stats_root.iterdir())
    if not entries:
        if merge_only:
            raise RuntimeError("--merge-only cannot use an empty stats root")
        if not dry_run:
            atomic_json(contract_path, contract)
        return
    if not resume and not merge_only:
        raise FileExistsError(
            f"Refusing to reuse non-empty stats root without --resume: {stats_root}"
        )
    if not contract_path.is_file():
        raise RuntimeError(
            f"Existing stats root has no {CONTRACT_FILE}; refusing to mix assets"
        )
    existing = json.loads(contract_path.read_text(encoding="utf-8"))
    if existing != contract:
        raise RuntimeError("Existing slot-stat directory contract/hash does not match")


def _expected_indices(sample_count: int, num_shards: int, shard_id: int) -> list[int]:
    return list(range(shard_id, sample_count, num_shards))


def _validate_partial(
    path: Path,
    *,
    shard_id: int,
    contract: dict,
    expected_tokens: list[str],
) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise RuntimeError(f"Cannot load completed shard {path}: {error}") from error
    required = {
        "count",
        "pooled_feature_slot_mean",
        "pooled_scene_descriptor_sum",
        "token_descriptors",
        "tokens",
        "source_indices",
        "shard_id",
        "contract_sha256",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise RuntimeError(f"Partial shard has an invalid schema: {path}")
    count = int(payload["count"])
    if (
        int(payload["shard_id"]) != shard_id
        or payload["contract_sha256"] != _contract_sha(contract)
        or count != len(expected_tokens)
        or payload["tokens"] != expected_tokens
    ):
        raise RuntimeError(f"Partial shard contract/token order mismatch: {path}")
    if payload["pooled_feature_slot_mean"].shape != (180, 2048):
        raise RuntimeError(f"Partial shard mean shape mismatch: {path}")
    if payload["pooled_feature_slot_mean"].dtype != torch.float64:
        raise RuntimeError(f"Partial shard mean must be float64: {path}")
    if payload["pooled_scene_descriptor_sum"].shape != (DESCRIPTOR_DIMENSION,):
        raise RuntimeError(f"Partial descriptor sum shape mismatch: {path}")
    descriptors = payload["token_descriptors"]
    if descriptors.shape != (count, DESCRIPTOR_DIMENSION) or descriptors.dtype != torch.float16:
        raise RuntimeError(f"Partial descriptor matrix mismatch: {path}")
    if not torch.isfinite(payload["pooled_feature_slot_mean"]).all() or not torch.isfinite(descriptors).all():
        raise RuntimeError(f"Partial shard contains non-finite values: {path}")
    return payload


def _compute_shard(
    cache_root_string: str,
    stats_root_string: str,
    tokens: list[str],
    num_shards: int,
    shard_id: int,
    contract: dict,
) -> dict:
    cache_root = Path(cache_root_string)
    stats_root = Path(stats_root_string)
    indices = _expected_indices(len(tokens), num_shards, shard_id)
    shard_tokens = [tokens[index] for index in indices]
    partial_path = stats_root / f"partial_{shard_id}.pt"
    if partial_path.exists():
        _validate_partial(
            partial_path,
            shard_id=shard_id,
            contract=contract,
            expected_tokens=shard_tokens,
        )
        return {"shard_id": shard_id, "count": len(indices), "resumed": True}

    reader = NavsimFeatureCacheReader(
        cache_root, components=("vggt_dense",), strict=True
    )
    projection = descriptor_projection()
    mean = torch.zeros((180, 2048), dtype=torch.float64)
    descriptor_sum = torch.zeros(DESCRIPTOR_DIMENSION, dtype=torch.float64)
    descriptors = torch.empty(
        (len(indices), DESCRIPTOR_DIMENSION), dtype=torch.float16
    )
    count = 0
    started = time.monotonic()
    for local_index, (source_index, token) in enumerate(zip(indices, shard_tokens)):
        payload = reader.get("vggt_dense", source_index, token)
        if payload is None:
            raise RuntimeError(
                f"Dense cache miss at source index={source_index} token={token}"
            )
        pooled = pool_dense_vggt_geometry_per_view(
            [payload], device="cpu", dtype=torch.float32
        )["features"][0]
        count += 1
        mean.add_((pooled.double() - mean) / count)
        descriptor = pooled_scene_descriptor(pooled, projection).cpu()
        descriptor_sum.add_(descriptor.double())
        descriptors[local_index].copy_(descriptor.to(torch.float16))
        if count % 250 == 0:
            elapsed = max(time.monotonic() - started, 1e-6)
            print(
                f"[gp-slot-stats shard={shard_id}] {count}/{len(indices)} "
                f"({count / elapsed:.2f} samples/s)",
                flush=True,
            )
    result = {
        "schema_version": SCHEMA_VERSION,
        "shard_id": int(shard_id),
        "contract_sha256": _contract_sha(contract),
        "count": int(count),
        "pooled_feature_slot_mean": mean,
        "pooled_scene_descriptor_sum": descriptor_sum,
        "token_descriptors": descriptors,
        "tokens": shard_tokens,
        "source_indices": indices,
    }
    atomic_torch_save(partial_path, result)
    _validate_partial(
        partial_path,
        shard_id=shard_id,
        contract=contract,
        expected_tokens=shard_tokens,
    )
    return {"shard_id": shard_id, "count": count, "resumed": False}


def _merge(stats_root: Path, tokens: list[str], contract: dict) -> dict:
    descriptors = torch.empty(
        (len(tokens), DESCRIPTOR_DIMENSION), dtype=torch.float16
    )
    total_mean = torch.zeros((180, 2048), dtype=torch.float64)
    descriptor_sum = torch.zeros(DESCRIPTOR_DIMENSION, dtype=torch.float64)
    total_count = 0
    completed = []
    for shard_id in range(int(contract["shard_count"])):
        indices = _expected_indices(len(tokens), int(contract["shard_count"]), shard_id)
        expected_tokens = [tokens[index] for index in indices]
        partial_path = stats_root / f"partial_{shard_id}.pt"
        if not partial_path.is_file():
            raise RuntimeError(f"Cannot merge: missing shard {shard_id}")
        payload = _validate_partial(
            partial_path,
            shard_id=shard_id,
            contract=contract,
            expected_tokens=expected_tokens,
        )
        count = int(payload["count"])
        if count:
            combined = total_count + count
            total_mean.mul_(total_count / combined).add_(
                payload["pooled_feature_slot_mean"], alpha=count / combined
            )
            total_count = combined
        descriptor_sum.add_(payload["pooled_scene_descriptor_sum"])
        descriptors[torch.as_tensor(indices, dtype=torch.long)] = payload[
            "token_descriptors"
        ]
        completed.append(shard_id)
    if total_count != len(tokens):
        raise RuntimeError(
            f"Merged count {total_count} does not match datalist count {len(tokens)}"
        )
    stats_path = stats_root / "gp_sq3dmix_pooled_stats.pt"
    descriptor_path = stats_root / "pooled_scene_descriptors.pt"
    manifest_path = stats_root / "manifest.json"
    for path in (stats_path, descriptor_path, manifest_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite completed asset: {path}")
    atomic_torch_save(
        stats_path, {"pooled_feature_slot_mean": total_mean.float()}
    )
    atomic_torch_save(
        descriptor_path,
        {
            "tokens": tokens,
            "descriptors": descriptors,
            "pooled_scene_descriptor_sum": descriptor_sum,
        },
    )
    manifest = {
        **contract,
        "complete": True,
        "code_commit": git_commit(),
        "completed_shards": completed,
        "stats_file_sha256": sha256_file(stats_path),
        "descriptor_file_sha256": sha256_file(descriptor_path),
    }
    atomic_json(manifest_path, manifest)
    return manifest


def main() -> None:
    args = parse_args()
    if args.num_workers < 1 or args.num_shards < 1:
        raise ValueError("--num-workers and --num-shards must be positive")
    if args.shard_id is not None and not 0 <= args.shard_id < args.num_shards:
        raise ValueError("--shard-id must be in [0, num-shards)")
    if args.merge_only and args.shard_id is not None:
        raise ValueError("--merge-only cannot be combined with --shard-id")
    cache_root = Path(args.cache_root).resolve()
    datalist = Path(args.datalist).resolve()
    stats_root = Path(args.stats_root).resolve()
    if not datalist.is_file():
        raise FileNotFoundError(datalist)
    tokens = _read_tokens(datalist)
    contract, _ = _build_contract(
        cache_root, datalist, stats_root, args.num_shards, tokens
    )
    resolved = {
        **contract,
        "num_workers": args.num_workers,
        "shard_id": args.shard_id,
        "resume": args.resume,
        "merge_only": args.merge_only,
    }
    if args.dry_run:
        print(json.dumps(resolved, indent=2, sort_keys=True))
        return
    _validate_or_create_root(
        stats_root,
        contract,
        resume=args.resume,
        merge_only=args.merge_only,
        dry_run=False,
    )

    complete_manifest = stats_root / "manifest.json"
    if complete_manifest.is_file():
        if not args.resume and not args.merge_only:
            raise FileExistsError(complete_manifest)
        manifest = json.loads(complete_manifest.read_text(encoding="utf-8"))
        for name in ("gp_sq3dmix_pooled_stats.pt", "pooled_scene_descriptors.pt"):
            if not (stats_root / name).is_file():
                raise RuntimeError("Completed manifest exists but an output file is missing")
        if any(manifest.get(key) != contract.get(key) for key in contract):
            raise RuntimeError("Completed stats manifest does not match active contract")
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    if not args.merge_only:
        shard_ids = (
            [args.shard_id]
            if args.shard_id is not None
            else list(range(args.num_shards))
        )
        if args.num_workers == 1 or len(shard_ids) == 1:
            for shard_id in shard_ids:
                print(
                    _compute_shard(
                        str(cache_root),
                        str(stats_root),
                        tokens,
                        args.num_shards,
                        int(shard_id),
                        contract,
                    ),
                    flush=True,
                )
        else:
            with ProcessPoolExecutor(max_workers=min(args.num_workers, len(shard_ids))) as pool:
                futures = {
                    pool.submit(
                        _compute_shard,
                        str(cache_root),
                        str(stats_root),
                        tokens,
                        args.num_shards,
                        int(shard_id),
                        contract,
                    ): shard_id
                    for shard_id in shard_ids
                }
                for future in as_completed(futures):
                    print(future.result(), flush=True)
        if args.shard_id is not None:
            return
    manifest = _merge(stats_root, tokens, contract)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
