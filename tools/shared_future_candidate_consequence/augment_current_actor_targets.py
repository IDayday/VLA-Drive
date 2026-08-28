#!/usr/bin/env python3
"""Atomically add current-time actor slots to existing derived v3 target caches."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tools.navsim_candidate_relative_audit.common import load_scenes_for_tokens

from .build_gate_c_targets import current_actor_slots
from .common import (
    DEFAULT_CACHE_DIR,
    DEFAULT_REPORT_DIR,
    append_command,
    ensure_dir,
    navsim_paths,
    write_json,
    write_markdown,
)
from .run_all_log_pipeline import _create_readonly_log_view


def augment(args: argparse.Namespace) -> dict[str, Any]:
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index is out of range")
    report_dir = ensure_dir(args.output_dir)
    cache_dir = ensure_dir(args.cache_dir)
    manifest = pd.read_parquet(report_dir / "balanced_scene_manifest.parquet")
    logs = sorted(manifest.log_name.unique())
    selected_logs = [
        log_name for position, log_name in enumerate(logs)
        if position % args.num_shards == args.shard_index
    ]
    selected = manifest[manifest.log_name.isin(selected_logs)].copy()
    if args.max_scenes > 0:
        selected = selected.head(args.max_scenes).copy()
        selected_logs = sorted(selected.log_name.unique())
    paths = navsim_paths(args.split)
    log_view = cache_dir / "current_actor_log_views" / f"shard_{args.shard_index:04d}"
    _create_readonly_log_view(paths.log_path, log_view, selected_logs)
    loader = load_scenes_for_tokens(replace(paths, log_path=log_view), selected.scene_token.tolist())
    augmented = 0
    reused = 0
    failures: list[dict[str, str]] = []
    started = time.time()
    for position, row in enumerate(selected.itertuples(index=False)):
        token = str(row.scene_token)
        safe_log = str(row.log_name).replace("/", "_")
        target_path = cache_dir / "targets_v3" / safe_log / f"{token}.npz"
        try:
            if not target_path.is_file():
                raise FileNotFoundError(target_path)
            with np.load(target_path, allow_pickle=False) as existing:
                arrays = {key: existing[key] for key in existing.files}
            if (
                not args.force
                and arrays.get("current_actor_state", np.empty(0)).shape == (args.actor_slots, 8)
                and arrays.get("current_actor_mask", np.empty(0)).shape == (args.actor_slots,)
            ):
                reused += 1
                continue
            scene = loader.get_scene_from_token(token)
            actor, mask, _ = current_actor_slots(scene, paths, args.actor_slots)
            arrays["current_actor_state"] = actor
            arrays["current_actor_mask"] = mask
            temporary = target_path.with_suffix(target_path.suffix + ".current-actor.tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
            with np.load(temporary, allow_pickle=False) as check:
                if check["current_actor_state"].shape != (args.actor_slots, 8):
                    raise RuntimeError("Temporary current-actor target failed validation")
            temporary.replace(target_path)
            augmented += 1
        except Exception as exc:
            failures.append(
                {
                    "scene_token": token,
                    "log_name": str(row.log_name),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        if (position + 1) % 1000 == 0:
            print(
                f"shard={args.shard_index} scenes={position + 1}/{len(selected)} "
                f"augmented={augmented} reused={reused} failures={len(failures)}",
                flush=True,
            )
    result = {
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "assigned_logs": len(selected_logs),
        "assigned_scenes": len(selected),
        "augmented": augmented,
        "reused": reused,
        "failure_count": len(failures),
        "failure_examples": failures[:30],
        "elapsed_seconds": time.time() - started,
    }
    write_json(
        cache_dir / "current_actor_augmentation" / f"shard_{args.shard_index:04d}.json",
        result,
    )
    return result


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    report_dir = ensure_dir(args.output_dir)
    cache_dir = ensure_dir(args.cache_dir)
    manifest = pd.read_parquet(
        report_dir / "balanced_scene_manifest.parquet",
        columns=["scene_token", "log_name"],
    )
    summaries = []
    missing_shards = []
    for shard in range(args.num_shards):
        path = cache_dir / "current_actor_augmentation" / f"shard_{shard:04d}.json"
        if path.is_file():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            missing_shards.append(shard)
    expected = int(sum(item["assigned_scenes"] for item in summaries))
    failures = int(sum(item["failure_count"] for item in summaries))
    covered = int(sum(item["augmented"] + item["reused"] for item in summaries))
    # The cache may intentionally contain targets from the immutable 103k-scene
    # inventory.  Audit only the formal log-balanced subset; otherwise a valid
    # 45k-scene run can be failed by unused superset entries that were never
    # augmented (and never enter the oracle store).
    paths = sorted(
        cache_dir
        / "targets_v3"
        / str(row.log_name).replace("/", "_")
        / f"{row.scene_token}.npz"
        for row in manifest.itertuples(index=False)
    )
    paths = [path for path in paths if path.is_file()]
    sample_indices = np.unique(
        np.linspace(0, len(paths) - 1, min(args.audit_samples, len(paths)), dtype=int)
    ) if paths else np.asarray([], dtype=int)
    sample_valid = 0
    sample_errors = []
    for index in sample_indices:
        path = paths[int(index)]
        try:
            with np.load(path, allow_pickle=False) as target:
                valid = (
                    target["current_actor_state"].shape == (args.actor_slots, 8)
                    and target["current_actor_mask"].shape == (args.actor_slots,)
                    and np.isfinite(target["current_actor_state"]).all()
                )
            if not valid:
                raise ValueError("invalid current actor shape or finite mask")
            sample_valid += 1
        except Exception as exc:
            sample_errors.append(f"{path}: {type(exc).__name__}: {exc}")
    result = {
        "num_shards": args.num_shards,
        "completed_shards": len(summaries),
        "missing_shards": missing_shards,
        "expected_scenes": expected,
        "covered_scenes": covered,
        "formal_target_files": len(paths),
        "failure_count": failures,
        "coverage": covered / expected if expected else 0.0,
        "sample_count": len(sample_indices),
        "sample_valid_count": sample_valid,
        "sample_errors": sample_errors[:20],
        "passed": (
            not missing_shards
            and failures == 0
            and covered == expected
            and len(paths) == expected
            and sample_valid == len(sample_indices)
        ),
    }
    write_json(report_dir / "current_actor_augmentation_summary.json", result)
    write_markdown(
        report_dir / "CURRENT_ACTOR_CONDITIONING_REPORT.md",
        f"""# Current-actor Conditioning Audit

- Derived target scenes covered: {covered:,}/{expected:,} ({result['coverage']:.3%})
- Shards: {len(summaries)}/{args.num_shards}; failures: {failures}
- Sampled shape/finite checks: {sample_valid}/{len(sample_indices)}
- Status: {'PASS' if result['passed'] else 'FAIL'}

O1–O13 condition on current-time dynamic actor slots in the current-ego frame,
in addition to the six current-scene summary values. These fields use current
annotations only, never logged-future annotations. They are a conservative
structured oracle control; the deployable model must infer this information from
its current images rather than consume annotation tensors.
""",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("shard", "aggregate"), default="shard")
    parser.add_argument("--split", choices=("train", "trainval"), default="trainval")
    parser.add_argument("--actor-slots", type=int, default=16)
    parser.add_argument("--num-shards", type=int, default=16)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--audit-samples", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR / "all")
    args = parser.parse_args()
    result = aggregate(args) if args.mode == "aggregate" else augment(args)
    print(json.dumps(result, sort_keys=True))
    append_command(
        args.output_dir,
        "python -m tools.shared_future_candidate_consequence.augment_current_actor_targets "
        + " ".join(sys.argv[1:]),
    )


if __name__ == "__main__":
    main()
