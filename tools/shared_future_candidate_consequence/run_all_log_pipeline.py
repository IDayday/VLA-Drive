#!/usr/bin/env python3
"""Resumable per-log candidate scoring and v3 target construction for all logs."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tools.navsim_candidate_relative_audit.build_candidate_relative_targets import build_scene_targets
from tools.navsim_candidate_relative_audit.candidate_generator import TARGET_TIMES, kinematic_summary
from tools.navsim_candidate_relative_audit.common import load_scenes_for_tokens, metric_cache_loader
from tools.navsim_candidate_relative_audit.score_candidates import _rows_from_score, score_pose_batch

from .build_controlled_candidates import generate_randomized_candidates, resample_gt_by_timestamp
from .build_gate_c_targets import split_v3_targets
from .common import (
    DEFAULT_CACHE_DIR,
    DEFAULT_REPORT_DIR,
    append_command,
    ensure_dir,
    navsim_paths,
    require_gate,
    stable_scene_seed,
    write_json,
    write_parquet,
)


def assign_log_shards(scene_manifest: pd.DataFrame, num_shards: int) -> dict[int, list[str]]:
    counts = scene_manifest.groupby("log_name").size().sort_values(ascending=False)
    shards = {index: [] for index in range(num_shards)}
    totals = [0] * num_shards
    for log_name, count in counts.items():
        shard = int(np.argmin(totals))
        shards[shard].append(str(log_name))
        totals[shard] += int(count)
    return shards


def _create_readonly_log_view(source: Path, view: Path, log_names: list[str]) -> None:
    view.mkdir(parents=True, exist_ok=True)
    for log_name in log_names:
        source_file = source / f"{log_name}.pkl"
        target = view / source_file.name
        if not source_file.is_file():
            raise FileNotFoundError(source_file)
        if target.is_symlink():
            if target.resolve() != source_file.resolve():
                raise RuntimeError(f"Unexpected log-view symlink target: {target}")
        elif target.exists():
            raise RuntimeError(f"Log view contains a non-symlink file: {target}")
        else:
            target.symlink_to(source_file)


def _manifest_rows(scene: Any, row: Any, num_candidates: int, global_seed: int) -> pd.DataFrame:
    base = resample_gt_by_timestamp(scene)
    subseed = stable_scene_seed(str(row.scene_token), global_seed)
    candidates, specs = generate_randomized_candidates(base, num_candidates, subseed)
    records = []
    for index, (poses, spec) in enumerate(zip(candidates, specs)):
        records.append(
            {
                "scene_token": str(row.scene_token),
                "scene_metadata_token": scene.scene_metadata.scene_token,
                "log_name": scene.scene_metadata.log_name,
                "map_name": scene.scene_metadata.map_name,
                "fold": int(row.fold),
                "candidate_index": index,
                "candidate_type": spec.candidate_type,
                "candidate_family": spec.family,
                "candidate_parameters": json.dumps(spec.parameters, sort_keys=True),
                "candidate_source": "randomized_smooth_controlled",
                "is_gt": spec.family == "gt",
                "time_s": TARGET_TIMES.astype(np.float32).tolist(),
                "pose_x_m": poses[:, 0].tolist(),
                "pose_y_m": poses[:, 1].tolist(),
                "pose_heading_rad": poses[:, 2].tolist(),
                "implicit_start_x_m": 0.0,
                "implicit_start_y_m": 0.0,
                "implicit_start_heading_rad": 0.0,
                "metric_cache_available": True,
                "seed": int(subseed),
                **kinematic_summary(poses),
            }
        )
    return pd.DataFrame(records)


def _cache_covers_selection(
    candidate_path: Path,
    metric_path: Path,
    target_dir: Path,
    scene_tokens: list[str],
    num_candidates: int,
) -> bool:
    """Return whether a possibly larger cache contains the formal selection.

    The all-log inventory was intentionally preserved as an immutable superset
    when the formal protocol was tightened to at most 50 scenes per log.  A
    per-log cache produced from that inventory is valid for the balanced subset
    as long as every requested `(scene, candidate)` row and target is present.
    Requiring equal row counts would needlessly recompute tens of thousands of
    already audited candidates and, more importantly, would make resharding
    dependent on stale shard-summary files.
    """

    if not candidate_path.is_file() or not metric_path.is_file() or not target_dir.is_dir():
        return False
    if not all((target_dir / f"{token}.npz").is_file() for token in scene_tokens):
        return False
    try:
        candidates = pd.read_parquet(
            candidate_path, columns=["scene_token", "candidate_index"]
        )
        metrics = pd.read_parquet(
            metric_path, columns=["scene_token", "candidate_index", "scoring_success"]
        )
    except Exception:
        return False
    expected = {
        (str(token), candidate_index)
        for token in scene_tokens
        for candidate_index in range(num_candidates)
    }
    candidate_keys = set(
        zip(candidates.scene_token.astype(str), candidates.candidate_index.astype(int))
    )
    successful_metrics = metrics[metrics.scoring_success.astype(bool)]
    metric_keys = set(
        zip(
            successful_metrics.scene_token.astype(str),
            successful_metrics.candidate_index.astype(int),
        )
    )
    return expected.issubset(candidate_keys) and expected.issubset(metric_keys)


def run(args: argparse.Namespace) -> dict[str, Any]:
    require_gate(args.output_dir, "gate_c0")
    report_dir = ensure_dir(args.output_dir)
    cache_dir = ensure_dir(args.cache_dir)
    manifest_path = report_dir / "balanced_scene_manifest.parquet"
    scene_manifest = pd.read_parquet(manifest_path)
    shards = assign_log_shards(scene_manifest, args.num_shards)
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index is out of range")
    log_names = shards[args.shard_index]
    if args.max_logs > 0:
        log_names = log_names[: args.max_logs]
    selected = scene_manifest[scene_manifest.log_name.isin(log_names)].copy()
    paths = navsim_paths(args.split)
    view = cache_dir / "log_views" / f"shard_{args.shard_index:04d}"
    _create_readonly_log_view(paths.log_path, view, log_names)
    shard_paths = replace(paths, log_path=view)
    loader = load_scenes_for_tokens(shard_paths, selected.scene_token.tolist())
    caches = metric_cache_loader(paths)
    candidate_dir = ensure_dir(cache_dir / "candidate_shards")
    metric_dir = ensure_dir(cache_dir / "metric_shards")
    target_dir = ensure_dir(cache_dir / "targets_v3")
    summary_dir = ensure_dir(cache_dir / "log_summaries")

    processed_logs = 0
    cached_logs = 0
    processed_scenes = 0
    processed_candidates = 0
    failures: list[dict[str, str]] = []
    started = time.time()
    for log_position, log_name in enumerate(log_names):
        log_rows = selected[selected.log_name == log_name].sort_values("selection_index")
        safe_name = log_name.replace("/", "_")
        candidate_path = candidate_dir / f"{safe_name}.parquet"
        metric_path = metric_dir / f"{safe_name}.parquet"
        summary_path = summary_dir / f"{safe_name}.json"
        log_target_dir = target_dir / safe_name
        if summary_path.is_file() and not args.force:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("complete") and _cache_covers_selection(
                candidate_path,
                metric_path,
                log_target_dir,
                log_rows.scene_token.astype(str).tolist(),
                args.num_candidates,
            ):
                cached_logs += 1
                continue
        candidate_frames = []
        metric_records: list[dict[str, Any]] = []
        target_success = 0
        for row in log_rows.itertuples(index=False):
            try:
                scene = loader.get_scene_from_token(row.scene_token)
                candidate_frame = _manifest_rows(scene, row, args.num_candidates, args.seed)
                poses = np.stack(
                    [
                        np.column_stack([item.pose_x_m, item.pose_y_m, item.pose_heading_rad])
                        for item in candidate_frame.sort_values("candidate_index").itertuples(index=False)
                    ],
                    axis=0,
                ).astype(np.float32)
                cache = caches.get_from_token(row.scene_token)
                score_started = time.perf_counter()
                score = score_pose_batch(cache, poses)
                scored_rows = _rows_from_score(candidate_frame, score, time.perf_counter() - score_started)
                scored = pd.DataFrame(scored_rows)
                old = build_scene_targets(scene, cache, scored, paths, 16, 64)
                arrays = split_v3_targets(scene, old, paths, args.actor_slots)
                log_target_dir = ensure_dir(log_target_dir)
                target_path = log_target_dir / f"{row.scene_token}.npz"
                if not target_path.exists() or args.force:
                    np.savez_compressed(target_path, **arrays)
                candidate_frames.append(candidate_frame)
                metric_records.extend(scored_rows)
                target_success += 1
            except Exception as exc:
                failures.append(
                    {
                        "scene_token": str(row.scene_token),
                        "log_name": log_name,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(limit=12),
                    }
                )
        candidate_frame = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
        metrics = pd.DataFrame(metric_records)
        write_parquet(candidate_frame, candidate_path)
        write_parquet(metrics, metric_path)
        log_summary = {
            "complete": target_success == len(log_rows),
            "log_name": log_name,
            "scene_count": int(len(log_rows)),
            "target_success": target_success,
            "candidate_count": int(len(candidate_frame)),
            "scoring_success_rate": float(metrics.scoring_success.mean()) if len(metrics) else 0.0,
            "failure_count": int(len(log_rows) - target_success),
            "shard_index": args.shard_index,
        }
        write_json(summary_path, log_summary)
        processed_logs += 1
        processed_scenes += target_success
        processed_candidates += len(metrics)
        if (log_position + 1) % 10 == 0:
            print(
                f"shard={args.shard_index} logs={log_position + 1}/{len(log_names)} "
                f"scenes={processed_scenes} failures={len(failures)}",
                flush=True,
            )
    result = {
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "assigned_logs": len(log_names),
        "assigned_scenes": int(len(selected)),
        "processed_logs_this_run": processed_logs,
        "cached_logs_reused": cached_logs,
        "processed_scenes_this_run": processed_scenes,
        "processed_candidates_this_run": processed_candidates,
        "failure_count_this_run": len(failures),
        "failure_examples": failures[:20],
        "elapsed_seconds": time.time() - started,
    }
    write_json(cache_dir / f"shard_{args.shard_index:04d}_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "trainval"), default="trainval")
    parser.add_argument("--num-candidates", type=int, default=16)
    parser.add_argument("--actor-slots", type=int, default=16)
    parser.add_argument("--num-shards", type=int, default=16)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument(
        "--max-logs",
        type=int,
        default=0,
        help="Debug-only cap after deterministic shard assignment; zero processes the full shard.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR / "all")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, sort_keys=True))
    append_command(args.output_dir, "python -m tools.shared_future_candidate_consequence.run_all_log_pipeline " + " ".join(sys.argv[1:]))


if __name__ == "__main__":
    main()
