#!/usr/bin/env python3
"""Score frozen EpisodeDrive proposals and build a diverse K=16 model bank."""

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

from tools.navsim_candidate_relative_audit.build_candidate_relative_targets import build_scene_targets
from tools.navsim_candidate_relative_audit.candidate_generator import TARGET_TIMES, kinematic_summary
from tools.navsim_candidate_relative_audit.common import load_scenes_for_tokens, metric_cache_loader
from tools.navsim_candidate_relative_audit.score_candidates import _rows_from_score, score_pose_batch

from .build_gate_c_targets import split_v3_targets
from .common import (
    DEFAULT_CACHE_DIR,
    DEFAULT_REPORT_DIR,
    append_command,
    ensure_dir,
    navsim_paths,
    stable_scene_seed,
    write_json,
    write_markdown,
    write_parquet,
)
from .export_episode_drive_candidates import EXPECTED_BASE_SHA256
from .run_all_log_pipeline import _create_readonly_log_view


SCALAR_SCORE_FIELDS = (
    "score",
    "no_at_fault_collision",
    "dac",
    "ddc",
    "progress",
    "raw_progress_m",
    "ttc",
    "comfort",
    "first_collision_time_s",
    "first_ttc_violation_time_s",
)


def trajectory_distance(proposals: np.ndarray) -> np.ndarray:
    xy = proposals[..., :2]
    heading = proposals[..., 2]
    position = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1).mean(axis=-1)
    delta_heading = np.abs(
        (heading[:, None] - heading[None, :] + np.pi) % (2 * np.pi) - np.pi
    ).mean(axis=-1)
    return (position + 0.5 * delta_heading).astype(np.float32)


def select_diverse_candidates(
    proposals: np.ndarray,
    baseline_scores: np.ndarray,
    official_scores: np.ndarray,
    k: int,
    seed: int,
) -> tuple[np.ndarray, dict[int, list[str]], dict[str, Any]]:
    """Combine baseline-high, farthest geometry and close/outcome-hard proposals."""

    if len(proposals) < k:
        raise ValueError(f"Only {len(proposals)} unique proposals for K={k}")
    distance = trajectory_distance(proposals)
    baseline_order = np.argsort(-baseline_scores, kind="stable")
    selected: list[int] = []
    reasons: dict[int, list[str]] = {}

    def add(index: int, reason: str) -> None:
        index = int(index)
        if index not in selected:
            selected.append(index)
            reasons[index] = [reason]
        elif reason not in reasons[index]:
            reasons[index].append(reason)

    for index in baseline_order[:5]:
        add(int(index), "baseline_high")
    baseline_selected = int(baseline_order[0])
    add(baseline_selected, "baseline_selected")

    # Farthest-point sampling fills six additional geometrically distinct slots.
    while len(selected) < min(k, 11):
        remaining = [index for index in range(len(proposals)) if index not in selected]
        minimum = np.min(distance[np.ix_(remaining, selected)], axis=1)
        best = remaining[int(np.argmax(minimum))]
        add(best, "geometry_farthest")

    anchor_indices = baseline_order[: min(12, len(baseline_order))]
    candidates = []
    nonzero = distance[np.triu_indices(len(distance), k=1)]
    close_threshold = float(np.quantile(nonzero[nonzero > 1e-6], 0.35)) if np.any(nonzero > 1e-6) else 0.0
    for index in range(len(proposals)):
        if index in selected:
            continue
        anchor_position = int(np.argmin(distance[index, anchor_indices]))
        anchor = int(anchor_indices[anchor_position])
        geometry = float(distance[index, anchor])
        outcome_difference = float(abs(official_scores[index] - official_scores[anchor]))
        if geometry <= close_threshold and outcome_difference >= 0.05:
            candidates.append((outcome_difference / max(geometry, 0.05), outcome_difference, -geometry, index))
    candidates.sort(reverse=True)
    for _, _, _, index in candidates:
        if len(selected) >= min(k, 15):
            break
        add(index, "geometry_close_official_outcome_different")

    # Fill any sparse hard-negative quota using baseline rank, then FPS.
    for index in baseline_order:
        if len(selected) >= k:
            break
        add(int(index), "baseline_fill")
    while len(selected) < k:
        remaining = [index for index in range(len(proposals)) if index not in selected]
        minimum = np.min(distance[np.ix_(remaining, selected)], axis=1)
        add(remaining[int(np.argmax(minimum))], "geometry_fill")

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(selected))
    selected_array = np.asarray(selected, dtype=np.int64)[permutation]
    stats = {
        "raw_proposal_count": int(len(proposals)),
        "selected_count": int(k),
        "baseline_selected_original_index": baseline_selected,
        "baseline_selected_retained": baseline_selected in selected,
        "selected_official_best": float(np.max(official_scores[selected_array])),
        "raw_official_best": float(np.max(official_scores)),
        "selection_best_of_raw_regret": float(np.max(official_scores) - np.max(official_scores[selected_array])),
        "selected_pairwise_geometry_mean": float(
            distance[np.ix_(selected_array, selected_array)][np.triu_indices(k, 1)].mean()
        ),
        "hard_outcome_count": int(
            sum("geometry_close_official_outcome_different" in reasons[index] for index in selected)
        ),
    }
    return selected_array, reasons, stats


def _slice_score(score: dict[str, Any], indices: np.ndarray) -> dict[str, Any]:
    count = len(score["score"])
    sliced: dict[str, Any] = {}
    for key, value in score.items():
        if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == count:
            sliced[key] = value[indices]
        elif isinstance(value, list) and len(value) == count:
            sliced[key] = [value[int(index)] for index in indices]
        else:
            sliced[key] = value
    return sliced


def _raw_score_rows(
    token: str,
    log_name: str,
    fold: int,
    baseline_scores: np.ndarray,
    baseline_factors: np.ndarray,
    factor_names: np.ndarray,
    score: dict[str, Any],
    original_indices: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for index in range(len(baseline_scores)):
        row = {
            "scene_token": token,
            "log_name": log_name,
            "fold": fold,
            "original_proposal_index": int(original_indices[index]),
            "baseline_scorer_score": float(baseline_scores[index]),
        }
        for factor_index, name in enumerate(factor_names):
            row[f"baseline_logit_{name}"] = float(baseline_factors[index, factor_index])
        for key in SCALAR_SCORE_FIELDS:
            value = score[key][index]
            row[f"official_{'aggregate_score' if key == 'score' else key}"] = float(value)
        rows.append(row)
    return rows


def _candidate_manifest(
    token: str,
    log_name: str,
    map_name: str,
    fold: int,
    proposals: np.ndarray,
    baseline_scores: np.ndarray,
    baseline_factors: np.ndarray,
    factor_names: np.ndarray,
    selected: np.ndarray,
    reasons: dict[int, list[str]],
    seed: int,
) -> pd.DataFrame:
    rows = []
    for candidate_index, original_index in enumerate(selected):
        pose = proposals[int(original_index)]
        row = {
            "scene_token": token,
            "log_name": log_name,
            "map_name": map_name,
            "fold": fold,
            "candidate_index": candidate_index,
            "original_proposal_index": int(original_index),
            "candidate_source": "frozen_episode_drive_base_proposal",
            "candidate_family": "model_proposal",
            "candidate_type": "model_proposal",
            "selection_reason": json.dumps(reasons[int(original_index)]),
            "is_gt": False,
            "time_s": TARGET_TIMES.astype(np.float32).tolist(),
            "pose_x_m": pose[:, 0].astype(np.float32).tolist(),
            "pose_y_m": pose[:, 1].astype(np.float32).tolist(),
            "pose_heading_rad": pose[:, 2].astype(np.float32).tolist(),
            "baseline_scorer_score": float(baseline_scores[int(original_index)]),
            "seed": int(seed),
            **kinematic_summary(pose),
        }
        for factor_index, name in enumerate(factor_names):
            row[f"baseline_logit_{name}"] = float(baseline_factors[int(original_index), factor_index])
        rows.append(row)
    return pd.DataFrame(rows)


def _load_export_manifest(cache_dir: Path) -> pd.DataFrame:
    shard_paths = sorted((cache_dir / "model_candidates" / "export_shards").glob("shard_*.parquet"))
    frames = [pd.read_parquet(path) for path in shard_paths]
    frames = [frame for frame in frames if len(frame)]
    if not frames:
        raise FileNotFoundError("No EpisodeDrive raw export shards")
    frame = pd.concat(frames, ignore_index=True)
    if frame.scene_token.duplicated().any():
        raise RuntimeError("Duplicate EpisodeDrive raw exports")
    return frame


def run_shard(args: argparse.Namespace) -> dict[str, Any]:
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError(
            f"shard-index {args.shard_index} is outside [0, {args.num_shards})"
        )
    if args.num_candidates < 2:
        raise ValueError("num-candidates must be at least two")
    cache_dir = ensure_dir(args.cache_dir)
    report_dir = ensure_dir(args.output_dir)
    aggregate_manifest = report_dir / "candidates/episode_drive_raw_exports.parquet"
    exports = (
        pd.read_parquet(aggregate_manifest)
        if aggregate_manifest.is_file()
        else _load_export_manifest(cache_dir)
    )
    logs = sorted(exports.log_name.unique())
    selected_logs = [name for position, name in enumerate(logs) if position % args.num_shards == args.shard_index]
    exports = exports[exports.log_name.isin(selected_logs)].copy()
    if args.max_scenes > 0:
        exports = exports.head(args.max_scenes).copy()
        selected_logs = sorted(exports.log_name.unique())
    paths = navsim_paths(args.split)
    log_view = cache_dir / "model_candidates" / "log_views" / f"shard_{args.shard_index:04d}"
    _create_readonly_log_view(paths.log_path, log_view, selected_logs)
    loader = load_scenes_for_tokens(replace(paths, log_path=log_view), exports.scene_token.tolist())
    caches = metric_cache_loader(paths)
    manifest_dir = ensure_dir(cache_dir / "model_candidates" / "selected_by_log")
    metric_dir = ensure_dir(cache_dir / "model_candidates" / "metrics_by_log")
    raw_metric_dir = ensure_dir(cache_dir / "model_candidates" / "raw_metrics_by_log")
    target_dir = ensure_dir(cache_dir / "model_candidates" / "targets_v3")
    summary_dir = ensure_dir(cache_dir / "model_candidates" / "log_summaries")
    failures = []
    processed_scenes = 0
    started = time.time()
    for log_position, log_name in enumerate(selected_logs):
        log_exports = exports[exports.log_name == log_name]
        safe_name = str(log_name).replace("/", "_")
        manifest_path = manifest_dir / f"{safe_name}.parquet"
        metric_path = metric_dir / f"{safe_name}.parquet"
        raw_metric_path = raw_metric_dir / f"{safe_name}.parquet"
        summary_path = summary_dir / f"{safe_name}.json"
        log_target_dir = target_dir / safe_name
        if (
            summary_path.is_file()
            and manifest_path.is_file()
            and metric_path.is_file()
            and raw_metric_path.is_file()
            and not args.force
        ):
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            target_count = len(list(log_target_dir.glob("*.npz"))) if log_target_dir.is_dir() else 0
            if summary.get("complete") and target_count == len(log_exports):
                continue
        candidate_frames = []
        metric_rows = []
        raw_rows = []
        selection_stats = []
        success = 0
        for export_row in log_exports.itertuples(index=False):
            token = str(export_row.scene_token)
            try:
                with np.load(export_row.path, allow_pickle=False) as raw:
                    if str(raw["checkpoint_sha256"].item()) != EXPECTED_BASE_SHA256:
                        raise RuntimeError("Raw proposal checkpoint identity mismatch")
                    proposals = raw["proposals"].astype(np.float32)
                    baseline_scores = raw["baseline_scorer_score"].astype(np.float32)
                    baseline_factors = raw["baseline_factor_logits"].astype(np.float32)
                    factor_names = raw["factor_names"].astype(str)
                signatures = {}
                unique_indices = []
                for index, proposal in enumerate(proposals):
                    signature = np.round(proposal, 5).tobytes()
                    if signature not in signatures:
                        signatures[signature] = index
                        unique_indices.append(index)
                unique_indices = np.asarray(unique_indices, dtype=np.int64)
                unique_proposals = proposals[unique_indices]
                cache = caches.get_from_token(token)
                score_started = time.perf_counter()
                raw_score = score_pose_batch(cache, unique_proposals)
                raw_runtime = time.perf_counter() - score_started
                scene_seed = stable_scene_seed(token, args.seed ^ 0xC411D)
                selected_unique, reasons_unique, stats = select_diverse_candidates(
                    unique_proposals,
                    baseline_scores[unique_indices],
                    raw_score["score"],
                    args.num_candidates,
                    scene_seed,
                )
                selected_original = unique_indices[selected_unique]
                reasons_original = {
                    int(unique_indices[index]): values for index, values in reasons_unique.items()
                }
                scene = loader.get_scene_from_token(token)
                manifest = _candidate_manifest(
                    token,
                    str(log_name),
                    str(scene.scene_metadata.map_name),
                    int(export_row.fold),
                    proposals,
                    baseline_scores,
                    baseline_factors,
                    factor_names,
                    selected_original,
                    reasons_original,
                    scene_seed,
                )
                # Translate original proposal indices back into the de-duplicated score tensor.
                original_to_unique = {int(original): index for index, original in enumerate(unique_indices)}
                score_indices = np.asarray(
                    [original_to_unique[int(index)] for index in selected_original], dtype=np.int64
                )
                selected_score = _slice_score(raw_score, score_indices)
                scored = pd.DataFrame(_rows_from_score(manifest, selected_score, raw_runtime))
                old = build_scene_targets(scene, cache, scored, paths, 16, 64)
                arrays = split_v3_targets(scene, old, paths, args.actor_slots)
                ensure_dir(log_target_dir)
                np.savez_compressed(log_target_dir / f"{token}.npz", **arrays)
                candidate_frames.append(manifest)
                metric_rows.extend(scored.to_dict("records"))
                raw_rows.extend(
                    _raw_score_rows(
                        token,
                        str(log_name),
                        int(export_row.fold),
                        baseline_scores[unique_indices],
                        baseline_factors[unique_indices],
                        factor_names,
                        raw_score,
                        unique_indices,
                    )
                )
                stats.update(
                    {
                        "scene_token": token,
                        "log_name": str(log_name),
                        "duplicate_proposal_count": int(len(proposals) - len(unique_indices)),
                    }
                )
                selection_stats.append(stats)
                success += 1
                processed_scenes += 1
            except Exception as exc:
                failures.append(
                    {"scene_token": token, "log_name": str(log_name), "error": f"{type(exc).__name__}: {exc}"}
                )
        candidates = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
        metrics = pd.DataFrame(metric_rows)
        raw_metrics = pd.DataFrame(raw_rows)
        write_parquet(candidates, manifest_path)
        write_parquet(metrics, metric_path)
        write_parquet(raw_metrics, raw_metric_path)
        log_summary = {
            "complete": success == len(log_exports),
            "scene_count": int(len(log_exports)),
            "success_count": success,
            "candidate_count": int(len(candidates)),
            "raw_metric_count": int(len(raw_metrics)),
            "selection_stats": selection_stats,
            "failure_count": int(len(log_exports) - success),
        }
        write_json(summary_path, log_summary)
        if (log_position + 1) % 25 == 0:
            print(
                f"shard={args.shard_index} logs={log_position + 1}/{len(selected_logs)} "
                f"scenes={processed_scenes} failures={len(failures)}",
                flush=True,
            )
    result = {
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "assigned_logs": len(selected_logs),
        "assigned_scenes": int(len(exports)),
        "processed_scenes_this_run": processed_scenes,
        "failure_count_this_run": len(failures),
        "failure_examples": failures[:20],
        "elapsed_seconds": time.time() - started,
    }
    write_json(cache_dir / "model_candidates" / f"scoring_shard_{args.shard_index:04d}.json", result)
    return result


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    report_dir = ensure_dir(args.output_dir)
    cache_dir = ensure_dir(args.cache_dir)
    export_manifest = pd.read_parquet(report_dir / "candidates/episode_drive_raw_exports.parquet")
    manifests = []
    metrics = []
    raw_metrics = []
    stats = []
    target_count = 0
    complete_logs = 0
    for log_name, group in export_manifest.groupby("log_name", sort=True):
        safe_name = str(log_name).replace("/", "_")
        paths = {
            "manifest": cache_dir / "model_candidates" / "selected_by_log" / f"{safe_name}.parquet",
            "metric": cache_dir / "model_candidates" / "metrics_by_log" / f"{safe_name}.parquet",
            "raw": cache_dir / "model_candidates" / "raw_metrics_by_log" / f"{safe_name}.parquet",
            "summary": cache_dir / "model_candidates" / "log_summaries" / f"{safe_name}.json",
        }
        if all(path.is_file() for path in paths.values()):
            summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
            manifests.append(pd.read_parquet(paths["manifest"]))
            metrics.append(pd.read_parquet(paths["metric"]))
            raw_metrics.append(pd.read_parquet(paths["raw"]))
            stats.extend(summary.get("selection_stats", []))
            target_count += len(
                list((cache_dir / "model_candidates" / "targets_v3" / safe_name).glob("*.npz"))
            )
            complete_logs += int(bool(summary.get("complete")))
    manifest = pd.concat(manifests, ignore_index=True) if manifests else pd.DataFrame()
    metric = pd.concat(metrics, ignore_index=True) if metrics else pd.DataFrame()
    raw_metric = pd.concat(raw_metrics, ignore_index=True) if raw_metrics else pd.DataFrame()
    stat_frame = pd.DataFrame(stats)
    fold_manifest = pd.read_parquet(report_dir / "balanced_scene_manifest.parquet")
    fold_by_log = fold_manifest.groupby("log_name").fold.first().to_dict()
    for frame in (manifest, metric, raw_metric):
        if len(frame):
            frame["fold"] = frame.log_name.map(fold_by_log).astype(int)
    # Full simulated-state columns are training caches, not lightweight Git
    # artifacts. Keep them outside reports and expose scalar-only audit tables.
    write_parquet(manifest, cache_dir / "model_candidates/combined_selected_manifest.parquet")
    write_parquet(metric, cache_dir / "model_candidates/combined_selected_metrics.parquet")
    write_parquet(raw_metric, cache_dir / "model_candidates/combined_raw_metrics.parquet")
    write_parquet(manifest, report_dir / "candidates/episode_drive_candidate_manifest.parquet")
    metric_scalar_columns = [
        column
        for column in (
            "scene_token", "log_name", "fold", "candidate_index",
            "original_proposal_index", "baseline_scorer_score",
            "aggregate_score", "no_at_fault_collision", "dac", "ddc",
            "progress", "raw_progress_m", "ttc", "comfort",
            "first_collision_time_s", "first_ttc_violation_time_s",
        )
        if column in metric
    ]
    write_parquet(
        metric[metric_scalar_columns],
        report_dir / "candidates/episode_drive_candidate_metrics.parquet",
    )
    write_parquet(stat_frame, report_dir / "candidates/episode_drive_selection_stats.parquet")
    expected_scenes = len(export_manifest)
    coverage = manifest.scene_token.nunique() / expected_scenes if expected_scenes and len(manifest) else 0.0
    selection_rows = []
    if len(metric) and len(raw_metric):
        for token, selected_group in metric.groupby("scene_token", sort=False):
            raw_group = raw_metric[raw_metric.scene_token == token]
            if raw_group.empty:
                continue
            selected_by_base = selected_group.iloc[
                int(np.argmax(selected_group.baseline_scorer_score.to_numpy()))
            ]
            raw_by_base = raw_group.iloc[
                int(np.argmax(raw_group.baseline_scorer_score.to_numpy()))
            ]
            selected_best = float(selected_group.aggregate_score.max())
            raw_best = float(raw_group.official_aggregate_score.max())
            selection_rows.append(
                {
                    "scene_token": token,
                    "log_name": str(selected_group.log_name.iloc[0]),
                    "fold": int(selected_group.fold.iloc[0]),
                    "baseline_selected_original_index": int(raw_by_base.original_proposal_index),
                    "baseline_selected_official_score_raw64": float(raw_by_base.official_aggregate_score),
                    "baseline_selected_official_score_selected16": float(selected_by_base.aggregate_score),
                    "random_expected_official_score_selected16": float(selected_group.aggregate_score.mean()),
                    "oracle_best_official_score_selected16": selected_best,
                    "oracle_best_official_score_raw64": raw_best,
                    "baseline_top1_regret_selected16": selected_best - float(selected_by_base.aggregate_score),
                    "baseline_top1_regret_raw64": raw_best - float(raw_by_base.official_aggregate_score),
                    "selection_best_of_raw_regret": raw_best - selected_best,
                    "baseline_selected_collision_free": float(selected_by_base.no_at_fault_collision),
                    "baseline_selected_ttc": float(selected_by_base.ttc),
                    "baseline_selected_dac": float(selected_by_base.dac),
                    "baseline_selected_ddc": float(selected_by_base.ddc),
                    "baseline_selected_progress": float(selected_by_base.progress),
                    "baseline_selected_comfort": float(selected_by_base.comfort),
                }
            )
    selection_frame = pd.DataFrame(selection_rows)
    write_parquet(
        selection_frame,
        report_dir / "candidates/episode_drive_selection_evaluation.parquet",
    )
    result = {
        "expected_scene_count": expected_scenes,
        "selected_scene_count": int(manifest.scene_token.nunique()) if len(manifest) else 0,
        "log_count": int(manifest.log_name.nunique()) if len(manifest) else 0,
        "complete_logs": complete_logs,
        "coverage": coverage,
        "candidates_per_scene": args.num_candidates,
        "selected_candidate_rows": len(manifest),
        "raw_scored_candidate_rows": len(raw_metric),
        "target_count": target_count,
        "mean_raw_to_selected_best_regret": float(stat_frame.selection_best_of_raw_regret.mean()) if len(stat_frame) else None,
        "mean_hard_outcome_pairs_retained": float(stat_frame.hard_outcome_count.mean()) if len(stat_frame) else None,
        "baseline_selected_mean_score": float(selection_frame.baseline_selected_official_score_selected16.mean()) if len(selection_frame) else None,
        "random_expected_mean_score": float(selection_frame.random_expected_official_score_selected16.mean()) if len(selection_frame) else None,
        "oracle_best_selected_mean_score": float(selection_frame.oracle_best_official_score_selected16.mean()) if len(selection_frame) else None,
        "oracle_best_raw_mean_score": float(selection_frame.oracle_best_official_score_raw64.mean()) if len(selection_frame) else None,
        "baseline_selected_top1_regret": float(selection_frame.baseline_top1_regret_selected16.mean()) if len(selection_frame) else None,
        "gt_forced": False,
    }
    write_json(report_dir / "candidates/model_candidate_bank_summary.json", result)
    write_markdown(
        report_dir / "MODEL_CANDIDATE_AUDIT.md",
        f"""# EpisodeDrive Model Candidate Audit

- Frozen checkpoint: `{EXPECTED_BASE_SHA256}`
- Exported/scored scenes: {result['selected_scene_count']:,}/{expected_scenes:,} ({coverage:.3%})
- Logs: {result['log_count']:,}; raw proposals / retained candidates: 64 / {args.num_candidates}
- Raw candidates scored offline: {len(raw_metric):,}
- Selected candidates scored/targeted: {len(metric):,} / {target_count:,} scenes
- Mean raw-64 to selected-16 oracle regret: {result['mean_raw_to_selected_best_regret']}
- Mean close-geometry/different-outcome candidates retained: {result['mean_hard_outcome_pairs_retained']}
- Mean baseline-selected / random / selected-16 oracle score: {result['baseline_selected_mean_score']} / {result['random_expected_mean_score']} / {result['oracle_best_selected_mean_score']}
- Selected-16 oracle headroom (mean baseline Top-1 regret): {result['baseline_selected_top1_regret']}
- Ground truth forcibly inserted: no

K=16 combines original-scorer-high proposals, farthest-point trajectory
diversity and close-geometry proposals with different offline outcomes. Candidate
order is deterministically shuffled. Official metrics are used only to build and
evaluate the offline bank; they are absent from deployable model inputs.

The existing `configs/base_model_navtest.yaml` file is used only to instantiate
the frozen EpisodeDrive architecture/checkpoint. Every exported sample is
selected from the legal trainval manifest and `feature_cache_navtrain_full`;
no navtest/navhard scene or reactive-response cache is opened.
""",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("shard", "aggregate"), default="shard")
    parser.add_argument("--split", choices=("train", "trainval"), default="trainval")
    parser.add_argument("--num-candidates", type=int, default=16)
    parser.add_argument("--actor-slots", type=int, default=16)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR / "all")
    args = parser.parse_args()
    result = aggregate(args) if args.mode == "aggregate" else run_shard(args)
    print(json.dumps(result, sort_keys=True))
    append_command(
        args.output_dir,
        "python -m tools.shared_future_candidate_consequence.build_model_candidate_bank "
        + " ".join(sys.argv[1:]),
    )


if __name__ == "__main__":
    main()
