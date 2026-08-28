#!/usr/bin/env python3
"""Audit and aggregate resumable all-log candidate/target shards."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .build_gate_c_targets import target_schema_v3
from .common import (
    DEFAULT_CACHE_DIR,
    DEFAULT_REPORT_DIR,
    append_command,
    ensure_dir,
    require_gate,
    update_gate,
    write_json,
    write_markdown,
)


REQUIRED_TARGET_KEYS = {
    "time_s",
    "candidate_index",
    "is_gt",
    "K_exact",
    "S_static",
    "D_state_summary",
    "D_state_actor",
    "D_state_actor_mask",
    "D_risk",
    "D_signal",
    "shared_actor_future_current_ego",
    "shared_actor_future_mask",
    "current_scene_features",
    "current_actor_state",
    "current_actor_mask",
}


def _safe_log_name(log_name: str) -> str:
    return str(log_name).replace("/", "_")


def _read_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _candidate_audit(
    path: Path,
    expected_tokens: set[str],
    k: int,
) -> dict[str, Any]:
    if not path.is_file():
        return {"candidate_rows": 0, "candidate_scenes": 0, "candidate_valid": False}
    frame = pd.read_parquet(
        path,
        columns=["scene_token", "candidate_index", "candidate_family", "is_gt"],
    )
    frame = frame[frame.scene_token.isin(expected_tokens)].copy()
    counts = frame.groupby("scene_token").size()
    gt_counts = frame.groupby("scene_token").is_gt.sum()
    index_valid = frame.groupby("scene_token").candidate_index.apply(
        lambda values: sorted(values.astype(int).tolist()) == list(range(k))
    )
    valid = (
        set(counts.index.astype(str)) == expected_tokens
        and bool((counts == k).all())
        and bool((gt_counts == 1).all())
        and bool(index_valid.all())
        and not frame.duplicated(["scene_token", "candidate_index"]).any()
    )
    return {
        "candidate_rows": int(len(frame)),
        "candidate_scenes": int(len(counts)),
        "candidate_valid": bool(valid),
        "gt_exactly_once_rate": float((gt_counts == 1).mean()) if len(gt_counts) else 0.0,
        "candidate_index_valid_rate": float(index_valid.mean()) if len(index_valid) else 0.0,
        "_family_counts": frame.candidate_family.value_counts().astype(int).to_dict(),
        "_gt_index_counts": (
            frame[frame.is_gt].candidate_index.astype(int).value_counts().astype(int).to_dict()
        ),
    }


def _metric_audit(path: Path, expected_tokens: set[str], k: int) -> dict[str, Any]:
    if not path.is_file():
        return {
            "metric_rows": 0,
            "metric_scenes": 0,
            "metric_valid": False,
            "scoring_success_rate": 0.0,
        }
    frame = pd.read_parquet(
        path,
        columns=["scene_token", "candidate_index", "scoring_success", "aggregate_score"],
    )
    frame = frame[frame.scene_token.isin(expected_tokens)].copy()
    counts = frame.groupby("scene_token").size()
    finite = np.isfinite(frame.aggregate_score.to_numpy(dtype=np.float64))
    success = frame.scoring_success.to_numpy(dtype=bool) & finite
    valid = (
        set(counts.index.astype(str)) == expected_tokens
        and bool((counts == k).all())
        and not frame.duplicated(["scene_token", "candidate_index"]).any()
    )
    return {
        "metric_rows": int(len(frame)),
        "metric_scenes": int(len(counts)),
        "metric_valid": bool(valid),
        "scoring_success_rate": float(success.mean()) if len(frame) else 0.0,
    }


def _sample_target_audit(paths: list[Path], k: int, actor_slots: int, samples: int) -> dict[str, Any]:
    if not paths:
        return {"sample_count": 0, "valid_rate": 0.0, "errors": ["no target files"]}
    indices = np.unique(np.linspace(0, len(paths) - 1, min(samples, len(paths)), dtype=int))
    errors: list[str] = []
    valid = 0
    forbidden = {"aggregate_score", "official_score", "pdm_score", "candidate_type", "candidate_family"}
    for index in indices:
        path = paths[int(index)]
        try:
            with np.load(path, allow_pickle=False) as target:
                keys = set(target.files)
                missing = REQUIRED_TARGET_KEYS - keys
                leaked = forbidden & keys
                shape_ok = (
                    target["candidate_index"].shape == (k,)
                    and target["K_exact"].shape[:2] == (k, 8)
                    and target["D_state_actor"].shape[:3] == (k, 8, actor_slots)
                    and target["shared_actor_future_current_ego"].shape[:2] == (8, actor_slots)
                    and target["current_actor_state"].shape == (actor_slots, 8)
                    and target["current_actor_mask"].shape == (actor_slots,)
                    and int(target["is_gt"].sum()) == 1
                )
                finite = all(
                    np.isfinite(target[key]).all()
                    for key in ("K_exact", "S_static", "D_state_summary", "D_risk", "D_signal")
                )
                if missing or leaked or not shape_ok or not finite:
                    raise ValueError(
                        f"missing={sorted(missing)} leaked={sorted(leaked)} "
                        f"shape_ok={shape_ok} finite={finite}"
                    )
            valid += 1
        except Exception as exc:  # preserve evidence and continue auditing
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
    return {
        "sample_count": int(len(indices)),
        "valid_count": int(valid),
        "valid_rate": float(valid / len(indices)),
        "errors": errors[:20],
    }


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    require_gate(args.output_dir, "gate_c0")
    report_dir = ensure_dir(args.output_dir)
    cache_dir = ensure_dir(args.cache_dir)
    manifest_path = report_dir / "balanced_scene_manifest.parquet"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = pd.read_parquet(manifest_path)
    expected_logs = manifest.groupby("log_name").size().sort_index()
    rows: list[dict[str, Any]] = []
    all_target_paths: list[Path] = []
    family_counts: Counter[str] = Counter()
    gt_index_counts: Counter[int] = Counter()
    for log_name, expected_scenes in expected_logs.items():
        log_manifest = manifest[manifest.log_name == log_name]
        expected_tokens = set(log_manifest.scene_token.astype(str))
        safe_name = _safe_log_name(str(log_name))
        summary = _read_summary(cache_dir / "log_summaries" / f"{safe_name}.json")
        candidate = _candidate_audit(
            cache_dir / "candidate_shards" / f"{safe_name}.parquet",
            expected_tokens,
            args.num_candidates,
        )
        family_counts.update(candidate.pop("_family_counts", {}))
        gt_index_counts.update(candidate.pop("_gt_index_counts", {}))
        metric = _metric_audit(
            cache_dir / "metric_shards" / f"{safe_name}.parquet",
            expected_tokens,
            args.num_candidates,
        )
        target_paths = [
            cache_dir / "targets_v3" / safe_name / f"{token}.npz"
            for token in sorted(expected_tokens)
            if (cache_dir / "targets_v3" / safe_name / f"{token}.npz").is_file()
        ]
        all_target_paths.extend(target_paths)
        target_count = len(target_paths)
        rows.append(
            {
                "log_name": str(log_name),
                "fold": int(manifest.loc[manifest.log_name == log_name, "fold"].iloc[0]),
                "expected_scenes": int(expected_scenes),
                "summary_present": bool(summary),
                "summary_complete": bool(summary.get("complete", False)),
                "summary_failure_count": int(summary.get("failure_count", expected_scenes)),
                "target_scenes": target_count,
                "target_coverage": target_count / int(expected_scenes),
                **candidate,
                **metric,
            }
        )
    coverage = pd.DataFrame(rows)
    coverage.to_csv(report_dir / "all_log_coverage.csv", index=False)
    # Keep the canonical deliverable name synchronized with the formal all-log
    # audit rather than leaving the earlier pilot scene-level file in place.
    coverage.to_csv(report_dir / "target_coverage_v3.csv", index=False)
    expected_scenes = int(coverage.expected_scenes.sum())
    expected_candidates = expected_scenes * args.num_candidates
    target_scenes = int(coverage.target_scenes.sum())
    candidate_rows = int(coverage.candidate_rows.sum())
    metric_rows = int(coverage.metric_rows.sum())
    weighted_success = (
        float(np.average(coverage.scoring_success_rate, weights=coverage.metric_rows))
        if metric_rows
        else 0.0
    )
    target_rate = target_scenes / expected_scenes if expected_scenes else 0.0
    candidate_rate = candidate_rows / expected_candidates if expected_candidates else 0.0
    metric_rate = metric_rows / expected_candidates if expected_candidates else 0.0
    sample_audit = _sample_target_audit(
        all_target_paths, args.num_candidates, args.actor_slots, args.sample_targets,
    )
    failure_examples: list[dict[str, str]] = []
    seen_failures: set[tuple[str, str]] = set()
    for summary_path in sorted(cache_dir.glob("shard_*_summary.json")):
        summary = _read_summary(summary_path)
        for failure in summary.get("failure_examples", []):
            key = (str(failure.get("log_name", "")), str(failure.get("scene_token", "")))
            if key in seen_failures:
                continue
            seen_failures.add(key)
            failure_examples.append(
                {
                    "log_name": key[0],
                    "scene_token": key[1],
                    "error": str(failure.get("error", "unknown failure")),
                }
            )
    complete_logs = int(
        (
            coverage.summary_complete
            & coverage.candidate_valid
            & coverage.metric_valid
            & (coverage.target_scenes == coverage.expected_scenes)
        ).sum()
    )
    passed = (
        target_rate > 0.98
        and candidate_rate > 0.98
        and metric_rate > 0.98
        and weighted_success > 0.98
        and sample_audit["valid_rate"] == 1.0
    )
    result = {
        "expected_logs": int(len(coverage)),
        "complete_logs": complete_logs,
        "expected_scenes": expected_scenes,
        "target_scenes": target_scenes,
        "target_coverage": target_rate,
        "expected_candidates": expected_candidates,
        "candidate_rows": candidate_rows,
        "candidate_coverage": candidate_rate,
        "metric_rows": metric_rows,
        "metric_coverage": metric_rate,
        "scoring_success_rate": weighted_success,
        "sample_target_audit": sample_audit,
        "failure_examples": failure_examples[:20],
        "passed": passed,
    }
    write_json(report_dir / "all_log_pipeline_summary.json", result)
    write_json(
        report_dir / "candidate_scoring_summary.json",
        {
            "scene_count": target_scenes,
            "expected_scene_count": expected_scenes,
            "candidate_rows": metric_rows,
            "expected_candidate_rows": expected_candidates,
            "candidate_coverage": metric_rate,
            "scoring_success_rate_among_computed_candidates": weighted_success,
            "traffic_policy": "non_reactive",
        },
    )
    controlled_summary = {
        "scene_count": target_scenes,
        "expected_scene_count": expected_scenes,
        "log_count": int(len(coverage)),
        "candidates_per_scene": args.num_candidates,
        "candidate_count": candidate_rows,
        "candidate_coverage": candidate_rate,
        "family_counts": dict(sorted(family_counts.items())),
        "gt_position_histogram": {
            str(index): int(gt_index_counts.get(index, 0))
            for index in range(args.num_candidates)
        },
        "gt_position_unique_count": int(sum(value > 0 for value in gt_index_counts.values())),
        "global_seed": 20260828,
        "source": "randomized smooth controlled candidates",
        "candidate_index_semantic": False,
    }
    write_json(report_dir / "candidates/controlled_candidate_summary.json", controlled_summary)
    write_markdown(
        report_dir / "candidates/CONTROLLED_CANDIDATE_AUDIT.md",
        f"""# Randomized Controlled Candidate Audit

- Formal scenes/logs: {target_scenes:,}/{len(coverage):,}
- Candidates: {candidate_rows:,}/{expected_candidates:,} ({candidate_rate:.3%})
- Candidates per successful scene: {args.num_candidates}
- GT appeared in {controlled_summary['gt_position_unique_count']} different candidate indices
- Global seed: 20260828; every scene uses a stable SHA256-derived subseed
- Families: {dict(sorted(family_counts.items()))}

All non-GT candidates use continuous scene-specific parameters, smooth temporal
profiles and deterministic de-duplication. Candidate order and GT position are
independently shuffled per scene, so candidate index does not identify a fixed
behavior. These are controlled perturbations, not true futures.
""",
    )
    schema = target_schema_v3(args.actor_slots)
    schema.update(
        {
            "actual_scene_success_rate": target_rate,
            "actual_scene_count": target_scenes,
            "expected_scene_count": expected_scenes,
            "target_cache_dir": str(cache_dir / "targets_v3"),
            "storage_layout": "one subdirectory per log; one compressed NPZ per scene",
        }
    )
    write_json(report_dir / "target_schema_v3.json", schema)
    update_gate(report_dir, "target_v3", {"passed": passed, **result})
    report_text = f"""# All-log Candidate Scoring and Target Construction

- Eligible logs/scenes: {len(coverage):,} / {expected_scenes:,}
- Complete logs: {complete_logs:,}/{len(coverage):,}
- Candidate rows: {candidate_rows:,}/{expected_candidates:,} ({candidate_rate:.3%})
- Metric rows: {metric_rows:,}/{expected_candidates:,} ({metric_rate:.3%})
- Official offline scoring success: {weighted_success:.3%}
- v3 target scenes: {target_scenes:,}/{expected_scenes:,} ({target_rate:.3%})
- Sampled target schema/leakage audit: {sample_audit['valid_count']}/{sample_audit['sample_count']}
- Audited failure examples: {failure_examples[:5]}
- Gate target-v3: {'PASS' if passed else 'INCOMPLETE/FAIL'}

The model-target NPZ files contain no official aggregate score, official factor,
candidate family or candidate index semantics beyond the deterministic tensor row
index. Official scores remain in physically separate offline-evaluation Parquet
files. Dynamic targets are candidate-conditioned relabeling of the shared logged
future under the non-reactive assumption.
"""
    write_markdown(report_dir / "ALL_LOG_TARGET_REPORT.md", report_text)
    write_markdown(report_dir / "TARGET_V3_REPORT.md", report_text)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-candidates", type=int, default=16)
    parser.add_argument("--actor-slots", type=int, default=16)
    parser.add_argument("--sample-targets", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR / "all")
    args = parser.parse_args()
    result = aggregate(args)
    print(json.dumps(result, sort_keys=True))
    append_command(
        args.output_dir,
        "python -m tools.shared_future_candidate_consequence.aggregate_all_log_pipeline "
        + " ".join(sys.argv[1:]),
    )


if __name__ == "__main__":
    main()
