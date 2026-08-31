#!/usr/bin/env python3
"""Compare two full Navtest proposal audits on exactly matched scene tokens.

The comparison is paired at scene level and bootstrapped at log level.  It is
intentionally separate from model inference and PDM scoring so that completed
audits remain immutable and can be re-analysed cheaply.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


METRICS = (
    "selected_pdms",
    "best_of_64_pdms",
    "scorer_regret",
    "mean_candidate_pdms",
    "median_candidate_pdms",
    "top5_oracle_mean_pdms",
    "fraction_candidates_pdms_ge_0_9",
    "fraction_candidates_pdms_ge_0_8",
    "unique_candidate_count",
    "mean_pairwise_endpoint_distance_m",
    "mean_pairwise_ade_m",
    "selected_no_at_fault_collisions",
    "selected_drivable_area_compliance",
    "selected_ego_progress",
    "selected_time_to_collision_within_bound",
    "selected_comfort",
    "selected_driving_direction_compliance",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-a", type=Path, required=True)
    parser.add_argument("--audit-b", type=Path, required=True)
    parser.add_argument("--label-a", default="candidate_a")
    parser.add_argument("--label-b", default="candidate_b")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-selected-csv", type=Path)
    parser.add_argument("--reference-parity-tolerance", type=float, default=1e-8)
    parser.add_argument("--allow-reference-mismatch", action="store_true")
    parser.add_argument("--expected-scene-count", type=int, default=12_146)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260831)
    return parser.parse_args()


def _load_audit(path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    summary = json.loads((path / "summary.json").read_text())
    frame = pd.read_csv(path / "per_scene_candidate_quality.csv")
    frame = frame.loc[frame["valid"].astype(bool)].copy()
    if frame["token"].duplicated().any():
        raise ValueError(f"Duplicate scene token in {path}")
    if not (frame["candidate_count"] == 64).all():
        raise ValueError(f"Non-64 candidate scene in {path}")
    if float(frame["selected_score_parity_abs"].max()) > 1e-8:
        raise ValueError(f"Selected-score parity failed in {path}")
    return summary, frame


def _cluster_bootstrap(
    merged: pd.DataFrame,
    column: str,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    per_log = merged.groupby("log_name", sort=True)[column].agg(["sum", "count"])
    sums = per_log["sum"].to_numpy(dtype=np.float64)
    counts = per_log["count"].to_numpy(dtype=np.float64)
    number_logs = len(per_log)
    draws = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 512):
        width = min(512, samples - start)
        indices = rng.integers(0, number_logs, size=(width, number_logs))
        draws[start : start + width] = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    return tuple(float(x) for x in np.quantile(draws, [0.025, 0.975]))


def _reference_parity(reference_csv: Path, merged: pd.DataFrame) -> dict[str, Any]:
    reference = pd.read_csv(reference_csv)
    reference = reference.loc[reference["token"].astype(str) != "average", ["token", "score"]]
    reference = reference.rename(columns={"score": "reference_score"})
    joined = merged[["token", "selected_pdms_b"]].merge(reference, on="token", how="outer", indicator=True)
    both = joined.loc[joined["_merge"] == "both"]
    error = np.abs(both["selected_pdms_b"].to_numpy() - both["reference_score"].to_numpy())
    return {
        "reference_path": str(reference_csv.resolve()),
        "matched_scene_count": int(len(both)),
        "audit_only_scene_count": int((joined["_merge"] == "left_only").sum()),
        "reference_only_scene_count": int((joined["_merge"] == "right_only").sum()),
        "mean_abs_error": float(error.mean()) if len(error) else None,
        "max_abs_error": float(error.max()) if len(error) else None,
    }


def main() -> None:
    args = _parse_args()
    summary_a, frame_a = _load_audit(args.audit_a)
    summary_b, frame_b = _load_audit(args.audit_b)

    columns = ["token", "log_name", *METRICS]
    merged = frame_a[columns].merge(
        frame_b[columns],
        on="token",
        how="outer",
        suffixes=("_a", "_b"),
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        counts = merged["_merge"].value_counts().to_dict()
        raise ValueError(f"Audit token sets differ: {counts}")
    if not (merged["log_name_a"] == merged["log_name_b"]).all():
        raise ValueError("log_name differs for matched scene token")
    merged = merged.rename(columns={"log_name_a": "log_name"}).drop(columns=["log_name_b", "_merge"])
    if len(merged) != args.expected_scene_count:
        raise ValueError(
            f"Expected {args.expected_scene_count} matched Navtest scenes, got {len(merged)}"
        )

    rows: list[dict[str, Any]] = []
    for metric_index, metric in enumerate(METRICS):
        a = merged[f"{metric}_a"].to_numpy(dtype=np.float64)
        b = merged[f"{metric}_b"].to_numpy(dtype=np.float64)
        delta_column = f"delta_{metric}"
        merged[delta_column] = a - b
        ci_low, ci_high = _cluster_bootstrap(
            merged,
            delta_column,
            args.bootstrap_samples,
            args.seed + metric_index,
        )
        rows.append(
            {
                "metric": metric,
                f"mean_{args.label_a}": float(a.mean()),
                f"mean_{args.label_b}": float(b.mean()),
                "paired_delta_a_minus_b": float((a - b).mean()),
                "log_bootstrap_ci95_low": ci_low,
                "log_bootstrap_ci95_high": ci_high,
                "a_better_scene_count": int((a > b + 1e-12).sum()),
                "tied_scene_count": int((np.abs(a - b) <= 1e-12).sum()),
                "b_better_scene_count": int((b > a + 1e-12).sum()),
            }
        )

    result: dict[str, Any] = {
        "label_a": args.label_a,
        "label_b": args.label_b,
        "scene_count": int(len(merged)),
        "log_count": int(merged["log_name"].nunique()),
        "audit_a": summary_a,
        "audit_b": summary_b,
        "comparisons": rows,
    }
    if args.reference_selected_csv is not None:
        result["released_selected_evaluator_parity"] = _reference_parity(
            args.reference_selected_csv, merged
        )
        parity = result["released_selected_evaluator_parity"]
        parity["tolerance"] = args.reference_parity_tolerance
        parity["passed"] = bool(
            parity["matched_scene_count"] == args.expected_scene_count
            and parity["audit_only_scene_count"] == 0
            and parity["reference_only_scene_count"] == 0
            and parity["max_abs_error"] is not None
            and parity["max_abs_error"] <= args.reference_parity_tolerance
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "comparison.json"
    table_path = args.output_dir / "paired_metrics.csv"
    report_path = args.output_dir / "COMPARISON.md"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    pd.DataFrame(rows).to_csv(table_path, index=False)

    key_metrics = {
        "selected_pdms",
        "best_of_64_pdms",
        "scorer_regret",
        "mean_candidate_pdms",
        "mean_pairwise_endpoint_distance_m",
        "mean_pairwise_ade_m",
    }
    report_lines = [
        "# Paired Navtest 64-candidate comparison",
        "",
        f"- Scene tokens: {len(merged)}",
        f"- Logs: {merged['log_name'].nunique()}",
        f"- A: `{args.label_a}`",
        f"- B: `{args.label_b}`",
        "- Delta convention: A - B",
        "- Confidence intervals: paired log-cluster bootstrap, 95%",
        "",
        "| Metric | A | B | Delta | 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["metric"] not in key_metrics:
            continue
        report_lines.append(
            "| {metric} | {a:.6f} | {b:.6f} | {d:+.6f} | [{lo:+.6f}, {hi:+.6f}] |".format(
                metric=row["metric"],
                a=row[f"mean_{args.label_a}"],
                b=row[f"mean_{args.label_b}"],
                d=row["paired_delta_a_minus_b"],
                lo=row["log_bootstrap_ci95_low"],
                hi=row["log_bootstrap_ci95_high"],
            )
        )
    if "released_selected_evaluator_parity" in result:
        parity = result["released_selected_evaluator_parity"]
        report_lines.extend(
            [
                "",
                "## Released selected-trajectory evaluator parity",
                "",
                f"- Matched scenes: {parity['matched_scene_count']}",
                f"- Mean absolute error: {parity['mean_abs_error']}",
                f"- Maximum absolute error: {parity['max_abs_error']}",
            ]
        )
    report_path.write_text("\n".join(report_lines) + "\n")
    print(json.dumps({"comparison": str(result_path), "report": str(report_path)}, indent=2))
    if (
        "released_selected_evaluator_parity" in result
        and not result["released_selected_evaluator_parity"]["passed"]
        and not args.allow_reference_mismatch
    ):
        raise RuntimeError(
            "Released-weight selected-trajectory parity failed; do not use this comparison. "
            "Check inference precision, attention backend, batch size, checkpoint class, and sensor root."
        )


if __name__ == "__main__":
    main()
