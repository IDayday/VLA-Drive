#!/usr/bin/env python3
"""Aggregate distributed O0–O13 oracle-ranking jobs and decide Gate C1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import (
    DEFAULT_CACHE_DIR,
    DEFAULT_REPORT_DIR,
    append_command,
    ensure_dir,
    log_bootstrap_ci,
    require_gate,
    update_gate,
    write_json,
    write_markdown,
)
from .run_oracle_decomposition import GROUP_NAMES, _aggregate_fold_results


def _discover_jobs(job_root: Path) -> list[Path]:
    jobs = sorted(path.parent for path in job_root.glob("*/job_summary.json"))
    if not jobs:
        raise FileNotFoundError(f"No completed oracle jobs under {job_root}")
    return jobs


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    require_gate(args.output_dir, "target_v3")
    report_dir = ensure_dir(args.output_dir)
    cache_dir = ensure_dir(args.cache_dir)
    job_root = cache_dir / "oracle_jobs"
    jobs = _discover_jobs(job_root)
    summaries = [json.loads((path / "job_summary.json").read_text(encoding="utf-8")) for path in jobs]
    scene_counts = {int(item["scene_count"]) for item in summaries}
    log_counts = {int(item["log_count"]) for item in summaries}
    candidate_counts = {int(item["candidates_per_scene"]) for item in summaries}
    if len(scene_counts) != 1 or len(log_counts) != 1 or len(candidate_counts) != 1:
        raise RuntimeError("Oracle jobs were generated from different datasets")
    fold_frames = [pd.read_csv(path / "oracle_fold_results.csv") for path in jobs]
    fold_frame = pd.concat(fold_frames, ignore_index=True)
    duplicated = fold_frame.duplicated(["group", "model", "fold"], keep=False)
    if duplicated.any():
        duplicate_keys = fold_frame.loc[duplicated, ["group", "model", "fold"]]
        raise RuntimeError(f"Duplicate oracle jobs: {duplicate_keys.drop_duplicates().to_dict('records')}")
    expected_models = sorted({model for item in summaries for model in item["models"]})
    expected_folds = list(range(args.num_folds))
    expected_keys = {
        (group, model, fold)
        for group in GROUP_NAMES
        for model in expected_models
        for fold in expected_folds
    }
    actual_keys = {
        (str(row.group), str(row.model), int(row.fold))
        for row in fold_frame.itertuples(index=False)
    }
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    if missing or extra:
        raise RuntimeError(f"Incomplete oracle job matrix: missing={missing[:20]}, extra={extra[:20]}")
    per_scene = pd.concat(
        [pd.read_parquet(path / "oracle_per_scene_results.parquet") for path in jobs],
        ignore_index=True,
    )
    calibration_paths = [path / "oracle_factor_calibration.csv" for path in jobs]
    if not all(path.is_file() for path in calibration_paths):
        raise RuntimeError("One or more oracle jobs are missing factor calibration bins")
    calibration = pd.concat(
        [pd.read_csv(path) for path in calibration_paths], ignore_index=True
    )
    if per_scene.duplicated(["scene_token", "group", "model"]).any():
        raise RuntimeError("Duplicate per-scene oracle predictions across jobs")
    heldout_frames = [
        pd.read_csv(path / "heldout_candidate_type_results.csv")
        for path in jobs
        if (path / "heldout_candidate_type_results.csv").is_file()
    ]
    if not heldout_frames:
        raise RuntimeError("No held-out candidate-family job was completed")
    heldout = pd.concat(heldout_frames, ignore_index=True).drop_duplicates(
        ["family", "fold", "group"], keep="last"
    )

    fold_frame = fold_frame.sort_values(["group", "model", "fold"])
    fold_frame.to_csv(report_dir / "oracle_fold_results.csv", index=False)
    calibration.to_csv(report_dir / "oracle_factor_calibration.csv", index=False)
    per_scene.to_parquet(
        cache_dir / "oracle_per_scene_results.parquet",
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    heldout.to_csv(report_dir / "heldout_candidate_type_results.csv", index=False)
    aggregated = _aggregate_fold_results(fold_frame)
    primary_model = "mlp" if "mlp" in expected_models else expected_models[0]
    primary = {group: aggregated[f"{group}:{primary_model}"] for group in GROUP_NAMES}
    dynamic_gain = primary["O8"]["pairwise_mean"] - primary["O3"]["pairwise_mean"]
    state_gain = primary["O9"]["pairwise_mean"] - primary["O3"]["pairwise_mean"]
    state_retention = state_gain / dynamic_gain if abs(dynamic_gain) > 1e-9 else float("nan")
    regret_reduction = (
        (primary["O3"]["top1_regret_mean"] - primary["O8"]["top1_regret_mean"])
        / max(primary["O3"]["top1_regret_mean"], 1e-9)
    )
    paired = per_scene[
        (per_scene.model == primary_model) & per_scene.group.isin(["O3", "O8"])
    ].pivot(index=["scene_token", "log_name"], columns="group", values="pairwise_accuracy").dropna()
    paired["dynamic_gain"] = paired.O8 - paired.O3
    # This is deliberately a different estimand from the mean of five fold
    # aggregate accuracies above: every complete log receives equal weight,
    # irrespective of how many non-tied scenes/pairs it contributes.  Report
    # its point estimate alongside the log-bootstrap interval so the interval
    # is never mistaken for a CI around ``dynamic_gain``.
    equal_log_dynamic_gain = float(
        paired.reset_index().groupby("log_name").dynamic_gain.mean().mean()
    )
    ci_low, ci_high = log_bootstrap_ci(
        paired.reset_index(), "dynamic_gain", seed=args.seed, samples=args.bootstrap_samples
    )
    family_gain = heldout[heldout.group == "gain"].copy()
    heldout_gain_mean = float(family_gain.pairwise_accuracy.mean())
    heldout_gain_worst = float(family_gain.pairwise_accuracy.min())
    controls_limit = primary["O3"]["pairwise_mean"] + args.control_tolerance
    factor_improvement = (
        primary["O8"]["collision_auroc_mean"] > primary["O3"]["collision_auroc_mean"] + 0.01
        or primary["O8"]["ttc_auroc_mean"] > primary["O3"]["ttc_auroc_mean"] + 0.01
    )
    criteria = {
        "dynamic_pairwise_gain_at_least_0p03": dynamic_gain >= 0.03,
        "bootstrap_ci_lower_above_zero": ci_low > 0,
        "top1_regret_reduction_at_least_20pct": regret_reduction >= 0.20,
        "collision_or_ttc_improved": factor_improvement,
        "within_scene_shuffle_gain_disappears": primary["O10"]["pairwise_mean"] <= controls_limit,
        "cross_scene_shuffle_gain_disappears": primary["O11"]["pairwise_mean"] <= controls_limit,
        "random_dimension_control_fails": primary["O12"]["pairwise_mean"] <= controls_limit,
        "repeated_static_control_fails": primary["O13"]["pairwise_mean"] <= controls_limit,
        "state_recomputed_risk_retention_at_least_0p40": state_retention >= 0.40,
        "every_heldout_candidate_type_has_positive_gain": heldout_gain_worst > 0,
    }
    passed = all(criteria.values())
    risk_only = (
        primary["O5"]["pairwise_mean"] > primary["O4"]["pairwise_mean"] + 0.02
        and state_retention < 0.40
    )
    interpretation = (
        "shared future state world-model supervision is supported"
        if passed
        else (
            "evaluation-metric distillation only; shared future state is not supported"
            if risk_only
            else "oracle dynamic evidence is incomplete under Gate C1"
        )
    )
    mlp_rows = fold_frame[fold_frame.model == "mlp"]
    parameter_ratio = (
        float(mlp_rows.parameter_count.max() / mlp_rows.parameter_count.min())
        if len(mlp_rows)
        else None
    )
    result = {
        "mode": "full-all-logs",
        "scene_count": scene_counts.pop(),
        "log_count": log_counts.pop(),
        "candidates_per_scene": candidate_counts.pop(),
        "folds": expected_folds,
        "models": expected_models,
        "primary_model": primary_model,
        "aggregated": aggregated,
        "primary": primary,
        "dynamic_gain": dynamic_gain,
        "equal_log_dynamic_gain": equal_log_dynamic_gain,
        "state_recomputed_risk_gain_retention": state_retention,
        "top1_regret_reduction": regret_reduction,
        "dynamic_gain_log_bootstrap_95ci": [ci_low, ci_high],
        "heldout_candidate_type_gain_mean": heldout_gain_mean,
        "heldout_candidate_type_gain_worst": heldout_gain_worst,
        "mlp_parameter_max_to_min_ratio": parameter_ratio,
        "criteria": criteria,
        "gate_c1": "PASS" if passed else "FAIL",
        "interpretation": interpretation,
        "leakage_audit": {
            "official_score_in_features": False,
            "official_factor_in_features": False,
            "official_values_used_as_targets_only": True,
        },
        "job_directories": [str(path) for path in jobs],
    }
    write_json(report_dir / "oracle_decomposition_results.json", result)
    update_gate(report_dir, "gate_c1", {"passed": passed, **result})
    rows = []
    for group in GROUP_NAMES:
        item = primary[group]
        rows.append(
            f"| {group} | {item['pairwise_mean']:.4f} ± {item['pairwise_std']:.4f} | "
            f"{item['pairwise_worst_fold']:.4f} | {item['top1_regret_mean']:.4f} | "
            f"{item['collision_auroc_mean']:.4f} | {item['ttc_auroc_mean']:.4f} |"
        )
    write_markdown(
        report_dir / "ORACLE_DECOMPOSITION_REPORT.md",
        f"""# Oracle Dynamic-value Decomposition

## Gate C1 (all eligible trainval logs): {'PASS' if passed else 'FAIL'}

| Group | Pairwise mean ± std | Worst fold | Top-1 regret | Collision AUROC | TTC AUROC |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

- Scenes/logs/K: {result['scene_count']:,} / {result['log_count']:,} / {result['candidates_per_scene']}
- Dynamic gain O8−O3: {dynamic_gain:.4f}
- Equal-log dynamic-gain point estimate: {equal_log_dynamic_gain:.4f}
- Equal-log bootstrap 95% CI: [{ci_low:.4f}, {ci_high:.4f}]
- Top-1 regret reduction: {regret_reduction:.2%}
- State/recomputed-risk retention R_state: {state_retention:.3f}
- Held-out-family gain mean/worst: {heldout_gain_mean:.4f} / {heldout_gain_worst:.4f}
- MLP parameter max/min ratio: {parameter_ratio:.3f}
- Interpretation: {interpretation}

O9 never receives collision/TTC labels: those risk features are recomputed from
actor-relative state and mask. O10/O11/O12/O13 are within-scene shuffle,
cross-scene shuffle, random-dimensional and repeated-static controls. Official
PDM aggregate/factor columns are isolated as targets and never enter O0–O13.
""",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--control-tolerance", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR / "all")
    args = parser.parse_args()
    result = aggregate(args)
    print(json.dumps(result, sort_keys=True))
    append_command(
        args.output_dir,
        "python -m tools.shared_future_candidate_consequence.aggregate_oracle_jobs "
        + " ".join(sys.argv[1:]),
    )


if __name__ == "__main__":
    main()
