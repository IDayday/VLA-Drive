#!/usr/bin/env python3
"""Summarize oracle decomposition on frozen EpisodeDrive model proposals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .common import (
    DEFAULT_CACHE_DIR,
    DEFAULT_REPORT_DIR,
    append_command,
    ensure_dir,
    log_bootstrap_ci,
    write_json,
    write_markdown,
)
from .run_oracle_decomposition import _aggregate_fold_results


REQUIRED_GROUPS = ("O3", "O4", "O5", "O8", "O9", "O10", "O11", "O12", "O13")


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    report_dir = ensure_dir(args.output_dir)
    cache_dir = ensure_dir(args.cache_dir)
    job_dir = args.job_dir or cache_dir / "model_candidates/oracle_jobs/model_oracle"
    fold = pd.read_csv(job_dir / "oracle_fold_results.csv")
    per_scene = pd.read_parquet(job_dir / "oracle_per_scene_results.parquet")
    missing = {
        (group, model, fold_index)
        for group in REQUIRED_GROUPS
        for model in ("linear", "mlp")
        for fold_index in range(5)
    } - {
        (str(row.group), str(row.model), int(row.fold))
        for row in fold.itertuples(index=False)
    }
    if missing:
        raise RuntimeError(f"Incomplete model-candidate oracle matrix: {sorted(missing)[:20]}")
    aggregated = _aggregate_fold_results(fold)
    primary = {group: aggregated[f"{group}:mlp"] for group in REQUIRED_GROUPS}
    dynamic_gain = primary["O8"]["pairwise_mean"] - primary["O3"]["pairwise_mean"]
    raw_state_gain = primary["O4"]["pairwise_mean"] - primary["O3"]["pairwise_mean"]
    direct_risk_gain = primary["O5"]["pairwise_mean"] - primary["O3"]["pairwise_mean"]
    state_gain = primary["O9"]["pairwise_mean"] - primary["O3"]["pairwise_mean"]
    state_retention = state_gain / dynamic_gain if abs(dynamic_gain) > 1e-9 else None
    regret_reduction = (
        (primary["O3"]["top1_regret_mean"] - primary["O8"]["top1_regret_mean"])
        / max(primary["O3"]["top1_regret_mean"], 1e-9)
    )
    paired = per_scene[
        (per_scene.model == "mlp") & per_scene.group.isin(["O3", "O8"])
    ].pivot(index=["scene_token", "log_name"], columns="group", values="pairwise_accuracy").dropna()
    paired["dynamic_gain"] = paired.O8 - paired.O3
    ci = log_bootstrap_ci(paired.reset_index(), "dynamic_gain", samples=args.bootstrap_samples)
    selection_path = report_dir / "candidates/episode_drive_selection_evaluation.parquet"
    selection = pd.read_parquet(selection_path)
    baseline_score = float(selection.baseline_selected_official_score_selected16.mean())
    oracle_score = float(selection.oracle_best_official_score_selected16.mean())
    control_gain_max = max(
        primary[group]["pairwise_mean"] - primary["O3"]["pairwise_mean"]
        for group in ("O10", "O11", "O12", "O13")
    )
    result = {
        "scene_count": int(selection.scene_token.nunique()),
        "log_count": int(selection.log_name.nunique()),
        "candidates_per_scene": 16,
        "groups": list(REQUIRED_GROUPS),
        "models": ["linear", "mlp"],
        "aggregated": aggregated,
        "primary": primary,
        "dynamic_pairwise_gain": dynamic_gain,
        "raw_dynamic_state_pairwise_gain": raw_state_gain,
        "direct_physical_risk_pairwise_gain": direct_risk_gain,
        "state_recomputed_risk_pairwise_gain": state_gain,
        "maximum_control_pairwise_gain": control_gain_max,
        "dynamic_gain_log_bootstrap_95ci": list(ci),
        "state_recomputed_risk_gain_retention": state_retention,
        "oracle_top1_regret_reduction_O8_vs_O3": regret_reduction,
        "baseline_selected_mean_official_score": baseline_score,
        "best_of_16_mean_official_score": oracle_score,
        "best_of_16_headroom": oracle_score - baseline_score,
        "o8_ranker_selected_mean_official_score": (
            oracle_score - primary["O8"]["top1_regret_mean"]
        ),
        "o8_ranker_beats_original_scorer": (
            oracle_score - primary["O8"]["top1_regret_mean"] > baseline_score
        ),
        "ground_truth_inserted": False,
        "official_scores_are_targets_only": True,
    }
    write_json(report_dir / "model_candidate_oracle_results.json", result)
    rows = []
    for group in REQUIRED_GROUPS:
        item = primary[group]
        rows.append(
            f"| {group} | {item['pairwise_mean']:.4f} | {item['pairwise_worst_fold']:.4f} | "
            f"{item['top1_regret_mean']:.4f} | {item['collision_auroc_mean']:.4f} | {item['ttc_auroc_mean']:.4f} |"
        )
    write_markdown(
        report_dir / "MODEL_CANDIDATE_ORACLE_REPORT.md",
        f"""# Frozen EpisodeDrive Proposal Oracle Audit

| Group | Pairwise | Worst fold | Top-1 regret | Collision AUROC | TTC AUROC |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

- Scenes/logs/K: {result['scene_count']:,} / {result['log_count']:,} / 16
- Dynamic O8−O3 pairwise gain: {dynamic_gain:.4f}, log-bootstrap 95% CI [{ci[0]:.4f}, {ci[1]:.4f}]
- Raw state O4−O3 / direct risk O5−O3 / state+recomputed-risk O9−O3: {raw_state_gain:.4f} / {direct_risk_gain:.4f} / {state_gain:.4f}
- Largest shuffled/noise/repeated-static control gain: {control_gain_max:.4f}
- State/recomputed-risk retention: {state_retention}
- Baseline-selected / best-of-16 mean official score: {baseline_score:.4f} / {oracle_score:.4f}
- Best-of-16 headroom: {oracle_score - baseline_score:.4f}
- O8 ranker selected mean score: {oracle_score - primary['O8']['top1_regret_mean']:.4f}; it {'beats' if result['o8_ranker_beats_original_scorer'] else 'does not beat'} the original scorer
- Ground-truth proposal inserted: no

This is an offline upper-bound analysis on frozen proposals. Official outcomes
are labels only and are not available to a deployable scorer.
""",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--job-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR / "all")
    args = parser.parse_args()
    result = summarize(args)
    print(json.dumps(result, sort_keys=True))
    append_command(
        args.output_dir,
        "python -m tools.shared_future_candidate_consequence.summarize_model_candidate_oracle "
        + " ".join(sys.argv[1:]),
    )


if __name__ == "__main__":
    main()
