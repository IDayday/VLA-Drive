#!/usr/bin/env python3
"""Generate leakage-aware Gate C diagnostic figures from completed artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import DEFAULT_CACHE_DIR, DEFAULT_REPORT_DIR, append_command, ensure_dir


GROUP_LABELS = {
    "O3": "static baseline",
    "O4": "dynamic state",
    "O5": "direct physical risk",
    "O6": "signal",
    "O8": "full dynamic",
    "O9": "state + recomputed risk",
    "O10": "within-scene swap",
    "O11": "cross-scene shuffle",
    "O12": "random dimensions",
    "O13": "repeated static",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finish(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _not_run_figure(path: Path, title: str, reason: str) -> None:
    fig, axis = plt.subplots(figsize=(8.0, 4.5))
    axis.axis("off")
    axis.set_title(title, fontsize=13)
    axis.text(
        0.5,
        0.5,
        f"NOT RUN\n\n{reason}",
        ha="center",
        va="center",
        fontsize=12,
        transform=axis.transAxes,
    )
    _finish(fig, path)


def _gain_decomposition(result: dict[str, Any], path: Path) -> None:
    primary = result["primary"]
    baseline = primary["O3"]["pairwise_mean"]
    groups = ["O4", "O5", "O6", "O8", "O9"]
    gains = [primary[group]["pairwise_mean"] - baseline for group in groups]
    colors = ["#4C78A8" if value >= 0 else "#E45756" for value in gains]
    fig, axis = plt.subplots(figsize=(9.0, 4.8))
    axis.bar([GROUP_LABELS[group] for group in groups], gains, color=colors)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.axhline(0.03, color="#F58518", linewidth=1.2, linestyle="--", label="Gate +0.03")
    axis.set_ylabel("Pairwise accuracy gain over O3")
    axis.set_title("Static, dynamic-state, risk and signal oracle decomposition")
    axis.tick_params(axis="x", rotation=22)
    axis.legend()
    _finish(fig, path)


def _fold_accuracy(fold: pd.DataFrame, path: Path) -> None:
    groups = ["O3", "O4", "O5", "O8", "O9"]
    frame = fold[(fold.model == "mlp") & fold.group.isin(groups)]
    fig, axis = plt.subplots(figsize=(9.0, 4.8))
    for group in groups:
        part = frame[frame.group == group].sort_values("fold")
        axis.plot(part.fold, part.pairwise_accuracy, marker="o", label=GROUP_LABELS[group])
    axis.set_xticks(range(5))
    axis.set_xlabel("Log-disjoint validation fold")
    axis.set_ylabel("Non-tied pairwise accuracy")
    axis.set_title("Oracle pairwise accuracy by fold")
    axis.legend(ncol=2, fontsize=8)
    _finish(fig, path)


def _per_log_regret(per_scene_path: Path, path: Path) -> None:
    if not per_scene_path.is_file():
        _not_run_figure(path, "Per-log Top-1 regret", "Formal per-scene predictions are unavailable.")
        return
    frame = pd.read_parquet(per_scene_path)
    frame = frame[(frame.model == "mlp") & frame.group.isin(["O3", "O8"])]
    per_log = frame.groupby(["log_name", "group"], as_index=False).top1_regret.mean()
    pivot = per_log.pivot(index="log_name", columns="group", values="top1_regret").dropna()
    pivot = pivot.sort_values("O3").reset_index(drop=True)
    fig, axis = plt.subplots(figsize=(10.0, 4.8))
    axis.plot(pivot.index, pivot.O3, linewidth=0.9, label="O3 static")
    axis.plot(pivot.index, pivot.O8, linewidth=0.9, label="O8 full dynamic")
    axis.set_xlabel("Logs sorted by O3 regret")
    axis.set_ylabel("Mean Top-1 regret")
    axis.set_title("Per-log Top-1 regret (log-disjoint predictions)")
    axis.legend()
    _finish(fig, path)


def _oracle_predicted_gap(
    oracle: dict[str, Any], training: dict[str, Any] | None, path: Path
) -> None:
    primary = oracle["primary"]
    labels = ["O3 static oracle", "O8 dynamic oracle"]
    values = [primary["O3"]["pairwise_mean"], primary["O8"]["pairwise_mean"]]
    if training and training.get("predicted_consequence_pairwise") is not None:
        labels.append("predicted consequence")
        values.append(float(training["predicted_consequence_pairwise"]))
    fig, axis = plt.subplots(figsize=(8.0, 4.8))
    axis.bar(labels, values, color=["#9D755D", "#4C78A8", "#59A14F"][: len(values)])
    axis.set_ylim(max(0.0, min(values) - 0.08), min(1.0, max(values) + 0.08))
    axis.set_ylabel("Pairwise accuracy")
    axis.set_title("Oracle–predicted consequence gap")
    if len(values) == 2:
        axis.text(
            0.5,
            0.05,
            "Predicted model NOT RUN (blocked by Gate C1)",
            transform=axis.transAxes,
            ha="center",
            color="#B22222",
        )
    axis.tick_params(axis="x", rotation=12)
    _finish(fig, path)


def _shared_vs_direct(
    oracle: dict[str, Any], training: dict[str, Any] | None, path: Path
) -> None:
    if not training or training.get("status") != "COMPLETE":
        primary = oracle["primary"]
        labels = ["raw state O4", "direct risk O5", "state→risk O9"]
        values = [primary[group]["pairwise_mean"] for group in ("O4", "O5", "O9")]
        fig, axis = plt.subplots(figsize=(8.0, 4.8))
        axis.bar(labels, values, color=["#4C78A8", "#E45756", "#72B7B2"])
        axis.set_ylabel("Oracle pairwise accuracy")
        axis.set_title("Oracle targets only; direct/shared prediction NOT RUN")
        axis.text(
            0.5,
            0.04,
            "These are logged-future upper bounds, not current-observation models.",
            transform=axis.transAxes,
            ha="center",
            fontsize=9,
        )
        _finish(fig, path)
        return
    labels = ["direct candidate", "shared future factorized"]
    values = [training["direct_pairwise"], training["shared_pairwise"]]
    fig, axis = plt.subplots(figsize=(8.0, 4.8))
    axis.bar(labels, values, color=["#F58518", "#4C78A8"])
    axis.set_ylabel("Pairwise accuracy")
    axis.set_title("Direct consequence vs shared-future factorization")
    _finish(fig, path)


def _calibration(calibration_path: Path, path: Path) -> None:
    if not calibration_path.is_file():
        _not_run_figure(path, "Collision and TTC calibration", "Calibration bins are unavailable.")
        return
    frame = pd.read_csv(calibration_path)
    frame = frame[(frame.model == "mlp") & frame.group.isin(["O3", "O8"])]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4), sharex=True, sharey=True)
    for axis, factor in zip(axes, ("collision", "ttc")):
        axis.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=0.8)
        for group in ("O3", "O8"):
            part = frame[(frame.factor == factor) & (frame.group == group)].copy()
            part["weighted_probability"] = part.mean_probability * part["count"]
            part["weighted_rate"] = part.observed_rate * part["count"]
            aggregated = part.groupby("bin", as_index=False).agg(
                count=("count", "sum"),
                weighted_probability=("weighted_probability", "sum"),
                weighted_rate=("weighted_rate", "sum"),
            )
            aggregated = aggregated[aggregated["count"] > 0]
            x = aggregated.weighted_probability / aggregated["count"]
            y = aggregated.weighted_rate / aggregated["count"]
            axis.plot(x, y, marker="o", label=group)
        axis.set_title(factor.upper())
        axis.set_xlabel("Predicted probability")
        axis.legend()
    axes[0].set_ylabel("Observed rate")
    fig.suptitle("Oracle factor-head reliability (official factors are targets only)")
    _finish(fig, path)


def _controls(result: dict[str, Any], path: Path) -> None:
    groups = ["O3", "O8", "O10", "O11", "O12", "O13"]
    values = [result["primary"][group]["pairwise_mean"] for group in groups]
    fig, axis = plt.subplots(figsize=(9.0, 4.8))
    axis.bar([GROUP_LABELS[group] for group in groups], values, color="#4C78A8")
    axis.set_ylabel("Pairwise accuracy")
    axis.set_title("Future shuffle, candidate swap and dimensional controls")
    axis.tick_params(axis="x", rotation=20)
    _finish(fig, path)


def _candidate_swap(result: dict[str, Any], path: Path) -> None:
    values = [result["primary"][group]["pairwise_mean"] for group in ("O8", "O10")]
    fig, axis = plt.subplots(figsize=(7.0, 4.6))
    axis.bar(["matched consequence", "same-scene candidate swap"], values, color=["#59A14F", "#E45756"])
    axis.set_ylabel("Pairwise accuracy")
    axis.set_title(f"Candidate–consequence swap drop: {values[0] - values[1]:.4f}")
    _finish(fig, path)


def _heldout(path_csv: Path, path: Path) -> None:
    if not path_csv.is_file():
        _not_run_figure(path, "Held-out candidate type", "Held-out-family audit is unavailable.")
        return
    frame = pd.read_csv(path_csv)
    frame = frame[frame.group == "gain"].sort_values("pairwise_accuracy")
    fig, axis = plt.subplots(figsize=(9.0, 4.8))
    colors = ["#59A14F" if value > 0 else "#E45756" for value in frame.pairwise_accuracy]
    axis.barh(frame.family, frame.pairwise_accuracy, color=colors)
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("O8−O3 held-out-family pairwise gain")
    axis.set_title("Candidate-template holdout generalization")
    _finish(fig, path)


def _model_headroom(model: dict[str, Any], candidate_summary: dict[str, Any], path: Path) -> None:
    labels = ["random K16", "original scorer", "O8 oracle ranker", "best of K16"]
    values = [
        candidate_summary["random_expected_mean_score"],
        model["baseline_selected_mean_official_score"],
        model["o8_ranker_selected_mean_official_score"],
        model["best_of_16_mean_official_score"],
    ]
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    axis.bar(labels, values, color=["#BAB0AC", "#F58518", "#4C78A8", "#59A14F"])
    axis.set_ylim(max(0.0, min(values) - 0.08), 1.01)
    axis.set_ylabel("Mean offline official score")
    axis.set_title("Frozen EpisodeDrive proposals: best-of-K headroom")
    axis.tick_params(axis="x", rotation=15)
    _finish(fig, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    report_dir = args.output_dir
    cache_dir = args.cache_dir
    figure_dir = ensure_dir(report_dir / "figures")
    oracle = _load_json(report_dir / "oracle_decomposition_results.json")
    model = _load_json(report_dir / "model_candidate_oracle_results.json")
    candidate_summary = _load_json(report_dir / "candidates/model_candidate_bank_summary.json")
    training_path = report_dir / "training_results.json"
    training = _load_json(training_path) if training_path.is_file() else None
    fold = pd.read_csv(report_dir / "oracle_fold_results.csv")

    _gain_decomposition(oracle, figure_dir / "oracle_gain_decomposition.png")
    _fold_accuracy(fold, figure_dir / "fold_pairwise_accuracy.png")
    _per_log_regret(cache_dir / "oracle_per_scene_results.parquet", figure_dir / "per_log_top1_regret.png")
    _oracle_predicted_gap(oracle, training, figure_dir / "oracle_vs_predicted_gap.png")
    _shared_vs_direct(oracle, training, figure_dir / "shared_vs_direct.png")
    _calibration(report_dir / "oracle_factor_calibration.csv", figure_dir / "collision_ttc_calibration.png")
    _controls(oracle, figure_dir / "future_shuffle_control.png")
    _candidate_swap(oracle, figure_dir / "candidate_consequence_swap.png")
    _heldout(report_dir / "heldout_candidate_type_results.csv", figure_dir / "heldout_candidate_type.png")
    _model_headroom(model, candidate_summary, figure_dir / "model_candidate_best_of_k_headroom.png")
    if training and training.get("visual_anchor"):
        _not_run_figure(
            figure_dir / "visual_anchor_prediction_error.png",
            "Visual-anchor prediction error",
            "The visual-anchor plotting adapter has no cached per-horizon errors.",
        )
    else:
        _not_run_figure(
            figure_dir / "visual_anchor_prediction_error.png",
            "Visual-anchor prediction error",
            "GT future visual-anchor training was not entered after Gate C1.",
        )
    return {"figure_count": len(list(figure_dir.glob("*.png"))), "figure_dir": str(figure_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR / "all")
    args = parser.parse_args()
    print(json.dumps(run(args), sort_keys=True))
    append_command(
        args.output_dir,
        "python -m tools.shared_future_candidate_consequence.visualize_gate_c "
        + " ".join(sys.argv[1:]),
    )


if __name__ == "__main__":
    main()
