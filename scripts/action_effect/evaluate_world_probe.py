#!/usr/bin/env python3
"""Evaluate factual world probes for action-conditioning collapse."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.action_effect.cache_io import write_json, write_jsonl  # noqa: E402
from research.action_effect.metrics import compute_action_collapse_metrics  # noqa: E402
from research.action_effect.probe_data import (  # noqa: E402
    HARD_TARGET_FIELDS,
    load_probe_arrays,
)


METRIC_ORDER = (
    "factual_prediction_error",
    "action_shuffle_gap",
    "candidate_sensitivity",
    "action_gap",
    "equivalence_leakage",
    "effect_alignment",
    "risk_false_safe_rate",
)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"configuration must be a mapping: {path}")
    return value


def _environment_root(variable: str) -> Path:
    value = os.environ.get(variable, "").strip()
    if not value:
        raise ValueError(f"source load_env.sh or set {variable}")
    return Path(value).resolve()


def _resolve_cache(explicit: Path | None, relative: str) -> Path:
    return explicit.resolve() if explicit is not None else _environment_root("ACTION_EFFECT_CACHE_ROOT") / relative


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _plot_metric_summary(summary: list[dict[str, Any]], output: Path) -> None:
    selected = [
        row
        for row in summary
        if row["control"]
        in {
            "constant_mean_control",
            "random_untrained_probe",
            "scene_only_probe",
            "trajectory_only_probe",
            "scene_action_probe",
            "shuffled_action_probe",
            "same_parameter_no_action",
        }
    ]
    metrics = ("factual_prediction_error", "action_shuffle_gap", "effect_alignment", "risk_false_safe_rate")
    figure, axes = plt.subplots(2, 2, figsize=(13, 8))
    for axis, metric in zip(axes.flat, metrics):
        values = np.asarray([row[f"{metric}_mean"] for row in selected], dtype=float)
        low = np.asarray([row[f"{metric}_ci_low"] for row in selected], dtype=float)
        high = np.asarray([row[f"{metric}_ci_high"] for row in selected], dtype=float)
        error = np.vstack((np.maximum(values - low, 0), np.maximum(high - values, 0)))
        axis.bar(np.arange(len(selected)), values, yerr=error, capsize=3)
        axis.set_title(metric.replace("_", " "))
        axis.set_xticks(np.arange(len(selected)))
        axis.set_xticklabels([row["control"].replace("_probe", "") for row in selected], rotation=40, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_effect_scatter(details: dict[str, Any], output: Path, seed: int) -> None:
    true = details["pair_true_distance"]
    predicted = details["pair_predicted_distance"]
    rng = np.random.default_rng(seed)
    if len(true) > 5000:
        indices = rng.choice(len(true), size=5000, replace=False)
        true, predicted = true[indices], predicted[indices]
    figure, axis = plt.subplots(figsize=(7, 5.5))
    axis.scatter(true, predicted, s=8, alpha=0.25, edgecolors="none")
    axis.set_xlabel("Replay-grounded consequence distance")
    axis.set_ylabel("Predicted consequence-output distance")
    axis.set_title("Factual-only scene-action probe on unseen candidates")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _gate2(summary_by_control: dict[str, dict[str, Any]]) -> dict[str, Any]:
    scene_action = summary_by_control["scene_action_probe"]
    random_control = summary_by_control["random_untrained_probe"]
    constant = summary_by_control["constant_mean_control"]
    factual = float(scene_action["factual_prediction_error_mean"])
    reference = min(
        float(random_control["factual_prediction_error_mean"]),
        float(constant["factual_prediction_error_mean"]),
    )
    learnable = factual < 0.95 * reference
    shuffle_small = abs(float(scene_action["action_shuffle_gap_mean"])) <= max(0.02 * factual, 0.005)
    alignment_low = float(scene_action["effect_alignment_mean"]) < 0.25
    gap = float(scene_action["action_gap_mean"])
    leakage = float(scene_action["equivalence_leakage_mean"])
    divergent_compressed = gap <= 1.25 * max(leakage, 1.0e-8)
    false_safe_high = float(scene_action["risk_false_safe_rate_mean"]) >= 0.20
    collapse_indicators = sum((shuffle_small, alignment_low, divergent_compressed, false_safe_high))
    passed = bool(learnable and collapse_indicators >= 3)
    return {
        "decision": "PASS" if passed else "FAIL",
        "supports_action_collapse": passed,
        "exploratory_thresholds": {
            "learnability_relative_improvement": 0.05,
            "shuffle_gap_max_fraction_of_factual_error": 0.02,
            "shuffle_gap_absolute_floor": 0.005,
            "effect_alignment_upper": 0.25,
            "action_gap_to_leakage_ratio_upper": 1.25,
            "false_safe_rate_lower": 0.20,
            "required_collapse_indicators": 3,
        },
        "evidence": {
            "factual_probe_learnable": learnable,
            "action_shuffle_gap_small": shuffle_small,
            "effect_alignment_low": alignment_low,
            "divergent_outputs_compressed": divergent_compressed,
            "false_safe_rate_high": false_safe_high,
            "collapse_indicator_count": collapse_indicators,
        },
        "note": "Thresholds are pilot diagnostics, not official NAVSIM acceptance criteria.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPOSITORY_ROOT / "configs/action_effect/factual_only.yaml"
    )
    parser.add_argument("--candidate-cache", type=Path)
    parser.add_argument("--consequence-cache", type=Path)
    parser.add_argument("--pair-cache", type=Path)
    parser.add_argument("--scene-feature-cache", type=Path)
    parser.add_argument("--probe-dir", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--bootstrap-samples", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_yaml(args.config.resolve())
    data = config["data"]
    candidate_cache = _resolve_cache(args.candidate_cache, str(data["candidate_cache"]))
    consequence_cache = _resolve_cache(args.consequence_cache, str(data["consequence_cache"]))
    pair_cache = _resolve_cache(args.pair_cache, str(data["pair_cache"]))
    scene_feature_cache = _resolve_cache(args.scene_feature_cache, str(data["scene_feature_cache"]))
    probe_dir = (
        args.probe_dir.resolve()
        if args.probe_dir is not None
        else _environment_root("ACTION_EFFECT_OUTPUT_ROOT") / "factual_only" / "pilot_tiny"
    )
    report_dir = (
        args.report_dir.resolve()
        if args.report_dir is not None
        else REPOSITORY_ROOT / "reports/action_effect_world_model/action_collapse_artifacts"
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    with (probe_dir / "split.json").open("r", encoding="utf-8") as stream:
        split = json.load(stream)
    fit_scenes, heldout_scenes = split["fit"], split["heldout"]
    arrays, _, _, scales, _, _ = load_probe_arrays(
        candidate_cache=candidate_cache,
        consequence_cache=consequence_cache,
        scene_feature_cache=scene_feature_cache,
        fit_scene_ids=fit_scenes,
        assumption=str(data["target_assumption"]),
    )
    eligible_scenes = sorted(
        set(arrays.scene_ids[arrays.accepted & arrays.anchor].tolist())
    )
    if set(fit_scenes).intersection(heldout_scenes) or set(fit_scenes + heldout_scenes) != set(eligible_scenes):
        raise RuntimeError("saved probe split is not a disjoint partition of eligible factual scenes")
    bootstrap_samples = int(
        args.bootstrap_samples
        if args.bootstrap_samples is not None
        else config["experiment"]["bootstrap_samples"]
    )
    confidence = float(config["experiment"]["bootstrap_confidence"])
    risk = config["risk"]

    run_rows: list[dict[str, Any]] = []
    details_by_run: dict[tuple[str, int], dict[str, Any]] = {}
    controls = sorted(path.name for path in probe_dir.iterdir() if path.is_dir())
    for control in controls:
        for seed_dir in sorted((probe_dir / control).glob("seed_*")):
            prediction_path = seed_dir / "predictions.npz"
            result_path = seed_dir / "result.json"
            if not prediction_path.is_file() or not result_path.is_file():
                continue
            seed = int(seed_dir.name.removeprefix("seed_"))
            with np.load(prediction_path) as payload:
                prediction = np.asarray(payload["consequence_prediction"], dtype=np.float32)
            intervals, details = compute_action_collapse_metrics(
                arrays=arrays,
                raw_prediction=prediction,
                heldout_scene_ids=heldout_scenes,
                pair_path=pair_cache / "pairs.jsonl",
                scales=scales,
                low_ttc_seconds=float(risk["low_ttc_seconds"]),
                safe_threshold=float(risk["hard_safe_threshold"]),
                bootstrap_samples=bootstrap_samples,
                confidence=confidence,
                seed=seed,
            )
            with result_path.open("r", encoding="utf-8") as stream:
                training_result = json.load(stream)
            row: dict[str, Any] = {
                "control": control,
                "seed": seed,
                "total_parameters": training_result["total_parameters"],
                "trainable_parameters": training_result["trainable_parameters"],
                "best_validation_loss": training_result["best_validation_loss"],
            }
            for name in METRIC_ORDER:
                interval = intervals[name]
                row[f"{name}_point"] = interval.point
                row[f"{name}_ci_low"] = interval.ci_low
                row[f"{name}_ci_high"] = interval.ci_high
            run_rows.append(row)
            details_by_run[(control, seed)] = details
            write_json(
                seed_dir / "collapse_metrics.json",
                {name: asdict(value) for name, value in intervals.items()},
            )
    if not run_rows:
        raise RuntimeError(f"no completed prediction runs found under {probe_dir}")
    _write_csv(report_dir / "metrics_by_run.csv", run_rows)

    summary_rows: list[dict[str, Any]] = []
    for control in sorted({row["control"] for row in run_rows}):
        selected = [row for row in run_rows if row["control"] == control]
        summary: dict[str, Any] = {
            "control": control,
            "seeds": len(selected),
            "total_parameters": selected[0]["total_parameters"],
            "trainable_parameters": selected[0]["trainable_parameters"],
        }
        for metric in METRIC_ORDER:
            points = np.asarray([row[f"{metric}_point"] for row in selected], dtype=float)
            summary[f"{metric}_mean"] = float(np.nanmean(points))
            bootstrap_arrays = [
                details_by_run[(row["control"], int(row["seed"]))]["bootstrap_values"][metric]
                for row in selected
            ]
            minimum = min(len(values) for values in bootstrap_arrays)
            if minimum:
                seed_averaged = np.nanmean(
                    np.stack([values[:minimum] for values in bootstrap_arrays]), axis=0
                )
                alpha = (1.0 - confidence) / 2.0
                low, high = np.nanquantile(seed_averaged, [alpha, 1.0 - alpha])
            else:
                low = high = float("nan")
            summary[f"{metric}_ci_low"] = float(low)
            summary[f"{metric}_ci_high"] = float(high)
        summary_rows.append(summary)
    _write_csv(report_dir / "metrics_summary.csv", summary_rows)
    summary_by_control = {row["control"]: row for row in summary_rows}
    required = {
        "scene_action_probe",
        "random_untrained_probe",
        "constant_mean_control",
    }
    if not required <= set(summary_by_control):
        raise RuntimeError(f"Gate 2 controls are incomplete: {sorted(required - set(summary_by_control))}")
    gate = _gate2(summary_by_control)
    write_json(report_dir / "gate2.json", gate)

    _plot_metric_summary(summary_rows, report_dir / "collapse_metric_comparison.png")
    scene_action_runs = sorted(key for key in details_by_run if key[0] == "scene_action_probe")
    representative_key = scene_action_runs[0]
    representative = details_by_run[representative_key]
    _plot_effect_scatter(
        representative,
        report_dir / "predicted_vs_true_effect_distance.png",
        int(config["experiment"]["seed"]),
    )

    false_safe_indices = np.flatnonzero(representative["false_safe"])
    failure_rows: list[dict[str, Any]] = []
    candidate_error = representative["candidate_error"]
    for index in false_safe_indices[np.argsort(candidate_error[false_safe_indices])[::-1][:50]]:
        failure_rows.append(
            {
                "scene_id": str(arrays.scene_ids[index]),
                "candidate_id": str(arrays.candidate_ids[index]),
                "candidate_index": int(arrays.candidate_indices[index]),
                "candidate_error": float(candidate_error[index]),
                "true_no_at_fault_collision": float(arrays.raw_hard_targets[index, 0]),
                "true_dac": float(arrays.raw_hard_targets[index, 1]),
                "true_dynamic_collision": float(arrays.raw_hard_targets[index, 5]),
                "true_ttc_time_s": float(arrays.raw_soft_targets[index, 0]),
                "failure_type": "factual_prediction_accurate_but_candidate_false_safe",
            }
        )
    write_jsonl(report_dir / "false_safe_examples.jsonl", failure_rows)

    report_path = REPOSITORY_ROOT / "reports/action_effect_world_model/action_collapse_diagnosis.md"
    lines = [
        "# Action-conditioning collapse diagnosis",
        "",
        "## Scope and isolation",
        "",
        "This Phase 5 pilot freezes Qwen and DiT, caches only current-observation action-query tokens, and trains a lightweight consequence probe on one factual expert trajectory per fit scene. Candidate consequences are used only for held-out evaluation. This is a quick consequence-probe result, not yet a structured-future world-model conclusion.",
        "",
        f"- Fit / held-out scenes: {len(fit_scenes)} / {len(heldout_scenes)} (scene-disjoint deterministic split).",
        f"- Seeds: {len({row['seed'] for row in run_rows})}.",
        f"- Bootstrap: {bootstrap_samples} scene-clustered resamples at {confidence:.0%} confidence.",
        "- Target statistics: factual anchors in the fit scenes only.",
        "- World input: current images, navigation instruction, current ego state, and the candidate trajectory; no logged future enters Qwen or the probe condition.",
        "",
        "## Results",
        "",
        "Distances below are computed on predicted consequence outputs (hard probabilities plus normalized soft predictions), so arbitrary unsupervised trajectory-encoder variation cannot masquerade as action sensitivity.",
        "",
        "| Method | Factual error ↓ | Shuffle gap ↑ | Candidate sensitivity ↑ | Action gap ↑ | Equivalence leakage ↓ | Effect alignment ↑ | False-safe ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        def cell(metric: str) -> str:
            return f"{row[f'{metric}_mean']:.4f} [{row[f'{metric}_ci_low']:.4f}, {row[f'{metric}_ci_high']:.4f}]"
        lines.append(
            f"| {row['control']} | {cell('factual_prediction_error')} | {cell('action_shuffle_gap')} | {cell('candidate_sensitivity')} | {cell('action_gap')} | {cell('equivalence_leakage')} | {cell('effect_alignment')} | {cell('risk_false_safe_rate')} |"
        )
    evidence = gate["evidence"]
    lines.extend(
        [
            "",
            "## Gate 2",
            "",
            f"**Decision: {gate['decision']}.** The pilot {'supports' if gate['supports_action_collapse'] else 'does not support'} the action-collapse pattern required before implementing multi-candidate/AEE training.",
            "",
            f"- Factual probe learnable versus low-information controls: {evidence['factual_probe_learnable']}.",
            f"- Small action-shuffle gap: {evidence['action_shuffle_gap_small']}.",
            f"- Low effect alignment: {evidence['effect_alignment_low']}.",
            f"- Divergent outputs compressed relative to equivalence leakage: {evidence['divergent_outputs_compressed']}.",
            f"- High unsafe-candidate false-safe rate: {evidence['false_safe_rate_high']}.",
            "",
            "The numerical gate thresholds are exploratory diagnostics recorded in `action_collapse_artifacts/gate2.json`; they are not changes to NAVSIM evaluator semantics.",
            "",
            "## Artifacts",
            "",
            "- `action_collapse_artifacts/metrics_by_run.csv`",
            "- `action_collapse_artifacts/metrics_summary.csv`",
            "- `action_collapse_artifacts/collapse_metric_comparison.png`",
            "- `action_collapse_artifacts/predicted_vs_true_effect_distance.png`",
            "- `action_collapse_artifacts/false_safe_examples.jsonl`",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "gate2": gate}, indent=2))


if __name__ == "__main__":
    main()
