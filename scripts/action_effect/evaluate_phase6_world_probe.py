#!/usr/bin/env python3
"""Evaluate Phase-6 probes, bootstrap scenes, and make the Gate-3 decision."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.action_effect.cache_io import (  # noqa: E402
    content_hash,
    file_sha256,
    read_manifest,
    write_json,
    write_jsonl,
)
from research.action_effect.effect_tube import EFFECT_TUBE_CHANNELS  # noqa: E402
from research.action_effect.gate2_5 import (  # noqa: E402
    calibrate_risk_threshold,
    calibrated_scene_bootstrap,
    consequence_risk_score,
    intervals_to_json,
    unsafe_labels,
)
from research.action_effect.metrics import MetricInterval, decoded_prediction  # noqa: E402
from research.action_effect.phase6_metrics import (  # noqa: E402
    channel_metrics,
    decoded_effect_prediction,
    effect_action_shuffle_gap,
    effect_primary_channel_shuffle_gap,
    effect_channel_pair_sensitivity,
    gate3_conditions,
    intervals_as_json,
    latent_diagnostics,
    representation_metrics,
)
from research.action_effect.probe_data import (  # noqa: E402
    HARD_TARGET_FIELDS,
    SOFT_TARGET_FIELDS,
    iter_jsonl,
    load_probe_arrays,
    load_structured_targets,
)
from research.action_effect.reversal import build_reversal_cases, reversal_accuracy  # noqa: E402
from research.action_effect.world_probe import ActionEffectWorldProbe  # noqa: E402


TRAINED_METHODS = (
    "factual_only",
    "multi_candidate_absolute",
    "global_separation",
    "aee",
    "confidence_aee",
    "scene_only_control",
)
CONTROL_METHODS = ("mean_control", "zero_control")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"configuration must be a mapping: {path}")
    return value


def _root(variable: str) -> Path:
    value = os.environ.get(variable, "").strip()
    if not value:
        raise ValueError(f"source load_env.sh or set {variable}")
    return Path(value).resolve()


def _cache(explicit: Path | None, relative: str) -> Path:
    return explicit.resolve() if explicit is not None else _root("ACTION_EFFECT_CACHE_ROOT") / relative


def _model(config: Mapping[str, Any], method: str) -> ActionEffectWorldProbe:
    probe = config["probe"]
    return ActionEffectWorldProbe(
        scene_input_dim=int(probe["scene_input_dim"]),
        consequence_dim=len(HARD_TARGET_FIELDS) + len(SOFT_TARGET_FIELDS),
        latent_dim=int(probe["latent_dim"]),
        trajectory_input_dim=int(probe["trajectory_input_dim"]),
        trajectory_token_dim=int(probe["trajectory_token_dim"]),
        dropout=float(probe["dropout"]),
        structured_future_shape=tuple(int(value) for value in probe["structured_future_shape"]),
        input_mode="scene_only" if method == "scene_only_control" else "scene_action",
    )


@torch.no_grad()
def _predict(
    model: ActionEffectWorldProbe,
    indices: np.ndarray,
    *,
    arrays: Any,
    scene_features: np.ndarray,
    device: torch.device,
    batch_size: int,
    collect_latent: bool = True,
    collect_effect: bool = True,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray | None]:
    model.eval()
    latent: list[np.ndarray] = []
    consequence: list[np.ndarray] = []
    effect: list[np.ndarray] = []
    for start in range(0, len(indices), batch_size):
        batch = indices[start : start + batch_size]
        output = model(
            torch.from_numpy(scene_features[arrays.scene_feature_indices[batch]]).to(device),
            torch.from_numpy(arrays.trajectories[batch]).to(device),
        )
        latent_value = output["effect_latent"]
        consequence_value = output["consequence_prediction"]
        effect_value = output["structured_future_prediction"]
        assert isinstance(latent_value, torch.Tensor)
        assert isinstance(consequence_value, torch.Tensor)
        assert isinstance(effect_value, torch.Tensor)
        if collect_latent:
            latent.append(latent_value.float().cpu().numpy())
        consequence.append(consequence_value.float().cpu().numpy())
        if collect_effect:
            effect.append(effect_value.float().cpu().numpy().astype(np.float16))
    return (
        np.concatenate(latent) if collect_latent else None,
        np.concatenate(consequence),
        np.concatenate(effect) if collect_effect else None,
    )


def _bootstrap_mean(
    values: Mapping[str, float],
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> MetricInterval:
    finite = {key: value for key, value in values.items() if math.isfinite(float(value))}
    if not finite:
        return MetricInterval(float("nan"), float("nan"), float("nan"), samples, confidence)
    scenes = np.asarray(sorted(finite), dtype=str)
    vector = np.asarray([finite[scene] for scene in scenes], dtype=np.float64)
    point = float(np.mean(vector))
    rng = np.random.default_rng(seed)
    draws = vector[rng.integers(0, len(vector), size=(samples, len(vector)))].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(draws, [alpha, 1.0 - alpha])
    return MetricInterval(point, float(low), float(high), samples, confidence)


def _channel_scene_metrics(
    *,
    target: np.ndarray,
    prediction: np.ndarray,
    scene_ids: np.ndarray,
    positive_weight: Sequence[float],
    bootstrap_samples: int,
    confidence: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    per_scene: dict[str, dict[str, float]] = {}
    for scene in sorted(set(scene_ids.tolist())):
        indices = np.flatnonzero(scene_ids == scene)
        metrics = channel_metrics(
            target[indices], prediction[indices], positive_weight=positive_weight
        )
        flat: dict[str, float] = {}
        for channel, channel_metrics_value in enumerate(metrics):
            for name, value in channel_metrics_value.items():
                flat[f"channel_{channel}_{name}"] = float(value)
        flat["structured_effect_mae"] = float(
            np.mean(np.abs(prediction[indices] - target[indices]))
        )
        per_scene[str(scene)] = flat
    keys = sorted({key for values in per_scene.values() for key in values})
    summary = []
    for key in keys:
        interval = _bootstrap_mean(
            {scene: values.get(key, float("nan")) for scene, values in per_scene.items()},
            samples=bootstrap_samples,
            confidence=confidence,
            seed=seed,
        )
        if key == "structured_effect_mae":
            channel = -1
            metric = key
        else:
            _, channel_text, metric = key.split("_", 2)
            channel = int(channel_text)
        summary.append(
            {
                "channel_index": channel,
                "channel": "all" if channel < 0 else EFFECT_TUBE_CHANNELS[channel],
                "metric": metric,
                **asdict(interval),
            }
        )
    return summary, per_scene


def _consequence_utility(raw_prediction: np.ndarray) -> np.ndarray:
    prediction = decoded_prediction(raw_prediction, len(HARD_TARGET_FIELDS))
    hard = np.concatenate((prediction[:, :4], 1.0 - prediction[:, 4:6]), axis=1).mean(axis=1)
    soft = prediction[:, len(HARD_TARGET_FIELDS) :]
    positive = np.mean(1.0 / (1.0 + np.exp(-np.clip(soft[:, :5], -20, 20))), axis=1)
    negative = np.mean(1.0 / (1.0 + np.exp(np.clip(soft[:, 5:], -20, 20))), axis=1)
    return 0.7 * hard + 0.15 * positive + 0.15 * negative


def _order_accuracy_by_scene(
    *,
    pair_rows: Sequence[Mapping[str, Any]],
    candidate_lookup: Mapping[str, int],
    utility: np.ndarray,
    selected_scene_ids: Sequence[str],
    order_field: str,
) -> dict[str, float]:
    selected = set(selected_scene_ids)
    correct: dict[str, list[float]] = {}
    for row in pair_rows:
        scene = str(row["scene_id"])
        order = row.get(order_field)
        if scene not in selected or order is None:
            continue
        left = candidate_lookup[str(row["candidate_i"])]
        right = candidate_lookup[str(row["candidate_j"])]
        delta = float(utility[left] - utility[right])
        predicted = 0 if abs(delta) <= 1.0e-6 else (1 if delta > 0 else -1)
        correct.setdefault(scene, []).append(float(predicted == int(order)))
    return {scene: float(np.mean(values)) for scene, values in correct.items()}


def _reversal_interval(
    cases: Sequence[Mapping[str, Any]],
    correct: np.ndarray,
    scenes: Sequence[str],
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> MetricInterval:
    if not len(cases):
        return MetricInterval(float("nan"), float("nan"), float("nan"), samples, confidence)
    point = float(np.mean(correct))
    scene_array = np.asarray(scenes, dtype=str)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(samples):
        selected = rng.choice(scene_array, size=len(scene_array), replace=True)
        counts = {scene: int(np.sum(selected == scene)) for scene in set(selected.tolist())}
        weights = np.asarray(
            [
                counts.get(str(case["positive"]["scene_id"]), 0)
                * counts.get(str(case["negative"]["scene_id"]), 0)
                for case in cases
            ],
            dtype=np.float64,
        )
        if weights.sum() > 0:
            draws.append(float(np.average(correct, weights=weights)))
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(draws, [alpha, 1.0 - alpha])
    return MetricInterval(point, float(low), float(high), samples, confidence)


def _constant_train_mean(target: np.ndarray, indices: np.ndarray) -> np.ndarray:
    accumulator = np.zeros(target.shape[1:], dtype=np.float64)
    count = 0
    for start in range(0, len(indices), 128):
        batch = indices[start : start + 128]
        accumulator += target[batch].astype(np.float64).sum(axis=0)
        count += len(batch)
    return (accumulator / count).astype(np.float32)


def _primary_channel_metric(channel: int) -> tuple[str, bool]:
    if channel in {0, 7}:
        return "balanced_bce", False
    if channel in {1, 2, 3, 6}:
        return "normalized_l1", False
    if channel in {4, 5}:
        return "masked_l1", False
    return "dice", True


def _paired_delta(
    run_scene_metrics: Mapping[tuple[str, int], Mapping[str, Mapping[str, float]]],
    *,
    left_method: str,
    right_method: str,
    seeds: Sequence[int],
    metric: str,
    samples: int,
    confidence: float,
    seed: int,
) -> MetricInterval:
    per_scene: dict[str, list[float]] = {}
    for run_seed in seeds:
        left = run_scene_metrics[(left_method, run_seed)]
        right = run_scene_metrics[(right_method, run_seed)]
        for scene in sorted(set(left) & set(right)):
            left_value = left[scene].get(metric, float("nan"))
            right_value = right[scene].get(metric, float("nan"))
            if math.isfinite(left_value) and math.isfinite(right_value):
                per_scene.setdefault(scene, []).append(left_value - right_value)
    averaged = {
        scene: float(np.mean(values)) for scene, values in per_scene.items() if len(values) == len(seeds)
    }
    return _bootstrap_mean(
        averaged,
        samples=samples,
        confidence=confidence,
        seed=seed,
    )


def _render_phase6_report(
    run_rows: Sequence[Mapping[str, Any]],
    channel_rows: Sequence[Mapping[str, Any]],
    confidence_enabled: bool,
    data_summary: Mapping[str, Any],
    split: Mapping[str, Any],
) -> str:
    methods = sorted(set(str(row["method"]) for row in run_rows))

    def display(value: float, digits: int = 4) -> str:
        return f"{value:.{digits}f}" if math.isfinite(value) else "N/A"

    def method_mean(method: str, name: str) -> float:
        values = [
            float(row.get(name, float("nan")))
            for row in run_rows
            if str(row["method"]) == method
        ]
        finite = [value for value in values if math.isfinite(value)]
        return float(np.mean(finite)) if finite else float("nan")

    lines = [
        "# Phase 6 world-probe ablation",
        "",
        "All trained methods use the same probe, normalized effect-latent dimension, effect decoder, "
        "consequence decoder, steps, 16-scene batch, and three seeds. Factual-only contributes its one "
        "unique expert candidate per scene; multi-candidate methods draw four candidates and average "
        "within scene before the scene batch is averaged. Pair distances are computed only "
        "after L2 normalization. Scenes—not pairs—are the bootstrap unit.",
        "",
        "## Data protocol",
        "",
        f"- Scene-disjoint train/validation/test: {len(split['train'])} / "
        f"{len(split['validation'])} / {len(split['test'])}.",
        f"- Accepted training candidates after validity filtering and held-out-family exclusion: "
        f"{int(data_summary['train_candidate_count'])}.",
        f"- Candidate family held out from training: `{data_summary['held_out_perturbation_family']}`.",
        f"- Pair-bearing training scenes: {int(data_summary['pair_scene_count'])}; each sampled scene "
        "receives equal weight and missing pair categories remain masked.",
        "- Pair thresholds and target normalization use train scenes only; validation is used only "
        "for risk-threshold calibration, and test remains held out until evaluation.",
        "",
        "## Representation and risk results",
        "",
        "| Method | Per-scene alignment | Pooled alignment | AG | EL | AG/EL | Eq/Div AUPRC | Safety AUPRC | False-safe | Shuffle gap | Held-out alignment | Reversal | Rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        selected = [row for row in run_rows if row["method"] == method]
        def mean(name: str) -> float:
            values = [float(row.get(name, float("nan"))) for row in selected]
            values = [value for value in values if math.isfinite(value)]
            return float(np.mean(values)) if values else float("nan")
        lines.append(
            f"| {method} | {display(mean('per_scene_effect_alignment'))} | "
            f"{display(mean('pooled_effect_alignment'))} | {display(mean('action_gap'))} | "
            f"{display(mean('equivalence_leakage'))} | {display(mean('separation_ratio'), 3)} | "
            f"{display(mean('equivalent_divergent_auprc'))} | {display(mean('safety_boundary_auprc'))} | "
            f"{display(mean('false_safe_rate'))} | {display(mean('action_shuffle_gap'), 5)} | "
            f"{display(mean('heldout_family_per_scene_effect_alignment'))} | "
            f"{display(mean('cross_scene_reversal_accuracy'))} | {display(mean('covariance_effective_rank'), 1)} |"
        )
    lines.extend(
        [
            "",
            "## Attribution checks",
            "",
            "- **Action branch engineering failure:** ruled out by the separate Gate-2.5 synthetic fit, "
            "candidate overfit, gradient, Jacobian, variance, and optimizer-membership audit.",
            "- **Factual-only statistical shortcut:** retained as a distinct diagnosis; the factual-only "
            f"test alignment is {method_mean('factual_only', 'per_scene_effect_alignment'):.4f} and its "
            f"structured action-shuffle gap is {method_mean('factual_only', 'action_shuffle_gap'):.5f}.",
            "- **Action-invariant target:** raw-map channels remain diagnostic and never enter the main "
            "collapse/Gate criteria; the nine-channel effect tube is evaluated channel by channel.",
            "- **Multi-candidate data benefit:** absolute minus factual alignment is "
            f"{method_mean('multi_candidate_absolute', 'per_scene_effect_alignment') - method_mean('factual_only', 'per_scene_effect_alignment'):+.4f}; "
            "absolute minus factual false-safe rate is "
            f"{method_mean('multi_candidate_absolute', 'false_safe_rate') - method_mean('factual_only', 'false_safe_rate'):+.4f}.",
            "- **AEE-specific benefit:** AEE minus absolute alignment is "
            f"{method_mean('aee', 'per_scene_effect_alignment') - method_mean('multi_candidate_absolute', 'per_scene_effect_alignment'):+.4f}; "
            "AEE minus global-separation equivalence leakage is "
            f"{method_mean('aee', 'equivalence_leakage') - method_mean('global_separation', 'equivalence_leakage'):+.4f}.",
            "",
            "## Calibrated candidate risk",
            "",
            "Thresholds are selected on validation scenes only and frozen for test. Values below are "
            "three-seed means; run-level scene-bootstrap intervals are in `phase6_artifacts/*.json`.",
            "",
            "| Method | Unsafe prevalence | Balanced accuracy | AUROC | AUPRC | False-safe rate |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method in methods:
        selected = [row for row in run_rows if row["method"] == method]
        def risk_mean(name: str) -> float:
            values = [float(row.get(name, float("nan"))) for row in selected]
            finite = [value for value in values if math.isfinite(value)]
            return float(np.mean(finite)) if finite else float("nan")
        lines.append(
            f"| {method} | {display(risk_mean('unsafe_prevalence'))} | "
            f"{display(risk_mean('balanced_accuracy'))} | {display(risk_mean('auroc'))} | "
            f"{display(risk_mean('auprc'))} | {display(risk_mean('false_safe_rate'))} |"
        )
    lines.extend(
        [
            "",
            "## Structured effect channels",
            "",
            "Per-channel rows, control comparisons, and scene-bootstrap intervals are saved in "
            "`phase6_artifacts/channel_metrics.csv`. Binary fields use balanced BCE/AUPRC/IoU; "
            "SDF and clearance use Huber/normalized L1; occupied velocity uses masked Huber/L1; "
            "footprint uses Dice/IoU.",
            "",
            f"Confidence-weighted AEE required in the matrix: **{confidence_enabled}**. It is retained in config "
            "but disabled when the LR/IDM disagreement subset is insufficient.",
            "",
            "No Qwen/DiT parameter was updated and PDMS/EPDMS remain N/A.",
            "",
        ]
    )
    lines.extend(
        [
            "## Structured-channel controls",
            "",
            "The table reports the declared primary metric for each channel. Lower is better except "
            "for footprint Dice. Formal paired scene-bootstrap control and shuffle checks are reported "
            "in `gate3_decision.md`.",
            "",
            "| Channel | Metric | AEE | Scene-only | Mean | Zero |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )

    def channel_mean(method: str, channel: int, metric: str) -> float:
        values = [
            float(row.get("point", float("nan")))
            for row in channel_rows
            if str(row.get("method")) == method
            and int(row.get("channel_index", -2)) == channel
            and str(row.get("metric")) == metric
        ]
        finite = [value for value in values if math.isfinite(value)]
        return float(np.mean(finite)) if finite else float("nan")

    for channel, name in enumerate(EFFECT_TUBE_CHANNELS):
        metric, _ = _primary_channel_metric(channel)
        lines.append(
            f"| {name} | {metric} | {display(channel_mean('aee', channel, metric), 5)} | "
            f"{display(channel_mean('scene_only_control', channel, metric), 5)} | "
            f"{display(channel_mean('mean_control', channel, metric), 5)} | "
            f"{display(channel_mean('zero_control', channel, metric), 5)} |"
        )
    lines.extend(
        [
            "",
            "## Latent anti-collapse audit",
            "",
            "| Method | Raw norm | Normalized variance | Covariance rank |",
            "|---|---:|---:|---:|",
        ]
    )
    for method in methods:
        lines.append(
            f"| {method} | {display(method_mean(method, 'latent_norm_mean'), 5)} | "
            f"{display(method_mean(method, 'normalized_latent_variance'), 7)} | "
            f"{display(method_mean(method, 'covariance_effective_rank'), 1)} |"
        )
    lines.append("")
    idm_values = [
        float(row.get("idm_order_accuracy", float("nan"))) for row in run_rows
    ]
    if not any(math.isfinite(value) for value in idm_values):
        lines.extend(
            [
                "## LR-to-IDM transfer",
                "",
                "N/A in the scene-disjoint test split: all 128 scenes with reactive-model labels "
                "belong to the training partition used for the agreement/confidence audit. No reactive "
                "test label is copied across scenes, and this report does not invent a transfer score.",
                "",
            ]
        )
    return "\n".join(lines)


def _write_unified_results(run_rows: Sequence[Mapping[str, Any]]) -> None:
    """Refresh the required table without inventing unavailable planning metrics."""

    columns = (
        "Method",
        "Factual Prediction Error",
        "Action Shuffle Gap",
        "Action Gap",
        "Equivalence Leakage",
        "Effect Alignment",
        "False-Safe Rate",
        "PDMS",
        "EPDMS",
        "NC",
        "DAC",
        "TTC",
        "EP",
        "Inference Latency",
        "Trainable Parameters",
    )

    def mean(method: str, metric: str) -> str:
        values = [
            float(row.get(metric, float("nan")))
            for row in run_rows
            if str(row["method"]) == method
        ]
        finite = [value for value in values if math.isfinite(value)]
        return f"{np.mean(finite):.6f}" if finite else "N/A"

    labels = {
        "factual_only": "Factual only",
        "multi_candidate_absolute": "Multi-candidate absolute",
        "global_separation": "Global separation",
        "aee": "AEE",
        "confidence_aee": "Confidence-weighted AEE",
        "scene_only_control": "Scene-only equal-capacity control",
        "mean_control": "Effect-target mean control",
        "zero_control": "Effect-target zero control",
    }
    rows: list[dict[str, str]] = [
        {
            "Method": "Original Qwen+DiT (frozen feature source)",
            **{name: "N/A" for name in columns if name != "Method"},
            "Trainable Parameters": "0",
        }
    ]
    observed = {str(row["method"]) for row in run_rows}
    for method, label in labels.items():
        if method not in observed:
            if method == "confidence_aee":
                rows.append(
                    {
                        "Method": label,
                        **{name: "Not run: insufficient LR/IDM disagreement" for name in columns if name != "Method"},
                    }
                )
            continue
        selected = [row for row in run_rows if str(row["method"]) == method]
        parameter_values = [
            int(row["trainable_parameters"])
            for row in selected
            if row.get("trainable_parameters") is not None
        ]
        rows.append(
            {
                "Method": label,
                "Factual Prediction Error": mean(method, "factual_prediction_error"),
                "Action Shuffle Gap": mean(method, "action_shuffle_gap"),
                "Action Gap": mean(method, "action_gap"),
                "Equivalence Leakage": mean(method, "equivalence_leakage"),
                "Effect Alignment": mean(method, "per_scene_effect_alignment"),
                "False-Safe Rate": mean(method, "false_safe_rate"),
                "PDMS": "N/A",
                "EPDMS": "N/A",
                "NC": "N/A",
                "DAC": "N/A",
                "TTC": "N/A",
                "EP": "N/A",
                "Inference Latency": "N/A (training-only probe)",
                "Trainable Parameters": str(parameter_values[0]) if parameter_values else "0",
            }
        )
    report_root = REPOSITORY_ROOT / "reports/action_effect_world_model"
    csv_path = report_root / "results_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Unified Gate-3 probe result table",
        "",
        "World probes are offline/training-only and were not attached to Qwen+DiT. Accordingly, "
        "planning latency and all PDMS/EPDMS component columns remain N/A rather than copying "
        "numbers from an unmatched baseline run.",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" if index == 0 else "---:" for index in range(len(columns))) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(column, "N/A") for column in columns) + " |")
    lines.append("")
    (report_root / "results_table.md").write_text("\n".join(lines), encoding="utf-8")


def _plot_phase6_summary(run_rows: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    """Plot the core matched-method metrics without selecting favorable runs."""

    import matplotlib.pyplot as plt

    methods = [
        method
        for method in (
            "factual_only",
            "multi_candidate_absolute",
            "global_separation",
            "aee",
            "confidence_aee",
        )
        if any(str(row["method"]) == method for row in run_rows)
    ]
    metrics = (
        ("per_scene_effect_alignment", "Per-scene alignment"),
        ("action_gap", "Action Gap"),
        ("equivalence_leakage", "Equivalence Leakage"),
        ("separation_ratio", "AG / EL"),
        ("safety_boundary_auprc", "Safety-boundary AUPRC"),
        ("false_safe_rate", "False-safe rate"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for axis, (metric, title) in zip(axes.flat, metrics):
        means, errors = [], []
        for method in methods:
            values = np.asarray(
                [float(row[metric]) for row in run_rows if str(row["method"]) == method],
                dtype=np.float64,
            )
            means.append(float(np.mean(values)))
            errors.append(float(np.std(values)))
        axis.bar(np.arange(len(methods)), means, yerr=errors, capsize=3)
        axis.set_xticks(np.arange(len(methods)), [value.replace("_", "\n") for value in methods])
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(output_dir / "phase6_metric_comparison.png", dpi=180)
    plt.close(figure)


def _select_failure_examples(
    pair_rows: Sequence[Mapping[str, Any]],
    distances: Mapping[str, Mapping[tuple[str, str], float]],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Select deterministic favorable and unfavorable cases from one fixed seed."""

    by_key = {
        (str(row["candidate_i"]), str(row["candidate_j"])): row for row in pair_rows
    }
    required = ("factual_only", "multi_candidate_absolute", "global_separation", "aee")
    common = set(by_key)
    for method in required:
        common &= set(distances.get(method, {}))
    examples: list[dict[str, Any]] = []

    def add(category: str, ranked: Sequence[tuple[float, tuple[str, str]]]) -> None:
        for score, key in ranked[:limit]:
            row = by_key[key]
            examples.append(
                {
                    "category": category,
                    "scene_id": str(row["scene_id"]),
                    "candidate_i": key[0],
                    "candidate_j": key[1],
                    "pair_type": str(row["pair_type"]),
                    "safety_boundary": bool(row.get("safety_boundary")),
                    "geometric_distance": float(row.get("geometric_distance", float("nan"))),
                    "consequence_distance": float(row.get("consequence_distance", float("nan"))),
                    "selection_score": float(score),
                    **{
                        f"{method}_latent_distance": float(distances[method][key])
                        for method in required
                    },
                }
            )

    factual = sorted(
        [
            (
                float(by_key[key]["consequence_distance"])
                / max(distances["factual_only"][key], 1.0e-6),
                key,
            )
            for key in common
            if str(by_key[key]["pair_type"]) == "effect_divergent"
        ],
        reverse=True,
    )
    add("factual_accurate_but_action_insensitive", factual)
    over_separation = sorted(
        [
            (distances["global_separation"][key] - distances["aee"][key], key)
            for key in common
            if str(by_key[key]["pair_type"]) == "effect_equivalent"
        ],
        reverse=True,
    )
    add("global_separation_over_separates_equivalent", over_separation)
    safety = sorted(
        [
            (distances["aee"][key] - distances["multi_candidate_absolute"][key], key)
            for key in common
            if bool(by_key[key].get("safety_boundary"))
        ],
        reverse=True,
    )
    add("aee_safety_boundary_diagnostic", safety)
    conflicts = sorted(
        [
            row
            for row in pair_rows
            if row.get("reactive_order") is not None
            and (
                not bool(row.get("pairwise_hard_relation_agreement", True))
                or not bool(row.get("soft_rank_agreement", True))
            )
        ],
        key=lambda row: (
            float(row.get("geometric_distance", math.inf)),
            str(row["scene_id"]),
        ),
    )
    for row in conflicts[:limit]:
        examples.append(
            {
                "category": "log_replay_reactive_model_conflict",
                "scene_id": str(row["scene_id"]),
                "candidate_i": str(row["candidate_i"]),
                "candidate_j": str(row["candidate_j"]),
                "pair_type": str(row["pair_type"]),
                "safety_boundary": bool(row.get("safety_boundary")),
                "geometric_distance": float(row.get("geometric_distance", float("nan"))),
                "consequence_distance": float(row.get("consequence_distance", float("nan"))),
                "log_replay_order": int(row["log_replay_order"]),
                "reactive_order": int(row["reactive_order"]),
                "hard_relation_agreement": bool(
                    row.get("pairwise_hard_relation_agreement", False)
                ),
            }
        )
    return examples


def _render_failure_analysis(examples: Sequence[Mapping[str, Any]]) -> str:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in examples:
        grouped.setdefault(str(row["category"]), []).append(row)
    titles = {
        "factual_accurate_but_action_insensitive": "Factual prediction with weak action sensitivity",
        "global_separation_over_separates_equivalent": "Global separation over-separates equivalent actions",
        "aee_safety_boundary_diagnostic": "AEE safety-boundary diagnostics",
        "log_replay_reactive_model_conflict": "Log-replay / reactive-model conflicts",
    }
    lines = [
        "# Gate-3 failure and case analysis",
        "",
        "Cases are selected deterministically from the full fixed test/IDM subsets, including "
        "unfavorable deltas; they are diagnostics rather than causal counterfactuals.",
        "",
    ]
    for category, title in titles.items():
        lines.extend(
            [
                f"## {title}",
                "",
                "| Scene | Candidate pair | Pair type | Safety boundary | Geometry | Consequence | Selection delta |",
                "|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in grouped.get(category, []):
            lines.append(
                f"| `{row['scene_id']}` | `{row['candidate_i']}` / `{row['candidate_j']}` | "
                f"{row['pair_type']} | {row['safety_boundary']} | "
                f"{float(row['geometric_distance']):.4f} | "
                f"{float(row['consequence_distance']):.4f} | "
                f"{float(row.get('selection_score', float('nan'))):.4f} |"
            )
        if not grouped.get(category):
            lines.append("| N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
        lines.append("")
    lines.extend(
        [
            "## World metric improved but planning did not",
            "",
            "Not evaluated by instruction: no world loss was connected to Qwen+DiT, no planning "
            "training was run, and no PDMS/EPDMS value is populated. Therefore this delivery makes "
            "no world-to-planning transfer or gradient-conflict claim.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPOSITORY_ROOT / "configs/action_effect/phase6.yaml"
    )
    parser.add_argument("--candidate-cache", type=Path)
    parser.add_argument("--consequence-cache", type=Path)
    parser.add_argument("--pair-cache", type=Path)
    parser.add_argument("--scene-feature-cache", type=Path)
    parser.add_argument("--effect-tube-cache", type=Path)
    parser.add_argument("--split-cache", type=Path)
    parser.add_argument("--probe-dir", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "reports/action_effect_world_model/phase6_artifacts",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=REPOSITORY_ROOT / "reports/action_effect_world_model/phase6_probe_ablation.md",
    )
    parser.add_argument(
        "--gate-report-path",
        type=Path,
        default=REPOSITORY_ROOT / "reports/action_effect_world_model/gate3_decision.md",
    )
    parser.add_argument("--methods", nargs="+", choices=TRAINED_METHODS)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--bootstrap-samples", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_yaml(args.config.resolve())
    data = config["data"]
    paths = {
        "candidate": _cache(args.candidate_cache, str(data["candidate_cache"])),
        "consequence": _cache(args.consequence_cache, str(data["consequence_cache"])),
        "pair": _cache(args.pair_cache, str(data["pair_cache"])),
        "scene_feature": _cache(args.scene_feature_cache, str(data["scene_feature_cache"])),
        "effect_tube": _cache(args.effect_tube_cache, str(data["effect_tube_cache"])),
        "split": _cache(args.split_cache, str(data["split_cache"])),
    }
    for name, path in paths.items():
        if read_manifest(path) is None:
            raise FileNotFoundError(f"published {name} cache is missing: {path}")
    probe_dir = (
        args.probe_dir.resolve()
        if args.probe_dir is not None
        else _root("ACTION_EFFECT_OUTPUT_ROOT") / "phase6/pilot_small_v1"
    )
    with (paths["split"] / "split.json").open("r", encoding="utf-8") as stream:
        split = json.load(stream)
    train_scenes = [str(value) for value in split["train"]]
    validation_scenes = [str(value) for value in split["validation"]]
    test_scenes = [str(value) for value in split["test"]]
    arrays, scene_features, _, scales, _, _ = load_probe_arrays(
        candidate_cache=paths["candidate"],
        consequence_cache=paths["consequence"],
        scene_feature_cache=paths["scene_feature"],
        fit_scene_ids=train_scenes,
        assumption=str(data["target_assumption"]),
    )
    effect_target, effect_valid = load_structured_targets(paths["effect_tube"], arrays)
    metadata = list(iter_jsonl(paths["candidate"] / "metadata.jsonl"))
    perturbation = np.asarray([str(row["perturbation_type"]) for row in metadata], dtype=str)
    heldout_family = str(split["held_out_perturbation_family"])
    train_mask = (
        arrays.accepted
        & effect_valid
        & np.isin(arrays.scene_ids, train_scenes)
        & (perturbation != heldout_family)
    )
    validation_mask = arrays.accepted & effect_valid & np.isin(arrays.scene_ids, validation_scenes)
    test_mask = arrays.accepted & effect_valid & np.isin(arrays.scene_ids, test_scenes)
    validation_indices = np.flatnonzero(validation_mask)
    test_indices = np.flatnonzero(test_mask)
    test_effect_target = effect_target[test_indices].astype(np.float32)
    pair_rows = list(iter_jsonl(paths["pair"] / "pairs.jsonl"))
    consequence_rows = list(iter_jsonl(paths["consequence"] / "consequences.jsonl"))
    candidate_lookup = {candidate: index for index, candidate in enumerate(arrays.candidate_ids)}
    reversal_cases = build_reversal_cases(
        consequence_rows,
        pair_rows,
        selected_scene_ids=test_scenes,
        maximum_cases=5000,
        seed=int(config["experiment"]["seeds"][0]),
    )
    methods = args.methods or [str(value) for value in config["methods"]]
    seeds = args.seeds or [int(value) for value in config["experiment"]["seeds"]]
    bootstrap_samples = int(
        args.bootstrap_samples
        if args.bootstrap_samples is not None
        else config["experiment"]["bootstrap_samples"]
    )
    confidence = float(config["experiment"]["bootstrap_confidence"])
    representation_bootstrap_samples = min(
        bootstrap_samples,
        int(config["experiment"].get("representation_bootstrap_samples", bootstrap_samples)),
    )
    data_summary = json.loads((probe_dir / "data_summary.json").read_text(encoding="utf-8"))
    positive_weight = data_summary["binary_positive_weight"]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict[str, Any]] = []
    all_channel_rows: list[dict[str, Any]] = []
    run_scene_metrics: dict[tuple[str, int], dict[str, dict[str, float]]] = {}
    example_pair_distances: dict[str, dict[tuple[str, str], float]] = {}
    labels = unsafe_labels(arrays, low_ttc_seconds=float(config["risk"]["low_ttc_seconds"]))
    requested = str(config["training"]["device"])
    device = torch.device(requested if torch.cuda.is_available() else "cpu")
    for method in methods:
        for run_seed in seeds:
            checkpoint_path = probe_dir / method / f"seed_{run_seed}/probe.pt"
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Phase-6 checkpoint is missing: {checkpoint_path}")
            model = _model(config, method).to(device)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            validation_latent, validation_consequence, validation_effect_raw = _predict(
                model,
                validation_indices,
                arrays=arrays,
                scene_features=scene_features,
                device=device,
                batch_size=int(config["training"]["evaluation_batch_size"]),
                collect_latent=False,
                collect_effect=False,
            )
            test_latent, test_consequence, test_effect_raw = _predict(
                model,
                test_indices,
                arrays=arrays,
                scene_features=scene_features,
                device=device,
                batch_size=int(config["training"]["evaluation_batch_size"]),
            )
            assert test_latent is not None and test_effect_raw is not None
            latent_full = np.zeros((len(arrays.scene_ids), test_latent.shape[1]), dtype=np.float32)
            latent_full[test_indices] = test_latent
            representation, representation_details = representation_metrics(
                latent_by_candidate=latent_full,
                candidate_ids=arrays.candidate_ids,
                pair_rows=pair_rows,
                selected_scene_ids=test_scenes,
                candidate_valid=test_mask,
                perturbation=perturbation,
                heldout_family=heldout_family,
                bootstrap_samples=representation_bootstrap_samples,
                confidence=confidence,
                seed=run_seed,
            )
            if run_seed == seeds[0] and method in {
                "factual_only",
                "multi_candidate_absolute",
                "global_separation",
                "aee",
            }:
                example_pair_distances[method] = {
                    (str(left), str(right)): float(distance)
                    for left, right, distance in zip(
                        representation_details["candidate_i"],
                        representation_details["candidate_j"],
                        representation_details["pair_distance"],
                    )
                }
            validation_scores = consequence_risk_score(
                validation_consequence,
                scales,
                low_ttc_seconds=float(config["risk"]["low_ttc_seconds"]),
            )
            threshold, _ = calibrate_risk_threshold(
                labels[validation_indices], validation_scores
            )
            test_scores = consequence_risk_score(
                test_consequence,
                scales,
                low_ttc_seconds=float(config["risk"]["low_ttc_seconds"]),
            )
            risk_intervals, risk_counts = calibrated_scene_bootstrap(
                labels=labels[test_indices],
                scores=test_scores,
                scene_ids=arrays.scene_ids[test_indices],
                selected_scene_ids=test_scenes,
                threshold=threshold,
                samples=bootstrap_samples,
                confidence=confidence,
                seed=run_seed,
            )
            test_effect = decoded_effect_prediction(test_effect_raw)
            compact_candidate_ids = arrays.candidate_ids[test_indices]
            compact_scene_ids = arrays.scene_ids[test_indices]
            channel_summary, channel_scene = _channel_scene_metrics(
                target=test_effect_target,
                prediction=test_effect,
                scene_ids=arrays.scene_ids[test_indices],
                positive_weight=positive_weight,
                bootstrap_samples=bootstrap_samples,
                confidence=confidence,
                seed=run_seed,
            )
            shuffle_gap, shuffle_by_scene = effect_action_shuffle_gap(
                target=test_effect_target,
                prediction=test_effect,
                scene_ids=arrays.scene_ids[test_indices],
            )
            channel_shuffle = effect_primary_channel_shuffle_gap(
                target=test_effect_target,
                prediction=test_effect,
                scene_ids=arrays.scene_ids[test_indices],
                positive_weight=positive_weight,
            )
            channel_sensitivity = effect_channel_pair_sensitivity(
                target=test_effect_target,
                prediction=test_effect,
                scene_ids=compact_scene_ids,
                candidate_ids=compact_candidate_ids,
                pair_rows=pair_rows,
                selected_scene_ids=test_scenes,
            )
            for key in sorted(
                {
                    key
                    for values in channel_sensitivity.values()
                    for key in values
                }
            ):
                interval = _bootstrap_mean(
                    {
                        scene: values.get(key, float("nan"))
                        for scene, values in channel_sensitivity.items()
                    },
                    samples=bootstrap_samples,
                    confidence=confidence,
                    seed=run_seed,
                )
                _, channel_text, metric = key.split("_", 2)
                channel = int(channel_text)
                channel_summary.append(
                    {
                        "channel_index": channel,
                        "channel": EFFECT_TUBE_CHANNELS[channel],
                        "metric": metric,
                        **asdict(interval),
                    }
                )
            shuffle_interval = _bootstrap_mean(
                shuffle_by_scene,
                samples=bootstrap_samples,
                confidence=confidence,
                seed=run_seed,
            )
            utility_full = np.zeros(len(arrays.scene_ids), dtype=np.float64)
            utility_full[test_indices] = _consequence_utility(test_consequence)
            reversal_point, reversal_correct = reversal_accuracy(
                reversal_cases,
                candidate_ids=arrays.candidate_ids,
                predicted_utility=utility_full,
            )
            reversal_interval = _reversal_interval(
                reversal_cases,
                reversal_correct,
                test_scenes,
                samples=bootstrap_samples,
                confidence=confidence,
                seed=run_seed,
            )
            lr_accuracy = _order_accuracy_by_scene(
                pair_rows=pair_rows,
                candidate_lookup=candidate_lookup,
                utility=utility_full,
                selected_scene_ids=test_scenes,
                order_field="log_replay_order",
            )
            idm_accuracy = _order_accuracy_by_scene(
                pair_rows=pair_rows,
                candidate_lookup=candidate_lookup,
                utility=utility_full,
                selected_scene_ids=test_scenes,
                order_field="reactive_order",
            )
            lr_interval = _bootstrap_mean(
                lr_accuracy,
                samples=bootstrap_samples,
                confidence=confidence,
                seed=run_seed,
            )
            idm_interval = _bootstrap_mean(
                idm_accuracy,
                samples=bootstrap_samples,
                confidence=confidence,
                seed=run_seed,
            )
            diagnostics = latent_diagnostics(
                test_latent,
                relative_tolerance=float(config["latent_audit"]["covariance_rank_relative_tolerance"]),
            )
            predicted_unsafe = test_scores >= threshold
            decoded_consequence = decoded_prediction(test_consequence, len(HARD_TARGET_FIELDS))
            factual_error_by_scene: dict[str, float] = {}
            for scene in test_scenes:
                local = np.flatnonzero(
                    (compact_scene_ids == scene) & arrays.anchor[test_indices]
                )
                if len(local):
                    factual_error_by_scene[scene] = float(
                        np.mean(
                            np.abs(
                                decoded_consequence[local]
                                - arrays.targets[test_indices[local]]
                            )
                        )
                    )
            factual_error = _bootstrap_mean(
                factual_error_by_scene,
                samples=bootstrap_samples,
                confidence=confidence,
                seed=run_seed,
            )
            per_scene = {
                scene: dict(values)
                for scene, values in representation_details["scene_metrics"].items()
            }
            for scene in test_scenes:
                per_scene.setdefault(scene, {})
                local = np.flatnonzero(arrays.scene_ids[test_indices] == scene)
                unsafe = labels[test_indices][local]
                per_scene[scene]["false_safe_rate"] = (
                    float(np.sum(unsafe & ~predicted_unsafe[local]) / np.sum(unsafe))
                    if np.any(unsafe)
                    else float("nan")
                )
                per_scene[scene].update(channel_scene[scene])
                per_scene[scene].update(channel_sensitivity.get(scene, {}))
                per_scene[scene]["action_shuffle_gap"] = shuffle_by_scene.get(scene, float("nan"))
                if scene in channel_shuffle:
                    for channel, value in enumerate(channel_shuffle[scene]):
                        per_scene[scene][f"channel_{channel}_shuffle_gap"] = float(value)
                per_scene[scene]["lr_order_accuracy"] = lr_accuracy.get(scene, float("nan"))
                per_scene[scene]["idm_order_accuracy"] = idm_accuracy.get(scene, float("nan"))
            run_scene_metrics[(method, run_seed)] = per_scene
            row = {
                "method": method,
                "seed": run_seed,
                **{name: value.point for name, value in representation.items()},
                **{name: value.point for name, value in risk_intervals.items()},
                "calibrated_threshold": threshold,
                "action_shuffle_gap": shuffle_gap,
                "factual_prediction_error": factual_error.point,
                "cross_scene_reversal_accuracy": reversal_point,
                "lr_order_accuracy": lr_interval.point,
                "idm_order_accuracy": idm_interval.point,
                "lr_to_idm_transfer_gap": idm_interval.point - lr_interval.point,
                "trainable_parameters": int(
                    sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
                ),
                **diagnostics,
                **risk_counts,
            }
            run_rows.append(row)
            write_json(
                output_dir / f"{method}_seed_{run_seed}.json",
                {
                    "metrics": row,
                    "representation_intervals": intervals_as_json(representation),
                    "risk_intervals": intervals_to_json(risk_intervals),
                    "action_shuffle_interval": asdict(shuffle_interval),
                    "factual_prediction_error_interval": asdict(factual_error),
                    "reversal_interval": asdict(reversal_interval),
                    "lr_order_interval": asdict(lr_interval),
                    "idm_order_interval": asdict(idm_interval),
                    "channel_metrics": channel_summary,
                    "reversal_case_count": len(reversal_cases),
                },
            )
            for channel_row in channel_summary:
                all_channel_rows.append({"method": method, "seed": run_seed, **channel_row})
            write_jsonl(
                output_dir / f"{method}_seed_{run_seed}_per_scene.jsonl",
                [{"scene_id": scene, **values} for scene, values in sorted(per_scene.items())],
            )
            del model, validation_latent, validation_effect_raw, test_effect_raw, test_effect
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Target-only controls use train statistics and receive the same scene-level
    # structured metrics. They have no representation/risk values by design.
    mean_target = _constant_train_mean(effect_target, np.flatnonzero(train_mask))
    control_predictions = {
        "mean_control": np.broadcast_to(
            mean_target, (len(test_indices), *mean_target.shape)
        ),
        "zero_control": np.zeros(
            (len(test_indices), *effect_target.shape[1:]), dtype=np.float32
        ),
    }
    for control, prediction in control_predictions.items():
        summary, per_scene_channel = _channel_scene_metrics(
            target=test_effect_target,
            prediction=prediction,
            scene_ids=arrays.scene_ids[test_indices],
            positive_weight=positive_weight,
            bootstrap_samples=bootstrap_samples,
            confidence=confidence,
            seed=seeds[0],
        )
        shuffle_gap, shuffle_scene = effect_action_shuffle_gap(
            target=test_effect_target,
            prediction=prediction,
            scene_ids=arrays.scene_ids[test_indices],
        )
        per_scene = {scene: dict(values) for scene, values in per_scene_channel.items()}
        for scene in per_scene:
            per_scene[scene]["action_shuffle_gap"] = shuffle_scene.get(scene, float("nan"))
        run_scene_metrics[(control, seeds[0])] = per_scene
        run_rows.append({
            "method": control,
            "seed": seeds[0],
            "action_shuffle_gap": shuffle_gap,
            "covariance_effective_rank": 0,
        })
        for channel_row in summary:
            all_channel_rows.append({"method": control, "seed": seeds[0], **channel_row})

    write_jsonl(output_dir / "metrics_by_run.jsonl", run_rows)
    write_jsonl(output_dir / "channel_metrics.jsonl", all_channel_rows)
    with (output_dir / "metrics_by_run.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = sorted({key for row in run_rows for key in row})
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(run_rows)
    with (output_dir / "channel_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = sorted({key for row in all_channel_rows for key in row})
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_channel_rows)
    _write_unified_results(run_rows)
    _plot_phase6_summary(run_rows, output_dir)

    # Gate-3 paired scene-bootstrap comparisons. Positive deltas mean the left
    # method has a numerically larger metric; lower-is-better metrics are handled
    # explicitly below.
    delta = {}
    for name, left, right, metric in (
        ("aee_vs_absolute_alignment", "aee", "multi_candidate_absolute", "per_scene_effect_alignment"),
        ("aee_vs_absolute_pair_auprc", "aee", "multi_candidate_absolute", "equivalent_divergent_auprc"),
        ("aee_vs_absolute_safety_auprc", "aee", "multi_candidate_absolute", "safety_boundary_auprc"),
        ("aee_vs_absolute_false_safe", "aee", "multi_candidate_absolute", "false_safe_rate"),
        ("aee_vs_absolute_heldout_alignment", "aee", "multi_candidate_absolute", "heldout_family_per_scene_effect_alignment"),
        ("aee_vs_global_separation_ratio", "aee", "global_separation", "separation_ratio"),
        ("aee_vs_global_structured_error", "aee", "global_separation", "structured_effect_mae"),
    ):
        delta[name] = _paired_delta(
            run_scene_metrics,
            left_method=left,
            right_method=right,
            seeds=seeds,
            metric=metric,
            samples=bootstrap_samples,
            confidence=confidence,
            seed=seeds[0],
        )
    confidence_delta: dict[str, MetricInterval] = {}
    if "confidence_aee" in methods:
        for name, right, metric in (
            ("confidence_vs_aee_alignment", "aee", "per_scene_effect_alignment"),
            ("confidence_vs_aee_pair_auprc", "aee", "equivalent_divergent_auprc"),
            ("confidence_vs_aee_safety_auprc", "aee", "safety_boundary_auprc"),
            ("confidence_vs_aee_false_safe", "aee", "false_safe_rate"),
            ("confidence_vs_aee_heldout_alignment", "aee", "heldout_family_per_scene_effect_alignment"),
            ("confidence_vs_absolute_alignment", "multi_candidate_absolute", "per_scene_effect_alignment"),
            ("confidence_vs_absolute_safety_auprc", "multi_candidate_absolute", "safety_boundary_auprc"),
            ("confidence_vs_absolute_false_safe", "multi_candidate_absolute", "false_safe_rate"),
            ("confidence_vs_absolute_heldout_alignment", "multi_candidate_absolute", "heldout_family_per_scene_effect_alignment"),
        ):
            confidence_delta[name] = _paired_delta(
                run_scene_metrics,
                left_method="confidence_aee",
                right_method=right,
                seeds=seeds,
                metric=metric,
                samples=bootstrap_samples,
                confidence=confidence,
                seed=seeds[0],
            )
    method_means = {}
    for method in ("multi_candidate_absolute", "global_separation", "aee"):
        selected = [row for row in run_rows if row["method"] == method]
        method_means[method] = {
            key: float(np.mean([float(row[key]) for row in selected]))
            for key in ("action_gap", "equivalence_leakage", "separation_ratio")
        }
    structured_channels = []
    for channel in range(len(EFFECT_TUBE_CHANNELS)):
        metric, higher_better = _primary_channel_metric(channel)
        key = f"channel_{channel}_{metric}"
        comparisons = {}
        for control in ("scene_only_control", "mean_control", "zero_control"):
            if control in CONTROL_METHODS:
                # Replicate a deterministic target-only control across seeds for
                # the paired scene calculation without pretending it was trained.
                control_values = run_scene_metrics[(control, seeds[0])]
                for run_seed in seeds:
                    run_scene_metrics[(control, run_seed)] = control_values
            comparisons[control] = _paired_delta(
                run_scene_metrics,
                left_method="aee",
                right_method=control,
                seeds=seeds,
                metric=key,
                samples=bootstrap_samples,
                confidence=confidence,
                seed=seeds[0],
            )
        shuffle = _bootstrap_mean(
            {
                scene: float(
                    np.mean(
                        [
                            run_scene_metrics[("aee", run_seed)][scene].get(
                                f"channel_{channel}_shuffle_gap", float("nan")
                            )
                            for run_seed in seeds
                        ]
                    )
                )
                for scene in test_scenes
            },
            samples=bootstrap_samples,
            confidence=confidence,
            seed=seeds[0],
        )
        beats_controls = all(
            (value.ci_low > 0 if higher_better else value.ci_high < 0)
            for value in comparisons.values()
        )
        structured_channels.append(
            {
                "channel": EFFECT_TUBE_CHANNELS[channel],
                "primary_metric": metric,
                "higher_better": higher_better,
                "beats_all_controls": beats_controls,
                "shuffle_gap": asdict(shuffle),
                "shuffle_significantly_hurts": shuffle.ci_low > 0,
                "comparisons": {
                    name: asdict(value) for name, value in comparisons.items()
                },
            }
        )
    structured_success = any(
        row["beats_all_controls"] and row["shuffle_significantly_hurts"]
        for row in structured_channels
    )
    alignment_improves_each_seed = all(
        next(
            float(row["per_scene_effect_alignment"])
            for row in run_rows
            if row["method"] == "aee" and int(row["seed"]) == run_seed
        )
        > next(
            float(row["per_scene_effect_alignment"])
            for row in run_rows
            if row["method"] == "multi_candidate_absolute" and int(row["seed"]) == run_seed
        )
        for run_seed in seeds
    )
    condition_result = gate3_conditions(
        paired_deltas=delta,
        aee_action_gap=method_means["aee"]["action_gap"],
        global_action_gap=method_means["global_separation"]["action_gap"],
        aee_equivalence_leakage=method_means["aee"]["equivalence_leakage"],
        global_equivalence_leakage=method_means["global_separation"]["equivalence_leakage"],
        alignment_improves_each_seed=alignment_improves_each_seed,
        structured_effect_success=structured_success,
    )
    aee_vs_absolute = condition_result["aee_vs_absolute"]
    aee_vs_global = condition_result["aee_vs_global"]
    gate_pass = condition_result["decision"] == "PASS"
    gate = {
        "decision": "PASS" if gate_pass else "FAIL",
        "aee_vs_absolute": aee_vs_absolute,
        "aee_vs_global": aee_vs_global,
        "structured_effect_success": structured_success,
        "structured_channel_checks": structured_channels,
        "paired_deltas": {name: asdict(value) for name, value in delta.items()},
        "confidence_paired_deltas": {
            name: asdict(value) for name, value in confidence_delta.items()
        },
        "confidence_aee_primary": bool(config["confidence_aee"]["enabled"]),
        "qwen_dit_world_loss": "NOT IMPLEMENTED",
        "planning_training": "NOT RUN",
        "pdms": None,
        "epdms": None,
    }
    write_json(
        output_dir / "evaluation_identity.json",
        {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
            ).strip(),
            "config": str(args.config.resolve()),
            "methods": methods,
            "seeds": seeds,
            "bootstrap_samples": bootstrap_samples,
            "evaluator_tree_hash": content_hash(
                {
                    str(path.relative_to(REPOSITORY_ROOT)): file_sha256(path)
                    for path in (
                        Path(__file__),
                        REPOSITORY_ROOT / "research/action_effect/phase6_metrics.py",
                        REPOSITORY_ROOT / "research/action_effect/gate2_5.py",
                    )
                }
            ),
            "source_cache_manifests": {
                name: read_manifest(path).compatibility_identity()  # type: ignore[union-attr]
                for name, path in paths.items()
            },
        },
    )
    failure_examples = _select_failure_examples(pair_rows, example_pair_distances)
    write_jsonl(output_dir / "failure_examples.jsonl", failure_examples)
    (REPOSITORY_ROOT / "reports/action_effect_world_model/failure_analysis.md").write_text(
        _render_failure_analysis(failure_examples), encoding="utf-8"
    )
    args.report_path.resolve().write_text(
        _render_phase6_report(
            run_rows,
            all_channel_rows,
            bool(config["confidence_aee"]["enabled"]),
            data_summary,
            split,
        ),
        encoding="utf-8",
    )
    failed = []
    for group_name, group in (("AEE vs absolute", aee_vs_absolute), ("AEE vs global", aee_vs_global)):
        for name, value in group.items():
            if isinstance(value, bool) and not value:
                failed.append(f"{group_name}: {name}")
    if not structured_success:
        failed.append("no action-dependent structured channel passes all control/shuffle conditions")
    confidence_risk_benefit = bool(
        confidence_delta
        and confidence_delta["confidence_vs_aee_false_safe"].ci_high < 0
    )
    confidence_alignment_benefit = bool(
        confidence_delta
        and confidence_delta["confidence_vs_aee_alignment"].ci_low > 0
    )
    confidence_safety_benefit = bool(
        confidence_delta
        and max(
            confidence_delta["confidence_vs_aee_pair_auprc"].ci_low,
            confidence_delta["confidence_vs_aee_safety_auprc"].ci_low,
        )
        > 0
    )
    confidence_has_broad_benefit = bool(
        confidence_risk_benefit
        and (confidence_alignment_benefit or confidence_safety_benefit)
    )
    gate["confidence_aee_significant_false_safe_benefit"] = confidence_risk_benefit
    gate["confidence_aee_significant_alignment_benefit"] = confidence_alignment_benefit
    gate["confidence_aee_significant_safety_auprc_benefit"] = confidence_safety_benefit
    gate["confidence_aee_broad_empirical_benefit"] = confidence_has_broad_benefit
    write_json(output_dir / "gate3.json", gate)

    if gate_pass:
        recommendation = (
            "Probe evidence supports AEE-WM as the next research direction, but this delivery stops "
            "before any shared-backbone or planning experiment."
        )
    elif not structured_success:
        recommendation = (
            "Do not attach AEE to Qwen+DiT: any consequence-vector gain is not supported by a "
            "learnable action-dependent structured effect channel. Refine the effect target or stop "
            "the current AEE formulation."
        )
    elif bool(config["confidence_aee"]["enabled"]) and confidence_risk_benefit:
        recommendation = (
            "Do not attach the current AEE objective. Select Direction B for the next probe-only "
            "study: partially identified / uncertainty-aware world supervision. This is a provisional "
            "direction because confidence weighting significantly reduces false-safe errors relative "
            "to unweighted AEE, but does not significantly improve alignment and lacks reactive-model "
            "validation/test coverage."
        )
    else:
        recommendation = (
            "Do not attach AEE to Qwen+DiT. Retain multi-candidate absolute supervision as the "
            "strong probe baseline and revise the equivalence objective before further work."
        )
    gate_lines = [
        "# Gate 3 decision",
        "",
        f"**Gate 3: {gate['decision']}.**",
        "",
        "This decision uses the predeclared joint criteria; a consequence-vector-only gain cannot pass.",
        "",
        f"- AEE vs absolute: `{json.dumps(aee_vs_absolute, sort_keys=True)}`",
        f"- AEE vs global: `{json.dumps(aee_vs_global, sort_keys=True)}`",
        f"- Structured action-dependent channel success: {structured_success}.",
        f"- Failed conditions: {failed if failed else 'none'}.",
        f"- Recommendation: {recommendation}",
        "",
        "## Paired scene-bootstrap deltas",
        "",
        "Positive values favor AEE for alignment/AUPRC/separation ratio; negative values favor AEE "
        "for false-safe rate and structured error.",
        "",
        "| Comparison | Point | 95% CI |",
        "|---|---:|---:|",
    ]
    for name, interval in delta.items():
        gate_lines.append(
            f"| {name} | {interval.point:.6f} | [{interval.ci_low:.6f}, {interval.ci_high:.6f}] |"
        )
    if confidence_delta:
        gate_lines.extend(
            [
                "",
                "## Confidence-AEE paired deltas",
                "",
                "Positive values favor confidence-AEE for alignment/AUPRC; negative values favor it "
                "for false-safe rate.",
                "",
                "| Comparison | Point | 95% CI |",
                "|---|---:|---:|",
            ]
        )
        for name, interval in confidence_delta.items():
            gate_lines.append(
                f"| {name} | {interval.point:.6f} | "
                f"[{interval.ci_low:.6f}, {interval.ci_high:.6f}] |"
            )
    gate_lines.extend(
        [
            "",
            "## Three-seed alignment stability",
            "",
            "| Seed | Multi-candidate absolute | AEE | Delta |",
            "|---:|---:|---:|---:|",
        ]
    )
    for run_seed in seeds:
        absolute_value = next(
            float(row["per_scene_effect_alignment"])
            for row in run_rows
            if row["method"] == "multi_candidate_absolute" and int(row["seed"]) == run_seed
        )
        aee_value = next(
            float(row["per_scene_effect_alignment"])
            for row in run_rows
            if row["method"] == "aee" and int(row["seed"]) == run_seed
        )
        gate_lines.append(
            f"| {run_seed} | {absolute_value:.6f} | {aee_value:.6f} | "
            f"{aee_value - absolute_value:+.6f} |"
        )
    gate_lines.extend(
        [
            "",
            "## Structured action-dependence checks",
            "",
            "A channel passes only if AEE beats scene-only, train-mean, and zero controls under "
            "scene bootstrap and within-scene action shuffling significantly harms its primary metric.",
            "",
            "| Channel | Metric | Beats all controls | Shuffle gap | 95% CI | Pass |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for channel in structured_channels:
        shuffle = channel["shuffle_gap"]
        channel_pass = bool(
            channel["beats_all_controls"] and channel["shuffle_significantly_hurts"]
        )
        gate_lines.append(
            f"| {channel['channel']} | {channel['primary_metric']} | "
            f"{channel['beats_all_controls']} | {float(shuffle['point']):.6f} | "
            f"[{float(shuffle['ci_low']):.6f}, {float(shuffle['ci_high']):.6f}] | "
            f"{channel_pass} |"
        )
    gate_lines.extend(
        [
            "",
            f"Confidence-AEE significant false-safe benefit: {confidence_risk_benefit}; "
            f"significant alignment benefit: {confidence_alignment_benefit}; significant safety-AUPRC "
            f"benefit: {confidence_safety_benefit}; broad benefit: {confidence_has_broad_benefit}.",
            "",
        "Development stops here. Qwen+DiT world loss is not implemented, planning training is not run, "
        "and PDMS/EPDMS are not populated.",
        "",
        ]
    )
    args.gate_report_path.resolve().write_text("\n".join(gate_lines), encoding="utf-8")
    print(json.dumps({"gate3": gate["decision"], "failed_conditions": failed}, indent=2))


if __name__ == "__main__":
    main()
