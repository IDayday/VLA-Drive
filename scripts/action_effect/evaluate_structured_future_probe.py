#!/usr/bin/env python3
"""Evaluate trajectory-aligned structured probes for action collapse."""

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
import torch  # noqa: E402
import yaml  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.action_effect.cache_io import write_json, write_jsonl  # noqa: E402
from research.action_effect.losses import StructuredFutureLoss  # noqa: E402
from research.action_effect.metrics import compute_structured_collapse_metrics  # noqa: E402
from research.action_effect.probe_data import load_probe_arrays, load_structured_targets  # noqa: E402
from research.action_effect.structured_future import FutureTubeConfig, candidate_aligned_grid  # noqa: E402


METRICS = (
    "factual_prediction_error",
    "action_shuffle_gap",
    "candidate_sensitivity",
    "action_gap",
    "equivalence_leakage",
    "effect_alignment",
    "risk_false_safe_rate",
)


def _yaml(path: Path) -> dict[str, Any]:
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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _plots(summary: list[dict[str, Any]], representative: dict[str, Any], output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for axis, metric in zip(
        axes, ("factual_prediction_error", "action_shuffle_gap", "effect_alignment")
    ):
        values = [row[f"{metric}_mean"] for row in summary]
        axis.bar(np.arange(len(summary)), values)
        axis.set_xticks(np.arange(len(summary)))
        axis.set_xticklabels([row["control"].replace("_probe", "") for row in summary], rotation=40, ha="right")
        axis.set_title(metric.replace("_", " "))
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "structured_collapse_comparison.png", dpi=180)
    plt.close(figure)

    true = representative["pair_true_distance"]
    predicted = representative["pair_predicted_distance"]
    figure, axis = plt.subplots(figsize=(7, 5.5))
    axis.scatter(true, predicted, s=8, alpha=0.25, edgecolors="none")
    axis.set_xlabel("Replay-grounded consequence distance")
    axis.set_ylabel("Predicted future-tube RMS distance")
    axis.set_title("Structured factual-only action-effect alignment")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "structured_predicted_vs_true_effect.png", dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/action_effect/structured_factual_only.yaml",
    )
    parser.add_argument(
        "--target-config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/action_effect/structured_future.yaml",
    )
    parser.add_argument("--candidate-cache", type=Path)
    parser.add_argument("--consequence-cache", type=Path)
    parser.add_argument("--pair-cache", type=Path)
    parser.add_argument("--scene-feature-cache", type=Path)
    parser.add_argument("--structured-future-cache", type=Path)
    parser.add_argument("--factual-probe-dir", type=Path)
    parser.add_argument("--probe-dir", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--bootstrap-samples", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _yaml(args.config.resolve())
    target_config = FutureTubeConfig.from_mapping(_yaml(args.target_config.resolve()))
    data = config["data"]
    candidate_cache = _cache(args.candidate_cache, str(data["candidate_cache"]))
    consequence_cache = _cache(args.consequence_cache, str(data["consequence_cache"]))
    pair_cache = _cache(args.pair_cache, str(data["pair_cache"]))
    scene_feature_cache = _cache(args.scene_feature_cache, str(data["scene_feature_cache"]))
    structured_cache = _cache(args.structured_future_cache, str(data["structured_future_cache"]))
    factual_probe_dir = (
        args.factual_probe_dir.resolve()
        if args.factual_probe_dir is not None
        else _root("ACTION_EFFECT_OUTPUT_ROOT") / str(data["factual_probe_dir"])
    )
    probe_dir = (
        args.probe_dir.resolve()
        if args.probe_dir is not None
        else _root("ACTION_EFFECT_OUTPUT_ROOT") / "structured_factual_only" / "pilot_tiny"
    )
    report_dir = (
        args.report_dir.resolve()
        if args.report_dir is not None
        else REPOSITORY_ROOT / "reports/action_effect_world_model/structured_collapse_artifacts"
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    with (factual_probe_dir / "split.json").open("r", encoding="utf-8") as stream:
        split = json.load(stream)
    arrays, _, _, _, _, _ = load_probe_arrays(
        candidate_cache=candidate_cache,
        consequence_cache=consequence_cache,
        scene_feature_cache=scene_feature_cache,
        fit_scene_ids=split["fit"],
        assumption="log_replay",
    )
    target, valid = load_structured_targets(structured_cache, arrays)
    with (probe_dir / "data_summary.json").open("r", encoding="utf-8") as stream:
        probe_data_summary = json.load(stream)
    structured_loss = StructuredFutureLoss(
        torch.tensor(probe_data_summary["binary_positive_weight"], dtype=torch.float32)
    )
    factual_heldout_indices = np.flatnonzero(
        arrays.anchor
        & arrays.accepted
        & valid
        & np.isin(arrays.scene_ids, split["heldout"])
    )
    _, local_x, local_y = candidate_aligned_grid(np.asarray([0, 0, 0, 0, 0]), target_config)
    ego_mask = (local_x >= -1.0) & (local_x <= 4.0) & (np.abs(local_y) <= 1.25)
    bootstrap = int(args.bootstrap_samples or config["experiment"]["bootstrap_samples"])
    confidence = float(config["experiment"]["bootstrap_confidence"])
    rows: list[dict[str, Any]] = []
    details: dict[tuple[str, int], dict[str, Any]] = {}
    for control_dir in sorted(path for path in probe_dir.iterdir() if path.is_dir()):
        for seed_dir in sorted(control_dir.glob("seed_*")):
            prediction_path = seed_dir / "predictions.npz"
            result_path = seed_dir / "result.json"
            if not prediction_path.is_file() or not result_path.is_file():
                continue
            seed = int(seed_dir.name.removeprefix("seed_"))
            with np.load(prediction_path) as payload:
                prediction = np.asarray(payload["structured_future_prediction"], dtype=np.float16)
            intervals, run_details = compute_structured_collapse_metrics(
                arrays=arrays,
                structured_target=target,
                structured_valid=valid,
                raw_prediction=prediction,
                heldout_scene_ids=split["heldout"],
                pair_path=pair_cache / "pairs.jsonl",
                safe_threshold=float(config["risk"]["hard_safe_threshold"]),
                minimum_clearance_normalized=float(config["risk"]["minimum_clearance_normalized"]),
                ego_grid_mask=ego_mask,
                bootstrap_samples=bootstrap,
                confidence=confidence,
                seed=seed,
            )
            with result_path.open("r", encoding="utf-8") as stream:
                training = json.load(stream)
            objective = structured_loss(
                torch.from_numpy(prediction[factual_heldout_indices].astype(np.float32)),
                torch.from_numpy(target[factual_heldout_indices].astype(np.float32)),
            )
            row: dict[str, Any] = {
                "control": control_dir.name,
                "seed": seed,
                "total_parameters": training["total_parameters"],
                "trainable_parameters": training["trainable_parameters"],
                "best_validation_loss": training["best_validation_loss"],
                "structured_objective": float(objective["total"]),
                "structured_binary_loss": float(objective["binary"]),
                "structured_velocity_loss": float(objective["velocity"]),
                "structured_clearance_loss": float(objective["clearance"]),
            }
            for metric in METRICS:
                interval = intervals[metric]
                row[f"{metric}_point"] = interval.point
                row[f"{metric}_ci_low"] = interval.ci_low
                row[f"{metric}_ci_high"] = interval.ci_high
            rows.append(row)
            details[(control_dir.name, seed)] = run_details
            write_json(seed_dir / "collapse_metrics.json", {name: asdict(value) for name, value in intervals.items()})
    if not rows:
        raise RuntimeError("no structured prediction runs found")
    _write_csv(report_dir / "metrics_by_run.csv", rows)
    summary: list[dict[str, Any]] = []
    for control in sorted({row["control"] for row in rows}):
        selected = [row for row in rows if row["control"] == control]
        item: dict[str, Any] = {
            "control": control,
            "seeds": len(selected),
            "total_parameters": selected[0]["total_parameters"],
            "trainable_parameters": selected[0]["trainable_parameters"],
            "structured_objective_mean": float(
                np.mean([row["structured_objective"] for row in selected])
            ),
            "structured_binary_loss_mean": float(
                np.mean([row["structured_binary_loss"] for row in selected])
            ),
            "structured_velocity_loss_mean": float(
                np.mean([row["structured_velocity_loss"] for row in selected])
            ),
            "structured_clearance_loss_mean": float(
                np.mean([row["structured_clearance_loss"] for row in selected])
            ),
        }
        for metric in METRICS:
            item[f"{metric}_mean"] = float(np.mean([row[f"{metric}_point"] for row in selected]))
            bootstrap_arrays = [details[(row["control"], int(row["seed"]))]["bootstrap_values"][metric] for row in selected]
            minimum = min(len(value) for value in bootstrap_arrays)
            averaged = np.mean(np.stack([value[:minimum] for value in bootstrap_arrays]), axis=0)
            alpha = (1.0 - confidence) / 2
            low, high = np.quantile(averaged, [alpha, 1 - alpha])
            item[f"{metric}_ci_low"] = float(low)
            item[f"{metric}_ci_high"] = float(high)
        summary.append(item)
    _write_csv(report_dir / "metrics_summary.csv", summary)
    by_control = {row["control"]: row for row in summary}
    required = {"constant_mean_control", "scene_action_probe", "shuffled_action_probe", "scene_only_probe"}
    if not required <= set(by_control):
        raise RuntimeError(f"structured controls are incomplete: {sorted(required - set(by_control))}")
    scene_action = by_control["scene_action_probe"]
    constant = by_control["constant_mean_control"]
    factual = float(scene_action["factual_prediction_error_mean"])
    gate = {
        "factual_target_learnable": float(scene_action["structured_objective_mean"])
        < 0.95 * float(constant["structured_objective_mean"]),
        "unweighted_map_mae_improves": factual
        < float(constant["factual_prediction_error_mean"]),
        "small_action_shuffle_gap": abs(float(scene_action["action_shuffle_gap_mean"])) <= max(0.02 * factual, 0.002),
        "low_effect_alignment": float(scene_action["effect_alignment_mean"]) < 0.25,
        "high_false_safe_rate": float(scene_action["risk_false_safe_rate_mean"]) >= 0.20,
        "note": "Single-seed structured confirmation; the consequence probe carries the three-seed diagnostic.",
    }
    gate["supports_structured_action_collapse"] = bool(
        gate["factual_target_learnable"]
        and sum((gate["small_action_shuffle_gap"], gate["low_effect_alignment"], gate["high_false_safe_rate"])) >= 2
    )
    write_json(report_dir / "gate2_structured.json", gate)
    representative_key = sorted(key for key in details if key[0] == "scene_action_probe")[0]
    _plots(summary, details[representative_key], report_dir)
    false_safe = np.flatnonzero(details[representative_key]["false_safe"])
    write_jsonl(
        report_dir / "false_safe_examples.jsonl",
        [
            {
                "scene_id": str(arrays.scene_ids[index]),
                "candidate_id": str(arrays.candidate_ids[index]),
                "candidate_index": int(arrays.candidate_indices[index]),
                "structured_mae": float(details[representative_key]["candidate_error"][index]),
                "failure_type": "structured_future_false_safe",
            }
            for index in false_safe[:50]
        ],
    )
    print(json.dumps({"report_dir": str(report_dir), "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
