"""Aggregate whole-log temporal-consequence cross-validation results.

This module deliberately chooses one epoch and one deployment policy across
folds.  It prevents the final full-data/Navtest model from inheriting a
different validation-tuned policy from every fold.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


DEPLOYMENT_KEYS = (
    "residual_scale",
    "switch_penalty",
    "safety_floor",
    "safety_relative_tolerance",
)
SAFETY_FACTORS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "time_to_collision_within_bound",
)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _deployment_key(item: Mapping[str, object]) -> Tuple[float, ...]:
    return tuple(float(item[key]) for key in DEPLOYMENT_KEYS)


def _weighted_mean(values: Iterable[Tuple[float, int]]) -> float:
    pairs = list(values)
    total = sum(weight for _value, weight in pairs)
    if total <= 0:
        raise ValueError("Cannot compute a weighted mean with zero weight")
    return sum(value * weight for value, weight in pairs) / total


def _validate_folds(payloads: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    if not payloads:
        raise ValueError("No training_results.json payloads were supplied")
    fold_records = [payload["metadata"]["fold"] for payload in payloads]
    num_folds = {int(record["num_folds"]) for record in fold_records}
    fold_seeds = {int(record["fold_seed"]) for record in fold_records}
    if len(num_folds) != 1 or len(fold_seeds) != 1:
        raise RuntimeError("Cross-validation fold metadata disagree")
    expected = next(iter(num_folds))
    indices = [int(record["fold_index"]) for record in fold_records]
    if len(indices) != len(set(indices)):
        raise RuntimeError("Duplicate fold index")

    validation_sets = [set(record["validation_logs"]) for record in fold_records]
    for index, validation in enumerate(validation_sets):
        training = set(fold_records[index]["train_logs"])
        if validation.intersection(training):
            raise RuntimeError(f"Fold {indices[index]} has train/validation leakage")
        for other in range(index):
            if validation.intersection(validation_sets[other]):
                raise RuntimeError("Validation logs overlap across folds")
    return {
        "declared_num_folds": expected,
        "observed_fold_count": len(payloads),
        "complete": sorted(indices) == list(range(expected)),
        "fold_seed": next(iter(fold_seeds)),
        "fold_indices": sorted(indices),
        "validation_log_count": len(set().union(*validation_sets)),
        "validation_scene_count": sum(
            int(record["validation_scene_count"]) for record in fold_records
        ),
    }


def _aggregate_deployments(
    payloads: Sequence[Mapping[str, object]],
    safety_tolerance: float,
    required_epoch: int,
) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[float, ...], List[Mapping[str, object]]] = defaultdict(list)
    for payload in payloads:
        retained_epoch = int(payload["metadata"]["retained_epoch"])
        if retained_epoch != required_epoch or int(payload["best_epoch"]) != required_epoch:
            raise RuntimeError(
                "Deployment sweep weights do not match the common epoch: "
                f"required={required_epoch} retained={retained_epoch} "
                f"best={payload['best_epoch']}"
            )
        for item in payload["deployment_sweep"]:
            if int(item.get("weight_epoch", -1)) != required_epoch:
                raise RuntimeError(
                    "Deployment sweep row has no matching weight_epoch: "
                    f"required={required_epoch} row={item.get('weight_epoch')}"
                )
            grouped[_deployment_key(item)].append(item)
    if not grouped:
        raise RuntimeError("Deployment sweeps are empty")
    if any(len(items) != len(payloads) for items in grouped.values()):
        raise RuntimeError("Deployment grids differ across folds")

    rows: List[Dict[str, object]] = []
    for key, items in sorted(grouped.items()):
        weights = [int(item["scene_count"]) for item in items]
        base_regret = _weighted_mean(
            (float(item["base_top1_regret"]), weight)
            for item, weight in zip(items, weights)
        )
        model_regret = _weighted_mean(
            (float(item["model_top1_regret"]), weight)
            for item, weight in zip(items, weights)
        )
        factor_means = {
            factor: _weighted_mean(
                (
                    float(item["selected_factor_delta"][factor]),
                    weight,
                )
                for item, weight in zip(items, weights)
            )
            for factor in items[0]["selected_factor_delta"]
        }
        factor_worst = {
            factor: min(
                float(item["selected_factor_delta"][factor]) for item in items
            )
            for factor in items[0]["selected_factor_delta"]
        }
        deltas = [float(item["selected_pdms_delta"]) for item in items]
        row = {
            **dict(zip(DEPLOYMENT_KEYS, key)),
            "scene_count": sum(weights),
            "fold_count": len(items),
            "weighted_pdms_delta": _weighted_mean(zip(deltas, weights)),
            "worst_fold_pdms_delta": min(deltas),
            "all_folds_positive": all(delta > 0.0 for delta in deltas),
            "weighted_regret_reduction_fraction": (
                1.0 - model_regret / max(base_regret, 1e-12)
            ),
            "weighted_factor_delta": factor_means,
            "worst_fold_factor_delta": factor_worst,
        }
        row["safety_nonregressing"] = all(
            factor_worst[factor] >= -safety_tolerance for factor in SAFETY_FACTORS
        )
        rows.append(row)
    return rows


def _aggregate_epochs(
    payloads: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    grouped: Dict[int, List[Mapping[str, object]]] = defaultdict(list)
    for payload in payloads:
        for item in payload["history"]:
            epoch = int(item["epoch"])
            if epoch >= 0:
                grouped[epoch].append(item["validation"])
    rows = []
    for epoch, items in sorted(grouped.items()):
        if len(items) != len(payloads):
            continue
        weights = [int(item["scene_count"]) for item in items]
        deltas = [float(item["selected_pdms_delta"]) for item in items]
        base_regret = _weighted_mean(
            (float(item["base_top1_regret"]), weight)
            for item, weight in zip(items, weights)
        )
        model_regret = _weighted_mean(
            (float(item["model_top1_regret"]), weight)
            for item, weight in zip(items, weights)
        )
        rows.append(
            {
                "epoch": epoch,
                "weighted_pdms_delta": _weighted_mean(zip(deltas, weights)),
                "worst_fold_pdms_delta": min(deltas),
                "all_folds_positive": all(delta > 0.0 for delta in deltas),
                "weighted_pairwise_accuracy_delta_ge_0_02": _weighted_mean(
                    (
                        float(item["pairwise_accuracy_delta_ge_0_02"]),
                        weight,
                    )
                    for item, weight in zip(items, weights)
                ),
                "weighted_regret_reduction_fraction": (
                    1.0 - model_regret / max(base_regret, 1e-12)
                ),
            }
        )
    return rows


def summarize_cv(
    payloads: Sequence[Mapping[str, object]],
    *,
    safety_tolerance: float = 5e-4,
) -> Dict[str, object]:
    fold_audit = _validate_folds(payloads)
    epochs = _aggregate_epochs(payloads)
    if not epochs:
        raise RuntimeError("No epoch is complete across every fold")
    positive_epochs = [item for item in epochs if item["all_folds_positive"]]
    forced_epochs = {
        payload["metadata"].get("forced_retained_epoch") for payload in payloads
    }
    if forced_epochs == {None}:
        best_epoch = max(
            positive_epochs or epochs,
            key=lambda item: (
                float(item["weighted_pdms_delta"]),
                float(item["worst_fold_pdms_delta"]),
            ),
        )
        epoch_selection_mode = "discovery"
    elif len(forced_epochs) == 1 and None not in forced_epochs:
        locked_epoch = int(next(iter(forced_epochs)))
        matches = [item for item in epochs if int(item["epoch"]) == locked_epoch]
        if len(matches) != 1:
            raise RuntimeError(f"Locked epoch {locked_epoch} is not complete across folds")
        best_epoch = matches[0]
        epoch_selection_mode = "locked_replay"
    else:
        raise RuntimeError("CV folds mix discovery and locked replay epochs")

    common_epoch = int(best_epoch["epoch"])
    epoch_weights_aligned = all(
        int(payload.get("best_epoch", -1)) == common_epoch
        and int(payload["metadata"].get("retained_epoch", -1)) == common_epoch
        for payload in payloads
    )
    if epoch_weights_aligned:
        deployments = _aggregate_deployments(
            payloads, safety_tolerance, common_epoch
        )
        robust_deployments = [
            item
            for item in deployments
            if item["all_folds_positive"] and item["safety_nonregressing"]
        ]
        best_deployment = max(
            robust_deployments or deployments,
            key=lambda item: (
                float(item["weighted_pdms_delta"]),
                float(item["worst_fold_pdms_delta"]),
                -float(item["residual_scale"]),
            ),
        )
    else:
        deployments = []
        robust_deployments = []
        best_deployment = None
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "fold_audit": fold_audit,
        "safety_tolerance": safety_tolerance,
        "epoch_selection_mode": epoch_selection_mode,
        "common_epoch_weights_aligned": epoch_weights_aligned,
        "common_epoch": best_epoch,
        "common_deployment": best_deployment,
        "robust_deployment_available": bool(robust_deployments),
        "epoch_results": epochs,
        "deployment_results": deployments,
    }


def _markdown(summary: Mapping[str, object]) -> str:
    fold = summary["fold_audit"]
    epoch = summary["common_epoch"]
    deployment = summary["common_deployment"]
    if deployment is None:
        return "\n".join(
            (
                "# Temporal Consequence Cross-Validation",
                "",
                f"- Complete folds: `{fold['complete']}` "
                f"({fold['observed_fold_count']}/{fold['declared_num_folds']})",
                f"- Validation logs/scenes: `{fold['validation_log_count']}` / "
                f"`{fold['validation_scene_count']}`",
                f"- Discovered common epoch: `{epoch['epoch']}`",
                f"- Common-epoch PDMS delta: `{epoch['weighted_pdms_delta']:.8f}`",
                "- Epoch weights aligned: `False`",
                "- Deployment policy: `NOT MATERIALIZABLE`",
                "",
                "The discovery pass selected a common epoch, but one or more "
                "fold artifacts contain weights from a different epoch. Replay "
                "every fold with `--retained-epoch` before policy selection or "
                "Navtest promotion.",
                "",
            )
        )
    factors = deployment["weighted_factor_delta"]
    return "\n".join(
        (
            "# Temporal Consequence Cross-Validation",
            "",
            f"- Complete folds: `{fold['complete']}` "
            f"({fold['observed_fold_count']}/{fold['declared_num_folds']})",
            f"- Validation logs/scenes: `{fold['validation_log_count']}` / "
            f"`{fold['validation_scene_count']}`",
            f"- Common epoch: `{epoch['epoch']}`",
            f"- Common-epoch PDMS delta: `{epoch['weighted_pdms_delta']:.8f}`",
            f"- Common deployment: `{dict((key, deployment[key]) for key in DEPLOYMENT_KEYS)}`",
            f"- Deployment PDMS delta: `{deployment['weighted_pdms_delta']:.8f}`",
            f"- Worst-fold delta: `{deployment['worst_fold_pdms_delta']:.8f}`",
            f"- Regret reduction: `{deployment['weighted_regret_reduction_fraction']:.2%}`",
            f"- Safety-nonregressing: `{deployment['safety_nonregressing']}`",
            "",
            "## Weighted factor deltas",
            "",
            *[f"- {key}: `{value:+.8f}`" for key, value in factors.items()],
            "",
            "The common epoch and policy are chosen only from whole-log Navtrain "
            "folds. Navtest is not consulted.",
            "",
        )
    )


def _console_summary(summary: Mapping[str, object]) -> Dict[str, object]:
    """Return a compact promotion record while full search grids stay on disk."""

    return {
        "schema_version": summary["schema_version"],
        "created_at_utc": summary["created_at_utc"],
        "fold_audit": summary["fold_audit"],
        "common_epoch": summary["common_epoch"],
        "common_deployment": summary["common_deployment"],
        "robust_deployment_available": summary["robust_deployment_available"],
        "output_contains_full_grids": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--safety-tolerance", type=float, default=5e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.run_root.glob("fold_*/training_results.json"))
    payloads = [json.loads(path.read_text()) for path in paths]
    summary = summarize_cv(payloads, safety_tolerance=args.safety_tolerance)
    summary["input_paths"] = [str(path.resolve()) for path in paths]
    _atomic_json(args.output, summary)
    report = args.report or args.output.with_suffix(".md")
    report.parent.mkdir(parents=True, exist_ok=True)
    temporary = report.with_name(f".{report.name}.tmp-{os.getpid()}")
    temporary.write_text(_markdown(summary))
    os.replace(temporary, report)
    print(json.dumps(_console_summary(summary), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
