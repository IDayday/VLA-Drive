#!/usr/bin/env python3
"""Select one conservative deployment policy across disjoint Navtrain folds."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from local_stage2.sweep_m0_conservative_reference_fold import (
    REFERENCE_POLICY_FIELDS,
)
from local_stage2.train_independent_scorer import (
    _atomic_json_dump,
    _atomic_torch_save,
    _sha256,
)
from navsim.agents.EpisodeDrive.score_module.independent_ranker import FACTOR_KEYS


SAFETY_FACTORS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "time_to_collision_within_bound",
)

_POLICY_TO_ARGUMENT = {
    "gain_quantile_index": "reference_gain_quantile_index",
    "minimum_lcb_gain": "reference_minimum_lcb_gain",
    "maximum_safety_worse_probability": (
        "reference_maximum_safety_worse_probability"
    ),
    "minimum_safe_improvement_probability": (
        "reference_minimum_safe_improvement_probability"
    ),
}


def log_cluster_bootstrap_from_sufficient_statistics(
    log_statistics: Mapping[str, Mapping[str, object]],
    *,
    seed: int,
    replicates: int,
) -> Tuple[float, float]:
    """Resample physical logs while retaining the scene-weighted estimand."""

    if replicates <= 0:
        return float("nan"), float("nan")
    names = sorted(log_statistics)
    if not names:
        raise ValueError("log statistics are empty")
    counts = np.asarray(
        [int(log_statistics[name]["scene_count"]) for name in names],
        dtype=np.int64,
    )
    sums = np.asarray(
        [float(log_statistics[name]["delta_sum"]) for name in names],
        dtype=np.float64,
    )
    if np.any(counts <= 0):
        raise ValueError("every physical log must contain at least one scene")
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(names), size=(replicates, len(names)))
    values = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _policy_key(policy: Mapping[str, object]) -> Tuple[object, ...]:
    return tuple(policy[name] for name in REFERENCE_POLICY_FIELDS)


def _merge_log_statistics(
    rows: Sequence[Mapping[str, Mapping[str, object]]],
) -> Dict[str, Mapping[str, object]]:
    merged: Dict[str, Mapping[str, object]] = {}
    for row in rows:
        overlap = set(merged).intersection(row)
        if overlap:
            raise RuntimeError(
                f"validation physical logs overlap across folds: {sorted(overlap)[:3]}"
            )
        merged.update({str(key): value for key, value in row.items()})
    return merged


def aggregate_common_reference_policy(
    payloads: Sequence[Mapping[str, object]],
    *,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 20260902,
    safety_tolerance: float = 5.0e-4,
) -> Dict[str, object]:
    """Aggregate identical policy grids and apply the predeclared robust gate."""

    if len(payloads) < 2:
        raise ValueError("at least two fold sweeps are required")
    ordered = sorted(payloads, key=lambda payload: int(payload["fold_index"]))
    expected_folds = int(ordered[0]["num_folds"])
    if len(ordered) != expected_folds:
        raise RuntimeError("not every declared fold has a policy sweep")
    if [int(row["fold_index"]) for row in ordered] != list(range(expected_folds)):
        raise RuntimeError("fold indices are incomplete or duplicated")
    if any(int(row["artifact_epoch"]) != 7 for row in ordered):
        raise RuntimeError("common-policy selection requires locked epoch 7")
    selection_sources = {str(row["selection_source"]) for row in ordered}
    if len(selection_sources) != 1:
        raise RuntimeError("fold selection sources differ")
    validation_sets = [set(row["validation_physical_logs"]) for row in ordered]
    for left in range(len(validation_sets)):
        for right in range(left):
            if validation_sets[left].intersection(validation_sets[right]):
                raise RuntimeError("validation logs overlap across folds")

    grids = [row["policy_grid"] for row in ordered]
    grid_size = int(ordered[0]["policy_grid_size"])
    if any(int(row["policy_grid_size"]) != grid_size for row in ordered):
        raise RuntimeError("policy grid sizes differ")
    if any(len(grid) != grid_size for grid in grids):
        raise RuntimeError("a policy grid is incomplete")
    rows: List[Dict[str, object]] = []
    for policy_index in range(grid_size):
        fold_items = [grid[policy_index] for grid in grids]
        policy_ids = {int(item["policy_id"]) for item in fold_items}
        policy_keys = {_policy_key(item["policy"]) for item in fold_items}
        if policy_ids != {policy_index} or len(policy_keys) != 1:
            raise RuntimeError("policy grids are not aligned across folds")
        fold_metrics: List[Dict[str, object]] = []
        for fold_index, item in enumerate(fold_items):
            interval = log_cluster_bootstrap_from_sufficient_statistics(
                item["per_log_sufficient_statistics"],
                seed=bootstrap_seed + policy_index * 101 + fold_index,
                replicates=bootstrap_replicates,
            )
            fold_metrics.append(
                {
                    "fold_index": fold_index,
                    "scene_count": int(item["scene_count"]),
                    "physical_log_count": int(item["physical_log_count"]),
                    "selected_pdms": float(item["selected_pdms"]),
                    "base_selected_pdms": float(item["base_selected_pdms"]),
                    "selected_delta": float(item["selected_delta"]),
                    "bootstrap_95ci": [float(interval[0]), float(interval[1])],
                    "switch_rate": float(item["switch_rate"]),
                    "wins": int(item["wins"]),
                    "losses": int(item["losses"]),
                    "ties": int(item["ties"]),
                    "factor_delta": dict(item["factor_delta"]),
                    "selected_factors": dict(item["selected_factors"]),
                    "base_selected_factors": dict(item["base_selected_factors"]),
                }
            )
        scene_counts = np.asarray(
            [item["scene_count"] for item in fold_metrics], dtype=np.float64
        )
        weights = scene_counts / scene_counts.sum()
        deltas = np.asarray(
            [item["selected_delta"] for item in fold_metrics], dtype=np.float64
        )
        combined_logs = _merge_log_statistics(
            [item["per_log_sufficient_statistics"] for item in fold_items]
        )
        combined_interval = log_cluster_bootstrap_from_sufficient_statistics(
            combined_logs,
            seed=bootstrap_seed + policy_index * 101 + expected_folds,
            replicates=bootstrap_replicates,
        )
        factor_worst = {
            key: min(float(item["factor_delta"][key]) for item in fold_metrics)
            for key in FACTOR_KEYS
        }
        factor_weighted = {
            key: float(
                np.sum(
                    weights
                    * np.asarray(
                        [item["factor_delta"][key] for item in fold_metrics],
                        dtype=np.float64,
                    )
                )
            )
            for key in FACTOR_KEYS
        }
        all_points_positive = bool(np.all(deltas > 0.0))
        all_lowers_positive = all(
            float(item["bootstrap_95ci"][0]) > 0.0 for item in fold_metrics
        )
        safety_nonregressing = all(
            factor_worst[key] >= -safety_tolerance for key in SAFETY_FACTORS
        )
        rows.append(
            {
                "policy_id": policy_index,
                "policy": dict(fold_items[0]["policy"]),
                "scene_count": int(scene_counts.sum()),
                "physical_log_count": len(combined_logs),
                "scene_weighted_selected_delta": float(np.sum(weights * deltas)),
                "worst_fold_selected_delta": float(deltas.min()),
                "worst_fold_bootstrap_95ci_lower": min(
                    float(item["bootstrap_95ci"][0]) for item in fold_metrics
                ),
                "combined_log_bootstrap_95ci": [
                    float(combined_interval[0]),
                    float(combined_interval[1]),
                ],
                "scene_weighted_switch_rate": float(
                    np.sum(
                        weights
                        * np.asarray(
                            [item["switch_rate"] for item in fold_metrics],
                            dtype=np.float64,
                        )
                    )
                ),
                "all_fold_point_deltas_positive": all_points_positive,
                "all_fold_bootstrap_lowers_positive": all_lowers_positive,
                "safety_nonregressing": bool(safety_nonregressing),
                "robust_eligible": bool(
                    all_points_positive
                    and all_lowers_positive
                    and safety_nonregressing
                ),
                "weighted_factor_delta": factor_weighted,
                "worst_fold_factor_delta": factor_worst,
                "folds": fold_metrics,
            }
        )

    robust = [row for row in rows if row["robust_eligible"]]
    diagnostic_best = max(
        rows,
        key=lambda row: (
            float(row["worst_fold_selected_delta"]),
            float(row["scene_weighted_selected_delta"]),
            -float(row["scene_weighted_switch_rate"]),
            -int(row["policy_id"]),
        ),
    )
    selected: Optional[Dict[str, object]] = None
    if robust:
        selected = max(
            robust,
            key=lambda row: (
                float(row["worst_fold_selected_delta"]),
                float(row["combined_log_bootstrap_95ci"][0]),
                float(row["scene_weighted_selected_delta"]),
                -float(row["scene_weighted_switch_rate"]),
                -int(row["policy_id"]),
            ),
        )
    default_policy = ordered[0]["artifact_policy"]
    default_matches = [row for row in rows if row["policy"] == default_policy]
    if len(default_matches) != 1:
        raise RuntimeError("artifact default policy is not common across folds")
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "strategy": "locked_epoch_common_conservative_policy_across_five_log_folds",
        "fold_count": expected_folds,
        "validation_scene_count": sum(
            int(row["validation_scene_count"]) for row in ordered
        ),
        "validation_physical_log_count": len(set().union(*validation_sets)),
        "policy_grid_size": grid_size,
        "safety_tolerance": safety_tolerance,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
        "selection_priority": [
            "all_fold_point_delta_positive",
            "all_fold_log_bootstrap_lower_positive",
            "worst_fold_NOC_DAC_TTC_delta_at_least_minus_tolerance",
            "maximize_worst_fold_delta",
            "maximize_combined_log_bootstrap_lower",
            "maximize_scene_weighted_delta",
            "minimize_switch_rate",
        ],
        "robust_policy_count": len(robust),
        "robust_refit_gate_passed": selected is not None,
        "selected_policy_result": selected,
        "diagnostic_best_without_gate": diagnostic_best,
        "fixed_artifact_policy_result": default_matches[0],
        "policy_results": rows,
        "selection_source": next(iter(selection_sources)),
        "fold_artifacts": [
            {
                "fold_index": int(row["fold_index"]),
                "artifact": row["artifact"],
                "artifact_sha256": row["artifact_sha256"],
                "split_manifest": row["split_manifest"],
                "split_manifest_sha256": row["split_manifest_sha256"],
            }
            for row in ordered
        ],
        "validation_logs_disjoint": True,
        "navtest_used_for_selection": False,
    }


def _selected_fold_validation(
    source_validation: Mapping[str, object],
    fold: Mapping[str, object],
) -> Dict[str, object]:
    best_of_64 = float(source_validation["best_of_64_pdms"])
    selected_pdms = float(fold["selected_pdms"])
    base_pdms = float(fold["base_selected_pdms"])
    return {
        "scene_count": int(fold["scene_count"]),
        "physical_log_count": int(fold["physical_log_count"]),
        "selected_pdms": selected_pdms,
        "base_selected_pdms": base_pdms,
        "best_of_64_pdms": best_of_64,
        "selected_delta": float(fold["selected_delta"]),
        "selected_delta_log_bootstrap_95ci": [
            float(value) for value in fold["bootstrap_95ci"]
        ],
        "selected_regret": best_of_64 - selected_pdms,
        "base_regret": best_of_64 - base_pdms,
        "switch_rate": float(fold["switch_rate"]),
        "wins": int(fold["wins"]),
        "losses": int(fold["losses"]),
        "ties": int(fold["ties"]),
        "selected_factors": dict(fold["selected_factors"]),
        "base_selected_factors": dict(fold["base_selected_factors"]),
    }


def materialize_common_policy_artifact(
    source_artifact: Mapping[str, object],
    summary: Mapping[str, object],
    *,
    source_artifact_path: Path,
) -> Dict[str, object]:
    """Attach the common deployment thresholds without changing weights."""

    selected = summary.get("selected_policy_result")
    if not isinstance(selected, Mapping) or not summary.get(
        "robust_refit_gate_passed"
    ):
        raise RuntimeError("no robust common policy can be materialized")
    derived = copy.deepcopy(dict(source_artifact))
    policy = dict(selected["policy"])
    residual_config = dict(derived["residual_config"])
    residual_config.update(policy)
    derived["residual_config"] = residual_config
    fold_manifest = copy.deepcopy(dict(derived["fold_manifest"]))
    fold_args = dict(fold_manifest["args"])
    for policy_name, argument_name in _POLICY_TO_ARGUMENT.items():
        fold_args[argument_name] = policy[policy_name]
    fold_manifest["args"] = fold_args
    fold_residual = dict(fold_manifest.get("residual_config", residual_config))
    fold_residual.update(policy)
    fold_manifest["residual_config"] = fold_residual
    derived["fold_manifest"] = fold_manifest

    source_name = str(derived["checkpoint_selection_source"])
    source_validation = derived.get("validation_by_source", {}).get(
        source_name, derived.get("validation")
    )
    if not isinstance(source_validation, Mapping):
        raise RuntimeError("source artifact lacks validation metrics")
    source_fold_rows = [
        int(row["fold_index"])
        for row in summary["fold_artifacts"]
        if Path(str(row["artifact"])).resolve() == source_artifact_path.resolve()
    ]
    source_fold = source_fold_rows[0] if len(source_fold_rows) == 1 else 0
    fold_matches = [
        row for row in selected["folds"] if int(row["fold_index"]) == source_fold
    ]
    if len(fold_matches) != 1:
        # The materialization source is conventionally fold 0.  Fall back to
        # its fold metadata if a custom path lacks the standard name.
        fold_matches = [
            row for row in selected["folds"] if int(row["fold_index"]) == 0
        ]
    if len(fold_matches) != 1:
        raise RuntimeError("cannot identify source fold metrics")
    validation = _selected_fold_validation(source_validation, fold_matches[0])
    derived["validation"] = validation
    derived["validation_by_source"] = {source_name: validation}
    evidence_core = {
        "strategy": summary["strategy"],
        "policy": policy,
        "validation_scene_count": summary["validation_scene_count"],
        "validation_physical_log_count": summary[
            "validation_physical_log_count"
        ],
        "selected_policy_result": selected,
        "fold_artifacts": summary["fold_artifacts"],
    }
    evidence_digest = hashlib.sha256(
        json.dumps(evidence_core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    derived.update(
        {
            "source_artifact": str(source_artifact_path.resolve()),
            "source_artifact_sha256": _sha256(source_artifact_path),
            "derived_conservative_policy": True,
            "policy_selection_uses_navtest": False,
            "policy_selection_uses_disjoint_physical_logs": True,
            "policy_calibration": {
                "strategy": summary["strategy"],
                "common_policy": policy,
                "fold_count": summary["fold_count"],
                "validation_scene_count": summary["validation_scene_count"],
                "validation_physical_log_count": summary[
                    "validation_physical_log_count"
                ],
                "worst_fold_selected_delta": selected[
                    "worst_fold_selected_delta"
                ],
                "worst_fold_bootstrap_95ci_lower": selected[
                    "worst_fold_bootstrap_95ci_lower"
                ],
                "combined_log_bootstrap_95ci": selected[
                    "combined_log_bootstrap_95ci"
                ],
                "navtest_used_for_selection": False,
            },
            "cross_validation_selection": {
                "schema_version": 2,
                "strategy": summary["strategy"],
                "locked_epoch": 7,
                "fold_count": summary["fold_count"],
                "validation_scene_count": summary["validation_scene_count"],
                "validation_physical_log_count": summary[
                    "validation_physical_log_count"
                ],
                "scene_weighted_selected_delta": selected[
                    "scene_weighted_selected_delta"
                ],
                "worst_fold_selected_delta": selected[
                    "worst_fold_selected_delta"
                ],
                "worst_fold_bootstrap_95ci_lower": selected[
                    "worst_fold_bootstrap_95ci_lower"
                ],
                "robust_refit_gate_passed": True,
                "navtest_used_for_selection": False,
                "evidence_digest": evidence_digest,
            },
        }
    )
    return derived


def _markdown(summary: Mapping[str, object]) -> str:
    selected = summary.get("selected_policy_result")
    lines = [
        "# Wave-12 common conservative-reference policy",
        "",
        "The final epoch and neural weights are fixed independently in each of five disjoint Navtrain physical-log folds. Navtest is not read.",
        "",
        f"- Validation coverage: `{summary['validation_scene_count']}` scenes / `{summary['validation_physical_log_count']}` logs",
        f"- Policies evaluated: `{summary['policy_grid_size']}`",
        f"- Robust eligible policies: `{summary['robust_policy_count']}`",
        f"- All-log refit gate: `{'PASS' if summary['robust_refit_gate_passed'] else 'FAIL'}`",
    ]
    if isinstance(selected, Mapping):
        lines.extend(
            [
                f"- Common policy: `{selected['policy']}`",
                f"- Scene-weighted delta: `{selected['scene_weighted_selected_delta']:+.8f}`",
                f"- Worst-fold delta: `{selected['worst_fold_selected_delta']:+.8f}`",
                f"- Worst-fold bootstrap lower: `{selected['worst_fold_bootstrap_95ci_lower']:+.8f}`",
                f"- Combined log-bootstrap CI: `[{selected['combined_log_bootstrap_95ci'][0]:+.8f}, {selected['combined_log_bootstrap_95ci'][1]:+.8f}]`",
                f"- Switch rate: `{selected['scene_weighted_switch_rate']:.4%}`",
                "",
                "| Fold | Scenes | Logs | Base | Selected | Delta | 95% CI |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in selected["folds"]:
            interval = row["bootstrap_95ci"]
            lines.append(
                f"| {row['fold_index']} | {row['scene_count']} | {row['physical_log_count']} | "
                f"{row['base_selected_pdms']:.6f} | {row['selected_pdms']:.6f} | "
                f"{row['selected_delta']:+.6f} | [{interval[0]:+.6f}, {interval[1]:+.6f}] |"
            )
    else:
        diagnostic = summary["diagnostic_best_without_gate"]
        lines.extend(
            [
                "",
                "No policy passed the all-fold point, all-fold clustered-CI and safety-factor gate.",
                f"The diagnostic-only best worst-fold delta was `{diagnostic['worst_fold_selected_delta']:+.8f}`.",
            ]
        )
    lines.extend(
        [
            "",
            "Policy priority is fixed as worst-fold robustness, combined clustered lower bound, weighted gain, then lower switch rate. Per-fold tuning is forbidden.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-result", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--selection-artifact", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260902)
    parser.add_argument("--safety-tolerance", type=float, default=5.0e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in args.fold_result:
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (args.output_json, args.output_md, args.selection_artifact):
        if path.exists():
            raise FileExistsError(path)
    payloads = [json.loads(path.read_text()) for path in args.fold_result]
    summary = aggregate_common_reference_policy(
        payloads,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        safety_tolerance=args.safety_tolerance,
    )
    summary["fold_result_paths"] = [
        str(path.resolve()) for path in args.fold_result
    ]
    summary["fold_result_sha256"] = [_sha256(path) for path in args.fold_result]
    if summary["robust_refit_gate_passed"]:
        source_path = Path(summary["fold_artifacts"][0]["artifact"])
        if _sha256(source_path) != summary["fold_artifacts"][0]["artifact_sha256"]:
            raise RuntimeError("materialization source artifact hash mismatch")
        source_artifact = torch.load(
            source_path, map_location="cpu", weights_only=False
        )
        materialized = materialize_common_policy_artifact(
            source_artifact,
            summary,
            source_artifact_path=source_path,
        )
        _atomic_torch_save(materialized, args.selection_artifact)
        summary["selection_artifact"] = str(args.selection_artifact.resolve())
        summary["selection_artifact_sha256"] = _sha256(args.selection_artifact)
    else:
        summary["selection_artifact"] = None
        summary["selection_artifact_sha256"] = None
    _atomic_json_dump(summary, args.output_json)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(_markdown(summary))
    print(
        json.dumps(
            {
                "output_json": str(args.output_json.resolve()),
                "robust_refit_gate_passed": summary[
                    "robust_refit_gate_passed"
                ],
                "robust_policy_count": summary["robust_policy_count"],
                "selected_policy_result": summary["selected_policy_result"],
                "selection_artifact": summary["selection_artifact"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
