#!/usr/bin/env python3
"""Audit a fixed-epoch M0 scorer across disjoint physical-log folds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_torch(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def aggregate_fixed_epoch_folds(
    fold_rows: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    if len(fold_rows) < 2:
        raise ValueError("at least two folds are required")
    scenes = np.asarray([int(row["scene_count"]) for row in fold_rows], dtype=np.float64)
    deltas = np.asarray([float(row["selected_delta"]) for row in fold_rows])
    lowers = np.asarray([float(row["bootstrap_95ci"][0]) for row in fold_rows])
    uppers = np.asarray([float(row["bootstrap_95ci"][1]) for row in fold_rows])
    if np.any(scenes <= 0):
        raise ValueError("every fold must contain validation scenes")
    weights = scenes / scenes.sum()
    gate = bool(np.all(deltas > 0.0) and np.all(lowers > 0.0))
    return {
        "fold_count": len(fold_rows),
        "validation_scene_count": int(scenes.sum()),
        "scene_weighted_selected_delta": float(np.sum(weights * deltas)),
        "mean_selected_delta": float(deltas.mean()),
        "std_selected_delta": float(deltas.std(ddof=0)),
        "worst_fold_selected_delta": float(deltas.min()),
        "worst_fold_bootstrap_95ci_lower": float(lowers.min()),
        "best_fold_bootstrap_95ci_upper": float(uppers.max()),
        "all_fold_point_deltas_positive": bool(np.all(deltas > 0.0)),
        "all_fold_bootstrap_lowers_positive": bool(np.all(lowers > 0.0)),
        "robust_refit_gate_passed": gate,
    }


def _normalized_locked_args(payload: Mapping[str, object]) -> Dict[str, object]:
    args = dict(payload)
    for key in ("split_manifest", "output_dir"):
        args.pop(key, None)
    return args


def _markdown(payload: Mapping[str, object]) -> str:
    aggregate = payload["aggregate"]
    lines = [
        "# No-VQA E35 risk-stratified five-fold scorer audit",
        "",
        "This audit uses the predeclared final epoch across disjoint Navtrain physical logs. Navtest is not read.",
        "",
        f"- Locked epoch: `{payload['locked_epoch']}`",
        f"- Validation coverage: `{aggregate['validation_scene_count']}` scenes / `{payload['validation_physical_log_count']}` logs",
        f"- Scene-weighted delta: `{aggregate['scene_weighted_selected_delta']:+.8f}`",
        f"- Worst-fold delta: `{aggregate['worst_fold_selected_delta']:+.8f}`",
        f"- Worst-fold bootstrap lower: `{aggregate['worst_fold_bootstrap_95ci_lower']:+.8f}`",
        f"- Robust all-log refit gate: `{'PASS' if aggregate['robust_refit_gate_passed'] else 'FAIL'}`",
        "",
        "| Fold | Scenes | Logs | Base PDMS | Selected PDMS | Delta | 95% CI |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["folds"]:
        interval = row["bootstrap_95ci"]
        lines.append(
            f"| {row['fold_index']} | {row['scene_count']} | {row['physical_log_count']} | "
            f"{row['base_selected_pdms']:.6f} | {row['selected_pdms']:.6f} | "
            f"{row['selected_delta']:+.6f} | [{interval[0]:+.6f}, {interval[1]:+.6f}] |"
        )
    lines.extend(
        [
            "",
            "The five validation-log sets are pairwise disjoint and cover every available Navtrain physical log exactly once. The ordinary per-fold best epochs are not used by the gate.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--locked-epoch", type=int, default=7)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--selection-artifact", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.run_root, args.fold_root, args.fold_root / "index.json"):
        if not path.exists():
            raise FileNotFoundError(path)
    for output in (args.output_json, args.output_md, args.selection_artifact):
        if output.exists():
            raise FileExistsError(output)
    fold_index = json.loads((args.fold_root / "index.json").read_text())
    expected_folds = int(fold_index["num_folds"])
    if expected_folds != 5:
        raise RuntimeError("Wave-12 requires exactly five folds")
    validation_sets: List[set[str]] = []
    fold_rows: List[Dict[str, object]] = []
    locked_args = None
    selection_payload = None
    for fold in range(expected_folds):
        split_path = args.fold_root / f"fold_{fold}.json"
        summary_path = args.run_root / f"fold_{fold}" / "training_summary.json"
        if not split_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(split_path if not split_path.is_file() else summary_path)
        split = json.loads(split_path.read_text())
        summary = json.loads(summary_path.read_text())
        history = summary.get("history", [])
        if len(history) != args.locked_epoch + 1:
            raise RuntimeError(f"fold {fold} is incomplete: {len(history)} epochs")
        record = history[args.locked_epoch]
        if int(record["epoch"]) != args.locked_epoch:
            raise RuntimeError(f"fold {fold} history does not contain locked epoch")
        metrics = record["validation"]
        last_artifact = Path(summary["last_artifact"])
        if not last_artifact.is_file():
            raise FileNotFoundError(last_artifact)
        if _sha256(last_artifact) != summary["last_artifact_sha256"]:
            raise RuntimeError(f"fold {fold} last-artifact hash mismatch")
        artifact = torch.load(last_artifact, map_location="cpu", weights_only=False)
        if int(artifact.get("epoch", -1)) != args.locked_epoch:
            raise RuntimeError(f"fold {fold} last artifact has wrong epoch")
        manifest = artifact.get("fold_manifest")
        if not isinstance(manifest, Mapping):
            raise RuntimeError(f"fold {fold} artifact lacks fold manifest")
        manifest_args = _normalized_locked_args(manifest["args"])
        if locked_args is None:
            locked_args = manifest_args
            selection_payload = artifact
        elif manifest_args != locked_args:
            raise RuntimeError("fold training arguments differ beyond split/output path")
        validation_logs = set(str(value) for value in split["validation_physical_logs"])
        artifact_validation_logs = set(
            str(value) for value in manifest["validation_physical_logs"]
        )
        if validation_logs != artifact_validation_logs:
            raise RuntimeError(f"fold {fold} validation lineage mismatch")
        validation_sets.append(validation_logs)
        factor_delta = {
            key: float(metrics["selected_factors"][key])
            - float(metrics["base_selected_factors"][key])
            for key in metrics["selected_factors"]
        }
        fold_rows.append(
            {
                "fold_index": fold,
                "split_manifest": str(split_path.resolve()),
                "split_manifest_sha256": _sha256(split_path),
                "training_summary": str(summary_path.resolve()),
                "training_summary_sha256": _sha256(summary_path),
                "last_artifact": str(last_artifact.resolve()),
                "last_artifact_sha256": _sha256(last_artifact),
                "scene_count": int(metrics["scene_count"]),
                "physical_log_count": int(metrics["physical_log_count"]),
                "base_selected_pdms": float(metrics["base_selected_pdms"]),
                "selected_pdms": float(metrics["selected_pdms"]),
                "selected_delta": float(metrics["selected_delta"]),
                "bootstrap_95ci": [
                    float(value)
                    for value in metrics["selected_delta_log_bootstrap_95ci"]
                ],
                "factor_delta": factor_delta,
                "ordinary_best_epoch_diagnostic_only": int(summary["best_epoch"]),
            }
        )
    union = set().union(*validation_sets)
    for left in range(expected_folds):
        for right in range(left + 1, expected_folds):
            if validation_sets[left].intersection(validation_sets[right]):
                raise RuntimeError("cross-validation physical-log overlap")
    expected_logs = set(fold_index["per_log_stats"])
    if union != expected_logs:
        raise RuntimeError("cross-validation folds do not cover all physical logs")
    aggregate = aggregate_fixed_epoch_folds(fold_rows)
    evidence_core = {
        "locked_epoch": args.locked_epoch,
        "folds": fold_rows,
        "aggregate": aggregate,
    }
    evidence_digest = hashlib.sha256(
        json.dumps(evidence_core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    payload: Dict[str, object] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "strategy": "fixed_epoch_risk_stratified_disjoint_physical_log_fivefold",
        "locked_epoch": args.locked_epoch,
        "fold_index_manifest": str((args.fold_root / "index.json").resolve()),
        "fold_index_manifest_sha256": _sha256(args.fold_root / "index.json"),
        "folds": fold_rows,
        "aggregate": aggregate,
        "validation_physical_log_count": len(union),
        "validation_logs_cover_all_available_logs_once": True,
        "training_arguments_identical_across_folds": True,
        "navtest_used_for_selection": False,
        "evidence_digest": evidence_digest,
    }
    if aggregate["robust_refit_gate_passed"]:
        assert selection_payload is not None
        materialized = dict(selection_payload)
        materialized["cross_validation_selection"] = {
            "schema_version": 1,
            "strategy": payload["strategy"],
            "locked_epoch": args.locked_epoch,
            "fold_count": expected_folds,
            "validation_scene_count": aggregate["validation_scene_count"],
            "validation_physical_log_count": len(union),
            "scene_weighted_selected_delta": aggregate[
                "scene_weighted_selected_delta"
            ],
            "worst_fold_selected_delta": aggregate["worst_fold_selected_delta"],
            "worst_fold_bootstrap_95ci_lower": aggregate[
                "worst_fold_bootstrap_95ci_lower"
            ],
            "robust_refit_gate_passed": True,
            "navtest_used_for_selection": False,
            "evidence_digest": evidence_digest,
        }
        _atomic_torch(args.selection_artifact, materialized)
        payload["selection_artifact"] = str(args.selection_artifact.resolve())
        payload["selection_artifact_sha256"] = _sha256(args.selection_artifact)
    else:
        payload["selection_artifact"] = None
        payload["selection_artifact_sha256"] = None
    _atomic_json(args.output_json, payload)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(_markdown(payload))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
