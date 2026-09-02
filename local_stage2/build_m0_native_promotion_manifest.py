#!/usr/bin/env python3
"""Build the strict held-out-log promotion set for M0-native scorer runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping

import torch


INDEPENDENT_ARTIFACTS = {
    "direct": (
        "best_independent_scorer.pt",
        "selected_pdms",
        "selected_delta",
        "selected_delta_log_bootstrap_95ci",
    ),
    "coarse": (
        "best_coarse_independent_scorer.pt",
        "coarse_selected_pdms",
        "coarse_selected_delta",
        "coarse_selected_delta_log_bootstrap_95ci",
    ),
    "factor": (
        "best_factor_independent_scorer.pt",
        "factor_selected_pdms",
        "factor_selected_delta",
        "factor_selected_delta_log_bootstrap_95ci",
    ),
}


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _completed_training(run_dir: Path) -> Mapping[str, object]:
    summary_path = run_dir / "training_summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"training summary is not ready: {summary_path}")
    summary = json.loads(summary_path.read_text())
    history = summary.get("history")
    if not isinstance(history, list) or not history:
        raise RuntimeError(f"training history is empty: {summary_path}")
    fold_path = run_dir / "fold_manifest.json"
    if not fold_path.is_file():
        raise RuntimeError(f"fold manifest is not ready: {fold_path}")
    fold = json.loads(fold_path.read_text())
    requested = int(fold["args"]["epochs"])
    if len(history) != requested:
        raise RuntimeError(
            f"training is incomplete for {run_dir}: {len(history)}/{requested}"
        )
    return summary


def _validation_for_selection_source(
    artifact: Mapping[str, object],
) -> Mapping[str, object]:
    source = str(artifact["checkpoint_selection_source"])
    by_source = artifact.get("validation_by_source")
    if isinstance(by_source, Mapping) and source in by_source:
        validation = by_source[source]
    else:
        validation = artifact.get("validation")
    if not isinstance(validation, Mapping):
        raise RuntimeError("artifact lacks held-out validation metrics")
    return validation


def _record(
    *,
    run_dir: Path,
    path: Path,
    architecture: str,
    score_mode: str,
    selected_key: str,
    delta_key: str,
    interval_key: str,
    minimum_ci_lower: float,
) -> Dict[str, object]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if artifact.get("architecture") != architecture:
        raise RuntimeError(f"artifact architecture mismatch: {path}")
    validation = _validation_for_selection_source(artifact)
    interval = validation.get(interval_key)
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        raise RuntimeError(f"artifact lacks {interval_key}: {path}")
    selected = float(validation[selected_key])
    delta = float(validation[delta_key])
    lower, upper = float(interval[0]), float(interval[1])
    promoted = delta > 0.0 and lower > minimum_ci_lower
    run_name = _safe_name(run_dir.name)
    return {
        "name": f"{run_name}__{score_mode}",
        "run_dir": str(run_dir.resolve()),
        "artifact": str(path.resolve()),
        "artifact_sha256": _sha256(path),
        "architecture": architecture,
        "score_mode": score_mode,
        "epoch": int(artifact["epoch"]),
        "validation_scene_count": int(validation["scene_count"]),
        "validation_physical_log_count": int(validation["physical_log_count"]),
        "validation_selected_pdms": selected,
        "validation_base_selected_pdms": float(
            validation["base_selected_pdms"]
        ),
        "validation_delta": delta,
        "validation_delta_log_bootstrap_95ci": [lower, upper],
        "promoted": promoted,
        "promotion_reason": (
            "positive_heldout_log_bootstrap_lower_bound"
            if promoted
            else "heldout_log_bootstrap_lower_bound_not_positive"
        ),
        "future_or_evaluator_input": False,
        "external_model_representation_or_weight_used": False,
    }


def _refit_record(*, run_dir: Path, path: Path) -> Dict[str, object]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if artifact.get("architecture") != "M0PrivateResidualRanker":
        raise RuntimeError(f"refit artifact architecture mismatch: {path}")
    if not bool(artifact.get("refit_all_logs")):
        raise RuntimeError(f"artifact is not an all-log refit: {path}")
    if artifact.get("validation_performed") is not False:
        raise RuntimeError(f"all-log refit unexpectedly reports validation: {path}")
    provenance = artifact.get("refit_provenance")
    if not isinstance(provenance, Mapping):
        raise RuntimeError(f"refit artifact lacks selection provenance: {path}")
    interval = provenance.get("selected_delta_log_bootstrap_95ci")
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        raise RuntimeError(f"refit artifact lacks selection CI: {path}")
    lower, upper = float(interval[0]), float(interval[1])
    if lower <= 0.0:
        raise RuntimeError(f"refit selection did not pass the held-out gate: {path}")
    fold = artifact.get("fold_manifest")
    if not isinstance(fold, Mapping):
        raise RuntimeError(f"refit artifact lacks fold manifest: {path}")
    if fold.get("validation_physical_logs") not in ([], ()):
        raise RuntimeError(f"refit artifact retains validation logs: {path}")
    score_mode = f"residual_{artifact['residual_config']['score_mode']}"
    return {
        "name": f"{_safe_name(run_dir.name)}__{score_mode}",
        "run_dir": str(run_dir.resolve()),
        "artifact": str(path.resolve()),
        "artifact_sha256": _sha256(path),
        "architecture": "M0PrivateResidualRanker",
        "score_mode": score_mode,
        "epoch": int(artifact["epoch"]),
        "validation_scene_count": int(
            provenance["selected_validation_scene_count"]
        ),
        "validation_physical_log_count": int(
            provenance["selected_validation_physical_log_count"]
        ),
        "validation_selected_pdms": float(
            provenance["selected_validation_pdms"]
        ),
        "validation_base_selected_pdms": float(
            provenance["selected_validation_base_pdms"]
        ),
        "validation_delta": float(provenance["selected_validation_delta"]),
        "validation_delta_log_bootstrap_95ci": [lower, upper],
        "promoted": True,
        "promotion_reason": (
            "all_log_refit_locked_by_positive_heldout_log_bootstrap_lower_bound"
        ),
        "selection_artifact_sha256": str(
            provenance["selection_artifact_sha256"]
        ),
        "refit_all_logs": True,
        "validation_performed_after_refit": False,
        "future_or_evaluator_input": False,
        "external_model_representation_or_weight_used": False,
    }


def build_manifest(
    independent_runs: List[Path],
    residual_runs: List[Path],
    refit_residual_runs: List[Path],
    minimum_ci_lower: float,
) -> Dict[str, object]:
    records: List[Dict[str, object]] = []
    for run_dir in independent_runs:
        _completed_training(run_dir)
        for mode, (filename, selected, delta, interval) in INDEPENDENT_ARTIFACTS.items():
            path = run_dir / filename
            if not path.is_file():
                raise RuntimeError(f"missing completed ranker artifact: {path}")
            records.append(
                _record(
                    run_dir=run_dir,
                    path=path,
                    architecture="IndependentProposalRanker",
                    score_mode=mode,
                    selected_key=selected,
                    delta_key=delta,
                    interval_key=interval,
                    minimum_ci_lower=minimum_ci_lower,
                )
            )
    for run_dir in residual_runs:
        _completed_training(run_dir)
        path = run_dir / "best_m0_private_residual_scorer.pt"
        if not path.is_file():
            raise RuntimeError(f"missing completed residual artifact: {path}")
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        residual_mode = str(artifact["residual_config"]["score_mode"])
        records.append(
            _record(
                run_dir=run_dir,
                path=path,
                architecture="M0PrivateResidualRanker",
                score_mode=f"residual_{residual_mode}",
                selected_key="selected_pdms",
                delta_key="selected_delta",
                interval_key="selected_delta_log_bootstrap_95ci",
                minimum_ci_lower=minimum_ci_lower,
            )
        )
    for run_dir in refit_residual_runs:
        _completed_training(run_dir)
        path = run_dir / "refit_m0_private_residual_scorer.pt"
        if not path.is_file():
            raise RuntimeError(f"missing completed refit artifact: {path}")
        records.append(_refit_record(run_dir=run_dir, path=path))
    promoted = [record for record in records if bool(record["promoted"])]
    if len({str(record["artifact_sha256"]) for record in promoted}) != len(promoted):
        raise RuntimeError("promoted artifacts unexpectedly share a SHA256")
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "promotion_rule": "positive held-out physical-log bootstrap lower bound",
        "minimum_ci_lower": minimum_ci_lower,
        "scanned_artifact_count": len(records),
        "promoted_artifact_count": len(promoted),
        "promoted": promoted,
        "excluded": [record for record in records if not bool(record["promoted"])],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--independent-run", action="append", type=Path, default=[])
    parser.add_argument("--residual-run", action="append", type=Path, default=[])
    parser.add_argument(
        "--refit-residual-run", action="append", type=Path, default=[]
    )
    parser.add_argument("--minimum-ci-lower", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.independent_run and not args.residual_run and not args.refit_residual_run:
        raise ValueError("at least one scorer run is required")
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = build_manifest(
        args.independent_run,
        args.residual_run,
        args.refit_residual_run,
        args.minimum_ci_lower,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
