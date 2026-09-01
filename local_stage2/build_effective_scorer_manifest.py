"""Build an auditable Navtest promotion manifest from scorer artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping

import torch

from local_stage2.public_base_residual_scorer import PublicBaseResidualScorerAgent
from local_stage2.temporal_consequence_scorer import TemporalConsequenceScorerAgent


SAFETY_FACTORS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "time_to_collision_within_bound",
)


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _artifact_record(
    path: Path,
    minimum_ci_lower: float,
    promotion_rule: str = "positive_mean",
) -> Dict[str, object]:
    payload = torch.load(path, map_location="cpu")
    supported_types = {
        PublicBaseResidualScorerAgent.ARTIFACT_TYPE,
        TemporalConsequenceScorerAgent.ARTIFACT_TYPE,
    }
    if payload.get("artifact_type") not in supported_types:
        return {"path": str(path.resolve()), "promoted": False, "reason": "wrong_artifact_type"}
    metadata: Mapping[str, object] = payload.get("metadata", {})
    validation: Mapping[str, object] = metadata.get("validation", {})
    delta = validation.get("selected_pdms_delta")
    interval = validation.get("selected_pdms_delta_log_bootstrap_95ci")
    if delta is None or not isinstance(interval, (list, tuple)) or len(interval) != 2:
        return {"path": str(path.resolve()), "promoted": False, "reason": "missing_validation_ci"}
    future_inputs_used = bool(metadata.get("future_inputs_used", False))
    official_scores_at_inference = bool(
        metadata.get("official_scores_used_at_inference", False)
    )
    if promotion_rule == "positive_mean":
        positive = float(delta) > 0.0
    elif promotion_rule == "positive_ci":
        positive = float(delta) > 0.0 and float(interval[0]) > minimum_ci_lower
    else:
        raise ValueError(f"Unknown promotion rule: {promotion_rule}")
    deployable_inputs = not future_inputs_used and not official_scores_at_inference
    factor_delta = validation.get("selected_factor_delta", {})
    safety_values = {
        key: float(factor_delta[key]) if key in factor_delta else None
        for key in SAFETY_FACTORS
    }
    safety_non_regressing = all(
        value is not None and value >= -1e-8 for value in safety_values.values()
    )
    reasons: List[str] = []
    if not positive:
        reasons.append(
            "non_positive_validation_mean"
            if promotion_rule == "positive_mean"
            else "non_positive_log_bootstrap_lower_bound"
        )
    if future_inputs_used:
        reasons.append("future_input_used")
    if official_scores_at_inference:
        reasons.append("official_score_used_at_inference")
    return {
        "name": _safe_name(path.parent.name),
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "promoted": bool(positive and deployable_inputs),
        "tier": "deployable" if safety_non_regressing else "diagnostic_safety_regression",
        "reason": ",".join(reasons) if reasons else promotion_rule,
        "promotion_rule": promotion_rule,
        "validation_scene_count": metadata.get("val_scene_count"),
        "validation_log_count": metadata.get("val_log_count"),
        "validation_pdms_delta": float(delta),
        "validation_pdms_delta_ci95": [float(interval[0]), float(interval[1])],
        "validation_safety_factor_delta": safety_values,
        "future_inputs_used": future_inputs_used,
        "official_scores_used_at_inference": official_scores_at_inference,
        "model_config": payload.get("model_config"),
        "artifact_type": payload.get("artifact_type"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-root", type=Path, required=True)
    parser.add_argument("--pattern", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-ci-lower", type=float, default=0.0)
    parser.add_argument(
        "--promotion-rule",
        choices=("positive_mean", "positive_ci"),
        default="positive_mean",
        help=(
            "positive_mean sends every validation-improving method to Navtest; "
            "positive_ci is the stricter legacy gate"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(
        {
            path.resolve()
            for pattern in args.pattern
            for path in args.search_root.glob(pattern)
            if path.is_file()
        }
    )
    records = [
        _artifact_record(path, args.minimum_ci_lower, args.promotion_rule)
        for path in paths
    ]
    promoted = [record for record in records if record.get("promoted")]

    # Exact duplicate artifacts are evaluated once; all experiment aliases are
    # retained so the final report remains complete.
    unique: Dict[str, Dict[str, object]] = {}
    for record in promoted:
        digest = str(record["sha256"])
        if digest not in unique:
            unique[digest] = dict(record, aliases=[record["name"]])
        else:
            unique[digest]["aliases"].append(record["name"])
    artifacts = sorted(unique.values(), key=lambda value: str(value["name"]))
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "search_root": str(args.search_root.resolve()),
        "patterns": list(args.pattern),
        "minimum_ci_lower": args.minimum_ci_lower,
        "promotion_rule": args.promotion_rule,
        "scanned_artifact_count": len(records),
        "promoted_experiment_count": len(promoted),
        "unique_promoted_artifact_count": len(artifacts),
        "artifacts": artifacts,
        "excluded": [record for record in records if not record.get("promoted")],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
