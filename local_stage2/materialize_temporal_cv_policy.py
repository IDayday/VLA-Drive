"""Apply one Navtrain-CV deployment policy to temporal scorer artifacts.

The learned weights are not changed.  Only the four inference-time policy
fields selected jointly across whole-log folds are replaced.  Each derived
artifact keeps the validation result for that exact policy, so promotion and
Navtest evaluation remain auditable and never consult Navtest for tuning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Mapping, Sequence

import torch

from local_stage2.summarize_temporal_consequence_cv import DEPLOYMENT_KEYS
from local_stage2.temporal_consequence_scorer import TemporalConsequenceScorerAgent


POLICY_TO_CONFIG = {
    "residual_scale": "inference_scale",
    "switch_penalty": "switch_penalty",
    "safety_floor": "safety_floor",
    "safety_relative_tolerance": "safety_relative_tolerance",
}


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _policy_values(summary: Mapping[str, object]) -> Dict[str, float]:
    if not bool(summary.get("robust_deployment_available")):
        raise ValueError("CV summary has no robust common deployment")
    deployment = summary["common_deployment"]
    return {key: float(deployment[key]) for key in DEPLOYMENT_KEYS}


def _matching_validation(
    sweep: Sequence[Mapping[str, object]],
    policy: Mapping[str, float],
    *,
    tolerance: float = 1e-12,
) -> Mapping[str, object]:
    matches = [
        item
        for item in sweep
        if all(abs(float(item[key]) - float(policy[key])) <= tolerance for key in DEPLOYMENT_KEYS)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one exact deployment-policy match, found {len(matches)}")
    return matches[0]


def materialize_common_policy_artifact(
    source: Path,
    output: Path,
    cv_summary_path: Path,
) -> Dict[str, object]:
    summary = json.loads(cv_summary_path.read_text())
    fold_audit = summary.get("fold_audit", {})
    if not bool(fold_audit.get("complete")):
        raise ValueError("Refusing to materialize policy from incomplete CV folds")
    policy = _policy_values(summary)
    common_epoch = int(summary["common_epoch"]["epoch"])

    payload = torch.load(source, map_location="cpu")
    if payload.get("artifact_type") != TemporalConsequenceScorerAgent.ARTIFACT_TYPE:
        raise ValueError(f"Not a temporal consequence artifact: {source}")
    config = dict(payload["model_config"])
    for policy_key, value in policy.items():
        config[POLICY_TO_CONFIG[policy_key]] = value

    metadata = dict(payload.get("metadata", {}))
    retained_epoch = metadata.get("retained_epoch")
    if retained_epoch is None or int(retained_epoch) != common_epoch:
        raise ValueError(
            "Artifact weights do not match the CV common epoch: "
            f"artifact={retained_epoch} common={common_epoch}"
        )
    validation = _matching_validation(metadata.get("deployment_sweep", ()), policy)
    if int(validation.get("weight_epoch", -1)) != common_epoch:
        raise ValueError(
            "Matched deployment row does not correspond to the CV common epoch"
        )
    metadata["validation"] = dict(validation)
    metadata["common_cv_policy"] = {
        "summary_path": str(cv_summary_path.resolve()),
        "summary_sha256": _sha256(cv_summary_path),
        "source_artifact_path": str(source.resolve()),
        "source_artifact_sha256": _sha256(source),
        "fold_audit": fold_audit,
        "policy": policy,
        "common_epoch": common_epoch,
        "navtest_used_for_policy_selection": False,
    }
    derived = dict(payload)
    derived["model_config"] = config
    derived["metadata"] = metadata

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    torch.save(derived, temporary)
    os.replace(temporary, output)
    return {
        "source": str(source.resolve()),
        "source_sha256": metadata["common_cv_policy"]["source_artifact_sha256"],
        "output": str(output.resolve()),
        "output_sha256": _sha256(output),
        "validation_pdms_delta": float(validation["selected_pdms_delta"]),
        "policy": policy,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, action="append", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = []
    for source in args.artifact:
        fold_name = source.parent.name
        output = args.output_root / fold_name / "common_policy_temporal_consequence_scorer.pt"
        records.append(
            materialize_common_policy_artifact(source, output, args.cv_summary)
        )
    manifest = {
        "cv_summary": str(args.cv_summary.resolve()),
        "cv_summary_sha256": _sha256(args.cv_summary),
        "artifacts": records,
    }
    manifest_path = args.output_root / "materialization_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
