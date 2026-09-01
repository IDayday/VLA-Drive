#!/usr/bin/env python3
"""Build an artifact-SHA promotion manifest for independent DrivOR scorers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping

import torch


def _sha256(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _ranker_record(
    artifact: Mapping[str, object], path: Path
) -> Dict[str, object]:
    mode = str(artifact["selection_mode"])
    manifest = artifact["training_manifest"]
    assert isinstance(manifest, Mapping)
    source = str(manifest["checkpoint_selection_source"])
    by_source = artifact["validation_by_source"]
    assert isinstance(by_source, Mapping)
    metrics = by_source[source]
    assert isinstance(metrics, Mapping)
    if mode == "factor":
        selected_key = "factor_selected_pdms"
        delta_key = "factor_selected_delta"
        ci_key = "factor_selected_delta_log_bootstrap_95ci"
    elif mode == "direct":
        selected_key = "selected_pdms"
        delta_key = "selected_delta"
        ci_key = "selected_delta_log_bootstrap_95ci"
    else:
        raise RuntimeError(f"unsupported DrivOR ranker selection mode: {mode}")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "architecture": "DrivORInitializedProposalRanker",
        "selection_mode": mode,
        "selection_source": source,
        "epoch": int(artifact["epoch"]),
        "validation_selected_pdms": float(metrics[selected_key]),
        "validation_base_pdms": float(metrics["base_selected_pdms"]),
        "validation_delta": float(metrics[delta_key]),
        "validation_delta_log_bootstrap_95ci": [
            float(value) for value in metrics[ci_key]
        ],
    }


def _gate_record(
    artifact: Mapping[str, object], path: Path
) -> Dict[str, object]:
    manifest = artifact["training_manifest"]
    assert isinstance(manifest, Mapping)
    source = str(manifest["selection_source"])
    by_source = artifact["validation_by_source"]
    assert isinstance(by_source, Mapping)
    metrics = by_source[source]
    assert isinstance(metrics, Mapping)
    policy = artifact["selected_policy"]
    assert isinstance(policy, Mapping)
    if policy != metrics["best_policy"]:
        raise RuntimeError("gate artifact and source-locked best policy differ")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "architecture": "DrivORReferenceGateRanker",
        "selection_mode": str(artifact["alternative_mode"]),
        "alternative_count": int(artifact.get("alternative_count", 1)),
        "selection_source": source,
        "epoch": int(artifact["epoch"]),
        "validation_selected_pdms": float(policy["selected_pdms"]),
        "validation_base_pdms": float(policy["base_selected_pdms"]),
        "validation_delta": float(policy["delta"]),
        "validation_delta_log_bootstrap_95ci": [
            float(value) for value in policy["delta_log_bootstrap_95ci"]
        ],
        "validation_locked_policy": dict(policy),
    }


def _artifact_paths(roots: Iterable[Path]) -> list[Path]:
    paths: set[Path] = set()
    patterns = ("best_factor_ranker.pt", "best_direct_ranker.pt", "best_drivor_reference_gate.pt")
    for root in roots:
        for pattern in patterns:
            paths.update(root.glob(f"**/{pattern}"))
    return sorted(paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-ci-lower", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    for root in args.search_root:
        if not root.is_dir():
            raise FileNotFoundError(root)
    audited = []
    failures = []
    for path in _artifact_paths(args.search_root):
        try:
            artifact = torch.load(path, map_location="cpu", weights_only=False)
            architecture = str(artifact.get("architecture"))
            if architecture == "DrivORInitializedProposalRanker":
                record = _ranker_record(artifact, path)
            elif architecture == "DrivORReferenceGateRanker":
                record = _gate_record(artifact, path)
            else:
                continue
            record["promoted"] = (
                float(record["validation_delta_log_bootstrap_95ci"][0])
                > args.minimum_ci_lower
            )
            audited.append(record)
        except Exception as error:  # preserve a complete campaign audit
            failures.append({"path": str(path.resolve()), "error": repr(error)})
    promoted = [record for record in audited if bool(record["promoted"])]
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "search_roots": [str(root.resolve()) for root in args.search_root],
        "minimum_ci_lower": args.minimum_ci_lower,
        "audited_artifact_count": len(audited),
        "promoted_artifact_count": len(promoted),
        "failed_artifact_count": len(failures),
        "artifacts": audited,
        "promoted_artifacts": promoted,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if failures:
        raise RuntimeError("one or more scorer artifacts failed promotion audit")


if __name__ == "__main__":
    main()
