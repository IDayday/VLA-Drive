#!/usr/bin/env python3
"""Select the best distribution-matched CLOVER generator/scorer pair."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from starVLA.candidate_bank.reader import read_candidate_bank_manifest
from starVLA.candidate_bank.schema import manifest_hash
from starVLA.model.modules.register_planner.checkpoint import sha256_file


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        nargs=4,
        metavar=("LABEL", "GENERATOR", "SCORER", "TRAIN_BANK"),
        required=True,
        help="A generator and scorer trained on that generator's train bank.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def select_best_pair(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return the highest true validation-PDMS record with stable tie-breaking."""

    if not records:
        raise ValueError("checkpoint-pair selection requires at least one record")
    for record in records:
        value = float(record["selected_true_pdms"])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("selected validation PDMS must be finite and in [0,1]")
    return max(records, key=lambda record: float(record["selected_true_pdms"]))


def _candidate_record(raw: Sequence[str]) -> dict[str, Any]:
    label, generator_raw, scorer_raw, bank_raw = raw
    generator = Path(generator_raw).expanduser().resolve()
    scorer = Path(scorer_raw).expanduser().resolve()
    bank_root = Path(bank_raw).expanduser().resolve()
    for path in (generator, scorer, bank_root / "manifest.json"):
        if not path.exists():
            raise FileNotFoundError(path)

    manifest = read_candidate_bank_manifest(bank_root)
    payload = torch.load(scorer, map_location="cpu", weights_only=False)
    metadata = dict(payload.get("metadata", {}))
    if metadata.get("schema_version") != 1 or metadata.get("stage") != "drivor_scorer":
        raise RuntimeError(f"{label}: scorer is not a Register64 DrivoR checkpoint")
    if metadata.get("generator_checkpoint_sha256") != manifest.generator_checkpoint_sha256:
        raise RuntimeError(f"{label}: scorer and bank generator identities differ")
    if metadata.get("candidate_bank_manifest_hash") != manifest_hash(manifest):
        raise RuntimeError(f"{label}: scorer was not fitted on the supplied bank")
    if metadata.get("label_protocol") != "navsim_v1_1_pdms_two_way":
        raise RuntimeError(f"{label}: scorer does not use exact NAVSIM-v1.1 labels")
    validation = metadata.get("validation")
    if not isinstance(validation, dict) or "selected_true_score" not in validation:
        raise RuntimeError(f"{label}: scorer checkpoint has no validation selection score")
    selected_true_pdms = float(validation["selected_true_score"])
    if not math.isfinite(selected_true_pdms) or not 0.0 <= selected_true_pdms <= 1.0:
        raise RuntimeError(f"{label}: invalid validation selected PDMS")
    return {
        "label": str(label),
        "generator_checkpoint": str(generator),
        "scorer_checkpoint": str(scorer),
        "train_bank_root": str(bank_root),
        "generator_checkpoint_sha256": manifest.generator_checkpoint_sha256,
        "candidate_bank_manifest_hash": manifest_hash(manifest),
        "selected_true_pdms": selected_true_pdms,
        "oracle_true_pdms": float(validation["oracle_true_score"]),
        "regret": float(validation["regret"]),
        "selector_alpha": float(metadata.get("selection_alpha", 0.0)),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    args = _parse_args()
    records = [_candidate_record(raw) for raw in args.candidate]
    selected = dict(select_best_pair(records))
    actual_generator_sha = sha256_file(selected["generator_checkpoint"])
    if actual_generator_sha != selected["generator_checkpoint_sha256"]:
        raise RuntimeError("selected generator file does not match its bank identity")
    selected["generator_checkpoint_sha256"] = actual_generator_sha
    selected["scorer_checkpoint_sha256"] = sha256_file(
        selected["scorer_checkpoint"]
    )
    result = {
        "schema_version": 1,
        "selection_metric": "validation_selected_true_pdms",
        "num_distribution_matched_pairs": len(records),
        "selected": selected,
        "candidates": records,
    }
    _atomic_json(Path(args.output).expanduser().resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
