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
        nargs=5,
        metavar=(
            "LABEL",
            "GENERATOR",
            "SCORER",
            "TRAIN_BANK",
            "SELECTION_BANK",
        ),
        required=True,
        help=(
            "A matched generator/scorer, its train bank, and its untouched "
            "log-disjoint selection bank."
        ),
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def select_best_pair(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return a conservative holdout winner, using paired scenes when present."""

    if not records:
        raise ValueError("checkpoint-pair selection requires at least one record")
    for record in records:
        value = float(record["selected_true_pdms"])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("selected validation PDMS must be finite and in [0,1]")
        lower_bound = float(record.get("selected_true_pdms_lcb95", value))
        if not math.isfinite(lower_bound) or not 0.0 <= lower_bound <= value:
            raise ValueError("selected PDMS confidence bound is invalid")
    if all(isinstance(record.get("_scene_scores"), dict) for record in records):
        incumbent = records[0]
        incumbent["paired_selection"] = {
            "incumbent_label": None,
            "improvement_mean": 0.0,
            "improvement_lcb95": 0.0,
            "accepted": True,
        }
        reference_tokens = set(incumbent["_scene_scores"])
        for candidate in records[1:]:
            if set(candidate["_scene_scores"]) != reference_tokens:
                raise ValueError(
                    "paired checkpoint selection requires identical scene tokens"
                )
            differences = torch.tensor(
                [
                    candidate["_scene_scores"][token]
                    - incumbent["_scene_scores"][token]
                    for token in sorted(reference_tokens)
                ],
                dtype=torch.float64,
            )
            mean = float(differences.mean())
            stderr = (
                float(differences.std(unbiased=True) / math.sqrt(len(differences)))
                if len(differences) > 1
                else float("inf")
            )
            lower_bound = mean - 1.96 * stderr
            accepted = bool(lower_bound > 0.0)
            candidate["paired_selection"] = {
                "incumbent_label": str(incumbent["label"]),
                "improvement_mean": mean,
                "improvement_stderr": stderr,
                "improvement_lcb95": lower_bound,
                "accepted": accepted,
            }
            if accepted:
                incumbent = candidate
        return incumbent

    return max(
        records,
        key=lambda record: (
            float(
                record.get(
                    "selected_true_pdms_lcb95", record["selected_true_pdms"]
                )
            ),
            float(record["selected_true_pdms"]),
        ),
    )


def _candidate_record(raw: Sequence[str]) -> dict[str, Any]:
    label, generator_raw, scorer_raw, bank_raw, selection_bank_raw = raw
    generator = Path(generator_raw).expanduser().resolve()
    scorer = Path(scorer_raw).expanduser().resolve()
    bank_root = Path(bank_raw).expanduser().resolve()
    selection_bank_root = Path(selection_bank_raw).expanduser().resolve()
    for path in (
        generator,
        scorer,
        bank_root / "manifest.json",
        selection_bank_root / "manifest.json",
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    manifest = read_candidate_bank_manifest(bank_root)
    selection_manifest = read_candidate_bank_manifest(selection_bank_root)
    if manifest.split != "train" or selection_manifest.split != "selection":
        raise RuntimeError(
            f"{label}: supplied bank roles must be train and selection"
        )
    payload = torch.load(scorer, map_location="cpu", weights_only=False)
    metadata = dict(payload.get("metadata", {}))
    if metadata.get("schema_version") != 1 or metadata.get("stage") != "drivor_scorer":
        raise RuntimeError(f"{label}: scorer is not a Register64 DrivoR checkpoint")
    if metadata.get("generator_checkpoint_sha256") != manifest.generator_checkpoint_sha256:
        raise RuntimeError(f"{label}: scorer and bank generator identities differ")
    if metadata.get("candidate_bank_manifest_hash") != manifest_hash(manifest):
        raise RuntimeError(f"{label}: scorer was not fitted on the supplied bank")
    if (
        selection_manifest.generator_checkpoint_sha256
        != manifest.generator_checkpoint_sha256
    ):
        raise RuntimeError(f"{label}: train and selection banks use different generators")
    if metadata.get("selection_bank_manifest_hash") != manifest_hash(
        selection_manifest
    ):
        raise RuntimeError(
            f"{label}: scorer was not evaluated on the supplied selection bank"
        )
    if metadata.get("label_protocol") != "navsim_v1_1_pdms_two_way":
        raise RuntimeError(f"{label}: scorer does not use exact NAVSIM-v1.1 labels")
    validation = metadata.get("selection_validation")
    required = {
        "selected_true_score",
        "selected_true_score_lcb95",
        "selected_true_score_stderr",
        "num_scenes",
        "scene_scores",
    }
    if not isinstance(validation, dict) or not required.issubset(validation):
        raise RuntimeError(
            f"{label}: scorer checkpoint has no untouched selection statistics"
        )
    selected_true_pdms = float(validation["selected_true_score"])
    selected_true_pdms_lcb95 = float(validation["selected_true_score_lcb95"])
    if not math.isfinite(selected_true_pdms) or not 0.0 <= selected_true_pdms <= 1.0:
        raise RuntimeError(f"{label}: invalid validation selected PDMS")
    if (
        not math.isfinite(selected_true_pdms_lcb95)
        or not 0.0 <= selected_true_pdms_lcb95 <= selected_true_pdms
    ):
        raise RuntimeError(f"{label}: invalid selection PDMS confidence bound")
    raw_scene_scores = validation["scene_scores"]
    if not isinstance(raw_scene_scores, list):
        raise RuntimeError(f"{label}: selection scene scores must be a list")
    scene_scores = {
        str(item["token"]): float(item["selected_true_score"])
        for item in raw_scene_scores
    }
    expected_tokens = {record.token for record in selection_manifest.records}
    if set(scene_scores) != expected_tokens or len(scene_scores) != len(raw_scene_scores):
        raise RuntimeError(f"{label}: selection scene-score tokens differ from bank")
    if int(validation["num_scenes"]) != len(scene_scores):
        raise RuntimeError(f"{label}: selection scene count differs from bank")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in scene_scores.values()):
        raise RuntimeError(f"{label}: selection scene scores are invalid")
    scene_mean = sum(scene_scores.values()) / max(len(scene_scores), 1)
    if not math.isclose(
        scene_mean, selected_true_pdms, rel_tol=0.0, abs_tol=1.0e-6
    ):
        raise RuntimeError(
            f"{label}: selection scene-score mean differs from checkpoint summary"
        )
    return {
        "label": str(label),
        "generator_checkpoint": str(generator),
        "scorer_checkpoint": str(scorer),
        "train_bank_root": str(bank_root),
        "selection_bank_root": str(selection_bank_root),
        "generator_checkpoint_sha256": manifest.generator_checkpoint_sha256,
        "candidate_bank_manifest_hash": manifest_hash(manifest),
        "selected_true_pdms": selected_true_pdms,
        "selected_true_pdms_lcb95": selected_true_pdms_lcb95,
        "selected_true_pdms_stderr": float(
            validation["selected_true_score_stderr"]
        ),
        "selection_scenes": int(validation["num_scenes"]),
        "oracle_true_pdms": float(validation["oracle_true_score"]),
        "regret": float(validation["regret"]),
        "selector_alpha": float(metadata.get("selection_alpha", 0.0)),
        "_scene_scores": scene_scores,
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
    selected.pop("_scene_scores", None)
    actual_generator_sha = sha256_file(selected["generator_checkpoint"])
    if actual_generator_sha != selected["generator_checkpoint_sha256"]:
        raise RuntimeError("selected generator file does not match its bank identity")
    selected["generator_checkpoint_sha256"] = actual_generator_sha
    selected["scorer_checkpoint_sha256"] = sha256_file(
        selected["scorer_checkpoint"]
    )
    result = {
        "schema_version": 1,
        "selection_metric": "paired_log_disjoint_pdms_improvement_lcb95",
        "num_distribution_matched_pairs": len(records),
        "selected": selected,
        "candidates": [
            {key: value for key, value in record.items() if key != "_scene_scores"}
            for record in records
        ],
    }
    _atomic_json(Path(args.output).expanduser().resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
