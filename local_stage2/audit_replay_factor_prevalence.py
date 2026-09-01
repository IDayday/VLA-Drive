#!/usr/bin/env python3
"""Audit train/validation factor prevalence without reading Navtest labels."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from local_stage2.train_independent_scorer import (
    _atomic_json_dump,
    _sha256,
    physical_log_name,
)


def audit(label_root: Path, split_manifest: Path) -> dict[str, object]:
    split = json.loads(split_manifest.read_text())
    log_sets = {
        "train": {str(value) for value in split["train_physical_logs"]},
        "validation": {
            str(value) for value in split["validation_physical_logs"]
        },
    }
    if log_sets["train"].intersection(log_sets["validation"]):
        raise RuntimeError("physical-log split overlaps")
    accumulators = {
        name: {
            "scene_count": 0,
            "candidate_count": 0,
            "sum": None,
            "failure": None,
        }
        for name in log_sets
    }
    seen: set[str] = set()
    target_keys = None
    chunks = sorted(label_root.glob("*_shard_*-of-*/chunk_*.pt"))
    if not chunks:
        raise RuntimeError("no replay label chunks found")
    for path in chunks:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        tokens = [str(value) for value in payload["tokens"]]
        duplicate = seen.intersection(tokens)
        if duplicate:
            raise RuntimeError(f"duplicate replay token: {sorted(duplicate)[:1]}")
        seen.update(tokens)
        keys = tuple(str(value) for value in payload["target_factor_keys"])
        if target_keys is None:
            target_keys = keys
        elif keys != target_keys:
            raise RuntimeError("factor field order changes across chunks")
        factors = payload["target_factors"].double()
        valid = payload["valid_mask"].bool()
        for split_name, logs in log_sets.items():
            selected = torch.tensor(
                [
                    bool(valid[index])
                    and physical_log_name(log_name) in logs
                    for index, log_name in enumerate(payload["log_names"])
                ],
                dtype=torch.bool,
            )
            if not bool(selected.any()):
                continue
            values = factors[selected]
            accumulator = accumulators[split_name]
            total = values.sum(dim=(0, 1))
            failures = torch.stack(
                (
                    values[..., 0] != 1.0,
                    values[..., 1] < 0.5,
                    torch.zeros_like(values[..., 2], dtype=torch.bool),
                    values[..., 3] < 0.5,
                    values[..., 4] < 0.5,
                    values[..., 5] != 1.0,
                    torch.zeros_like(values[..., 6], dtype=torch.bool),
                ),
                dim=-1,
            ).sum(dim=(0, 1))
            accumulator["sum"] = (
                total
                if accumulator["sum"] is None
                else accumulator["sum"] + total
            )
            accumulator["failure"] = (
                failures
                if accumulator["failure"] is None
                else accumulator["failure"] + failures
            )
            accumulator["scene_count"] += int(values.shape[0])
            accumulator["candidate_count"] += int(values.shape[0] * values.shape[1])
    if target_keys is None:
        raise RuntimeError("factor keys were not loaded")
    result_splits = {}
    for split_name, accumulator in accumulators.items():
        count = int(accumulator["candidate_count"])
        if count <= 0:
            raise RuntimeError(f"split has no candidates: {split_name}")
        result_splits[split_name] = {
            "scene_count": int(accumulator["scene_count"]),
            "candidate_count": count,
            "factor_mean": {
                key: float(accumulator["sum"][index] / count)
                for index, key in enumerate(target_keys)
            },
            "failure_rate": {
                key: float(accumulator["failure"][index] / count)
                for index, key in enumerate(target_keys)
                if key not in {"ego_progress", "score"}
            },
        }
    manifests = sorted(label_root.glob("*_shard_*-of-*/manifest.json"))
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_scope": "trainval replay only; no Navtest/private-test labels",
        "label_root": str(label_root.resolve()),
        "split_manifest": str(split_manifest.resolve()),
        "split_manifest_sha256": _sha256(split_manifest),
        "chunk_count": len(chunks),
        "unique_scene_count": len(seen),
        "target_factor_keys": list(target_keys),
        "manifest_sha256": {
            str(path.relative_to(label_root)): _sha256(path) for path in manifests
        },
        "splits": result_splits,
        "predeclared_safety_negative_weight": 5.0,
        "weight_selection_basis": (
            "rounded conservative cap below inverse-frequency weights for "
            "2-9% train safety-failure prevalence; selected without Navtest"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.label_root, args.split_manifest):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.label_root, args.split_manifest)
    _atomic_json_dump(result, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
