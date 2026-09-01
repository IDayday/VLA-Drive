#!/usr/bin/env python3
"""Decompose two deployable policies' selected-candidate target differences."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import torch


FACTOR_KEYS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
    "score",
)


def _sha256(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-csv", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.selection_csv.is_file() or not args.label_root.is_dir():
        raise FileNotFoundError("selection CSV or label root is missing")
    selections: Dict[str, tuple[int, int, float]] = {}
    with args.selection_csv.open(newline="") as stream:
        for row in csv.DictReader(stream):
            token = str(row["token"])
            if token in selections:
                raise RuntimeError(f"duplicate selection token: {token}")
            selections[token] = (
                int(row["base_index"]),
                int(row["selected_index"]),
                float(row["pdms_delta"]),
            )
    accumulators = {
        name: {
            "count": 0,
            "sum": np.zeros(len(FACTOR_KEYS), dtype=np.float64),
            "regression": np.zeros(len(FACTOR_KEYS), dtype=np.int64),
            "improvement": np.zeros(len(FACTOR_KEYS), dtype=np.int64),
        }
        for name in ("win", "loss", "tie")
    }
    seen: set[str] = set()
    chunks = sorted(args.label_root.glob("**/chunk_*.pt"))
    if not chunks:
        raise RuntimeError("label root contains no chunk_*.pt files")
    for path in chunks:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        factors = payload["target_factors"].float().numpy()
        if factors.shape[1:] != (64, 7):
            raise RuntimeError(f"unexpected target shape in {path}: {factors.shape}")
        for index, raw_token in enumerate(payload["tokens"]):
            token = str(raw_token)
            if token not in selections:
                continue
            if token in seen:
                raise RuntimeError(f"duplicate label token: {token}")
            seen.add(token)
            base_index, selected_index, aggregate_delta = selections[token]
            factor_delta = factors[index, selected_index] - factors[index, base_index]
            group = (
                "win"
                if aggregate_delta > 1.0e-9
                else "loss"
                if aggregate_delta < -1.0e-9
                else "tie"
            )
            accumulator = accumulators[group]
            accumulator["count"] += 1
            accumulator["sum"] += factor_delta
            accumulator["regression"] += factor_delta < -1.0e-6
            accumulator["improvement"] += factor_delta > 1.0e-6
    if seen != set(selections):
        missing = sorted(set(selections).difference(seen))
        raise RuntimeError(f"labels missing {len(missing)} selections: {missing[:3]}")

    groups = {}
    for name, accumulator in accumulators.items():
        count = int(accumulator["count"])
        denominator = max(count, 1)
        groups[name] = {
            "scene_count": count,
            "mean_factor_delta": dict(
                zip(FACTOR_KEYS, (accumulator["sum"] / denominator).tolist())
            ),
            "factor_regression_rate": dict(
                zip(
                    FACTOR_KEYS,
                    (accumulator["regression"] / denominator).tolist(),
                )
            ),
            "factor_improvement_rate": dict(
                zip(
                    FACTOR_KEYS,
                    (accumulator["improvement"] / denominator).tolist(),
                )
            ),
        }
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_csv": str(args.selection_csv.resolve()),
        "selection_csv_sha256": _sha256(args.selection_csv),
        "label_root": str(args.label_root.resolve()),
        "scene_count": len(selections),
        "factor_keys": FACTOR_KEYS,
        "groups": groups,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
