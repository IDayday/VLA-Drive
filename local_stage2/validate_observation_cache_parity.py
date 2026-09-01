#!/usr/bin/env python3
"""Stream two current-observation caches and verify tensor parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch


FLOAT_FIELDS = (
    "visual_tokens",
    "status_feature",
    "history_trajectory",
    "high_command_one_hot",
)
EXACT_FIELDS = ("visual_valid_mask",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--expected-scenes", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _chunk_map(root: Path) -> Dict[Path, Path]:
    paths = sorted(root.glob("*/chunk_*.pt"))
    if not paths:
        raise RuntimeError(f"No observation chunks in {root}")
    return {path.relative_to(root): path for path in paths}


def main() -> None:
    args = parse_args()
    if args.tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    reference = _chunk_map(args.reference_root)
    candidate = _chunk_map(args.candidate_root)
    if set(reference) != set(candidate):
        raise RuntimeError("Observation caches have different chunk inventories")

    scene_count = 0
    maxima = {field: 0.0 for field in FLOAT_FIELDS}
    for relative in sorted(reference):
        left = torch.load(reference[relative], map_location="cpu", weights_only=False)
        right = torch.load(candidate[relative], map_location="cpu", weights_only=False)
        if list(left["tokens"]) != list(right["tokens"]):
            raise RuntimeError(f"Token order differs in {relative}")
        if list(left["log_names"]) != list(right["log_names"]):
            raise RuntimeError(f"Log order differs in {relative}")
        scene_count += len(left["tokens"])
        for field in FLOAT_FIELDS:
            if left[field].shape != right[field].shape:
                raise RuntimeError(f"{field} shape differs in {relative}")
            if left[field].numel():
                maxima[field] = max(
                    maxima[field],
                    float((left[field].float() - right[field].float()).abs().max()),
                )
        for field in EXACT_FIELDS:
            if not torch.equal(left[field], right[field]):
                raise RuntimeError(f"{field} differs in {relative}")
    if args.expected_scenes and scene_count != args.expected_scenes:
        raise RuntimeError(
            f"Expected {args.expected_scenes} scenes, compared {scene_count}"
        )
    failed = {field: value for field, value in maxima.items() if value > args.tolerance}
    if failed:
        raise RuntimeError(f"Observation cache parity exceeds tolerance: {failed}")

    payload = {
        "reference_root": str(args.reference_root.resolve()),
        "candidate_root": str(args.candidate_root.resolve()),
        "scene_count": scene_count,
        "chunk_count": len(reference),
        "tolerance": args.tolerance,
        "max_abs_error": maxima,
        "exact_mask_parity": True,
        "token_and_log_order_parity": True,
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
