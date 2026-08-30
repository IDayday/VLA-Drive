#!/usr/bin/env python3
"""Compare artifacts emitted by ``audit_stage2_numerics.py``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def difference(left: torch.Tensor, right: torch.Tensor):
    delta = (left.float() - right.float()).abs()
    scale = left.float().abs().clamp_min(1e-8)
    return {
        "equal": bool(torch.equal(left, right)),
        "max_abs": float(delta.max()) if delta.numel() else 0.0,
        "mean_abs": float(delta.mean()) if delta.numel() else 0.0,
        "rmse": float(delta.square().mean().sqrt()) if delta.numel() else 0.0,
        "max_relative": float((delta / scale).max()) if delta.numel() else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--path", choices=("raw", "optimized"), default="raw")
    args = parser.parse_args()
    if len(args.artifacts) < 2:
        raise SystemExit("Provide at least two artifacts")

    loaded = [torch.load(path, map_location="cpu", weights_only=False) for path in args.artifacts]
    reference = loaded[0]
    report = {
        "reference": reference["name"],
        "path": args.path,
        "comparisons": [],
    }
    for candidate in loaded[1:]:
        item = {
            "candidate": candidate["name"],
            "same_initial_action": (
                candidate["initial_action_sha256"]
                == reference["initial_action_sha256"]
            ),
            "loss_abs": abs(
                reference[args.path]["loss"] - candidate[args.path]["loss"]
            ),
            "tensors": {
                key: difference(
                    reference[args.path][key], candidate[args.path][key]
                )
                for key in (
                    "last_hidden_state",
                    "proposals",
                    "pdm_score",
                    "gradients",
                    "parameter_delta",
                )
            },
        }
        report["comparisons"].append(item)

    output = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
