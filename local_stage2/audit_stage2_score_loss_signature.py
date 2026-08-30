#!/usr/bin/env python3
"""Infer which score heads were optimized in a completed Stage-2 checkpoint.

The paper describes a BCE loss on the aggregate PDM target, while the released
training code applies independent BCE losses to six PDM factors.  The deployed
aggregate gives the driving-direction head zero weight.  Consequently, a
driving-direction head that moved away from its seeded initialization rules
out aggregate-only supervision through the released aggregation function.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from audit_stage2_initialization_fingerprint import (
    DEFAULT_CONFIG,
    DEFAULT_PUBLIC_CHECKPOINT,
    _checkpoint_state,
    _seed_action_decoder,
)


SCORE_HEADS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "time_to_collision_within_bound",
    "ego_progress",
    "driving_direction_compliance",
    "comfort",
)


def _summarize_head(
    checkpoint_state: dict[str, torch.Tensor],
    initial_state: dict[str, torch.Tensor],
    head: str,
) -> dict[str, float | int | bool]:
    prefix = f"scorer.pred_score.{head}."
    keys = sorted(key for key in initial_state if key.startswith(prefix))
    if not keys:
        raise RuntimeError(f"No initialized tensors found for score head {head!r}")

    square_difference = 0.0
    square_initial = 0.0
    element_count = 0
    exact_tensor_count = 0
    max_abs_difference = 0.0
    for key in keys:
        checkpoint_key = f"agent.action_head.{key}"
        if checkpoint_key not in checkpoint_state:
            raise KeyError(f"Checkpoint is missing {checkpoint_key}")
        initial = initial_state[key].float()
        trained = checkpoint_state[checkpoint_key].float()
        difference = trained - initial
        square_difference += difference.square().sum().item()
        square_initial += initial.square().sum().item()
        element_count += difference.numel()
        exact_tensor_count += int(torch.equal(trained, initial))
        max_abs_difference = max(
            max_abs_difference, difference.abs().max().item()
        )

    return {
        "tensor_count": len(keys),
        "exact_tensor_count": exact_tensor_count,
        "all_tensors_exact": exact_tensor_count == len(keys),
        "element_count": element_count,
        "delta_rms": (square_difference / element_count) ** 0.5,
        "relative_l2": (square_difference / square_initial) ** 0.5,
        "max_abs_difference": max_abs_difference,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_PUBLIC_CHECKPOINT
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    initial_state = _seed_action_decoder(args.seed, args.config)
    checkpoint_state = _checkpoint_state(args.checkpoint)
    heads = {
        head: _summarize_head(checkpoint_state, initial_state, head)
        for head in SCORE_HEADS
    }
    ddc_moved = not heads["driving_direction_compliance"][
        "all_tensors_exact"
    ]
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "initialization_seed": args.seed,
        "released_aggregate_ddc_weight": 0.0,
        "score_heads": heads,
        "inference": {
            "ddc_head_moved": ddc_moved,
            "aggregate_only_released_objective_ruled_out": ddc_moved,
            "reason": (
                "The released aggregate has zero DDC weight, yet the DDC "
                "head moved from initialization; it therefore received an "
                "additional factor-level or otherwise private loss."
                if ddc_moved
                else "The DDC head stayed initialized, which is compatible "
                "with aggregate-only supervision but does not prove it."
            ),
        },
    }
    rendered = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
