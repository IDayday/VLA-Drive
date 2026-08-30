#!/usr/bin/env python3
"""Compare Stage-2 action-head update directions from a shared initialization.

The released checkpoint exposes an exact initialization fingerprint for seed 2
through its unused trajectory heads.  For checkpoints produced from that same
seed, this tool compares the *trained* parameter displacement vectors rather
than only their final norms.  A high cosine similarity is evidence that two
training runs follow the same optimization objective even when their step
counts or learning-rate schedules differ.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from audit_stage2_initialization_fingerprint import (
    _checkpoint_state,
    _fingerprint_keys,
    _seed_action_decoder,
    DEFAULT_CONFIG,
)


def _accumulator() -> dict[str, float | int]:
    return {
        "element_count": 0,
        "left_square": 0.0,
        "right_square": 0.0,
        "dot": 0.0,
        "difference_square": 0.0,
        "same_sign_count": 0,
        "nonzero_union_count": 0,
    }


def _add(
    accumulator: dict[str, float | int],
    left_update: torch.Tensor,
    right_update: torch.Tensor,
) -> None:
    left = left_update.float().reshape(-1)
    right = right_update.float().reshape(-1)
    accumulator["element_count"] += left.numel()
    accumulator["left_square"] += left.square().sum().item()
    accumulator["right_square"] += right.square().sum().item()
    accumulator["dot"] += (left * right).sum().item()
    accumulator["difference_square"] += (left - right).square().sum().item()
    nonzero_union = (left != 0) | (right != 0)
    accumulator["nonzero_union_count"] += nonzero_union.sum().item()
    accumulator["same_sign_count"] += (
        ((torch.sign(left) == torch.sign(right)) & nonzero_union).sum().item()
    )


def _finalize(accumulator: dict[str, float | int]) -> dict[str, float | int | None]:
    count = int(accumulator["element_count"])
    left_square = float(accumulator["left_square"])
    right_square = float(accumulator["right_square"])
    dot = float(accumulator["dot"])
    nonzero_union_count = int(accumulator["nonzero_union_count"])
    denominator = (left_square * right_square) ** 0.5
    optimal_right_scale = dot / right_square if right_square else None
    scaled_residual_square = (
        left_square - 2.0 * optimal_right_scale * dot
        + optimal_right_scale * optimal_right_scale * right_square
        if optimal_right_scale is not None
        else None
    )
    return {
        "element_count": count,
        "left_update_rms": (left_square / count) ** 0.5 if count else None,
        "right_update_rms": (right_square / count) ** 0.5 if count else None,
        "left_to_right_norm_ratio": (
            (left_square / right_square) ** 0.5 if right_square else None
        ),
        "cosine_similarity": dot / denominator if denominator else None,
        "raw_difference_rms": (
            (float(accumulator["difference_square"]) / count) ** 0.5
            if count
            else None
        ),
        "optimal_right_scale_to_left": optimal_right_scale,
        "scaled_residual_relative_l2": (
            (max(0.0, scaled_residual_square) / left_square) ** 0.5
            if scaled_residual_square is not None and left_square
            else None
        ),
        "same_sign_fraction_nonzero_union": (
            int(accumulator["same_sign_count"]) / nonzero_union_count
            if nonzero_union_count
            else None
        ),
    }


def compare(
    left_checkpoint: Path,
    right_checkpoint: Path,
    left_seed: int,
    right_seed: int,
    config_path: Path,
) -> dict:
    left_initial_state = _seed_action_decoder(left_seed, config_path)
    right_initial_state = (
        left_initial_state
        if right_seed == left_seed
        else _seed_action_decoder(right_seed, config_path)
    )
    left_state = _checkpoint_state(left_checkpoint)
    right_state = _checkpoint_state(right_checkpoint)
    left_fingerprint_keys = set(_fingerprint_keys(left_state))
    right_fingerprint_keys = set(_fingerprint_keys(right_state))
    fingerprint_keys = left_fingerprint_keys | right_fingerprint_keys

    total = _accumulator()
    modules: dict[str, dict[str, float | int]] = {}
    tensor_count = 0
    for initial_key, left_initial_tensor in left_initial_state.items():
        checkpoint_key = f"agent.action_head.{initial_key}"
        if checkpoint_key in fingerprint_keys:
            continue
        if checkpoint_key not in left_state or checkpoint_key not in right_state:
            raise KeyError(f"Missing action-head tensor: {checkpoint_key}")
        left_tensor = left_state[checkpoint_key]
        right_tensor = right_state[checkpoint_key]
        right_initial_tensor = right_initial_state[initial_key]
        if not (
            torch.is_floating_point(left_initial_tensor)
            and torch.is_floating_point(right_initial_tensor)
            and torch.is_floating_point(left_tensor)
            and torch.is_floating_point(right_tensor)
        ):
            continue
        left_update = left_tensor.float() - left_initial_tensor.float()
        right_update = right_tensor.float() - right_initial_tensor.float()
        module_name = initial_key.split(".", 1)[0]
        module = modules.setdefault(module_name, _accumulator())
        _add(total, left_update, right_update)
        _add(module, left_update, right_update)
        tensor_count += 1

    return {
        "left_checkpoint": str(left_checkpoint),
        "right_checkpoint": str(right_checkpoint),
        "left_initialization_seed": left_seed,
        "right_initialization_seed": right_seed,
        "excluded_fingerprint_tensor_count": len(fingerprint_keys),
        "compared_tensor_count": tensor_count,
        "all_effective_action_head": _finalize(total),
        "by_module": {
            name: _finalize(values) for name, values in sorted(modules.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument(
        "--seed",
        type=int,
        help="Shared initialization seed (legacy shorthand for both checkpoints)",
    )
    parser.add_argument("--left-seed", type=int)
    parser.add_argument("--right-seed", type=int)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    left_seed = args.left_seed if args.left_seed is not None else args.seed
    right_seed = args.right_seed if args.right_seed is not None else args.seed
    if left_seed is None or right_seed is None:
        parser.error("provide --seed or both --left-seed and --right-seed")
    report = compare(args.left, args.right, left_seed, right_seed, args.config)
    output = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
