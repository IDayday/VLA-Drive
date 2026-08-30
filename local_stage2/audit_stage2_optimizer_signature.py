#!/usr/bin/env python3
"""Recover optimizer/schedule clues from the released Stage-2 checkpoint.

For every floating-point ActionDecoder tensor, fit the released value as a
single scalar multiple of its deterministic seed-2 initialization.  A tensor
whose gradient is identically zero but materialized (rather than ``None``)
would carry an almost pure AdamW decay signature.  Such tensors can reveal the
cumulative learning rate without depending on dataset order.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from navsim.agents.EpisodeDrive.action_decoder import ActionDecoder


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT / "navsim/planning/script/config/common/agent/episode_drive.yaml"
)
DEFAULT_CHECKPOINT = Path(
    "/mnt/project/DriveVLA-M0-modelscope/"
    "best-epoch_26-step_174312.server_merged.ckpt"
)


def _initial_state(seed: int, config_path: Path) -> dict[str, torch.Tensor]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return ActionDecoder(OmegaConf.load(config_path).action_head_config).state_dict()


def _checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    state = payload.get("state_dict", payload)
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint payload: {type(state)}")
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=174312)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    initial = _initial_state(args.seed, args.config)
    released = _checkpoint_state(args.checkpoint)
    rows = []
    for initial_key, initial_tensor in initial.items():
        checkpoint_key = f"agent.action_head.{initial_key}"
        if checkpoint_key not in released:
            continue
        checkpoint_tensor = released[checkpoint_key]
        if not (
            torch.is_floating_point(initial_tensor)
            and torch.is_floating_point(checkpoint_tensor)
        ):
            continue
        x = initial_tensor.float().reshape(-1)
        y = checkpoint_tensor.float().reshape(-1)
        x_norm_sq = torch.dot(x, x).item()
        y_norm_sq = torch.dot(y, y).item()
        if x_norm_sq == 0.0 or y_norm_sq == 0.0:
            scale = None
            residual = None
        else:
            scale = torch.dot(x, y).item() / x_norm_sq
            residual = torch.linalg.vector_norm(y - scale * x).item() / math.sqrt(
                y_norm_sq
            )
        exact = torch.equal(
            checkpoint_tensor, initial_tensor.to(checkpoint_tensor.dtype)
        )
        implied_sum_lr = None
        implied_constant_lr = None
        if scale is not None and 0.0 < scale < 1.0:
            implied_sum_lr = -math.log(scale) / args.weight_decay
            implied_constant_lr = implied_sum_lr / args.steps
        rows.append(
            {
                "key": initial_key,
                "numel": initial_tensor.numel(),
                "exact": exact,
                "scale": scale,
                "relative_residual_after_scale": residual,
                "implied_sum_lr_if_pure_decay": implied_sum_lr,
                "implied_constant_lr_if_pure_decay": implied_constant_lr,
            }
        )

    proportional = sorted(
        (
            row
            for row in rows
            if not row["exact"]
            and row["scale"] is not None
            and row["relative_residual_after_scale"] is not None
        ),
        key=lambda row: row["relative_residual_after_scale"],
    )
    report = {
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "steps": args.steps,
        "weight_decay": args.weight_decay,
        "tensor_count": len(rows),
        "exact_tensor_count": sum(row["exact"] for row in rows),
        "best_proportional_candidates": proportional[:40],
        "interpretation": (
            "Only candidates with a near-zero proportional residual can be "
            "treated as pure AdamW decay. A non-negligible residual means the "
            "tensor also received task gradients and cannot identify the LR."
        ),
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
