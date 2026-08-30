#!/usr/bin/env python3
"""Compare checkpoint displacement with candidate Stage-2 LR schedules.

This is a diagnostic, not an optimizer reconstruction.  It treats parameter
updates as a noisy random walk, for which RMS displacement scales with the
square root of the accumulated squared learning rate.  The approximation is
useful for ruling out schedules whose update budget is grossly inconsistent
with the released checkpoint, but a full training run remains the final test.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _relative_lr(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        progress = step / warmup_steps
        return 1e-6 + (1.0 - 1e-6) * progress
    decay_steps = total_steps - warmup_steps
    progress = min(1.0, (step - warmup_steps) / decay_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("comparison", type=Path)
    parser.add_argument("--epochs", type=int, default=27)
    parser.add_argument("--steps-per-epoch", type=int, default=6456)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    comparison = json.loads(args.comparison.read_text())
    updates = comparison["all_effective_action_head"]
    released_rms = float(updates["left_update_rms"])
    epoch_rms = float(updates["right_update_rms"])
    total_steps = args.epochs * args.steps_per_epoch
    warmup_steps = int(total_steps * args.warmup_ratio)

    cosine_square_budget = sum(
        _relative_lr(step, total_steps, warmup_steps) ** 2
        for step in range(total_steps)
    )
    schedules = {
        "constant_peak_lr": float(total_steps),
        "constant_half_peak_lr": float(total_steps) * 0.25,
        "source_10pct_warmup_cosine": cosine_square_budget,
    }
    epoch_square_budget = float(args.steps_per_epoch)
    estimates = {}
    for name, square_budget in schedules.items():
        predicted_rms = epoch_rms * math.sqrt(
            square_budget / epoch_square_budget
        )
        estimates[name] = {
            "relative_lr_squared_budget": square_budget / total_steps,
            "predicted_final_update_rms": predicted_rms,
            "prediction_minus_released": predicted_rms - released_rms,
            "relative_error": predicted_rms / released_rms - 1.0,
        }

    result = {
        "method": "random_walk_squared_lr_budget_approximation",
        "caveat": (
            "Gradient statistics evolve during training; use this to prioritize "
            "full runs, not as proof of the private training schedule."
        ),
        "released_final_update_rms": released_rms,
        "constant_peak_epoch0_update_rms": epoch_rms,
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "estimates": estimates,
    }
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
