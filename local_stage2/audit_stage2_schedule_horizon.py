#!/usr/bin/env python3
"""Quantify the unresolved 25-vs-27-epoch Stage-2 scheduler hypothesis.

The public checkpoint proves 27 blocks of 6,456 optimizer updates and proves
that a scheduler was stepped each time.  Its optimizer and scheduler state was
stripped, however, while the original run directory contains ``25epochs``.
This audit evaluates the released SequentialLR exactly under both possible
configured horizons without treating the directory label as ground truth.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any
import warnings

import torch


def released_lr_trace(
    *,
    schedule_epochs: int,
    training_epochs: int,
    steps_per_epoch: int,
    warmup_ratio: float = 0.1,
    start_factor: float = 1e-6,
) -> list[float]:
    """Return post-update LR multipliers from the released scheduler code."""

    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    total_steps = steps_per_epoch * schedule_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=start_factor,
        total_iters=warmup_steps,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=0.0,
        last_epoch=-1,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )

    trace = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for _ in range(steps_per_epoch * training_epochs):
            optimizer.step()
            scheduler.step()
            trace.append(float(scheduler.get_last_lr()[0]))
    return trace


def summarize_trace(
    trace: list[float], *, schedule_epochs: int, steps_per_epoch: int
) -> dict[str, Any]:
    total = len(trace)
    configured_steps = schedule_epochs * steps_per_epoch
    minimum = min(trace)
    min_step = trace.index(minimum) + 1
    post_horizon = trace[configured_steps:] if configured_steps < total else []
    return {
        "configured_schedule_epochs": schedule_epochs,
        "configured_schedule_steps": configured_steps,
        "actual_training_steps": total,
        "warmup_steps": int(configured_steps * 0.1),
        "lr_multiplier_sum": sum(trace),
        "lr_multiplier_square_sum": sum(value * value for value in trace),
        "sqrt_lr_square_budget": math.sqrt(
            sum(value * value for value in trace)
        ),
        "minimum_lr_multiplier": minimum,
        "minimum_lr_step": min_step,
        "final_lr_multiplier": trace[-1],
        "maximum_post_horizon_lr_multiplier": max(post_horizon, default=0.0),
        "epoch_end_lr_multipliers": {
            str(epoch): trace[epoch * steps_per_epoch - 1]
            for epoch in range(1, total // steps_per_epoch + 1)
        },
    }


def audit_horizons(
    *, steps_per_epoch: int = 6_456, training_epochs: int = 27
) -> dict[str, Any]:
    trace_27 = released_lr_trace(
        schedule_epochs=27,
        training_epochs=training_epochs,
        steps_per_epoch=steps_per_epoch,
    )
    trace_25 = released_lr_trace(
        schedule_epochs=25,
        training_epochs=training_epochs,
        steps_per_epoch=steps_per_epoch,
    )
    summary_27 = summarize_trace(
        trace_27, schedule_epochs=27, steps_per_epoch=steps_per_epoch
    )
    summary_25 = summarize_trace(
        trace_25, schedule_epochs=25, steps_per_epoch=steps_per_epoch
    )
    differences = [abs(a - b) for a, b in zip(trace_27, trace_25)]
    first_material = next(
        (index + 1 for index, value in enumerate(differences) if value >= 1e-3),
        None,
    )
    return {
        "audit": "stage2_scheduler_horizon_counterfactual",
        "direct_checkpoint_facts": {
            "global_step": steps_per_epoch * training_epochs,
            "optimizer_steps_per_epoch": steps_per_epoch,
            "optimizer_step_epochs": training_epochs,
            "checkpoint_epoch_index": 26,
            "scheduler_stepped_every_optimizer_step": True,
            "scheduler_state_stripped": True,
            "hyperparameters_absent": True,
        },
        "indirect_artifact_clue": {
            "run_directory_label": "training_episode_Nav1_traj_long_25epochs_visionlora",
            "encoded_epoch_count": 25,
            "is_authoritative_config": False,
        },
        "hypotheses": {
            "schedule_horizon_27_epochs": summary_27,
            "schedule_horizon_25_epochs_overrun_to_27": summary_25,
        },
        "comparison": {
            "first_step_with_absolute_multiplier_difference_ge_1e-3": first_material,
            "max_absolute_lr_multiplier_difference": max(differences),
            "horizon25_over_horizon27_lr_sum": (
                summary_25["lr_multiplier_sum"]
                / summary_27["lr_multiplier_sum"]
            ),
            "horizon25_over_horizon27_sqrt_lr2_budget": (
                summary_25["sqrt_lr_square_budget"]
                / summary_27["sqrt_lr_square_budget"]
            ),
            "horizon25_restarts_after_step": 25 * steps_per_epoch,
            "horizon25_final_actual_lr_at_peak_1e-4": (
                summary_25["final_lr_multiplier"] * 1e-4
            ),
            "horizon27_final_actual_lr_at_peak_1e-4": (
                summary_27["final_lr_multiplier"] * 1e-4
            ),
        },
        "conclusion": (
            "The checkpoint proves 27 optimizer-step epochs, but it cannot prove "
            "whether the released cosine scheduler was configured for 27 epochs "
            "or for the 25 epochs encoded in the stale run-directory label. A "
            "25-epoch released CosineAnnealingLR reaches zero at epoch 25 and "
            "then rises again during epochs 26-27. This is a real late-training "
            "counterfactual, but the directory name alone is too weak to justify "
            "a second full run before the predeclared epoch-9 checkpoint or a "
            "controlled continuation test."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps-per-epoch", type=int, default=6_456)
    parser.add_argument("--training-epochs", type=int, default=27)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_horizons(
        steps_per_epoch=args.steps_per_epoch,
        training_epochs=args.training_epochs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
