#!/usr/bin/env python3
"""Verify that the live Stage-2 TensorBoard LR trace follows the source schedule.

The checkpoint loop state proves that a scheduler is stepped, but it does not
prove which values Lightning applied.  This audit compares every persisted LR
sample against the configured linear-warmup/cosine formula and records the
exact observed step range.  It is safe to run while training is active because
TensorBoard event files are append-only.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


def expected_lr(
    step: int,
    *,
    peak_lr: float,
    total_steps: int,
    warmup_steps: int,
    start_lr_ratio: float,
    min_lr_ratio: float,
) -> float:
    """Return the source LambdaLR value logged at ``global_step=step``."""

    if step < warmup_steps:
        progress = step / warmup_steps
        multiplier = start_lr_ratio + (1.0 - start_lr_ratio) * progress
    else:
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, (step - warmup_steps) / decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        multiplier = min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return peak_lr * multiplier


def audit_samples(
    samples: Iterable[tuple[int, float]],
    *,
    peak_lr: float,
    total_steps: int,
    warmup_steps: int,
    start_lr_ratio: float,
    min_lr_ratio: float,
) -> dict[str, Any]:
    """Compare ``(step, value)`` samples with the expected curve."""

    rows = list(samples)
    if not rows:
        raise ValueError("LR trace contains no scalar samples")
    steps = [step for step, _ in rows]
    if steps != sorted(steps) or len(steps) != len(set(steps)):
        raise ValueError("LR trace steps must be strictly increasing")

    errors = []
    relative_errors = []
    for step, observed in rows:
        expected = expected_lr(
            step,
            peak_lr=peak_lr,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            start_lr_ratio=start_lr_ratio,
            min_lr_ratio=min_lr_ratio,
        )
        error = abs(observed - expected)
        errors.append(error)
        relative_errors.append(error / max(abs(expected), 1e-30))

    intervals = [right - left for left, right in zip(steps, steps[1:])]
    before_boundary = [row for row in rows if row[0] < warmup_steps]
    after_boundary = [row for row in rows if row[0] >= warmup_steps]
    return {
        "sample_count": len(rows),
        "first_step": steps[0],
        "last_step": steps[-1],
        "step_interval_values": sorted(set(intervals)),
        "max_absolute_lr_error": max(errors),
        "max_relative_lr_error": max(relative_errors),
        "first_observed_lr": rows[0][1],
        "first_expected_lr": expected_lr(
            steps[0],
            peak_lr=peak_lr,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            start_lr_ratio=start_lr_ratio,
            min_lr_ratio=min_lr_ratio,
        ),
        "last_observed_lr": rows[-1][1],
        "last_expected_lr": expected_lr(
            steps[-1],
            peak_lr=peak_lr,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            start_lr_ratio=start_lr_ratio,
            min_lr_ratio=min_lr_ratio,
        ),
        "covers_warmup_boundary": bool(before_boundary and after_boundary),
        "last_sample_before_warmup_end": (
            {"step": before_boundary[-1][0], "lr": before_boundary[-1][1]}
            if before_boundary
            else None
        ),
        "first_sample_after_warmup_end": (
            {"step": after_boundary[0][0], "lr": after_boundary[0][1]}
            if after_boundary
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-dir", type=Path, required=True)
    parser.add_argument("--tag", default="lr-AdamW/action_head_decay")
    parser.add_argument("--peak-lr", type=float, default=1e-4)
    parser.add_argument("--total-steps", type=int, default=174_312)
    parser.add_argument("--warmup-steps", type=int, default=17_431)
    parser.add_argument("--start-lr-ratio", type=float, default=1e-6)
    parser.add_argument("--min-lr-ratio", type=float, default=0.0)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    event_files = sorted(args.event_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event file in {args.event_dir}")
    event_file = max(event_files, key=lambda path: path.stat().st_mtime_ns)
    accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    accumulator.Reload()
    samples = [(item.step, float(item.value)) for item in accumulator.Scalars(args.tag)]
    comparison = audit_samples(
        samples,
        peak_lr=args.peak_lr,
        total_steps=args.total_steps,
        warmup_steps=args.warmup_steps,
        start_lr_ratio=args.start_lr_ratio,
        min_lr_ratio=args.min_lr_ratio,
    )
    result = {
        "audit": "active_stage2_tensorboard_lr_trace",
        "event_file": str(event_file.resolve()),
        "event_file_size_at_audit": event_file.stat().st_size,
        "tag": args.tag,
        "configured_schedule": {
            "peak_lr": args.peak_lr,
            "total_steps": args.total_steps,
            "warmup_steps": args.warmup_steps,
            "start_lr_ratio": args.start_lr_ratio,
            "min_lr_ratio": args.min_lr_ratio,
        },
        "comparison": comparison,
        "absolute_tolerance": args.absolute_tolerance,
        "all_persisted_samples_match": (
            comparison["max_absolute_lr_error"] <= args.absolute_tolerance
        ),
        "interpretation": (
            "Every TensorBoard LR sample is compared at its persisted global "
            "step. A passing result rules out Lightning call-order, missed-step, "
            "duplicate-step, and off-by-one errors over the observed interval."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(payload)
    print(payload, end="")
    if not result["all_persisted_samples_match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
