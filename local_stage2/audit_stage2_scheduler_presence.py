#!/usr/bin/env python3
"""Prove whether a Lightning Stage-2 checkpoint executed an LR scheduler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _loop_progress(path: Path) -> dict:
    payload = torch.load(
        path, map_location="cpu", mmap=True, weights_only=False
    )
    epoch_loop = payload["loops"]["fit_loop"]
    return {
        "checkpoint": str(path.resolve()),
        "global_step": int(payload["global_step"]),
        "scheduler_progress": epoch_loop["epoch_loop.scheduler_progress"],
        "optimizer_step_progress": epoch_loop[
            "epoch_loop.automatic_optimization.optim_progress"
        ]["optimizer"]["step"],
        "saved_scheduler_state_count": len(payload.get("lr_schedulers", [])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-audit", type=Path, required=True)
    parser.add_argument("--no-scheduler", type=Path, required=True)
    parser.add_argument("--with-scheduler", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    public = json.loads(args.public_audit.read_text())
    no_scheduler = _loop_progress(args.no_scheduler)
    with_scheduler = _loop_progress(args.with_scheduler)
    public_rows = [
        {
            "name": shard["name"],
            "global_step": shard["global_step"],
            "scheduler_progress": shard["scheduler_progress"],
            "optimizer_step_progress": shard["optimizer_step_progress"],
        }
        for shard in public["shards"]
    ]

    public_all_steps = all(
        row["scheduler_progress"]["total"]["completed"]
        == row["optimizer_step_progress"]["total"]["completed"]
        == row["global_step"]
        for row in public_rows
    )
    no_scheduler_is_zero = (
        no_scheduler["scheduler_progress"]["total"]["completed"] == 0
    )
    control_all_steps = (
        with_scheduler["scheduler_progress"]["total"]["completed"]
        == with_scheduler["optimizer_step_progress"]["total"]["completed"]
        == with_scheduler["global_step"]
    )
    if not (public_all_steps and no_scheduler_is_zero and control_all_steps):
        raise RuntimeError("scheduler-presence control invariants failed")

    report = {
        "public_historical_shards": public_rows,
        "local_no_scheduler_control": no_scheduler,
        "local_step_scheduler_control": with_scheduler,
        "invariants": {
            "public_scheduler_completed_every_optimizer_step": public_all_steps,
            "no_scheduler_control_progress_is_zero": no_scheduler_is_zero,
            "step_scheduler_control_completed_every_optimizer_step": control_all_steps,
        },
        "conclusion": (
            "The public Stage-2 run executed an LR scheduler at every optimizer "
            "step. Stripped scheduler state prevents identifying its class from "
            "the artifact alone."
        ),
    }
    output = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
