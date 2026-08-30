#!/usr/bin/env python3
"""Export comparable TensorBoard scalars from controlled Stage-2 runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


DEFAULT_TAGS = (
    "lr-AdamW/action_head_decay",
    "train/loss",
    "train/trajectory_loss",
    "train/final_score_loss",
    "val/score_epoch",
    "val/best_score",
    "val/lost_score",
    "val/score_hit_rate",
    "val/top_5_score_hit_rate",
    "val/collision",
    "val/ttc",
    "val/dac",
    "val/progress",
    "val/comfort",
    "val/l2",
)


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must use NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("run must use nonempty NAME=PATH")
    return name, Path(path)


def _load_run(path: Path, tags: tuple[str, ...]) -> dict:
    event_files = sorted(path.glob("lightning_logs/version_*/events.out.tfevents.*"))
    if len(event_files) != 1:
        raise RuntimeError(f"Expected one event file under {path}, found {event_files}")
    accumulator = EventAccumulator(
        str(event_files[0]), size_guidance={"scalars": 0}
    )
    accumulator.Reload()
    available = set(accumulator.Tags()["scalars"])
    scalars = {}
    for tag in tags:
        if tag not in available:
            continue
        scalars[tag] = [
            {"step": item.step, "value": item.value, "wall_time": item.wall_time}
            for item in accumulator.Scalars(tag)
        ]
    return {"path": str(path), "event_file": str(event_files[0]), "scalars": scalars}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=_parse_run, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    runs = {name: _load_run(path, DEFAULT_TAGS) for name, path in args.run}
    report = {"runs": runs}
    if len(runs) == 2:
        left_name, right_name = runs
        deltas = {}
        for tag in DEFAULT_TAGS:
            left = runs[left_name]["scalars"].get(tag, [])
            right = runs[right_name]["scalars"].get(tag, [])
            if left and right:
                deltas[tag] = {
                    "left": left[-1]["value"],
                    "right": right[-1]["value"],
                    "right_minus_left": right[-1]["value"] - left[-1]["value"],
                }
        report["comparison"] = {
            "left": left_name,
            "right": right_name,
            "final_scalar_deltas": deltas,
        }

    output = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
