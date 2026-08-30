#!/usr/bin/env python3
"""Snapshot a Stage-2 checkpoint after a requested validation milestone.

This helper watches TensorBoard scalars rather than the training log, whose
progress-bar carriage returns make it unsuitable as a synchronization API.  It
never imports the live training modules and does not touch the source
checkpoint.  A retained best checkpoint is hard-linked; a mutable ``last``
checkpoint is copied with copy-on-write when the filesystem supports it.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


VALIDATION_TAGS = (
    "val/score_epoch",
    "val/best_score",
    "val/lost_score",
    "val/l2",
)


def _validation_row(event_file: Path, epoch_index: int) -> dict | None:
    accumulator = EventAccumulator(
        str(event_file), size_guidance={"scalars": 0}
    )
    accumulator.Reload()
    available = set(accumulator.Tags()["scalars"])
    if VALIDATION_TAGS[0] not in available:
        return None
    primary = accumulator.Scalars(VALIDATION_TAGS[0])
    if len(primary) <= epoch_index:
        return None

    row = {
        "validation_index": epoch_index,
        "step": primary[epoch_index].step,
        "wall_time": primary[epoch_index].wall_time,
    }
    for tag in VALIDATION_TAGS:
        values = accumulator.Scalars(tag) if tag in available else []
        row[tag] = values[epoch_index].value if len(values) > epoch_index else None
    return row


def _snapshot(source_link: Path, destination: Path) -> tuple[Path, str]:
    source = source_link.resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite milestone: {destination}")

    if source_link.is_symlink():
        os.link(source, destination)
        method = "hardlink_retained_best"
    else:
        subprocess.run(
            ["cp", "--reflink=auto", "--sparse=always", str(source), str(destination)],
            check=True,
        )
        method = "reflink_or_copy_mutable_last"

    if destination.stat().st_size != source.stat().st_size:
        raise RuntimeError("Milestone checkpoint size differs from its source")
    return source, method


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--validation-index",
        type=int,
        default=9,
        help="Zero-based validation index; 9 snapshots the tenth validation.",
    )
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--training-pid", type=int)
    args = parser.parse_args()
    if args.validation_index < 0:
        parser.error("--validation-index must be non-negative")
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")

    event_files = sorted(args.event_dir.glob("events.out.tfevents.*"))
    if len(event_files) != 1:
        raise RuntimeError(
            f"Expected one TensorBoard event file, found: {event_files}"
        )
    event_file = event_files[0]
    last_checkpoint = args.checkpoint_dir / "last.ckpt"

    while True:
        row = _validation_row(event_file, args.validation_index)
        if row is not None:
            break
        if args.training_pid is not None and not Path(
            f"/proc/{args.training_pid}"
        ).exists():
            raise RuntimeError("Training exited before the requested milestone")
        time.sleep(args.poll_seconds)

    # Scalar logging precedes checkpoint serialization.  Require a source
    # timestamp at or after the validation scalar and a stable size.
    previous_size = None
    while True:
        if last_checkpoint.exists():
            source = last_checkpoint.resolve(strict=True)
            size = source.stat().st_size
            if source.stat().st_mtime >= row["wall_time"] and size == previous_size:
                break
            previous_size = size
        time.sleep(args.poll_seconds)

    source, method = _snapshot(last_checkpoint, args.output)
    metadata = {
        "event_file": str(event_file.resolve()),
        "source_checkpoint": str(source),
        "output_checkpoint": str(args.output.resolve()),
        "snapshot_method": method,
        "size_bytes": args.output.stat().st_size,
        "validation": row,
        "created_at_unix": time.time(),
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
