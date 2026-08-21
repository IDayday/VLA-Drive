#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Cache-field conventions are adapted from William-Yao-2000/DriveSuprim,
# navsim/agents/tools/split_final_ensemble_pickle.py.  This project version
# validates the official eight-metric schema and writes atomic NPZ shards.

"""Split DriveSuprim's aggregate navtrain pickle into lazy per-token shards.

The official cache is a large Python mapping and therefore has to be loaded
once during conversion.  Training ranks subsequently read only their current
batch tokens instead of each unpickling the complete aggregate file.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Mapping

import numpy as np


METRICS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "time_to_collision_within_bound",
    "ego_progress",
    "driving_direction_compliance",
    "lane_keeping",
    "traffic_light_compliance",
    "history_comfort",
)


def _entry(value: object, token: str) -> Mapping[str, object]:
    if isinstance(value, Mapping) and isinstance(value.get("scores"), Mapping):
        value = value["scores"]
    if not isinstance(value, Mapping):
        raise TypeError(f"cache entry for token {token!r} is not a mapping")
    missing = set(METRICS).difference(value)
    if missing:
        raise KeyError(f"cache entry {token!r} is missing {sorted(missing)}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--vocab-size", default=8192, type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = args.input.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"DriveSuprim aggregate cache not found: {source}")
    if args.vocab_size <= 0:
        raise ValueError("--vocab-size must be positive")
    output_root = args.output_root.expanduser().resolve()
    split_root = output_root / args.split
    success_path = output_root / "_SUCCESS.json"
    if success_path.is_file() and not args.overwrite:
        print(f"DriveSuprim shard cache already complete: {success_path}")
        return
    split_root.mkdir(parents=True, exist_ok=True)

    print(
        "Loading the official aggregate cache once; peak host RAM is roughly "
        f"the uncompressed pickle size ({source.stat().st_size / 2**30:.1f} GiB).",
        flush=True,
    )
    with source.open("rb") as stream:
        aggregate = pickle.load(stream)
    if not isinstance(aggregate, Mapping) or not aggregate:
        raise TypeError("official DriveSuprim cache must be a non-empty mapping")

    count = 0
    for token, raw_value in aggregate.items():
        token = str(token)
        raw = _entry(raw_value, token)
        arrays = {}
        for name in METRICS:
            value = np.asarray(raw[name], dtype=np.float32).squeeze()
            if value.shape != (args.vocab_size,):
                raise ValueError(
                    f"{token}/{name} has shape {value.shape}, expected "
                    f"({args.vocab_size},)"
                )
            if not np.isfinite(value).all():
                raise ValueError(f"{token}/{name} contains NaN or Inf")
            arrays[name] = np.ascontiguousarray(value)
        destination = split_root / f"{token}.npz"
        if destination.exists() and not args.overwrite:
            count += 1
            continue
        temporary = destination.with_name(
            f".{destination.name}.tmp-{os.getpid()}.npz"
        )
        np.savez(temporary, **arrays)
        os.replace(temporary, destination)
        count += 1
        if count % 1000 == 0:
            print(f"wrote {count}/{len(aggregate)} static-score shards", flush=True)

    manifest = {
        "schema_version": 1,
        "split": args.split,
        "vocab_size": args.vocab_size,
        "metrics": list(METRICS),
        "num_tokens": count,
        "source_file": str(source),
        "source_size": source.stat().st_size,
    }
    temporary_manifest = success_path.with_name(
        f".{success_path.name}.tmp-{os.getpid()}"
    )
    with temporary_manifest.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_manifest, success_path)
    print(f"DriveSuprim static-score cache ready: {output_root} ({count} tokens)")


if __name__ == "__main__":
    main()
