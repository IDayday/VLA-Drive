#!/usr/bin/env python3
"""Audit numeric integrity of the reconstructed Stage-2 long target cache."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import pickle

import numpy as np


DEFAULT_CACHE = Path(
    "/mnt/project/DriveVLA-M0-stage2/cache/feature_cache_navtrain_long2"
)


def _heading_deltas(trajectory: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = np.diff(trajectory[:, 2])
    wrapped = (raw + np.pi) % (2 * np.pi) - np.pi
    return raw, wrapped


def audit(cache: Path, samples_per_log: int = 1) -> dict:
    if samples_per_log <= 0:
        raise ValueError("samples_per_log must be positive")
    log_paths = [path for path in cache.iterdir() if path.is_dir()]
    rows = []
    for log_path in log_paths:
        token_paths = [path for path in log_path.iterdir() if path.is_dir()]
        if not token_paths:
            raise RuntimeError(f"empty cache log: {log_path}")
        # CacheOnlyDataset depends on filesystem enumeration order. Audit in
        # that same order and choose evenly spread deterministic positions.
        indices = np.linspace(
            0,
            len(token_paths) - 1,
            min(samples_per_log, len(token_paths)),
            dtype=np.int64,
        )
        for index in np.unique(indices):
            token_path = token_paths[int(index)]
            with gzip.open(token_path / "trajectory_target.gz", "rb") as stream:
                target = pickle.load(stream)
            for field in ("trajectory", "trajectory_long"):
                if field not in target:
                    raise KeyError(f"{field} missing from {token_path}")
                trajectory = np.asarray(target[field], dtype=np.float64)
                if trajectory.shape != (8, 3):
                    raise ValueError(
                        f"wrong {field} shape at {token_path}: {trajectory.shape}"
                    )
                raw, wrapped = _heading_deltas(trajectory)
                rows.append(
                    {
                        "field": field,
                        "log_name": log_path.name,
                        "scene_token": token_path.name,
                        "finite": bool(np.isfinite(trajectory).all()),
                        "max_abs_heading_rad": float(
                            np.max(np.abs(trajectory[:, 2]))
                        ),
                        "max_raw_heading_step_rad": float(np.max(np.abs(raw))),
                        "max_wrapped_heading_step_rad": float(
                            np.max(np.abs(wrapped))
                        ),
                    }
                )

    fields = {}
    for field in ("trajectory", "trajectory_long"):
        field_rows = [row for row in rows if row["field"] == field]
        worst_raw = max(field_rows, key=lambda row: row["max_raw_heading_step_rad"])
        worst_wrapped = max(
            field_rows, key=lambda row: row["max_wrapped_heading_step_rad"]
        )
        fields[field] = {
            "target_count": len(field_rows),
            "finite_count": sum(row["finite"] for row in field_rows),
            "raw_heading_jump_over_pi_count": sum(
                row["max_raw_heading_step_rad"] > np.pi for row in field_rows
            ),
            "wrapped_heading_step_over_0_5_count": sum(
                row["max_wrapped_heading_step_rad"] > 0.5 for row in field_rows
            ),
            "max_abs_heading_rad": max(
                row["max_abs_heading_rad"] for row in field_rows
            ),
            "worst_raw_heading_step": worst_raw,
            "worst_wrapped_heading_step": worst_wrapped,
        }

    report = {
        "cache": str(cache.resolve()),
        "log_count": len(log_paths),
        "samples_per_log": samples_per_log,
        "selection": "filesystem-order evenly spaced positions per log",
        "fields": fields,
    }
    long_stats = fields["trajectory_long"]
    report["pass"] = bool(
        long_stats["finite_count"] == long_stats["target_count"]
        and long_stats["raw_heading_jump_over_pi_count"] == 0
        and long_stats["wrapped_heading_step_over_0_5_count"] == 0
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--samples-per-log", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.cache, args.samples_per_log)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
