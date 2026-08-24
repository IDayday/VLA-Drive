#!/usr/bin/env python3
"""Summarize paired trajectory and NAVSIM intervention results."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


MODES = ("real", "zero", "shuffled", "slot_mean")


def read_scores(path: Path) -> tuple[dict[str, float], float]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    per_token = {}
    aggregate = float("nan")
    for row in rows:
        token = row.get("token", "")
        raw = row.get("score", "")
        if not raw:
            continue
        value = float(raw)
        if token in {"average", "average_all_frames"}:
            aggregate = value
        else:
            per_token[token] = value
    if not np.isfinite(aggregate) and per_token:
        aggregate = float(np.mean(list(per_token.values())))
    return per_token, aggregate


def bootstrap(values: np.ndarray, seed: int, draws: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 256):
        size = min(256, draws - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        means[start : start + size] = values[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def atomic_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--datalist", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-slot-mean", action="store_true")
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()
    root = Path(args.root)
    modes = MODES if args.include_slot_mean else MODES[:3]
    tokens = json.loads(Path(args.datalist).read_text(encoding="utf-8"))
    trajectories = {}
    pdms = {}
    epdms = {}
    summary_rows = []
    for mode in modes:
        prediction_dir = root / "predictions" / mode / "test"
        if not prediction_dir.is_dir():
            candidates = list((root / "predictions" / mode).glob("*/test"))
            if len(candidates) != 1:
                raise RuntimeError(
                    f"Expected exactly one prediction directory for {mode}, found {candidates}"
                )
            prediction_dir = candidates[0]
        arrays = []
        for token in tokens:
            path = prediction_dir / f"{token}.npy"
            if not path.is_file():
                raise FileNotFoundError(path)
            arrays.append(np.load(path))
        trajectories[mode] = np.stack(arrays)
        pdms[mode], pdms_mean = read_scores(root / "scores" / mode / "pdms.csv")
        epdms[mode], epdms_mean = read_scores(root / "scores" / mode / "epdms.csv")
        summary_rows.append(
            {
                "mode": mode,
                "sample_count": len(tokens),
                "pdms": pdms_mean,
                "epdms": epdms_mean,
            }
        )
    output = Path(args.output)
    atomic_csv(output, ["mode", "sample_count", "pdms", "epdms"], summary_rows)
    paired_rows = []
    real = trajectories["real"]
    for mode in modes[1:]:
        other = trajectories[mode]
        l2 = np.linalg.norm((real - other).reshape(len(tokens), -1), axis=1)
        common_pdms = [token for token in tokens if token in pdms["real"] and token in pdms[mode]]
        common_epdms = [token for token in tokens if token in epdms["real"] and token in epdms[mode]]
        for metric, source, common in (
            ("pdms", pdms, common_pdms),
            ("epdms", epdms, common_epdms),
        ):
            delta = np.asarray(
                [source["real"][token] - source[mode][token] for token in common],
                dtype=np.float64,
            )
            if len(delta):
                lower, upper = bootstrap(delta, args.seed, args.bootstrap_draws)
                mean_delta = float(delta.mean())
            else:
                lower = upper = mean_delta = float("nan")
            paired_rows.append(
                {
                    "comparison": f"real-{mode}",
                    "metric": metric,
                    "paired_count": len(delta),
                    "mean_delta": mean_delta,
                    "bootstrap_ci_lower": lower,
                    "bootstrap_ci_upper": upper,
                    "trajectory_l2_mean": float(l2.mean()),
                    "trajectory_l2_median": float(np.median(l2)),
                    "identical_trajectory_ratio": float(
                        np.asarray(
                            [np.array_equal(a, b) for a, b in zip(real, other)]
                        ).mean()
                    ),
                }
            )
    paired_output = output.with_name(output.stem + "_paired.csv")
    atomic_csv(paired_output, list(paired_rows[0]), paired_rows)
    print(json.dumps({"summary": str(output), "paired": str(paired_output)}, indent=2))


if __name__ == "__main__":
    main()
