#!/usr/bin/env python3
"""Apply paired full-navtest Stage-B gates and issue formal 30k permission."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from tools.summarize_gp_sq3dmix_stage_b_v2 import aggregate_seed_paired


SEEDS = (20260824, 20260825)
MODES = ("real", "hard_shuffled", "spatial_shuffled", "control")
SUBMETRICS = {
    "NC": "no_at_fault_collisions",
    "DAC": "drivable_area_compliance",
    "TTC": "time_to_collision_within_bound",
}


def read_scores(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for row in rows:
        token = row.get("token")
        if not token or token in {"average", "average_all_frames"}:
            continue
        result[token] = {
            key: float(value)
            for key, value in row.items()
            if key and value not in (None, "") and key not in {"token", "valid"}
            and key != ""
        }
    if len(result) != 12146:
        raise RuntimeError(f"Expected 12,146 per-token scores in {path}, found {len(result)}")
    return result


def delta(first, second, column: str) -> np.ndarray:
    if set(first) != set(second):
        raise RuntimeError("Full-navtest paired token sets differ")
    return np.asarray(
        [first[token][column] - second[token][column] for token in sorted(first)],
        dtype=np.float64,
    )


def ci(values: np.ndarray, seed: int, draws: int = 10000) -> list[float]:
    rng = np.random.default_rng(seed)
    output = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 64):
        size = min(64, draws - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        output[start : start + size] = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(output, (0.025, 0.975))]


def atomic_json(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--stage-b-decision", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--decision-json", required=True)
    parser.add_argument("--permission-json", required=True)
    args = parser.parse_args()
    stage_b = json.loads(Path(args.stage_b_decision).read_text(encoding="utf-8"))
    if stage_b.get("all_passed") is not True:
        raise RuntimeError("Stage-B 2k gates did not pass; full-navtest is invalid")
    root = Path(args.root).resolve()
    aggregate_arrays = {"pdms": {mode: [] for mode in MODES[1:]}, "epdms": {"control": []}}
    submetric_arrays = {name: [] for name in SUBMETRICS}
    rows = []
    for seed in SEEDS:
        score_root = root / f"seed{seed}" / "scores"
        per_metric = {
            metric: {
                mode: read_scores(score_root / mode / f"{metric}.csv")
                for mode in MODES
            }
            for metric in ("pdms", "epdms")
        }
        for metric in ("pdms", "epdms"):
            for mode in MODES:
                values = np.asarray(
                    [row["score"] for row in per_metric[metric][mode].values()],
                    dtype=np.float64,
                )
                mode_delta = (
                    np.zeros_like(values)
                    if mode == "real"
                    else delta(per_metric[metric]["real"], per_metric[metric][mode], "score")
                )
                bounds = [0.0, 0.0] if mode == "real" else ci(mode_delta, seed + len(rows))
                rows.append(
                    {
                        "phase": "stage_b_full_navtest",
                        "status": "complete",
                        "seed": seed,
                        "mode": mode,
                        "metric": metric,
                        "sample_count": len(values),
                        "score": float(values.mean()),
                        "real_minus_mode_delta": float(mode_delta.mean()),
                        "bootstrap_ci_lower": bounds[0],
                        "bootstrap_ci_upper": bounds[1],
                    }
                )
                if mode in aggregate_arrays.get(metric, {}):
                    aggregate_arrays[metric][mode].append(mode_delta)
        for name, column in SUBMETRICS.items():
            submetric_arrays[name].append(
                delta(
                    per_metric["pdms"]["real"],
                    per_metric["pdms"]["control"],
                    column,
                )
            )

    aggregate = {}
    counter = 0
    for metric, modes in aggregate_arrays.items():
        for mode, arrays in modes.items():
            values = aggregate_seed_paired(arrays)
            aggregate[f"real_minus_{mode}_{metric}"] = {
                "mean": float(values.mean()),
                "bootstrap_ci": ci(values, 20260824 + counter),
            }
            counter += 1
    submetrics = {}
    for name, arrays in submetric_arrays.items():
        values = aggregate_seed_paired(arrays)
        submetrics[name] = {
            "mean": float(values.mean()),
            "bootstrap_ci": ci(values, 20260824 + counter),
        }
        counter += 1
    checks = {
        "real_control_pdms_mean": aggregate["real_minus_control_pdms"]["mean"] >= 0.0015,
        "real_control_pdms_ci": aggregate["real_minus_control_pdms"]["bootstrap_ci"][0] > 0.0,
        "real_hard_pdms_ci": aggregate["real_minus_hard_shuffled_pdms"]["bootstrap_ci"][0] > 0.0,
        "real_spatial_pdms_ci": aggregate["real_minus_spatial_shuffled_pdms"]["bootstrap_ci"][0] > 0.0,
        "real_control_epdms_mean": aggregate["real_minus_control_epdms"]["mean"] >= 0.0,
        "real_control_epdms_ci": aggregate["real_minus_control_epdms"]["bootstrap_ci"][0] >= -0.001,
        "NC_not_significantly_lower": submetrics["NC"]["bootstrap_ci"][1] >= 0.0,
        "DAC_not_significantly_lower": submetrics["DAC"]["bootstrap_ci"][1] >= 0.0,
        "TTC_not_significantly_lower": submetrics["TTC"]["bootstrap_ci"][1] >= 0.0,
        "one_safety_submetric_positive": any(value["mean"] > 0.0 for value in submetrics.values()),
    }
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    decision = {
        "schema_version": 2,
        "stage": "stage_b_full_navtest",
        "code_commit": commit,
        "stage_b_multiseed_decision": str(Path(args.stage_b_decision).resolve()),
        "aggregate": aggregate,
        "safety_submetrics": submetrics,
        "checks": checks,
        "all_passed": all(checks.values()),
    }
    permission = {
        "schema_version": 2,
        "code_commit": commit,
        "stage_a_passed": True,
        "stage_b_two_seed_passed": True,
        "stage_b_full_navtest_passed": decision["all_passed"],
        "formal_30k_allowed": decision["all_passed"],
        "formal_100k_allowed": False,
    }
    atomic_csv(Path(args.output_csv), rows)
    atomic_json(Path(args.decision_json), decision)
    atomic_json(Path(args.permission_json), permission)
    print(json.dumps({"decision": decision, "permission": permission}, indent=2, sort_keys=True))
    if not decision["all_passed"]:
        raise SystemExit("Full-navtest gates failed; formal 30k remains forbidden")


if __name__ == "__main__":
    main()
