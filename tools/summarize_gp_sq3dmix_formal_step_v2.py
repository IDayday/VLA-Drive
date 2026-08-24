#!/usr/bin/env python3
"""Gate one paired Stage-C checkpoint on fixed navtest-2k."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.summarize_gp_sq3dmix_stage_b_v2 import (
    bootstrap_ci,
    gradient_active,
    inference_diagnostics,
    paired_delta,
    scores,
)


MODES = ("real", "hard_shuffled", "spatial_shuffled", "control")


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
    parser.add_argument("--gp-run", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    per_metric = {"pdms": {}, "epdms": {}}
    means = {"pdms": {}, "epdms": {}}
    for metric in per_metric:
        for mode in MODES:
            per_metric[metric][mode], means[metric][mode] = scores(
                root / "scores" / mode / f"{metric}.csv"
            )
    rows = []
    deltas = {}
    for metric in per_metric:
        deltas[metric] = {}
        for mode in MODES:
            value = (
                np.zeros(len(per_metric[metric]["real"]), dtype=np.float64)
                if mode == "real"
                else paired_delta(per_metric[metric]["real"], per_metric[metric][mode])
            )
            deltas[metric][mode] = value
            bounds = [0.0, 0.0] if mode == "real" else bootstrap_ci(value, args.seed + len(rows))
            rows.append(
                {
                    "phase": "formal_30k",
                    "status": "complete",
                    "seed": args.seed,
                    "step": args.step,
                    "mode": mode,
                    "metric": metric,
                    "score": means[metric][mode],
                    "real_minus_mode_delta": float(value.mean()),
                    "bootstrap_ci_lower": bounds[0],
                    "bootstrap_ci_upper": bounds[1],
                }
            )
    diagnostics = inference_diagnostics(root)
    gradients_ok, gradients = gradient_active(Path(args.gp_run), args.variant)
    checks = {
        "real_control_pdms_positive": float(deltas["pdms"]["control"].mean()) > 0.0,
        "real_hard_pdms_positive": float(deltas["pdms"]["hard_shuffled"].mean()) > 0.0,
        "real_spatial_pdms_positive": float(deltas["pdms"]["spatial_shuffled"].mean()) > 0.0,
        "epdms_nonnegative": float(deltas["epdms"]["control"].mean()) >= 0.0,
        "residual_p95": diagnostics["residual_p95"] <= 0.30,
        "residual_max": diagnostics["residual_max"] <= 0.60,
        "alpha_not_collapsed": diagnostics["alpha_min"] > 0.05 and diagnostics["alpha_max"] < 0.30,
        "geometry_grad_sustained": gradients_ok,
    }
    if args.variant == "gated_residual":
        checks["retention_not_collapsed"] = (
            diagnostics["retention_near_lower_fraction"] is not None
            and diagnostics["retention_near_upper_fraction"] is not None
            and diagnostics["retention_near_lower_fraction"] < 0.80
            and diagnostics["retention_near_upper_fraction"] < 0.80
        )
    report = {
        "schema_version": 2,
        "stage": (
            "formal_30k_intermediate"
            if args.step < 30000
            else "formal_30k_final"
            if args.step == 30000
            else "formal_100k_midterm"
        ),
        "seed": args.seed,
        "step": args.step,
        "variant": args.variant,
        "metrics": means,
        "diagnostics": diagnostics,
        "gradient_diagnostics": gradients,
        "checks": checks,
        "all_passed": all(checks.values()),
    }
    atomic_csv(Path(args.output_csv), rows)
    atomic_json(Path(args.output_json), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["all_passed"]:
        raise SystemExit("Formal paired checkpoint failed; matched continuation must stop")


if __name__ == "__main__":
    main()
