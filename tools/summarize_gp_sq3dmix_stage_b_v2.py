#!/usr/bin/env python3
"""Summarize paired two-seed Stage-B-v2 PDMS/EPDMS gates."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

import numpy as np


MODES = ("real", "hard_shuffled", "spatial_shuffled", "slot_mean", "zero", "control")
SEEDS = (20260824, 20260825)


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
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def scores(path: Path) -> tuple[dict[str, float], float]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    per_token = {
        row["token"]: float(row["score"])
        for row in rows
        if row.get("score")
        and row.get("token") not in {"average", "average_all_frames"}
    }
    if not per_token:
        raise RuntimeError(f"No per-token scores in {path}")
    aggregate = float(np.mean(list(per_token.values())))
    return per_token, aggregate


def paired_delta(first: dict[str, float], second: dict[str, float]) -> np.ndarray:
    if set(first) != set(second):
        raise RuntimeError("Paired scoring token sets differ")
    return np.asarray(
        [first[token] - second[token] for token in sorted(first)], dtype=np.float64
    )


def bootstrap_ci(values: np.ndarray, seed: int, draws: int = 10000) -> list[float]:
    generator = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 128):
        size = min(128, draws - start)
        indices = generator.integers(0, len(values), size=(size, len(values)))
        means[start : start + size] = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, (0.025, 0.975))]


def aggregate_seed_paired(values: list[np.ndarray]) -> np.ndarray:
    """Average matched seed deltas per scene before resampling scenes.

    The two policy seeds evaluate the same immutable token set.  Concatenating
    them would incorrectly treat two measurements of one scene as independent
    observations and make the interval too narrow.
    """

    arrays = [np.asarray(value, dtype=np.float64) for value in values]
    if not arrays or len({array.shape for array in arrays}) != 1:
        raise ValueError("seed-paired arrays must have one identical shape")
    return np.stack(arrays, axis=0).mean(axis=0)


def inference_diagnostics(step_root: Path) -> dict:
    paths = list(
        (step_root / "predictions" / "real").glob(
            "*/test/gp_diagnostics/*.json"
        )
    )
    if not paths:
        raise RuntimeError(f"No real-mode GP diagnostics under {step_root}")
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    ratios = np.asarray([row["residual_action_ratio"] for row in rows], dtype=np.float64)
    alpha = np.asarray([row["alpha"] for row in rows], dtype=np.float64)
    retention_rows = [row.get("retention") for row in rows]
    return {
        "residual_p95": float(np.quantile(ratios, 0.95)),
        "residual_max": float(ratios.max()),
        "alpha_min": float(alpha.min()),
        "alpha_max": float(alpha.max()),
        "retention_near_lower_fraction": (
            float(np.mean([row["near_lower_fraction"] for row in retention_rows]))
            if all(row is not None for row in retention_rows)
            else None
        ),
        "retention_near_upper_fraction": (
            float(np.mean([row["near_upper_fraction"] for row in retention_rows]))
            if all(row is not None for row in retention_rows)
            else None
        ),
    }


def gradient_active(run_dir: Path, variant: str) -> tuple[bool, dict]:
    path = run_dir / "gp_sq3dmix_diagnostics.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    names = ["adapter_grad_norm", "reader_grad_norm"]
    if variant == "gated_residual":
        names.append("gate_grad_norm")
    summary = {}
    for name in names:
        values = np.asarray(
            [
                float(value)
                for row in rows
                for key, value in row.items()
                if key.endswith(name)
            ],
            dtype=np.float64,
        )
        summary[name] = {
            "count": len(values),
            "positive_fraction": float((values > 0).mean()) if len(values) else 0.0,
            "mean": float(values.mean()) if len(values) else 0.0,
        }
    return all(value["positive_fraction"] >= 0.95 for value in summary.values()), summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--variant", choices=("projected_residual", "gated_residual"), required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--interventions-csv", required=True)
    parser.add_argument("--decision-json", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    run_root = Path(args.run_root).resolve()
    all_rows = []
    seed_best = {}
    intervention_rows = []
    for seed in SEEDS:
        candidates = []
        for step in (2000, 4000, 6000, 8000, 10000):
            score_root = root / f"seed{seed}" / f"step{step}" / "scores"
            per_metric = {"pdms": {}, "epdms": {}}
            means = {"pdms": {}, "epdms": {}}
            for metric in per_metric:
                for mode in MODES:
                    per_metric[metric][mode], means[metric][mode] = scores(
                        score_root / mode / f"{metric}.csv"
                    )
            deltas = {
                metric: {
                    mode: paired_delta(per_metric[metric]["real"], per_metric[metric][mode])
                    for mode in MODES[1:]
                }
                for metric in per_metric
            }
            row = {
                "phase": "stage_b_2k",
                "status": "complete",
                "seed": seed,
                "step": step,
                **{
                    f"{mode}_{metric}": means[metric][mode]
                    for metric in means
                    for mode in MODES
                },
                **{
                    f"real_minus_{mode}_{metric}": float(deltas[metric][mode].mean())
                    for metric in deltas
                    for mode in MODES[1:]
                },
            }
            candidates.append((row, deltas, per_metric))
            all_rows.append(row)
        best = max(
            candidates,
            key=lambda value: (
                value[0]["real_pdms"],
                value[0]["real_minus_hard_shuffled_pdms"],
                -abs(value[0]["real_minus_control_pdms"]),
            ),
        )
        row, deltas, per_metric = best
        step_root = root / f"seed{seed}" / f"step{row['step']}"
        diagnostics = inference_diagnostics(step_root)
        gp_run = run_root / f"gp-sq3dmix-stage-b-{args.variant}-{seed}"
        gradients_ok, gradients = gradient_active(gp_run, args.variant)
        checks = {
            "real_control_pdms_positive": row["real_minus_control_pdms"] > 0.0,
            "real_hard_pdms_positive": row["real_minus_hard_shuffled_pdms"] > 0.0,
            "real_spatial_pdms_positive": row["real_minus_spatial_shuffled_pdms"] > 0.0,
            "real_slot_mean_pdms_positive": row["real_minus_slot_mean_pdms"] > 0.0,
            "real_control_epdms_nonnegative_margin": row["real_minus_control_epdms"] >= -0.001,
            "residual_p95": diagnostics["residual_p95"] <= 0.30,
            "residual_max": diagnostics["residual_max"] <= 0.60,
            "alpha_not_at_boundary": diagnostics["alpha_min"] > 0.05
            and diagnostics["alpha_max"] < 0.30,
            "geometry_grad_sustained": gradients_ok,
        }
        if args.variant == "gated_residual":
            checks["retention_not_at_boundary"] = (
                diagnostics["retention_near_lower_fraction"] is not None
                and diagnostics["retention_near_upper_fraction"] is not None
                and diagnostics["retention_near_lower_fraction"] < 0.80
                and diagnostics["retention_near_upper_fraction"] < 0.80
            )
        seed_best[seed] = {
            "selected_step": row["step"],
            "selected_checkpoint": str(
                gp_run / "checkpoints" / f"steps_{row['step']}_pytorch_model.pt"
            ),
            "metrics": row,
            "diagnostics": diagnostics,
            "gradient_diagnostics": gradients,
            "checks": checks,
            "all_passed": all(checks.values()),
            "deltas": deltas,
            "per_metric": per_metric,
        }
        for metric in ("pdms", "epdms"):
            for mode in MODES:
                delta = (
                    np.zeros_like(next(iter(deltas[metric].values())))
                    if mode == "real"
                    else deltas[metric][mode]
                )
                ci = [0.0, 0.0] if mode == "real" else bootstrap_ci(delta, seed + len(intervention_rows))
                intervention_rows.append(
                    {
                        "phase": "stage_b_2k",
                        "status": "complete",
                        "seed": seed,
                        "selected_step": row["step"],
                        "mode": mode,
                        "metric": metric,
                        "score": row[f"{mode}_{metric}"],
                        "real_minus_mode_delta": float(delta.mean()),
                        "bootstrap_ci_lower": ci[0],
                        "bootstrap_ci_upper": ci[1],
                    }
                )

    aggregate = {}
    for mode in ("control", "hard_shuffled", "spatial_shuffled"):
        values = aggregate_seed_paired(
            [seed_best[seed]["deltas"]["pdms"][mode] for seed in SEEDS]
        )
        aggregate[mode] = {
            "mean": float(values.mean()),
            "bootstrap_ci": bootstrap_ci(values, 20260824 + len(aggregate)),
        }
    aggregate_checks = {
        "both_seeds_passed": all(seed_best[seed]["all_passed"] for seed in SEEDS),
        "both_seeds_control_positive": all(
            seed_best[seed]["metrics"]["real_minus_control_pdms"] > 0.0
            for seed in SEEDS
        ),
        "aggregate_control_mean": aggregate["control"]["mean"] >= 0.0015,
        "aggregate_control_ci": aggregate["control"]["bootstrap_ci"][0] >= -0.0005,
        "aggregate_hard_ci": aggregate["hard_shuffled"]["bootstrap_ci"][0] > 0.0,
        "aggregate_spatial_ci": aggregate["spatial_shuffled"]["bootstrap_ci"][0] > 0.0,
    }
    decision = {
        "schema_version": 2,
        "stage": "stage_b_multiseed_2k",
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "selected_variant": args.variant,
        "seeds": {
            str(seed): {
                key: value
                for key, value in seed_best[seed].items()
                if key not in {"deltas", "per_metric"}
            }
            for seed in SEEDS
        },
        "aggregate_paired_pdms": aggregate,
        "checks": aggregate_checks,
        "all_passed": all(aggregate_checks.values()),
    }
    atomic_csv(Path(args.output_csv), all_rows)
    atomic_csv(Path(args.interventions_csv), intervention_rows)
    atomic_json(Path(args.decision_json), decision)
    print(json.dumps(decision, indent=2, sort_keys=True))
    if not decision["all_passed"]:
        raise SystemExit("Stage B two-seed 2k gates failed")


if __name__ == "__main__":
    main()
