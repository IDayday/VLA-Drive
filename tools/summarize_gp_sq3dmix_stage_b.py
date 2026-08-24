#!/usr/bin/env python3
"""Joint checkpoint selection and Stage-B go/no-go decision."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np


MODES = ("real", "zero", "shuffled", "slot_mean", "control")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(path: Path, rows: list[dict]) -> None:
    if path.exists():
        with path.open(newline="", encoding="utf-8") as stream:
            existing = list(csv.DictReader(stream))
        if len(existing) != 1 or existing[0].get("status") != "not_run":
            raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def scores(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    values = {
        row["token"]: float(row["score"])
        for row in rows
        if row.get("score") and row.get("token") not in {"average", "average_all_frames"}
    }
    aggregate_rows = [
        row
        for row in rows
        if row.get("token") in {"average", "average_all_frames"} and row.get("score")
    ]
    aggregate = float(aggregate_rows[-1]["score"]) if aggregate_rows else float(np.mean(list(values.values())))
    return values, aggregate


def ci(values: np.ndarray, rng: np.random.Generator, draws: int = 10000):
    means = []
    for _ in range(draws // 250):
        index = rng.integers(0, len(values), size=(250, len(values)))
        means.extend(values[index].mean(axis=1))
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--gp-run-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--interventions-csv", required=True)
    parser.add_argument("--decision-json", required=True)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()
    root = Path(args.root)
    rng = np.random.default_rng(args.seed)
    rows = []
    for step_dir in sorted(root.glob("step*"), key=lambda path: int(path.name[4:])):
        step = int(step_dir.name[4:])
        mode_pdms, mode_epdms = {}, {}
        pdms_per_token = {}
        for mode in MODES:
            pdms_per_token[mode], mode_pdms[mode] = scores(step_dir / "scores" / mode / "pdms.csv")
            _, mode_epdms[mode] = scores(step_dir / "scores" / mode / "epdms.csv")
        paired = {}
        for mode in MODES[1:]:
            common = sorted(set(pdms_per_token["real"]) & set(pdms_per_token[mode]))
            delta = np.asarray(
                [pdms_per_token["real"][token] - pdms_per_token[mode][token] for token in common]
            )
            paired[mode] = {"mean": float(delta.mean()), "ci": ci(delta, rng)}
        rows.append(
            {
                "phase": "stage_b",
                "status": "complete",
                "reason": "",
                "step": step,
                "sample_count": len(pdms_per_token["real"]),
                **{f"{mode}_pdms": mode_pdms[mode] for mode in MODES},
                **{f"{mode}_epdms": mode_epdms[mode] for mode in MODES},
                "real_shuffled_pdms_delta": paired["shuffled"]["mean"],
                "real_shuffled_ci_lower": paired["shuffled"]["ci"][0],
                "real_shuffled_ci_upper": paired["shuffled"]["ci"][1],
                "real_control_pdms_delta": paired["control"]["mean"],
                "real_control_ci_lower": paired["control"]["ci"][0],
                "real_control_ci_upper": paired["control"]["ci"][1],
            }
        )
    if not rows:
        raise RuntimeError("No Stage-B step results were found")
    best = max(
        rows,
        key=lambda row: (
            row["real_pdms"],
            row["real_shuffled_pdms_delta"],
            -abs(row["real_control_pdms_delta"]),
        ),
    )
    diagnostics_path = Path(args.gp_run_dir) / "gp_sq3dmix_diagnostics.jsonl"
    diagnostic_rows = [
        json.loads(line)
        for line in diagnostics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    def series(suffix):
        return np.asarray(
            [float(row[key]) for row in diagnostic_rows for key in row if key.endswith(suffix)],
            dtype=np.float64,
        )
    residual = series("residual_action_ratio")
    alpha = series("/alpha")
    lower = series("retention_near_lower_fraction")
    upper = series("retention_near_upper_fraction")
    gradients = {
        name: series(name)
        for name in ("adapter_grad_norm", "gate_grad_norm", "reader_grad_norm")
    }
    checks = {
        "deterministic_real_best": all(
            best["real_pdms"] > best[f"{mode}_pdms"]
            for mode in ("slot_mean", "zero", "shuffled")
        ),
        "positive_paired_pdms_vs_control": best["real_control_pdms_delta"] > 0.0,
        "control_ci_not_significantly_negative": best["real_control_ci_upper"] >= 0.0,
        "bounded_residual": len(residual) > 0 and 0.01 <= float(residual.mean()) <= 0.20,
        "geometry_gradients_sustained": all(
            len(values) > 0 and float((values > 0).mean()) >= 0.95
            for values in gradients.values()
        ),
        "no_gate_alpha_boundary_collapse": len(alpha) > 0
        and np.all((alpha >= 0.05) & (alpha <= 0.20))
        and len(lower) > 0 and float(lower.mean()) < 0.80
        and len(upper) > 0 and float(upper.mean()) < 0.80,
    }
    decision = {
        "schema_version": 1,
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "selected_step": best["step"],
        "selection_order": ["real_pdms", "real_shuffled_paired_delta", "fidelity_to_control"],
        "checks": checks,
        "all_passed": all(checks.values()),
        "recommend_full_training": all(checks.values()),
    }
    gp_run_dir = Path(args.gp_run_dir)
    selected_checkpoint = (
        gp_run_dir
        / "checkpoints"
        / f"steps_{best['step']}_pytorch_model.pt"
    )
    config_path = gp_run_dir / "config.yaml"
    if not selected_checkpoint.is_file() or not config_path.is_file():
        raise FileNotFoundError(selected_checkpoint if not selected_checkpoint.is_file() else config_path)
    decision.update(
        {
            "selected_checkpoint": str(selected_checkpoint.resolve()),
            "selected_checkpoint_sha256": sha256_file(selected_checkpoint),
            "resolved_config_sha256": sha256_file(config_path),
        }
    )
    output = Path(args.output_csv)
    interventions_output = Path(args.interventions_csv)
    decision_path = Path(args.decision_json)
    if decision_path.exists():
        raise FileExistsError(decision_path)
    for path in (output, interventions_output):
        if path.exists():
            with path.open(newline="", encoding="utf-8") as stream:
                existing = list(csv.DictReader(stream))
            if len(existing) != 1 or existing[0].get("status") != "not_run":
                raise FileExistsError(path)
    atomic_csv(output, rows)

    selected_root = root / f"step{best['step']}" / "scores"
    selected_pdms = {}
    selected_epdms = {}
    selected_pdms_mean = {}
    selected_epdms_mean = {}
    for mode in MODES:
        selected_pdms[mode], selected_pdms_mean[mode] = scores(
            selected_root / mode / "pdms.csv"
        )
        selected_epdms[mode], selected_epdms_mean[mode] = scores(
            selected_root / mode / "epdms.csv"
        )
    intervention_rows = []
    for mode in MODES:
        for metric, per_token, aggregates in (
            ("pdms", selected_pdms, selected_pdms_mean),
            ("epdms", selected_epdms, selected_epdms_mean),
        ):
            common = sorted(set(per_token["real"]) & set(per_token[mode]))
            delta = np.asarray(
                [per_token["real"][token] - per_token[mode][token] for token in common],
                dtype=np.float64,
            )
            lower, upper = ([0.0, 0.0] if mode == "real" else ci(delta, rng))
            intervention_rows.append(
                {
                    "phase": "stage_b",
                    "status": "complete",
                    "reason": "",
                    "selected_step": best["step"],
                    "mode": mode,
                    "metric": metric,
                    "sample_count": len(common),
                    "score": aggregates[mode],
                    "real_minus_mode_delta": float(delta.mean()),
                    "bootstrap_ci_lower": lower,
                    "bootstrap_ci_upper": upper,
                }
            )
    atomic_csv(interventions_output, intervention_rows)
    temporary = decision_path.with_name(decision_path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, decision_path)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
