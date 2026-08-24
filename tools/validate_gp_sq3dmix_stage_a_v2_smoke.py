#!/usr/bin/env python3
"""Validate non-negotiable Stage-A-v2 smoke invariants from runtime records."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def metric(record: dict, suffix: str, default=float("nan")) -> float:
    for key, value in record.items():
        if key == suffix or key.endswith("/" + suffix):
            return float(value)
    return float(default)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--variant", choices=("projected_residual", "gated_residual"), required=True)
    parser.add_argument("--evaluation-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    diagnostics_path = run_dir / "gp_sq3dmix_diagnostics.jsonl"
    evaluation_path = Path(args.evaluation_report).resolve()
    output = Path(args.output).resolve()
    for path in (diagnostics_path, evaluation_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(output)
    records = [
        json.loads(line)
        for line in diagnostics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_step = {int(record["step"]): record for record in records}
    if 1 not in by_step or 10 not in by_step:
        raise RuntimeError("Smoke diagnostics must contain optimizer steps 1 and 10")
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    step1_reader = metric(by_step[1], "gp_sq3dmix/reader_grad_norm")
    step10_adapter = metric(by_step[10], "gp_sq3dmix/adapter_grad_norm")
    step10_reader = metric(by_step[10], "gp_sq3dmix/reader_grad_norm")
    step10_gate = metric(by_step[10], "gp_sq3dmix/gate_grad_norm")
    slot_values = [
        metric(record, "gp_sq3dmix/slot_mean_identity_max_abs")
        for record in records
    ]
    fixed_points = [
        metric(record, "gp_sq3dmix/spatial_permutation_fixed_point_count")
        for record in records
    ]
    shared_flow_counts = [
        metric(record, "gp_sq3dmix/shared_flow_state_condition_count")
        for record in records
    ]
    shared_dropout_counts = [
        metric(record, "gp_sq3dmix/shared_dropout_stream_condition_count")
        for record in records
    ]
    loss_values = [
        float(value)
        for record in records
        for key, value in record.items()
        if key.startswith("weighted_loss/")
    ]
    checks = {
        "step_1_up_projection_grad_nonzero": math.isfinite(step1_reader)
        and step1_reader > 0.0,
        "step_10_adapter_grad_nonzero": math.isfinite(step10_adapter)
        and step10_adapter > 0.0,
        "step_10_reader_grad_nonzero": math.isfinite(step10_reader)
        and step10_reader > 0.0,
        "slot_mean_identity": bool(slot_values)
        and all(math.isfinite(value) and value < 1e-6 for value in slot_values),
        "all_losses_finite": bool(loss_values)
        and all(math.isfinite(value) for value in loss_values),
        "spatial_derangement": bool(fixed_points)
        and all(value == 0.0 for value in fixed_points),
        "real_hard_spatial_share_flow_state": bool(shared_flow_counts)
        and all(value == 4.0 for value in shared_flow_counts),
        "real_hard_spatial_share_dropout_stream": bool(shared_dropout_counts)
        and all(value == 4.0 for value in shared_dropout_counts),
        "checkpoint_strict_reload": evaluation.get("sample_count") == 128,
        "paired_flow_samples_present": set(evaluation.get("loss_means", {}))
        == {"base", "real", "hard", "spatial"},
    }
    if args.variant == "gated_residual":
        checks["step_10_gate_grad_nonzero"] = math.isfinite(step10_gate) and step10_gate > 0.0
    report = {
        "schema_version": 2,
        "stage": "stage_a_v2_smoke",
        "variant": args.variant,
        "run_dir": str(run_dir),
        "evaluation_report": str(evaluation_path),
        "checks": checks,
        "all_passed": all(checks.values()),
        "observed": {
            "step_1_reader_grad_norm": step1_reader,
            "step_10_adapter_grad_norm": step10_adapter,
            "step_10_reader_grad_norm": step10_reader,
            "step_10_gate_grad_norm": step10_gate,
            "slot_mean_identity_max_abs": max(slot_values, default=float("nan")),
            "spatial_fixed_point_max": max(fixed_points, default=float("nan")),
            "shared_flow_state_condition_count": min(
                shared_flow_counts, default=float("nan")
            ),
            "shared_dropout_stream_condition_count": min(
                shared_dropout_counts, default=float("nan")
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["all_passed"]:
        raise SystemExit("Stage-A-v2 smoke invariants failed")


if __name__ == "__main__":
    main()
