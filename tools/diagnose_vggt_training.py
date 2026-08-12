#!/usr/bin/env python3
"""Summarize measured VGGT alignment and planner-use signals from a run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median


SIGNALS = (
    "vggt/alignment_cosine_all",
    "vggt/alignment_cosine_special",
    "vggt/alignment_cosine_spatial",
    "vggt/alignment_distributed_retrieval_top1",
    "vggt/alignment_distributed_retrieval_top5",
    "vggt/alignment_correct_minus_slot_mean",
    "vggt/alignment_correct_minus_shuffled",
    "vggt/alignment_scene_residual_cosine",
    "vggt/alignment_student_teacher_scene_variance_ratio",
    "vggt/alignment_student_std",
    "vggt/geometry_x_over_z_mae",
    "vggt/geometry_y_over_z_mae",
    "vggt/geometry_log_depth_mae",
    "vggt/aux_plan_ade",
    "vggt/aux_plan_fde",
    "vggt/planning_context_grad_norm",
    "vggt/geometry_adapter_grad_norm",
    "vggt/waypoint_reader_grad_norm",
    "vggt/geometry_probe_grad_norm",
    "vggt/aux_plan_head_grad_norm",
    "vggt/attention_entropy",
    "vggt/attention_waypoint_js_divergence",
    "vggt/attention_front_view_mass",
    "vggt/attention_left_view_mass",
    "vggt/attention_right_view_mass",
    "vggt/intervention_zero_minus_real",
    "vggt/intervention_shuffled_minus_real",
    "vggt/intervention_slot_mean_minus_real",
    "vggt/intervention_zero_trajectory_l2",
    "vggt/intervention_shuffled_trajectory_l2",
    "vggt/intervention_slot_mean_trajectory_l2",
    "action_dit_loss",
)


def finite_values(records: list[dict], key: str) -> list[float]:
    values = []
    for record in records:
        value = record.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def main(args: argparse.Namespace) -> None:
    path = Path(args.path)
    if path.is_dir():
        path = path / "vggt_diagnostics.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing VGGT diagnostic log: {path}")
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise RuntimeError(f"VGGT diagnostic log is empty: {path}")
    window = records[-args.window :]
    summary = {
        "path": str(path.resolve()),
        "records": len(records),
        "window": len(window),
        "last_step": records[-1].get("step"),
        "signals": {},
    }
    for key in SIGNALS:
        values = finite_values(window, key)
        summary["signals"][key] = (
            {"median": median(values), "last": values[-1], "count": len(values)}
            if values
            else "MISSING"
        )

    context_grad = finite_values(window, "vggt/planning_context_grad_norm")
    reader_grad = finite_values(window, "vggt/waypoint_reader_grad_norm")
    geometry_error = finite_values(window, "vggt/geometry_log_depth_mae")
    aux_ade = finite_values(window, "vggt/aux_plan_ade")
    shuffled_change = finite_values(window, "vggt/intervention_shuffled_trajectory_l2")
    student_std = finite_values(window, "vggt/alignment_student_std")
    summary["checks"] = {
        "planner_context_receives_action_gradient": (
            median(context_grad) > args.gradient_epsilon if context_grad else "MISSING"
        ),
        "waypoint_reader_receives_gradient": (
            median(reader_grad) > args.gradient_epsilon if reader_grad else "MISSING"
        ),
        "student_queries_not_collapsed": (
            median(student_std) > args.std_epsilon if student_std else "MISSING"
        ),
        "physical_geometry_is_finite": (
            math.isfinite(median(geometry_error)) if geometry_error else "MISSING"
        ),
        "auxiliary_planning_is_finite": (
            math.isfinite(median(aux_ade)) if aux_ade else "MISSING"
        ),
        "trajectory_changes_when_scene_memory_is_shuffled": (
            median(shuffled_change) > args.intervention_epsilon
            if shuffled_change
            else "MISSING"
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Run directory or vggt_diagnostics.jsonl")
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--gradient-epsilon", type=float, default=1e-10)
    parser.add_argument("--std-epsilon", type=float, default=1e-4)
    parser.add_argument("--intervention-epsilon", type=float, default=1e-5)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
