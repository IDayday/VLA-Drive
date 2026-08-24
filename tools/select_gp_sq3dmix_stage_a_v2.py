#!/usr/bin/env python3
"""Select Stage-A-v2 checkpoints and the projected/gated final variant."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def atomic_json(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_report(path: str | Path) -> dict:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "variant",
        "code_commit",
        "checkpoint",
        "checkpoint_sha256",
        "relative_real_minus_base",
        "relative_hard_real_gap",
        "relative_spatial_real_gap",
        "loss_means",
        "checks",
        "all_passed",
    }
    if not required.issubset(report):
        raise RuntimeError(f"Incomplete Stage-A-v2 report: {path}")
    return report


def checkpoint_score(report: dict) -> tuple:
    causal_floor = min(
        float(report["relative_hard_real_gap"]) / 0.05,
        float(report["relative_spatial_real_gap"]) / 0.02,
    )
    utility_penalty = max(float(report["relative_real_minus_base"]), 0.0)
    return (
        int(report["all_passed"] is True),
        causal_floor,
        -utility_penalty,
        -float(report["loss_means"]["real"]),
        str(report["checkpoint"]),
    )


def select_checkpoint(args: argparse.Namespace) -> None:
    reports = [load_report(path) for path in args.reports]
    variants = {report["variant"] for report in reports}
    commits = {report["code_commit"] for report in reports}
    if len(variants) != 1 or len(commits) != 1:
        raise RuntimeError("Model-selection reports must share variant and commit")
    selected = max(reports, key=checkpoint_score)
    output = {
        "schema_version": 2,
        "stage": "stage_a_v2_model_selection",
        "variant": selected["variant"],
        "code_commit": selected["code_commit"],
        "candidate_count": len(reports),
        "selected_checkpoint": selected["checkpoint"],
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "selected_report": args.reports[reports.index(selected)],
        "selected_model_selection_all_passed": selected["all_passed"],
        "selection_rule": [
            "all_passed",
            "max_min_normalized_hard_spatial_gap",
            "min_positive_utility_regression",
            "min_real_loss",
            "checkpoint_path_tiebreak",
        ],
    }
    atomic_json(Path(args.output), output)
    print(json.dumps(output, indent=2, sort_keys=True))


def decide_variant(args: argparse.Namespace) -> None:
    projected = load_report(args.projected_report)
    gated = load_report(args.gated_report)
    if projected["variant"] != "projected_residual" or gated["variant"] != "gated_residual":
        raise RuntimeError("Final reports do not match projected/gated variants")
    if projected["code_commit"] != gated["code_commit"]:
        raise RuntimeError("Final variant reports use different code commits")
    passing = [
        report for report in (projected, gated) if report["all_passed"] is True
    ]
    scene = gated.get("scene_conditioning_diagnostic", {})
    scene_ci = scene.get("scene_shuffled_minus_real_loss_bootstrap_ci") or [
        float("-inf"),
        float("inf"),
    ]
    gated_selection_checks = {
        "real_loss_within_projected_1p005": float(gated["loss_means"]["real"])
        <= 1.005 * float(projected["loss_means"]["real"]),
        "hard_gap_not_lower_than_projected": float(gated["relative_hard_real_gap"])
        >= float(projected["relative_hard_real_gap"]),
        "spatial_gap_not_lower_than_projected": float(
            gated["relative_spatial_real_gap"]
        )
        >= float(projected["relative_spatial_real_gap"]),
        "scene_shuffled_loss_higher": float(
            scene.get("scene_shuffled_minus_real_loss", float("-inf"))
        )
        > 0.0
        and float(scene_ci[0]) > 0.0,
    }
    scene_useful = all(gated_selection_checks.values())
    if not passing:
        selected = None
        reason = "both variants failed one or more immutable final gates"
    elif len(passing) == 1:
        selected = passing[0]
        reason = "only one variant passed every immutable final gate"
    elif scene_useful:
        selected = gated
        reason = "both passed and gated met every scene-conditioning selection rule"
    else:
        selected = projected
        reason = "both passed but gated lacked causal scene-conditioning evidence; selected simpler projected residual"
    output = {
        "schema_version": 2,
        "stage": "stage_a_v2_final_decision",
        "code_commit": projected["code_commit"],
        "projected_report": str(Path(args.projected_report).resolve()),
        "gated_report": str(Path(args.gated_report).resolve()),
        "projected_all_passed": projected["all_passed"],
        "gated_all_passed": gated["all_passed"],
        "gated_selection_checks": gated_selection_checks,
        "scene_conditioning_useful": scene_useful,
        "selected_variant": selected["variant"] if selected else None,
        "selected_checkpoint": selected["checkpoint"] if selected else None,
        "selected_checkpoint_sha256": selected["checkpoint_sha256"] if selected else None,
        "all_passed": selected is not None,
        "reason": reason,
    }
    atomic_json(Path(args.output), output)
    print(json.dumps(output, indent=2, sort_keys=True))
    if selected is None:
        raise SystemExit("Stage A NO-GO: both variants failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("select-checkpoint")
    select.add_argument("--reports", nargs="+", required=True)
    select.add_argument("--output", required=True)
    select.set_defaults(function=select_checkpoint)
    decide = subparsers.add_parser("decide-variant")
    decide.add_argument("--projected-report", required=True)
    decide.add_argument("--gated-report", required=True)
    decide.add_argument("--output", required=True)
    decide.set_defaults(function=decide_variant)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
