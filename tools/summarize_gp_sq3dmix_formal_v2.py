#!/usr/bin/env python3
"""Select and gate the two-seed Stage-C matched continuation.

The ``select`` command consumes the fixed-navtest-2k score artifacts produced
at 10k/20k/30k.  The ``full`` command consumes the paired full-navtest scores
for the two selected checkpoints and is the only code path that may issue a
permission with ``formal_100k_allowed=true``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.summarize_gp_sq3dmix_full_navtest_v2 import (
    MODES,
    SUBMETRICS,
    ci,
    delta,
    read_scores,
)
from tools.summarize_gp_sq3dmix_stage_b_v2 import (
    aggregate_seed_paired,
    bootstrap_ci,
    paired_delta,
    scores,
)


FORMAL_SEEDS = (20260826, 20260827)
FORMAL_STEPS = (10000, 20000, 30000)


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
    if not rows:
        raise RuntimeError("Cannot write an empty formal summary")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _current_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _formal_run(run_root: Path, variant: str, seed: int, step: int) -> Path:
    segment = "10k" if step == 10000 else "30k"
    return run_root / f"gp-sq3dmix-stage-c-{segment}-{variant}-{seed}"


def select(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    run_root = Path(args.run_root).resolve()
    stage_a = json.loads(Path(args.stage_a_decision).read_text(encoding="utf-8"))
    if stage_a.get("all_passed") is not True:
        raise RuntimeError("Stage A did not pass; Stage C selection is invalid")
    variant = str(stage_a["selected_variant"])
    candidates: dict[int, list[dict]] = {seed: [] for seed in FORMAL_SEEDS}
    rows: list[dict] = []
    for seed in FORMAL_SEEDS:
        for step in FORMAL_STEPS:
            step_root = root / f"seed{seed}" / f"step{step}"
            decision_path = step_root / "decision.json"
            if not decision_path.is_file():
                raise FileNotFoundError(decision_path)
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            if (
                int(decision.get("seed", -1)) != seed
                or int(decision.get("step", -1)) != step
                or decision.get("variant") != variant
            ):
                raise RuntimeError(f"Formal step decision binding mismatch: {decision_path}")
            per_metric = {"pdms": {}, "epdms": {}}
            means = {"pdms": {}, "epdms": {}}
            for metric in per_metric:
                for mode in MODES:
                    per_metric[metric][mode], means[metric][mode] = scores(
                        step_root / "scores" / mode / f"{metric}.csv"
                    )
            deltas = {
                metric: {
                    mode: paired_delta(per_metric[metric]["real"], per_metric[metric][mode])
                    for mode in MODES[1:]
                }
                for metric in per_metric
            }
            row = {
                "phase": "formal_30k_2k",
                "status": "complete",
                "seed": seed,
                "step": step,
                "all_passed": bool(decision.get("all_passed")),
                "real_pdms": means["pdms"]["real"],
                "control_pdms": means["pdms"]["control"],
                "real_minus_control_pdms": float(deltas["pdms"]["control"].mean()),
                "real_minus_hard_pdms": float(deltas["pdms"]["hard_shuffled"].mean()),
                "real_minus_spatial_pdms": float(deltas["pdms"]["spatial_shuffled"].mean()),
                "real_minus_control_epdms": float(deltas["epdms"]["control"].mean()),
            }
            rows.append(row)
            candidates[seed].append(
                {
                    "row": row,
                    "decision": decision,
                    "decision_path": str(decision_path),
                    "deltas": deltas,
                }
            )

    selected: dict[int, dict] = {}
    for seed, values in candidates.items():
        passing = [value for value in values if value["decision"].get("all_passed") is True]
        if not passing:
            selected[seed] = max(
                values,
                key=lambda value: (
                    value["row"]["real_minus_control_pdms"],
                    value["row"]["real_minus_hard_pdms"],
                    value["row"]["real_minus_spatial_pdms"],
                    -value["row"]["step"],
                ),
            )
        else:
            selected[seed] = max(
                passing,
                key=lambda value: (
                    value["row"]["real_minus_control_pdms"],
                    value["row"]["real_minus_hard_pdms"],
                    value["row"]["real_minus_spatial_pdms"],
                    -value["row"]["step"],
                ),
            )

    aggregate: dict[str, dict] = {}
    for mode in ("control", "hard_shuffled", "spatial_shuffled"):
        values = aggregate_seed_paired(
            [selected[seed]["deltas"]["pdms"][mode] for seed in FORMAL_SEEDS]
        )
        aggregate[mode] = {
            "mean": float(values.mean()),
            "bootstrap_ci": bootstrap_ci(values, 20260826 + len(aggregate)),
        }
    checks = {
        "both_seeds_passed": all(
            selected[seed]["decision"].get("all_passed") is True
            for seed in FORMAL_SEEDS
        ),
        "both_seeds_control_positive": all(
            selected[seed]["row"]["real_minus_control_pdms"] > 0.0
            for seed in FORMAL_SEEDS
        ),
        "both_seeds_hard_positive": all(
            selected[seed]["row"]["real_minus_hard_pdms"] > 0.0
            for seed in FORMAL_SEEDS
        ),
        "both_seeds_spatial_positive": all(
            selected[seed]["row"]["real_minus_spatial_pdms"] > 0.0
            for seed in FORMAL_SEEDS
        ),
        "both_seeds_epdms_nonnegative": all(
            selected[seed]["row"]["real_minus_control_epdms"] >= 0.0
            for seed in FORMAL_SEEDS
        ),
        "aggregate_control_mean": aggregate["control"]["mean"] >= 0.002,
        "aggregate_control_ci": aggregate["control"]["bootstrap_ci"][0] > 0.0,
        "aggregate_hard_positive": aggregate["hard_shuffled"]["mean"] > 0.0,
        "aggregate_spatial_positive": aggregate["spatial_shuffled"]["mean"] > 0.0,
    }
    report = {
        "schema_version": 2,
        "stage": "formal_30k_two_seed_selection",
        "code_commit": _current_commit(),
        "selected_variant": variant,
        "seeds": {
            str(seed): {
                "selected_step": int(selected[seed]["row"]["step"]),
                "selected_checkpoint": str(
                    _formal_run(run_root, variant, seed, selected[seed]["row"]["step"])
                    / "checkpoints"
                    / f"steps_{selected[seed]['row']['step']}_pytorch_model.pt"
                ),
                "selected_control_checkpoint": str(
                    _formal_run(run_root, "control", seed, selected[seed]["row"]["step"])
                    / "checkpoints"
                    / f"steps_{selected[seed]['row']['step']}_pytorch_model.pt"
                ),
                "selected_decision": selected[seed]["decision_path"],
                "metrics": selected[seed]["row"],
            }
            for seed in FORMAL_SEEDS
        },
        "aggregate_paired_pdms": aggregate,
        "checks": checks,
        "all_passed": all(checks.values()),
    }
    atomic_csv(Path(args.output_csv), rows)
    atomic_json(Path(args.output_json), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["all_passed"]:
        raise SystemExit("Formal 30k two-seed gate failed; full-navtest/100k forbidden")


def full(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    permission_before = json.loads(
        Path(args.permission_before).read_text(encoding="utf-8")
    )
    if selection.get("all_passed") is not True:
        raise RuntimeError("Formal two-seed 2k selection did not pass")
    if permission_before.get("formal_30k_allowed") is not True:
        raise RuntimeError("Input permission does not allow formal 30k")

    aggregate_arrays = {
        "pdms": {mode: [] for mode in MODES[1:]},
        "epdms": {"control": []},
    }
    submetric_arrays = {name: [] for name in SUBMETRICS}
    rows: list[dict] = []
    per_seed_checks: dict[str, dict] = {}
    for seed in FORMAL_SEEDS:
        score_root = root / f"seed{seed}" / "scores"
        per_metric = {
            metric: {
                mode: read_scores(score_root / mode / f"{metric}.csv")
                for mode in MODES
            }
            for metric in ("pdms", "epdms")
        }
        seed_deltas: dict[str, dict[str, np.ndarray]] = {"pdms": {}, "epdms": {}}
        for metric in ("pdms", "epdms"):
            for mode in MODES:
                values = np.asarray(
                    [item["score"] for item in per_metric[metric][mode].values()],
                    dtype=np.float64,
                )
                mode_delta = (
                    np.zeros_like(values)
                    if mode == "real"
                    else delta(per_metric[metric]["real"], per_metric[metric][mode], "score")
                )
                seed_deltas[metric][mode] = mode_delta
                bounds = [0.0, 0.0] if mode == "real" else ci(mode_delta, seed + len(rows))
                rows.append(
                    {
                        "phase": "formal_30k_full_navtest",
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
        per_seed_checks[str(seed)] = {
            "real_control_pdms_positive": seed_deltas["pdms"]["control"].mean() > 0.0,
            "real_hard_pdms_positive": seed_deltas["pdms"]["hard_shuffled"].mean() > 0.0,
            "real_spatial_pdms_positive": seed_deltas["pdms"]["spatial_shuffled"].mean() > 0.0,
            "real_control_epdms_nonnegative": seed_deltas["epdms"]["control"].mean() >= 0.0,
        }
        for name, column in SUBMETRICS.items():
            submetric_arrays[name].append(
                delta(per_metric["pdms"]["real"], per_metric["pdms"]["control"], column)
            )

    aggregate: dict[str, dict] = {}
    counter = 0
    for metric, modes in aggregate_arrays.items():
        for mode, arrays in modes.items():
            values = aggregate_seed_paired(arrays)
            aggregate[f"real_minus_{mode}_{metric}"] = {
                "mean": float(values.mean()),
                "bootstrap_ci": ci(values, 20260826 + counter),
            }
            counter += 1
    submetrics: dict[str, dict] = {}
    for name, arrays in submetric_arrays.items():
        values = aggregate_seed_paired(arrays)
        submetrics[name] = {
            "mean": float(values.mean()),
            "bootstrap_ci": ci(values, 20260826 + counter),
        }
        counter += 1
    checks = {
        "both_seeds_passed": all(all(value.values()) for value in per_seed_checks.values()),
        "real_control_pdms_mean": aggregate["real_minus_control_pdms"]["mean"] >= 0.002,
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
    passed = all(checks.values())
    commit = _current_commit()
    decision = {
        "schema_version": 2,
        "stage": "formal_30k_full_navtest",
        "code_commit": commit,
        "selection": str(Path(args.selection).resolve()),
        "per_seed_checks": per_seed_checks,
        "aggregate": aggregate,
        "safety_submetrics": submetrics,
        "checks": checks,
        "all_passed": passed,
    }
    permission = {
        "schema_version": 2,
        "status": "complete",
        "code_commit": commit,
        "stage_a_passed": True,
        "stage_b_two_seed_passed": True,
        "stage_b_full_navtest_passed": True,
        "formal_30k_allowed": True,
        "formal_30k_two_seed_passed": bool(selection["all_passed"]),
        "formal_30k_full_navtest_passed": passed,
        "formal_100k_allowed": passed,
    }
    atomic_csv(Path(args.output_csv), rows)
    atomic_json(Path(args.output_json), decision)
    atomic_json(Path(args.permission_json), permission)
    print(json.dumps({"decision": decision, "permission": permission}, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit("Formal full-navtest failed; 100k extension remains forbidden")


def midterm(args: argparse.Namespace) -> None:
    """Apply the mandatory 50k early-stop gate to both matched seed pairs."""

    root = Path(args.root).resolve()
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    permission = json.loads(Path(args.permission).read_text(encoding="utf-8"))
    if selection.get("all_passed") is not True:
        raise RuntimeError("Formal 30k selection did not pass")
    if permission.get("formal_100k_allowed") is not True:
        raise RuntimeError("formal_100k_allowed=false; 50k checkpoint is invalid")
    old_control = float(selection["aggregate_paired_pdms"]["control"]["mean"])
    old_hard = float(selection["aggregate_paired_pdms"]["hard_shuffled"]["mean"])
    old_spatial = float(selection["aggregate_paired_pdms"]["spatial_shuffled"]["mean"])
    new_deltas: dict[str, list[np.ndarray]] = {
        "control": [],
        "hard_shuffled": [],
        "spatial_shuffled": [],
    }
    per_seed = {}
    residual_improvements = []
    for seed in FORMAL_SEEDS:
        step_root = root / f"seed{seed}" / "step50000"
        decision = json.loads((step_root / "decision.json").read_text(encoding="utf-8"))
        if int(decision.get("seed", -1)) != seed or int(decision.get("step", -1)) != 50000:
            raise RuntimeError(f"50k decision binding mismatch for seed {seed}")
        for mode in new_deltas:
            real, _ = scores(step_root / "scores" / "real" / "pdms.csv")
            other, _ = scores(step_root / "scores" / mode / "pdms.csv")
            new_deltas[mode].append(paired_delta(real, other))
        old_decision_path = Path(selection["seeds"][str(seed)]["selected_decision"])
        old_decision = json.loads(old_decision_path.read_text(encoding="utf-8"))
        old_diag = old_decision["diagnostics"]
        new_diag = decision["diagnostics"]
        residual_improved = (
            float(new_diag["residual_p95"]) < float(old_diag["residual_p95"])
            or float(new_diag["residual_max"]) < float(old_diag["residual_max"])
        )
        residual_improvements.append(residual_improved)
        per_seed[str(seed)] = {
            "decision": str(step_root / "decision.json"),
            "all_passed": bool(decision.get("all_passed")),
            "residual_improved": residual_improved,
        }
    aggregate = {}
    for mode, values in new_deltas.items():
        paired = aggregate_seed_paired(values)
        aggregate[mode] = {
            "mean": float(paired.mean()),
            "bootstrap_ci": bootstrap_ci(paired, 20260850 + len(aggregate)),
        }
    performance_not_declined = aggregate["control"]["mean"] >= old_control
    structural_improved = (
        aggregate["hard_shuffled"]["mean"] > old_hard
        and aggregate["spatial_shuffled"]["mean"] > old_spatial
        and any(residual_improvements)
    )
    checks = {
        "both_50k_seed_gates_passed": all(value["all_passed"] for value in per_seed.values()),
        "no_unexplained_30k_to_50k_decline": performance_not_declined or structural_improved,
        "aggregate_control_positive": aggregate["control"]["mean"] > 0.0,
        "aggregate_hard_positive": aggregate["hard_shuffled"]["mean"] > 0.0,
        "aggregate_spatial_positive": aggregate["spatial_shuffled"]["mean"] > 0.0,
    }
    report = {
        "schema_version": 2,
        "stage": "formal_100k_50k_midterm",
        "code_commit": _current_commit(),
        "selection": str(Path(args.selection).resolve()),
        "permission": str(Path(args.permission).resolve()),
        "old_30k_aggregate_paired_pdms": {
            "control": old_control,
            "hard_shuffled": old_hard,
            "spatial_shuffled": old_spatial,
        },
        "new_50k_aggregate_paired_pdms": aggregate,
        "performance_not_declined": performance_not_declined,
        "structural_improved": structural_improved,
        "per_seed": per_seed,
        "checks": checks,
        "all_passed": all(checks.values()),
    }
    atomic_json(Path(args.output_json), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["all_passed"]:
        raise SystemExit("50k midterm failed; both GP/control continuations must stop")


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    selection = commands.add_parser("select")
    selection.add_argument("--root", required=True)
    selection.add_argument("--run-root", required=True)
    selection.add_argument("--stage-a-decision", required=True)
    selection.add_argument("--output-json", required=True)
    selection.add_argument("--output-csv", required=True)
    selection.set_defaults(function=select)
    full_parser = commands.add_parser("full")
    full_parser.add_argument("--root", required=True)
    full_parser.add_argument("--selection", required=True)
    full_parser.add_argument("--permission-before", required=True)
    full_parser.add_argument("--output-json", required=True)
    full_parser.add_argument("--output-csv", required=True)
    full_parser.add_argument("--permission-json", required=True)
    full_parser.set_defaults(function=full)
    middle = commands.add_parser("midterm")
    middle.add_argument("--root", required=True)
    middle.add_argument("--selection", required=True)
    middle.add_argument("--permission", required=True)
    middle.add_argument("--output-json", required=True)
    middle.set_defaults(function=midterm)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
