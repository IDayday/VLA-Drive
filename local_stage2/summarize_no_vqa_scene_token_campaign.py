#!/usr/bin/env python3
"""Summarize validation promotion and strict Navtest scorer coverage."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List


def _load(path: Path) -> Dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def summarize(args: argparse.Namespace) -> Dict[str, object]:
    promotion = _load(args.promotion_manifest)
    baseline = _load(args.baseline_audit / "summary.json")
    promoted = list(promotion.get("promoted", []))
    expected = {str(row["artifact_sha256"]): row for row in promoted}
    if len(expected) != len(promoted):
        raise RuntimeError("Promotion manifest contains duplicate artifact SHA256")
    rows: List[Dict[str, object]] = []
    observed: set[str] = set()
    for directory in sorted(args.navtest_root.iterdir()):
        if not directory.is_dir() or not (directory / "summary.json").is_file():
            continue
        summary = _load(directory / "summary.json")
        source_sha = str(summary.get("source_ranker_artifact_sha256"))
        if source_sha not in expected:
            raise RuntimeError(
                f"Navtest result was not validation-promoted: {directory}"
            )
        if source_sha in observed:
            raise RuntimeError(f"Duplicate Navtest result for {source_sha}")
        observed.add(source_sha)
        hard_expected = {
            "scene_count": 12_146,
            "valid_scene_count": 12_146,
            "invalid_scene_count": 0,
            "log_count": 136,
            "candidate_count": 64,
            "precision": 32,
        }
        for key, value in hard_expected.items():
            if summary.get(key) != value:
                raise RuntimeError(
                    f"Strict Navtest gate failed for {directory}/{key}: "
                    f"{summary.get(key)!r} != {value!r}"
                )
        for key in (
            "inference_inputs_only",
            "official_candidate_matrix_joined_after_selection",
        ):
            if summary.get(key) is not True:
                raise RuntimeError(f"Inference-boundary gate failed: {directory}/{key}")
        for key in (
            "future_target_present_during_inference",
            "official_score_input_present_during_inference",
            "external_model_representation_or_weight_used",
            "drivor_representation_or_weight_used",
        ):
            if summary.get(key) is not False:
                raise RuntimeError(f"Forbidden input gate failed: {directory}/{key}")
        parity = float(summary.get("base_score_cache_parity_max_abs", float("inf")))
        if parity > 1.0e-8:
            raise RuntimeError(f"Base cache parity failed for {directory}: {parity}")
        metrics = dict(summary["metrics"])
        comparison = dict(summary["comparison_to_public_base"])
        validation = expected[source_sha]
        selected = float(metrics["selected_pdms"])
        base_selected = float(comparison["public_selected_pdms"])
        row = {
            "name": str(validation["name"]),
            "artifact_sha256": source_sha,
            "score_mode": str(validation["score_mode"]),
            "validation_selected_pdms": float(
                validation["validation_selected_pdms"]
            ),
            "validation_delta": float(validation["validation_delta"]),
            "validation_ci_low": float(
                validation["validation_delta_log_bootstrap_95ci"][0]
            ),
            "validation_ci_high": float(
                validation["validation_delta_log_bootstrap_95ci"][1]
            ),
            "navtest_selected_pdms": selected,
            "navtest_base_pdms": base_selected,
            "navtest_delta": float(comparison["selected_delta"]),
            "navtest_ci_low": float(
                comparison["selected_delta_log_bootstrap_95ci"][0]
            ),
            "navtest_ci_high": float(
                comparison["selected_delta_log_bootstrap_95ci"][1]
            ),
            "best_of_64_pdms": float(
                summary["offline_oracle_candidate_bank_upper_bound"]
            ),
            "scorer_regret": float(metrics["scorer_regret"]),
            "selected_noc": float(metrics["selected_no_at_fault_collisions"]),
            "selected_dac": float(metrics["selected_drivable_area_compliance"]),
            "selected_ddc": float(metrics["selected_driving_direction_compliance"]),
            "selected_ttc": float(
                metrics["selected_time_to_collision_within_bound"]
            ),
            "selected_progress": float(metrics["selected_ego_progress"]),
            "selected_comfort": float(metrics["selected_comfort"]),
            "exceeds_0_93": selected > 0.93,
            "navtest_result_dir": str(directory.resolve()),
        }
        if args.parity_root is not None:
            parity_path = args.parity_root / f"{validation['name']}.json"
            parity = _load(parity_path)
            if not bool(parity.get("passed")) or int(parity.get("scene_count", 0)) < 4:
                raise RuntimeError(f"Online/cache parity failed: {parity_path}")
            if float(parity.get("proposal_max_abs", float("inf"))) > 1.0e-6:
                raise RuntimeError(f"Online proposal parity failed: {parity_path}")
            if float(parity.get("score_max_abs", float("inf"))) > 1.0e-6:
                raise RuntimeError(f"Online score parity failed: {parity_path}")
            row["online_cache_parity_passed"] = True
            row["online_cache_score_max_abs"] = float(parity["score_max_abs"])
            row["online_cache_proposal_max_abs"] = float(
                parity["proposal_max_abs"]
            )
        rows.append(row)
    missing = sorted(set(expected) - observed)
    extra = sorted(observed - set(expected))
    if missing or extra:
        raise RuntimeError(
            f"Promoted/Navtest artifact coverage mismatch: missing={missing}, extra={extra}"
        )
    rows.sort(key=lambda value: float(value["navtest_selected_pdms"]), reverse=True)
    primary = [
        row
        for row in rows
        if str(row["name"]).startswith("primary_hybrid_actor050_seed")
    ]
    primary_scores = [float(row["navtest_selected_pdms"]) for row in primary]
    primary_deltas = [float(row["navtest_delta"]) for row in primary]
    return {
        "status": "PASS",
        "baseline_checkpoint": baseline.get("checkpoint"),
        "baseline_checkpoint_sha256": baseline.get("checkpoint_sha256"),
        "baseline_selected_pdms": float(baseline["metrics"]["selected_pdms"]),
        "baseline_best_of_64_pdms": float(
            baseline["metrics"]["best_of_64_pdms"]
        ),
        "promoted_artifact_count": len(promoted),
        "evaluated_artifact_count": len(rows),
        "coverage_complete": True,
        "online_cache_parity_required": args.parity_root is not None,
        "best": rows[0] if rows else None,
        "exceeds_0_93_count": sum(bool(row["exceeds_0_93"]) for row in rows),
        "primary_seed_count": len(primary),
        "primary_seed_selected_pdms_mean": mean(primary_scores) if primary_scores else None,
        "primary_seed_selected_pdms_std": pstdev(primary_scores) if len(primary_scores) > 1 else 0.0 if primary_scores else None,
        "primary_seed_delta_mean": mean(primary_deltas) if primary_deltas else None,
        "primary_seed_direction_consistent": (
            all(value > 0 for value in primary_deltas)
            or all(value < 0 for value in primary_deltas)
            if primary_deltas
            else None
        ),
        "rows": rows,
        "excluded_validation_artifacts": promotion.get("excluded", []),
    }


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _markdown(result: Dict[str, object]) -> str:
    lines = [
        "# No-VQA scene-token scorer campaign",
        "",
        f"- Strict coverage: **{'PASS' if result['coverage_complete'] else 'FAIL'}**",
        f"- Validation-promoted / Navtest-evaluated: `{result['promoted_artifact_count']}` / `{result['evaluated_artifact_count']}`",
        f"- Matching No-VQA Base PDMS: `{result['baseline_selected_pdms']:.6f}`",
        f"- Matching candidate-bank Best-of-64: `{result['baseline_best_of_64_pdms']:.6f}`",
        f"- Models above 0.93: `{result['exceeds_0_93_count']}`",
        "",
        "| model | val delta | Navtest PDMS | Navtest delta | 95% CI | regret | >0.93 |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in result["rows"]:
        lines.append(
            f"| {row['name']} | {row['validation_delta']:+.6f} | "
            f"{row['navtest_selected_pdms']:.6f} | {row['navtest_delta']:+.6f} | "
            f"[{row['navtest_ci_low']:+.6f}, {row['navtest_ci_high']:+.6f}] | "
            f"{row['scorer_regret']:.6f} | {'yes' if row['exceeds_0_93'] else 'no'} |"
        )
    lines.extend(
        (
            "",
            "All rows use FP32, 12,146 scenes, 136 logs, 64 fixed No-VQA",
            "proposals, zero invalid scenes, and join official PDM factors only after",
            "the scorer has frozen its selected index. Best-of-64 is an offline oracle",
            "candidate-bank upper bound, not deployable PDMS.",
            "",
        )
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--promotion-manifest", type=Path, required=True)
    parser.add_argument("--baseline-audit", type=Path, required=True)
    parser.add_argument("--navtest-root", type=Path, required=True)
    parser.add_argument("--parity-root", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = summarize(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _write_csv(args.output_csv, result["rows"])
    args.output_md.write_text(_markdown(result))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
