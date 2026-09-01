"""Merge promoted-scorer manifests with complete Navtest campaign results."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Sequence


FACTOR_NAMES = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "time_to_collision_within_bound",
    "ego_progress",
    "comfort",
    "score",
)


def _load_json(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text())


def _status(delta: float, low: float, high: float) -> str:
    if low > 0.0:
        return "TEST_POSITIVE_SIGNIFICANT"
    if delta > 0.0:
        return "TEST_POSITIVE_INCONCLUSIVE"
    if high >= 0.0:
        return "TEST_NEGATIVE_INCONCLUSIVE"
    return "TEST_NEGATIVE_SIGNIFICANT"


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value)
    temporary.replace(path)


def _manifest_records(paths: Sequence[Path]) -> Dict[str, Dict[str, object]]:
    records: Dict[str, Dict[str, object]] = {}
    for path in paths:
        manifest = _load_json(path)
        for raw in manifest.get("artifacts", []):
            record = dict(raw)
            digest = str(record["sha256"])
            if digest in records:
                raise RuntimeError(f"Duplicate promoted artifact SHA256: {digest}")
            record["promotion_manifest"] = str(path.resolve())
            records[digest] = record
    if not records:
        raise RuntimeError("No promoted artifacts found")
    return records


def _campaign_records(paths: Sequence[Path]) -> tuple[Dict[str, Dict[str, object]], Dict[str, object]]:
    records: Dict[str, Dict[str, object]] = {}
    invariant: Dict[str, object] = {}
    for path in paths:
        campaign = _load_json(path)
        current_invariant = {
            "feature_cache_sha256": campaign["feature_cache_sha256"],
            "candidate_matrix_sha256": campaign["candidate_matrix_sha256"],
            "candidate_proposal_lineage_match": campaign[
                "candidate_proposal_lineage_match"
            ],
            "base_score_max_abs": campaign["base_score_max_abs"],
        }
        if invariant and current_invariant != invariant:
            raise RuntimeError(
                f"Campaign invariants differ: {path}: {current_invariant} != {invariant}"
            )
        invariant = current_invariant
        for raw in campaign.get("methods", []):
            record = dict(raw)
            digest = str(record["artifact_sha256"])
            if digest in records:
                raise RuntimeError(f"Duplicate tested artifact SHA256: {digest}")
            if not record.get("accepted"):
                raise RuntimeError(f"Navtest audit gates failed for {record['method']}")
            record["campaign_summary"] = str(path.resolve())
            records[digest] = record
    if not records:
        raise RuntimeError("No campaign methods found")
    return records, invariant


def _build_rows(
    promoted: Mapping[str, Mapping[str, object]],
    tested: Mapping[str, Mapping[str, object]],
) -> List[Dict[str, object]]:
    missing = sorted(set(promoted) - set(tested))
    extra = sorted(set(tested) - set(promoted))
    if missing or extra:
        raise RuntimeError(
            f"Promotion/test SHA coverage mismatch: missing={missing}, extra={extra}"
        )
    rows: List[Dict[str, object]] = []
    for digest, promotion in promoted.items():
        result = tested[digest]
        metrics = result["metrics"]
        interval = result["pdms_delta_log_cluster_bootstrap"]
        validation_interval = promotion["validation_pdms_delta_ci95"]
        row: Dict[str, object] = {
            "method": result["method"],
            "artifact_sha256": digest,
            "tier": promotion["tier"],
            "validation_pdms_delta": promotion["validation_pdms_delta"],
            "validation_ci95_low": validation_interval[0],
            "validation_ci95_high": validation_interval[1],
            "navtest_selected_pdms": metrics["selected_pdms"],
            "navtest_public_base_pdms": metrics["public_base_selected_pdms"],
            "navtest_best_of_64_pdms": metrics["best_of_64_pdms"],
            "navtest_pdms_delta": metrics["pdms_delta"],
            "navtest_ci95_low": interval["ci95_low"],
            "navtest_ci95_high": interval["ci95_high"],
            "navtest_status": _status(
                float(metrics["pdms_delta"]),
                float(interval["ci95_low"]),
                float(interval["ci95_high"]),
            ),
            "navtest_scorer_regret": metrics["scorer_regret"],
            "navtest_regret_reduction_fraction": metrics[
                "regret_reduction_fraction"
            ],
            "switch_rate": metrics["switch_rate"],
            "win_rate": metrics["win_rate"],
            "loss_rate": metrics["loss_rate"],
            "scene_count": result["scene_count"],
            "log_count": result["log_count"],
            "candidate_count": result["candidate_count"],
            "invalid_scene_count": result["invalid_scene_count"],
            "inference_inputs_only": result["inference_inputs_only"],
            "official_pdm_used_only_after_selection": result[
                "official_pdm_used_only_after_selection"
            ],
            "promotion_manifest": promotion["promotion_manifest"],
            "campaign_summary": result["campaign_summary"],
        }
        for factor in FACTOR_NAMES:
            row[f"navtest_{factor}_delta"] = (
                float(metrics[f"selected_{factor}"])
                - float(metrics[f"base_{factor}"])
            )
        rows.append(row)
    return sorted(rows, key=lambda item: float(item["navtest_pdms_delta"]), reverse=True)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _markdown(summary: Mapping[str, object], rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "# All Effective Scorers: Complete Navtest Audit",
        "",
        "Every artifact promoted by a positive held-out-log bootstrap lower bound was evaluated on the same complete Navtest cache. Official PDM factors were joined only after candidate selection.",
        "",
        "## Coverage and outcome",
        "",
        f"- Promoted and tested artifacts: {summary['method_count']} / {summary['method_count']}",
        f"- Navtest scenes / logs / candidates: {summary['scene_count']} / {summary['log_count']} / {summary['candidate_count']}",
        f"- Public Base PDMS: {summary['public_base_pdms']:.9f}",
        f"- Best tested scorer PDMS: {summary['best_test_pdms']:.9f}",
        f"- Best tested delta: {summary['best_test_delta']:+.9f}",
        f"- Positive test deltas: {summary['positive_test_count']}",
        f"- Positive test 95% CI lower bounds: {summary['positive_test_ci_count']}",
        f"- Validation-positive to test-negative sign flips: {summary['validation_positive_test_negative_count']}",
        f"- Methods above 0.93 PDMS: {summary['above_093_count']}",
        "",
        "## Ranked results",
        "",
        "| Method | Validation delta | Navtest PDMS | Navtest delta | 95% log-bootstrap CI | Switch rate | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {validation_pdms_delta:+.6f} | {navtest_selected_pdms:.6f} | "
            "{navtest_pdms_delta:+.6f} | [{navtest_ci95_low:+.6f}, {navtest_ci95_high:+.6f}] | "
            "{switch_rate:.3f} | {navtest_status} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A validation improvement is treated only as a promotion signal. It is not reported as a planning improvement unless the complete Navtest result is also positive. The current campaign shows a systematic validation-to-test sign reversal, so none of these fine-rankers is deployable as an improvement over the released scorer.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--campaign", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    promoted = _manifest_records(args.manifest)
    tested, invariant = _campaign_records(args.campaign)
    rows = _build_rows(promoted, tested)
    baseline_values = {float(row["navtest_public_base_pdms"]) for row in rows}
    scene_counts = {int(row["scene_count"]) for row in rows}
    log_counts = {int(row["log_count"]) for row in rows}
    candidate_counts = {int(row["candidate_count"]) for row in rows}
    if len(baseline_values) != 1 or scene_counts != {12_146} or log_counts != {136}:
        raise RuntimeError(
            f"Navtest invariants failed: baseline={baseline_values}, "
            f"scenes={scene_counts}, logs={log_counts}"
        )
    if candidate_counts != {64}:
        raise RuntimeError(f"Candidate-count invariant failed: {candidate_counts}")
    best = rows[0]
    summary: Dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method_count": len(rows),
        "scene_count": 12_146,
        "log_count": 136,
        "candidate_count": 64,
        "public_base_pdms": next(iter(baseline_values)),
        "best_method": best["method"],
        "best_test_pdms": best["navtest_selected_pdms"],
        "best_test_delta": best["navtest_pdms_delta"],
        "best_test_ci95": [best["navtest_ci95_low"], best["navtest_ci95_high"]],
        "positive_test_count": sum(float(row["navtest_pdms_delta"]) > 0 for row in rows),
        "positive_test_ci_count": sum(float(row["navtest_ci95_low"]) > 0 for row in rows),
        "validation_positive_test_negative_count": sum(
            float(row["validation_pdms_delta"]) > 0
            and float(row["navtest_pdms_delta"]) < 0
            for row in rows
        ),
        "above_093_count": sum(float(row["navtest_selected_pdms"]) > 0.93 for row in rows),
        "feature_and_candidate_invariants": invariant,
        "all_promoted_artifacts_tested": True,
        "rows": rows,
    }
    _atomic_text(args.output_json, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_csv(args.output_csv, rows)
    _atomic_text(args.output_md, _markdown(summary, rows))
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
