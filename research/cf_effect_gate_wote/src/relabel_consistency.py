"""Exact run-to-run consistency and non-blocking published-label audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .feature_store import atomic_write_json, sha256_file
from .independent_label_store import (
    FACTOR_ORDER,
    IndependentCandidateLabelStore,
    IndependentLabelStoreError,
)


PUBLISHED_FACTOR_KEYS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
)


def compare_independent_runs(
    first_root: Path,
    second_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare all arrays exactly; a mean-error fallback is intentionally absent."""

    first = IndependentCandidateLabelStore(first_root)
    second = IndependentCandidateLabelStore(second_root)
    if first.scene_tokens != second.scene_tokens:
        raise IndependentLabelStoreError("run1/run2 scene token order differs")
    first_scenes = first.scene_index()
    second_scenes = second.scene_index()
    rows: list[dict[str, Any]] = []
    factor_equal_all = True
    score_equal_all = True
    oracle_equal_all = True
    max_factor_error = 0.0
    max_score_error = 0.0
    for token in first.scene_tokens:
        left = first_scenes[token]
        right = second_scenes[token]
        if left.record != right.record:
            raise IndependentLabelStoreError(f"run identity differs for {token}")
        factor_equal = bool(np.array_equal(left.factors, right.factors))
        score_equal = bool(np.array_equal(left.score, right.score))
        oracle_equal = left.oracle_index == right.oracle_index
        factor_error = float(np.max(np.abs(left.factors - right.factors)))
        score_error = float(np.max(np.abs(left.score - right.score)))
        factor_equal_all &= factor_equal
        score_equal_all &= score_equal
        oracle_equal_all &= oracle_equal
        max_factor_error = max(max_factor_error, factor_error)
        max_score_error = max(max_score_error, score_error)
        rows.append(
            {
                "scene_token": token,
                "factors_array_equal": factor_equal,
                "score_array_equal": score_equal,
                "oracle_index_equal": oracle_equal,
                "max_factor_absolute_error": factor_error,
                "max_score_absolute_error": score_error,
            }
        )
    logical_equal = first.logical_content_sha256 == second.logical_content_sha256
    first_manifest_sha = sha256_file(first_root / "manifest.json")
    second_manifest_sha = sha256_file(second_root / "manifest.json")
    summary = {
        "status": (
            "PASS"
            if logical_equal
            and factor_equal_all
            and score_equal_all
            and oracle_equal_all
            and first_manifest_sha == second_manifest_sha
            else "FAIL"
        ),
        "scenes": len(rows),
        "candidates": len(rows) * 256,
        "run1_logical_sha256": first.logical_content_sha256,
        "run2_logical_sha256": second.logical_content_sha256,
        "logical_sha256_equal": logical_equal,
        "run1_manifest_sha256": first_manifest_sha,
        "run2_manifest_sha256": second_manifest_sha,
        "manifest_bytes_equal": first_manifest_sha == second_manifest_sha,
        "factor_arrays_exactly_equal": factor_equal_all,
        "score_arrays_exactly_equal": score_equal_all,
        "oracle_indices_exactly_equal": oracle_equal_all,
        "max_factor_absolute_error": max_factor_error,
        "max_score_absolute_error": max_score_error,
    }
    return rows, summary


def published_audit(
    independent_root: Path,
    published_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare published labels only as a quarantined, non-Gate audit."""

    store = IndependentCandidateLabelStore(independent_root)
    payload = np.load(published_path, allow_pickle=True).item()
    if not isinstance(payload, dict):
        raise ValueError("published audit input is not a dictionary")
    rows: list[dict[str, Any]] = []
    candidate_mismatches = 0
    argmax_mismatches = 0
    top5_mismatches = 0
    factor_errors: list[np.ndarray] = []
    ep_errors: list[np.ndarray] = []
    for scene in store.iter_scenes():
        token = scene.record.scene_token
        if token not in payload:
            raise ValueError(f"published audit is missing token {token}")
        table = payload[token]["trajectory_scores"][0]
        published_factors = np.stack(
            [np.asarray(table[key], dtype=np.float32) for key in PUBLISHED_FACTOR_KEYS],
            axis=-1,
        )
        if published_factors.shape != scene.factors.shape:
            raise ValueError(f"published factor shape mismatch for {token}")
        errors = np.abs(published_factors - scene.factors)
        factor_errors.append(errors)
        ep_errors.append(errors[:, 2])
        mismatch = np.any(errors > 1e-6, axis=1)
        candidate_mismatches += int(mismatch.sum())
        published_score = np.asarray(table.get("score"), dtype=np.float32)
        if published_score.shape != scene.score.shape:
            raise ValueError(f"published score shape mismatch for {token}")
        published_argmax = int(np.argmax(published_score))
        argmax_mismatches += int(published_argmax != scene.oracle_index)
        published_top5 = set(np.argsort(published_score)[-5:].tolist())
        independent_top5 = set(np.argsort(scene.score)[-5:].tolist())
        top5_mismatches += int(published_top5 != independent_top5)
        for factor_index, factor_name in enumerate(FACTOR_ORDER):
            values = errors[:, factor_index]
            rows.append(
                {
                    "scene_token": token,
                    "factor": factor_name,
                    "mismatch_rate": float(np.mean(values > 1e-6)),
                    "max_absolute_error": float(values.max()),
                    "mean_absolute_error": float(values.mean()),
                }
            )
    all_errors = np.stack(factor_errors)
    ep_values = np.stack(ep_errors)
    scene_count = len(store.scene_tokens)
    summary = {
        "status": "UPSTREAM_REPRODUCTION_AUDIT_ONLY",
        "gate_blocking": False,
        "scenes": scene_count,
        "factor_level_mismatch_rate": float(np.mean(all_errors > 1e-6)),
        "candidate_level_mismatch_rate": candidate_mismatches / (scene_count * 256),
        "maximum_absolute_error": float(all_errors.max()),
        "mean_absolute_error": float(all_errors.mean()),
        "argmax_candidate_disagreement_rate": argmax_mismatches / scene_count,
        "top5_set_disagreement_rate": top5_mismatches / scene_count,
        "ep_specific_mismatch_rate": float(np.mean(ep_values > 1e-6)),
        "ep_maximum_absolute_error": float(ep_values.max()),
        "ep_mean_absolute_error": float(ep_values.mean()),
    }
    return rows, summary


def _write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing existing CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--run1", type=Path, required=True)
    compare.add_argument("--run2", type=Path, required=True)
    compare.add_argument("--output-csv", type=Path, required=True)
    compare.add_argument("--output-summary", type=Path, required=True)
    audit = commands.add_parser("published-audit")
    audit.add_argument("--independent-labels", type=Path, required=True)
    audit.add_argument("--published-scores", type=Path, required=True)
    audit.add_argument("--output-csv", type=Path, required=True)
    audit.add_argument("--output-summary", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "compare":
        rows, summary = compare_independent_runs(args.run1, args.run2)
        _write_rows(args.output_csv, rows, tuple(rows[0].keys()))
        atomic_write_json(args.output_summary, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["status"] == "PASS" else 4
    rows, summary = published_audit(args.independent_labels, args.published_scores)
    _write_rows(args.output_csv, rows, tuple(rows[0].keys()))
    atomic_write_json(args.output_summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
