#!/usr/bin/env python3
"""Fail-closed validation for smoke or full Navtest proposal audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--reference-selected-csv", type=Path)
    parser.add_argument("--expected-scenes", type=int)
    parser.add_argument("--expected-candidates", type=int, default=64)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--require-reference-token-set", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def validate(args: argparse.Namespace) -> dict[str, Any]:
    summary = json.loads((args.audit_dir / "summary.json").read_text())
    frame = pd.read_csv(args.audit_dir / "per_scene_candidate_quality.csv")
    with np.load(args.audit_dir / "candidate_scores.npz") as archive:
        tokens = archive["tokens"].astype(str)
        candidate_scores = archive["candidate_scores"].astype(np.float64)
        selected_indices = archive["selected_indices"].astype(np.int64)
        oracle_indices = archive["oracle_indices"].astype(np.int64)

    failures: list[str] = []
    expected_scenes = args.expected_scenes
    if expected_scenes is not None and int(summary["scene_count"]) != expected_scenes:
        failures.append(
            f"scene_count={summary['scene_count']} expected={expected_scenes}"
        )
    if int(summary["invalid_scene_count"]) != 0:
        failures.append(f"invalid_scene_count={summary['invalid_scene_count']}")
    if int(summary["candidate_count"]) != args.expected_candidates:
        failures.append(
            f"candidate_count={summary['candidate_count']} expected={args.expected_candidates}"
        )
    if len(frame) != int(summary["scene_count"]):
        failures.append("CSV row count differs from summary scene_count")
    if frame["token"].astype(str).duplicated().any():
        failures.append("duplicate token in per-scene CSV")
    if len(tokens) != int(summary["valid_scene_count"]):
        failures.append("NPZ token count differs from valid_scene_count")
    if len(set(tokens)) != len(tokens):
        failures.append("duplicate token in candidate-score NPZ")
    if candidate_scores.shape != (len(tokens), args.expected_candidates):
        failures.append(f"unexpected candidate score shape {candidate_scores.shape}")

    valid_frame = frame.loc[frame["valid"].astype(bool)].copy()
    valid_frame["token"] = valid_frame["token"].astype(str)
    valid_frame = valid_frame.set_index("token").loc[tokens]
    row_indices = np.arange(len(tokens))
    selected_from_array = candidate_scores[row_indices, selected_indices]
    oracle_from_array = candidate_scores[row_indices, oracle_indices]
    array_tolerance = max(args.tolerance, 2e-6)  # candidate_scores.npz is float32
    selected_error = np.abs(
        selected_from_array - valid_frame["selected_pdms"].to_numpy(dtype=np.float64)
    )
    oracle_error = np.abs(
        oracle_from_array - valid_frame["best_of_64_pdms"].to_numpy(dtype=np.float64)
    )
    regret_error = np.abs(
        valid_frame["best_of_64_pdms"].to_numpy(dtype=np.float64)
        - valid_frame["selected_pdms"].to_numpy(dtype=np.float64)
        - valid_frame["scorer_regret"].to_numpy(dtype=np.float64)
    )
    if float(selected_error.max()) > array_tolerance:
        failures.append(f"selected NPZ/CSV mismatch={selected_error.max()}")
    if float(oracle_error.max()) > array_tolerance:
        failures.append(f"oracle NPZ/CSV mismatch={oracle_error.max()}")
    if float(regret_error.max()) > args.tolerance:
        failures.append(f"regret identity mismatch={regret_error.max()}")
    if float(frame["selected_score_parity_abs"].max()) > args.tolerance:
        failures.append(
            "batch/single selected-score parity mismatch="
            f"{frame['selected_score_parity_abs'].max()}"
        )

    reference_result = None
    if args.reference_selected_csv is not None:
        reference = pd.read_csv(args.reference_selected_csv)
        reference = reference.loc[
            reference["token"].astype(str) != "average", ["token", "score"]
        ].copy()
        reference["token"] = reference["token"].astype(str)
        merged = valid_frame.reset_index()[["token", "standard_selected_pdms"]].merge(
            reference, on="token", how="left", validate="one_to_one", indicator=True
        )
        missing = int((merged["_merge"] != "both").sum())
        error = np.abs(
            merged.loc[merged["_merge"] == "both", "standard_selected_pdms"].to_numpy()
            - merged.loc[merged["_merge"] == "both", "score"].to_numpy()
        )
        max_error = float(error.max()) if len(error) else None
        if missing:
            failures.append(f"{missing} audit tokens missing from reference CSV")
        if max_error is None or max_error > args.tolerance:
            failures.append(f"released selected-score parity mismatch={max_error}")
        if args.require_reference_token_set and len(reference) != len(valid_frame):
            failures.append(
                f"reference token count={len(reference)} audit token count={len(valid_frame)}"
            )
        reference_result = {
            "path": str(args.reference_selected_csv.resolve()),
            "matched_tokens": int(len(error)),
            "missing_tokens": missing,
            "mean_abs_error": float(error.mean()) if len(error) else None,
            "max_abs_error": max_error,
        }

    result = {
        "audit_dir": str(args.audit_dir.resolve()),
        "passed": not failures,
        "failures": failures,
        "scene_count": int(summary["scene_count"]),
        "candidate_count": int(summary["candidate_count"]),
        "invalid_scene_count": int(summary["invalid_scene_count"]),
        "max_batch_single_parity_error": float(
            frame["selected_score_parity_abs"].max()
        ),
        "max_selected_npz_csv_error": float(selected_error.max()),
        "max_oracle_npz_csv_error": float(oracle_error.max()),
        "max_regret_identity_error": float(regret_error.max()),
        "reference": reference_result,
    }
    return result


def main() -> None:
    args = _parse_args()
    result = validate(args)
    value = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(value)
    print(value, end="")
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
