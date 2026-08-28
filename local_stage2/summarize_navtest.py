#!/usr/bin/env python3
"""Validate a full Navtest CSV and compare it with the public Base result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd


EXPECTED_NAVTEST_SCENES = 12_146


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_full_navtest(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(path)
    required = {"token", "valid", "score"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{path} is missing columns: {missing}")

    average_rows = frame[frame["token"] == "average"]
    scenarios = frame[frame["token"] != "average"].copy()
    if len(average_rows) != 1:
        raise RuntimeError(f"{path} has {len(average_rows)} average rows; expected one")
    if len(scenarios) != EXPECTED_NAVTEST_SCENES:
        raise RuntimeError(
            f"{path} has {len(scenarios)} scenarios; expected {EXPECTED_NAVTEST_SCENES}"
        )
    if scenarios["token"].duplicated().any():
        duplicates = scenarios.loc[scenarios["token"].duplicated(), "token"].tolist()
        raise RuntimeError(f"{path} has duplicate tokens: {duplicates[:10]}")

    valid = scenarios["valid"]
    if valid.dtype != bool:
        normalized = valid.astype(str).str.lower()
        if not normalized.isin({"true", "false"}).all():
            raise RuntimeError(f"{path} contains invalid values in the valid column")
        valid = normalized == "true"
    valid_count = int(valid.sum())
    if valid_count != EXPECTED_NAVTEST_SCENES:
        raise RuntimeError(
            f"{path} has {valid_count}/{EXPECTED_NAVTEST_SCENES} valid scenarios"
        )

    scores = pd.to_numeric(scenarios["score"], errors="raise")
    if not all(math.isfinite(float(score)) for score in scores):
        raise RuntimeError(f"{path} contains non-finite scores")
    pdms = float(scores.mean())
    recorded_average = float(average_rows.iloc[0]["score"])
    if not math.isclose(pdms, recorded_average, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(
            f"{path} average-row mismatch: computed={pdms}, recorded={recorded_average}"
        )

    return scenarios, {
        "csv": str(path.resolve()),
        "csv_sha256": sha256(path),
        "scenario_count": len(scenarios),
        "valid_count": valid_count,
        "pdms": pdms,
        "pdms_percent": pdms * 100.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_csv", type=Path)
    parser.add_argument("reference_csv", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()

    candidate_rows, candidate = load_full_navtest(args.candidate_csv)
    reference_rows, reference = load_full_navtest(args.reference_csv)
    candidate_tokens = set(candidate_rows["token"])
    reference_tokens = set(reference_rows["token"])
    if candidate_tokens != reference_tokens:
        raise RuntimeError(
            "Candidate/reference token sets differ: "
            f"candidate_only={sorted(candidate_tokens - reference_tokens)[:10]}, "
            f"reference_only={sorted(reference_tokens - candidate_tokens)[:10]}"
        )

    delta = candidate["pdms"] - reference["pdms"]
    report: dict[str, Any] = {
        "candidate": candidate,
        "reference": reference,
        "delta_pdms": delta,
        "delta_points": delta * 100.0,
        "reached_public_base": delta >= -1e-12,
        "token_sets_equal": True,
    }
    if args.checkpoint is not None:
        if not args.checkpoint.is_file():
            raise FileNotFoundError(args.checkpoint)
        report["checkpoint"] = {
            "path": str(args.checkpoint.resolve()),
            "sha256": sha256(args.checkpoint),
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output_json)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
