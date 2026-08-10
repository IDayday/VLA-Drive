"""Fail-fast validation and compact summaries for local NAVSIM-v2 CSV output."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


NAVTEST_SUMMARY = "average_all_frames"
NAVHARD_SUMMARIES = {
    "stage_one_score": "extended_pdm_score_stage_one",
    "stage_two_score": "extended_pdm_score_stage_two",
    "combined_score": "extended_pdm_score_combined",
}


def _valid_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    normalized = series.astype(str).str.strip().str.lower()
    unknown = ~normalized.isin(("true", "false", "1", "0"))
    if unknown.any():
        raise ValueError(f"unrecognized valid values: {sorted(normalized[unknown].unique())}")
    return normalized.isin(("true", "1"))


def _finite(value: object, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite: {value}")
    return result


def validate_results(csv_path: Path, suite: str, expected_scenarios: int) -> dict[str, Any]:
    """Validate scorer completeness and return explicit stage-level scores."""
    if suite not in ("navtest", "navhard_two_stage"):
        raise ValueError(f"unsupported suite: {suite}")
    if expected_scenarios <= 0:
        raise ValueError("expected_scenarios must be positive")
    frame = pd.read_csv(csv_path)
    required = {"token", "valid", "score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"result CSV is missing columns: {sorted(missing)}")

    summary_tokens = {NAVTEST_SUMMARY, *NAVHARD_SUMMARIES.values()}
    scenarios = frame[~frame["token"].isin(summary_tokens)].copy()
    if len(scenarios) != expected_scenarios:
        raise ValueError(
            f"scored scenario count {len(scenarios)} != expected {expected_scenarios}"
        )
    valid = _valid_series(scenarios["valid"])
    if not valid.all():
        failed = scenarios.loc[~valid, "token"].astype(str).tolist()
        raise ValueError(f"NAVSIM scorer returned {len(failed)} invalid scenarios: {failed[:20]}")
    scores = pd.to_numeric(scenarios["score"], errors="coerce")
    if not scores.map(math.isfinite).all():
        raise ValueError("one or more scenario scores are not finite")

    result: dict[str, Any] = {
        "suite": suite,
        "csv": str(csv_path.resolve()),
        "expected_scenarios": expected_scenarios,
        "valid_scenarios": int(valid.sum()),
        "mean_score": float(scores.mean()),
    }
    if suite == "navtest":
        rows = frame[frame["token"] == NAVTEST_SUMMARY]
        if len(rows) != 1:
            raise ValueError(f"expected exactly one {NAVTEST_SUMMARY} row")
        result["official_summary_score"] = _finite(rows.iloc[0]["score"], "navtest score")
    else:
        for key, token in NAVHARD_SUMMARIES.items():
            rows = frame[frame["token"] == token]
            if len(rows) != 1:
                raise ValueError(f"expected exactly one {token} row")
            if not bool(_valid_series(rows["valid"]).iloc[0]):
                raise ValueError(f"navhard summary {token} is invalid")
            result[key] = _finite(rows.iloc[0]["score"], key)
    return result


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--suite", choices=("navtest", "navhard_two_stage"), required=True)
    parser.add_argument("--expected-scenarios", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = validate_results(args.csv, args.suite, args.expected_scenarios)
    _atomic_json(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
