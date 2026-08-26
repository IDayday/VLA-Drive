"""Strict readers for official NAVSIM v1 PDMS and v2 navtest EPDMS CSVs."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Sequence


_SUMMARY_TOKENS = {
    "average",
    "average_all_frames",
    "extended_pdm_score_stage_one",
    "extended_pdm_score_stage_two",
    "extended_pdm_score_combined",
}


def latest_score_csv(results_dir: str | Path) -> Path:
    root = Path(results_dir).expanduser().resolve()
    candidates = [path for path in root.rglob("*.csv") if path.is_file()]
    if not candidates:
        raise FileNotFoundError(f"no official score CSV found under {root}")
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)))


def _as_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "1.0"}:
        return True
    if normalized in {"false", "0", "0.0"}:
        return False
    raise ValueError(f"invalid CSV boolean value: {value!r}")


def validate_score_directory(
    results_dir: str | Path,
    *,
    protocol: str,
    expected_scenarios: int | None = None,
    expected_tokens: Sequence[str] | None = None,
) -> dict[str, Any]:
    protocol = protocol.lower()
    summary_token = {
        "pdms": "average",
        "epdms": "average_all_frames",
    }.get(protocol)
    if summary_token is None:
        raise ValueError("protocol must be pdms or epdms")
    csv_path = latest_score_csv(results_dir)
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if (
        not rows
        or "token" not in rows[0]
        or "valid" not in rows[0]
        or "score" not in rows[0]
    ):
        raise ValueError(f"official score CSV has an invalid schema: {csv_path}")
    summaries = [row for row in rows if row["token"] == summary_token]
    if len(summaries) != 1:
        raise RuntimeError(
            f"expected exactly one {summary_token!r} row in {csv_path}, found {len(summaries)}"
        )
    scenarios = [row for row in rows if row["token"] not in _SUMMARY_TOKENS]
    tokens = [row["token"] for row in scenarios]
    if len(tokens) != len(set(tokens)):
        raise RuntimeError("official score CSV contains duplicate scenario tokens")
    if expected_scenarios is not None and len(scenarios) != expected_scenarios:
        raise RuntimeError(
            f"official {protocol.upper()} scenario count mismatch: "
            f"{len(scenarios)} != {expected_scenarios}"
        )
    if expected_tokens is not None:
        expected = list(expected_tokens)
        if len(expected) != len(set(expected)):
            raise ValueError("expected NAVSIM tokens must be unique")
        missing = sorted(set(expected) - set(tokens))
        extra = sorted(set(tokens) - set(expected))
        if missing or extra:
            raise RuntimeError(
                f"official {protocol.upper()} token set mismatch: "
                f"missing={len(missing)} extra={len(extra)}"
            )
    failed = [row["token"] for row in scenarios if not _as_bool(row["valid"])]
    if failed:
        raise RuntimeError(
            f"official {protocol.upper()} evaluation has {len(failed)} failed scenarios; first={failed[:5]}"
        )
    summary = summaries[0]
    if not _as_bool(summary["valid"]):
        raise RuntimeError(f"official {protocol.upper()} aggregate row is invalid")
    score = float(summary["score"])
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"official {protocol.upper()} score is outside [0,1]: {score}")
    return {
        "protocol": protocol,
        "csv": str(csv_path),
        "summary_token": summary_token,
        "score": score,
        "score_percent": 100.0 * score,
        "num_scenarios": len(scenarios),
        "num_failed": 0,
    }
