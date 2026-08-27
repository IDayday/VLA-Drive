#!/usr/bin/env python3
"""Validate and summarize Register64 scorer-vs-Oracle NAVSIM-v1.1 PDMS."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from starVLA.training.navtest_score_io import validate_score_directory


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _source_pdms(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    matches = [row for row in rows if row.get("protocol") == "pdms_v1_1"]
    if len(matches) != 1:
        raise RuntimeError(
            f"source summary must contain one pdms_v1_1 row: {path}"
        )
    row = matches[0]
    score = float(row["score"])
    scenarios = int(row["num_scenarios"])
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"source PDMS is invalid: {score}")
    return {
        "score": score,
        "score_percent": score * 100.0,
        "num_scenarios": scenarios,
        "csv": row.get("official_csv"),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-report", required=True)
    parser.add_argument("--official-results-dir", required=True)
    parser.add_argument("--source-summary", required=True)
    parser.add_argument("--datalist", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-scenarios", type=int, required=True)
    parser.add_argument("--parity-tolerance", type=float, default=1.0e-4)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.expected_scenarios <= 0 or args.parity_tolerance < 0:
        raise ValueError("scenario count must be positive and tolerance non-negative")
    report_path = Path(args.oracle_report).expanduser().resolve()
    source_path = Path(args.source_summary).expanduser().resolve()
    datalist_path = Path(args.datalist).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    tokens = json.loads(datalist_path.read_text(encoding="utf-8"))
    if len(tokens) != args.expected_scenarios:
        raise RuntimeError("datalist size does not match --expected-scenarios")
    if int(report.get("num_scenes", -1)) != args.expected_scenarios:
        raise RuntimeError("Oracle pool report has the wrong scenario count")
    source = _source_pdms(source_path)
    if source["num_scenarios"] != args.expected_scenarios:
        raise RuntimeError("source result has the wrong scenario count")
    official = validate_score_directory(
        args.official_results_dir,
        protocol="pdms",
        expected_scenarios=args.expected_scenarios,
        expected_tokens=tokens,
    )
    pool_oracle = float(report["mean_oracle_at_64_pdms"])
    parity_gap = official["score"] - pool_oracle
    if abs(parity_gap) > args.parity_tolerance:
        raise RuntimeError(
            "batched Oracle and official Oracle PDMS disagree: "
            f"official={official['score']:.8f} pool={pool_oracle:.8f} "
            f"gap={parity_gap:.8f} tolerance={args.parity_tolerance:.8f}"
        )
    baseline = source["score"]
    oracle = official["score"]
    summary = {
        "schema_version": 1,
        "protocol": "navsim_v1.1_pdms_oracle_at_64",
        "num_scenarios": args.expected_scenarios,
        "source_drivor_official_pdms": baseline,
        "recomputed_drivor_selected_pdms": float(
            report["mean_drivor_selected_pdms"]
        ),
        "proposal0_pdms": float(report["mean_proposal0_pdms"]),
        "deterministic_random_pdms": float(
            report["mean_deterministic_random_pdms"]
        ),
        "oracle_at_64_pool_pdms": pool_oracle,
        "oracle_at_64_official_pdms": oracle,
        "selector_gap_to_oracle": oracle - baseline,
        "generator_ceiling_gap_to_one": 1.0 - oracle,
        "drivor_exact_oracle_rate": float(report["drivor_exact_oracle_rate"]),
        "pool_official_parity_gap": parity_gap,
        "pool_official_parity_tolerance": args.parity_tolerance,
        "source_official_csv": source["csv"],
        "oracle_official_csv": official["csv"],
        "oracle_report": str(report_path),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "summary.json", summary)

    csv_lines = ["selection,score,score_percent,num_scenarios"]
    rows = (
        ("drivor_official", baseline),
        ("drivor_recomputed", summary["recomputed_drivor_selected_pdms"]),
        ("proposal0", summary["proposal0_pdms"]),
        ("deterministic_random", summary["deterministic_random_pdms"]),
        ("oracle64_pool", pool_oracle),
        ("oracle64_official", oracle),
    )
    csv_lines.extend(
        f"{name},{score:.10f},{100.0 * score:.6f},{args.expected_scenarios}"
        for name, score in rows
    )
    _atomic_text(output_dir / "summary.csv", "\n".join(csv_lines) + "\n")

    markdown = f"""# Register64 navtest PDMS Oracle@64 diagnostic

- Scenarios: {args.expected_scenarios}
- DrivoR official PDMS: **{100.0 * baseline:.3f}**
- DrivoR selection recomputed from the same candidate pools: **{100.0 * summary['recomputed_drivor_selected_pdms']:.3f}**
- Proposal-0 PDMS: **{100.0 * summary['proposal0_pdms']:.3f}**
- Deterministic-random proposal PDMS: **{100.0 * summary['deterministic_random_pdms']:.3f}**
- Oracle@64 batched PDMS: **{100.0 * pool_oracle:.3f}**
- Oracle@64 official PDMS: **{100.0 * oracle:.3f}**
- Selector gap to Oracle@64: **{100.0 * (oracle - baseline):.3f}** points
- Generator ceiling gap to 100: **{100.0 * (1.0 - oracle):.3f}** points
- DrivoR exact Oracle-register hit rate: **{100.0 * summary['drivor_exact_oracle_rate']:.3f}%**
- Batched/official Oracle parity gap: `{parity_gap:.8f}`

Interpretation: the Oracle@64 official score is the trajectory-generator ceiling under NAVSIM-v1.1 PDMS. The gap from DrivoR official PDMS to that ceiling is attributable to candidate selection; the remaining gap from the ceiling to 100 is not recoverable by a better selector over this fixed 64-trajectory pool.
"""
    _atomic_text(output_dir / "summary.md", markdown)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
