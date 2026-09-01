#!/usr/bin/env python3
"""Create a deterministic table from candidate-metric JSON reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = (
    "selected_pdms",
    "best_of_k_pdms",
    "scorer_regret",
    "candidate_mean_pdms",
    "candidate_median_pdms",
    "top5_oracle_mean",
    "fraction_candidates_above_0p8",
    "fraction_candidates_above_0p9",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = []
    for path in args.reports:
        payload = json.loads(path.read_text())
        payload["experiment"] = path.stem
        rows.append(payload)
    by_name = {row["experiment"]: row for row in rows}
    if args.baseline not in by_name:
        raise KeyError(
            f"Baseline {args.baseline!r} is not one of {sorted(by_name)}"
        )
    baseline = by_name[args.baseline]
    output_fields = ["experiment", "scene_count", "candidate_count", *FIELDS]
    output_fields.extend(f"delta_{field}" for field in FIELDS)
    for row in rows:
        for field in FIELDS:
            row[f"delta_{field}"] = float(row[field]) - float(baseline[field])
    rows.sort(key=lambda row: row["experiment"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} experiment rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
