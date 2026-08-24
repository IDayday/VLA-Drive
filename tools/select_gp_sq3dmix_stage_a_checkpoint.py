#!/usr/bin/env python3
"""Select a Stage-A checkpoint using only the model-selection split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+")
    args = parser.parse_args()
    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.reports]
    # Prefer gate-passing candidates, then causal separation, then utility.
    eligible = [report for report in reports if report.get("all_passed")]
    pool = eligible or reports
    selected = max(
        pool,
        key=lambda report: (
            report["relative_shuffled_real_flow_gap"],
            -report["mean_real_minus_base"],
        ),
    )
    print(selected["checkpoint"])


if __name__ == "__main__":
    main()
