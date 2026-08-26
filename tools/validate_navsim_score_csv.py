#!/usr/bin/env python3
"""Validate and print one official NAVSIM aggregate score."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from starVLA.training.navtest_score_io import validate_score_directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--protocol", choices=("pdms", "epdms"), required=True)
    parser.add_argument("--expected-scenarios", type=int)
    parser.add_argument("--expected-datalist")
    args = parser.parse_args()
    expected_tokens = None
    if args.expected_datalist:
        expected_tokens = json.loads(
            Path(args.expected_datalist)
            .expanduser()
            .resolve()
            .read_text(encoding="utf-8")
        )
    result = validate_score_directory(
        args.results_dir,
        protocol=args.protocol,
        expected_scenarios=args.expected_scenarios,
        expected_tokens=expected_tokens,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
