#!/usr/bin/env python3
"""Validate NAVSIM metric-cache metadata and optional token coverage."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--expected-datalist")
    parser.add_argument("--check-cache-files", action="store_true")
    return parser.parse_args()


def cache_paths(root: Path) -> dict[str, Path]:
    metadata = root / "metadata"
    csv_paths = sorted(metadata.glob("*.csv")) if metadata.is_dir() else []
    if not csv_paths:
        raise FileNotFoundError(
            f"metric-cache metadata CSV is missing under {metadata}"
        )
    result: dict[str, Path] = {}
    for metadata_csv in csv_paths:
        with metadata_csv.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
        for row in rows[1:]:
            values = [value.strip() for value in row if value.strip()]
            if not values:
                continue
            raw_path = Path(values[-1])
            token = raw_path.parent.name
            if not token:
                raise ValueError(
                    f"invalid metric-cache path in {metadata_csv}: {raw_path}"
                )
            previous = result.get(token)
            if previous is not None and previous != raw_path:
                raise RuntimeError(
                    f"duplicate metric-cache token with different paths: {token}"
                )
            result[token] = raw_path
    if not result:
        raise RuntimeError(f"metric-cache metadata contains no cache paths: {metadata}")
    return result


def main() -> None:
    args = _parse_args()
    root = Path(args.cache_root).expanduser().resolve()
    paths = cache_paths(root)
    missing_files = sorted(token for token, path in paths.items() if not path.is_file())
    if args.check_cache_files and missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} metric-cache files referenced by metadata are missing"
        )
    expected_count = missing_count = 0
    if args.expected_datalist:
        datalist = Path(args.expected_datalist).expanduser().resolve()
        expected = json.loads(datalist.read_text(encoding="utf-8"))
        if not isinstance(expected, list) or any(
            not isinstance(token, str) for token in expected
        ):
            raise TypeError("expected datalist must be a JSON list of tokens")
        expected_count = len(expected)
        missing = sorted(set(expected) - set(paths))
        missing_count = len(missing)
        if missing:
            raise RuntimeError(
                f"metric cache misses {len(missing)} / {len(expected)} expected tokens; first={missing[:5]}"
            )
    print(
        json.dumps(
            {
                "cache_root": str(root),
                "cache_tokens": len(paths),
                "expected_tokens": expected_count,
                "missing_expected_tokens": missing_count,
                "missing_cache_files": len(missing_files),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
