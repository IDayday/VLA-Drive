#!/usr/bin/env python3
"""Create a lightweight metadata view of a NAVSIM metric-cache subset."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--datalist", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    source = Path(args.source_root).resolve()
    output = Path(args.output_root).resolve()
    tokens = json.loads(Path(args.datalist).read_text(encoding="utf-8"))
    metadata_files = sorted((source / "metadata").glob("*.csv"))
    if len(metadata_files) != 1:
        raise RuntimeError(f"Expected one metadata CSV under {source}/metadata")
    with metadata_files[0].open(newline="", encoding="utf-8") as stream:
        paths = [row[0] for row in csv.reader(stream) if row]
    if paths and paths[0] == "file_name":
        paths = paths[1:]
    by_token = {Path(raw).parent.name: raw for raw in paths}
    missing = [token for token in tokens if token not in by_token]
    if missing:
        raise RuntimeError(f"Metric cache is missing {len(missing)} requested tokens")
    metadata_dir = output / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=False)
    destination = metadata_dir / "cache.csv"
    temporary = destination.with_name(destination.name + f".tmp-{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["file_name"])
        writer.writerows((by_token[token],) for token in tokens)
    os.replace(temporary, destination)


if __name__ == "__main__":
    main()
