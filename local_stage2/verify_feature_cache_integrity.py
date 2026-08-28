#!/usr/bin/env python3

"""Check gzip integrity of a DriveVLA feature cache in parallel."""

import argparse
import gzip
import multiprocessing as mp
from pathlib import Path
from typing import Optional, Tuple


def check_gzip(path_string: str) -> Tuple[str, Optional[str]]:
    path = Path(path_string)
    try:
        with gzip.open(path, "rb") as stream:
            while stream.read(1024 * 1024):
                pass
        return path_string, None
    except Exception as error:  # report the exact corrupt artifact to the caller
        return path_string, repr(error)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--workers", type=int, default=min(64, mp.cpu_count()))
    parser.add_argument("--quarantine-corrupt", action="store_true")
    args = parser.parse_args()

    paths = sorted(str(path) for path in args.root.glob("*/*/*.gz"))
    print(f"gzip files discovered: {len(paths):,}")
    corrupt = []
    with mp.Pool(args.workers) as pool:
        for index, (path_string, error) in enumerate(
            pool.imap_unordered(check_gzip, paths, chunksize=32), start=1
        ):
            if error is not None:
                corrupt.append((path_string, error))
            if index % 10_000 == 0 or index == len(paths):
                print(f"checked: {index:,} / {len(paths):,}; corrupt: {len(corrupt)}")

    for path_string, error in corrupt:
        print(f"CORRUPT {path_string}: {error}")
        if args.quarantine_corrupt:
            path = Path(path_string)
            destination = path.with_name(path.name + ".corrupt")
            path.replace(destination)
            print(f"QUARANTINED {destination}")

    if corrupt:
        raise SystemExit(f"feature cache contains {len(corrupt)} corrupt gzip files")
    print("feature cache gzip integrity: PASS")


if __name__ == "__main__":
    main()
