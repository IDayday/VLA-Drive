#!/usr/bin/env python3

"""Verify that Stage-2 feature caches contain real camera paths."""

import argparse
import gzip
import multiprocessing as mp
import pickle
from pathlib import Path
from typing import Optional, Tuple


def check_feature(arguments: Tuple[str, str]) -> Tuple[str, Optional[str]]:
    path_string, sensor_root_string = arguments
    path = Path(path_string)
    sensor_root = Path(sensor_root_string)
    try:
        with gzip.open(path, "rb") as stream:
            feature = pickle.load(stream)
        required = {
            "history_trajectory",
            "high_command_one_hot",
            "status_feature",
            "image_path_tensor",
        }
        missing = sorted(required - set(feature))
        if missing:
            raise ValueError(f"missing feature keys: {missing}")
        values = feature["image_path_tensor"].tolist()
        image_path = Path("".join(chr(int(value)) for value in values))
        image_path.relative_to(sensor_root)
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            raise ValueError(f"unexpected image suffix: {image_path}")
        return path_string, None
    except Exception as error:
        return path_string, repr(error)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("sensor_root", type=Path)
    parser.add_argument("--workers", type=int, default=min(64, mp.cpu_count()))
    args = parser.parse_args()

    paths = sorted(args.root.glob("*/*/internvl_feature.gz"))
    work = [(str(path), str(args.sensor_root)) for path in paths]
    print(f"DriveVLA feature files discovered: {len(work):,}")
    failures = []
    with mp.Pool(args.workers) as pool:
        for index, (path_string, error) in enumerate(
            pool.imap_unordered(check_feature, work, chunksize=32), start=1
        ):
            if error is not None:
                failures.append((path_string, error))
            if index % 10_000 == 0 or index == len(work):
                print(f"checked: {index:,} / {len(work):,}; invalid: {len(failures)}")

    for path_string, error in failures[:100]:
        print(f"INVALID {path_string}: {error}")
    if failures:
        raise SystemExit(f"feature cache contains {len(failures)} invalid entries")
    print("DriveVLA feature cache semantics: PASS")


if __name__ == "__main__":
    main()
