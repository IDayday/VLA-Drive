#!/usr/bin/env python3
"""Generate training videos directly from one complete camera archive.

Only the three camera views used by DriveDreamer are temporarily extracted to
local storage. The source archive and shared NAVSIM dataset are never modified.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
import shutil
import subprocess
from functools import partial
from pathlib import Path

from navsim_data_process.make_videos import _process_token
from tqdm import tqdm


CAMERA_PATTERNS = (
    "*/CAM_F0/*.jpg",
    "*/CAM_L0/*.jpg",
    "*/CAM_R0/*.jpg",
)


def _read_token_log(arguments: tuple[str, str]) -> tuple[str, str]:
    token, meta_dir = arguments
    with open(os.path.join(meta_dir, token + ".pkl"), "rb") as handle:
        metadata = pickle.load(handle)
    image_path = metadata["glo_images"]["cam_f0"]["image_paths"][3]
    _, separator, relative_path = image_path.partition("/trainval/")
    if not separator:
        raise ValueError(f"Cannot parse trainval-relative path: {image_path}")
    return relative_path.split("/", 1)[0], token


def build_index(
    datalist_path: Path,
    meta_dir: Path,
    index_path: Path,
    workers: int,
) -> None:
    with datalist_path.open("r", encoding="utf-8") as handle:
        tokens = json.load(handle)

    index: dict[str, list[str]] = {}
    arguments = ((token, str(meta_dir)) for token in tokens)
    with mp.Pool(processes=workers) as pool:
        results = pool.imap_unordered(_read_token_log, arguments, chunksize=64)
        for log_name, token in tqdm(results, total=len(tokens), desc="index video logs"):
            index.setdefault(log_name, []).append(token)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = index_path.with_name(index_path.name + f".tmp-{os.getpid()}")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(index, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, index_path)
    print(f"[OK] indexed {len(tokens)} tokens across {len(index)} logs: {index_path}")


def process_archive(
    archive_path: Path,
    index_path: Path,
    meta_dir: Path,
    video_dir: Path,
    marker_dir: Path,
    staging_root: Path,
    video_workers: int,
    encoder_preset: str,
) -> None:
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker_path = marker_dir / f"{archive_path.name}.done"
    if marker_path.is_file():
        print(f"[SKIP] {archive_path.name}")
        return

    staging_root.mkdir(parents=True, exist_ok=True)
    staging_dir = staging_root / archive_path.stem
    resolved_staging_root = staging_root.resolve()
    resolved_staging_dir = staging_dir.resolve()
    if not resolved_staging_dir.is_relative_to(resolved_staging_root):
        raise ValueError(f"Unsafe staging path: {resolved_staging_dir}")

    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    print(f"[EXTRACT] {archive_path.name} -> {staging_dir}")
    try:
        subprocess.run(
            [
                "tar",
                "--extract",
                "--gzip",
                "--file",
                str(archive_path),
                "--directory",
                str(staging_dir),
                "--strip-components=3",
                "--wildcards",
                "--no-anchored",
                *CAMERA_PATTERNS,
            ],
            check=True,
        )

        with index_path.open("r", encoding="utf-8") as handle:
            token_index: dict[str, list[str]] = json.load(handle)

        log_names = [
            path.name for path in staging_dir.iterdir() if path.is_dir()
        ]
        tokens = [
            token
            for log_name in log_names
            for token in token_index.get(log_name, [])
        ]
        worker = partial(
            _process_token,
            meta_dir=str(meta_dir),
            video_dir=str(video_dir),
            image_root=str(staging_dir),
            encoder_preset=encoder_preset,
        )

        failures: list[tuple[str, str]] = []
        with mp.Pool(processes=video_workers) as pool:
            results = pool.imap_unordered(worker, tokens, chunksize=1)
            for token, error in tqdm(
                results,
                total=len(tokens),
                desc=archive_path.stem,
            ):
                if error:
                    failures.append((token, error))

        if failures:
            for token, error in failures[:20]:
                print(f"[ERROR] {archive_path.name} {token}: {error}")
            raise RuntimeError(
                f"{archive_path.name}: {len(failures)}/{len(tokens)} tokens failed"
            )

        temporary_marker = marker_path.with_name(
            marker_path.name + f".tmp-{os.getpid()}"
        )
        temporary_marker.write_text(
            f"logs={len(log_names)} tokens={len(tokens)}\n",
            encoding="utf-8",
        )
        os.replace(temporary_marker, marker_path)
        print(
            f"[OK] {archive_path.name}: "
            f"{len(log_names)} logs, {len(tokens)} tokens"
        )
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--datalist", type=Path, required=True)
    parser.add_argument("--meta-dir", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--marker-dir", type=Path, required=True)
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=Path("/tmp/drivedreamer_policy_camera_stage"),
    )
    parser.add_argument("--index-workers", type=int, default=16)
    parser.add_argument("--video-workers", type=int, default=2)
    parser.add_argument("--encoder-preset", default="medium")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.build_index:
        build_index(
            args.datalist,
            args.meta_dir,
            args.index_path,
            args.index_workers,
        )
        return
    if args.archive is None:
        raise ValueError("--archive is required unless --build-index is used")
    process_archive(
        args.archive,
        args.index_path,
        args.meta_dir,
        args.video_dir,
        args.marker_dir,
        args.staging_root,
        args.video_workers,
        args.encoder_preset,
    )


if __name__ == "__main__":
    main()
