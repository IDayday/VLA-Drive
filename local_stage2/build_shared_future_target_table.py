#!/usr/bin/env python3
"""Aggregate Gate-C shared logged-future actor targets into mmap arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


EXPECTED_STATE_SHAPE = (8, 16, 8)
EXPECTED_MASK_SHAPE = (8, 16)


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _load_target(
    item: Tuple[int, str, str, str],
) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    index, targets_root, log_name, token = item
    path = Path(targets_root) / log_name / f"{token}.npz"
    if not path.is_file():
        return index, None, None, f"missing:{path}"
    try:
        with np.load(path, allow_pickle=False) as payload:
            state = np.asarray(
                payload["shared_actor_future_current_ego"], dtype=np.float32
            )
            mask = np.asarray(payload["shared_actor_future_mask"], dtype=bool)
        if state.shape != EXPECTED_STATE_SHAPE or mask.shape != EXPECTED_MASK_SHAPE:
            return index, None, None, f"shape:{path}:{state.shape}:{mask.shape}"
        if not np.isfinite(state[mask]).all():
            return index, None, None, f"nonfinite:{path}"
        valid_types = state[..., 0][mask]
        if len(valid_types) and (
            valid_types.min() < 0 or valid_types.max() > 2
        ):
            return index, None, None, f"type:{path}"
        return index, state, mask, None
    except Exception as error:  # recorded verbatim in the derived manifest
        return index, None, None, f"decode:{path}:{type(error).__name__}:{error}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--targets-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if not args.metadata.is_file() or not args.targets_root.is_dir():
        raise FileNotFoundError(
            args.metadata if not args.metadata.is_file() else args.targets_root
        )
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    metadata = pd.read_parquet(args.metadata)
    required = {
        "scene_token",
        "log_name",
        "scene_index",
        "target_preflight_available",
    }
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise RuntimeError(f"metadata misses columns: {missing}")
    metadata = metadata.sort_values("scene_index").reset_index(drop=True)
    indices = metadata["scene_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(indices, np.arange(len(metadata), dtype=np.int64)):
        raise RuntimeError("scene_index must be contiguous and ordered")
    tokens = metadata["scene_token"].astype(str).tolist()
    if len(set(tokens)) != len(tokens):
        raise RuntimeError("scene metadata contains duplicate tokens")

    temporary = args.output_dir.with_name(
        f".{args.output_dir.name}.tmp.{os.getpid()}"
    )
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.mkdir(parents=True)
    states = np.lib.format.open_memmap(
        temporary / "shared_actor_future.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(metadata), *EXPECTED_STATE_SHAPE),
    )
    masks = np.lib.format.open_memmap(
        temporary / "shared_actor_mask.npy",
        mode="w+",
        dtype=bool,
        shape=(len(metadata), *EXPECTED_MASK_SHAPE),
    )
    completed = np.lib.format.open_memmap(
        temporary / "completed.npy",
        mode="w+",
        dtype=bool,
        shape=(len(metadata),),
    )
    states[:] = 0
    masks[:] = False
    completed[:] = False
    tasks = (
        (
            int(row.scene_index),
            str(args.targets_root),
            str(row.log_name),
            str(row.scene_token),
        )
        for row in metadata.itertuples(index=False)
    )
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, state, mask, error in executor.map(
            _load_target,
            tasks,
            chunksize=16,
        ):
            if error is not None:
                failures.append(error)
                continue
            states[index] = state
            masks[index] = mask
            completed[index] = True
    states.flush()
    masks.flush()
    completed.flush()
    del states, masks, completed
    shutil.copy2(args.metadata, temporary / "scene_metadata.parquet")
    preflight = metadata["target_preflight_available"].to_numpy(dtype=bool)
    completed_array = np.load(temporary / "completed.npy", mmap_mode="r")
    supervision_valid = completed_array & preflight
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_metadata": str(args.metadata.resolve()),
        "source_metadata_sha256": _sha256(args.metadata),
        "source_targets_root": str(args.targets_root.resolve()),
        "scene_count": len(metadata),
        "valid_scene_count": int(supervision_valid.sum()),
        "failure_count": len(failures),
        "failures": failures[:100],
        "array_sha256": {
            name: _sha256(temporary / name)
            for name in (
                "shared_actor_future.npy",
                "shared_actor_mask.npy",
                "completed.npy",
            )
        },
        "state_shape": [len(metadata), *EXPECTED_STATE_SHAPE],
        "mask_shape": [len(metadata), *EXPECTED_MASK_SHAPE],
        "coordinate_frame": "current_ego",
        "horizons_seconds": [0.5 * (index + 1) for index in range(8)],
        "actor_fields": [
            "object_type_id",
            "x_m",
            "y_m",
            "vx_mps",
            "vy_mps",
            "heading_rad",
            "length_m",
            "width_m",
        ],
        "depends_on_logged_future": True,
        "training_only_target": True,
        "available_as_model_input_at_inference": False,
    }
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary, args.output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
