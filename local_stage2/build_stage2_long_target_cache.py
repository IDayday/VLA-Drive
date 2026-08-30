#!/usr/bin/env python3
"""Build a non-destructive Stage-2 cache with the released long target.

The public agent code contains a second trajectory target that warps the
eight proposal timestamps over a longer logged trajectory. The released
inference YAML disables it, so the ordinary feature cache contains only the
four-second target. This utility reconstructs that target from raw training
logs while symlinking the existing image features. It never modifies the
source cache.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
from typing import Any

import numpy as np
from pyquaternion import Quaternion
from scipy.interpolate import CubicSpline
import torch

from nuplan.common.actor_state.state_representation import StateSE2
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)


DEFAULT_SOURCE_CACHE = Path(
    "/mnt/project/DriveVLA-M0-stage2/cache/feature_cache_navtrain_full"
)
DEFAULT_RAW_LOGS = Path("/mnt/navsim/trainval_navsim_logs/trainval")
DEFAULT_OUTPUT_CACHE = Path(
    "/mnt/project/DriveVLA-M0-stage2/cache/feature_cache_navtrain_long2"
)


def build_long_trajectory(
    logged_trajectory: np.ndarray,
    *,
    num_poses: int = 8,
    additional_poses: int = 2,
) -> np.ndarray:
    """Reproduce ``DriveVLATargetBuilder``'s long-target interpolation."""

    logged_trajectory = np.asarray(logged_trajectory, dtype=np.float64)
    required_poses = num_poses + additional_poses
    if additional_poses <= 0:
        raise ValueError("additional_poses must be positive")
    if logged_trajectory.shape != (required_poses, 3):
        raise ValueError(
            f"expected logged trajectory shape {(required_poses, 3)}, "
            f"got {logged_trajectory.shape}"
        )

    # Keep these dtypes and operations identical to the released builder.
    x = np.arange(logged_trajectory.shape[0], dtype=np.float32)
    alpha = 2 * additional_poses / (num_poses * (num_poses + 1))
    x_new = np.arange(num_poses, dtype=np.float32)
    offsets = np.cumsum((x_new + 1) * alpha)
    x_new += offsets
    return np.stack(
        [CubicSpline(x, logged_trajectory[:, axis])(x_new) for axis in range(3)],
        axis=1,
    )


def _global_pose(frame: dict[str, Any]) -> np.ndarray:
    translation = frame["ego2global_translation"]
    quaternion = Quaternion(*frame["ego2global_rotation"])
    return np.asarray(
        [translation[0], translation[1], quaternion.yaw_pitch_roll[0]],
        dtype=np.float64,
    )


def _relative_future(
    frames: list[dict[str, Any]], current_index: int, future_poses: int
) -> np.ndarray:
    poses = np.stack(
        [
            _global_pose(frames[index])
            for index in range(current_index, current_index + future_poses + 1)
        ]
    )
    return convert_absolute_to_relative_se2_array(StateSE2(*poses[0]), poses[1:])


def _atomic_pickle_gzip(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with gzip.open(temporary, "wb", compresslevel=1) as stream:
            pickle.dump(value, stream)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_symlink(source: Path, destination: Path) -> None:
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise RuntimeError(f"wrong existing symlink: {destination}")
        return
    if destination.exists():
        raise FileExistsError(destination)
    destination.symlink_to(source.resolve())


def _build_one_log(task: tuple[str, str, str, str, int, int]) -> dict[str, Any]:
    (
        log_name,
        source_cache_text,
        raw_logs_text,
        output_cache_text,
        num_poses,
        additional_poses,
    ) = task
    source_log = Path(source_cache_text) / log_name
    raw_log = Path(raw_logs_text) / f"{log_name}.pkl"
    output_log = Path(output_cache_text) / log_name
    if not raw_log.is_file():
        raise FileNotFoundError(raw_log)

    with raw_log.open("rb") as stream:
        frames = pickle.load(stream)
    token_to_index = {frame["token"]: index for index, frame in enumerate(frames)}
    output_log.mkdir(parents=True, exist_ok=True)

    count = 0
    max_standard_error = 0.0
    displacement_sum = 0.0
    displacement_max = 0.0
    required_future = num_poses + additional_poses
    # CacheOnlyDataset maps sampler indices to the filesystem enumeration
    # order.  Preserve the source cache's order so a long-target A/B changes
    # only the target dictionary, not the sample-to-index mapping.
    source_tokens = [path for path in source_log.iterdir() if path.is_dir()]
    for source_token in source_tokens:
        token = source_token.name
        if token not in token_to_index:
            raise KeyError(f"token {token} missing from {raw_log}")
        current_index = token_to_index[token]
        if current_index + required_future >= len(frames):
            raise IndexError(
                f"token {token} has fewer than {required_future} future frames"
            )

        source_target = source_token / "trajectory_target.gz"
        source_feature = source_token / "internvl_feature.gz"
        with gzip.open(source_target, "rb") as stream:
            target = pickle.load(stream)

        relative_future = _relative_future(frames, current_index, required_future)
        standard = np.asarray(target["trajectory"], dtype=np.float64)
        standard_error = float(
            np.max(np.abs(standard - relative_future[:num_poses]))
        )
        max_standard_error = max(max_standard_error, standard_error)
        if standard_error > 1e-7:
            raise RuntimeError(
                f"standard target mismatch for {token}: {standard_error:.3e}"
            )

        long_trajectory = build_long_trajectory(
            relative_future,
            num_poses=num_poses,
            additional_poses=additional_poses,
        )
        displacement = float(
            np.linalg.norm(long_trajectory[-1, :2] - standard[-1, :2])
        )
        displacement_sum += displacement
        displacement_max = max(displacement_max, displacement)

        output_token = output_log / token
        output_token.mkdir(parents=True, exist_ok=True)
        _safe_symlink(source_feature, output_token / source_feature.name)
        target = dict(target)
        target["trajectory_long"] = torch.tensor(long_trajectory)
        target["token"] = token
        destination = output_token / source_target.name
        if destination.exists():
            with gzip.open(destination, "rb") as stream:
                existing = pickle.load(stream)
            if not torch.equal(existing["trajectory_long"], target["trajectory_long"]):
                raise RuntimeError(f"wrong existing target: {destination}")
        else:
            _atomic_pickle_gzip(destination, target)
        count += 1

    source_order = [path.name for path in source_tokens]
    output_order = [path.name for path in output_log.iterdir() if path.is_dir()]
    if output_order != source_order:
        raise RuntimeError(
            f"cache token order differs from source for {log_name}: "
            f"source={len(source_order)}, output={len(output_order)}"
        )
    token_order_sha256 = hashlib.sha256(
        "\n".join(source_order).encode("utf-8")
    ).hexdigest()

    return {
        "log_name": log_name,
        "count": count,
        "max_standard_error": max_standard_error,
        "mean_terminal_displacement": displacement_sum / count if count else 0.0,
        "max_terminal_displacement": displacement_max,
        "source_output_token_order_equal": True,
        "token_order_sha256": token_order_sha256,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-cache", type=Path, default=DEFAULT_SOURCE_CACHE)
    parser.add_argument("--raw-logs", type=Path, default=DEFAULT_RAW_LOGS)
    parser.add_argument("--output-cache", type=Path, default=DEFAULT_OUTPUT_CACHE)
    parser.add_argument("--num-poses", type=int, default=8)
    parser.add_argument("--additional-poses", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-logs", type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source_cache = args.source_cache.resolve()
    raw_logs = args.raw_logs.resolve()
    output_cache = args.output_cache.resolve()
    if not source_cache.is_dir():
        raise FileNotFoundError(source_cache)
    if not raw_logs.is_dir():
        raise FileNotFoundError(raw_logs)
    if output_cache == source_cache:
        raise ValueError("output cache must differ from source cache")
    output_cache.mkdir(parents=True, exist_ok=True)

    log_names = sorted(path.name for path in source_cache.iterdir() if path.is_dir())
    if args.max_logs is not None:
        log_names = log_names[: args.max_logs]
    tasks = [
        (
            log_name,
            str(source_cache),
            str(raw_logs),
            str(output_cache),
            args.num_poses,
            args.additional_poses,
        )
        for log_name in log_names
    ]

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_build_one_log, task): task[0] for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if completed % 25 == 0 or completed == len(futures):
                print(f"LONG_TARGET_PROGRESS logs={completed}/{len(futures)}")

    results.sort(key=lambda item: item["log_name"])
    total_targets = sum(item["count"] for item in results)
    manifest = {
        "schema_version": 1,
        "source_cache": str(source_cache),
        "raw_logs": str(raw_logs),
        "output_cache": str(output_cache),
        "num_poses": args.num_poses,
        "additional_poses": args.additional_poses,
        "nominal_horizon_seconds": args.num_poses * 0.5,
        "long_horizon_seconds": (args.num_poses + args.additional_poses) * 0.5,
        "num_logs": len(results),
        "num_targets": total_targets,
        "max_standard_target_error": max(
            (item["max_standard_error"] for item in results), default=0.0
        ),
        "mean_terminal_displacement": (
            sum(
                item["mean_terminal_displacement"] * item["count"]
                for item in results
            )
            / total_targets
            if total_targets
            else 0.0
        ),
        "max_terminal_displacement": max(
            (item["max_terminal_displacement"] for item in results), default=0.0
        ),
        "logs": results,
    }
    manifest_path = output_cache / "_long_target_cache_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in manifest.items() if key != "logs"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
