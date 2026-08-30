#!/usr/bin/env python3
"""Compare the Stage-2 tensor cache against the underlying NAVSIM logs.

This is deliberately a read-only audit.  It samples tokens across independent
logs, reconstructs the exact four-history/ten-future NAVSIM window used by the
released training configuration, and compares every cached Stage-2 input and
trajectory target with a fresh computation from the raw log.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import pickle
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from nuplan.common.actor_state.state_representation import StateSE2
from pyquaternion import Quaternion

from navsim.agents.EpisodeDrive.drivevla_features import DriveVLAFeatureBuilder
from navsim.common.dataclasses import AgentInput, SensorConfig
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)


DEFAULT_CACHE_ROOT = Path(
    "/mnt/project/DriveVLA-M0-stage2/cache/feature_cache_navtrain_full"
)
DEFAULT_LOG_ROOT = Path("/mnt/project/onevl_navsim_data/navsim_logs/trainval")
DEFAULT_SENSOR_ROOT = Path("/mnt/project/onevl_navsim_data/sensor_blobs/trainval")


def _load_gzip(path: Path) -> Dict[str, Any]:
    with gzip.open(path, "rb") as stream:
        return pickle.load(stream)


def _decode_path(path_tensor: torch.Tensor) -> Path:
    return Path("".join(chr(int(value)) for value in path_tensor.flatten().tolist()))


def _select_samples(
    cache_root: Path, sample_count: int, log_count: int, seed: int
) -> List[Path]:
    rng = random.Random(seed)
    log_dirs = sorted(path for path in cache_root.iterdir() if path.is_dir())
    rng.shuffle(log_dirs)
    selected_logs = log_dirs[: min(log_count, sample_count, len(log_dirs))]
    if not selected_logs:
        raise RuntimeError(f"No log directories found below {cache_root}")

    candidates: Dict[Path, List[Path]] = {}
    for log_dir in selected_logs:
        complete = [
            path
            for path in log_dir.iterdir()
            if path.is_dir()
            and (path / "internvl_feature.gz").is_file()
            and (path / "trajectory_target.gz").is_file()
        ]
        rng.shuffle(complete)
        if complete:
            candidates[log_dir] = complete

    selected: List[Path] = []
    while len(selected) < sample_count and candidates:
        for log_dir in list(candidates):
            paths = candidates[log_dir]
            if paths:
                selected.append(paths.pop())
                if len(selected) == sample_count:
                    break
            if not paths:
                candidates.pop(log_dir)
    if len(selected) != sample_count:
        raise RuntimeError(
            f"Requested {sample_count} complete samples, selected {len(selected)}"
        )
    return selected


def _global_pose(frame: Dict[str, Any]) -> np.ndarray:
    translation = frame["ego2global_translation"]
    quaternion = Quaternion(*frame["ego2global_rotation"])
    return np.asarray(
        [translation[0], translation[1], quaternion.yaw_pitch_roll[0]],
        dtype=np.float64,
    )


def _future_trajectory(window: Sequence[Dict[str, Any]]) -> torch.Tensor:
    # Mirrors Scene.get_future_trajectory(num_trajectory_frames=8): current
    # frame is history index three and the eight following frames are targets.
    poses = np.asarray([_global_pose(frame) for frame in window[3:12]], dtype=np.float64)
    relative = convert_absolute_to_relative_se2_array(StateSE2(*poses[0]), poses[1:])
    return torch.tensor(relative)


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        return math.inf
    return float(torch.max(torch.abs(left.to(torch.float64) - right.to(torch.float64))))


def _percentile(values: Iterable[float], percentile: float) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.percentile(array, percentile)) if len(array) else math.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--sensor-root", type=Path, default=DEFAULT_SENSOR_ROOT)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--logs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    sample_dirs = _select_samples(args.cache_root, args.samples, args.logs, args.seed)
    by_log: Dict[str, List[Path]] = {}
    for sample_dir in sample_dirs:
        by_log.setdefault(sample_dir.parent.name, []).append(sample_dir)

    sensor_config = SensorConfig.build_no_sensors()
    sensor_config.cam_f0 = [3]
    builder = DriveVLAFeatureBuilder(cache_hidden_state=False)
    rows: List[Dict[str, Any]] = []

    for log_index, (log_name, paths) in enumerate(sorted(by_log.items()), start=1):
        raw_log_path = args.log_root / f"{log_name}.pkl"
        if not raw_log_path.is_file():
            raise FileNotFoundError(raw_log_path)
        with raw_log_path.open("rb") as stream:
            frames: List[Dict[str, Any]] = pickle.load(stream)

        requested = {path.name: path for path in paths}
        windows: Dict[str, Sequence[Dict[str, Any]]] = {}
        for start in range(0, len(frames) - 14 + 1):
            token = frames[start + 3]["token"]
            if token in requested:
                windows[token] = frames[start : start + 14]
        missing = sorted(set(requested) - set(windows))
        if missing:
            raise RuntimeError(
                f"{len(missing)} cached tokens are absent from raw log {log_name}: "
                f"{missing[:3]}"
            )

        for token, sample_dir in requested.items():
            window = windows[token]
            agent_input = AgentInput.from_scene_dict_list(
                list(window),
                args.sensor_root,
                num_history_frames=4,
                sensor_config=sensor_config,
                load_image_path=True,
            )
            raw_feature = builder.compute_features(agent_input)
            raw_target = _future_trajectory(window)
            cached_feature = _load_gzip(sample_dir / "internvl_feature.gz")
            cached_target = _load_gzip(sample_dir / "trajectory_target.gz")

            cached_path = _decode_path(cached_feature["image_path_tensor"])
            raw_path = _decode_path(raw_feature["image_path_tensor"])
            row = {
                "log_name": log_name,
                "token": token,
                "raw_current_token": window[3]["token"],
                "cached_target_token": cached_target.get("token"),
                "trajectory_max_abs": _max_abs(cached_target["trajectory"], raw_target),
                "history_max_abs": _max_abs(
                    cached_feature["history_trajectory"],
                    raw_feature["history_trajectory"],
                ),
                "command_max_abs": _max_abs(
                    cached_feature["high_command_one_hot"],
                    raw_feature["high_command_one_hot"],
                ),
                "status_max_abs": _max_abs(
                    cached_feature["status_feature"], raw_feature["status_feature"]
                ),
                "image_path_equal": cached_path == raw_path,
                "image_resolved_equal": cached_path.resolve() == raw_path.resolve(),
                "image_exists": cached_path.is_file(),
                "cached_image_path": str(cached_path),
                "raw_image_path": str(raw_path),
            }
            rows.append(row)
        print(f"audited raw logs: {log_index}/{len(by_log)}", flush=True)

    numeric_fields = [
        "trajectory_max_abs",
        "history_max_abs",
        "command_max_abs",
        "status_max_abs",
    ]
    numeric_summary = {
        field: {
            "max": max(float(row[field]) for row in rows),
            "p95": _percentile((float(row[field]) for row in rows), 95),
            "mismatch_count": sum(
                float(row[field]) > args.tolerance for row in rows
            ),
        }
        for field in numeric_fields
    }
    identity_failures = sum(
        row["token"] != row["raw_current_token"]
        or row["token"] != row["cached_target_token"]
        for row in rows
    )
    image_path_failures = sum(not row["image_path_equal"] for row in rows)
    missing_images = sum(not row["image_exists"] for row in rows)
    passed = (
        not identity_failures
        and not image_path_failures
        and not missing_images
        and all(summary["mismatch_count"] == 0 for summary in numeric_summary.values())
    )
    result = {
        "status": "PASS" if passed else "FAIL",
        "seed": args.seed,
        "requested_samples": args.samples,
        "audited_samples": len(rows),
        "audited_logs": len(by_log),
        "history_frames": 4,
        "future_frames_in_scene": 10,
        "trajectory_target_frames": 8,
        "tolerance": args.tolerance,
        "numeric": numeric_summary,
        "identity_failure_count": identity_failures,
        "image_path_failure_count": image_path_failures,
        "missing_image_count": missing_images,
        "cache_root": str(args.cache_root),
        "log_root": str(args.log_root),
        "sensor_root": str(args.sensor_root),
        "failures": [
            row
            for row in rows
            if row["token"] != row["raw_current_token"]
            or row["token"] != row["cached_target_token"]
            or not row["image_path_equal"]
            or not row["image_exists"]
            or any(float(row[field]) > args.tolerance for field in numeric_fields)
        ][:20],
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
