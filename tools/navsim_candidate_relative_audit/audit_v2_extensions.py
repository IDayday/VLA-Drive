#!/usr/bin/env python3
"""Audit locally deployed NAVSIM v2 reactive and synthetic extensions.

This module deliberately does not score the deployed v2 evaluation cache: its
recorded Hydra split is ``navtest``, which is outside the allowed supervision
scope of this audit.
"""

from __future__ import annotations

import argparse
import ast
import itertools
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import (
    add_common_arguments,
    append_command,
    ensure_output_dir,
    paths_from_args,
    run_text,
    write_json,
    write_markdown,
)


def _read_version(setup_path: Path) -> str | None:
    if not setup_path.is_file():
        return None
    match = re.search(r"version\s*=\s*['\"]([^'\"]+)", setup_path.read_text(encoding="utf-8"))
    return match.group(1) if match else None


def _read_split_from_hydra(cache_root: Path) -> str | None:
    candidates = [
        cache_root / "metadata/code/hydra/hydra.yaml",
        cache_root / "metadata/code/hydra/config.yaml",
        cache_root / "metadata/code/hydra/overrides.yaml",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"train_test_split(?:/scene_filter)?[=:]\s*([A-Za-z0-9_-]+)", text)
        if match:
            return match.group(1)
    return None


def _numeric_array(value: Any, expected: int = 3) -> np.ndarray | None:
    try:
        parsed = np.asarray(ast.literal_eval(str(value)), dtype=np.float64)
    except (SyntaxError, ValueError):
        return None
    if parsed.shape != (expected,) or not np.isfinite(parsed).all():
        return None
    return parsed


def _pairwise_spread(values: list[np.ndarray]) -> tuple[float, float]:
    distances = [float(np.linalg.norm(a[:2] - b[:2])) for a, b in itertools.combinations(values, 2)]
    return (float(np.mean(distances)), float(np.max(distances))) if distances else (0.0, 0.0)


def _inspect_synthetic_pickles(scene_dir: Path, max_pickles: int = 204) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(scene_dir.glob("*.pkl"))[:max_pickles]:
        try:
            with path.open("rb") as stream:
                data = pickle.load(stream)
            metadata = data.get("scene_metadata", {})
            frames = data.get("frames", [])
            current = frames[-1] if frames else {}
            annotations = [frame.get("annotations") for frame in frames]
            camera_dicts = [frame.get("camera_dict", {}) for frame in frames]
            current_pose = np.asarray(current.get("ego_status", {}).get("ego_pose", [np.nan] * 3), dtype=float)
            current_velocity = np.asarray(current.get("ego_status", {}).get("ego_velocity", [np.nan] * 2), dtype=float)
            rows.append(
                {
                    "pickle_path": str(path),
                    "synthetic_scene_token": metadata.get("scene_token", path.stem),
                    "log_name": metadata.get("log_name"),
                    "corresponding_original_scene": metadata.get("corresponding_original_scene"),
                    "corresponding_original_initial_token": metadata.get("corresponding_original_initial_token"),
                    "frame_count": len(frames),
                    "annotations_all_frames": bool(frames and all(item is not None for item in annotations)),
                    "track_tokens_all_frames": bool(
                        frames and all(item is not None and item.get("track_tokens") is not None for item in annotations)
                    ),
                    "camera_records_all_frames": bool(frames and all(bool(item) for item in camera_dicts)),
                    "extended_track_steps": len(data.get("extended_detections_tracks") or []),
                    "extended_traffic_light_steps": len(data.get("extended_traffic_light_data") or []),
                    "current_x_m": current_pose[0],
                    "current_y_m": current_pose[1],
                    "current_heading_rad": current_pose[2],
                    "current_speed_mps": float(np.linalg.norm(current_velocity)),
                    "current_front_path": str(
                        next(
                            (
                                value.get("data_path")
                                for key, value in current.get("camera_dict", {}).items()
                                if str(key).lower() == "cam_f0" and isinstance(value, dict)
                            ),
                            "",
                        )
                    ),
                    "load_success": True,
                    "load_error": None,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "pickle_path": str(path),
                    "synthetic_scene_token": path.stem,
                    "load_success": False,
                    "load_error": f"{type(exc).__name__}: {exc}",
                }
            )
    frame = pd.DataFrame(rows)
    successful = frame[frame.load_success].copy() if len(frame) else frame
    group_spreads = []
    if len(successful):
        for token, group in successful.groupby("corresponding_original_scene"):
            poses = [np.asarray([row.current_x_m, row.current_y_m, row.current_heading_rad]) for row in group.itertuples()]
            mean_distance, max_distance = _pairwise_spread(poses)
            group_spreads.append(
                {
                    "corresponding_original_scene": token,
                    "synthetic_count": len(group),
                    "current_pose_pairwise_mean_m": mean_distance,
                    "current_pose_pairwise_max_m": max_distance,
                    "unique_current_front_images": group.current_front_path.nunique(),
                    "current_speed_range_mps": float(group.current_speed_mps.max() - group.current_speed_mps.min()),
                }
            )
    summary = {
        "pickle_count_audited": len(frame),
        "pickle_load_success_rate": float(frame.load_success.mean()) if len(frame) else 0.0,
        "four_history_frame_rate": float((successful.frame_count == 4).mean()) if len(successful) else 0.0,
        "annotations_all_frames_rate": float(successful.annotations_all_frames.mean()) if len(successful) else 0.0,
        "track_tokens_all_frames_rate": float(successful.track_tokens_all_frames.mean()) if len(successful) else 0.0,
        "camera_records_all_frames_rate": float(successful.camera_records_all_frames.mean()) if len(successful) else 0.0,
        "extended_tracks_at_least_8_steps_rate": float((successful.extended_track_steps >= 8).mean()) if len(successful) else 0.0,
        "extended_traffic_lights_at_least_8_steps_rate": float((successful.extended_traffic_light_steps >= 8).mean()) if len(successful) else 0.0,
        "extended_track_step_distribution": {
            str(int(key)): int(value) for key, value in successful.extended_track_steps.value_counts().sort_index().items()
        } if len(successful) else {},
        "group_spreads": group_spreads,
    }
    return frame, summary


def audit(args: argparse.Namespace) -> dict[str, Any]:
    paths = paths_from_args(args)
    output_dir = ensure_output_dir(args.output_dir)
    v2_root = paths.v2_devkit_root
    scene_dir = paths.synthetic_scene_path
    sensor_dir = paths.synthetic_sensor_path
    csv_path = scene_dir.parent / "synthetic_scenes_attributes.csv" if scene_dir else None
    v2_available = bool(v2_root and (v2_root / "navsim").is_dir())
    synthetic_available = bool(scene_dir and scene_dir.is_dir() and csv_path and csv_path.is_file())

    idm_path = v2_root / "navsim/traffic_agents_policies/navsim_IDM_traffic_agents.py" if v2_root else Path()
    abstract_path = v2_root / "navsim/traffic_agents_policies/abstract_traffic_agents_policy.py" if v2_root else Path()
    idm_text = idm_path.read_text(encoding="utf-8") if idm_path.is_file() else ""
    abstract_text = abstract_path.read_text(encoding="utf-8") if abstract_path.is_file() else ""
    reactive_code_available = "NavsimIDMTrafficAgents" in idm_text and "simulate_traffic_agents" in idm_text
    vehicles_only = "return [TrackedObjectType.VEHICLE]" in idm_text
    remaining_log_replayed = "remaining_object_detections_tracks" in abstract_text and "future_tracked_objects" in abstract_text

    v2_cache = Path("/mnt/project/DriveDreamer-Policy/navsim_exp/eval_v2/metric_cache")
    cache_split = _read_split_from_hydra(v2_cache) if v2_cache.is_dir() else None
    eligible_reactive_cache = cache_split in {"mini", "trainval", "navtrain"}
    reactive_empirical_run = False
    reactive_blocker = (
        f"Only deployed v2 cache records train_test_split={cache_split!r}; test/navtest labels are excluded."
        if not eligible_reactive_cache
        else "Reactive candidate comparison was not requested by the current traffic-policy setting."
    )

    attributes = pd.DataFrame()
    synthetic_rows = pd.DataFrame()
    synthetic_summary: dict[str, Any] = {}
    sensor_coverage = 0.0
    camera_sensor_coverage = 0.0
    lidar_sensor_coverage = 0.0
    viewpoint_match_error: list[float] = []
    original_log_deployed_rate = 0.0
    if synthetic_available and csv_path is not None and scene_dir is not None:
        attributes = pd.read_csv(csv_path)
        sensor_columns = [column for column in attributes if column.endswith("_path")]
        exists = []
        camera_exists: list[bool] = []
        lidar_exists: list[bool] = []
        if sensor_dir and sensor_dir.is_dir():
            for column in sensor_columns:
                column_exists = [(sensor_dir / str(value)).is_file() for value in attributes[column].dropna()]
                exists.extend(column_exists)
                (lidar_exists if "lidar" in column.lower() else camera_exists).extend(column_exists)
        sensor_coverage = float(np.mean(exists)) if exists else 0.0
        camera_sensor_coverage = float(np.mean(camera_exists)) if camera_exists else 0.0
        lidar_sensor_coverage = float(np.mean(lidar_exists)) if lidar_exists else 0.0
        synthetic_rows, synthetic_summary = _inspect_synthetic_pickles(scene_dir, args.max_synthetic_scenes)
        successful = synthetic_rows[synthetic_rows.load_success]
        by_token = successful.set_index("synthetic_scene_token") if len(successful) else pd.DataFrame()
        for row in attributes.itertuples():
            viewpoint = _numeric_array(row.viewpoint)
            if viewpoint is not None and len(by_token) and row.synthetic_scene_token in by_token.index:
                item = by_token.loc[row.synthetic_scene_token]
                current_pose = np.asarray([item.current_x_m, item.current_y_m, item.current_heading_rad])
                viewpoint_match_error.append(float(np.linalg.norm(viewpoint[:2] - current_pose[:2])))
        original_logs = attributes.log_name.dropna().unique().tolist()
        deployed = [(paths.log_path / f"{log_name}.pkl").is_file() for log_name in original_logs]
        original_log_deployed_rate = float(np.mean(deployed)) if deployed else 0.0
        synthetic_rows.to_csv(output_dir / "synthetic_scene_inventory.csv", index=False)

    commit = run_text(["git", "-C", str(v2_root), "rev-parse", "HEAD"]) if v2_root else None
    group_spreads = synthetic_summary.get("group_spreads", [])
    nonidentical_state_rate = (
        float(np.mean([item["current_pose_pairwise_max_m"] > 1e-3 for item in group_spreads]))
        if group_spreads
        else 0.0
    )
    result = {
        "v2_devkit_available": v2_available,
        "v2_version": _read_version(v2_root / "setup.py") if v2_root else None,
        "v2_git_commit": commit,
        "reactive_policy_code_available": reactive_code_available,
        "reactive_simulated_object_types": ["VEHICLE"] if vehicles_only else [],
        "remaining_object_types_use_log_replay": remaining_log_replayed,
        "deployed_v2_cache_path": str(v2_cache),
        "deployed_v2_cache_split": cache_split,
        "eligible_train_reactive_cache_available": eligible_reactive_cache,
        "reactive_empirical_run": reactive_empirical_run,
        "reactive_blocker": reactive_blocker,
        "synthetic_scenes_available": synthetic_available,
        "synthetic_scene_count": int(len(attributes)),
        "synthetic_unique_original_scenes": int(attributes.corresponding_original_scene_token.nunique()) if len(attributes) else 0,
        "synthetic_followups_per_original": {
            "min": int(attributes.groupby("corresponding_original_scene_token").size().min()) if len(attributes) else 0,
            "median": float(attributes.groupby("corresponding_original_scene_token").size().median()) if len(attributes) else 0.0,
            "max": int(attributes.groupby("corresponding_original_scene_token").size().max()) if len(attributes) else 0,
        },
        "synthetic_sensor_file_coverage": sensor_coverage,
        "synthetic_camera_file_coverage": camera_sensor_coverage,
        "synthetic_lidar_file_coverage": lidar_sensor_coverage,
        "synthetic_current_pose_matches_csv_viewpoint_max_error_m": max(viewpoint_match_error, default=None),
        "mapped_original_logs_present_in_allowed_trainval_path_rate": original_log_deployed_rate,
        "same_original_groups_with_nonidentical_synthetic_current_pose_rate": nonidentical_state_rate,
        "synthetic_pickle_inventory": synthetic_summary,
        "interpretation": (
            "Synthetic follow-up scenes provide neighborhood-state augmentation / weak multi-future supervision only. "
            "They do not establish identical-current-observation, different-action ground-truth futures."
        ),
        "code_evidence": [str(idm_path), str(abstract_path)],
    }
    write_json(output_dir / "v2_extension_audit.json", result)
    report = f"""# NAVSIM v2 Reactive and Synthetic Extension Audit

## Reactive traffic policy

- v2 devkit/version/commit: `{v2_available}` / `{result['v2_version']}` / `{commit}`
- IDM traffic policy code present: `{reactive_code_available}`
- Simulated object types: `{', '.join(result['reactive_simulated_object_types']) or 'not resolved'}`
- Remaining object types merged from logged future: `{remaining_log_replayed}`
- Deployed v2 metric-cache split: `{cache_split}`
- Eligible mini/trainval reactive cache available: `{eligible_reactive_cache}`
- Candidate-level reactive empirical comparison run: `{reactive_empirical_run}`
- Blocker: {reactive_blocker}

Consequently, this deployment supports the reactive mechanism in code, but the audit does not report actor endpoint/speed/braking deltas as measured training evidence.  Running those metrics on the only configured `navtest` cache would violate the split constraint.  The implementation is vehicle-only; pedestrians and other types remain log replay.

## Synthetic follow-up scenes

- CSV/pickle scenes: {len(attributes)} / {synthetic_summary.get('pickle_count_audited', 0)}
- Unique corresponding original scenes: {result['synthetic_unique_original_scenes']}
- Follow-ups per original (min/median/max): {result['synthetic_followups_per_original']['min']} / {result['synthetic_followups_per_original']['median']} / {result['synthetic_followups_per_original']['max']}
- Pickle load success: {synthetic_summary.get('pickle_load_success_rate', 0.0):.3%}
- Four history frames / annotations / track tokens: {synthetic_summary.get('four_history_frame_rate', 0.0):.3%} / {synthetic_summary.get('annotations_all_frames_rate', 0.0):.3%} / {synthetic_summary.get('track_tokens_all_frames_rate', 0.0):.3%}
- Extended tracks / traffic lights with at least 8 steps: {synthetic_summary.get('extended_tracks_at_least_8_steps_rate', 0.0):.3%} / {synthetic_summary.get('extended_traffic_lights_at_least_8_steps_rate', 0.0):.3%}
- Referenced synthetic camera / LiDAR / combined sensor file coverage: {camera_sensor_coverage:.3%} / {lidar_sensor_coverage:.3%} / {sensor_coverage:.3%}
- Same-original groups with non-identical synthetic current poses: {nonidentical_state_rate:.3%}
- Corresponding original log availability in the allowed trainval log directory: {original_log_deployed_rate:.3%}

The synthetic scene current pose agrees with its CSV `viewpoint` (maximum measured XY error `{max(viewpoint_match_error, default=float('nan')):.6g} m`), but different follow-ups mapped to one original have different synthetic current states and image paths.  Their start-state offset from the unavailable corresponding original warmup log cannot be reliably computed in this deployment.  These are synthetic follow-up scenes suitable for neighborhood-state augmentation or weak multi-future supervision, not real counterfactual futures from one identical current observation.
"""
    write_markdown(output_dir / "V2_EXTENSION_REPORT.md", report)
    append_command(output_dir, "python -m tools.navsim_candidate_relative_audit.audit_v2_extensions " + " ".join(sys.argv[1:]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--max-synthetic-scenes", type=int, default=204)
    args = parser.parse_args()
    audit(args)


if __name__ == "__main__":
    main()
