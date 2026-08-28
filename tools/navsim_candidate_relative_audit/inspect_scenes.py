#!/usr/bin/env python3
"""Phase 1: audit raw Scene fields, sparse future sensors, and MetricCache."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .common import (
    HORIZONS_S,
    add_common_arguments,
    bootstrap_navsim,
    discover_paths,
    load_metric_cache,
    load_metric_cache_index,
    metric_log_name,
    raw_log_path,
    resolve_horizon_index,
    write_dataframe,
    write_json,
    write_text,
)


MODE_LIMITS = {"smoke": 8, "audit": 64, "statistics": 500}


def describe_value(value: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python_type": f"{type(value).__module__}.{type(value).__qualname__}"
    }
    if isinstance(value, np.ndarray):
        result.update(shape=list(value.shape), dtype=str(value.dtype))
    elif isinstance(value, (list, tuple, dict, set)):
        result["length"] = len(value)
    elif value is None:
        result["is_none"] = True
    return result


def continuity(frames: list[dict[str, Any]]) -> tuple[float, float]:
    token_sets = [
        set(map(str, frame["anns"].get("track_tokens", []))) for frame in frames
    ]
    transition_denominator = 0
    transition_numerator = 0
    for left, right in zip(token_sets[:-1], token_sets[1:]):
        transition_denominator += len(left)
        transition_numerator += len(left & right)
    all_future = set.intersection(*token_sets) if token_sets else set()
    first = token_sets[0] if token_sets else set()
    return (
        transition_numerator / transition_denominator
        if transition_denominator
        else float("nan"),
        len(all_future) / len(first) if first else float("nan"),
    )


def cache_continuity(tracks: list[Any]) -> float:
    sets = [
        {
            str(obj.metadata.track_token)
            for obj in detections.tracked_objects.tracked_objects
        }
        for detections in tracks
    ]
    denominator = sum(len(left) for left in sets[:-1])
    numerator = sum(len(left & right) for left, right in zip(sets[:-1], sets[1:]))
    return numerator / denominator if denominator else float("nan")


def sensor_record(
    window: list[dict[str, Any]],
    sensor_root: Path,
    *,
    load_images: bool,
) -> dict[str, Any]:
    current = 3
    timestamps = [frame["timestamp"] for frame in window[current:]]
    relative_timestamps = np.asarray(timestamps, dtype=np.float64)
    target_indices = [
        current + resolve_horizon_index(relative_timestamps, horizon)
        for horizon in HORIZONS_S
    ]
    indices = [current, *target_indices]
    camera_exists: list[bool] = []
    lidar_exists: list[bool] = []
    image_details: list[dict[str, Any]] = []
    for frame_index in indices:
        frame = window[frame_index]
        camera = frame["cams"]["CAM_F0"]
        camera_path = sensor_root / camera["data_path"]
        lidar_path = sensor_root / frame["lidar_path"]
        camera_exists.append(camera_path.is_file())
        lidar_exists.append(lidar_path.is_file())
        detail: dict[str, Any] = {
            "frame_index": frame_index,
            "timestamp": int(frame["timestamp"]),
            "relative_camera_path": str(camera["data_path"]),
            "exists": camera_path.is_file(),
            "intrinsics_shape": list(np.asarray(camera.get("cam_intrinsic")).shape),
            "extrinsics_rotation_shape": list(
                np.asarray(camera.get("sensor2lidar_rotation")).shape
            ),
            "extrinsics_translation_shape": list(
                np.asarray(camera.get("sensor2lidar_translation")).shape
            ),
        }
        if load_images and camera_path.is_file():
            with Image.open(camera_path) as image:
                detail.update(image_size=list(image.size), image_mode=image.mode)
        image_details.append(detail)
    return {
        "target_window_indices": indices,
        "camera_exists": camera_exists,
        "lidar_exists": lidar_exists,
        "camera_coverage": float(np.mean(camera_exists)),
        "lidar_coverage": float(np.mean(lidar_exists)),
        "image_details": image_details,
    }


def summarize_metric_cache(cache: Any) -> dict[str, Any]:
    attributes = {name: describe_value(value) for name, value in vars(cache).items()}
    observation_attrs = {
        name: describe_value(value)
        for name, value in vars(cache.observation).items()
        if "light" in name.lower() or "red" in name.lower() or "detect" in name.lower()
    }
    current = cache.current_tracked_objects[0].tracked_objects.tracked_objects
    sample_object = None
    if current:
        obj = current[0]
        sample_object = {
            "class": f"{type(obj).__module__}.{type(obj).__qualname__}",
            "track_token": str(obj.metadata.track_token),
            "instance_token": str(getattr(obj, "instance_token", "")),
            "object_type": str(obj.tracked_object_type),
            "center_global": [obj.center.x, obj.center.y, obj.center.heading],
            "velocity_global": [
                getattr(obj.velocity, "x", 0.0),
                getattr(obj.velocity, "y", 0.0),
            ]
            if hasattr(obj, "velocity")
            else None,
            "box_lwh_m": [obj.box.length, obj.box.width, obj.box.height],
        }
    return {
        "class": f"{type(cache).__module__}.{type(cache).__qualname__}",
        "attributes": attributes,
        "observation_traffic_light_evidence": observation_attrs,
        "sample_current_object": sample_object,
        "future_track_count": len(cache.future_tracked_objects),
        "observation_track_count": len(cache.observation.detections_tracks),
    }


def inspect(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split = "trainval" if args.split == "train" else args.split
    paths = discover_paths(args, split=split)
    runtime = bootstrap_navsim(paths)
    if (
        paths.metric_cache is None
        or paths.logs_root is None
        or paths.sensors_root is None
    ):
        raise FileNotFoundError("metric cache, raw logs, and sensor roots are required")
    from navsim.common.dataclasses import Scene, SensorConfig

    metric_index = load_metric_cache_index(paths.metric_cache)
    rng = np.random.default_rng(args.seed)
    available = np.asarray(sorted(metric_index), dtype=object)
    rng.shuffle(available)
    tokens = [str(token) for token in available[: min(args.max_scenes, len(available))]]
    grouped: dict[str, list[str]] = defaultdict(list)
    for token in tokens:
        grouped[metric_log_name(metric_index[token])].append(token)

    rows: list[dict[str, Any]] = []
    object_types: Counter[str] = Counter()
    errors: list[dict[str, str]] = []
    inventory_samples: dict[str, Any] = {}
    image_evidence: list[dict[str, Any]] = []
    scene_build_budget = min(args.scene_object_samples, args.max_scenes)
    built_scene_count = 0
    for log_name in sorted(grouped):
        try:
            with raw_log_path(paths.logs_root, log_name).open("rb") as stream:
                raw_frames = pickle.load(stream)
            positions = {
                str(frame["token"]): index for index, frame in enumerate(raw_frames)
            }
        except Exception as error:
            for token in grouped[log_name]:
                errors.append(
                    {
                        "scene_token": token,
                        "stage": "raw_log",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            continue
        for token in grouped[log_name]:
            try:
                current_position = positions[token]
                start, stop = current_position - 3, current_position + 9
                if start < 0 or stop > len(raw_frames):
                    raise IndexError(
                        "scene lacks four history plus eight future frames"
                    )
                window = list(raw_frames[start:stop])
                cache = load_metric_cache(metric_index[token])
                timestamps = np.asarray(
                    [frame["timestamp"] for frame in window], dtype=np.int64
                )
                delta_s = np.diff(timestamps).astype(np.float64) / 1e6
                future = window[4:]
                current = window[3]
                future_actor_counts = [
                    len(frame["anns"]["gt_boxes"]) for frame in future
                ]
                velocity_coverage = [
                    len(frame["anns"].get("gt_velocity_3d", []))
                    == len(frame["anns"].get("gt_boxes", []))
                    for frame in future
                ]
                track_coverage = [
                    len(frame["anns"].get("track_tokens", []))
                    == len(frame["anns"].get("gt_boxes", []))
                    for frame in future
                ]
                transition_continuity, full_continuity = continuity([current, *future])
                sensor = sensor_record(
                    window,
                    paths.sensors_root,
                    load_images=len(image_evidence) < args.sensor_scene_samples,
                )
                if len(image_evidence) < args.sensor_scene_samples:
                    image_evidence.append({"scene_token": token, **sensor})
                object_types.update(
                    str(name)
                    for frame in future
                    for name in frame["anns"].get("gt_names", [])
                )
                map_available = bool(
                    cache.map_parameters.map_root and cache.map_parameters.map_name
                )
                route_available = bool(
                    cache.route_lane_ids and current.get("roadblock_ids")
                )
                traffic_field = all("traffic_lights" in frame for frame in future)
                traffic_nonempty = any(
                    bool(frame["traffic_lights"]) for frame in future
                )
                scene_gt_ok = False
                scene_gt_shape = None
                scene_object_attempted = False
                if built_scene_count < scene_build_budget:
                    scene_object_attempted = True
                    scene = Scene.from_scene_dict_list(
                        window,
                        paths.sensors_root,
                        num_history_frames=4,
                        num_future_frames=8,
                        sensor_config=SensorConfig.build_no_sensors(),
                    )
                    trajectory = scene.get_future_trajectory()
                    scene_gt_shape = list(trajectory.poses.shape)
                    scene_gt_ok = (
                        trajectory.poses.shape == (8, 3)
                        and np.isfinite(trajectory.poses).all()
                    )
                    built_scene_count += 1
                    if not inventory_samples:
                        inventory_samples = {
                            "Scene": {
                                "class": f"{type(scene).__module__}.{type(scene).__qualname__}",
                                "scene_metadata": {
                                    name: describe_value(value)
                                    for name, value in vars(
                                        scene.scene_metadata
                                    ).items()
                                },
                                "frame": {
                                    name: describe_value(value)
                                    for name, value in vars(scene.frames[3]).items()
                                },
                                "annotations": {
                                    name: describe_value(value)
                                    for name, value in vars(
                                        scene.frames[3].annotations
                                    ).items()
                                },
                                "sensor_config": f"{SensorConfig.__module__}.{SensorConfig.__qualname__}",
                            },
                            "MetricCache": summarize_metric_cache(cache),
                        }
                row = {
                    "scene_token": token,
                    "log_name": log_name,
                    "split": args.split,
                    "metric_cache_loaded": True,
                    "scene_object_attempted": scene_object_attempted,
                    "scene_object_loaded": scene_gt_ok
                    if scene_object_attempted
                    else None,
                    "gt_future_4s_available": len(future) == 8,
                    "gt_future_shape": json.dumps(scene_gt_shape),
                    "timestamp_delta_mean_s": float(np.mean(delta_s)),
                    "timestamp_delta_min_s": float(np.min(delta_s)),
                    "timestamp_delta_max_s": float(np.max(delta_s)),
                    "future_horizon_s": float((timestamps[-1] - timestamps[3]) / 1e6),
                    "current_ego_pose_available": len(
                        current.get("ego2global_translation", [])
                    )
                    >= 2,
                    "current_ego_dynamic_state_available": len(
                        current.get("ego_dynamic_state", [])
                    )
                    >= 4,
                    "future_actor_frames_available": all(
                        count >= 0 for count in future_actor_counts
                    ),
                    "future_actor_mean_count": float(np.mean(future_actor_counts)),
                    "future_actor_velocity_coverage": float(np.mean(velocity_coverage)),
                    "future_actor_track_token_coverage": float(np.mean(track_coverage)),
                    "raw_track_transition_continuity": transition_continuity,
                    "raw_current_track_full_horizon_continuity": full_continuity,
                    "metric_track_transition_continuity": cache_continuity(
                        cache.future_tracked_objects
                    ),
                    "metric_future_track_frames": len(cache.future_tracked_objects),
                    "metric_future_track_frequency_hz": len(
                        cache.future_tracked_objects
                    )
                    / 4.0,
                    "future_cam_f0_coverage": sensor["camera_coverage"],
                    "future_lidar_coverage": sensor["lidar_coverage"],
                    "camera_intrinsics_available": all(
                        np.asarray(frame["cams"]["CAM_F0"].get("cam_intrinsic")).shape
                        == (3, 3)
                        for frame in [current, *future]
                    ),
                    "camera_extrinsics_available": all(
                        len(frame["cams"]["CAM_F0"].get("sensor2lidar_translation", []))
                        == 3
                        for frame in [current, *future]
                    ),
                    "annotations_available": all("anns" in frame for frame in future),
                    "traffic_light_field_available": traffic_field,
                    "traffic_light_nonempty": traffic_nonempty,
                    "map_available": map_available,
                    "route_available": route_available,
                    "route_lane_id_count": len(cache.route_lane_ids),
                    "roadblock_id_count": len(current.get("roadblock_ids", [])),
                    "centerline_available": getattr(cache.centerline, "length", 0.0)
                    > 0,
                    "drivable_area_available": bool(
                        getattr(cache.drivable_area_map, "tokens", [])
                    ),
                    "metric_scene_type": str(cache.scene_type),
                    "raw_metric_current_actor_count_delta": len(
                        current["anns"]["gt_boxes"]
                    )
                    - len(
                        cache.current_tracked_objects[0].tracked_objects.tracked_objects
                    ),
                    "raw_metric_future_actor_count_mean_abs_delta": float(
                        np.mean(
                            [
                                abs(
                                    len(future[min(index // 5, 7)]["anns"]["gt_boxes"])
                                    - len(detections.tracked_objects.tracked_objects)
                                )
                                for index, detections in enumerate(
                                    cache.future_tracked_objects
                                )
                            ]
                        )
                    ),
                }
                rows.append(row)
            except Exception as error:
                errors.append(
                    {
                        "scene_token": token,
                        "stage": "scene",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )

    frame = pd.DataFrame(rows)
    coverage: dict[str, Any] = {
        "requested_scene_count": args.max_scenes,
        "audited_scene_count": len(rows),
        "failure_count": len(errors),
        "runtime": runtime,
        "paths": paths.to_json(),
        "field_coverage": {},
        "object_type_counts": dict(object_types.most_common()),
        "sensor_evidence": image_evidence,
        "evidence_scene_tokens": [row["scene_token"] for row in rows[:16]],
        "errors": errors,
        "inventory_samples": inventory_samples,
    }
    if not frame.empty:
        for column in frame.columns:
            if frame[column].dtype == bool:
                coverage["field_coverage"][column] = float(frame[column].mean())
        for column in (
            "future_cam_f0_coverage",
            "future_lidar_coverage",
            "future_actor_velocity_coverage",
            "future_actor_track_token_coverage",
            "raw_track_transition_continuity",
            "metric_track_transition_continuity",
        ):
            coverage["field_coverage"][column] = float(frame[column].mean())
        attempted = frame["scene_object_attempted"].astype(bool)
        coverage["field_coverage"]["scene_object_attempt_coverage"] = float(
            attempted.mean()
        )
        coverage["field_coverage"]["scene_object_loaded_given_attempted"] = (
            float(frame.loc[attempted, "scene_object_loaded"].astype(bool).mean())
            if attempted.any()
            else 0.0
        )
        coverage["statistics"] = {
            "timestamp_delta_s": frame["timestamp_delta_mean_s"]
            .describe(percentiles=[0.5, 0.95, 0.99])
            .to_dict(),
            "future_actor_count": frame["future_actor_mean_count"]
            .describe(percentiles=[0.5, 0.95, 0.99])
            .to_dict(),
            "raw_track_continuity": frame["raw_track_transition_continuity"]
            .describe(percentiles=[0.5, 0.95, 0.99])
            .to_dict(),
        }
    return rows, coverage


def render_report(coverage: dict[str, Any]) -> str:
    values = coverage.get("field_coverage", {})
    return "\n".join(
        [
            "# NAVSIM Scene and MetricCache Field Audit",
            "",
            f"- Audited scenes: **{coverage['audited_scene_count']} / {coverage['requested_scene_count']}**",
            f"- Failures: **{coverage['failure_count']}**",
            f"- Runtime NAVSIM: `{coverage['runtime'].get('setup_version')}` from `{coverage['runtime'].get('import_path')}`",
            f"- GT future coverage: **{values.get('gt_future_4s_available', 0):.3%}**",
            f"- Full `Scene` construction: **{values.get('scene_object_loaded_given_attempted', 0):.3%}** of the intentionally sampled **{values.get('scene_object_attempt_coverage', 0):.3%}** subset",
            f"- Future front-camera coverage at current/0.5/1/2/4 s: **{values.get('future_cam_f0_coverage', 0):.3%}**",
            f"- Future LiDAR coverage at the same horizons: **{values.get('future_lidar_coverage', 0):.3%}**",
            f"- Raw track-token field coverage: **{values.get('future_actor_track_token_coverage', 0):.3%}**",
            f"- Raw adjacent-frame track continuity: **{values.get('raw_track_transition_continuity', 0):.3%}**",
            f"- MetricCache adjacent 10 Hz track continuity: **{values.get('metric_track_transition_continuity', 0):.3%}**",
            f"- Map coverage: **{values.get('map_available', 0):.3%}**",
            f"- Route coverage: **{values.get('route_available', 0):.3%}**",
            f"- Traffic-light field coverage: **{values.get('traffic_light_field_available', 0):.3%}**",
            "",
            "Raw annotations are ego-local at each logged frame; the official local code converts them to global objects in `navsim/planning/scenario_builder/navsim_scenario_utils.py`. MetricCache future tracks are the official 10 Hz interpolation of those logged 2 Hz annotations.",
            "",
            "Future camera and LiDAR checks only touch `cam_f0` and the requested sparse horizons. No all-camera/all-frame sensor materialization was performed.",
            "",
            "Evidence tokens: "
            + ", ".join(
                f"`{token}`" for token in coverage.get("evidence_scene_tokens", [])[:8]
            ),
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--mode", choices=tuple(MODE_LIMITS), default="smoke")
    parser.add_argument("--scene-object-samples", type=int, default=8)
    parser.add_argument("--sensor-scene-samples", type=int, default=4)
    args = parser.parse_args()
    if "--max-scenes" not in __import__("sys").argv:
        args.max_scenes = MODE_LIMITS[args.mode]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, inventory = inspect(args)
    write_json(args.output_dir / "field_inventory.json", inventory)
    write_dataframe(pd.DataFrame(rows), args.output_dir / "scene_coverage.csv")
    write_text(args.output_dir / "FIELD_AUDIT.md", render_report(inventory))
    print(
        json.dumps(
            {
                "mode": args.mode,
                "audited": inventory["audited_scene_count"],
                "failures": inventory["failure_count"],
                "output": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
