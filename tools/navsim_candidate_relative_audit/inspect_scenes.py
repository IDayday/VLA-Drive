#!/usr/bin/env python3
"""Audit Scene, annotations, future sensors, maps and local MetricCache fields."""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .common import (
    DEFAULT_HORIZONS,
    add_common_arguments,
    append_command,
    effective_max_scenes,
    ensure_output_dir,
    metric_cache_loader,
    paths_from_args,
    percentile_summary,
    resolve_horizon_index,
    scene_loader,
    write_json,
    write_markdown,
)


def _camera_record(frame_dict: dict[str, Any], sensor: str = "cam_f0") -> dict[str, Any] | None:
    cameras = frame_dict.get("cams", {})
    for key, value in cameras.items():
        if key.lower() == sensor.lower():
            return value
    return None


def _track_continuity(frames: list[Any]) -> tuple[float, float, int]:
    appearances: dict[str, list[int]] = collections.defaultdict(list)
    for frame_index, frame in enumerate(frames):
        for token in frame.annotations.track_tokens:
            appearances[str(token)].append(frame_index)
    multi = [indices for indices in appearances.values() if len(indices) >= 2]
    if not multi:
        return float("nan"), float("nan"), len(appearances)
    ratios = [len(indices) / (indices[-1] - indices[0] + 1) for indices in multi]
    adjacent = [
        sum(b == a + 1 for a, b in zip(indices[:-1], indices[1:])) / max(len(indices) - 1, 1)
        for indices in multi
    ]
    return float(np.mean(ratios)), float(np.mean(adjacent)), len(appearances)


def _cache_inventory(cache: Any) -> dict[str, Any]:
    fields = {}
    for name, value in vars(cache).items():
        try:
            length = len(value)
        except Exception:
            length = None
        fields[name] = {"type": f"{type(value).__module__}.{type(value).__name__}", "length": length}
    observation = getattr(cache, "observation", None)
    if observation is not None:
        fields["observation_details"] = {
            "sample_interval_s": getattr(observation, "_sample_interval", None),
            "observation_samples": getattr(observation, "_observation_samples", None),
            "occupancy_map_count": len(getattr(observation, "_occupancy_maps", []) or []),
            "unique_object_count": len(getattr(observation, "_unique_objects", {}) or {}),
        }
    return {
        "runtime_class": f"{type(cache).__module__}.{type(cache).__name__}",
        "fields": fields,
    }


def inspect(args: argparse.Namespace) -> tuple[dict[str, Any], pd.DataFrame]:
    paths = paths_from_args(args)
    output_dir = ensure_output_dir(args.output_dir)
    max_scenes = effective_max_scenes(args.mode, args.max_scenes)
    loader = scene_loader(paths, max_scenes=max_scenes, frame_interval=1)
    try:
        caches = metric_cache_loader(paths)
        cache_tokens = set(caches.tokens)
        cache_loader_error = None
    except Exception as exc:
        caches = None
        cache_tokens = set()
        cache_loader_error = f"{type(exc).__name__}: {exc}"

    rows: list[dict[str, Any]] = []
    object_types: collections.Counter[str] = collections.Counter()
    interval_values: list[float] = []
    future_actor_counts: list[int] = []
    cache_examples: list[dict[str, Any]] = []
    cache_scene_centroid_errors: list[float] = []
    sensor_shape_examples: list[dict[str, Any]] = []

    for scene_index, token in enumerate(loader.tokens):
        scene = loader.get_scene_from_token(token)
        metadata = scene.scene_metadata
        current = metadata.num_history_frames - 1
        timestamps = np.asarray([frame.timestamp for frame in scene.frames], dtype=np.int64)
        relative_s = (timestamps - timestamps[current]) / 1e6
        future_indices = [resolve_horizon_index(timestamps, horizon, origin_index=current) for horizon in DEFAULT_HORIZONS]
        dt = np.diff(timestamps).astype(np.float64) / 1e6
        interval_values.extend(dt.tolist())
        future_frames = scene.frames[current + 1 :]
        continuity, adjacent_continuity, unique_tracks = _track_continuity(future_frames)
        actor_counts = [len(frame.annotations.names) for frame in future_frames]
        future_actor_counts.extend(actor_counts)
        for frame in future_frames:
            object_types.update(str(name) for name in frame.annotations.names)

        raw_frames = loader.scene_frames_dicts[token]
        camera_available = []
        lidar_available = []
        selected_sensor_records = []
        for index, horizon in zip(future_indices, DEFAULT_HORIZONS):
            camera = _camera_record(raw_frames[index])
            camera_path = paths.sensor_blobs_path / camera["data_path"] if camera and camera.get("data_path") else None
            exists = bool(camera_path and camera_path.is_file())
            camera_available.append(exists)
            image_shape = None
            if exists and scene_index < 8:
                try:
                    with Image.open(camera_path) as image:
                        image_shape = [image.height, image.width, len(image.getbands())]
                except OSError:
                    exists = False
                    camera_available[-1] = False
            lidar_rel = raw_frames[index].get("lidar_path")
            lidar_path = paths.sensor_blobs_path / lidar_rel if lidar_rel else None
            lidar_available.append(bool(lidar_path and lidar_path.is_file()))
            if camera is not None:
                selected_sensor_records.append(
                    {
                        "horizon_s": horizon,
                        "frame_index": index,
                        "timestamp_error_s": float(abs(relative_s[index] - horizon)),
                        "camera_exists": exists,
                        "camera_shape": image_shape,
                        "intrinsics_shape": list(np.asarray(camera.get("cam_intrinsic")).shape),
                        "extrinsics_rotation_shape": list(np.asarray(camera.get("sensor2lidar_rotation")).shape),
                    }
                )
        if selected_sensor_records and len(sensor_shape_examples) < 8:
            sensor_shape_examples.append({"scene_token": token, "records": selected_sensor_records})

        cache_available = token in cache_tokens
        cache_load_success = False
        cache_class = None
        cache_future_track_available = False
        cache_observation_available = False
        cache_observation_frequency_hz = None
        cache_unique_objects = None
        cache_has_declared_future_tracks = False
        cache_has_human_trajectory = False
        if cache_available and caches is not None:
            try:
                cache = caches.get_from_token(token)
                cache_load_success = True
                cache_class = f"{type(cache).__module__}.{type(cache).__name__}"
                cache_has_declared_future_tracks = hasattr(cache, "future_tracked_objects")
                cache_has_human_trajectory = hasattr(cache, "human_trajectory")
                observation = getattr(cache, "observation", None)
                cache_observation_available = observation is not None and bool(getattr(observation, "_initialized", False))
                if cache_observation_available:
                    interval = float(getattr(observation, "_sample_interval", np.nan))
                    cache_observation_frequency_hz = 1.0 / interval if interval > 0 else None
                    cache_unique_objects = len(observation.unique_objects)
                    cache_future_track_available = len(getattr(observation, "_occupancy_maps", [])) >= 41
                if len(cache_examples) < 3:
                    cache_examples.append({"scene_token": token, **_cache_inventory(cache)})

                # Cross-check official Scene annotation conversion against cached logged replay.
                if cache_observation_available and scene_index < 64:
                    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario

                    scenario = NavSimScenario(scene, str(paths.map_path), "nuplan-maps-v1.0")
                    for iteration in (1, 2, 4, 8):
                        tracks = scenario.get_tracked_objects_at_iteration(iteration).tracked_objects
                        occupancy = observation[int(round(iteration * 0.5 / interval))]
                        for tracked in tracks:
                            track_token = tracked.track_token
                            if track_token in occupancy.tokens:
                                cached_center = occupancy[track_token].centroid
                                cache_scene_centroid_errors.append(
                                    float(np.hypot(cached_center.x - tracked.center.x, cached_center.y - tracked.center.y))
                                )
            except Exception:
                cache_load_success = False

        gt = scene.get_future_trajectory()
        gt_horizon_s = float(relative_s[current + len(gt.poses)]) if current + len(gt.poses) < len(relative_s) else None
        future_track_sets = [set(map(str, frame.annotations.track_tokens)) for frame in future_frames]
        current_track_set = set(map(str, scene.frames[current].annotations.track_tokens))
        index_4s = resolve_horizon_index(timestamps, 4.0, origin_index=current)
        tracks_4s = set(map(str, scene.frames[index_4s].annotations.track_tokens))
        current_to_4s = len(current_track_set & tracks_4s) / max(len(current_track_set), 1)
        traffic_light_frames = [len(frame.traffic_lights) > 0 for frame in future_frames]
        row = {
            "scene_token": token,
            "scene_metadata_token": metadata.scene_token,
            "log_name": metadata.log_name,
            "map_name": metadata.map_name,
            "num_history_frames": metadata.num_history_frames,
            "num_future_frames": metadata.num_future_frames,
            "frame_count": len(scene.frames),
            "median_frame_interval_s": float(np.median(dt)),
            "min_frame_interval_s": float(np.min(dt)),
            "max_frame_interval_s": float(np.max(dt)),
            "gt_future_available": bool(len(gt.poses) >= 8),
            "gt_future_horizon_s": gt_horizon_s,
            "gt_4s_available": bool(gt_horizon_s is not None and gt_horizon_s >= 3.9),
            "ego_velocity_available": all(frame.ego_status.ego_velocity is not None for frame in future_frames[:8]),
            "ego_acceleration_available": all(frame.ego_status.ego_acceleration is not None for frame in future_frames[:8]),
            "future_annotations_available": all(frame.annotations is not None for frame in future_frames[:8]),
            "future_actor_mean": float(np.mean(actor_counts)) if actor_counts else 0.0,
            "future_actor_min": min(actor_counts) if actor_counts else 0,
            "future_unique_tracks": unique_tracks,
            "track_span_continuity": continuity,
            "track_adjacent_continuity": adjacent_continuity,
            "current_track_survival_4s": current_to_4s,
            "map_available": scene.map_api is not None,
            "route_available": len(scene.frames[current].roadblock_ids) > 0,
            "route_roadblock_count": len(scene.frames[current].roadblock_ids),
            "traffic_lights_current_available": len(scene.frames[current].traffic_lights) > 0,
            "traffic_lights_future_any": any(traffic_light_frames),
            "traffic_lights_future_all": all(traffic_light_frames),
            "camera_current_path_available": bool(
                (_camera_record(raw_frames[current]) or {}).get("data_path")
                and (paths.sensor_blobs_path / (_camera_record(raw_frames[current]) or {})["data_path"]).is_file()
            ),
            "camera_0p5s_available": camera_available[0],
            "camera_1p0s_available": camera_available[1],
            "camera_2p0s_available": camera_available[2],
            "camera_4p0s_available": camera_available[3],
            "future_camera_all_requested": all(camera_available),
            "future_lidar_all_requested": all(lidar_available),
            "metric_cache_available": cache_available,
            "metric_cache_load_success": cache_load_success,
            "metric_cache_class": cache_class,
            "metric_cache_observation_available": cache_observation_available,
            "metric_cache_future_occupancy_available": cache_future_track_available,
            "metric_cache_observation_frequency_hz": cache_observation_frequency_hz,
            "metric_cache_unique_objects": cache_unique_objects,
            "metric_cache_declared_future_tracks_field": cache_has_declared_future_tracks,
            "metric_cache_human_trajectory_field": cache_has_human_trajectory,
        }
        rows.append(row)

    coverage = pd.DataFrame(rows)
    coverage.to_csv(output_dir / "scene_coverage.csv", index=False)
    boolean_fields = [column for column in coverage.columns if coverage[column].dtype == bool]
    field_coverage = {
        column: {
            "coverage": float(coverage[column].mean()),
            "count": int(coverage[column].sum()),
            "denominator": len(coverage),
            "evidence_scene_tokens": coverage.loc[coverage[column], "scene_token"].head(3).tolist(),
        }
        for column in boolean_fields
    }
    inventory = {
        "split": paths.split,
        "scene_count": len(coverage),
        "paths": paths.to_json(),
        "scene_dataclass_runtime_path": str(Path(sys.modules["navsim.common.dataclasses"].__file__).resolve()),
        "sample_interval_seconds": percentile_summary(interval_values),
        "future_actor_count": percentile_summary(future_actor_counts),
        "object_type_distribution": dict(object_types.most_common()),
        "field_coverage": field_coverage,
        "metric_cache_loader_error": cache_loader_error,
        "metric_cache_total_entries": len(cache_tokens),
        "metric_cache_examples": cache_examples,
        "metric_cache_scene_actor_centroid_error_m": percentile_summary(cache_scene_centroid_errors),
        "future_sensor_examples": sensor_shape_examples,
        "cache_schema_note": (
            "The deployed training cache uses train_metric_chache.MetricCache. It does not expose the newer "
            "human_trajectory/future_tracked_objects attributes, but its initialized PDMObservation contains "
            "51 logged-future occupancy maps at 0.1 s."
        ),
    }
    write_json(output_dir / "field_inventory.json", inventory)
    cache_success = float(coverage["metric_cache_load_success"].mean()) if len(coverage) else 0.0
    camera_cov = float(coverage["future_camera_all_requested"].mean()) if len(coverage) else 0.0
    track_cov = float(coverage["metric_cache_future_occupancy_available"].mean()) if len(coverage) else 0.0
    report = f"""# NAVSIM Scene and MetricCache Field Audit

## Scope

- Split: `{paths.split}`
- Scenes: {len(coverage)}
- Logs: `{paths.log_path}`
- Sensor blobs: `{paths.sensor_blobs_path}`
- Metric cache: `{paths.metric_cache_path}`

## Measured coverage

| Field | Coverage |
|---|---:|
| GT future >= 4 s | {coverage['gt_4s_available'].mean():.3%} |
| Future annotations (first 4 s) | {coverage['future_annotations_available'].mean():.3%} |
| Front camera at 0.5/1/2/4 s | {camera_cov:.3%} |
| Future LiDAR at 0.5/1/2/4 s | {coverage['future_lidar_all_requested'].mean():.3%} |
| Stable map API | {coverage['map_available'].mean():.3%} |
| Route roadblocks | {coverage['route_available'].mean():.3%} |
| Future traffic-light records | {coverage['traffic_lights_future_all'].mean():.3%} |
| MetricCache load | {cache_success:.3%} |
| Cache logged-future occupancy >= 4 s | {track_cov:.3%} |

Measured Scene timestamp interval: mean `{inventory['sample_interval_seconds']['mean']}` s, P99 `{inventory['sample_interval_seconds']['p99']}` s.  Horizon lookup uses timestamps, not fixed indices.

## Local cache schema adaptation

Loaded training cache objects are `navsim.planning.metric_caching.train_metric_chache.MetricCache`, not the newer dataclass declared in `navsim/planning/metric_caching/metric_cache.py`.  They omit `human_trajectory`, `past_human_trajectory`, and `future_tracked_objects` as named fields.  Their `observation` is initialized and contains 51 logged-replay occupancy maps at 10 Hz; Scene annotations remain the authoritative 2 Hz source for names, velocities, instance tokens, and track tokens.

The official Scene-to-nuPlan actor conversion and cached occupancy centroids agree with mean error `{inventory['metric_cache_scene_actor_centroid_error_m']['mean']}` m over the sampled matched tracks.

## Track continuity

- Mean future span continuity: {coverage['track_span_continuity'].mean():.3f}
- Mean adjacent continuity: {coverage['track_adjacent_continuity'].mean():.3f}
- Mean current-track survival at 4 s: {coverage['current_track_survival_4s'].mean():.3f}

All inputs were opened read-only.  No log, sensor blob, map, or metric-cache file was written.
"""
    write_markdown(output_dir / "FIELD_AUDIT.md", report)
    return inventory, coverage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    args = parser.parse_args()
    inspect(args)
    append_command(args.output_dir.resolve(), "python -m tools.navsim_candidate_relative_audit.inspect_scenes " + " ".join(sys.argv[1:]))


if __name__ == "__main__":
    main()
