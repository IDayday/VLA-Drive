#!/usr/bin/env python3
"""Build per-candidate consequences from one shared logged future world."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

from .common import (
    add_common_arguments,
    append_command,
    ensure_output_dir,
    global_to_local,
    load_scenes_for_tokens,
    metric_cache_loader,
    paths_from_args,
    read_parquet,
    stable_hash,
    wrap_angle,
    write_json,
    write_markdown,
    write_parquet,
)


TARGET_TIMES = np.arange(0.5, 4.0 + 1e-8, 0.5, dtype=np.float64)
ENVIRONMENT_FEATURES = (
    "min_actor_polygon_clearance_m",
    "min_actor_center_distance_m",
    "candidate_corridor_actor_count",
    "any_actor_collision",
    "instantaneous_min_ttc_s",
    "drivable_area_valid",
    "oncoming_lane",
    "intersection",
    "centerline_lateral_offset_m",
    "centerline_heading_error_rad",
    "route_progress_m",
    "red_light_polygon_clearance_m",
    "red_light_polygon_intersection",
    "nearest_actor_relative_x_m",
    "nearest_actor_relative_y_m",
)
TRAJECTORY_FEATURES = (
    "candidate_local_x_m",
    "candidate_local_y_m",
    "candidate_local_heading_rad",
    "speed_mps",
    "acceleration_mps2",
    "yaw_rate_radps",
    "curvature_1pm",
    "jerk_mps3",
    "terminal_displacement_m",
)
ACTOR_FEATURES = (
    "object_type_id",
    "relative_x_m",
    "relative_y_m",
    "relative_vx_mps",
    "relative_vy_mps",
    "relative_heading_rad",
    "length_m",
    "width_m",
    "polygon_clearance_m",
    "in_candidate_corridor",
)
SHARED_ACTOR_FEATURES = (
    "object_type_id",
    "global_x_m",
    "global_y_m",
    "global_vx_mps",
    "global_vy_mps",
    "global_heading_rad",
    "length_m",
    "width_m",
)


def _states_from_group(group: pd.DataFrame) -> np.ndarray:
    state_columns = [
        "sim_x_m",
        "sim_y_m",
        "sim_heading_rad",
        "sim_velocity_x_mps",
        "sim_velocity_y_mps",
        "sim_acceleration_x_mps2",
        "sim_acceleration_y_mps2",
        "sim_steering_angle_rad",
        "sim_steering_rate_radps",
        "sim_angular_velocity_radps",
        "sim_angular_acceleration_radps2",
    ]
    result = []
    for row in group.sort_values("candidate_index").itertuples():
        result.append(np.column_stack([getattr(row, column) for column in state_columns]))
    return np.asarray(result, dtype=np.float64)


def _nearest_centerline(line: LineString, point: Point) -> tuple[float, float, float]:
    progress = float(line.project(point))
    nearest = line.interpolate(progress)
    epsilon = min(0.5, max(line.length * 1e-4, 0.05))
    before = line.interpolate(max(0.0, progress - epsilon))
    after = line.interpolate(min(line.length, progress + epsilon))
    heading = math.atan2(after.y - before.y, after.x - before.x)
    return float(point.distance(nearest)), heading, progress


def _red_light_polygons(scene: Any, frame_index: int, geometry_cache: dict[str, Any]) -> list[Any]:
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer

    polygons = []
    for connector_id, is_red in scene.frames[frame_index].traffic_lights:
        if not is_red:
            continue
        key = str(connector_id)
        if key not in geometry_cache:
            try:
                map_object = scene.map_api.get_map_object(key, SemanticMapLayer.LANE_CONNECTOR)
                geometry_cache[key] = map_object.polygon if map_object is not None else None
            except Exception:
                geometry_cache[key] = None
        if geometry_cache[key] is not None:
            polygons.append(geometry_cache[key])
    return polygons


def _track_world_record(track: Any) -> dict[str, Any]:
    velocity = getattr(track, "velocity", None)
    vx = float(velocity.x) if velocity is not None else 0.0
    vy = float(velocity.y) if velocity is not None else 0.0
    return {
        "token": str(track.track_token),
        "token_hash": stable_hash(str(track.track_token)),
        "type_id": int(track.tracked_object_type.value),
        "x": float(track.center.x),
        "y": float(track.center.y),
        "heading": float(track.center.heading),
        "vx": vx,
        "vy": vy,
        "length": float(track.box.length),
        "width": float(track.box.width),
        "polygon": track.box.geometry,
    }


def _trajectory_features(states: np.ndarray, current_pose: np.ndarray) -> np.ndarray:
    # states: K x 41 x 11; sampled target steps are 5,10,...,40.
    indices = np.arange(5, 41, 5)
    sampled = states[:, indices]
    k, h = sampled.shape[:2]
    output = np.zeros((k, h, len(TRAJECTORY_FEATURES)), dtype=np.float32)
    for candidate in range(k):
        local_pose = global_to_local(current_pose, sampled[candidate, :, :3])
        speed = np.linalg.norm(sampled[candidate, :, 3:5], axis=-1)
        acceleration = np.linalg.norm(sampled[candidate, :, 5:7], axis=-1)
        yaw_rate = sampled[candidate, :, 9]
        curvature = np.divide(yaw_rate, np.maximum(speed, 1e-3))
        jerk = np.gradient(acceleration, TARGET_TIMES)
        terminal = float(np.linalg.norm(local_pose[-1, :2]))
        output[candidate] = np.column_stack(
            [
                local_pose[:, 0],
                local_pose[:, 1],
                local_pose[:, 2],
                speed,
                acceleration,
                yaw_rate,
                curvature,
                jerk,
                np.full(h, terminal),
            ]
        )
    return output


def _current_scene_features(scene: Any, scenario: Any) -> np.ndarray:
    current = scene.scene_metadata.num_history_frames - 1
    ego = scene.frames[current].ego_status
    tracks = list(scenario.get_tracked_objects_at_iteration(0).tracked_objects)
    ego_xy = np.asarray(ego.ego_pose[:2], dtype=np.float64)
    distances = [np.linalg.norm(np.asarray([track.center.x, track.center.y]) - ego_xy) for track in tracks]
    return np.asarray(
        [
            np.linalg.norm(ego.ego_velocity),
            np.linalg.norm(ego.ego_acceleration),
            len(tracks),
            min(distances, default=100.0),
            len(scene.frames[current].roadblock_ids),
            sum(bool(is_red) for _, is_red in scene.frames[current].traffic_lights),
        ],
        dtype=np.float32,
    )


def build_scene_targets(
    scene: Any,
    metric_cache: Any,
    group: pd.DataFrame,
    paths: Any,
    max_actors: int,
    max_shared_actors: int,
) -> dict[str, np.ndarray]:
    from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
        coords_array_to_polygon_array,
        state_array_to_coords_array,
    )

    ordered = group.sort_values("candidate_index").reset_index(drop=True)
    states = _states_from_group(ordered)
    if states.shape[1:] != (41, 11):
        raise ValueError(f"Unexpected simulated state shape {states.shape}")
    k = len(ordered)
    h_count = len(TARGET_TIMES)
    current_index = scene.scene_metadata.num_history_frames - 1
    current_pose = np.asarray(scene.frames[current_index].ego_status.ego_pose, dtype=np.float64)
    trajectory = _trajectory_features(states, current_pose)
    scenario = NavSimScenario(scene, str(paths.map_path), "nuplan-maps-v1.0")

    coords = state_array_to_coords_array(states, get_pacifica_parameters())
    ego_polygons_all = coords_array_to_polygon_array(coords)
    target_state_indices = np.arange(5, 41, 5)
    centerline = metric_cache.centerline
    initial_center = Point(*coords[0, 0, -1])
    _, _, initial_progress = _nearest_centerline(centerline, initial_center)

    actor_values = np.zeros((k, h_count, max_actors, len(ACTOR_FEATURES)), dtype=np.float32)
    actor_mask = np.zeros((k, h_count, max_actors), dtype=bool)
    actor_token_hash = np.zeros((k, h_count, max_actors), dtype=np.int64)
    shared_values = np.zeros((h_count, max_shared_actors, len(SHARED_ACTOR_FEATURES)), dtype=np.float32)
    shared_mask = np.zeros((h_count, max_shared_actors), dtype=bool)
    shared_token_hash = np.zeros((h_count, max_shared_actors), dtype=np.int64)
    env = np.zeros((k, h_count, len(ENVIRONMENT_FEATURES)), dtype=np.float32)
    shared_actor_count = np.zeros(h_count, dtype=np.int32)
    shared_red_light_count = np.zeros(h_count, dtype=np.int32)
    connector_geometry_cache: dict[str, Any] = {}

    for h_index, (time_s, state_index) in enumerate(zip(TARGET_TIMES, target_state_indices)):
        iteration = int(round(time_s / 0.5))
        frame_index = current_index + iteration
        tracks = list(scenario.get_tracked_objects_at_iteration(iteration).tracked_objects)
        world = [_track_world_record(track) for track in tracks]
        dynamic = [record for record, track in zip(world, tracks) if track.tracked_object_type in AGENT_TYPES]
        shared_actor_count[h_index] = len(dynamic)
        shared_sorted = sorted(dynamic, key=lambda item: item["token_hash"])[:max_shared_actors]
        for actor_index, actor in enumerate(shared_sorted):
            shared_mask[h_index, actor_index] = True
            shared_token_hash[h_index, actor_index] = actor["token_hash"]
            shared_values[h_index, actor_index] = [
                actor["type_id"],
                actor["x"],
                actor["y"],
                actor["vx"],
                actor["vy"],
                actor["heading"],
                actor["length"],
                actor["width"],
            ]
        red_polygons = _red_light_polygons(scene, frame_index, connector_geometry_cache)
        shared_red_light_count[h_index] = len(red_polygons)

        for candidate in range(k):
            ego_state = states[candidate, state_index]
            ego_pose = ego_state[:3]
            ego_velocity = ego_state[3:5]
            ego_polygon = ego_polygons_all[candidate, state_index]
            prefix_centers = coords[candidate, : state_index + 1, -1]
            if np.unique(np.round(prefix_centers, 4), axis=0).shape[0] >= 2:
                corridor = LineString(prefix_centers).buffer(get_pacifica_parameters().half_width + 0.5)
            else:
                corridor = ego_polygon.buffer(0.5)

            relative_records = []
            min_clearance = 100.0
            min_center_distance = 100.0
            collision = False
            corridor_count = 0
            min_ttc = 10.0
            for actor in dynamic:
                actor_pose = np.asarray([[actor["x"], actor["y"], actor["heading"]]], dtype=np.float64)
                relative_pose = global_to_local(ego_pose, actor_pose)[0]
                c, s = math.cos(ego_pose[2]), math.sin(ego_pose[2])
                dvx = actor["vx"] - ego_velocity[0]
                dvy = actor["vy"] - ego_velocity[1]
                relative_vx = c * dvx + s * dvy
                relative_vy = -s * dvx + c * dvy
                clearance = float(ego_polygon.distance(actor["polygon"]))
                center_distance = float(np.hypot(relative_pose[0], relative_pose[1]))
                in_corridor = bool(corridor.intersects(actor["polygon"]))
                is_collision = bool(ego_polygon.intersects(actor["polygon"]))
                half_width = (get_pacifica_parameters().width + actor["width"]) / 2.0 + 0.5
                front_clearance = relative_pose[0] - (get_pacifica_parameters().length + actor["length"]) / 2.0
                closing_speed = -relative_vx
                if front_clearance > 0 and closing_speed > 1e-3 and abs(relative_pose[1]) <= half_width:
                    min_ttc = min(min_ttc, front_clearance / closing_speed)
                min_clearance = min(min_clearance, clearance)
                min_center_distance = min(min_center_distance, center_distance)
                collision = collision or is_collision
                corridor_count += int(in_corridor)
                relative_records.append(
                    {
                        "distance": center_distance,
                        "token_hash": actor["token_hash"],
                        "values": [
                            actor["type_id"],
                            relative_pose[0],
                            relative_pose[1],
                            relative_vx,
                            relative_vy,
                            relative_pose[2],
                            actor["length"],
                            actor["width"],
                            clearance,
                            float(in_corridor),
                        ],
                    }
                )
            relative_records.sort(key=lambda item: (item["distance"], item["token_hash"]))
            for actor_index, actor in enumerate(relative_records[:max_actors]):
                actor_mask[candidate, h_index, actor_index] = True
                actor_token_hash[candidate, h_index, actor_index] = actor["token_hash"]
                actor_values[candidate, h_index, actor_index] = actor["values"]

            center = Point(*coords[candidate, state_index, -1])
            lateral_offset, centerline_heading, progress = _nearest_centerline(centerline, center)
            heading_error = abs(float(wrap_angle(ego_pose[2] - centerline_heading)))
            try:
                intersection = bool(
                    metric_cache.drivable_area_map.is_in_layer(center, SemanticMapLayer.INTERSECTION)
                )
            except Exception:
                intersection = False
            non_drivable = bool(ordered.iloc[candidate].non_drivable_by_step[state_index])
            oncoming = bool(ordered.iloc[candidate].oncoming_by_step[state_index])
            if red_polygons:
                red_clearance = min(float(ego_polygon.distance(polygon)) for polygon in red_polygons)
                red_intersection = any(ego_polygon.intersects(polygon) for polygon in red_polygons)
            else:
                red_clearance = 100.0
                red_intersection = False
            nearest_x = relative_records[0]["values"][1] if relative_records else 0.0
            nearest_y = relative_records[0]["values"][2] if relative_records else 0.0
            env[candidate, h_index] = [
                min_clearance,
                min_center_distance,
                corridor_count,
                float(collision),
                min(min_ttc, 10.0),
                float(not non_drivable),
                float(oncoming),
                float(intersection),
                lateral_offset,
                heading_error,
                progress - initial_progress,
                min(red_clearance, 100.0),
                float(red_intersection),
                nearest_x,
                nearest_y,
            ]

    full = np.concatenate([trajectory, env], axis=-1).astype(np.float32)
    return {
        "time_s": TARGET_TIMES.astype(np.float32),
        "candidate_index": ordered["candidate_index"].to_numpy(dtype=np.int16),
        "is_gt": ordered["is_gt"].to_numpy(dtype=bool),
        "trajectory_derived": trajectory,
        "shared_logged_future_actor": shared_values,
        "shared_logged_future_actor_mask": shared_mask,
        "shared_logged_future_actor_token_hash": shared_token_hash,
        "shared_logged_future_actor_count": shared_actor_count,
        "shared_logged_future_red_light_count": shared_red_light_count,
        "candidate_relative_actor": actor_values,
        "candidate_relative_actor_mask": actor_mask,
        "candidate_relative_actor_token_hash": actor_token_hash,
        "C_environment_only": env,
        "C_full": full,
        "current_scene_features": _current_scene_features(scene, scenario),
    }


def target_schema(max_actors: int, max_shared_actors: int) -> dict[str, Any]:
    def field(
        shape: list[Any],
        dtype: str,
        unit: str,
        coordinate: str,
        candidate: bool,
        logged: bool,
        inference: bool,
        description: str,
        mask: str | None = None,
    ) -> dict[str, Any]:
        return {
            "shape": shape,
            "dtype": dtype,
            "unit": unit,
            "coordinate_frame": coordinate,
            "time_frequency_hz": 2.0 if "H" in shape else None,
            "valid_mask": mask,
            "depends_on_candidate": candidate,
            "depends_on_logged_future": logged,
            "reactive_only": False,
            "available_as_observation_at_inference": inference,
            "description": description,
        }

    return {
        "schema_version": "0.1.0",
        "horizons_s": TARGET_TIMES.tolist(),
        "actor_limit": max_actors,
        "shared_actor_limit": max_shared_actors,
        "trajectory_feature_names": list(TRAJECTORY_FEATURES),
        "environment_feature_names": list(ENVIRONMENT_FEATURES),
        "actor_feature_names": list(ACTOR_FEATURES),
        "shared_actor_feature_names": list(SHARED_ACTOR_FEATURES),
        "fields": {
            "trajectory_derived": field(
                ["K", "H", len(TRAJECTORY_FEATURES)], "float32", "mixed; see feature names", "current ego local", True, False, True,
                "Directly recoverable from the candidate/simulated candidate. Excluded from the environment-only schema."
            ),
            "shared_logged_future_actor": field(
                ["H", max_shared_actors, len(SHARED_ACTOR_FEATURES)], "float32", "mixed", "global map", False, True, False,
                "One shared logged future actor world, from official Scene annotation conversion after the deployed cache lacked a named future_tracked_objects field.",
                "shared_logged_future_actor_mask",
            ),
            "candidate_relative_actor": field(
                ["K", "H", max_actors, len(ACTOR_FEATURES)], "float32", "mixed", "candidate ego at each horizon", True, True, False,
                "Nearest dynamic actors after candidate-conditioned relabeling of the same logged future.",
                "candidate_relative_actor_mask",
            ),
            "C_environment_only": field(
                ["K", "H", len(ENVIRONMENT_FEATURES)], "float32", "mixed", "candidate-relative/global map relations", True, True, False,
                "No candidate waypoint copy, candidate type, final PDM score, or aggregate PDM factor is included."
            ),
            "C_full": field(
                ["K", "H", len(TRAJECTORY_FEATURES) + len(ENVIRONMENT_FEATURES)], "float32", "mixed", "mixed", True, True, False,
                "Completeness/debug schema: trajectory-derived concatenated with candidate-relative environment features."
            ),
            "reactive_response": {
                "shape": None,
                "dtype": None,
                "unit": None,
                "coordinate_frame": None,
                "time_frequency_hz": None,
                "valid_mask": None,
                "depends_on_candidate": True,
                "depends_on_logged_future": False,
                "reactive_only": True,
                "available_as_observation_at_inference": False,
                "description": "Not generated by the runtime NAVSIM 1.1 non-reactive scorer.",
            },
        },
        "leakage_exclusions": [
            "candidate future x/y/heading from C_environment_only",
            "candidate waypoint values",
            "candidate type/family/index identity",
            "official final PDM score",
            "official aggregate PDM factor columns",
        ],
        "world_source_note": (
            "The deployed train_metric_chache.MetricCache stores a 10 Hz PDMObservation but no named "
            "future_tracked_objects list. Actor identities/types/velocities therefore come from Scene future "
            "annotations via NavSimScenario.get_tracked_objects_at_iteration; their global polygons were "
            "numerically cross-checked against the cache observation in Gate A."
        ),
    }


def build_targets(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    paths = paths_from_args(args)
    output_dir = ensure_output_dir(args.output_dir)
    gate_path = output_dir / "gate_status.json"
    gates = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.exists() else {}
    if not gates.get("gate_a", {}).get("passed") or not gates.get("gate_b", {}).get("passed"):
        raise SystemExit("Gate A and Gate B must pass before building candidate-relative targets")
    metrics = read_parquet(output_dir / "candidate_metrics.parquet")
    metrics = metrics[metrics["scoring_success"] & (metrics["traffic_policy"] == "non_reactive")].copy()
    if args.max_scenes > 0:
        tokens = metrics["scene_token"].drop_duplicates().head(args.max_scenes).tolist()
        metrics = metrics[metrics["scene_token"].isin(tokens)].copy()
    else:
        tokens = metrics["scene_token"].drop_duplicates().tolist()
    loader = load_scenes_for_tokens(paths, tokens)
    caches = metric_cache_loader(paths)
    targets_dir = ensure_output_dir(output_dir / "targets")

    coverage_rows = []
    index_rows = []
    for token, group in metrics.groupby("scene_token", sort=False):
        try:
            scene = loader.get_scene_from_token(token)
            cache = caches.get_from_token(token)
            arrays = build_scene_targets(
                scene,
                cache,
                group,
                paths,
                max_actors=args.max_actors,
                max_shared_actors=args.max_shared_actors,
            )
            target_path = targets_dir / f"{token}.npz"
            np.savez_compressed(target_path, **arrays)
            env = arrays["C_environment_only"]
            mask = arrays["candidate_relative_actor_mask"]
            coverage_rows.append(
                {
                    "scene_token": token,
                    "log_name": scene.scene_metadata.log_name,
                    "success": True,
                    "error": None,
                    "candidate_count": env.shape[0],
                    "horizon_count": env.shape[1],
                    "finite_environment_fraction": float(np.isfinite(env).mean()),
                    "actor_slot_valid_fraction": float(mask.mean()),
                    "future_steps_with_actor": float(mask.any(axis=(0, 2)).mean()),
                    "future_steps_with_red_light": float((arrays["shared_logged_future_red_light_count"] > 0).mean()),
                    "map_relation_coverage": float(np.isfinite(env[..., 8:11]).mean()),
                    "traffic_light_relation_coverage": 1.0,
                    "source": "logged future; non-reactive candidate-conditioned relabeling",
                }
            )
            index_rows.append(
                {
                    "scene_token": token,
                    "log_name": scene.scene_metadata.log_name,
                    "target_path": str(target_path.relative_to(output_dir)),
                    "candidate_count": env.shape[0],
                }
            )
        except Exception as exc:
            coverage_rows.append(
                {
                    "scene_token": token,
                    "log_name": str(group.iloc[0].log_name),
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "candidate_count": len(group),
                    "horizon_count": 0,
                    "finite_environment_fraction": 0.0,
                    "actor_slot_valid_fraction": 0.0,
                    "future_steps_with_actor": 0.0,
                    "future_steps_with_red_light": 0.0,
                    "map_relation_coverage": 0.0,
                    "traffic_light_relation_coverage": 0.0,
                    "source": "failed",
                }
            )
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(output_dir / "target_coverage.csv", index=False)
    index = pd.DataFrame(index_rows)
    write_parquet(index, targets_dir / "index.parquet")
    schema = target_schema(args.max_actors, args.max_shared_actors)
    schema["actual_scene_success_rate"] = float(coverage["success"].mean()) if len(coverage) else 0.0
    schema["evidence_scene_tokens"] = coverage.loc[coverage["success"], "scene_token"].head(8).tolist()
    write_json(output_dir / "target_schema.json", schema)
    success_rate = schema["actual_scene_success_rate"]
    report = f"""# Candidate-relative Target Construction

- Successful scenes: {int(coverage['success'].sum())}/{len(coverage)} ({success_rate:.3%})
- `C_full`: trajectory-derived candidate motion plus world-relative relations.
- `C_environment_only`: {len(ENVIRONMENT_FEATURES)} relationship features; no waypoint copy, candidate type/index, official final score or official aggregate factor.
- Dynamic actor tensor: nearest {args.max_actors} per candidate/horizon, deterministically sorted by center distance then stable token hash.
- Shared logged-world tensor: up to {args.max_shared_actors} dynamic actors per horizon in global coordinates.
- Time horizons: {TARGET_TIMES.tolist()} s at 2 Hz.

The construction is a **non-reactive candidate-relative consequence**: every candidate is related to the same logged future actors, traffic-light records and static map. It is not a candidate-specific ground-truth image, true multi-agent response, causal effect or complete counterfactual future.

The deployed training MetricCache has no named `future_tracked_objects` field. Actor identity/type/velocity therefore comes from official `NavSimScenario.get_tracked_objects_at_iteration`, backed by Scene annotations; Gate A verifies its global polygons against MetricCache's 10 Hz logged occupancy.
"""
    write_markdown(output_dir / "TARGET_CONSTRUCTION_REPORT.md", report)
    if success_rate < 0.98:
        raise SystemExit(f"Candidate-relative target coverage {success_rate:.3%} is below 98%")
    return coverage, schema


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, include_candidates=True)
    parser.add_argument("--max-actors", type=int, default=16)
    parser.add_argument("--max-shared-actors", type=int, default=64)
    args = parser.parse_args()
    build_targets(args)
    append_command(args.output_dir.resolve(), "python -m tools.navsim_candidate_relative_audit.build_candidate_relative_targets " + " ".join(sys.argv[1:]))


if __name__ == "__main__":
    main()
