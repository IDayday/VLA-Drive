#!/usr/bin/env python3
"""Phase 5: build candidate-relative targets from one shared logged future."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.ops import unary_union

from .common import (
    HORIZONS_S,
    add_common_arguments,
    bootstrap_navsim,
    discover_paths,
    load_metric_cache,
    load_metric_cache_index,
    rotate_global_vector_to_local,
    se2_global_to_local,
    stable_token_hash,
    trajectory_kinematics,
    uniform_horizon_index,
    wrap_heading,
    write_dataframe,
    write_json,
)


TRAJECTORY_FIELDS = (
    "candidate_local_x_m",
    "candidate_local_y_m",
    "candidate_local_heading_rad",
    "candidate_speed_mps",
    "candidate_acceleration_mps2",
    "candidate_yaw_rate_radps",
    "candidate_curvature_inv_m",
    "candidate_jerk_mps3",
    "terminal_displacement_m",
)

ENVIRONMENT_FIELDS = (
    "minimum_object_clearance_m",
    "minimum_dynamic_clearance_m",
    "actor_count_within_10m",
    "candidate_corridor_occupancy_count",
    "object_collision",
    "dynamic_actor_collision",
    "minimum_linear_ttc_s",
    "nearest_actor_relative_x_m",
    "nearest_actor_relative_y_m",
    "nearest_actor_relative_vx_mps",
    "nearest_actor_relative_vy_mps",
    "candidate_center_in_drivable_map",
    "candidate_center_in_lane_or_connector",
    "candidate_heading_oncoming_relation",
    "candidate_center_in_intersection",
    "centerline_lateral_offset_m",
    "centerline_heading_error_rad",
    "route_progress_m",
    "red_light_route_zone_count",
    "red_light_minimum_clearance_m",
    "red_light_zone_intersection",
)

ACTOR_FIELDS = (
    "object_type_code",
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

SHARED_ACTOR_FIELDS = (
    "object_type_code",
    "global_x_m",
    "global_y_m",
    "global_vx_mps",
    "global_vy_mps",
    "global_heading_rad",
    "length_m",
    "width_m",
)


def load_inputs(output_dir: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    manifest = pd.read_parquet(output_dir / "candidate_manifest.parquet").sort_values(
        ["scene_index", "candidate_index"]
    )
    metrics = pd.read_parquet(output_dir / "candidate_metrics.parquet").sort_values(
        ["scene_index", "candidate_index"]
    )
    with np.load(output_dir / "candidate_trajectories.npz") as payload:
        trajectories = np.asarray(payload["trajectories"], dtype=np.float32)
    with np.load(output_dir / "candidate_simulation_arrays.npz") as payload:
        simulated = np.asarray(payload["simulated_states"], dtype=np.float32)
    if len(manifest) != len(metrics) or trajectories.shape[:2] != simulated.shape[:2]:
        raise ValueError(
            "candidate manifest, metrics, trajectories, and simulation arrays are misaligned"
        )
    return manifest, trajectories, simulated


def object_type_mapping(tracks: Sequence[Any]) -> dict[str, int]:
    names = sorted(
        {
            str(obj.tracked_object_type)
            for detections in tracks
            for obj in detections.tracked_objects.tracked_objects
        }
    )
    return {name: index + 1 for index, name in enumerate(names)}


def linear_ttc(
    relative_position: np.ndarray,
    relative_velocity: np.ndarray,
    collision_radius_m: float,
) -> float:
    velocity_norm_sq = float(relative_velocity @ relative_velocity)
    if velocity_norm_sq < 1e-6:
        return 5.0
    time = -float(relative_position @ relative_velocity) / velocity_norm_sq
    if not 0.0 <= time <= 5.0:
        return 5.0
    closest = relative_position + time * relative_velocity
    return time if float(np.linalg.norm(closest)) <= collision_radius_m else 5.0


def map_relations(
    cache: Any,
    ego_polygon: Any,
    state: np.ndarray,
    start_progress: float,
    time_index: int,
) -> list[float]:
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer

    center = Point(float(state[0]), float(state[1]))
    drivable = cache.drivable_area_map
    indices = drivable.query(center, predicate="within")
    types = [drivable.map_types[int(index)] for index in indices]
    in_drivable = bool(indices.size)
    in_lane = any(
        layer in {SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR}
        for layer in types
    )
    in_intersection = SemanticMapLayer.INTERSECTION in types
    progress = float(cache.centerline.project(center))
    centerline_state = np.asarray(
        cache.centerline.interpolate([progress], as_array=True)
    )[0]
    heading_error = float(abs(wrap_heading(float(state[2]) - centerline_state[2])))
    lateral_offset = float(center.distance(cache.centerline.linestring))
    oncoming = heading_error > math.pi / 2
    red_tokens, red_polygons = cache.observation._occupancy_maps_tl[time_index]
    if len(red_polygons):
        red_clearance = min(
            float(ego_polygon.distance(polygon)) for polygon in red_polygons
        )
        red_intersection = any(
            bool(ego_polygon.intersects(polygon)) for polygon in red_polygons
        )
    else:
        red_clearance = 100.0
        red_intersection = False
    return [
        float(in_drivable),
        float(in_lane),
        float(oncoming),
        float(in_intersection),
        lateral_offset,
        heading_error,
        progress - start_progress,
        float(len(red_tokens)),
        red_clearance,
        float(red_intersection),
    ]


def actor_features(
    detections: Any,
    state: np.ndarray,
    ego_polygon: Any,
    corridor: Any,
    type_codes: dict[str, int],
    max_actors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES

    center_global = np.asarray(state[:3], dtype=np.float64)
    ego_velocity_global = np.asarray(state[3:5], dtype=np.float64)
    entries: list[tuple[float, int, Any, list[float], bool]] = []
    all_clearance = 100.0
    dynamic_clearance = 100.0
    collision = False
    dynamic_collision = False
    actor_count_10m = 0
    corridor_count = 0
    ttc_min = 5.0
    nearest: list[float] | None = None
    nearest_distance = float("inf")
    for obj in detections.tracked_objects.tracked_objects:
        global_pose = np.asarray(
            [obj.center.x, obj.center.y, obj.center.heading], dtype=np.float64
        )
        relative_pose = se2_global_to_local(global_pose, center_global)
        velocity = getattr(obj, "velocity", None)
        actor_velocity_global = np.asarray(
            [getattr(velocity, "x", 0.0), getattr(velocity, "y", 0.0)], dtype=np.float64
        )
        relative_velocity = rotate_global_vector_to_local(
            actor_velocity_global - ego_velocity_global, float(state[2])
        )
        clearance = float(ego_polygon.distance(obj.box.geometry))
        intersects = bool(ego_polygon.intersects(obj.box.geometry))
        is_dynamic = obj.tracked_object_type in AGENT_TYPES
        in_corridor = bool(corridor.intersects(obj.box.geometry))
        distance = float(np.linalg.norm(relative_pose[:2]))
        if distance <= 10.0:
            actor_count_10m += 1
        corridor_count += int(in_corridor)
        all_clearance = min(all_clearance, clearance)
        collision |= intersects
        if is_dynamic:
            dynamic_clearance = min(dynamic_clearance, clearance)
            dynamic_collision |= intersects
            radius = 0.5 * (float(obj.box.width) + 2.3)
            ttc_min = min(
                ttc_min, linear_ttc(relative_pose[:2], relative_velocity, radius)
            )
        values = [
            float(type_codes[str(obj.tracked_object_type)]),
            float(relative_pose[0]),
            float(relative_pose[1]),
            float(relative_velocity[0]),
            float(relative_velocity[1]),
            float(relative_pose[2]),
            float(obj.box.length),
            float(obj.box.width),
            clearance,
            float(in_corridor),
        ]
        token_hash = stable_token_hash(str(obj.metadata.track_token))
        entries.append((distance, token_hash, obj, values, is_dynamic))
        if is_dynamic and distance < nearest_distance:
            nearest_distance = distance
            nearest = [
                float(relative_pose[0]),
                float(relative_pose[1]),
                float(relative_velocity[0]),
                float(relative_velocity[1]),
            ]
    entries.sort(key=lambda entry: (entry[0], entry[1]))
    tensor = np.zeros((max_actors, len(ACTOR_FIELDS)), dtype=np.float32)
    mask = np.zeros(max_actors, dtype=bool)
    hashes = np.zeros(max_actors, dtype=np.uint64)
    for index, (_, token_hash, _, values, _) in enumerate(entries[:max_actors]):
        tensor[index] = values
        mask[index] = True
        hashes[index] = token_hash
    nearest = nearest or [100.0, 100.0, 0.0, 0.0]
    summary = [
        all_clearance,
        dynamic_clearance,
        float(actor_count_10m),
        float(corridor_count),
        float(collision),
        float(dynamic_collision),
        ttc_min,
        *nearest,
    ]
    return tensor, mask, hashes, summary


def shared_actor_features(
    detections: Any,
    initial_xy: np.ndarray,
    type_codes: dict[str, int],
    max_actors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    entries = []
    for obj in detections.tracked_objects.tracked_objects:
        velocity = getattr(obj, "velocity", None)
        values = [
            float(type_codes[str(obj.tracked_object_type)]),
            float(obj.center.x),
            float(obj.center.y),
            float(getattr(velocity, "x", 0.0)),
            float(getattr(velocity, "y", 0.0)),
            float(obj.center.heading),
            float(obj.box.length),
            float(obj.box.width),
        ]
        distance = float(
            np.linalg.norm(np.asarray([obj.center.x, obj.center.y]) - initial_xy)
        )
        token_hash = stable_token_hash(str(obj.metadata.track_token))
        entries.append((distance, token_hash, values))
    entries.sort(key=lambda item: (item[0], item[1]))
    tensor = np.zeros((max_actors, len(SHARED_ACTOR_FIELDS)), dtype=np.float64)
    mask = np.zeros(max_actors, dtype=bool)
    hashes = np.zeros(max_actors, dtype=np.uint64)
    for index, (_, token_hash, values) in enumerate(entries[:max_actors]):
        tensor[index] = values
        mask[index] = True
        hashes[index] = token_hash
    return tensor, mask, hashes


def build_scene_target(
    cache: Any,
    trajectories: np.ndarray,
    states: np.ndarray,
    *,
    max_actors: int,
    traffic_policy: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
        coords_array_to_polygon_array,
        state_array_to_coords_array,
    )
    from navsim.traffic_agents_policies.log_replay_traffic_agents import (
        LogReplayTrafficAgents,
    )
    from nuplan.planning.simulation.trajectory.trajectory_sampling import (
        TrajectorySampling,
    )

    candidate_count = len(trajectories)
    horizon_count = len(HORIZONS_S)
    actor_tensor = np.zeros(
        (candidate_count, horizon_count, max_actors, len(ACTOR_FIELDS)),
        dtype=np.float32,
    )
    actor_mask = np.zeros((candidate_count, horizon_count, max_actors), dtype=bool)
    actor_hash = np.zeros((candidate_count, horizon_count, max_actors), dtype=np.uint64)
    environment = np.zeros(
        (candidate_count, horizon_count, len(ENVIRONMENT_FIELDS)), dtype=np.float32
    )
    trajectory_target = np.zeros(
        (candidate_count, horizon_count, len(TRAJECTORY_FIELDS)), dtype=np.float32
    )
    shared = np.zeros(
        (horizon_count, max_actors, len(SHARED_ACTOR_FIELDS)), dtype=np.float64
    )
    shared_mask = np.zeros((horizon_count, max_actors), dtype=bool)
    shared_hash = np.zeros((horizon_count, max_actors), dtype=np.uint64)
    reactive_response = np.full(
        (candidate_count, horizon_count, 4), np.nan, dtype=np.float32
    )
    reactive_mask = np.zeros((candidate_count, horizon_count), dtype=bool)

    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    if traffic_policy != "non_reactive":
        raise NotImplementedError(
            "reactive actor response targets require per-candidate captured policy tracks; run audit_v2_extensions for this deployment audit"
        )
    replay_policy = LogReplayTrafficAgents(sampling)
    shared_tracks = replay_policy.simulate_environment(states[0], cache)
    type_codes = object_type_mapping(shared_tracks)
    horizon_indices = [
        uniform_horizon_index(h, 0.1, 41, includes_current=True) for h in HORIZONS_S
    ]
    initial_pose = np.asarray(
        [
            cache.ego_state.rear_axle.x,
            cache.ego_state.rear_axle.y,
            cache.ego_state.rear_axle.heading,
        ]
    )
    start_progress = float(
        cache.centerline.project(Point(initial_pose[0], initial_pose[1]))
    )
    for horizon_slot, time_index in enumerate(horizon_indices):
        shared[horizon_slot], shared_mask[horizon_slot], shared_hash[horizon_slot] = (
            shared_actor_features(
                shared_tracks[time_index], initial_pose[:2], type_codes, max_actors
            )
        )

    for candidate_index in range(candidate_count):
        candidate_states = np.asarray(states[candidate_index], dtype=np.float64)
        coords = state_array_to_coords_array(
            candidate_states[None], cache.ego_state.car_footprint.vehicle_parameters
        )[0]
        ego_polygons = coords_array_to_polygon_array(coords)
        tracks = replay_policy.simulate_environment(candidate_states, cache)
        kinematics = trajectory_kinematics(trajectories[candidate_index])
        for horizon_slot, (horizon, time_index) in enumerate(
            zip(HORIZONS_S, horizon_indices)
        ):
            corridor = unary_union(list(ego_polygons[: time_index + 1])).buffer(0.25)
            tensor, mask, hashes, actor_summary = actor_features(
                tracks[time_index],
                candidate_states[time_index],
                ego_polygons[time_index],
                corridor,
                type_codes,
                max_actors,
            )
            actor_tensor[candidate_index, horizon_slot] = tensor
            actor_mask[candidate_index, horizon_slot] = mask
            actor_hash[candidate_index, horizon_slot] = hashes
            map_summary = map_relations(
                cache,
                ego_polygons[time_index],
                candidate_states[time_index],
                start_progress,
                time_index,
            )
            environment[candidate_index, horizon_slot] = np.asarray(
                actor_summary + map_summary, dtype=np.float32
            )
            candidate_pose_index = uniform_horizon_index(
                horizon, 0.5, 8, includes_current=False
            )
            pose = trajectories[candidate_index, candidate_pose_index]
            trajectory_target[candidate_index, horizon_slot] = np.asarray(
                [
                    pose[0],
                    pose[1],
                    pose[2],
                    np.asarray(kinematics["speed"])[candidate_pose_index],
                    np.asarray(kinematics["acceleration"])[candidate_pose_index],
                    np.asarray(kinematics["yaw_rate"])[candidate_pose_index],
                    np.asarray(kinematics["curvature"])[candidate_pose_index],
                    np.asarray(kinematics["jerk"])[candidate_pose_index],
                    kinematics["terminal_displacement"],
                ],
                dtype=np.float32,
            )
    return (
        {
            "C_full": np.concatenate([trajectory_target, environment], axis=-1),
            "C_environment_only": environment,
            "trajectory_derived": trajectory_target,
            "shared_logged_future": shared,
            "shared_logged_future_mask": shared_mask,
            "shared_logged_future_track_hash": shared_hash,
            "candidate_relative_actor_tensor": actor_tensor,
            "candidate_relative_actor_mask": actor_mask,
            "candidate_relative_actor_track_hash": actor_hash,
            "reactive_response": reactive_response,
            "reactive_response_mask": reactive_mask,
            "horizons_s": np.asarray(HORIZONS_S, dtype=np.float32),
        },
        {
            "object_type_codes": type_codes,
            "horizon_state_indices": horizon_indices,
        },
    )


REACTIVE_FIELDS = (
    "actor_endpoint_change_m",
    "actor_speed_change_mps",
    "braking_response_mps",
    "headway_change_m",
)

MAP_ONLY_ENVIRONMENT_FIELDS = {
    "candidate_center_in_drivable_map",
    "candidate_center_in_lane_or_connector",
    "candidate_heading_oncoming_relation",
    "candidate_center_in_intersection",
    "centerline_lateral_offset_m",
    "centerline_heading_error_rad",
    "route_progress_m",
}


def field_unit(name: str) -> str:
    suffix_units = (
        ("_mps3", "m/s^3"),
        ("_mps2", "m/s^2"),
        ("_radps", "rad/s"),
        ("_inv_m", "1/m"),
        ("_mps", "m/s"),
        ("_rad", "rad"),
        ("_s", "s"),
        ("_m", "m"),
    )
    for suffix, unit in suffix_units:
        if name.endswith(suffix):
            return unit
    return "1"


def field_group(name: str) -> str:
    if name in TRAJECTORY_FIELDS:
        return "trajectory_derived"
    if name in ENVIRONMENT_FIELDS:
        return "candidate_relative_future"
    if name in SHARED_ACTOR_FIELDS:
        return "shared_logged_future"
    if name in ACTOR_FIELDS:
        return "candidate_relative_actor"
    if name in REACTIVE_FIELDS:
        return "reactive_response"
    raise KeyError(name)


def field_metadata(
    fields: tuple[str, ...], *, valid_mask: str | None = None
) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    for name in fields:
        group = field_group(name)
        is_map_only = name in MAP_ONLY_ENVIRONMENT_FIELDS
        if group == "trajectory_derived":
            coordinate_frame = "current rear-axle local or ego kinematic scalar"
            source_frequency_hz = 2.0
            logged_future_dependent = False
            inference_availability: bool | str = True
        elif group == "shared_logged_future":
            coordinate_frame = "global map frame"
            source_frequency_hz = 10.0
            logged_future_dependent = True
            inference_availability = False
        elif group == "candidate_relative_actor":
            coordinate_frame = "candidate rear-axle heading-aligned at each horizon"
            source_frequency_hz = 10.0
            logged_future_dependent = True
            inference_availability = False
        elif group == "reactive_response":
            coordinate_frame = "actor response scalar relative to non-reactive replay"
            source_frequency_hz = 10.0
            logged_future_dependent = "non-reactive baseline only"
            inference_availability = "reactive simulator only"
        else:
            coordinate_frame = (
                "global map queried at candidate pose"
                if is_map_only
                else "candidate/world interaction scalar; relative vectors use candidate frame"
            )
            source_frequency_hz = 10.0
            logged_future_dependent = not is_map_only
            inference_availability = is_map_only
        metadata.append(
            {
                "name": name,
                "unit": field_unit(name),
                "coordinate_frame": coordinate_frame,
                "target_horizons_s": list(HORIZONS_S),
                "source_frequency_hz": source_frequency_hz,
                "valid_mask": valid_mask,
                "candidate_dependent": group != "shared_logged_future",
                "logged_future_dependent": logged_future_dependent,
                "reactive_only": group == "reactive_response",
                "inference_availability": inference_availability,
            }
        )
    return metadata


def schema(max_actors: int) -> dict[str, Any]:
    units = {
        "m": [
            name
            for name in (*TRAJECTORY_FIELDS, *ENVIRONMENT_FIELDS, *ACTOR_FIELDS)
            if name.endswith("_m")
        ],
        "m/s": [
            name
            for name in (*TRAJECTORY_FIELDS, *ENVIRONMENT_FIELDS, *ACTOR_FIELDS)
            if name.endswith("_mps")
        ],
        "rad": [
            name
            for name in (*TRAJECTORY_FIELDS, *ENVIRONMENT_FIELDS, *ACTOR_FIELDS)
            if name.endswith("_rad")
        ],
    }
    return {
        "version": "candidate_relative_target_v1",
        "horizons_s": list(HORIZONS_S),
        "time_frequency": {
            "candidate_waypoints_hz": 2.0,
            "metric_future_tracks_hz": 10.0,
        },
        "coordinate_frames": {
            "trajectory_derived": "current rear-axle local",
            "shared_logged_future": "global map frame",
            "candidate_relative_actor_tensor": "candidate rear-axle heading-aligned at each horizon",
        },
        "arrays": {
            "C_full": {
                "shape": [
                    "K",
                    len(HORIZONS_S),
                    len(TRAJECTORY_FIELDS) + len(ENVIRONMENT_FIELDS),
                ],
                "dtype": "float32",
                "fields": list(TRAJECTORY_FIELDS + ENVIRONMENT_FIELDS),
                "field_metadata": field_metadata(
                    TRAJECTORY_FIELDS + ENVIRONMENT_FIELDS
                ),
                "valid_mask": None,
                "candidate_dependent": True,
                "logged_future_dependent": "partly",
                "reactive_only": False,
                "inference_availability": "partly; future-world fields require prediction",
            },
            "C_environment_only": {
                "shape": ["K", len(HORIZONS_S), len(ENVIRONMENT_FIELDS)],
                "dtype": "float32",
                "fields": list(ENVIRONMENT_FIELDS),
                "field_metadata": field_metadata(ENVIRONMENT_FIELDS),
                "valid_mask": None,
                "candidate_dependent": True,
                "logged_future_dependent": True,
                "reactive_only": False,
                "inference_availability": "static-map subset only; future-world fields require prediction",
                "contains_pdm_score_or_factor": False,
                "contains_candidate_waypoint_copy": False,
            },
            "trajectory_derived": {
                "shape": ["K", len(HORIZONS_S), len(TRAJECTORY_FIELDS)],
                "dtype": "float32",
                "fields": list(TRAJECTORY_FIELDS),
                "field_metadata": field_metadata(TRAJECTORY_FIELDS),
                "valid_mask": None,
                "candidate_dependent": True,
                "logged_future_dependent": False,
                "reactive_only": False,
                "inference_availability": True,
            },
            "shared_logged_future": {
                "shape": [len(HORIZONS_S), max_actors, len(SHARED_ACTOR_FIELDS)],
                "dtype": "float64",
                "fields": list(SHARED_ACTOR_FIELDS),
                "field_metadata": field_metadata(
                    SHARED_ACTOR_FIELDS, valid_mask="shared_logged_future_mask"
                ),
                "valid_mask": "shared_logged_future_mask",
                "candidate_dependent": False,
                "logged_future_dependent": True,
                "reactive_only": False,
                "inference_availability": False,
            },
            "candidate_relative_actor_tensor": {
                "shape": ["K", len(HORIZONS_S), max_actors, len(ACTOR_FIELDS)],
                "dtype": "float32",
                "fields": list(ACTOR_FIELDS),
                "field_metadata": field_metadata(
                    ACTOR_FIELDS, valid_mask="candidate_relative_actor_mask"
                ),
                "valid_mask": "candidate_relative_actor_mask",
                "sorting": "distance then stable SHA-256 track hash",
                "candidate_dependent": True,
                "logged_future_dependent": True,
                "reactive_only": False,
                "inference_availability": False,
            },
            "reactive_response": {
                "shape": ["K", len(HORIZONS_S), 4],
                "dtype": "float32",
                "fields": list(REACTIVE_FIELDS),
                "field_metadata": field_metadata(
                    REACTIVE_FIELDS, valid_mask="reactive_response_mask"
                ),
                "reactive_only": True,
                "valid_mask": "reactive_response_mask",
                "candidate_dependent": True,
                "logged_future_dependent": "non-reactive baseline only",
                "inference_availability": "reactive simulator only",
            },
        },
        "provenance_classes": {
            "trajectory_derived": "candidate trajectory only; not world-model supervision",
            "shared_logged_future": "one logged world shared by all candidates",
            "candidate_relative_future": "candidate-conditioned relabeling under non-reactive logged replay",
            "reactive_response": "NAVSIM v2 reactive-policy simulated consequence only",
        },
        "inference_availability": {
            "trajectory_derived": True,
            "static_map_subset_of_environment": True,
            "future_actor_or_future_traffic_light_environment": False,
            "C_environment_only_as_a_whole": False,
        },
        "units_by_suffix": units,
        "leakage_exclusions": [
            "candidate future waypoint values",
            "candidate type",
            "official aggregate PDM score",
            "official PDM factor scores",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--max-actors", type=int, default=16, choices=(16, 32))
    parser.add_argument(
        "--traffic-policy", choices=("non_reactive", "reactive"), default="non_reactive"
    )
    args = parser.parse_args()
    split = "trainval" if args.split == "train" else args.split
    paths = discover_paths(args, split=split)
    bootstrap_navsim(paths)
    if paths.metric_cache is None:
        raise FileNotFoundError("MetricCache is required")
    manifest, trajectories, states = load_inputs(args.output_dir)
    scene_count = min(args.max_scenes, trajectories.shape[0])
    trajectories, states = trajectories[:scene_count], states[:scene_count]
    metric_index = load_metric_cache_index(paths.metric_cache)
    target_root = args.output_dir / "targets"
    target_root.mkdir(parents=True, exist_ok=True)
    coverage_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    object_type_codes: dict[str, dict[str, int]] = {}
    for scene_index in range(scene_count):
        scene_manifest = manifest[manifest["scene_index"] == scene_index].sort_values(
            "candidate_index"
        )
        scene_token = str(scene_manifest.iloc[0]["scene_token"])
        try:
            cache = load_metric_cache(metric_index[scene_token])
            arrays, metadata = build_scene_target(
                cache,
                trajectories[scene_index],
                states[scene_index],
                max_actors=args.max_actors,
                traffic_policy=args.traffic_policy,
            )
            np.savez_compressed(target_root / f"{scene_token}.npz", **arrays)
            object_type_codes[scene_token] = metadata["object_type_codes"]
            environment = arrays["C_environment_only"]
            actor_mask = arrays["candidate_relative_actor_mask"]
            coverage_rows.append(
                {
                    "scene_token": scene_token,
                    "scene_index": scene_index,
                    "candidate_count": len(trajectories[scene_index]),
                    "target_file": f"targets/{scene_token}.npz",
                    "C_environment_only_finite_coverage": float(
                        np.isfinite(environment).mean()
                    ),
                    "actor_slot_valid_coverage": float(actor_mask.mean()),
                    "horizon_coverage": float(
                        np.isfinite(environment).all(axis=-1).mean()
                    ),
                    "reactive_response_coverage": float(
                        arrays["reactive_response_mask"].mean()
                    ),
                    "traffic_policy": args.traffic_policy,
                }
            )
        except Exception as error:
            failures.append(
                {
                    "scene_token": scene_token,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    coverage = pd.DataFrame(coverage_rows)
    write_dataframe(coverage, args.output_dir / "target_coverage.csv")
    schema_value = schema(args.max_actors)
    schema_value.update(
        {
            "scene_count": len(coverage_rows),
            "failure_count": len(failures),
            "traffic_policy": args.traffic_policy,
            "object_type_codes_by_scene": object_type_codes,
            "failures": failures,
        }
    )
    write_json(args.output_dir / "target_schema.json", schema_value)
    summary = {
        "scene_count": len(coverage_rows),
        "requested_scene_count": scene_count,
        "failure_count": len(failures),
        "candidate_relative_target_coverage": float(coverage["horizon_coverage"].mean())
        if len(coverage)
        else 0.0,
        "actor_slot_valid_coverage": float(coverage["actor_slot_valid_coverage"].mean())
        if len(coverage)
        else 0.0,
        "traffic_policy": args.traffic_policy,
        "provenance": "non-reactive candidate-relative consequence"
        if args.traffic_policy == "non_reactive"
        else "reactive-policy simulated consequence",
    }
    write_json(args.output_dir / "candidate_relative_target_summary.json", summary)
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
