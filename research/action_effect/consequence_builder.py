"""Replay-grounded and reactive-model consequences for local ego actions.

The functions in this module preserve provenance explicitly.  Map/geometry
quantities are returned under ``exact``; non-responsive logged traffic under
``log_replay``; IDM outputs under ``reactive_model``.  None of these are named
or treated as a true causal counterfactual.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from shapely import Point, union_all

from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
    PDMScorer,
    PDMScorerConfig,
)
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    coords_array_to_polygon_array,
    state_array_to_coords_array,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    BBCoordsIndex,
    StateIndex,
)
from navsim.planning.simulation.observation.navsim_idm_agents import NavsimIDMAgents
from navsim.traffic_agents_policies.log_replay_traffic_agents import LogReplayTrafficAgents
from navsim.traffic_agents_policies.navsim_IDM_traffic_agents import NavsimIDMTrafficAgents


@dataclass(frozen=True)
class ConsequenceConfig:
    """Numerical settings that define cached consequence values."""

    proposal_num_poses: int = 40
    proposal_interval_length: float = 0.1
    candidate_interval_length: float = 0.5
    clearance_cap_m: float = 20.0
    occupancy_radius_m: float = 10.0
    no_event_time_s: float = 5.0

    @property
    def proposal_sampling(self) -> TrajectorySampling:
        return TrajectorySampling(
            num_poses=self.proposal_num_poses,
            interval_length=self.proposal_interval_length,
        )


class CapturingTrafficPolicy:
    """Transparent policy proxy retaining the exact tracks seen by the scorer."""

    def __init__(self, policy: Any):
        self.policy = policy
        self.last_detections_tracks: Sequence[Any] | None = None

    def simulate_environment(self, simulated_ego_states: np.ndarray, metric_cache: MetricCache) -> Sequence[Any]:
        tracks = self.policy.simulate_environment(simulated_ego_states, metric_cache)
        self.last_detections_tracks = tracks
        return tracks


def build_log_replay_policy(config: ConsequenceConfig) -> CapturingTrafficPolicy:
    """Construct the official non-responsive log replay policy."""

    return CapturingTrafficPolicy(LogReplayTrafficAgents(config.proposal_sampling))


def build_reactive_policy(config: ConsequenceConfig, map_root: str) -> CapturingTrafficPolicy:
    """Construct NAVSIM-v2's official IDM policy with published defaults."""

    agents = NavsimIDMAgents(
        target_velocity=10.0,
        min_gap_to_lead_agent=1.0,
        headway_time=1.5,
        accel_max=1.0,
        decel_max=2.0,
        open_loop_detections_types=[],
        minimum_path_length=20.0,
        planned_trajectory_samples=None,
        planned_trajectory_sample_interval=None,
        radius=100.0,
        add_open_loop_parked_vehicles=True,
        idm_snap_threshold=3.0,
    )
    return CapturingTrafficPolicy(
        NavsimIDMTrafficAgents(
            config.proposal_sampling,
            agents,
            map_root_override=map_root,
        )
    )


def physical_kinematics(trajectory: np.ndarray, interval_length: float = 0.5) -> dict[str, float]:
    """Compute exact geometry/kinematics in the current rear-axle frame."""

    poses = np.asarray(trajectory, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 3:
        raise ValueError(f"trajectory must be [T,3], got {poses.shape}")
    points = np.concatenate([np.zeros((1, 2)), poses[:, :2]], axis=0)
    displacement = np.linalg.norm(np.diff(points, axis=0), axis=1)
    speed = displacement / interval_length
    acceleration = np.diff(speed) / interval_length
    jerk = np.diff(acceleration) / interval_length
    yaw_step = np.diff(np.unwrap(np.concatenate([[0.0], poses[:, 2]])))
    curvature = np.divide(
        np.abs(yaw_step),
        displacement,
        out=np.zeros_like(displacement),
        where=displacement >= 0.05,
    )
    return {
        "ego_progress_m": float(np.sum(displacement)),
        "max_speed_mps": float(np.max(speed, initial=0.0)),
        "max_acceleration_mps2": float(np.max(acceleration, initial=0.0)),
        "max_deceleration_mps2": float(max(0.0, -np.min(acceleration, initial=0.0))),
        "max_abs_jerk_mps3": float(np.max(np.abs(jerk), initial=0.0)),
        "max_abs_curvature_inv_m": float(np.max(curvature, initial=0.0)),
        "max_abs_yaw_step_rad": float(np.max(np.abs(yaw_step), initial=0.0)),
    }


def _finite_or_cap(value: float, cap: float) -> float:
    return float(min(value, cap)) if math.isfinite(value) else float(cap)


def geometry_against_tracks(
    simulated_states: np.ndarray,
    detections_tracks: Sequence[Any],
    vehicle_parameters: Any,
    *,
    interval_length: float,
    clearance_cap_m: float,
    occupancy_radius_m: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Measure footprint overlap and clearance against one traffic assumption."""

    states = np.asarray(simulated_states, dtype=np.float64)
    if states.ndim != 2 or len(states) != len(detections_tracks):
        raise ValueError(f"states/tracks length mismatch: {states.shape} vs {len(detections_tracks)}")
    coords = state_array_to_coords_array(states[None], vehicle_parameters)[0]
    ego_polygons = coords_array_to_polygon_array(coords)
    num_steps = len(states)
    min_all_t = np.full(num_steps, clearance_cap_m, dtype=np.float32)
    min_dynamic_t = np.full(num_steps, clearance_cap_m, dtype=np.float32)
    overlap_t = np.zeros(num_steps, dtype=np.bool_)
    dynamic_overlap_t = np.zeros(num_steps, dtype=np.bool_)
    dynamic_count_t = np.zeros(num_steps, dtype=np.int16)
    nearest_relative_lon = 0.0
    nearest_relative_lat = 0.0
    nearest_dynamic = float("inf")
    static_collision = False
    dynamic_collision = False

    for time_index, (ego_polygon, detections) in enumerate(zip(ego_polygons, detections_tracks)):
        objects = detections.tracked_objects.tracked_objects
        for tracked_object in objects:
            geometry = tracked_object.box.geometry
            distance = float(ego_polygon.distance(geometry))
            intersects = bool(ego_polygon.intersects(geometry))
            min_all_t[time_index] = min(float(min_all_t[time_index]), distance)
            is_dynamic = tracked_object.tracked_object_type in AGENT_TYPES
            if is_dynamic:
                min_dynamic_t[time_index] = min(float(min_dynamic_t[time_index]), distance)
                dynamic_count_t[time_index] += int(distance <= occupancy_radius_m)
                dynamic_overlap_t[time_index] |= intersects
                dynamic_collision |= intersects
                if distance < nearest_dynamic:
                    nearest_dynamic = distance
                    velocity = getattr(tracked_object, "velocity", None)
                    object_velocity = np.array(
                        [getattr(velocity, "x", 0.0), getattr(velocity, "y", 0.0)],
                        dtype=np.float64,
                    )
                    ego_velocity = states[time_index, StateIndex.VELOCITY_2D]
                    relative = object_velocity - ego_velocity
                    heading = states[time_index, StateIndex.HEADING]
                    longitudinal = np.array([np.cos(heading), np.sin(heading)])
                    lateral = np.array([-np.sin(heading), np.cos(heading)])
                    nearest_relative_lon = float(np.dot(relative, longitudinal))
                    nearest_relative_lat = float(np.dot(relative, lateral))
            else:
                static_collision |= intersects
            overlap_t[time_index] |= intersects

    any_collision_indices = np.flatnonzero(overlap_t)
    dynamic_collision_indices = np.flatnonzero(dynamic_overlap_t)
    swept_footprint = union_all(list(ego_polygons))
    summary = {
        "minimum_clearance_m": _finite_or_cap(float(np.min(min_all_t)), clearance_cap_m),
        "minimum_dynamic_clearance_m": _finite_or_cap(float(np.min(min_dynamic_t)), clearance_cap_m),
        "static_object_collision": bool(static_collision),
        "dynamic_collision": bool(dynamic_collision),
        "collision_time_s": (
            float(any_collision_indices[0] * interval_length) if len(any_collision_indices) else None
        ),
        "dynamic_collision_time_s": (
            float(dynamic_collision_indices[0] * interval_length)
            if len(dynamic_collision_indices)
            else None
        ),
        "nearest_relative_longitudinal_velocity_mps": nearest_relative_lon,
        "nearest_relative_lateral_velocity_mps": nearest_relative_lat,
        "max_dynamic_agents_within_radius": int(np.max(dynamic_count_t, initial=0)),
        "dynamic_occupancy_fraction": float(np.mean(dynamic_count_t > 0)),
        "ego_swept_footprint_area_m2": float(swept_footprint.area),
    }
    arrays = {
        "minimum_clearance_t": min_all_t,
        "minimum_dynamic_clearance_t": min_dynamic_t,
        "time_indexed_overlap": overlap_t,
        "time_indexed_dynamic_overlap": dynamic_overlap_t,
        "dynamic_agents_within_radius_t": dynamic_count_t,
    }
    return summary, arrays


def score_under_assumption(
    metric_cache: MetricCache,
    trajectory: np.ndarray,
    simulator: PDMSimulator,
    scorer: PDMScorer,
    policy: CapturingTrafficPolicy,
    config: ConsequenceConfig,
) -> tuple[dict[str, Any], dict[str, float], np.ndarray, dict[str, np.ndarray]]:
    """Run the unmodified NAVSIM-v2 scorer and retain disaggregated labels."""

    model_trajectory = Trajectory(
        poses=np.asarray(trajectory, dtype=np.float32),
        trajectory_sampling=TrajectorySampling(num_poses=len(trajectory), interval_length=0.5),
    )
    score_frame, simulated_states = pdm_score(
        metric_cache=metric_cache,
        model_trajectory=model_trajectory,
        future_sampling=config.proposal_sampling,
        simulator=simulator,
        scorer=scorer,
        traffic_agents_policy=policy,
    )
    if policy.last_detections_tracks is None:
        raise RuntimeError("traffic policy did not capture detections tracks")
    geometry, arrays = geometry_against_tracks(
        simulated_states,
        policy.last_detections_tracks,
        metric_cache.ego_state.car_footprint.vehicle_parameters,
        interval_length=config.proposal_interval_length,
        clearance_cap_m=config.clearance_cap_m,
        occupancy_radius_m=config.occupancy_radius_m,
    )
    row = score_frame.iloc[0]
    ttc_infraction_s = float(scorer.time_to_ttc_infraction(1))
    collision_infraction_s = float(scorer.time_to_at_fault_collision(1))
    result = {
        "available": True,
        "no_at_fault_collision": float(row["no_at_fault_collisions"]),
        "traffic_light_compliance": float(row["traffic_light_compliance"]),
        "time_to_collision_within_bound": float(row["time_to_collision_within_bound"]),
        "ttc_infraction_time_s": _finite_or_cap(ttc_infraction_s, config.no_event_time_s),
        "ttc_infraction_observed": bool(math.isfinite(ttc_infraction_s)),
        "at_fault_collision_time_s": _finite_or_cap(collision_infraction_s, config.no_event_time_s),
        "at_fault_collision_observed": bool(math.isfinite(collision_infraction_s)),
        "pdm_score": float(row["pdm_score"]),
        **geometry,
    }
    exact_score_components = {
        "drivable_area_compliance": float(row["drivable_area_compliance"]),
        "driving_direction_compliance": float(row["driving_direction_compliance"]),
        "lane_keeping": float(row["lane_keeping"]),
        "history_comfort": float(row["history_comfort"]),
        "ego_progress": float(row["ego_progress"]),
    }
    arrays["simulated_states"] = np.asarray(simulated_states, dtype=np.float32)
    return result, exact_score_components, simulated_states, arrays


def exact_map_consequence(
    metric_cache: MetricCache,
    trajectory: np.ndarray,
    simulated_states: np.ndarray,
    score_row: Any,
    geometry: dict[str, Any],
    config: ConsequenceConfig,
) -> dict[str, Any]:
    """Assemble map/vehicle consequences independent of agent response."""

    coords = state_array_to_coords_array(
        np.asarray(simulated_states, dtype=np.float64)[None],
        metric_cache.ego_state.car_footprint.vehicle_parameters,
    )[0]
    centers = coords[:, BBCoordsIndex.CENTER]
    centerline = metric_cache.centerline.linestring
    route_deviations = np.asarray([Point(*point).distance(centerline) for point in centers])
    in_intersection = np.asarray(
        [
            metric_cache.drivable_area_map.is_in_layer(Point(*point), SemanticMapLayer.INTERSECTION)
            for point in centers
        ],
        dtype=np.bool_,
    )
    return {
        "available": True,
        "drivable_area_compliance": float(score_row["drivable_area_compliance"]),
        "driving_direction_compliance": float(score_row["driving_direction_compliance"]),
        "lane_keeping": float(score_row["lane_keeping"]),
        "history_comfort": float(score_row["history_comfort"]),
        "extended_comfort": None,
        "official_ego_progress_score": float(score_row["ego_progress"]),
        "centerline_progress_m": float(metric_cache.centerline.project([Point(*centers[0]), Point(*centers[-1])])[1]
                                       - metric_cache.centerline.project([Point(*centers[0]), Point(*centers[-1])])[0]),
        "route_deviation_mean_m": float(np.mean(route_deviations)),
        "route_deviation_max_m": float(np.max(route_deviations, initial=0.0)),
        "route_consistency": bool(
            float(score_row["driving_direction_compliance"]) == 1.0
            and float(score_row["lane_keeping"]) == 1.0
        ),
        "intersection_fraction": float(np.mean(in_intersection)),
        "static_object_collision": bool(geometry["static_object_collision"]),
        "ego_swept_footprint_area_m2": float(geometry["ego_swept_footprint_area_m2"]),
        **physical_kinematics(trajectory, config.candidate_interval_length),
    }


def make_scorer(config: ConsequenceConfig) -> tuple[PDMSimulator, PDMScorer]:
    """Create scorer components without the human-error penalty filter."""

    sampling = config.proposal_sampling
    return (
        PDMSimulator(sampling),
        PDMScorer(sampling, PDMScorerConfig(human_penalty_filter=False)),
    )


def validate_consequence_namespaces(row: dict[str, Any]) -> None:
    """Assert that consequence fields retain their declared provenance."""

    expected = {
        "exact": "exact",
        "log_replay": "log_replay",
        "reactive_model": "reactive_model",
        "unknown": "unknown",
    }
    for namespace, provenance in expected.items():
        value = row.get(namespace)
        if not isinstance(value, dict) or value.get("provenance") != provenance:
            raise AssertionError(f"invalid {namespace} provenance: {value}")
    forbidden = {
        "exact": {
            "dynamic_collision",
            "minimum_dynamic_clearance_m",
            "ttc_infraction_time_s",
            "no_at_fault_collision",
        },
        "log_replay": {
            "drivable_area_compliance",
            "driving_direction_compliance",
            "static_object_collision",
        },
        "reactive_model": {
            "drivable_area_compliance",
            "driving_direction_compliance",
            "static_object_collision",
        },
    }
    for namespace, forbidden_fields in forbidden.items():
        overlap = forbidden_fields & set(row[namespace])
        if overlap:
            raise AssertionError(f"fields leaked into {namespace}: {sorted(overlap)}")
