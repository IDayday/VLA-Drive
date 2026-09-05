"""Read-only, three-task labels. These are NOT official NC/TTC/DAC/EP scores.

Input states must be the already executed official bicycle/controller rollout,
in global coordinates. No simulator, scorer, responsibility rule or RNG runs here.
Protocol uses logged 10 Hz occupancy through 4.9 s, never forecast/tail padding.
"""
from __future__ import annotations

import hashlib
import json
from typing import Optional

import numpy as np
from shapely import affinity
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.strtree import STRtree

from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    state_array_to_coords_array, coords_array_to_polygon_array,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import BBCoordsIndex, StateIndex

SOURCE_PROTOCOL = "task_future_lite_v1_projected_geometric_gap_10hz_lag0_3_6_9"
FIELDS = ("projected_gap", "road_margin", "route_progress")
TIME_BINS = tuple(tuple(range(0 if h == 0 else 5*h+1, 5*h+6)) for h in range(8))
LAG_INDICES = (0, 3, 6, 9)
DRIVABLE_LAYERS = (SemanticMapLayer.ROADBLOCK, SemanticMapLayer.INTERSECTION,
                    SemanticMapLayer.DRIVABLE_AREA, SemanticMapLayer.CARPARK_AREA)


def array_hash(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str((value.dtype.str, value.shape)).encode() + value.tobytes()).hexdigest()


def label_identity(candidates, initial_ego_hash, metric_cache_hash, *, source_conditions=None):
    identity = dict(source_protocol=SOURCE_PROTOCOL,
                    candidate_coordinates_hash=array_hash(candidates),
                    initial_ego_hash=initial_ego_hash, metric_cache_hash=metric_cache_hash,
                    source_conditions=source_conditions or {})
    identity["cache_key"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return identity


def gap_class_index(distance, contact, no_actor=False):
    if contact:
        return 0
    if no_actor:
        return 4
    # Geometrical contact is intersects, never the floating distance==0 test.
    return int(np.searchsorted([.5, 2., 5.], distance, side="left")) + 1


def logged_occupancy_frames(observation):
    """Explicit missing/resampled entries become unknown, not last-frame copies.

    The caller must establish logged provenance (metric-cache builder or manifest).
    A forecast PDMObservation is not rendered logged merely by its class name.
    """
    if not np.isclose(observation._sample_interval, .1):
        raise ValueError("Lite requires a 10 Hz logged occupancy source")
    if observation._observation_sample_res != 1:
        raise ValueError("Lite rejects resampled occupancy: observation_sample_res must be 1")
    maps = observation._occupancy_maps
    mapping = observation._global_to_local_idcs
    return [maps[t] if t < len(maps) and t < len(mapping) and mapping[t] == t else None
            for t in range(50)]


def road_union(drivable_map):
    indices = drivable_map.get_indices_of_map_type(list(DRIVABLE_LAYERS))
    polygons = [drivable_map[drivable_map.tokens[i]] for i in indices]
    if not polygons or any(p.is_empty or not p.is_valid for p in polygons):
        return None
    result = unary_union(polygons)
    return result if result.is_valid and not result.is_empty else None


def _project(centerline, points):
    # Runtime train caches contain Shapely LineString; official caches PDMPath.
    if hasattr(centerline, "geom_type"):
        return np.array([centerline.project(p) for p in points])
    return np.asarray(centerline.project(points))


def extract_minimal_physical_targets(
    candidates, simulated_states, *, vehicle_parameters, actor_frames,
    drivable_union, map_coverage, centerline, initial_ego_hash, metric_cache_hash,
    source_conditions, red_light_token="red_light", actor_coverage=None,
):
    """Extract immutable labels from one existing [K,41,11] rollout.

    `actor_frames[t] is None` means unknown. An empty map means observed/no actor.
    `map_coverage` must be independently known (not the road polygon union).
    Outside known coverage is unknown, including when the road margin is negative.
    Raw gap is NaN for incomplete coverage even if an observed contact exists;
    contact is retained as diagnostic evidence, not trained as a complete label.
    """
    candidates = np.asarray(candidates)
    states = np.asarray(simulated_states)
    k = len(candidates)
    if candidates.shape != (k, 8, 3) or states.shape != (k, 41, 11):
        raise ValueError("Expected candidates [K,8,3] and official rollout [K,41,11]")
    if not np.isfinite(candidates).all() or not np.isfinite(states).all():
        raise ValueError("Non-finite coordinates/states cannot generate physical labels")
    if source_conditions.get("actor_source") != "logged_interpolated_10hz":
        raise ValueError("Physical targets require explicit logged actor provenance, not forecasts")
    coords = state_array_to_coords_array(states, vehicle_parameters)
    polygons = coords_array_to_polygon_array(coords)
    speeds = np.hypot(states[..., StateIndex.VELOCITY_X], states[..., StateIndex.VELOCITY_Y])
    directions = np.stack((np.cos(states[..., StateIndex.HEADING]),
                           np.sin(states[..., StateIndex.HEADING])), axis=-1)
    values = np.full((k, 8, 3), np.nan, dtype=np.float32)
    valid = np.zeros((k, 8, 3), dtype=bool)
    classes = np.zeros((k, 8), dtype=np.int64)
    contacts = np.zeros((k, 8), dtype=bool)
    no_actor = np.zeros((k, 8), dtype=bool)
    road_known = (drivable_union is not None and map_coverage is not None
                  and drivable_union.is_valid and not drivable_union.is_empty
                  and map_coverage.is_valid and not map_coverage.is_empty)
    frame_trees = []
    for frame in actor_frames:
        if frame is None:
            frame_trees.append(None)
            continue
        actors = [frame[token] for token in frame.tokens
                  if not (red_light_token and red_light_token in token)]
        complete = all(not p.is_empty and p.is_valid for p in actors)
        geometries = [p for p in actors if not p.is_empty and p.is_valid]
        frame_trees.append((STRtree(geometries) if geometries else None, geometries, complete))
    for candidate in range(k):
        start = Point(coords[candidate, 0, BBCoordsIndex.CENTER])
        for h, indices in enumerate(TIME_BINS):
            gap, covered, seen_actor, contact = np.inf, True, False, False
            for t in indices:
                for lag in LAG_INDICES:
                    at = t + lag
                    frame = frame_trees[at] if at < len(frame_trees) else None
                    if frame is None:
                        covered = False
                        continue
                    offset = directions[candidate, t] * speeds[candidate, t] * (lag / 10.)
                    ego = affinity.translate(polygons[candidate, t], *offset)
                    if actor_coverage is not None and not actor_coverage.covers(ego):
                        covered = False
                    tree, geometries, complete = frame
                    covered &= complete
                    if tree is not None:
                        actor = geometries[tree.nearest(ego)]
                        seen_actor = True
                        contact |= ego.intersects(actor)
                        gap = min(gap, ego.distance(actor))
            contacts[candidate, h] = contact
            if covered:
                no_actor[candidate, h] = not seen_actor
                values[candidate, h, 0] = gap if seen_actor else 5.
                classes[candidate, h] = gap_class_index(gap, contact, not seen_actor)
                valid[candidate, h, 0] = True
            if road_known:
                corners = [Point(xy) for t in indices for xy in coords[candidate, t, :4]]
                if all(map_coverage.covers(p) for p in corners):
                    values[candidate, h, 1] = min(
                        p.distance(drivable_union.boundary) * (1 if drivable_union.covers(p) else -1)
                        for p in corners)
                    valid[candidate, h, 1] = True
            if centerline is not None:
                end = Point(coords[candidate, indices[-1], BBCoordsIndex.CENTER])
                arc = _project(centerline, [start, end])
                if arc.shape == (2,) and np.isfinite(arc).all():
                    values[candidate, h, 2] = max(0., arc[1] - arc[0])
                    valid[candidate, h, 2] = True
    return dict(physical_values=values, gap_class=classes, valid=valid,
                geometric_contact=contacts, no_actor=no_actor,
                **label_identity(candidates, initial_ego_hash, metric_cache_hash,
                                 source_conditions=source_conditions))


def extract_from_metric_cache(candidates, simulated_states, metric_cache, *, metric_cache_hash,
                              source_conditions, map_coverage=None, lossless_drivable_union=None):
    """No implicit assumptions about cache creation or map-query coverage."""
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import ego_state_to_state_array
    ego = metric_cache.ego_state
    return extract_minimal_physical_targets(
        candidates, simulated_states,
        vehicle_parameters=ego.car_footprint.vehicle_parameters,
        actor_frames=logged_occupancy_frames(metric_cache.observation),
        drivable_union=lossless_drivable_union, map_coverage=map_coverage,
        centerline=metric_cache.centerline, metric_cache_hash=metric_cache_hash,
        initial_ego_hash=array_hash(ego_state_to_state_array(ego)),
        source_conditions=source_conditions, red_light_token=metric_cache.observation.red_light_token,
    )
