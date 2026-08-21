"""Trajectory-aligned effect tubes for multi-candidate world supervision.

Unlike the retained Phase-5 raw map diagnostic, these channels emphasize
candidate-relative consequences. Static geometry is represented as signed
distance fields, while replayed actors, collision clearance, and the ego swept
footprint retain explicit action dependence and provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES
from scipy.ndimage import distance_transform_edt
import shapely

from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    coords_array_to_polygon_array,
    state_array_to_coords_array,
)

from research.action_effect.structured_future import candidate_aligned_grid, map_unions


EFFECT_TUBE_CHANNELS = (
    "candidate_relative_dynamic_occupancy",
    "drivable_area_sdf",
    "lane_sdf",
    "route_sdf",
    "relative_longitudinal_velocity",
    "relative_lateral_velocity",
    "dynamic_clearance",
    "dynamic_collision_field",
    "ego_swept_footprint",
)

BINARY_EFFECT_CHANNELS = (0, 7, 8)
SDF_EFFECT_CHANNELS = (1, 2, 3)
VELOCITY_EFFECT_CHANNELS = (4, 5)
CLEARANCE_EFFECT_CHANNEL = 6


@dataclass(frozen=True)
class EffectTubeConfig:
    """Spatial, temporal, and normalization contract for an effect tube."""

    horizons_s: tuple[float, ...] = (1.0, 2.0, 4.0)
    resolution: int = 32
    longitudinal_min_m: float = -8.0
    longitudinal_max_m: float = 24.0
    lateral_min_m: float = -16.0
    lateral_max_m: float = 16.0
    maximum_relative_speed_mps: float = 20.0
    distance_cap_m: float = 12.0
    clearance_cap_m: float = 20.0
    collision_radius_m: float = 2.8
    proposal_interval_s: float = 0.1

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "EffectTubeConfig":
        return cls(
            horizons_s=tuple(float(item) for item in value["horizons_s"]),
            resolution=int(value["resolution"]),
            longitudinal_min_m=float(value["longitudinal_extent_m"][0]),
            longitudinal_max_m=float(value["longitudinal_extent_m"][1]),
            lateral_min_m=float(value["lateral_extent_m"][0]),
            lateral_max_m=float(value["lateral_extent_m"][1]),
            maximum_relative_speed_mps=float(value["maximum_relative_speed_mps"]),
            distance_cap_m=float(value["distance_cap_m"]),
            clearance_cap_m=float(value["clearance_cap_m"]),
            collision_radius_m=float(value["collision_radius_m"]),
            proposal_interval_s=float(value["proposal_interval_s"]),
        )

    def validate(self) -> None:
        if self.resolution < 4 or self.resolution % 4:
            raise ValueError("effect-tube resolution must be >=4 and divisible by four")
        if not self.horizons_s or tuple(sorted(self.horizons_s)) != self.horizons_s:
            raise ValueError("effect-tube horizons must be non-empty and increasing")
        if any(value <= 0 for value in self.horizons_s):
            raise ValueError("effect-tube horizons must be positive")
        if self.longitudinal_min_m >= self.longitudinal_max_m:
            raise ValueError("invalid longitudinal extent")
        if self.lateral_min_m >= self.lateral_max_m:
            raise ValueError("invalid lateral extent")
        if min(
            self.maximum_relative_speed_mps,
            self.distance_cap_m,
            self.clearance_cap_m,
            self.collision_radius_m,
            self.proposal_interval_s,
        ) <= 0:
            raise ValueError("effect-tube scales must be positive")

    def state_indices(self) -> tuple[int, ...]:
        return tuple(int(round(value / self.proposal_interval_s)) for value in self.horizons_s)

    def track_indices(self) -> tuple[int, ...]:
        return tuple(value - 1 for value in self.state_indices())

    @property
    def cell_size_m(self) -> tuple[float, float]:
        return (
            (self.longitudinal_max_m - self.longitudinal_min_m) / self.resolution,
            (self.lateral_max_m - self.lateral_min_m) / self.resolution,
        )


def signed_distance_from_mask(mask: np.ndarray, config: EffectTubeConfig) -> np.ndarray:
    """Return a clipped signed distance, positive inside and negative outside."""

    mask = np.asarray(mask, dtype=bool)
    # Grid rows are lateral and columns are longitudinal.
    sampling = (config.cell_size_m[1], config.cell_size_m[0])
    inside = distance_transform_edt(mask, sampling=sampling)
    outside = distance_transform_edt(~mask, sampling=sampling)
    signed = inside - outside
    return np.clip(signed / config.distance_cap_m, -1.0, 1.0).astype(np.float32)


def _contains(geometry: Any, grid: np.ndarray) -> np.ndarray:
    if geometry is None or geometry.is_empty:
        return np.zeros(grid.shape[:2], dtype=bool)
    return np.asarray(shapely.contains_xy(geometry, grid[..., 0], grid[..., 1]), dtype=bool)


def _object_velocity(obj: Any) -> tuple[float, float]:
    velocity = getattr(obj, "velocity", None)
    return float(getattr(velocity, "x", 0.0)), float(getattr(velocity, "y", 0.0))


def _dynamic_fields(
    *,
    state: np.ndarray,
    tracked_objects: Iterable[Any],
    grid: np.ndarray,
    local_x: np.ndarray,
    local_y: np.ndarray,
    config: EffectTubeConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    occupancy = np.zeros(grid.shape[:2], dtype=np.float32)
    relative_longitudinal = np.zeros_like(occupancy)
    relative_lateral = np.zeros_like(occupancy)
    best_distance = np.full(grid.shape[:2], np.inf, dtype=np.float64)
    geometries: list[Any] = []
    cosine, sine = np.cos(state[2]), np.sin(state[2])
    ego_velocity_x, ego_velocity_y = float(state[3]), float(state[4])
    points = shapely.points(grid[..., 0], grid[..., 1])
    for obj in tracked_objects:
        if getattr(obj, "tracked_object_type", None) not in AGENT_TYPES:
            continue
        center = getattr(obj, "center", None)
        if center is None:
            continue
        dx, dy = float(center.x - state[0]), float(center.y - state[1])
        relative_x = cosine * dx + sine * dy
        relative_y = -sine * dx + cosine * dy
        if not (
            config.longitudinal_min_m - 8.0 <= relative_x <= config.longitudinal_max_m + 8.0
            and config.lateral_min_m - 8.0 <= relative_y <= config.lateral_max_m + 8.0
        ):
            continue
        geometry = obj.box.geometry
        geometries.append(geometry)
        inside = _contains(geometry, grid)
        if not inside.any():
            nearest = np.unravel_index(
                np.argmin(np.square(local_x - relative_x) + np.square(local_y - relative_y)),
                local_x.shape,
            )
            inside[nearest] = True
        distance = np.asarray(shapely.distance(points, geometry), dtype=np.float64)
        update = inside | (distance < best_distance)
        object_vx, object_vy = _object_velocity(obj)
        delta_vx, delta_vy = object_vx - ego_velocity_x, object_vy - ego_velocity_y
        longitudinal = cosine * delta_vx + sine * delta_vy
        lateral = -sine * delta_vx + cosine * delta_vy
        occupancy[inside] = 1.0
        relative_longitudinal[inside] = np.clip(
            longitudinal / config.maximum_relative_speed_mps, -1.0, 1.0
        )
        relative_lateral[inside] = np.clip(
            lateral / config.maximum_relative_speed_mps, -1.0, 1.0
        )
        best_distance[update] = np.minimum(best_distance[update], distance[update])
    if geometries:
        dynamic_union = shapely.union_all(geometries)
        clearance_m = np.asarray(shapely.distance(points, dynamic_union), dtype=np.float32)
    else:
        clearance_m = np.full(grid.shape[:2], config.clearance_cap_m, dtype=np.float32)
    clearance = np.clip(clearance_m / config.clearance_cap_m, 0.0, 1.0)
    collision = (clearance_m <= config.collision_radius_m).astype(np.float32)
    return occupancy, relative_longitudinal, relative_lateral, clearance, collision


def _swept_footprint(
    *,
    states_to_horizon: np.ndarray,
    vehicle_parameters: Any,
    grid: np.ndarray,
) -> np.ndarray:
    coords = state_array_to_coords_array(
        np.asarray(states_to_horizon, dtype=np.float64)[None], vehicle_parameters
    )[0]
    polygons = coords_array_to_polygon_array(coords)
    swept = shapely.union_all(list(polygons))
    mask = _contains(swept, grid)
    # Preserve a supervised cell when a deliberately coarse grid has no center
    # strictly inside the footprint. Do not project an entirely off-grid sweep
    # into the raster: the nearest center must still be within half a cell
    # diagonal of the polygon.
    if not mask.any() and not swept.is_empty:
        points = shapely.points(grid[..., 0], grid[..., 1])
        distance = np.asarray(shapely.distance(points, swept), dtype=np.float64)
        nearest = np.unravel_index(np.argmin(distance), distance.shape)
        dx = float(np.ptp(grid[0, :, 0]) / max(grid.shape[1] - 1, 1))
        dy = float(np.ptp(grid[:, 0, 1]) / max(grid.shape[0] - 1, 1))
        if distance[nearest] <= 0.5 * float(np.hypot(dx, dy)):
            mask[nearest] = True
    return mask.astype(np.float32)


def build_effect_tube(
    *,
    simulated_states: np.ndarray,
    future_tracks: Sequence[Any],
    static_unions: tuple[Any, Any, Any],
    vehicle_parameters: Any,
    config: EffectTubeConfig,
) -> np.ndarray:
    """Build an ``[horizon, channel, height, width]`` effect target."""

    config.validate()
    states = np.asarray(simulated_states, dtype=np.float64)
    state_indices = config.state_indices()
    track_indices = config.track_indices()
    if max(state_indices) >= len(states) or max(track_indices) >= len(future_tracks):
        raise ValueError("metric cache does not cover configured effect horizons")
    result = np.zeros(
        (len(state_indices), len(EFFECT_TUBE_CHANNELS), config.resolution, config.resolution),
        dtype=np.float32,
    )
    for horizon_index, (state_index, track_index) in enumerate(zip(state_indices, track_indices)):
        state = states[state_index]
        grid, local_x, local_y = candidate_aligned_grid(state, config)  # type: ignore[arg-type]
        for channel, geometry in zip((1, 2, 3), static_unions):
            result[horizon_index, channel] = signed_distance_from_mask(
                _contains(geometry, grid), config
            )
        objects = future_tracks[track_index].tracked_objects.tracked_objects
        dynamic = _dynamic_fields(
            state=state,
            tracked_objects=objects,
            grid=grid,
            local_x=local_x,
            local_y=local_y,
            config=config,
        )
        result[horizon_index, 0] = dynamic[0]
        result[horizon_index, 4] = dynamic[1]
        result[horizon_index, 5] = dynamic[2]
        result[horizon_index, 6] = dynamic[3]
        result[horizon_index, 7] = dynamic[4]
        result[horizon_index, 8] = _swept_footprint(
            states_to_horizon=states[: state_index + 1],
            vehicle_parameters=vehicle_parameters,
            grid=grid,
        )
    return result


def effect_map_unions(drivable_map: Any, route_lane_ids: Sequence[str]) -> tuple[Any, Any, Any]:
    """Compatibility wrapper documenting reuse of the official map cache."""

    return map_unions(drivable_map, route_lane_ids)
