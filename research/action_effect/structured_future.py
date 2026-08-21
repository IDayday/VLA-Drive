"""Trajectory-aligned structured future tubes from replay-grounded state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import shapely


STRUCTURED_CHANNELS = (
    "drivable_area",
    "lane_or_connector",
    "route",
    "dynamic_occupancy",
    "relative_longitudinal_velocity",
    "relative_lateral_velocity",
    "dynamic_clearance",
)


@dataclass(frozen=True)
class FutureTubeConfig:
    """Spatial/temporal specification for a candidate-aligned BEV tube."""

    horizons_s: tuple[float, ...] = (1.0, 2.0, 4.0)
    resolution: int = 32
    longitudinal_min_m: float = -8.0
    longitudinal_max_m: float = 24.0
    lateral_min_m: float = -16.0
    lateral_max_m: float = 16.0
    maximum_relative_speed_mps: float = 20.0
    clearance_cap_m: float = 20.0
    proposal_interval_s: float = 0.1

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "FutureTubeConfig":
        return cls(
            horizons_s=tuple(float(item) for item in value["horizons_s"]),
            resolution=int(value["resolution"]),
            longitudinal_min_m=float(value["longitudinal_extent_m"][0]),
            longitudinal_max_m=float(value["longitudinal_extent_m"][1]),
            lateral_min_m=float(value["lateral_extent_m"][0]),
            lateral_max_m=float(value["lateral_extent_m"][1]),
            maximum_relative_speed_mps=float(value["maximum_relative_speed_mps"]),
            clearance_cap_m=float(value["clearance_cap_m"]),
            proposal_interval_s=float(value["proposal_interval_s"]),
        )

    def validate(self) -> None:
        if self.resolution < 4:
            raise ValueError("structured-future resolution must be at least 4")
        if not self.horizons_s or any(value <= 0 for value in self.horizons_s):
            raise ValueError("structured-future horizons must be positive")
        if tuple(sorted(self.horizons_s)) != self.horizons_s:
            raise ValueError("structured-future horizons must be increasing")
        if self.longitudinal_min_m >= self.longitudinal_max_m:
            raise ValueError("invalid longitudinal extent")
        if self.lateral_min_m >= self.lateral_max_m:
            raise ValueError("invalid lateral extent")

    def state_indices(self) -> tuple[int, ...]:
        return tuple(int(round(horizon / self.proposal_interval_s)) for horizon in self.horizons_s)

    def track_indices(self) -> tuple[int, ...]:
        return tuple(index - 1 for index in self.state_indices())


def candidate_aligned_grid(
    state: np.ndarray, config: FutureTubeConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return global cell centers and their candidate-local x/y coordinates."""

    state = np.asarray(state, dtype=np.float64)
    if state.shape[-1] < 5:
        raise ValueError("simulated state must contain pose and planar velocity")
    longitudinal = np.linspace(
        config.longitudinal_min_m,
        config.longitudinal_max_m,
        config.resolution,
        endpoint=False,
    )
    longitudinal += (config.longitudinal_max_m - config.longitudinal_min_m) / (2 * config.resolution)
    lateral = np.linspace(
        config.lateral_max_m,
        config.lateral_min_m,
        config.resolution,
        endpoint=False,
    )
    lateral -= (config.lateral_max_m - config.lateral_min_m) / (2 * config.resolution)
    local_x, local_y = np.meshgrid(longitudinal, lateral)
    cosine, sine = np.cos(state[2]), np.sin(state[2])
    global_x = state[0] + cosine * local_x - sine * local_y
    global_y = state[1] + sine * local_x + cosine * local_y
    return np.stack((global_x, global_y), axis=-1), local_x, local_y


def map_unions(drivable_map: Any, route_lane_ids: Sequence[str]) -> tuple[Any, Any, Any]:
    """Union NAVSIM polygons once per scene for fast candidate rasterization."""

    all_geometries = [drivable_map[token] for token in drivable_map.tokens]
    lane_geometries = [
        drivable_map[token]
        for token, map_type in zip(drivable_map.tokens, drivable_map.map_types)
        if getattr(map_type, "name", str(map_type)) in {"LANE", "LANE_CONNECTOR"}
    ]
    route_geometries = [
        drivable_map[token] for token in route_lane_ids if token in drivable_map.token_to_idx
    ]
    empty = shapely.GeometryCollection()
    return (
        shapely.union_all(all_geometries) if all_geometries else empty,
        shapely.union_all(lane_geometries) if lane_geometries else empty,
        shapely.union_all(route_geometries) if route_geometries else empty,
    )


def _contains(geometry: Any, grid: np.ndarray) -> np.ndarray:
    if geometry is None or geometry.is_empty:
        return np.zeros(grid.shape[:2], dtype=np.float32)
    return shapely.contains_xy(geometry, grid[..., 0], grid[..., 1]).astype(np.float32)


def _object_velocity(obj: Any) -> tuple[float, float]:
    velocity = getattr(obj, "velocity", None)
    return (
        float(getattr(velocity, "x", 0.0)),
        float(getattr(velocity, "y", 0.0)),
    )


def rasterize_future_slice(
    *,
    state: np.ndarray,
    tracked_objects: Iterable[Any],
    static_unions: tuple[Any, Any, Any],
    config: FutureTubeConfig,
) -> np.ndarray:
    """Rasterize one horizon under the log-replay traffic assumption."""

    grid, local_x, local_y = candidate_aligned_grid(state, config)
    output = np.zeros((len(STRUCTURED_CHANNELS), config.resolution, config.resolution), dtype=np.float32)
    for channel, geometry in enumerate(static_unions):
        output[channel] = _contains(geometry, grid)

    cosine, sine = np.cos(state[2]), np.sin(state[2])
    ego_longitudinal = float(state[3])
    ego_lateral = float(state[4])
    nearby_geometries: list[Any] = []
    best_distance = np.full(grid.shape[:2], np.inf, dtype=np.float64)
    for obj in tracked_objects:
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
        nearby_geometries.append(geometry)
        inside = shapely.contains_xy(geometry, grid[..., 0], grid[..., 1])
        # A coarse grid can miss a small object. Preserve one occupied cell at
        # its center so dynamic supervision does not vanish by raster aliasing.
        if not inside.any():
            nearest = np.unravel_index(
                np.argmin(np.square(local_x - relative_x) + np.square(local_y - relative_y)),
                local_x.shape,
            )
            inside[nearest] = True
        points = shapely.points(grid[..., 0], grid[..., 1])
        distance = np.asarray(shapely.distance(points, geometry), dtype=np.float64)
        update = inside | (distance < best_distance)
        velocity_x, velocity_y = _object_velocity(obj)
        relative_global_x = velocity_x - (
            ego_longitudinal * cosine - ego_lateral * sine
        )
        relative_global_y = velocity_y - (
            ego_longitudinal * sine + ego_lateral * cosine
        )
        relative_longitudinal = cosine * relative_global_x + sine * relative_global_y
        relative_lateral = -sine * relative_global_x + cosine * relative_global_y
        output[4][inside] = np.clip(
            relative_longitudinal / config.maximum_relative_speed_mps, -1.0, 1.0
        )
        output[5][inside] = np.clip(
            relative_lateral / config.maximum_relative_speed_mps, -1.0, 1.0
        )
        output[3][inside] = 1.0
        best_distance[update] = np.minimum(best_distance[update], distance[update])
    if nearby_geometries:
        union = shapely.union_all(nearby_geometries)
        points = shapely.points(grid[..., 0], grid[..., 1])
        clearance = np.asarray(shapely.distance(points, union), dtype=np.float32)
        output[6] = np.clip(clearance / config.clearance_cap_m, 0.0, 1.0)
    else:
        output[6] = 1.0
    return output


def build_future_tube(
    *,
    simulated_states: np.ndarray,
    future_tracks: Sequence[Any],
    static_unions: tuple[Any, Any, Any],
    config: FutureTubeConfig,
) -> np.ndarray:
    """Build ``[horizon, channel, height, width]`` candidate target."""

    config.validate()
    state_indices, track_indices = config.state_indices(), config.track_indices()
    if max(state_indices) >= len(simulated_states) or max(track_indices) >= len(future_tracks):
        raise ValueError("metric cache does not cover all configured horizons")
    slices = []
    for state_index, track_index in zip(state_indices, track_indices):
        detections = future_tracks[track_index]
        objects = detections.tracked_objects.tracked_objects
        slices.append(
            rasterize_future_slice(
                state=simulated_states[state_index],
                tracked_objects=objects,
                static_unions=static_unions,
                config=config,
            )
        )
    return np.stack(slices).astype(np.float32)
