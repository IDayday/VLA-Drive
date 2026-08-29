"""Legally replay logged geometry against each fixed ego candidate.

This module never predicts or edits another actor's response. It transforms the
same logged actor continuation into every candidate ego frame and exposes an
uncertainty mask wherever that non-reactive replay may be unreliable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from .feature_store import stable_array_hash


EGO_EFFECT_NAMES = (
    "candidate_x",
    "candidate_y",
    "candidate_heading",
    "estimated_speed",
    "acceleration",
    "yaw_rate",
    "curvature",
    "jerk",
    "swept_corner_0_x",
    "swept_corner_0_y",
    "swept_corner_1_x",
    "swept_corner_1_y",
    "swept_corner_2_x",
    "swept_corner_2_y",
    "swept_corner_3_x",
    "swept_corner_3_y",
)
MAP_EFFECT_NAMES = (
    "distance_to_route_centerline",
    "route_longitudinal_progress",
    "heading_error_to_route",
    "left_drivable_clearance",
    "right_drivable_clearance",
    "footprint_outside_drivable_ratio",
    "distance_to_static_obstacle",
    "distance_to_map_boundary",
)
ACTOR_EFFECT_NAMES = (
    "relative_x",
    "relative_y",
    "relative_velocity_x",
    "relative_velocity_y",
    "relative_heading",
    "oriented_box_clearance",
    "center_distance",
    "longitudinal_relation",
    "lateral_relation",
    "time_to_closest_approach",
    "distance_at_closest_approach",
    "swept_box_distance",
    "relative_speed",
)

# The v1 tensors above stay intact for cache compatibility.  The independent
# relabel Gate makes the leakage-safe core/diagnostic boundary explicit through
# these projected schemas.
PRIMITIVE_EGO_EFFECT_NAMES = EGO_EFFECT_NAMES
PRIMITIVE_MAP_EFFECT_NAMES = (
    "signed_route_center_offset",
    "route_longitudinal_coordinate",
    "heading_error_to_route",
    "left_boundary_signed_clearance",
    "right_boundary_signed_clearance",
    "distance_to_static_obstacle",
    "distance_to_map_boundary",
)
PRIMITIVE_ACTOR_EFFECT_NAMES = (
    "relative_x",
    "relative_y",
    "relative_velocity_x",
    "relative_velocity_y",
    "relative_heading",
    "oriented_box_clearance",
    "center_distance",
    "longitudinal_relation",
    "lateral_relation",
    "relative_speed",
)
ENGINEERED_EFFECT_NAMES = (
    "footprint_outside_drivable_ratio",
    "time_to_closest_approach",
    "distance_at_closest_approach",
    "swept_box_distance",
)
PRIMITIVE_MAP_INDICES = (0, 1, 2, 3, 4, 6, 7)
PRIMITIVE_ACTOR_INDICES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 12)
ENGINEERED_MAP_INDICES = (5,)
ENGINEERED_ACTOR_INDICES = (9, 10, 11)
PRIMITIVE_EFFECT_SCHEMA_VERSION = "primitive_effect.v1"
ENGINEERED_EFFECT_SCHEMA_VERSION = "engineered_effect.v1"


class EffectConstructionError(RuntimeError):
    """The replay inputs violate the fixed, leakage-free schema."""


def _as_float32(value: npt.ArrayLike) -> npt.NDArray[np.float32]:
    return np.ascontiguousarray(np.asarray(value, dtype=np.float32))


def wrap_angle(value: npt.ArrayLike) -> npt.NDArray[np.float32]:
    angle = np.asarray(value, dtype=np.float32)
    return ((angle + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)


def _rotate(points: npt.ArrayLike, heading: float) -> npt.NDArray[np.float32]:
    values = np.asarray(points, dtype=np.float32)
    cosine, sine = np.cos(heading), np.sin(heading)
    matrix = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float32)
    return values @ matrix.T


def oriented_box_corners(
    center: npt.ArrayLike, heading: float, length: float, width: float
) -> npt.NDArray[np.float32]:
    if length <= 0 or width <= 0:
        raise ValueError("oriented-box length and width must be positive")
    half_length, half_width = length / 2.0, width / 2.0
    local = np.array(
        [
            [half_length, half_width],
            [half_length, -half_width],
            [-half_length, -half_width],
            [-half_length, half_width],
        ],
        dtype=np.float32,
    )
    return _rotate(local, heading) + np.asarray(center, dtype=np.float32)


def _cross(origin: npt.NDArray[np.float32], first: npt.NDArray[np.float32], second: npt.NDArray[np.float32]) -> float:
    return float(np.cross(first - origin, second - origin))


def convex_hull(points: npt.ArrayLike) -> npt.NDArray[np.float32]:
    values = np.unique(np.asarray(points, dtype=np.float32), axis=0)
    if len(values) <= 2:
        return values
    ordered = values[np.lexsort((values[:, 1], values[:, 0]))]
    lower: list[npt.NDArray[np.float32]] = []
    for point in ordered:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[npt.NDArray[np.float32]] = []
    for point in ordered[::-1]:
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float32)


def _point_segment_distance(
    point: npt.NDArray[np.float32],
    start: npt.NDArray[np.float32],
    end: npt.NDArray[np.float32],
) -> float:
    delta = end - start
    denominator = float(delta @ delta)
    if denominator <= 1e-12:
        return float(np.linalg.norm(point - start))
    fraction = float(np.clip(((point - start) @ delta) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + fraction * delta)))


def _points_to_segments_distance(
    points: npt.ArrayLike,
    starts: npt.ArrayLike,
    ends: npt.ArrayLike,
) -> npt.NDArray[np.float32]:
    """Return every point-to-segment distance without changing the geometry."""

    point_values = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    start_values = np.asarray(starts, dtype=np.float32).reshape(-1, 2)
    end_values = np.asarray(ends, dtype=np.float32).reshape(-1, 2)
    delta = end_values - start_values
    denominator = np.sum(delta * delta, axis=-1)
    relative = point_values[:, None, :] - start_values[None, :, :]
    numerator = np.sum(relative * delta[None, :, :], axis=-1)
    safe_denominator = np.where(denominator > 1.0e-12, denominator, 1.0)
    fraction = np.clip(numerator / safe_denominator[None, :], 0.0, 1.0)
    projection = start_values[None, :, :] + fraction[..., None] * delta[None, :, :]
    distance = np.linalg.norm(point_values[:, None, :] - projection, axis=-1)
    degenerate = denominator <= 1.0e-12
    if np.any(degenerate):
        distance[:, degenerate] = np.linalg.norm(
            point_values[:, None, :] - start_values[None, degenerate, :], axis=-1
        )
    return np.asarray(distance, dtype=np.float32)


def point_in_polygon(point: npt.ArrayLike, polygon: npt.ArrayLike) -> bool:
    value = np.asarray(point, dtype=np.float32)
    vertices = np.asarray(polygon, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
        raise ValueError(f"polygon must be [N>=3,2], got {vertices.shape}")
    previous = np.roll(vertices, 1, axis=0)
    if float(
        _points_to_segments_distance(value[None], previous, vertices).min()
    ) <= 1.0e-6:
        return True
    crosses = (vertices[:, 1] > value[1]) != (previous[:, 1] > value[1])
    x_intersection = (
        (previous[:, 0] - vertices[:, 0])
        * (value[1] - vertices[:, 1])
        / (previous[:, 1] - vertices[:, 1] + 1.0e-12)
        + vertices[:, 0]
    )
    return bool(np.count_nonzero(crosses & (value[0] < x_intersection)) % 2)


def _orientation(
    first: npt.NDArray[np.float32],
    second: npt.NDArray[np.float32],
    third: npt.NDArray[np.float32],
) -> float:
    return _cross(first, second, third)


def _segments_intersect(
    a: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    c: npt.NDArray[np.float32],
    d: npt.NDArray[np.float32],
) -> bool:
    first = _orientation(a, b, c)
    second = _orientation(a, b, d)
    third = _orientation(c, d, a)
    fourth = _orientation(c, d, b)
    if first * second < 0 and third * fourth < 0:
        return True
    return any(
        distance <= 1e-6
        for distance in (
            _point_segment_distance(c, a, b),
            _point_segment_distance(d, a, b),
            _point_segment_distance(a, c, d),
            _point_segment_distance(b, c, d),
        )
    )


def polygons_intersect(first: npt.ArrayLike, second: npt.ArrayLike) -> bool:
    left = np.asarray(first, dtype=np.float32)
    right = np.asarray(second, dtype=np.float32)
    left_end = np.roll(left, -1, axis=0)
    right_end = np.roll(right, -1, axis=0)
    left_delta = left_end - left
    right_delta = right_end - right

    def cross(first_value: npt.NDArray[np.float32], second_value: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return first_value[..., 0] * second_value[..., 1] - first_value[..., 1] * second_value[..., 0]

    o1 = cross(left_delta[:, None, :], right[None, :, :] - left[:, None, :])
    o2 = cross(left_delta[:, None, :], right_end[None, :, :] - left[:, None, :])
    o3 = cross(right_delta[None, :, :], left[:, None, :] - right[None, :, :])
    o4 = cross(right_delta[None, :, :], left_end[:, None, :] - right[None, :, :])
    if np.any((o1 * o2 < 0.0) & (o3 * o4 < 0.0)):
        return True
    near = np.minimum.reduce(
        [
            _points_to_segments_distance(right, left, left_end).T,
            _points_to_segments_distance(right_end, left, left_end).T,
            _points_to_segments_distance(left, right, right_end),
            _points_to_segments_distance(left_end, right, right_end),
        ]
    )
    if np.any(near <= 1.0e-6):
        return True
    return point_in_polygon(left[0], right) or point_in_polygon(right[0], left)


def polygon_clearance(first: npt.ArrayLike, second: npt.ArrayLike) -> float:
    left = np.asarray(first, dtype=np.float32)
    right = np.asarray(second, dtype=np.float32)
    if polygons_intersect(left, right):
        return 0.0
    left_end = np.roll(left, -1, axis=0)
    right_end = np.roll(right, -1, axis=0)
    return float(
        min(
            _points_to_segments_distance(left, right, right_end).min(),
            _points_to_segments_distance(right, left, left_end).min(),
        )
    )


def _points_to_segment_batches(
    points: npt.NDArray[np.float32],
    starts: npt.NDArray[np.float32],
    ends: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Distances from shared points [P,2] to polygon batches [N,Q,2]."""

    delta = ends - starts
    denominator = np.sum(delta * delta, axis=-1)
    relative = points[None, :, None, :] - starts[:, None, :, :]
    numerator = np.sum(relative * delta[:, None, :, :], axis=-1)
    safe = np.where(denominator > 1.0e-12, denominator, 1.0)
    fraction = np.clip(numerator / safe[:, None, :], 0.0, 1.0)
    projection = starts[:, None, :, :] + fraction[..., None] * delta[:, None, :, :]
    distance = np.linalg.norm(points[None, :, None, :] - projection, axis=-1)
    degenerate = denominator <= 1.0e-12
    if np.any(degenerate):
        distance = np.where(
            degenerate[:, None, :],
            np.linalg.norm(relative, axis=-1),
            distance,
        )
    return np.asarray(distance, dtype=np.float32)


def _batch_ray_inside(
    points: npt.NDArray[np.float32], polygons: npt.NDArray[np.float32]
) -> npt.NDArray[np.bool_]:
    previous = np.roll(polygons, 1, axis=1)
    y = points[:, None, 1]
    crosses = (polygons[:, :, 1] > y) != (previous[:, :, 1] > y)
    x_intersection = (
        (previous[:, :, 0] - polygons[:, :, 0])
        * (y - polygons[:, :, 1])
        / (previous[:, :, 1] - polygons[:, :, 1] + 1.0e-12)
        + polygons[:, :, 0]
    )
    return np.asarray(
        np.count_nonzero(crosses & (points[:, None, 0] < x_intersection), axis=1)
        % 2
        == 1,
        dtype=bool,
    )


def _polygon_clearance_many(
    first: npt.ArrayLike, seconds: npt.ArrayLike
) -> npt.NDArray[np.float32]:
    """Exact polygon clearance from one convex polygon to a polygon batch."""

    left = np.asarray(first, dtype=np.float32)
    right = np.asarray(seconds, dtype=np.float32)
    if right.ndim != 3 or right.shape[-1] != 2:
        raise ValueError(f"seconds must be [N,Q,2], got {right.shape}")
    if len(right) == 0:
        return np.zeros(0, dtype=np.float32)
    left_end = np.roll(left, -1, axis=0)
    right_end = np.roll(right, -1, axis=1)
    left_delta = left_end - left
    right_delta = right_end - right

    def cross(
        first_value: npt.NDArray[np.float32],
        second_value: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        return first_value[..., 0] * second_value[..., 1] - first_value[..., 1] * second_value[..., 0]

    o1 = cross(
        left_delta[None, :, None, :],
        right[:, None, :, :] - left[None, :, None, :],
    )
    o2 = cross(
        left_delta[None, :, None, :],
        right_end[:, None, :, :] - left[None, :, None, :],
    )
    o3 = cross(
        right_delta[:, None, :, :],
        left[None, :, None, :] - right[:, None, :, :],
    )
    o4 = cross(
        right_delta[:, None, :, :],
        left_end[None, :, None, :] - right[:, None, :, :],
    )
    proper = (o1 * o2 < 0.0) & (o3 * o4 < 0.0)
    left_to_right = _points_to_segment_batches(left, right, right_end)
    left_end_to_right = _points_to_segment_batches(left_end, right, right_end)
    right_to_left = _points_to_segment_batches(right.reshape(-1, 2), left[None], left_end[None])
    right_to_left = right_to_left.reshape(len(right), right.shape[1], len(left))
    right_end_to_left = _points_to_segment_batches(
        right_end.reshape(-1, 2), left[None], left_end[None]
    ).reshape(len(right), right.shape[1], len(left))
    near = np.minimum.reduce(
        [
            left_to_right,
            left_end_to_right,
            np.swapaxes(right_to_left, 1, 2),
            np.swapaxes(right_end_to_left, 1, 2),
        ]
    ) <= 1.0e-6
    edge_intersection = np.any(proper | near, axis=(1, 2))
    contains = _batch_ray_inside(
        np.broadcast_to(left[0], (len(right), 2)), right
    ) | _batch_ray_inside(right[:, 0], np.broadcast_to(left, (len(right), *left.shape)))
    clearance = np.minimum(
        left_to_right.min(axis=(1, 2)), right_to_left.min(axis=(1, 2))
    )
    clearance[edge_intersection | contains] = 0.0
    return np.asarray(clearance, dtype=np.float32)


def _distance_to_polygon_boundary(point: npt.ArrayLike, polygon: npt.ArrayLike) -> float:
    value = np.asarray(point, dtype=np.float32)
    vertices = np.asarray(polygon, dtype=np.float32)
    return float(
        _points_to_segments_distance(
            value[None], vertices, np.roll(vertices, -1, axis=0)
        ).min()
    )


def _polygon_bounds(
    polygons: Sequence[npt.NDArray[np.float32]],
) -> npt.NDArray[np.float32]:
    return np.asarray(
        [
            [polygon[:, 0].min(), polygon[:, 1].min(), polygon[:, 0].max(), polygon[:, 1].max()]
            for polygon in polygons
        ],
        dtype=np.float32,
    )


def _inside_any(
    point: npt.ArrayLike,
    polygons: Sequence[npt.NDArray[np.float32]],
    bounds: npt.NDArray[np.float32] | None = None,
) -> bool:
    value = np.asarray(point, dtype=np.float32)
    polygon_bounds = _polygon_bounds(polygons) if bounds is None else bounds
    tolerance = 1.0e-6
    eligible = np.flatnonzero(
        (value[0] >= polygon_bounds[:, 0] - tolerance)
        & (value[0] <= polygon_bounds[:, 2] + tolerance)
        & (value[1] >= polygon_bounds[:, 1] - tolerance)
        & (value[1] <= polygon_bounds[:, 3] + tolerance)
    )
    return any(point_in_polygon(value, polygons[index]) for index in eligible)


def _boundary_distance_any(
    point: npt.ArrayLike,
    polygons: Sequence[npt.NDArray[np.float32]],
    bounds: npt.NDArray[np.float32] | None = None,
) -> float:
    if not polygons:
        raise EffectConstructionError("at least one drivable polygon is required")
    value = np.asarray(point, dtype=np.float32)
    polygon_bounds = _polygon_bounds(polygons) if bounds is None else bounds
    dx = np.maximum(
        np.maximum(polygon_bounds[:, 0] - value[0], value[0] - polygon_bounds[:, 2]),
        0.0,
    )
    dy = np.maximum(
        np.maximum(polygon_bounds[:, 1] - value[1], value[1] - polygon_bounds[:, 3]),
        0.0,
    )
    lower_bound = np.hypot(dx, dy)
    best = np.inf
    for index in np.argsort(lower_bound, kind="stable"):
        if lower_bound[index] > best:
            break
        best = min(best, _distance_to_polygon_boundary(value, polygons[index]))
    return float(best)


def _polygon_clearance_any(
    polygon: npt.NDArray[np.float32],
    others: Sequence[npt.NDArray[np.float32]],
    bounds: npt.NDArray[np.float32],
) -> float:
    own_min = polygon.min(axis=0)
    own_max = polygon.max(axis=0)
    dx = np.maximum(np.maximum(bounds[:, 0] - own_max[0], own_min[0] - bounds[:, 2]), 0.0)
    dy = np.maximum(np.maximum(bounds[:, 1] - own_max[1], own_min[1] - bounds[:, 3]), 0.0)
    lower_bound = np.hypot(dx, dy)
    best = np.inf
    for index in np.argsort(lower_bound, kind="stable"):
        if lower_bound[index] > best:
            break
        best = min(best, polygon_clearance(polygon, others[index]))
        if best == 0.0:
            break
    return float(best)


def _nearest_route(
    point: npt.NDArray[np.float32], route: npt.NDArray[np.float32]
) -> tuple[float, float, float]:
    segments = route[1:] - route[:-1]
    lengths = np.linalg.norm(segments, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    valid = lengths > 1.0e-6
    if not np.any(valid):
        raise EffectConstructionError("route centerline has no non-degenerate segment")
    valid_indices = np.flatnonzero(valid)
    starts = route[:-1][valid]
    deltas = segments[valid]
    valid_lengths = lengths[valid]
    fractions = np.clip(
        np.sum((point[None] - starts) * deltas, axis=-1)
        / (valid_lengths * valid_lengths),
        0.0,
        1.0,
    )
    projections = starts + fractions[:, None] * deltas
    residuals = point[None] - projections
    distances = np.linalg.norm(residuals, axis=-1)
    local_index = int(np.argmin(distances))
    index = int(valid_indices[local_index])
    delta = deltas[local_index]
    residual = residuals[local_index]
    distance = float(distances[local_index])
    signed_cross = float(delta[0] * residual[1] - delta[1] * residual[0])
    return (
        float(np.sign(signed_cross) * distance),
        float(cumulative[index] + fractions[local_index] * valid_lengths[local_index]),
        float(np.arctan2(delta[1], delta[0])),
    )


@dataclass(frozen=True)
class InteractionThresholds:
    clearance_m: float = 6.0
    tca_seconds: float = 3.0
    tca_distance_m: float = 10.0
    conflict_zone_clearance_m: float = 1.0

    def validate(self) -> None:
        values = (
            self.clearance_m,
            self.tca_seconds,
            self.tca_distance_m,
            self.conflict_zone_clearance_m,
        )
        if any(value <= 0 for value in values):
            raise ValueError("interaction thresholds must all be positive")


@dataclass(frozen=True)
class EffectBuilderConfig:
    horizon: int = 8
    interval_seconds: float = 0.5
    actor_slots: int = 16
    ego_length_m: float = 4.87
    ego_width_m: float = 2.27
    interaction: InteractionThresholds = field(default_factory=InteractionThresholds)

    def validate(self) -> None:
        if self.horizon != 8:
            raise ValueError(f"the fixed Gate horizon must be eight, got {self.horizon}")
        if self.interval_seconds <= 0 or self.actor_slots <= 0:
            raise ValueError("interval and actor slots must be positive")
        if self.ego_length_m <= 0 or self.ego_width_m <= 0:
            raise ValueError("ego dimensions must be positive")
        self.interaction.validate()


@dataclass(frozen=True)
class LoggedActorFutures:
    track_tokens: tuple[str, ...]
    positions: npt.NDArray[np.float32]
    headings: npt.NDArray[np.float32]
    velocities: npt.NDArray[np.float32]
    sizes: npt.NDArray[np.float32]
    valid: npt.NDArray[np.bool_]

    def validate(self, horizon: int) -> None:
        actor_count = len(self.track_tokens)
        expected = {
            "positions": (horizon, actor_count, 2),
            "headings": (horizon, actor_count),
            "velocities": (horizon, actor_count, 2),
            "sizes": (horizon, actor_count, 2),
            "valid": (horizon, actor_count),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise EffectConstructionError(
                    f"logged actor {name} expected {shape}, got {value.shape}"
                )
        if len(set(self.track_tokens)) != actor_count or any(not token for token in self.track_tokens):
            raise EffectConstructionError("actor track tokens must be unique and non-empty")
        for name in ("positions", "headings", "velocities", "sizes"):
            if not np.isfinite(getattr(self, name)).all():
                raise EffectConstructionError(f"logged actor {name} contains NaN/Inf")
        if actor_count and np.any(self.sizes[self.valid] <= 0):
            raise EffectConstructionError("valid logged actor sizes must be positive")


@dataclass(frozen=True)
class ReplaySceneContext:
    route_centerline: npt.NDArray[np.float32]
    drivable_polygons: tuple[npt.NDArray[np.float32], ...]
    static_obstacles: npt.NDArray[np.float32]
    logged_actors: LoggedActorFutures
    current_speed_mps: float = 0.0
    current_acceleration_mps2: float = 0.0
    traffic_light_states: tuple[tuple[str, bool], ...] = ()

    def validate(self, horizon: int) -> None:
        route = np.asarray(self.route_centerline)
        if route.ndim != 2 or route.shape[1] != 2 or len(route) < 2:
            raise EffectConstructionError(f"route centerline must be [N>=2,2], got {route.shape}")
        if not np.isfinite(route).all():
            raise EffectConstructionError("route centerline contains NaN/Inf")
        if not self.drivable_polygons:
            raise EffectConstructionError("drivable polygons cannot be empty")
        for polygon in self.drivable_polygons:
            if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
                raise EffectConstructionError(f"invalid drivable polygon {polygon.shape}")
            if not np.isfinite(polygon).all():
                raise EffectConstructionError("drivable polygon contains NaN/Inf")
        obstacles = np.asarray(self.static_obstacles)
        if obstacles.ndim != 2 or obstacles.shape[1] != 5:
            raise EffectConstructionError(
                f"static obstacles must be [N,5] x/y/heading/length/width, got {obstacles.shape}"
            )
        if not np.isfinite(obstacles).all() or (len(obstacles) and np.any(obstacles[:, 3:] <= 0)):
            raise EffectConstructionError("invalid static obstacle geometry")
        if not np.isfinite(self.current_speed_mps) or not np.isfinite(
            self.current_acceleration_mps2
        ):
            raise EffectConstructionError("current ego kinematics must be finite")
        self.logged_actors.validate(horizon)


@dataclass(frozen=True)
class ReplayEffectTensors:
    ego_effect: npt.NDArray[np.float32]
    map_effect: npt.NDArray[np.float32]
    actor_effect: npt.NDArray[np.float32]
    actor_mask: npt.NDArray[np.bool_]
    interaction_mask: npt.NDArray[np.bool_]
    selected_actor_indices: npt.NDArray[np.int64]
    selected_actor_tokens: tuple[str, ...]

    def as_tensor_dict(self) -> dict[str, npt.NDArray[Any]]:
        return {
            "ego_effect": self.ego_effect,
            "map_effect": self.map_effect,
            "actor_effect": self.actor_effect,
            "actor_mask": self.actor_mask,
            "interaction_mask": self.interaction_mask,
        }

    def as_primitive_dict(self) -> dict[str, npt.NDArray[Any]]:
        """Return only core primitives; evaluator-like proxies stay excluded."""

        return {
            "ego_effect": self.ego_effect,
            "map_effect": self.map_effect[..., PRIMITIVE_MAP_INDICES],
            "actor_effect": self.actor_effect[..., PRIMITIVE_ACTOR_INDICES],
            "actor_mask": self.actor_mask,
            "interaction_mask": self.interaction_mask,
        }

    def as_engineered_dict(self) -> dict[str, npt.NDArray[Any]]:
        """Return quarantined engineered diagnostics under their own schema."""

        return {
            "map_engineered_effect": self.map_effect[..., ENGINEERED_MAP_INDICES],
            "actor_engineered_effect": self.actor_effect[
                ..., ENGINEERED_ACTOR_INDICES
            ],
        }

    def flattened_primitive_groups(self) -> dict[str, npt.NDArray[np.float32]]:
        """Produce mutually controlled candidate inputs for matched probes."""

        primitive = self.as_primitive_dict()
        candidate_count = self.ego_effect.shape[0]

        def flatten(value: npt.ArrayLike) -> npt.NDArray[np.float32]:
            return np.asarray(value, dtype=np.float32).reshape(candidate_count, -1)

        ego = flatten(primitive["ego_effect"])
        map_effect = flatten(primitive["map_effect"])
        actor_effect = flatten(primitive["actor_effect"])
        actor_mask = flatten(primitive["actor_mask"])
        interaction = flatten(primitive["interaction_mask"])
        static = np.concatenate([ego, map_effect], axis=-1)
        dynamic = np.concatenate([actor_effect, actor_mask, interaction], axis=-1)
        return {
            "ego_effect": ego,
            "static_effect": static,
            "dynamic_effect": dynamic,
            "full_primitive_effect": np.concatenate([static, dynamic], axis=-1),
        }


def _logged_actor_hashes(actors: LoggedActorFutures) -> dict[str, str]:
    return {
        name: stable_array_hash(np.asarray(getattr(actors, name)))
        for name in ("positions", "headings", "velocities", "sizes", "valid")
    }


class ReplayGroundedEffectBuilder:
    """Build structured candidate effects against one immutable logged future."""

    def __init__(self, config: EffectBuilderConfig = EffectBuilderConfig()):
        config.validate()
        self.config = config

    def _select_actor_indices(self, actors: LoggedActorFutures) -> npt.NDArray[np.int64]:
        actor_count = len(actors.track_tokens)
        priorities: list[tuple[float, str, int]] = []
        for actor_index, token in enumerate(actors.track_tokens):
            valid_steps = np.flatnonzero(actors.valid[:, actor_index])
            if len(valid_steps) == 0:
                continue
            first_step = int(valid_steps[0])
            distance = float(np.linalg.norm(actors.positions[first_step, actor_index]))
            priorities.append((distance, token, actor_index))
        priorities.sort()
        return np.asarray(
            [item[2] for item in priorities[: self.config.actor_slots]], dtype=np.int64
        )

    def build(
        self,
        candidates: npt.ArrayLike,
        context: ReplaySceneContext,
    ) -> ReplayEffectTensors:
        trajectories = _as_float32(candidates)
        if trajectories.ndim != 3 or trajectories.shape[1:] != (
            self.config.horizon,
            3,
        ):
            raise EffectConstructionError(
                f"candidates must be [K,{self.config.horizon},3], got {trajectories.shape}"
            )
        if len(trajectories) == 0 or not np.isfinite(trajectories).all():
            raise EffectConstructionError("candidates must be finite and non-empty")
        context.validate(self.config.horizon)
        before_hashes = _logged_actor_hashes(context.logged_actors)

        candidate_count = len(trajectories)
        horizon = self.config.horizon
        slots = self.config.actor_slots
        ego_effect = np.zeros((candidate_count, horizon, len(EGO_EFFECT_NAMES)), dtype=np.float32)
        map_effect = np.zeros((candidate_count, horizon, len(MAP_EFFECT_NAMES)), dtype=np.float32)
        actor_effect = np.zeros(
            (candidate_count, horizon, slots, len(ACTOR_EFFECT_NAMES)), dtype=np.float32
        )
        actor_mask = np.zeros((candidate_count, horizon, slots), dtype=bool)
        interaction_mask = np.zeros_like(actor_mask)

        selected = self._select_actor_indices(context.logged_actors)
        selected_tokens = tuple(context.logged_actors.track_tokens[index] for index in selected)
        selected_valid = context.logged_actors.valid[:, selected]
        selected_actor_boxes = np.zeros(
            (horizon, len(selected), 4, 2), dtype=np.float32
        )
        for time_index in range(horizon):
            for slot_index, actor_index in enumerate(selected):
                if selected_valid[time_index, slot_index]:
                    actor_length, actor_width = context.logged_actors.sizes[
                        time_index, actor_index
                    ]
                    selected_actor_boxes[time_index, slot_index] = oriented_box_corners(
                        context.logged_actors.positions[time_index, actor_index],
                        float(context.logged_actors.headings[time_index, actor_index]),
                        float(actor_length),
                        float(actor_width),
                    )
        previous_positions = np.concatenate(
            [np.zeros((candidate_count, 1, 2), dtype=np.float32), trajectories[:, :-1, :2]],
            axis=1,
        )
        previous_headings = np.concatenate(
            [np.zeros((candidate_count, 1), dtype=np.float32), trajectories[:, :-1, 2]],
            axis=1,
        )
        segment_distance = np.linalg.norm(trajectories[:, :, :2] - previous_positions, axis=-1)
        speed = segment_distance / self.config.interval_seconds
        previous_speed = np.concatenate(
            [
                np.full((candidate_count, 1), context.current_speed_mps, dtype=np.float32),
                speed[:, :-1],
            ],
            axis=1,
        )
        acceleration = (speed - previous_speed) / self.config.interval_seconds
        previous_acceleration = np.concatenate(
            [
                np.full(
                    (candidate_count, 1),
                    context.current_acceleration_mps2,
                    dtype=np.float32,
                ),
                acceleration[:, :-1],
            ],
            axis=1,
        )
        jerk = (acceleration - previous_acceleration) / self.config.interval_seconds
        yaw_rate = wrap_angle(trajectories[:, :, 2] - previous_headings) / self.config.interval_seconds
        curvature = yaw_rate / np.maximum(speed, 0.1)

        static_polygons = tuple(
            oriented_box_corners(obstacle[:2], obstacle[2], obstacle[3], obstacle[4])
            for obstacle in context.static_obstacles
        )
        drivable_bounds = _polygon_bounds(context.drivable_polygons)
        static_bounds = (
            _polygon_bounds(static_polygons)
            if static_polygons
            else np.zeros((0, 4), dtype=np.float32)
        )
        _, route_origin_progress, _ = _nearest_route(
            np.zeros(2, dtype=np.float32), context.route_centerline
        )

        for candidate_index in range(candidate_count):
            for time_index in range(horizon):
                position = trajectories[candidate_index, time_index, :2]
                heading = float(trajectories[candidate_index, time_index, 2])
                previous_position = previous_positions[candidate_index, time_index]
                previous_heading = float(previous_headings[candidate_index, time_index])
                current_box = oriented_box_corners(
                    position, heading, self.config.ego_length_m, self.config.ego_width_m
                )
                previous_box = oriented_box_corners(
                    previous_position,
                    previous_heading,
                    self.config.ego_length_m,
                    self.config.ego_width_m,
                )
                swept_polygon = convex_hull(np.concatenate([previous_box, current_box], axis=0))
                local_sweep = _rotate(swept_polygon - position, -heading)
                sweep_min = local_sweep.min(axis=0)
                sweep_max = local_sweep.max(axis=0)
                swept_rectangle_local = np.array(
                    [
                        [sweep_max[0], sweep_max[1]],
                        [sweep_max[0], sweep_min[1]],
                        [sweep_min[0], sweep_min[1]],
                        [sweep_min[0], sweep_max[1]],
                    ],
                    dtype=np.float32,
                )
                swept_rectangle = _rotate(swept_rectangle_local, heading) + position
                ego_effect[candidate_index, time_index] = np.concatenate(
                    [
                        trajectories[candidate_index, time_index],
                        np.array(
                            [
                                speed[candidate_index, time_index],
                                acceleration[candidate_index, time_index],
                                yaw_rate[candidate_index, time_index],
                                curvature[candidate_index, time_index],
                                jerk[candidate_index, time_index],
                            ],
                            dtype=np.float32,
                        ),
                        swept_rectangle.reshape(-1),
                    ]
                )

                signed_route_distance, route_progress, route_heading = _nearest_route(
                    position, context.route_centerline
                )
                normal = np.array([-np.sin(heading), np.cos(heading)], dtype=np.float32)
                left_point = position + normal * (self.config.ego_width_m / 2.0)
                right_point = position - normal * (self.config.ego_width_m / 2.0)
                left_clearance = _boundary_distance_any(
                    left_point, context.drivable_polygons, drivable_bounds
                )
                right_clearance = _boundary_distance_any(
                    right_point, context.drivable_polygons, drivable_bounds
                )
                if not _inside_any(left_point, context.drivable_polygons, drivable_bounds):
                    left_clearance *= -1.0
                if not _inside_any(right_point, context.drivable_polygons, drivable_bounds):
                    right_clearance *= -1.0
                footprint_points = np.concatenate(
                    [
                        current_box,
                        (current_box + np.roll(current_box, -1, axis=0)) / 2.0,
                        position[None],
                    ],
                    axis=0,
                )
                outside_ratio = np.mean(
                    [
                        not _inside_any(
                            point, context.drivable_polygons, drivable_bounds
                        )
                        for point in footprint_points
                    ]
                )
                static_clearance = (
                    _polygon_clearance_any(current_box, static_polygons, static_bounds)
                    if static_polygons
                    else 100.0
                )
                map_effect[candidate_index, time_index] = np.array(
                    [
                        signed_route_distance,
                        route_progress - route_origin_progress,
                        float(wrap_angle(heading - route_heading)),
                        left_clearance,
                        right_clearance,
                        outside_ratio,
                        static_clearance,
                        _boundary_distance_any(
                            position, context.drivable_polygons, drivable_bounds
                        ),
                    ],
                    dtype=np.float32,
                )

                ego_velocity = speed[candidate_index, time_index] * np.array(
                    [np.cos(heading), np.sin(heading)], dtype=np.float32
                )
                valid_slots = np.flatnonzero(selected_valid[time_index])
                clearance_by_slot = np.zeros(len(selected), dtype=np.float32)
                swept_distance_by_slot = np.zeros(len(selected), dtype=np.float32)
                if len(valid_slots):
                    valid_actor_boxes = selected_actor_boxes[time_index, valid_slots]
                    clearance_by_slot[valid_slots] = _polygon_clearance_many(
                        current_box, valid_actor_boxes
                    )
                    swept_distance_by_slot[valid_slots] = _polygon_clearance_many(
                        swept_polygon, valid_actor_boxes
                    )
                for slot_index, actor_index in enumerate(selected):
                    if not context.logged_actors.valid[time_index, actor_index]:
                        continue
                    actor_mask[candidate_index, time_index, slot_index] = True
                    actor_position = context.logged_actors.positions[time_index, actor_index]
                    actor_heading = float(context.logged_actors.headings[time_index, actor_index])
                    actor_velocity = context.logged_actors.velocities[time_index, actor_index]
                    actor_length, actor_width = context.logged_actors.sizes[
                        time_index, actor_index
                    ]
                    relative_position = _rotate(actor_position - position, -heading)
                    relative_velocity = _rotate(actor_velocity - ego_velocity, -heading)
                    clearance = float(clearance_by_slot[slot_index])
                    center_distance = float(np.linalg.norm(relative_position))
                    relative_speed_squared = float(relative_velocity @ relative_velocity)
                    if relative_speed_squared > 1e-8:
                        tca = float(
                            np.clip(
                                -(relative_position @ relative_velocity)
                                / relative_speed_squared,
                                0.0,
                                horizon * self.config.interval_seconds,
                            )
                        )
                    else:
                        tca = float(horizon * self.config.interval_seconds)
                    distance_at_tca = float(
                        np.linalg.norm(relative_position + tca * relative_velocity)
                    )
                    swept_distance = float(swept_distance_by_slot[slot_index])
                    actor_effect[candidate_index, time_index, slot_index] = np.array(
                        [
                            relative_position[0],
                            relative_position[1],
                            relative_velocity[0],
                            relative_velocity[1],
                            float(wrap_angle(actor_heading - heading)),
                            clearance,
                            center_distance,
                            np.sign(relative_position[0]),
                            np.sign(relative_position[1]),
                            tca,
                            distance_at_tca,
                            swept_distance,
                            np.sqrt(relative_speed_squared),
                        ],
                        dtype=np.float32,
                    )
                    thresholds = self.config.interaction
                    interaction_mask[candidate_index, time_index, slot_index] = bool(
                        clearance < thresholds.clearance_m
                        or swept_distance < thresholds.conflict_zone_clearance_m
                        or (
                            tca < thresholds.tca_seconds
                            and distance_at_tca < thresholds.tca_distance_m
                        )
                    )

        after_hashes = _logged_actor_hashes(context.logged_actors)
        if before_hashes != after_hashes:
            raise EffectConstructionError("logged actor future was mutated during replay")
        expected_shapes = {
            "ego_effect": (candidate_count, horizon, len(EGO_EFFECT_NAMES)),
            "map_effect": (candidate_count, horizon, len(MAP_EFFECT_NAMES)),
            "actor_effect": (candidate_count, horizon, slots, len(ACTOR_EFFECT_NAMES)),
            "actor_mask": (candidate_count, horizon, slots),
            "interaction_mask": (candidate_count, horizon, slots),
        }
        values: Mapping[str, npt.NDArray[Any]] = {
            "ego_effect": ego_effect,
            "map_effect": map_effect,
            "actor_effect": actor_effect,
            "actor_mask": actor_mask,
            "interaction_mask": interaction_mask,
        }
        for name, shape in expected_shapes.items():
            if values[name].shape != shape:
                raise AssertionError(f"{name} shape {values[name].shape} != {shape}")
        for name in ("ego_effect", "map_effect", "actor_effect"):
            if not np.isfinite(values[name]).all():
                raise EffectConstructionError(f"{name} contains NaN/Inf")
        if np.any(interaction_mask & ~actor_mask):
            raise EffectConstructionError("interaction mask marks a padded actor")
        return ReplayEffectTensors(
            ego_effect=ego_effect,
            map_effect=map_effect,
            actor_effect=actor_effect,
            actor_mask=actor_mask,
            interaction_mask=interaction_mask,
            selected_actor_indices=selected,
            selected_actor_tokens=selected_tokens,
        )


def _global_to_local(
    points: npt.ArrayLike, origin: npt.ArrayLike
) -> npt.NDArray[np.float32]:
    pose = np.asarray(origin, dtype=np.float32)
    return _rotate(np.asarray(points, dtype=np.float32) - pose[:2], -float(pose[2]))


def context_from_navsim_scene(scene: Any, metric_cache: Any) -> ReplaySceneContext:
    """Adapt official NAVSIM scene/cache geometry without using metric labels."""

    horizon = 8
    current_index = scene.scene_metadata.num_history_frames - 1
    current_frame = scene.frames[current_index]
    current_pose = np.asarray(current_frame.ego_status.ego_pose, dtype=np.float32)
    route_global = np.asarray(
        [[state.x, state.y] for state in metric_cache.centerline.discrete_path],
        dtype=np.float32,
    )
    route_local = _global_to_local(route_global, current_pose)

    polygons: list[npt.NDArray[np.float32]] = []
    for geometry in metric_cache.drivable_area_map._geometries:
        if geometry.is_empty:
            continue
        components = list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]
        for component in components:
            coordinates = np.asarray(component.exterior.coords[:-1], dtype=np.float32)
            if len(coordinates) >= 3:
                polygons.append(_global_to_local(coordinates, current_pose))
    if not polygons:
        raise EffectConstructionError("metric cache has no drivable polygons")

    static_names = {"barrier", "traffic_cone", "generic_object", "czone_sign"}
    static_obstacles: list[list[float]] = []
    for name, box in zip(current_frame.annotations.names, current_frame.annotations.boxes):
        if name in static_names:
            static_obstacles.append(
                [float(box[0]), float(box[1]), float(box[6]), float(box[3]), float(box[4])]
            )

    dynamic_names = {
        "vehicle",
        "pedestrian",
        "bicycle",
    }
    token_union: set[str] = set()
    future_frames = scene.frames[current_index + 1 : current_index + 1 + horizon]
    if len(future_frames) != horizon:
        raise EffectConstructionError(
            f"NAVSIM scene has {len(future_frames)} future frames, expected {horizon}"
        )
    for frame in future_frames:
        for name, token in zip(frame.annotations.names, frame.annotations.track_tokens):
            if name in dynamic_names:
                token_union.add(token)
    tokens = tuple(sorted(token_union))
    token_to_index = {token: index for index, token in enumerate(tokens)}
    actor_count = len(tokens)
    positions = np.zeros((horizon, actor_count, 2), dtype=np.float32)
    headings = np.zeros((horizon, actor_count), dtype=np.float32)
    velocities = np.zeros((horizon, actor_count, 2), dtype=np.float32)
    sizes = np.ones((horizon, actor_count, 2), dtype=np.float32)
    valid = np.zeros((horizon, actor_count), dtype=bool)
    for time_index, frame in enumerate(future_frames):
        future_pose = np.asarray(frame.ego_status.ego_pose, dtype=np.float32)
        relative_rotation = float(future_pose[2] - current_pose[2])
        for name, token, box, velocity in zip(
            frame.annotations.names,
            frame.annotations.track_tokens,
            frame.annotations.boxes,
            frame.annotations.velocity_3d,
        ):
            if name not in dynamic_names or token not in token_to_index:
                continue
            actor_index = token_to_index[token]
            actor_global = _rotate(np.asarray(box[:2], dtype=np.float32), float(future_pose[2])) + future_pose[:2]
            positions[time_index, actor_index] = _global_to_local(
                actor_global[None], current_pose
            )[0]
            headings[time_index, actor_index] = float(
                wrap_angle(float(box[6]) + relative_rotation)
            )
            velocities[time_index, actor_index] = _rotate(
                np.asarray(velocity[:2], dtype=np.float32), relative_rotation
            )
            sizes[time_index, actor_index] = np.asarray(box[3:5], dtype=np.float32)
            valid[time_index, actor_index] = True
    logged = LoggedActorFutures(
        track_tokens=tokens,
        positions=positions,
        headings=headings,
        velocities=velocities,
        sizes=sizes,
        valid=valid,
    )
    return ReplaySceneContext(
        route_centerline=route_local,
        drivable_polygons=tuple(polygons),
        static_obstacles=np.asarray(static_obstacles, dtype=np.float32).reshape(-1, 5),
        logged_actors=logged,
        current_speed_mps=float(np.linalg.norm(current_frame.ego_status.ego_velocity[:2])),
        current_acceleration_mps2=float(
            np.linalg.norm(current_frame.ego_status.ego_acceleration[:2])
        ),
        traffic_light_states=tuple(tuple(item) for item in current_frame.traffic_lights),
    )
