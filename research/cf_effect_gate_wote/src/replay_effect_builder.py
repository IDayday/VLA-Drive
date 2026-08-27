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


def point_in_polygon(point: npt.ArrayLike, polygon: npt.ArrayLike) -> bool:
    value = np.asarray(point, dtype=np.float32)
    vertices = np.asarray(polygon, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
        raise ValueError(f"polygon must be [N>=3,2], got {vertices.shape}")
    inside = False
    previous = vertices[-1]
    for current in vertices:
        if _point_segment_distance(value, previous, current) <= 1e-6:
            return True
        crosses = (current[1] > value[1]) != (previous[1] > value[1])
        if crosses:
            x_intersection = (
                (previous[0] - current[0])
                * (value[1] - current[1])
                / (previous[1] - current[1] + 1e-12)
                + current[0]
            )
            if value[0] < x_intersection:
                inside = not inside
        previous = current
    return inside


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
    for left_index in range(len(left)):
        for right_index in range(len(right)):
            if _segments_intersect(
                left[left_index],
                left[(left_index + 1) % len(left)],
                right[right_index],
                right[(right_index + 1) % len(right)],
            ):
                return True
    return point_in_polygon(left[0], right) or point_in_polygon(right[0], left)


def polygon_clearance(first: npt.ArrayLike, second: npt.ArrayLike) -> float:
    left = np.asarray(first, dtype=np.float32)
    right = np.asarray(second, dtype=np.float32)
    if polygons_intersect(left, right):
        return 0.0
    distances: list[float] = []
    for point in left:
        for index in range(len(right)):
            distances.append(
                _point_segment_distance(point, right[index], right[(index + 1) % len(right)])
            )
    for point in right:
        for index in range(len(left)):
            distances.append(
                _point_segment_distance(point, left[index], left[(index + 1) % len(left)])
            )
    return min(distances)


def _distance_to_polygon_boundary(point: npt.ArrayLike, polygon: npt.ArrayLike) -> float:
    value = np.asarray(point, dtype=np.float32)
    vertices = np.asarray(polygon, dtype=np.float32)
    return min(
        _point_segment_distance(value, vertices[index], vertices[(index + 1) % len(vertices)])
        for index in range(len(vertices))
    )


def _inside_any(point: npt.ArrayLike, polygons: Sequence[npt.NDArray[np.float32]]) -> bool:
    return any(point_in_polygon(point, polygon) for polygon in polygons)


def _boundary_distance_any(
    point: npt.ArrayLike, polygons: Sequence[npt.NDArray[np.float32]]
) -> float:
    if not polygons:
        raise EffectConstructionError("at least one drivable polygon is required")
    return min(_distance_to_polygon_boundary(point, polygon) for polygon in polygons)


def _nearest_route(
    point: npt.NDArray[np.float32], route: npt.NDArray[np.float32]
) -> tuple[float, float, float]:
    segments = route[1:] - route[:-1]
    lengths = np.linalg.norm(segments, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    best_distance = np.inf
    best_progress = 0.0
    best_heading = 0.0
    best_signed = 0.0
    for index, (start, delta, length) in enumerate(zip(route[:-1], segments, lengths)):
        if length <= 1e-6:
            continue
        fraction = float(np.clip(((point - start) @ delta) / (length * length), 0.0, 1.0))
        projection = start + fraction * delta
        residual = point - projection
        distance = float(np.linalg.norm(residual))
        if distance < best_distance:
            best_distance = distance
            best_progress = float(cumulative[index] + fraction * length)
            best_heading = float(np.arctan2(delta[1], delta[0]))
            best_signed = float(np.sign(np.cross(delta, residual)) * distance)
    if not np.isfinite(best_distance):
        raise EffectConstructionError("route centerline has no non-degenerate segment")
    return best_signed, best_progress, best_heading


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
                left_clearance = _boundary_distance_any(left_point, context.drivable_polygons)
                right_clearance = _boundary_distance_any(right_point, context.drivable_polygons)
                if not _inside_any(left_point, context.drivable_polygons):
                    left_clearance *= -1.0
                if not _inside_any(right_point, context.drivable_polygons):
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
                    [not _inside_any(point, context.drivable_polygons) for point in footprint_points]
                )
                static_clearance = (
                    min(polygon_clearance(current_box, obstacle) for obstacle in static_polygons)
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
                        _boundary_distance_any(position, context.drivable_polygons),
                    ],
                    dtype=np.float32,
                )

                ego_velocity = speed[candidate_index, time_index] * np.array(
                    [np.cos(heading), np.sin(heading)], dtype=np.float32
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
                    actor_box = oriented_box_corners(
                        actor_position, actor_heading, float(actor_length), float(actor_width)
                    )
                    clearance = polygon_clearance(current_box, actor_box)
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
                    swept_distance = polygon_clearance(swept_polygon, actor_box)
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
