"""Configuration-driven, kinematically filtered policy-local trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .trajectory_io import NAVSIM_INTERVAL_LENGTH, NAVSIM_NUM_FUTURE_POSES, wrap_to_pi


def _smoothstep(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def _headings_from_positions(
    positions: np.ndarray,
    initial_heading: float = 0.0,
    minimum_motion_m: float = 0.05,
) -> np.ndarray:
    points = np.concatenate([np.zeros((1, 2), dtype=np.float64), positions], axis=0)
    deltas = np.diff(points, axis=0)
    headings = np.empty(len(positions), dtype=np.float64)
    previous = float(initial_heading)
    for index, delta in enumerate(deltas):
        if np.linalg.norm(delta) > minimum_motion_m:
            measured = float(np.arctan2(delta[1], delta[0]))
            previous = previous + float(wrap_to_pi(measured - previous))
        headings[index] = previous
    return wrap_to_pi(np.unwrap(headings))


def _polyline_progress(trajectory: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.concatenate([np.zeros((1, 2), dtype=np.float64), trajectory[:, :2]], axis=0)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    progress = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    return points, progress


def _sample_polyline(points: np.ndarray, progress: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Sample a polyline by arc length, linearly extrapolating its terminal ray."""

    query = np.maximum(np.asarray(query, dtype=np.float64), 0.0)
    result = np.empty((len(query), 2), dtype=np.float64)
    unique_progress, unique_indices = np.unique(progress, return_index=True)
    unique_points = points[unique_indices]
    if len(unique_progress) == 1:
        return np.repeat(unique_points[:1], len(query), axis=0)
    for axis in range(2):
        result[:, axis] = np.interp(query, unique_progress, unique_points[:, axis])
    beyond = query > unique_progress[-1]
    if np.any(beyond):
        terminal_delta = unique_points[-1] - unique_points[-2]
        norm = np.linalg.norm(terminal_delta)
        direction = terminal_delta / norm if norm > 1e-6 else np.array([1.0, 0.0])
        result[beyond] = unique_points[-1] + (query[beyond] - unique_progress[-1])[:, None] * direction
    return result


def _sample_headings(trajectory: np.ndarray, progress: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Interpolate continuous anchor heading by arc length, including the current pose."""

    headings = np.unwrap(np.concatenate([[0.0], trajectory[:, 2]]))
    unique_progress, unique_indices = np.unique(progress, return_index=True)
    unique_headings = headings[unique_indices]
    if len(unique_progress) == 1:
        return np.full(len(query), wrap_to_pi(unique_headings[0]), dtype=np.float64)
    sampled = np.interp(np.asarray(query), unique_progress, unique_headings)
    sampled[np.asarray(query) > unique_progress[-1]] = unique_headings[-1]
    return wrap_to_pi(sampled)


def _point_to_polyline_distances(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    if len(polyline) < 2:
        return np.linalg.norm(points - polyline[0], axis=1)
    starts = polyline[:-1]
    segments = polyline[1:] - starts
    denom = np.sum(segments * segments, axis=1)
    distances = []
    for point in points:
        numer = np.sum((point - starts) * segments, axis=1)
        interpolation = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 1e-12)
        interpolation = np.clip(interpolation, 0.0, 1.0)
        projections = starts + interpolation[:, None] * segments
        distances.append(float(np.min(np.linalg.norm(point - projections, axis=1))))
    return np.asarray(distances)


@dataclass(frozen=True)
class KinematicLimits:
    """Physical and local-route filters applied after candidate projection."""

    max_speed_mps: float = 25.0
    max_acceleration_mps2: float = 5.0
    max_deceleration_mps2: float = 8.0
    max_jerk_mps3: float = 12.0
    max_curvature_inv_m: float = 0.35
    max_yaw_step_rad: float = 0.55
    max_step_distance_m: float = 12.5
    route_corridor_m: float = 1.5
    route_terminal_extension_m: float = 5.0
    min_terminal_forward_m: float = -0.25


@dataclass(frozen=True)
class CandidateGeneratorConfig:
    """All perturbation values; no trajectory magnitude is hard-coded in logic."""

    interval_length: float = NAVSIM_INTERVAL_LENGTH
    lateral_offsets_m: tuple[float, ...] = (-0.6, -0.3, 0.3, 0.6)
    speed_scales: tuple[float, ...] = (0.8, 0.9, 1.1)
    brake_onset_shifts_s: tuple[float, ...] = (-0.5, 0.5)
    brake_nominal_onset_s: float = 2.0
    brake_terminal_speed_scale: float = 0.45
    terminal_progress_shifts_m: tuple[float, ...] = (-1.0, 1.0)
    curvature_scales: tuple[float, ...] = (0.9, 1.1)
    turn_offsets_m: tuple[float, ...] = (-0.3, 0.3)
    limits: KinematicLimits = field(default_factory=KinematicLimits)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CandidateGeneratorConfig":
        """Create a typed config from the ``candidate_generation`` YAML node."""

        data = dict(value)
        limits = KinematicLimits(**data.pop("limits", {}))
        for name in (
            "lateral_offsets_m",
            "speed_scales",
            "brake_onset_shifts_s",
            "terminal_progress_shifts_m",
            "curvature_scales",
            "turn_offsets_m",
        ):
            if name in data:
                data[name] = tuple(float(item) for item in data[name])
        return cls(limits=limits, **data)

    def candidate_count(self) -> int:
        """Return the configured number, including the unperturbed anchor."""

        return 1 + sum(
            len(values)
            for values in (
                self.lateral_offsets_m,
                self.speed_scales,
                self.brake_onset_shifts_s,
                self.terminal_progress_shifts_m,
                self.curvature_scales,
                self.turn_offsets_m,
            )
        )


@dataclass(frozen=True)
class CandidateValidation:
    """Post-projection validity flags and diagnostic extrema."""

    kinematic_valid: bool
    route_valid: bool
    reasons: tuple[str, ...]
    max_speed_mps: float
    max_acceleration_mps2: float
    max_deceleration_mps2: float
    max_abs_jerk_mps3: float
    max_abs_curvature_inv_m: float
    max_abs_yaw_step_rad: float
    max_step_distance_m: float
    max_anchor_route_deviation_m: float


@dataclass(frozen=True)
class PolicyLocalCandidate:
    """One interpretable trajectory candidate and its filter outcome."""

    candidate_id: str
    trajectory: np.ndarray
    perturbation_type: str
    perturbation_parameters: Mapping[str, float]
    validation: CandidateValidation


class PolicyLocalCandidateGenerator:
    """Generate 8--16 smooth local alternatives around one physical anchor."""

    def __init__(self, config: CandidateGeneratorConfig):
        if config.interval_length <= 0:
            raise ValueError("interval_length must be positive")
        count = config.candidate_count()
        if not 8 <= count <= 16:
            raise ValueError(f"candidate count must be in [8,16], got {count}")
        self.config = config

    def _candidate_id(
        self,
        scene_id: str,
        anchor_type: str,
        perturbation_type: str,
        parameters: Mapping[str, float],
        seed: int,
    ) -> str:
        payload = json.dumps(
            [scene_id, anchor_type, perturbation_type, dict(parameters), int(seed)],
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

    def _trajectory_from_progress(self, anchor: np.ndarray, query_progress: np.ndarray) -> np.ndarray:
        points, progress = _polyline_progress(anchor)
        positions = _sample_polyline(points, progress, np.maximum.accumulate(query_progress))
        headings = _sample_headings(anchor, progress, np.maximum.accumulate(query_progress))
        return np.concatenate([positions, headings[:, None]], axis=1)

    def _lateral_offset(self, anchor: np.ndarray, offset: float, *, turn_weighted: bool) -> np.ndarray:
        u = np.linspace(1.0 / len(anchor), 1.0, len(anchor))
        headings = np.unwrap(anchor[:, 2])
        normals = np.stack([-np.sin(headings), np.cos(headings)], axis=1)
        if turn_weighted:
            curvature_activity = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(headings)))])
            if curvature_activity[-1] > 1e-5:
                activity = curvature_activity / curvature_activity[-1]
            else:
                activity = u
            profile = _smoothstep(u) * (0.35 + 0.65 * activity)
        else:
            profile = _smoothstep(u)
        positions = anchor[:, :2] + float(offset) * profile[:, None] * normals
        return np.concatenate([positions, _headings_from_positions(positions)[:, None]], axis=1)

    def _speed_scale(self, anchor: np.ndarray, scale: float) -> np.ndarray:
        _, progress = _polyline_progress(anchor)
        return self._trajectory_from_progress(anchor, progress[1:] * float(scale))

    def _brake(self, anchor: np.ndarray, onset_shift: float) -> np.ndarray:
        _, progress = _polyline_progress(anchor)
        base_speed = np.diff(progress) / self.config.interval_length
        times = np.arange(1, len(anchor) + 1, dtype=np.float64) * self.config.interval_length
        onset = self.config.brake_nominal_onset_s + float(onset_shift)
        denominator = max(times[-1] - onset, self.config.interval_length)
        fraction = np.clip((times - onset) / denominator, 0.0, 1.0)
        factor = 1.0 - fraction * (1.0 - self.config.brake_terminal_speed_scale)
        query = np.cumsum(base_speed * factor * self.config.interval_length)
        return self._trajectory_from_progress(anchor, query)

    def _terminal_progress(self, anchor: np.ndarray, shift: float) -> np.ndarray:
        _, progress = _polyline_progress(anchor)
        u = np.linspace(1.0 / len(anchor), 1.0, len(anchor))
        query = progress[1:] + float(shift) * _smoothstep(u)
        query = np.maximum.accumulate(np.maximum(query, 0.0))
        return self._trajectory_from_progress(anchor, query)

    def _curvature_scale(self, anchor: np.ndarray, scale: float) -> np.ndarray:
        points, _ = _polyline_progress(anchor)
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        base_headings = np.unwrap(anchor[:, 2])
        heading_steps = np.diff(np.concatenate([[0.0], base_headings])) * float(scale)
        segment_headings = np.cumsum(heading_steps)
        positions = np.cumsum(
            np.stack([np.cos(segment_headings), np.sin(segment_headings)], axis=1) * lengths[:, None],
            axis=0,
        )
        return np.concatenate([positions, wrap_to_pi(segment_headings)[:, None]], axis=1)

    def _project_kinematics(self, trajectory: np.ndarray) -> np.ndarray:
        """Project scalar speed changes onto configured acceleration/jerk bounds."""

        trajectory = np.asarray(trajectory, dtype=np.float64).copy()
        points, progress = _polyline_progress(trajectory)
        dt = self.config.interval_length
        speed = np.diff(progress) / dt
        limits = self.config.limits
        speed = np.clip(speed, 0.0, limits.max_speed_mps)
        for index in range(1, len(speed)):
            lower = max(0.0, speed[index - 1] - limits.max_deceleration_mps2 * dt)
            upper = speed[index - 1] + limits.max_acceleration_mps2 * dt
            speed[index] = np.clip(speed[index], lower, upper)
        for index in range(2, len(speed)):
            previous_acceleration = (speed[index - 1] - speed[index - 2]) / dt
            acceleration = (speed[index] - speed[index - 1]) / dt
            max_delta = limits.max_jerk_mps3 * dt
            acceleration = np.clip(acceleration, previous_acceleration - max_delta, previous_acceleration + max_delta)
            speed[index] = max(0.0, speed[index - 1] + acceleration * dt)
        projected_progress = np.cumsum(speed * dt)
        positions = _sample_polyline(points, progress, projected_progress)
        headings = _sample_headings(trajectory, progress, projected_progress)
        return np.concatenate([positions, headings[:, None]], axis=1)

    def validate(self, trajectory: np.ndarray, anchor: np.ndarray) -> CandidateValidation:
        """Check motion, yaw, jumps, and a conservative anchor-route proxy."""

        trajectory = np.asarray(trajectory, dtype=np.float64)
        reasons: list[str] = []
        if trajectory.shape != (NAVSIM_NUM_FUTURE_POSES, 3):
            raise ValueError(f"trajectory must be [8,3], got {trajectory.shape}")
        if not np.isfinite(trajectory).all():
            reasons.append("non_finite")
        dt = self.config.interval_length
        points = np.concatenate([np.zeros((1, 2)), trajectory[:, :2]], axis=0)
        steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        speed = steps / dt
        acceleration = np.diff(speed) / dt
        jerk = np.diff(acceleration) / dt
        yaw = np.unwrap(np.concatenate([[0.0], trajectory[:, 2]]))
        yaw_steps = np.diff(yaw)
        # Heading at standstill is not geometrically identifiable. Do not turn
        # centimetre-scale localization jitter into enormous curvature.
        curvature = np.divide(
            np.abs(yaw_steps),
            steps,
            out=np.zeros_like(steps),
            where=steps >= 0.05,
        )
        limits = self.config.limits
        anchor_polyline = np.concatenate([np.zeros((1, 2)), anchor[:, :2]], axis=0)
        # Faster candidates legitimately travel beyond the finite anchor endpoint.
        # Extend only its final *moving* tangent for the cheap route proxy; the
        # official map-based route metrics remain the authority in Phase 2.
        moving_segments = np.diff(anchor_polyline, axis=0)
        moving_indices = np.flatnonzero(np.linalg.norm(moving_segments, axis=1) > 0.05)
        if len(moving_indices) and limits.route_terminal_extension_m > 0:
            terminal_direction = moving_segments[moving_indices[-1]]
            terminal_direction /= np.linalg.norm(terminal_direction)
            anchor_polyline = np.concatenate(
                [
                    anchor_polyline,
                    (
                        anchor_polyline[-1]
                        + terminal_direction * limits.route_terminal_extension_m
                    )[None],
                ],
                axis=0,
            )
        route_distances = _point_to_polyline_distances(trajectory[:, :2], anchor_polyline)

        maxima = {
            "max_speed": float(np.max(speed, initial=0.0)),
            "max_acceleration": float(np.max(acceleration, initial=0.0)),
            "max_deceleration": float(max(0.0, -np.min(acceleration, initial=0.0))),
            "max_jerk": float(np.max(np.abs(jerk), initial=0.0)),
            "max_curvature": float(np.max(curvature, initial=0.0)),
            "max_yaw_step": float(np.max(np.abs(yaw_steps), initial=0.0)),
            "max_step": float(np.max(steps, initial=0.0)),
            "max_route_deviation": float(np.max(route_distances, initial=0.0)),
        }
        checks = (
            (maxima["max_speed"] <= limits.max_speed_mps + 1e-6, "max_speed"),
            (maxima["max_acceleration"] <= limits.max_acceleration_mps2 + 1e-6, "max_acceleration"),
            (maxima["max_deceleration"] <= limits.max_deceleration_mps2 + 1e-6, "max_deceleration"),
            (maxima["max_jerk"] <= limits.max_jerk_mps3 + 1e-6, "max_jerk"),
            (maxima["max_curvature"] <= limits.max_curvature_inv_m + 1e-6, "max_curvature"),
            (maxima["max_yaw_step"] <= limits.max_yaw_step_rad + 1e-6, "yaw_discontinuity"),
            (maxima["max_step"] <= limits.max_step_distance_m + 1e-6, "position_jump"),
        )
        reasons.extend(name for passed, name in checks if not passed)
        kinematic_valid = not reasons

        anchor_terminal = anchor[-1, :2]
        terminal_norm = np.linalg.norm(anchor_terminal)
        forward = anchor_terminal / terminal_norm if terminal_norm > 1e-5 else np.array([1.0, 0.0])
        terminal_forward = float(np.dot(trajectory[-1, :2], forward))
        route_valid = bool(
            maxima["max_route_deviation"] <= limits.route_corridor_m + 1e-6
            and terminal_forward >= limits.min_terminal_forward_m
        )
        if maxima["max_route_deviation"] > limits.route_corridor_m + 1e-6:
            reasons.append("anchor_route_corridor")
        if terminal_forward < limits.min_terminal_forward_m:
            reasons.append("terminal_reversal")
        return CandidateValidation(
            kinematic_valid=kinematic_valid,
            route_valid=route_valid,
            reasons=tuple(reasons),
            max_speed_mps=maxima["max_speed"],
            max_acceleration_mps2=maxima["max_acceleration"],
            max_deceleration_mps2=maxima["max_deceleration"],
            max_abs_jerk_mps3=maxima["max_jerk"],
            max_abs_curvature_inv_m=maxima["max_curvature"],
            max_abs_yaw_step_rad=maxima["max_yaw_step"],
            max_step_distance_m=maxima["max_step"],
            max_anchor_route_deviation_m=maxima["max_route_deviation"],
        )

    def generate(
        self,
        anchor: np.ndarray,
        *,
        scene_id: str,
        anchor_type: str,
        seed: int,
    ) -> list[PolicyLocalCandidate]:
        """Generate deterministic candidates around one expert or policy anchor."""

        anchor = np.asarray(anchor, dtype=np.float64).copy()
        if anchor.shape != (NAVSIM_NUM_FUTURE_POSES, 3):
            raise ValueError(f"anchor must be [8,3], got {anchor.shape}")
        anchor[:, 2] = wrap_to_pi(anchor[:, 2])
        specifications: list[tuple[str, dict[str, float], np.ndarray]] = [("anchor", {}, anchor.copy())]
        specifications.extend(
            ("lateral_terminal_offset", {"offset_m": float(value)}, self._lateral_offset(anchor, value, turn_weighted=False))
            for value in self.config.lateral_offsets_m
        )
        specifications.extend(
            ("speed_scale", {"scale": float(value)}, self._speed_scale(anchor, value))
            for value in self.config.speed_scales
        )
        specifications.extend(
            (
                "brake_onset_shift",
                {"shift_s": float(value), "nominal_onset_s": self.config.brake_nominal_onset_s},
                self._brake(anchor, value),
            )
            for value in self.config.brake_onset_shifts_s
        )
        specifications.extend(
            ("terminal_progress_shift", {"shift_m": float(value)}, self._terminal_progress(anchor, value))
            for value in self.config.terminal_progress_shifts_m
        )
        specifications.extend(
            ("curvature_scale", {"scale": float(value)}, self._curvature_scale(anchor, value))
            for value in self.config.curvature_scales
        )
        specifications.extend(
            ("turn_inner_outer_offset", {"offset_m": float(value)}, self._lateral_offset(anchor, value, turn_weighted=True))
            for value in self.config.turn_offsets_m
        )

        candidates: list[PolicyLocalCandidate] = []
        for perturbation_type, parameters, raw_trajectory in specifications:
            trajectory = raw_trajectory if perturbation_type == "anchor" else self._project_kinematics(raw_trajectory)
            validation = self.validate(trajectory, anchor)
            candidate_id = self._candidate_id(scene_id, anchor_type, perturbation_type, parameters, seed)
            candidates.append(
                PolicyLocalCandidate(
                    candidate_id=candidate_id,
                    trajectory=trajectory.astype(np.float32),
                    perturbation_type=perturbation_type,
                    perturbation_parameters=parameters,
                    validation=validation,
                )
            )
        return candidates
