#!/usr/bin/env python3
"""Create a deterministic, smooth fallback trajectory candidate bank."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .common import (
    add_common_arguments,
    append_command,
    effective_max_scenes,
    ensure_output_dir,
    load_scenes_for_tokens,
    metric_cache_loader,
    paths_from_args,
    scene_loader,
    wrap_angle,
    write_markdown,
    write_parquet,
)


TARGET_TIMES = np.arange(0.5, 4.0 + 1e-8, 0.5, dtype=np.float64)


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    family: str
    parameters: dict[str, float]


def smoothstep5(u: np.ndarray) -> np.ndarray:
    u = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0)
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def recompute_heading(xy: np.ndarray, start_heading: float = 0.0) -> np.ndarray:
    points = np.vstack([np.zeros((1, 2), dtype=np.float64), np.asarray(xy, dtype=np.float64)])
    delta = np.diff(points, axis=0)
    heading = np.empty(len(xy), dtype=np.float64)
    valid = np.linalg.norm(delta, axis=1) > 1e-5
    last = float(start_heading)
    for index, vector in enumerate(delta):
        if valid[index]:
            last = float(np.arctan2(vector[1], vector[0]))
        heading[index] = last
    return np.asarray(wrap_angle(np.unwrap(heading)), dtype=np.float64)


def resample_gt(scene: Any, target_times: np.ndarray = TARGET_TIMES) -> np.ndarray:
    current = scene.scene_metadata.num_history_frames - 1
    timestamps = np.asarray([frame.timestamp for frame in scene.frames], dtype=np.float64)
    gt = np.asarray(scene.get_future_trajectory().poses, dtype=np.float64)
    # Select the measured frame nearest each horizon.  This preserves the logged
    # waypoints exactly while avoiding a hard-coded "horizon * 2" array rule.
    from .common import resolve_horizon_index

    indices = [resolve_horizon_index(timestamps, float(t), origin_index=current) for t in target_times]
    future_offsets = [index - current - 1 for index in indices]
    if min(future_offsets) < 0 or max(future_offsets) >= len(gt) or len(set(future_offsets)) != len(future_offsets):
        raise RuntimeError(f"Cannot resolve distinct logged GT horizons: {future_offsets}")
    return gt[future_offsets].copy()


def _sample_path(base: np.ndarray, normalized_time: np.ndarray) -> np.ndarray:
    source_u = np.linspace(0.0, 1.0, len(base) + 1)
    points = np.vstack([np.zeros((1, 2)), base[:, :2]])
    result = np.column_stack(
        [
            np.interp(normalized_time, source_u, points[:, 0]),
            np.interp(normalized_time, source_u, points[:, 1]),
        ]
    )
    return result


def lateral_candidate(base: np.ndarray, amplitude: float, profile: str = "ramp") -> np.ndarray:
    u = np.linspace(1.0 / len(base), 1.0, len(base))
    if profile == "ramp":
        offset = amplitude * smoothstep5(u)
    elif profile == "middle":
        offset = amplitude * np.sin(np.pi * u) ** 2
    elif profile == "tail":
        offset = amplitude * smoothstep5(np.clip((u - 0.5) / 0.5, 0.0, 1.0))
    elif profile == "early_return":
        offset = amplitude * np.sin(np.pi * u)
    else:
        raise ValueError(profile)
    heading = recompute_heading(base[:, :2], 0.0)
    normal = np.column_stack([-np.sin(heading), np.cos(heading)])
    xy = base[:, :2] + offset[:, None] * normal
    perturbed_heading = recompute_heading(xy, 0.0)
    if profile == "tail":
        # Preserve the factual pose prefix exactly; only poses after the tail
        # branch starts receive headings derived from perturbed geometry.
        perturbed_heading[u <= 0.5] = base[u <= 0.5, 2]
    return np.column_stack([xy, perturbed_heading])


def time_scaled_candidate(base: np.ndarray, scale: float) -> np.ndarray:
    u = np.linspace(1.0 / len(base), 1.0, len(base))
    warped = np.clip(u * scale, 0.0, 1.0)
    xy = _sample_path(base, warped)
    return np.column_stack([xy, recompute_heading(xy, 0.0)])


def progress_warp_candidate(base: np.ndarray, strength: float) -> np.ndarray:
    u = np.linspace(1.0 / len(base), 1.0, len(base))
    warped = np.clip(u + strength * u * (1.0 - u), 0.0, 1.0)
    xy = _sample_path(base, warped)
    return np.column_stack([xy, recompute_heading(xy, 0.0)])


def candidate_specs() -> list[CandidateSpec]:
    return [
        CandidateSpec("gt", "gt", {}),
        CandidateSpec("smooth_left", "lateral", {"amplitude_m": 2.0}),
        CandidateSpec("smooth_right", "lateral", {"amplitude_m": -2.0}),
        CandidateSpec("slight_left", "lateral", {"amplitude_m": 0.75}),
        CandidateSpec("slight_right", "lateral", {"amplitude_m": -0.75}),
        CandidateSpec("slow_time_scale", "speed", {"scale": 0.72}),
        CandidateSpec("mild_deceleration", "speed", {"strength": -0.22}),
        CandidateSpec("mild_acceleration", "speed", {"strength": 0.18}),
        CandidateSpec("fast_time_scale", "speed", {"scale": 1.18}),
        CandidateSpec("same_endpoint_mid_curve", "curvature", {"amplitude_m": 1.4}),
        CandidateSpec("same_prefix_different_tail", "tail", {"amplitude_m": -2.4}),
        CandidateSpec("different_prefix_similar_endpoint", "curvature", {"amplitude_m": -1.4}),
    ]


def generate_candidates(base: np.ndarray, num_candidates: int = 12, seed: int = 20260828) -> tuple[np.ndarray, list[CandidateSpec]]:
    if num_candidates < 1:
        raise ValueError("num_candidates must be positive")
    base = np.asarray(base, dtype=np.float64)
    if base.shape != (8, 3) or not np.isfinite(base).all():
        raise ValueError(f"Expected finite (8, 3) GT poses, got {base.shape}")
    specs = candidate_specs()
    generated: list[np.ndarray] = []
    used_specs: list[CandidateSpec] = []
    rng = np.random.default_rng(seed)
    variant = 0
    spec_cursor = 0
    while len(generated) < num_candidates:
        spec = specs[spec_cursor] if spec_cursor < len(specs) else CandidateSpec(
            f"smooth_variant_{variant}",
            "lateral",
            {"amplitude_m": float(rng.choice([-1, 1]) * (1.0 + 0.2 * variant))},
        )
        spec_cursor += 1
        if spec.name == "gt":
            poses = base.copy()
        elif spec.name in {"smooth_left", "smooth_right", "slight_left", "slight_right"}:
            poses = lateral_candidate(base, spec.parameters["amplitude_m"], "ramp")
        elif spec.name in {"slow_time_scale", "fast_time_scale"}:
            poses = time_scaled_candidate(base, spec.parameters["scale"])
        elif spec.name in {"mild_deceleration", "mild_acceleration"}:
            poses = progress_warp_candidate(base, spec.parameters["strength"])
        elif spec.name == "same_endpoint_mid_curve":
            poses = lateral_candidate(base, spec.parameters["amplitude_m"], "middle")
        elif spec.name == "same_prefix_different_tail":
            poses = lateral_candidate(base, spec.parameters["amplitude_m"], "tail")
        elif spec.name == "different_prefix_similar_endpoint":
            poses = lateral_candidate(base, spec.parameters["amplitude_m"], "early_return")
        else:
            poses = lateral_candidate(base, spec.parameters["amplitude_m"], "ramp")
            variant += 1
        if spec.name not in {"gt", "same_prefix_different_tail"}:
            poses[:, 2] = recompute_heading(poses[:, :2], 0.0)
        if not np.isfinite(poses).all():
            raise RuntimeError(f"Non-finite candidate {spec.name}")
        signature = np.round(poses, decimals=5).tobytes()
        if any(signature == np.round(other, decimals=5).tobytes() for other in generated):
            # A stopped/near-stopped GT makes time warps identical.  Keep the
            # semantic candidate type but add a deterministic, smooth forward
            # nudge instead of accepting duplicates or injecting waypoint noise.
            u = np.linspace(1.0 / len(base), 1.0, len(base))
            nudge = (0.15 + 0.05 * spec_cursor) * smoothstep5(u)
            heading = recompute_heading(base[:, :2], 0.0)
            poses[:, 0] += nudge * np.cos(heading)
            poses[:, 1] += nudge * np.sin(heading)
            poses[:, 2] = recompute_heading(poses[:, :2], 0.0)
            signature = np.round(poses, decimals=5).tobytes()
            if any(signature == np.round(other, decimals=5).tobytes() for other in generated):
                variant += 1
                continue
        generated.append(poses.astype(np.float32))
        used_specs.append(spec)
        variant += int(spec_cursor > len(specs))
    result = np.stack(generated, axis=0)
    np.testing.assert_array_equal(result[0], base.astype(np.float32))
    return result, used_specs


def kinematic_summary(poses: np.ndarray, dt: float = 0.5) -> dict[str, float]:
    xy = np.vstack([np.zeros((1, 2)), poses[:, :2]])
    heading = np.unwrap(np.concatenate([[0.0], poses[:, 2]]))
    velocity = np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt
    acceleration = np.diff(np.concatenate([[0.0], velocity])) / dt
    curvature = np.divide(
        np.diff(heading),
        np.maximum(np.linalg.norm(np.diff(xy, axis=0), axis=1), 1e-3),
    )
    return {
        "max_speed_mps": float(np.max(np.abs(velocity))),
        "max_acceleration_mps2": float(np.max(np.abs(acceleration))),
        "max_abs_curvature_1pm": float(np.max(np.abs(curvature))),
        "terminal_displacement_m": float(np.linalg.norm(poses[-1, :2])),
    }


def build_manifest(args: argparse.Namespace) -> pd.DataFrame:
    paths = paths_from_args(args)
    output_dir = ensure_output_dir(args.output_dir)
    max_scenes = effective_max_scenes(args.mode, args.max_scenes)
    cache_loader = metric_cache_loader(paths)
    cache_tokens = set(cache_loader.tokens)
    loader = scene_loader(paths, max_scenes=max_scenes, frame_interval=1, tokens=cache_loader.tokens)
    rows: list[dict[str, Any]] = []
    for scene_index, token in enumerate(loader.tokens):
        scene = loader.get_scene_from_token(token)
        base = resample_gt(scene)
        candidates, specs = generate_candidates(base, args.num_candidates, args.seed + scene_index)
        for candidate_index, (poses, spec) in enumerate(zip(candidates, specs)):
            summary = kinematic_summary(poses)
            rows.append(
                {
                    "scene_token": token,
                    "scene_metadata_token": scene.scene_metadata.scene_token,
                    "log_name": scene.scene_metadata.log_name,
                    "map_name": scene.scene_metadata.map_name,
                    "candidate_index": candidate_index,
                    "candidate_type": spec.name,
                    "candidate_family": spec.family,
                    "candidate_parameters": str(spec.parameters),
                    "candidate_source": "logged_gt" if candidate_index == 0 else "deterministic_smooth_fallback",
                    "is_gt": candidate_index == 0,
                    "time_s": TARGET_TIMES.astype(np.float32).tolist(),
                    "pose_x_m": poses[:, 0].tolist(),
                    "pose_y_m": poses[:, 1].tolist(),
                    "pose_heading_rad": poses[:, 2].tolist(),
                    "implicit_start_x_m": 0.0,
                    "implicit_start_y_m": 0.0,
                    "implicit_start_heading_rad": 0.0,
                    "metric_cache_available": token in cache_tokens,
                    "seed": args.seed + scene_index,
                    **summary,
                }
            )
    frame = pd.DataFrame(rows).sort_values(["scene_token", "candidate_index"]).reset_index(drop=True)
    write_parquet(frame, output_dir / "candidate_manifest.parquet")
    duplicate_count = 0
    for _, group in frame.groupby("scene_token"):
        signatures = {
            tuple(np.round(np.column_stack([row.pose_x_m, row.pose_y_m, row.pose_heading_rad]), 5).ravel())
            for row in group.itertuples()
        }
        duplicate_count += len(group) - len(signatures)
    report = f"""# Candidate Bank Audit

- Source: deterministic smooth fallback; no existing multi-trajectory inference dump was found in the repository environment scan.
- Scenes: {frame['scene_token'].nunique()}
- Candidates per scene: {args.num_candidates}
- GT is candidate index 0 and is byte-identical to the resampled logged trajectory.
- Implicit trajectory start: `(0 m, 0 m, 0 rad)` for every candidate.
- Duplicate candidates after 1e-5 rounding: {duplicate_count}
- Non-finite rows: {int((~frame[['max_speed_mps', 'max_acceleration_mps2', 'max_abs_curvature_1pm']].apply(np.isfinite)).any(axis=1).sum())}

Non-GT rows are controlled trajectory perturbations, not real futures or real counterfactual futures.
"""
    write_markdown(output_dir / "CANDIDATE_GENERATION_AUDIT.md", report)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, include_candidates=True)
    args = parser.parse_args()
    build_manifest(args)
    append_command(args.output_dir.resolve(), "python -m tools.navsim_candidate_relative_audit.candidate_generator " + " ".join(sys.argv[1:]))


if __name__ == "__main__":
    main()
