#!/usr/bin/env python3
"""Build randomized, smooth controlled candidates with no index-template leak."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from tools.navsim_candidate_relative_audit.candidate_generator import (
    TARGET_TIMES,
    kinematic_summary,
    recompute_heading,
    smoothstep5,
)
from tools.navsim_candidate_relative_audit.common import load_scenes_for_tokens

from .common import (
    DEFAULT_CACHE_DIR,
    DEFAULT_REPORT_DIR,
    append_command,
    ensure_dir,
    navsim_paths,
    stable_scene_seed,
    write_json,
    write_markdown,
    write_parquet,
)


@dataclass(frozen=True)
class RandomCandidateSpec:
    family: str
    candidate_type: str
    parameters: dict[str, float]


def resample_gt_by_timestamp(scene, target_times: np.ndarray = TARGET_TIMES) -> np.ndarray:
    """Interpolate official future poses at requested seconds using measured timestamps.

    Some deployed trainval logs are sampled more coarsely than the six logs in
    the original audit. Nearest-frame indexing can therefore map two requested
    horizons to one frame. Position interpolation plus unwrapped-angle
    interpolation is the exact coordinate/time derivation needed by the 2 Hz
    planner output interface.
    """

    current = int(scene.scene_metadata.num_history_frames) - 1
    timestamps = np.asarray([frame.timestamp for frame in scene.frames], dtype=np.float64)
    scale = 1e6 if np.nanmax(np.abs(timestamps)) > 1e9 else 1.0
    relative_time = (timestamps[current + 1 :] - timestamps[current]) / scale
    future = np.asarray(scene.get_future_trajectory().poses, dtype=np.float64)
    count = min(len(relative_time), len(future))
    relative_time = relative_time[:count]
    future = future[:count]
    if count < 2 or relative_time[-1] + 1e-6 < float(np.max(target_times)):
        raise RuntimeError(
            f"Logged future horizon {relative_time[-1] if count else 0:.3f}s is shorter "
            f"than requested {float(np.max(target_times)):.3f}s"
        )
    if np.any(np.diff(relative_time) <= 0):
        raise RuntimeError("Future timestamps are not strictly increasing")
    heading = np.unwrap(future[:, 2])
    result = np.column_stack(
        [
            np.interp(target_times, relative_time, future[:, 0]),
            np.interp(target_times, relative_time, future[:, 1]),
            np.interp(target_times, relative_time, heading),
        ]
    )
    result[:, 2] = (result[:, 2] + np.pi) % (2 * np.pi) - np.pi
    return result


def _path_sample(base: np.ndarray, progress: np.ndarray) -> np.ndarray:
    source = np.linspace(0.0, 1.0, len(base) + 1)
    points = np.vstack([np.zeros((1, 2), dtype=np.float64), base[:, :2]])
    xy = np.column_stack(
        [
            np.interp(progress, source, points[:, 0]),
            np.interp(progress, source, points[:, 1]),
        ]
    )
    return np.column_stack([xy, recompute_heading(xy, 0.0)])


def _lateral(
    base: np.ndarray,
    amplitude: float,
    start: float,
    shape: str,
) -> np.ndarray:
    u = np.linspace(1.0 / len(base), 1.0, len(base))
    phase = np.clip((u - start) / max(1.0 - start, 1e-6), 0.0, 1.0)
    if shape == "offset":
        offset = amplitude * smoothstep5(phase)
    elif shape == "tail":
        offset = amplitude * smoothstep5(phase)
    elif shape == "return":
        offset = amplitude * np.sin(np.pi * phase) * (phase > 0)
    elif shape == "mid_curve":
        offset = amplitude * np.sin(np.pi * phase) ** 2 * (phase > 0)
    else:
        raise ValueError(shape)
    base_heading = recompute_heading(base[:, :2], 0.0)
    normal = np.column_stack([-np.sin(base_heading), np.cos(base_heading)])
    xy = base[:, :2] + offset[:, None] * normal
    result = np.column_stack([xy, recompute_heading(xy, 0.0)])
    # Same-prefix candidates are bit-identical before the randomized branch.
    prefix = u <= start
    result[prefix] = base[prefix]
    return result


def _speed_warp(base: np.ndarray, factor: float, start: float) -> np.ndarray:
    u = np.linspace(1.0 / len(base), 1.0, len(base))
    phase = np.clip((u - start) / max(1.0 - start, 1e-6), 0.0, 1.0)
    blend = smoothstep5(phase)
    target = np.clip(start + (u - start) * factor, 0.0, 1.0)
    warped = np.where(u <= start, u, (1.0 - blend) * u + blend * target)
    warped = np.maximum.accumulate(warped)
    return _path_sample(base, warped)


def _progress_bulge(base: np.ndarray, strength: float, start: float) -> np.ndarray:
    u = np.linspace(1.0 / len(base), 1.0, len(base))
    phase = np.clip((u - start) / max(1.0 - start, 1e-6), 0.0, 1.0)
    warped = u + strength * smoothstep5(phase) * phase * (1.0 - phase)
    warped = np.maximum.accumulate(np.clip(warped, 0.0, 1.0))
    return _path_sample(base, warped)


def _random_spec(family: str, rng: np.random.Generator) -> RandomCandidateSpec:
    if family == "lateral_offset":
        sign = float(rng.choice([-1.0, 1.0]))
        amplitude = sign * float(rng.uniform(0.45, 2.6))
        return RandomCandidateSpec(
            family, "left_offset" if sign > 0 else "right_offset",
            {"amplitude_m": amplitude, "start_fraction": float(rng.uniform(0.0, 0.42)), "shape_id": 0.0},
        )
    if family == "speed_change":
        slower = bool(rng.integers(0, 2))
        factor = float(rng.uniform(0.58, 0.92) if slower else rng.uniform(1.08, 1.32))
        return RandomCandidateSpec(
            family, "slow" if slower else "fast",
            {"factor": factor, "start_fraction": float(rng.uniform(0.0, 0.5))},
        )
    if family == "brake_timing":
        return RandomCandidateSpec(
            family, "brake",
            {"factor": float(rng.uniform(0.35, 0.78)), "start_fraction": float(rng.uniform(0.12, 0.72))},
        )
    if family == "same_prefix_different_tail":
        amplitude = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.8, 2.8))
        return RandomCandidateSpec(
            family, "tail_left" if amplitude > 0 else "tail_right",
            {"amplitude_m": amplitude, "start_fraction": float(rng.uniform(0.34, 0.72)), "shape_id": 1.0},
        )
    if family == "same_endpoint_mid_curve":
        amplitude = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.65, 2.2))
        return RandomCandidateSpec(
            family, "mid_curve",
            {"amplitude_m": amplitude, "start_fraction": float(rng.uniform(0.0, 0.28)), "shape_id": 2.0},
        )
    if family == "different_prefix_similar_endpoint":
        amplitude = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.65, 2.4))
        return RandomCandidateSpec(
            family, "early_return",
            {"amplitude_m": amplitude, "start_fraction": float(rng.uniform(0.0, 0.2)), "shape_id": 3.0},
        )
    if family == "progress_shape":
        return RandomCandidateSpec(
            family, "progress_bulge",
            {"strength": float(rng.uniform(-0.45, 0.45)), "start_fraction": float(rng.uniform(0.0, 0.45))},
        )
    raise ValueError(family)


def _materialize(base: np.ndarray, spec: RandomCandidateSpec) -> np.ndarray:
    p = spec.parameters
    if spec.family == "lateral_offset":
        return _lateral(base, p["amplitude_m"], p["start_fraction"], "offset")
    if spec.family == "speed_change" or spec.family == "brake_timing":
        return _speed_warp(base, p["factor"], p["start_fraction"])
    if spec.family == "same_prefix_different_tail":
        return _lateral(base, p["amplitude_m"], p["start_fraction"], "tail")
    if spec.family == "same_endpoint_mid_curve":
        return _lateral(base, p["amplitude_m"], p["start_fraction"], "mid_curve")
    if spec.family == "different_prefix_similar_endpoint":
        return _lateral(base, p["amplitude_m"], p["start_fraction"], "return")
    if spec.family == "progress_shape":
        return _progress_bulge(base, p["strength"], p["start_fraction"])
    raise ValueError(spec.family)


def generate_randomized_candidates(
    base: np.ndarray,
    num_candidates: int = 16,
    seed: int = 20260828,
) -> tuple[np.ndarray, list[RandomCandidateSpec]]:
    base = np.asarray(base, dtype=np.float64)
    if base.shape != (8, 3) or not np.isfinite(base).all():
        raise ValueError(f"Expected finite (8, 3) GT, got {base.shape}")
    if num_candidates < 2:
        raise ValueError("At least GT and one non-GT candidate are required")
    rng = np.random.default_rng(seed)
    families = [
        "lateral_offset",
        "speed_change",
        "brake_timing",
        "same_prefix_different_tail",
        "same_endpoint_mid_curve",
        "different_prefix_similar_endpoint",
        "progress_shape",
    ]
    specs = [RandomCandidateSpec("gt", "gt", {})]
    candidates = [base.copy()]
    cursor = 0
    retries = 0
    while len(candidates) < num_candidates:
        family = families[cursor % len(families)]
        cursor += 1
        spec = _random_spec(family, rng)
        candidate = _materialize(base, spec)
        if not np.isfinite(candidate).all():
            raise RuntimeError(f"Non-finite randomized candidate: {spec}")
        if np.linalg.norm(candidate[0, :2]) > max(np.linalg.norm(base[0, :2]) + 1e-5, 20.0):
            raise RuntimeError("Candidate start exploded")
        duplicate = any(np.allclose(candidate, other, atol=1e-5, rtol=0.0) for other in candidates)
        if duplicate:
            retries += 1
            if retries > 100:
                raise RuntimeError("Could not create unique smooth candidates")
            continue
        candidates.append(candidate)
        specs.append(spec)
    permutation = rng.permutation(num_candidates)
    shuffled = np.stack(candidates, axis=0)[permutation].astype(np.float32)
    shuffled_specs = [specs[index] for index in permutation]
    gt_indices = [index for index, spec in enumerate(shuffled_specs) if spec.family == "gt"]
    if len(gt_indices) != 1:
        raise AssertionError(gt_indices)
    np.testing.assert_array_equal(shuffled[gt_indices[0]], base.astype(np.float32))
    signatures = {np.round(item, 5).tobytes() for item in shuffled}
    if len(signatures) != num_candidates:
        raise AssertionError("Candidate deduplication failed")
    return shuffled, shuffled_specs


def build(args: argparse.Namespace) -> dict[str, object]:
    report_dir = ensure_dir(args.output_dir)
    cache_dir = ensure_dir(args.cache_dir)
    scene_manifest_path = report_dir / "balanced_scene_manifest.parquet"
    if not scene_manifest_path.is_file():
        raise FileNotFoundError(scene_manifest_path)
    selected = pd.read_parquet(scene_manifest_path)
    if args.num_scenes > 0:
        selected = selected.head(args.num_scenes).copy()
    paths = navsim_paths(args.split)
    loader = load_scenes_for_tokens(paths, selected.scene_token.tolist())
    rows: list[dict[str, object]] = []
    gt_positions: list[int] = []
    for scene_row in selected.itertuples(index=False):
        scene = loader.get_scene_from_token(scene_row.scene_token)
        base = resample_gt_by_timestamp(scene)
        subseed = stable_scene_seed(scene_row.scene_token, args.seed)
        candidates, specs = generate_randomized_candidates(base, args.num_candidates, subseed)
        for index, (poses, spec) in enumerate(zip(candidates, specs)):
            summary = kinematic_summary(poses)
            is_gt = spec.family == "gt"
            if is_gt:
                gt_positions.append(index)
            rows.append(
                {
                    "scene_token": scene_row.scene_token,
                    "scene_metadata_token": scene.scene_metadata.scene_token,
                    "log_name": scene.scene_metadata.log_name,
                    "map_name": scene.scene_metadata.map_name,
                    "fold": int(scene_row.fold),
                    "candidate_index": index,
                    "candidate_type": spec.candidate_type,
                    "candidate_family": spec.family,
                    "candidate_parameters": json.dumps(spec.parameters, sort_keys=True),
                    "candidate_source": "randomized_smooth_controlled",
                    "is_gt": is_gt,
                    "time_s": TARGET_TIMES.astype(np.float32).tolist(),
                    "pose_x_m": poses[:, 0].tolist(),
                    "pose_y_m": poses[:, 1].tolist(),
                    "pose_heading_rad": poses[:, 2].tolist(),
                    "implicit_start_x_m": 0.0,
                    "implicit_start_y_m": 0.0,
                    "implicit_start_heading_rad": 0.0,
                    "metric_cache_available": True,
                    "seed": int(subseed),
                    **summary,
                }
            )
    manifest = pd.DataFrame(rows)
    output_path = cache_dir / "controlled_candidate_manifest.parquet"
    write_parquet(manifest, output_path)
    family_counts = manifest.candidate_family.value_counts().to_dict()
    result = {
        "source": "randomized smooth controlled candidates",
        "scene_count": int(manifest.scene_token.nunique()),
        "candidate_count": int(len(manifest)),
        "candidates_per_scene": args.num_candidates,
        "log_count": int(manifest.log_name.nunique()),
        "unique_candidate_rate": 1.0,
        "gt_position_unique_count": len(set(gt_positions)),
        "gt_position_histogram": {str(key): gt_positions.count(key) for key in sorted(set(gt_positions))},
        "family_counts": family_counts,
        "global_seed": args.seed,
        "manifest_path": str(output_path),
    }
    write_json(report_dir / "candidates/controlled_candidate_summary.json", result)
    write_markdown(
        report_dir / "candidates/CONTROLLED_CANDIDATE_AUDIT.md",
        f"""# Randomized Controlled Candidate Audit

- Scenes/logs: {result['scene_count']} / {result['log_count']}
- Candidates per scene: {args.num_candidates}
- Candidate rows: {len(manifest)}
- GT appeared in {result['gt_position_unique_count']} different candidate indices
- Global seed: {args.seed}; every scene has a stable SHA256-derived subseed
- Manifest cache: `{output_path}`

All non-GT candidates use continuous scene-specific parameters, smooth temporal
profiles and deterministic de-duplication. Candidate order is independently
shuffled per scene. Candidate index therefore does not identify a fixed behavior.
These candidates are controlled perturbations, not true futures.
""",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "trainval"), default="trainval")
    parser.add_argument("--num-scenes", type=int, default=0)
    parser.add_argument("--num-candidates", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    args = parser.parse_args()
    build(args)
    append_command(args.output_dir, "python -m tools.shared_future_candidate_consequence.build_controlled_candidates " + " ".join(sys.argv[1:]))


if __name__ == "__main__":
    main()
