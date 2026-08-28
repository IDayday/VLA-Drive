#!/usr/bin/env python3
"""Phase 3: adapt an existing candidate bank or generate deterministic fallbacks."""

from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .common import (
    add_common_arguments,
    bootstrap_navsim,
    consequence_rows_by_scene,
    discover_paths,
    load_candidate_source,
    load_metric_cache,
    load_metric_cache_index,
    output_tokens,
    select_eligible_scenes,
    trajectory_kinematics,
    wrap_heading,
    write_dataframe,
    write_json,
    write_text,
)


def quintic_smoothstep(values: np.ndarray) -> np.ndarray:
    u = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def headings_from_positions(
    positions: np.ndarray, minimum_motion_m: float = 0.03
) -> np.ndarray:
    points = np.concatenate(
        [np.zeros((1, 2), dtype=np.float64), np.asarray(positions, dtype=np.float64)]
    )
    deltas = np.diff(points, axis=0)
    headings = np.zeros(len(positions), dtype=np.float64)
    previous = 0.0
    for index, delta in enumerate(deltas):
        if np.linalg.norm(delta) >= minimum_motion_m:
            measured = float(np.arctan2(delta[1], delta[0]))
            previous += float(wrap_heading(measured - previous))
        headings[index] = previous
    return wrap_heading(headings)


def polyline(anchor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.concatenate(
        [
            np.zeros((1, 2), dtype=np.float64),
            np.asarray(anchor, dtype=np.float64)[:, :2],
        ]
    )
    progress = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    )
    return points, progress


def resample_progress(anchor: np.ndarray, query: np.ndarray) -> np.ndarray:
    points, progress = polyline(anchor)
    unique_progress, unique_index = np.unique(progress, return_index=True)
    unique_points = points[unique_index]
    query = np.maximum.accumulate(np.maximum(np.asarray(query, dtype=np.float64), 0.0))
    result = np.stack(
        [
            np.interp(query, unique_progress, unique_points[:, axis])
            for axis in range(2)
        ],
        axis=1,
    )
    beyond = query > unique_progress[-1]
    if np.any(beyond) and len(unique_points) > 1:
        direction = unique_points[-1] - unique_points[-2]
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            result[beyond] = (
                unique_points[-1]
                + (query[beyond] - unique_progress[-1])[:, None] * direction / norm
            )
    return np.concatenate([result, headings_from_positions(result)[:, None]], axis=1)


def lateral_profile(
    anchor: np.ndarray, profile: np.ndarray, offset_m: float
) -> np.ndarray:
    headings = np.asarray(anchor)[:, 2]
    normals = np.stack([-np.sin(headings), np.cos(headings)], axis=1)
    positions = (
        np.asarray(anchor)[:, :2]
        + float(offset_m) * np.asarray(profile)[:, None] * normals
    )
    return np.concatenate(
        [positions, headings_from_positions(positions)[:, None]], axis=1
    )


def fallback_candidates(
    anchor: np.ndarray,
) -> list[tuple[str, dict[str, float], np.ndarray]]:
    """Build smooth, interpretable alternatives around a logged GT anchor."""

    anchor = np.asarray(anchor, dtype=np.float64)
    count = len(anchor)
    u = np.arange(1, count + 1, dtype=np.float64) / count
    ramp = quintic_smoothstep(u)
    points, progress = polyline(anchor)
    base = progress[1:]
    late_u = np.clip((u - 0.25) / 0.75, 0.0, 1.0)
    late_ramp = quintic_smoothstep(late_u)
    middle_bump = np.sin(np.pi * u) ** 2
    early_bump = quintic_smoothstep(np.clip(u / 0.6, 0.0, 1.0)) * (
        1.0 - quintic_smoothstep(np.clip((u - 0.4) / 0.6, 0.0, 1.0))
    )
    speed = np.diff(progress) / 0.5
    decel_factor = 1.0 - 0.18 * quintic_smoothstep(np.clip((u - 0.25) / 0.75, 0.0, 1.0))
    accel_factor = 1.0 + 0.10 * quintic_smoothstep(u)
    candidates = [
        ("gt", {}, anchor.copy()),
        ("smooth_left", {"offset_m": 0.8}, lateral_profile(anchor, ramp, 0.8)),
        ("smooth_right", {"offset_m": -0.8}, lateral_profile(anchor, ramp, -0.8)),
        ("slight_left", {"offset_m": 0.35}, lateral_profile(anchor, ramp, 0.35)),
        ("slight_right", {"offset_m": -0.35}, lateral_profile(anchor, ramp, -0.35)),
        ("slow_time_scale", {"scale": 0.8}, resample_progress(anchor, base * 0.8)),
        (
            "slight_deceleration",
            {"terminal_factor": 0.82},
            resample_progress(anchor, np.cumsum(speed * decel_factor * 0.5)),
        ),
        (
            "slight_acceleration",
            {"terminal_factor": 1.1},
            resample_progress(anchor, np.cumsum(speed * accel_factor * 0.5)),
        ),
        ("fast_time_scale", {"scale": 1.15}, resample_progress(anchor, base * 1.15)),
        (
            "same_endpoint_mid_curve",
            {"offset_m": 0.65},
            lateral_profile(anchor, middle_bump, 0.65),
        ),
        (
            "same_prefix_different_tail",
            {"offset_m": -0.75, "prefix_s": 1.0},
            lateral_profile(anchor, late_ramp, -0.75),
        ),
        (
            "different_prefix_similar_endpoint",
            {"offset_m": 0.55},
            lateral_profile(anchor, early_bump, 0.55),
        ),
    ]
    return candidates


def candidate_validity(
    trajectory: np.ndarray,
) -> tuple[bool, list[str], dict[str, float]]:
    trajectory = np.asarray(trajectory, dtype=np.float64)
    reasons: list[str] = []
    if trajectory.shape != (8, 3):
        reasons.append("shape")
    if not np.isfinite(trajectory).all():
        reasons.append("non_finite")
    kinematics = trajectory_kinematics(trajectory)
    speed = np.asarray(kinematics["speed"])
    acceleration = np.asarray(kinematics["acceleration"])
    curvature = np.asarray(kinematics["curvature"])
    jerk = np.asarray(kinematics["jerk"])
    if np.max(speed, initial=0.0) > 30.0:
        reasons.append("speed_explosion")
    if np.max(np.abs(acceleration), initial=0.0) > 10.0:
        reasons.append("acceleration_explosion")
    if np.max(np.abs(curvature), initial=0.0) > 0.5:
        reasons.append("curvature_explosion")
    if np.max(np.abs(jerk), initial=0.0) > 25.0:
        reasons.append("jerk_explosion")
    summary = {
        "max_speed_mps": float(np.max(speed, initial=0.0)),
        "max_abs_acceleration_mps2": float(np.max(np.abs(acceleration), initial=0.0)),
        "max_abs_curvature_inv_m": float(np.max(np.abs(curvature), initial=0.0)),
        "max_abs_jerk_mps3": float(np.max(np.abs(jerk), initial=0.0)),
    }
    return not reasons, reasons, summary


def deterministic_id(
    scene_token: str, source: str, candidate_type: str, parameters: Mapping[str, float]
) -> str:
    payload = json.dumps(
        [scene_token, source, candidate_type, dict(parameters)], sort_keys=True
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def generate(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    split = "trainval" if args.split == "train" else args.split
    paths = discover_paths(args, split=split)
    bootstrap_navsim(paths)
    if paths.metric_cache is None:
        raise FileNotFoundError("MetricCache is required to obtain GT anchors")
    metric_index = load_metric_cache_index(paths.metric_cache)
    preferred = output_tokens(args.output_dir)
    source = args.source
    existing_available = (
        paths.candidate_cache is not None and paths.consequence_cache is not None
    )
    if source == "auto":
        source = "existing" if existing_available else "fallback"
    if source == "existing" and not existing_available:
        raise FileNotFoundError(
            "--source existing requested but no compatible bank was found"
        )

    trajectories_out: list[np.ndarray] = []
    scene_tokens_out: list[str] = []
    candidate_ids_out: list[str] = []
    source_indices_out: list[int] = []
    rows: list[dict[str, Any]] = []
    gt_errors: list[float] = []
    if source == "existing":
        all_trajectories, metadata, scene_index = load_candidate_source(paths)
        consequences = consequence_rows_by_scene(paths)
        selected_scenes = select_eligible_scenes(
            paths,
            max_scenes=args.max_scenes,
            num_candidates=args.num_candidates,
            require_reactive=args.traffic_policy == "reactive",
            preferred_tokens=preferred,
        )
        metadata_by_index = {
            int(item["trajectory"]["index"]): item for item in metadata
        }
        for scene_number, scene_token in enumerate(selected_scenes):
            consequence_by_slot = {
                int(item["scene_candidate_index"]): item
                for item in consequences[scene_token]
            }
            entry = scene_index[scene_token]
            source_indices = list(
                range(int(entry["start"]), int(entry["start"]) + int(entry["count"]))
            )
            accepted = [
                source_index
                for slot, source_index in enumerate(source_indices)
                if consequence_by_slot[slot].get("candidate_accepted")
                and consequence_by_slot[slot].get("log_replay", {}).get("available")
            ]
            accepted.sort(
                key=lambda index: (
                    metadata_by_index[index].get("perturbation_type") != "anchor",
                    metadata_by_index[index].get("perturbation_type", ""),
                    index,
                )
            )
            chosen = accepted[: args.num_candidates]
            if (
                not chosen
                or metadata_by_index[chosen[0]].get("perturbation_type") != "anchor"
            ):
                raise AssertionError(
                    f"GT anchor was not selected first for {scene_token}"
                )
            gt = np.asarray(
                load_metric_cache(metric_index[scene_token]).human_trajectory.poses,
                dtype=np.float64,
            )
            gt_errors.append(
                float(
                    np.max(
                        np.linalg.norm(
                            all_trajectories[chosen[0], :, :2] - gt[:, :2], axis=1
                        )
                    )
                )
            )
            scene_trajectories = np.asarray(all_trajectories[chosen], dtype=np.float32)
            if len(np.unique(np.round(scene_trajectories, 5), axis=0)) != len(
                scene_trajectories
            ):
                raise AssertionError(
                    f"duplicate trajectories selected for {scene_token}"
                )
            for candidate_slot, source_index in enumerate(chosen):
                item = metadata_by_index[source_index]
                trajectory = scene_trajectories[candidate_slot]
                valid, reasons, summary = candidate_validity(trajectory)
                candidate_id = str(item["candidate_id"])
                trajectories_out.append(trajectory)
                scene_tokens_out.append(scene_token)
                candidate_ids_out.append(candidate_id)
                source_indices_out.append(source_index)
                rows.append(
                    {
                        "scene_token": scene_token,
                        "scene_index": scene_number,
                        "candidate_index": candidate_slot,
                        "candidate_id": candidate_id,
                        "candidate_type": "gt"
                        if item["perturbation_type"] == "anchor"
                        else item["perturbation_type"],
                        "is_gt": item["perturbation_type"] == "anchor",
                        "parameters_json": json.dumps(
                            item.get("perturbation_parameters", {}), sort_keys=True
                        ),
                        "source": "existing_expert_anchor_heuristic_bank",
                        "source_candidate_index": source_index,
                        "implicit_start_x_m": 0.0,
                        "implicit_start_y_m": 0.0,
                        "implicit_start_heading_rad": 0.0,
                        "finite": bool(np.isfinite(trajectory).all()),
                        "kinematic_valid": valid,
                        "validation_reasons": json.dumps(reasons),
                        **summary,
                    }
                )
    else:
        selected_scenes = [token for token in preferred if token in metric_index][
            : args.max_scenes
        ]
        if len(selected_scenes) < args.max_scenes:
            selected_scenes += [
                token
                for token in sorted(metric_index)
                if token not in set(selected_scenes)
            ][: args.max_scenes - len(selected_scenes)]
        for scene_number, scene_token in enumerate(selected_scenes):
            anchor = np.asarray(
                load_metric_cache(metric_index[scene_token]).human_trajectory.poses,
                dtype=np.float64,
            )
            candidates = fallback_candidates(anchor)[: args.num_candidates]
            rounded = [np.round(candidate[2], 5).tobytes() for candidate in candidates]
            if len(set(rounded)) != len(rounded):
                raise AssertionError(f"fallback generated duplicates for {scene_token}")
            for candidate_slot, (candidate_type, parameters, trajectory) in enumerate(
                candidates
            ):
                valid, reasons, summary = candidate_validity(trajectory)
                candidate_id = deterministic_id(
                    scene_token, "fallback", candidate_type, parameters
                )
                trajectories_out.append(np.asarray(trajectory, dtype=np.float32))
                scene_tokens_out.append(scene_token)
                candidate_ids_out.append(candidate_id)
                source_indices_out.append(-1)
                rows.append(
                    {
                        "scene_token": scene_token,
                        "scene_index": scene_number,
                        "candidate_index": candidate_slot,
                        "candidate_id": candidate_id,
                        "candidate_type": candidate_type,
                        "is_gt": candidate_type == "gt",
                        "parameters_json": json.dumps(parameters, sort_keys=True),
                        "source": "deterministic_quintic_fallback",
                        "source_candidate_index": -1,
                        "implicit_start_x_m": 0.0,
                        "implicit_start_y_m": 0.0,
                        "implicit_start_heading_rad": 0.0,
                        "finite": bool(np.isfinite(trajectory).all()),
                        "kinematic_valid": valid,
                        "validation_reasons": json.dumps(reasons),
                        **summary,
                    }
                )

    scene_count = len(set(scene_tokens_out))
    if scene_count == 0:
        raise RuntimeError("no candidate scenes were produced")
    arrays = {
        "trajectories": np.asarray(trajectories_out, dtype=np.float32).reshape(
            scene_count, args.num_candidates, 8, 3
        ),
        "scene_tokens": np.asarray(scene_tokens_out, dtype="U32").reshape(
            scene_count, args.num_candidates
        ),
        "candidate_ids": np.asarray(candidate_ids_out, dtype="U32").reshape(
            scene_count, args.num_candidates
        ),
        "source_candidate_indices": np.asarray(
            source_indices_out, dtype=np.int64
        ).reshape(scene_count, args.num_candidates),
    }
    manifest = pd.DataFrame(rows)
    summary = {
        "source": source,
        "source_description": (
            "Pre-existing deterministic expert-anchor perturbation cache; not model multi-sample output"
            if source == "existing"
            else "Deterministic local fallback using quintic smooth profiles; not a logged or true future"
        ),
        "scene_count": scene_count,
        "candidate_count": len(rows),
        "candidates_per_scene": args.num_candidates,
        "valid_rate": float(manifest["kinematic_valid"].mean()),
        "gt_anchor_position_error_max_m": max(gt_errors) if gt_errors else 0.0,
        "seed": args.seed,
        "traffic_policy_selection": args.traffic_policy,
        "selected_scene_tokens": list(dict.fromkeys(scene_tokens_out)),
    }
    return manifest, arrays, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--num-candidates", type=int, default=12, choices=range(8, 17))
    parser.add_argument(
        "--source", choices=("auto", "existing", "fallback"), default="auto"
    )
    parser.add_argument(
        "--traffic-policy", choices=("non_reactive", "reactive"), default="non_reactive"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest, arrays, summary = generate(args)
    write_dataframe(manifest, args.output_dir / "candidate_manifest.parquet")
    np.savez_compressed(args.output_dir / "candidate_trajectories.npz", **arrays)
    write_json(args.output_dir / "candidate_generation_summary.json", summary)
    write_text(
        args.output_dir / "CANDIDATE_GENERATION_AUDIT.md",
        "\n".join(
            [
                "# Candidate Bank Audit",
                "",
                f"- Source: **{summary['source']}**",
                f"- Scope: **{summary['scene_count']} scenes × {summary['candidates_per_scene']} candidates**",
                f"- Kinematic validity: **{summary['valid_rate']:.3%}**",
                f"- GT anchor max position mismatch: **{summary['gt_anchor_position_error_max_m']:.6g} m**",
                "",
                summary["source_description"] + ".",
                "",
                "The current pose is represented explicitly as the implicit `(0, 0, 0)` prefix. Stored waypoints are the eight future 0.5 s poses. Generated perturbations are never described as real futures.",
                "",
            ]
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
