#!/usr/bin/env python3
"""Phase 4: score candidates through the deployed official NAVSIM v2 path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import (
    add_common_arguments,
    bootstrap_navsim,
    consequence_rows_by_scene,
    discover_paths,
    load_metric_cache,
    load_metric_cache_index,
    metric_log_name,
    write_dataframe,
    write_json,
    write_text,
)


POLICY_NAMESPACE = {"non_reactive": "log_replay", "reactive": "reactive_model"}


def load_inputs(output_dir: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    manifest_path = output_dir / "candidate_manifest.parquet"
    trajectory_path = output_dir / "candidate_trajectories.npz"
    if not manifest_path.is_file() or not trajectory_path.is_file():
        raise FileNotFoundError("candidate_generator must run before score_candidates")
    manifest = pd.read_parquet(manifest_path).sort_values(
        ["scene_index", "candidate_index"]
    )
    with np.load(trajectory_path) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    return manifest, arrays


def scalar_metrics(row: dict[str, Any], namespace: str) -> dict[str, Any]:
    policy = row[namespace]
    exact = row["exact"]
    return {
        "no_at_fault_collision": policy.get("no_at_fault_collision"),
        "drivable_area_compliance": exact.get("drivable_area_compliance"),
        "driving_direction_compliance": exact.get("driving_direction_compliance"),
        "traffic_light_compliance": policy.get("traffic_light_compliance"),
        "ego_progress": exact.get("official_ego_progress_score"),
        "time_to_collision_within_bound": policy.get("time_to_collision_within_bound"),
        "lane_keeping": exact.get("lane_keeping"),
        "history_comfort": exact.get("history_comfort"),
        "extended_comfort": exact.get("extended_comfort"),
        "aggregate_score": policy.get("pdm_score"),
        "first_collision_time_s": policy.get("at_fault_collision_time_s"),
        "collision_observed": policy.get("at_fault_collision_observed"),
        "first_ttc_violation_time_s": policy.get("ttc_infraction_time_s"),
        "ttc_violation_observed": policy.get("ttc_infraction_observed"),
        "minimum_clearance_m": policy.get("minimum_clearance_m"),
        "minimum_dynamic_clearance_m": policy.get("minimum_dynamic_clearance_m"),
        "dynamic_collision": policy.get("dynamic_collision"),
        "dynamic_occupancy_fraction": policy.get("dynamic_occupancy_fraction"),
        "centerline_progress_m": exact.get("centerline_progress_m"),
        "route_deviation_mean_m": exact.get("route_deviation_mean_m"),
        "route_deviation_max_m": exact.get("route_deviation_max_m"),
        "intersection_fraction": exact.get("intersection_fraction"),
    }


def verify_official_repeat(
    paths: Any,
    metric_index: dict[str, Path],
    scene_token: str,
    trajectory: np.ndarray,
    namespace: str,
    cached_states: np.ndarray,
    cached_metrics: dict[str, Any],
    repeat_count: int,
) -> dict[str, Any]:
    if repeat_count <= 0:
        return {"performed": False}
    from research.action_effect.consequence_builder import (
        ConsequenceConfig,
        build_log_replay_policy,
        build_reactive_policy,
        make_scorer,
        score_under_assumption,
    )

    cache = load_metric_cache(metric_index[scene_token])
    config = ConsequenceConfig()
    runs: list[tuple[dict[str, Any], np.ndarray]] = []
    for _ in range(repeat_count):
        simulator, scorer = make_scorer(config)
        policy = (
            build_log_replay_policy(config)
            if namespace == "log_replay"
            else build_reactive_policy(config, cache.map_parameters.map_root)
        )
        result, _, states, _ = score_under_assumption(
            cache,
            trajectory,
            simulator,
            scorer,
            policy,
            config,
        )
        runs.append((result, np.asarray(states)))
    repeat_state_max_error = max(
        float(np.max(np.abs(states - runs[0][1]))) for _, states in runs
    )
    cached_state_max_error = float(np.nanmax(np.abs(cached_states - runs[0][1])))
    cached_state_float32_max_error = float(
        np.nanmax(np.abs(cached_states - runs[0][1].astype(np.float32)))
    )
    compared_fields = (
        "pdm_score",
        "no_at_fault_collision",
        "traffic_light_compliance",
        "time_to_collision_within_bound",
        "minimum_dynamic_clearance_m",
    )
    metric_errors = {
        field: abs(
            float(runs[0][0][field])
            - float(
                cached_metrics[field if field != "pdm_score" else "aggregate_score"]
            )
        )
        for field in compared_fields
    }
    repeat_metric_max_error = max(
        abs(float(run[0][field]) - float(runs[0][0][field]))
        for run in runs
        for field in compared_fields
    )
    return {
        "performed": True,
        "repeat_count": repeat_count,
        "repeat_state_max_abs_error": repeat_state_max_error,
        "repeat_metric_max_abs_error": repeat_metric_max_error,
        "cached_state_max_abs_error": cached_state_max_error,
        "cached_state_float32_max_abs_error": cached_state_float32_max_error,
        "cached_metric_abs_errors": metric_errors,
        "deterministic": repeat_state_max_error < 1e-9
        and repeat_metric_max_error < 1e-12,
    }


def score(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    split = "trainval" if args.split == "train" else args.split
    paths = discover_paths(args, split=split)
    runtime = bootstrap_navsim(paths)
    if paths.metric_cache is None or paths.consequence_cache is None:
        raise FileNotFoundError(
            "official MetricCache and existing consequence cache are required"
        )
    manifest, candidate_arrays = load_inputs(args.output_dir)
    trajectories = np.asarray(candidate_arrays["trajectories"], dtype=np.float32)
    scene_tokens = trajectories.shape[0]
    candidates_per_scene = trajectories.shape[1]
    if args.max_scenes < scene_tokens:
        trajectories = trajectories[: args.max_scenes]
        manifest = manifest[manifest["scene_index"] < args.max_scenes].copy()
        scene_tokens = args.max_scenes
    consequence_by_scene = consequence_rows_by_scene(paths)
    metric_index = load_metric_cache_index(paths.metric_cache)
    namespace = POLICY_NAMESPACE[args.traffic_policy]
    consequence_root = paths.consequence_cache
    simulated_states = np.full(
        (scene_tokens, candidates_per_scene, 41, 11), np.nan, dtype=np.float32
    )
    minimum_clearance_t = np.full(
        (scene_tokens, candidates_per_scene, 41), np.nan, dtype=np.float32
    )
    minimum_dynamic_clearance_t = np.full_like(minimum_clearance_t, np.nan)
    overlap_t = np.zeros((scene_tokens, candidates_per_scene, 41), dtype=bool)
    dynamic_overlap_t = np.zeros_like(overlap_t)
    dynamic_count_t = np.zeros((scene_tokens, candidates_per_scene, 41), dtype=np.int16)
    output_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for scene_index in range(scene_tokens):
        scene_manifest = manifest[manifest["scene_index"] == scene_index].sort_values(
            "candidate_index"
        )
        scene_token = str(scene_manifest.iloc[0]["scene_token"])
        rows_by_source = {
            int(row["candidate_index"]): row
            for row in consequence_by_scene[scene_token]
        }
        scene_file = consequence_root / "scenes" / f"{scene_token}.npz"
        if not scene_file.is_file():
            raise FileNotFoundError(
                f"official consequence arrays missing: {scene_file}"
            )
        with np.load(scene_file) as arrays:
            arrays_copy = {name: np.asarray(arrays[name]) for name in arrays.files}
        prefix = namespace
        for _, item in scene_manifest.iterrows():
            candidate_index = int(item["candidate_index"])
            source_index = int(item["source_candidate_index"])
            row = rows_by_source[source_index]
            source_slot = int(row["scene_candidate_index"])
            policy = row.get(namespace, {})
            success = bool(row.get("candidate_accepted") and policy.get("available"))
            if success:
                simulated_states[scene_index, candidate_index] = arrays_copy[
                    f"{prefix}_simulated_states"
                ][source_slot]
                minimum_clearance_t[scene_index, candidate_index] = arrays_copy[
                    f"{prefix}_minimum_clearance_t"
                ][source_slot]
                minimum_dynamic_clearance_t[scene_index, candidate_index] = arrays_copy[
                    f"{prefix}_minimum_dynamic_clearance_t"
                ][source_slot]
                overlap_t[scene_index, candidate_index] = arrays_copy[
                    f"{prefix}_time_indexed_overlap"
                ][source_slot]
                dynamic_overlap_t[scene_index, candidate_index] = arrays_copy[
                    f"{prefix}_time_indexed_dynamic_overlap"
                ][source_slot]
                dynamic_count_t[scene_index, candidate_index] = arrays_copy[
                    f"{prefix}_dynamic_agents_within_radius_t"
                ][source_slot]
                metrics = scalar_metrics(row, namespace)
            else:
                metrics = {
                    name: None
                    for name in scalar_metrics({"exact": {}, namespace: {}}, namespace)
                }
                failures.append(
                    {
                        "scene_token": scene_token,
                        "candidate_id": str(item["candidate_id"]),
                        "reason": str(policy.get("reason", "unavailable")),
                    }
                )
            state = simulated_states[scene_index, candidate_index]
            output_rows.append(
                {
                    "scene_token": scene_token,
                    "log_name": metric_log_name(metric_index[scene_token]),
                    "scene_index": scene_index,
                    "candidate_index": candidate_index,
                    "candidate_id": str(item["candidate_id"]),
                    "candidate_type": str(item["candidate_type"]),
                    "is_gt": bool(item["is_gt"]),
                    "traffic_policy": args.traffic_policy,
                    "traffic_policy_provenance": (
                        "non-reactive logged future replay"
                        if namespace == "log_replay"
                        else "NAVSIM v2 IDM reactive-policy simulated consequence"
                    ),
                    "success": success,
                    "official_cache_reused": True,
                    "simulated_state_count": int(np.sum(np.isfinite(state[:, 0])))
                    if success
                    else 0,
                    "simulated_x_m": state[:, 0].tolist() if success else [],
                    "simulated_y_m": state[:, 1].tolist() if success else [],
                    "simulated_heading_rad": state[:, 2].tolist() if success else [],
                    "simulated_velocity_x_mps": state[:, 3].tolist() if success else [],
                    "simulated_velocity_y_mps": state[:, 4].tolist() if success else [],
                    "simulated_acceleration_x_mps2": state[:, 5].tolist()
                    if success
                    else [],
                    "simulated_acceleration_y_mps2": state[:, 6].tolist()
                    if success
                    else [],
                    "simulated_steering_angle_rad": state[:, 7].tolist()
                    if success
                    else [],
                    "simulated_steering_rate_radps": state[:, 8].tolist()
                    if success
                    else [],
                    "simulated_angular_velocity_radps": state[:, 9].tolist()
                    if success
                    else [],
                    **metrics,
                }
            )

    result_frame = pd.DataFrame(output_rows)
    valid = result_frame[result_frame["success"]]
    per_scene_unique = valid.groupby("scene_token")[
        [
            "aggregate_score",
            "no_at_fault_collision",
            "drivable_area_compliance",
            "driving_direction_compliance",
            "traffic_light_compliance",
            "time_to_collision_within_bound",
            "lane_keeping",
            "ego_progress",
        ]
    ].nunique(dropna=True)
    factor_diverse_scene_count = int((per_scene_unique.max(axis=1) > 1).sum())
    first = valid.iloc[0]
    first_scene_index = int(first["scene_index"])
    first_candidate_index = int(first["candidate_index"])
    verification = verify_official_repeat(
        paths,
        metric_index,
        str(first["scene_token"]),
        trajectories[first_scene_index, first_candidate_index],
        namespace,
        simulated_states[first_scene_index, first_candidate_index],
        first.to_dict(),
        args.verify_runs,
    )
    success_rate = float(result_frame["success"].mean())
    gt_alignment_summary = json.loads(
        (args.output_dir / "candidate_generation_summary.json").read_text()
    )
    criteria = {
        "legal_candidate_success_rate_gt_98pct": success_rate > 0.98,
        "gt_coordinate_alignment": gt_alignment_summary[
            "gt_anchor_position_error_max_m"
        ]
        < 1e-4,
        "deterministic_repeat": bool(verification.get("deterministic", False)),
        "candidate_factor_difference_exists": factor_diverse_scene_count > 0,
        "state_horizon_alignment": bool((valid["simulated_state_count"] == 41).all()),
        "candidate_order_preserved": bool(
            result_frame.sort_values(["scene_index", "candidate_index"])[
                "candidate_id"
            ].tolist()
            == manifest.sort_values(["scene_index", "candidate_index"])[
                "candidate_id"
            ].tolist()
        ),
    }
    gate_b = all(criteria.values())
    arrays_out = {
        "simulated_states": simulated_states,
        "minimum_clearance_t": minimum_clearance_t,
        "minimum_dynamic_clearance_t": minimum_dynamic_clearance_t,
        "time_indexed_overlap": overlap_t,
        "time_indexed_dynamic_overlap": dynamic_overlap_t,
        "dynamic_agents_within_radius_t": dynamic_count_t,
    }
    summary = {
        "gate_b": "PASS" if gate_b else "FAIL",
        "criteria": criteria,
        "traffic_policy": args.traffic_policy,
        "runtime": runtime,
        "scene_count": scene_tokens,
        "candidate_count": len(result_frame),
        "success_count": int(result_frame["success"].sum()),
        "success_rate": success_rate,
        "factor_diverse_scene_count": factor_diverse_scene_count,
        "gt_score_statistics": valid[valid["is_gt"]]["aggregate_score"]
        .describe()
        .to_dict(),
        "repeat_verification": verification,
        "failures": failures,
        "normalization_note": "Each cached candidate was scored with the same PDM-closed reference proposal. No across-scene max-progress normalization is used as a training label. Official ego_progress remains an official within-call normalized factor.",
        "batch_note": "The deployed pdm_score API evaluates a candidate plus the fixed PDM reference. Candidates remain individually scored because proposal-set progress normalization makes all-K scorer batching a different protocol; the simulator itself is batch-capable.",
    }
    return result_frame, arrays_out, summary


def render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Candidate Scoring Audit",
            "",
            f"## Gate B: **{summary['gate_b']}**",
            "",
            *(
                f"- {name}: **{'PASS' if value else 'FAIL'}**"
                for name, value in summary["criteria"].items()
            ),
            "",
            f"- Traffic setting: `{summary['traffic_policy']}`",
            f"- Official scoring success: **{summary['success_count']} / {summary['candidate_count']} ({summary['success_rate']:.3%})**",
            f"- Scenes with at least one differing PDM factor: **{summary['factor_diverse_scene_count']} / {summary['scene_count']}**",
            f"- Repeat state max error: `{summary['repeat_verification'].get('repeat_state_max_abs_error')}`",
            f"- Cached-vs-repeat state max error: `{summary['repeat_verification'].get('cached_state_max_abs_error')}`",
            f"- Cached-vs-repeat after declared float32 storage cast: `{summary['repeat_verification'].get('cached_state_float32_max_abs_error')}`",
            "",
            summary["batch_note"],
            "",
            summary["normalization_note"],
            "",
            "The Parquet table keeps non-reactive and reactive labels in distinct runs/columns through the `traffic_policy` field. These are official simulated consequences, not true causal counterfactuals.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument(
        "--traffic-policy", choices=tuple(POLICY_NAMESPACE), default="non_reactive"
    )
    parser.add_argument("--verify-runs", type=int, default=2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame, arrays, summary = score(args)
    write_dataframe(frame, args.output_dir / "candidate_metrics.parquet")
    np.savez_compressed(args.output_dir / "candidate_simulation_arrays.npz", **arrays)
    write_json(args.output_dir / "candidate_scoring_summary.json", summary)
    write_json(
        args.output_dir / "gate_b.json",
        {key: summary[key] for key in ("gate_b", "criteria", "failures")},
    )
    write_text(args.output_dir / "CANDIDATE_SCORING_AUDIT.md", render_report(summary))
    print(
        json.dumps(
            {"gate_b": summary["gate_b"], "criteria": summary["criteria"]}, indent=2
        )
    )
    if summary["gate_b"] != "PASS":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
