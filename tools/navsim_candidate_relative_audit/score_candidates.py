#!/usr/bin/env python3
"""Batch-simulate and score candidate trajectories through the local official path."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import (
    add_common_arguments,
    append_command,
    ensure_output_dir,
    metric_cache_loader,
    paths_from_args,
    read_parquet,
    write_json,
    write_markdown,
    write_parquet,
)


STATE_COLUMNS = (
    "sim_x_m",
    "sim_y_m",
    "sim_heading_rad",
    "sim_velocity_x_mps",
    "sim_velocity_y_mps",
    "sim_acceleration_x_mps2",
    "sim_acceleration_y_mps2",
    "sim_steering_angle_rad",
    "sim_steering_rate_radps",
    "sim_angular_velocity_radps",
    "sim_angular_acceleration_radps2",
)


def _poses_from_group(group: pd.DataFrame) -> np.ndarray:
    poses = []
    for row in group.sort_values("candidate_index").itertuples():
        poses.append(np.column_stack([row.pose_x_m, row.pose_y_m, row.pose_heading_rad]))
    return np.asarray(poses, dtype=np.float32)


def score_pose_batch(metric_cache: Any, poses: np.ndarray) -> dict[str, Any]:
    """Score one scene's candidate batch using the repository's evaluator classes."""

    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.agents.EpisodeDrive.score_module.train_pdm_scorer import PDMScorer, PDMScorerConfig
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
        EgoAreaIndex,
        MultiMetricIndex,
        WeightedMetricIndex,
    )

    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim != 3 or poses.shape[1:] != (8, 3):
        raise ValueError(f"Expected candidates with shape (K, 8, 3), got {poses.shape}")
    proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    simulator = PDMSimulator(proposal_sampling)
    scorer = PDMScorer(proposal_sampling, PDMScorerConfig())
    initial_ego_state = metric_cache.ego_state
    state_batches = []
    for candidate in poses:
        trajectory = transform_trajectory(Trajectory(candidate), initial_ego_state)
        state_batches.append(
            get_trajectory_as_array(trajectory, proposal_sampling, initial_ego_state.time_point)
        )
    controller_inputs = np.stack(state_batches, axis=0)
    simulated_states = simulator.simulate_proposals(controller_inputs, initial_ego_state)
    scores = scorer.score_proposals(
        simulated_states,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
        metric_cache.pdm_progress,
    )
    collision_times = np.asarray(scorer._collision_time_idcs, dtype=np.float64) * proposal_sampling.interval_length
    ttc_times = np.asarray(scorer._ttc_time_idcs, dtype=np.float64) * proposal_sampling.interval_length
    return {
        "simulated_states": np.asarray(simulated_states, dtype=np.float64),
        "score": np.asarray(scores, dtype=np.float64),
        "no_at_fault_collision": np.asarray(
            scorer._multi_metrics[MultiMetricIndex.NO_COLLISION], dtype=np.float64
        ),
        "dac": np.asarray(scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA], dtype=np.float64),
        "progress": np.asarray(scorer._weighted_metrics[WeightedMetricIndex.PROGRESS], dtype=np.float64),
        "raw_progress_m": np.asarray(scorer._progress_raw, dtype=np.float64),
        "ttc": np.asarray(scorer._weighted_metrics[WeightedMetricIndex.TTC], dtype=np.float64),
        "comfort": np.asarray(scorer._weighted_metrics[WeightedMetricIndex.COMFORTABLE], dtype=np.float64),
        "ddc": np.asarray(scorer._weighted_metrics[WeightedMetricIndex.DRIVING_DIRECTION], dtype=np.float64),
        "first_collision_time_s": collision_times,
        "first_ttc_violation_time_s": ttc_times,
        "multiple_lanes_by_step": np.asarray(
            scorer._ego_areas[:, :, EgoAreaIndex.MULTIPLE_LANES], dtype=bool
        ),
        "non_drivable_by_step": np.asarray(
            scorer._ego_areas[:, :, EgoAreaIndex.NON_DRIVABLE_AREA], dtype=bool
        ),
        "oncoming_by_step": np.asarray(
            scorer._ego_areas[:, :, EgoAreaIndex.ONCOMING_TRAFFIC], dtype=bool
        ),
        "comfort_components": np.asarray(scorer.is_comfortable, dtype=bool),
        "collision_track_tokens": [
            list(scorer.proposal_fault_collided_track_ids[index]) for index in range(len(poses))
        ],
        "ttc_track_tokens": [list(scorer.ttc_collided_track_ids[index]) for index in range(len(poses))],
        "progress_reference_m": float(np.asarray(metric_cache.pdm_progress).reshape(-1)[0]),
        "proposal_interval_s": proposal_sampling.interval_length,
        "proposal_num_poses": proposal_sampling.num_poses,
    }


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _rows_from_score(group: pd.DataFrame, score: dict[str, Any], runtime_s: float) -> list[dict[str, Any]]:
    ordered = group.sort_values("candidate_index").reset_index(drop=True)
    states = score["simulated_states"]
    rows = []
    for index, manifest_row in ordered.iterrows():
        row = manifest_row.to_dict()
        row.update(
            {
                "traffic_policy": "non_reactive",
                "scoring_success": True,
                "scoring_error": None,
                "scene_batch_runtime_s": runtime_s,
                "simulated_num_states": states.shape[1],
                "simulation_interval_s": score["proposal_interval_s"],
                "no_at_fault_collision": float(score["no_at_fault_collision"][index]),
                "dac": float(score["dac"][index]),
                "ddc": float(score["ddc"][index]),
                "tlc": None,
                "progress": float(score["progress"][index]),
                "raw_progress_m": float(score["raw_progress_m"][index]),
                "ttc": float(score["ttc"][index]),
                "lane_keeping": None,
                "history_comfort": None,
                "comfort": float(score["comfort"][index]),
                "extended_comfort": None,
                "aggregate_score": float(score["score"][index]),
                "first_collision_time_s": _finite_or_none(score["first_collision_time_s"][index]),
                "first_ttc_violation_time_s": _finite_or_none(score["first_ttc_violation_time_s"][index]),
                "collision_track_tokens": score["collision_track_tokens"][index],
                "ttc_track_tokens": score["ttc_track_tokens"][index],
                "multiple_lanes_by_step": score["multiple_lanes_by_step"][index].tolist(),
                "non_drivable_by_step": score["non_drivable_by_step"][index].tolist(),
                "oncoming_by_step": score["oncoming_by_step"][index].tolist(),
                "comfort_components": score["comfort_components"][index].tolist(),
                "progress_reference_m": score["progress_reference_m"],
                "official_metric_availability": json.dumps(
                    {
                        "no_at_fault_collision": True,
                        "dac": True,
                        "ddc": True,
                        "tlc": False,
                        "progress": True,
                        "ttc": True,
                        "lane_keeping": False,
                        "history_comfort": False,
                        "comfort": True,
                        "extended_comfort": False,
                    },
                    sort_keys=True,
                ),
            }
        )
        for column_index, column in enumerate(STATE_COLUMNS):
            row[column] = states[index, :, column_index].astype(np.float32).tolist()
        rows.append(row)
    return rows


def _failure_rows(group: pd.DataFrame, error: str) -> list[dict[str, Any]]:
    rows = []
    for _, manifest_row in group.sort_values("candidate_index").iterrows():
        row = manifest_row.to_dict()
        row.update(
            {
                "traffic_policy": "non_reactive",
                "scoring_success": False,
                "scoring_error": error,
                "simulated_num_states": 0,
                "aggregate_score": None,
            }
        )
        rows.append(row)
    return rows


def score_candidates(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    paths = paths_from_args(args)
    output_dir = ensure_output_dir(args.output_dir)
    if args.traffic_policy != "non_reactive":
        raise SystemExit(
            "The runtime NAVSIM 1.1 scorer has no reactive traffic-policy argument. "
            "Reactive code/data are audited separately by audit_v2_extensions.py; results must not be mixed."
        )
    manifest_path = output_dir / "candidate_manifest.parquet"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing candidate manifest: {manifest_path}")
    manifest = read_parquet(manifest_path)
    if args.max_scenes > 0:
        keep_tokens = manifest["scene_token"].drop_duplicates().head(args.max_scenes)
        manifest = manifest[manifest["scene_token"].isin(set(keep_tokens))].copy()
    caches = metric_cache_loader(paths)

    all_rows: list[dict[str, Any]] = []
    batch_single_errors: list[float] = []
    deterministic_errors: list[float] = []
    order_checks: list[bool] = []
    state_count_checks: list[bool] = []
    scene_factor_differences: list[bool] = []
    gt_scores: list[float] = []
    error_examples: list[dict[str, str]] = []
    for scene_number, (token, group) in enumerate(manifest.groupby("scene_token", sort=False)):
        poses = _poses_from_group(group)
        try:
            cache = caches.get_from_token(token)
            start = time.perf_counter()
            score = score_pose_batch(cache, poses)
            runtime_s = time.perf_counter() - start
            rows = _rows_from_score(group, score, runtime_s)
            all_rows.extend(rows)
            order_checks.append(
                [row["candidate_index"] for row in rows]
                == group.sort_values("candidate_index")["candidate_index"].tolist()
            )
            state_count_checks.append(score["simulated_states"].shape[1:] == (41, 11))
            factors = np.column_stack(
                [
                    score["no_at_fault_collision"],
                    score["dac"],
                    score["ddc"],
                    score["progress"],
                    score["ttc"],
                    score["comfort"],
                    score["score"],
                ]
            )
            scene_factor_differences.append(bool(np.any(np.ptp(factors, axis=0) > 1e-9)))
            gt_index = int(np.flatnonzero(group.sort_values("candidate_index")["is_gt"].to_numpy())[0])
            gt_scores.append(float(score["score"][gt_index]))

            if scene_number < args.sanity_scenes:
                repeat = score_pose_batch(cache, poses)
                deterministic_errors.append(
                    float(np.max(np.abs(score["simulated_states"] - repeat["simulated_states"])))
                )
                deterministic_errors.append(float(np.max(np.abs(score["score"] - repeat["score"]))))
                for candidate_index in range(len(poses)):
                    single = score_pose_batch(cache, poses[candidate_index : candidate_index + 1])
                    batch_single_errors.append(
                        float(abs(score["score"][candidate_index] - single["score"][0]))
                    )
                    batch_single_errors.append(
                        float(
                            np.max(
                                np.abs(
                                    score["simulated_states"][candidate_index]
                                    - single["simulated_states"][0]
                                )
                            )
                        )
                    )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            all_rows.extend(_failure_rows(group, error))
            if len(error_examples) < 10:
                error_examples.append({"scene_token": token, "error": error})

    metrics = pd.DataFrame(all_rows)
    write_parquet(metrics, output_dir / "candidate_metrics.parquet")
    success_rate = float(metrics["scoring_success"].mean()) if len(metrics) else 0.0
    valid = metrics[metrics["scoring_success"]]
    gt_reasonable = bool(gt_scores and np.isfinite(gt_scores).all() and np.min(gt_scores) >= 0 and np.max(gt_scores) <= 1)
    deterministic_max = max(deterministic_errors, default=float("inf"))
    batch_single_max = max(batch_single_errors, default=float("inf"))
    varied_scene_count = int(sum(scene_factor_differences))
    blockers = []
    if success_rate <= 0.98:
        blockers.append(f"Scoring success {success_rate:.3%} does not exceed 98%")
    if not gt_reasonable:
        blockers.append("At least one GT candidate score is non-finite or outside [0, 1]")
    if deterministic_max > 1e-12:
        blockers.append(f"Repeated official scoring differs by {deterministic_max}")
    if batch_single_max > 1e-12:
        blockers.append(f"Batch and single-candidate scoring differ by {batch_single_max}")
    if not all(order_checks):
        blockers.append("Candidate order changed during scoring")
    if not all(state_count_checks):
        blockers.append("Simulated state shape is not (41, 11)")
    if varied_scene_count == 0:
        blockers.append("No audited scene has differing candidate PDM factors")
    gate_pass = not blockers
    audit = {
        "gate": "B",
        "passed": gate_pass,
        "traffic_policy": "non_reactive",
        "scene_count": int(metrics["scene_token"].nunique()) if len(metrics) else 0,
        "candidate_count": len(metrics),
        "success_rate": success_rate,
        "gt_score_min": min(gt_scores, default=None),
        "gt_score_mean": float(np.mean(gt_scores)) if gt_scores else None,
        "gt_score_max": max(gt_scores, default=None),
        "deterministic_max_abs_error": deterministic_max,
        "batch_vs_single_max_abs_error": batch_single_max,
        "candidate_order_all_preserved": all(order_checks),
        "state_count_all_aligned": all(state_count_checks),
        "scenes_with_factor_differences": varied_scene_count,
        "scenes_with_factor_differences_rate": varied_scene_count / max(len(scene_factor_differences), 1),
        "error_examples": error_examples,
        "blockers": blockers,
        "progress_normalization_note": (
            "Local custom PDMScorer normalizes each proposal's raw progress against the cached PDM baseline "
            "metric_cache.pdm_progress, not against the maximum of the candidate set."
        ),
        "official_paths": [
            "navsim/evaluate/pdm_score.py:transform_trajectory,get_trajectory_as_array",
            "navsim/planning/simulation/planner/pdm_planner/simulation/pdm_simulator.py:PDMSimulator",
            "navsim/agents/EpisodeDrive/score_module/train_pdm_scorer.py:PDMScorer",
        ],
    }
    write_json(output_dir / "candidate_scoring_audit.json", audit)
    gate_path = output_dir / "gate_status.json"
    existing = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.exists() else {}
    existing["gate_b"] = {
        "passed": gate_pass,
        "blockers": blockers,
        "success_rate": success_rate,
        "candidate_count": len(metrics),
    }
    write_json(gate_path, existing)
    report = f"""# Candidate Scoring Audit

## Gate B: {'PASS' if gate_pass else 'FAIL'}

- Traffic policy: `non_reactive`
- Scenes / candidates: {audit['scene_count']} / {len(metrics)}
- Successful candidates: {success_rate:.3%}
- GT score mean (min–max): {audit['gt_score_mean']} ({audit['gt_score_min']}–{audit['gt_score_max']})
- Repeated-run max absolute difference: {deterministic_max}
- Batch-vs-single max absolute difference: {batch_single_max}
- Candidate order preserved: {audit['candidate_order_all_preserved']}
- Simulated state shape: 41 states × 11 fields at 0.1 s for every successful candidate: {audit['state_count_all_aligned']}
- Scenes with at least one differing factor: {varied_scene_count}/{len(scene_factor_differences)}

The evaluator exposes no-at-fault collision, DAC, DDC, progress, TTC, comfort and aggregate score.  This v1/custom single-scene interface does not expose TLC, lane keeping, history comfort or extended comfort; those columns are null and accompanied by an availability map, not synthesized.

Progress comparability: the deployed scorer normalizes each candidate against the cached PDM baseline progress. It does not use the maximum progress of the submitted candidate set.

## Blockers

{chr(10).join('- ' + blocker for blocker in blockers) if blockers else '- None.'}
"""
    write_markdown(output_dir / "CANDIDATE_SCORING_AUDIT.md", report)
    if not gate_pass:
        raise SystemExit("Gate B failed; candidate-relative target construction must not run")
    return metrics, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, include_candidates=True)
    parser.add_argument("--traffic-policy", choices=("non_reactive", "reactive"), default="non_reactive")
    parser.add_argument("--sanity-scenes", type=int, default=2)
    args = parser.parse_args()
    score_candidates(args)
    append_command(args.output_dir.resolve(), "python -m tools.navsim_candidate_relative_audit.score_candidates " + " ".join(sys.argv[1:]))


if __name__ == "__main__":
    main()
