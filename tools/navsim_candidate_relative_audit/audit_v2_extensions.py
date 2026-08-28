#!/usr/bin/env python3
"""Phase 10: audit deployed NAVSIM-v2 reactive and synthetic extensions."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import pickle
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .common import (
    add_common_arguments,
    bootstrap_navsim,
    consequence_rows_by_scene,
    discover_paths,
    load_metric_cache,
    load_metric_cache_index,
    write_dataframe,
    write_json,
    write_text,
)


HORIZON_INDICES = (10, 20, 40)
HORIZONS_S = (1.0, 2.0, 4.0)


def tracked_state(detections: Any) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for actor in detections.tracked_objects.tracked_objects:
        velocity = getattr(actor, "velocity", None)
        output[str(actor.metadata.track_token)] = {
            "type": str(actor.tracked_object_type),
            "x": float(actor.center.x),
            "y": float(actor.center.y),
            "speed": float(
                math.hypot(getattr(velocity, "x", 0.0), getattr(velocity, "y", 0.0))
            ),
        }
    return output


def candidate_track_comparison(
    scene_token: str,
    candidate_slot: int,
    replay_tracks: Sequence[Any],
    reactive_tracks: Sequence[Any],
    ego_states: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon_s, time_index in zip(HORIZONS_S, HORIZON_INDICES):
        replay = tracked_state(replay_tracks[time_index])
        reactive = tracked_state(reactive_tracks[time_index])
        ego_xy = np.asarray(ego_states[time_index, :2], dtype=np.float64)
        for token in sorted(set(replay) & set(reactive)):
            logged = replay[token]
            response = reactive[token]
            delta_position = float(
                math.hypot(response["x"] - logged["x"], response["y"] - logged["y"])
            )
            logged_gap = float(
                math.hypot(logged["x"] - ego_xy[0], logged["y"] - ego_xy[1])
            )
            reactive_gap = float(
                math.hypot(response["x"] - ego_xy[0], response["y"] - ego_xy[1])
            )
            rows.append(
                {
                    "scene_token": scene_token,
                    "candidate_slot": candidate_slot,
                    "horizon_s": horizon_s,
                    "track_token": token,
                    "object_type": response["type"],
                    "endpoint_change_m": delta_position,
                    "speed_change_mps": response["speed"] - logged["speed"],
                    "braking_response": response["speed"] < logged["speed"] - 0.2,
                    "headway_distance_change_m": reactive_gap - logged_gap,
                    "replay_speed_mps": logged["speed"],
                    "reactive_speed_mps": response["speed"],
                }
            )
    return rows


def audit_reactive(args: argparse.Namespace, paths: Any) -> dict[str, Any]:
    if paths.consequence_cache is None or paths.metric_cache is None:
        return {"available": False, "reason": "consequence or MetricCache root missing"}
    grouped = consequence_rows_by_scene(paths)
    metric_index = load_metric_cache_index(paths.metric_cache)
    reactive_tokens = [
        token
        for token, rows in grouped.items()
        if token in metric_index
        and any(row.get("reactive_model", {}).get("available") for row in rows)
    ]
    cache_summary = json.loads((paths.consequence_cache / "summary.json").read_text())
    scalar_rows: list[dict[str, Any]] = []
    for token in reactive_tokens:
        for row in grouped[token]:
            if not row.get("reactive_model", {}).get("available"):
                continue
            replay = row["log_replay"]
            reactive = row["reactive_model"]
            scalar_rows.append(
                {
                    "scene_token": token,
                    "scene_candidate_index": int(row["scene_candidate_index"]),
                    "pdm_score_change": float(
                        reactive["pdm_score"] - replay["pdm_score"]
                    ),
                    "minimum_clearance_change_m": float(
                        reactive["minimum_clearance_m"] - replay["minimum_clearance_m"]
                    ),
                    "minimum_dynamic_clearance_change_m": float(
                        reactive["minimum_dynamic_clearance_m"]
                        - replay["minimum_dynamic_clearance_m"]
                    ),
                    "dynamic_collision_changed": bool(
                        reactive["dynamic_collision"] != replay["dynamic_collision"]
                    ),
                }
            )
    scalar_frame = pd.DataFrame(scalar_rows)
    response_rows: list[dict[str, Any]] = []
    candidate_dependence_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    if not args.skip_track_rerun:
        from research.action_effect.consequence_builder import (
            ConsequenceConfig,
            build_log_replay_policy,
            build_reactive_policy,
        )

        config = ConsequenceConfig()
        for token in reactive_tokens[: args.max_scenes]:
            try:
                available = [
                    row
                    for row in grouped[token]
                    if row.get("reactive_model", {}).get("available")
                ]
                available.sort(
                    key=lambda row: (
                        row.get("perturbation_type") != "anchor",
                        float(row["reactive_model"]["pdm_score"]),
                    )
                )
                anchor = next(
                    (
                        row
                        for row in available
                        if row.get("perturbation_type") == "anchor"
                    ),
                    available[0],
                )
                # Prefer an alternative for which the cached reactive policy
                # already changed dynamic clearance; this is a stronger and
                # still deterministic response audit than merely taking the
                # lowest map/progress score.
                alternative = max(
                    (row for row in available if row is not anchor),
                    key=lambda row: (
                        abs(
                            float(row["reactive_model"]["minimum_dynamic_clearance_m"])
                            - float(row["log_replay"]["minimum_dynamic_clearance_m"])
                        ),
                        bool(row["reactive_model"]["dynamic_collision"])
                        != bool(row["log_replay"]["dynamic_collision"]),
                        -float(row["reactive_model"]["pdm_score"]),
                    ),
                )
                selected = [anchor, alternative]
                with np.load(
                    paths.consequence_cache / "scenes" / f"{token}.npz"
                ) as payload:
                    states = np.asarray(
                        payload["reactive_model_simulated_states"], dtype=np.float64
                    )
                cache = load_metric_cache(metric_index[token])
                candidate_reactive_tracks: dict[int, Sequence[Any]] = {}
                for row in selected:
                    slot = int(row["scene_candidate_index"])
                    ego_states = states[slot]
                    replay_policy = build_log_replay_policy(config)
                    reactive_policy = build_reactive_policy(
                        config, cache.map_parameters.map_root
                    )
                    replay_tracks = replay_policy.simulate_environment(
                        ego_states, cache
                    )
                    reactive_tracks = reactive_policy.simulate_environment(
                        ego_states, cache
                    )
                    candidate_reactive_tracks[slot] = reactive_tracks
                    response_rows.extend(
                        candidate_track_comparison(
                            token, slot, replay_tracks, reactive_tracks, ego_states
                        )
                    )
                left_slot, right_slot = [
                    int(row["scene_candidate_index"]) for row in selected
                ]
                left = tracked_state(candidate_reactive_tracks[left_slot][40])
                right = tracked_state(candidate_reactive_tracks[right_slot][40])
                for actor_token in sorted(set(left) & set(right)):
                    if "VEHICLE" not in left[actor_token]["type"].upper():
                        continue
                    candidate_dependence_rows.append(
                        {
                            "scene_token": token,
                            "candidate_i": left_slot,
                            "candidate_j": right_slot,
                            "track_token": actor_token,
                            "candidate_dependent_endpoint_variance_m": math.hypot(
                                left[actor_token]["x"] - right[actor_token]["x"],
                                left[actor_token]["y"] - right[actor_token]["y"],
                            ),
                            "candidate_dependent_speed_difference_mps": abs(
                                left[actor_token]["speed"] - right[actor_token]["speed"]
                            ),
                        }
                    )
            except Exception as error:
                failures.append(
                    {"scene_token": token, "error": f"{type(error).__name__}: {error}"}
                )
    response = pd.DataFrame(response_rows)
    dependence = pd.DataFrame(candidate_dependence_rows)
    write_dataframe(scalar_frame, args.output_dir / "reactive_scalar_comparison.csv")
    write_dataframe(response, args.output_dir / "reactive_actor_response.csv")
    write_dataframe(dependence, args.output_dir / "reactive_candidate_dependence.csv")
    types = sorted(response["object_type"].unique().tolist()) if len(response) else []
    vehicle_mask = (
        response["object_type"].str.contains("VEHICLE", case=False)
        if len(response)
        else np.asarray([], dtype=bool)
    )
    nonvehicle = response[~vehicle_mask] if len(response) else response
    return {
        "available": bool(reactive_tokens),
        "official_cache_reactive_scene_count": int(
            cache_summary.get("reactive_scene_count", 0)
        ),
        "official_cache_reactive_candidate_count": int(
            cache_summary.get("reactive_available_count", 0)
        ),
        "track_rerun_requested_scenes": 0
        if args.skip_track_rerun
        else min(args.max_scenes, len(reactive_tokens)),
        "track_rerun_success_scenes": 0
        if args.skip_track_rerun
        else min(args.max_scenes, len(reactive_tokens)) - len(failures),
        "track_rerun_failure_rate": (
            len(failures) / min(args.max_scenes, len(reactive_tokens))
            if not args.skip_track_rerun and reactive_tokens
            else None
        ),
        "actor_comparison_count": len(response),
        "actor_endpoint_change_mean_m": float(response["endpoint_change_m"].mean())
        if len(response)
        else None,
        "actor_endpoint_change_p95_m": float(
            response["endpoint_change_m"].quantile(0.95)
        )
        if len(response)
        else None,
        "actor_speed_change_abs_mean_mps": float(
            response["speed_change_mps"].abs().mean()
        )
        if len(response)
        else None,
        "braking_response_rate": float(response["braking_response"].mean())
        if len(response)
        else None,
        "candidate_dependent_vehicle_comparison_count": len(dependence),
        "candidate_dependent_endpoint_change_nonzero_rate": float(
            (dependence["candidate_dependent_endpoint_variance_m"] > 1e-3).mean()
        )
        if len(dependence)
        else None,
        "candidate_dependent_endpoint_change_mean_m": float(
            dependence["candidate_dependent_endpoint_variance_m"].mean()
        )
        if len(dependence)
        else None,
        "object_types_observed": types,
        "policy_simulated_object_types": ["VEHICLE"],
        "log_replayed_object_types": "all non-VEHICLE types, including pedestrians and static objects",
        "nonvehicle_empirical_response_max_m": float(
            nonvehicle["endpoint_change_m"].max()
        )
        if len(nonvehicle)
        else None,
        "scalar_score_change_abs_mean": float(
            scalar_frame["pdm_score_change"].abs().mean()
        )
        if len(scalar_frame)
        else None,
        "scalar_dynamic_collision_change_rate": float(
            scalar_frame["dynamic_collision_changed"].mean()
        )
        if len(scalar_frame)
        else None,
        "failures": failures,
        "code_paths": [
            "navsim/navsim/traffic_agents_policies/navsim_IDM_traffic_agents.py",
            "navsim/navsim/traffic_agents_policies/abstract_traffic_agents_policy.py",
            "research/action_effect/consequence_builder.py",
        ],
    }


def synthetic_sensor_exists(root: Path | None, relative: Any) -> bool:
    if root is None or relative is None:
        return False
    return (root / Path(relative)).is_file()


def audit_synthetic(args: argparse.Namespace, paths: Any) -> dict[str, Any]:
    root = paths.synthetic_scenes
    if root is None or not root.is_dir():
        return {"available": False, "reason": "no synthetic scene directory resolved"}
    files = sorted(root.glob("*.pkl"))
    sampled = files[: min(args.synthetic_metadata_samples, len(files))]
    rows: list[dict[str, Any]] = []
    mapping_counts: Counter[str] = Counter()
    failures: list[dict[str, str]] = []
    initial_states_by_original: dict[str, list[np.ndarray]] = defaultdict(list)
    for path in sampled:
        try:
            with path.open("rb") as stream:
                data = pickle.load(stream)
            metadata = data["scene_metadata"]
            frames = data.get("frames", [])
            current_index = int(metadata.get("num_history_frames", 4)) - 1
            current = frames[current_index]
            mapping = str(metadata.get("corresponding_original_scene"))
            mapping_counts[mapping] += 1
            state = current["ego_status"]
            pose = np.asarray(state["ego_pose"], dtype=np.float64)
            speed = float(np.linalg.norm(state["ego_velocity"]))
            initial_states_by_original[mapping].append(
                np.asarray([pose[0], pose[1], speed])
            )
            camera = current.get("camera_dict", {}).get("cam_f0", {})
            rows.append(
                {
                    "synthetic_scene_token": metadata.get("scene_token"),
                    "initial_token": metadata.get("initial_token"),
                    "corresponding_original_scene": mapping,
                    "corresponding_original_initial_token": metadata.get(
                        "corresponding_original_initial_token"
                    ),
                    "history_frames": metadata.get("num_history_frames"),
                    "future_frames": metadata.get("num_future_frames"),
                    "frame_annotation_count": sum(
                        len(frame.get("annotations", {}).get("boxes", ()))
                        for frame in frames
                    ),
                    "extended_track_steps": len(
                        data.get("extended_detections_tracks") or ()
                    ),
                    "extended_traffic_light_steps": len(
                        data.get("extended_traffic_light_data") or ()
                    ),
                    "front_camera_path_declared": bool(camera.get("data_path")),
                    "front_camera_file_exists": synthetic_sensor_exists(
                        paths.synthetic_sensors, camera.get("data_path")
                    ),
                    "lidar_path_declared": bool(current.get("lidar_path")),
                    "lidar_file_exists": synthetic_sensor_exists(
                        paths.synthetic_sensors, current.get("lidar_path")
                    ),
                }
            )
        except Exception as error:
            failures.append(
                {"file": path.name, "error": f"{type(error).__name__}: {error}"}
            )
    frame = pd.DataFrame(rows)
    pair_position: list[float] = []
    pair_speed: list[float] = []
    for states in initial_states_by_original.values():
        for left in range(len(states)):
            for right in range(left + 1, len(states)):
                pair_position.append(
                    float(np.linalg.norm(states[left][:2] - states[right][:2]))
                )
                pair_speed.append(float(abs(states[left][2] - states[right][2])))
    write_dataframe(frame, args.output_dir / "synthetic_metadata_audit.csv")
    legal_train_deployed = (
        "navhard" not in str(root).lower() and "test" not in str(root).lower()
    )
    return {
        "available": bool(files),
        "root": str(root),
        "scene_file_count": len(files),
        "metadata_sample_count": len(frame),
        "metadata_failure_count": len(failures),
        "corresponding_original_scene_coverage": float(
            frame["corresponding_original_scene"].notna().mean()
        )
        if len(frame)
        else 0.0,
        "corresponding_original_initial_token_coverage": float(
            frame["corresponding_original_initial_token"].notna().mean()
        )
        if len(frame)
        else 0.0,
        "front_camera_declared_coverage": float(
            frame["front_camera_path_declared"].mean()
        )
        if len(frame)
        else 0.0,
        "front_camera_file_coverage": float(frame["front_camera_file_exists"].mean())
        if len(frame)
        else 0.0,
        "annotation_coverage": float((frame["frame_annotation_count"] > 0).mean())
        if len(frame)
        else 0.0,
        "extended_tracks_coverage": float((frame["extended_track_steps"] > 0).mean())
        if len(frame)
        else 0.0,
        "originals_with_multiple_followups_in_sample": sum(
            value > 1 for value in mapping_counts.values()
        ),
        "followup_pair_initial_position_difference_mean_m": float(
            np.mean(pair_position)
        )
        if pair_position
        else None,
        "followup_pair_initial_speed_difference_mean_mps": float(np.mean(pair_speed))
        if pair_speed
        else None,
        "legal_train_synthetic_root_deployed": legal_train_deployed,
        "training_eligible": legal_train_deployed,
        "scope_warning": (
            "Resolved data is NAVHARD/two-stage challenge data. Metadata was audited, but annotations are not used for training or supervision."
            if not legal_train_deployed
            else "Synthetic data is under a non-test path; split provenance still requires explicit approval before training."
        ),
        "same_current_state_claim_supported": False,
        "weak_supervision_interpretation": "At most neighborhood-state augmentation or weak multi-future supervision; not same-current-state, different-action ground truth.",
        "failures": failures,
        "code_paths": [
            "navsim/navsim/common/dataclasses.py",
            "navsim/navsim/common/dataloader.py",
        ],
    }


def render_report(result: dict[str, Any]) -> str:
    reactive = result["reactive"]
    synthetic = result["synthetic"]
    return "\n".join(
        [
            "# NAVSIM v2 Extension Audit",
            "",
            "## Reactive traffic policy",
            "",
            f"- Available: **{reactive.get('available')}**",
            f"- Official cached scope: **{reactive.get('official_cache_reactive_scene_count', 0)} scenes / {reactive.get('official_cache_reactive_candidate_count', 0)} candidates**",
            f"- Captured-track rerun: **{reactive.get('track_rerun_success_scenes', 0)} / {reactive.get('track_rerun_requested_scenes', 0)} scenes**",
            f"- Mean reactive-vs-replay actor endpoint change: **{reactive.get('actor_endpoint_change_mean_m')} m**",
            f"- Candidate-dependent vehicle response nonzero rate: **{reactive.get('candidate_dependent_endpoint_change_nonzero_rate')}**",
            "- NAVSIM IDM simulates `VEHICLE` only. Pedestrians and all remaining types are merged from logged replay by the abstract policy.",
            "",
            "These are reactive-policy simulated consequences, not observed multi-agent reactions and not causal counterfactuals.",
            "",
            "## Synthetic follow-up scenes",
            "",
            f"- Deployed: **{synthetic.get('available')}** at `{synthetic.get('root')}`",
            f"- Scene files: **{synthetic.get('scene_file_count', 0)}**; sampled metadata: **{synthetic.get('metadata_sample_count', 0)}**",
            f"- Legal train synthetic root deployed: **{synthetic.get('legal_train_synthetic_root_deployed')}**",
            f"- Camera file coverage in sample: **{synthetic.get('front_camera_file_coverage')}**",
            f"- Extended tracks coverage in sample: **{synthetic.get('extended_tracks_coverage')}**",
            "",
            synthetic.get("scope_warning", ""),
            "",
            synthetic.get("weak_supervision_interpretation", ""),
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, max_scenes=32)
    parser.add_argument("--skip-track-rerun", action="store_true")
    parser.add_argument("--synthetic-metadata-samples", type=int, default=128)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = discover_paths(
        args, split="trainval" if args.split == "train" else args.split
    )
    runtime = bootstrap_navsim(paths)
    result = {
        "runtime": runtime,
        "reactive": audit_reactive(args, paths),
        "synthetic": audit_synthetic(args, paths),
    }
    write_json(args.output_dir / "v2_extension_results.json", result)
    write_text(args.output_dir / "V2_EXTENSION_REPORT.md", render_report(result))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
