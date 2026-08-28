#!/usr/bin/env python3
"""Score randomized candidates and build leakage-audited Gate C v3 targets."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tools.navsim_candidate_relative_audit.build_candidate_relative_targets import build_scene_targets
from tools.navsim_candidate_relative_audit.common import load_scenes_for_tokens, metric_cache_loader, stable_hash
from tools.navsim_candidate_relative_audit.score_candidates import _rows_from_score, score_pose_batch

from .common import (
    DEFAULT_CACHE_DIR,
    DEFAULT_REPORT_DIR,
    append_command,
    ensure_dir,
    navsim_paths,
    require_gate,
    update_gate,
    write_json,
    write_markdown,
    write_parquet,
)


HORIZONS = np.arange(0.5, 4.0 + 1e-8, 0.5, dtype=np.float32)
K_EXACT_NAMES = (
    "speed_mps",
    "acceleration_mps2",
    "yaw_rate_radps",
    "curvature_1pm",
    "jerk_mps3",
    "terminal_displacement_m",
)
S_STATIC_NAMES = (
    "drivable_area_valid",
    "oncoming_lane",
    "intersection",
    "centerline_lateral_offset_m",
    "centerline_heading_error_rad",
    "route_progress_m",
)
D_STATE_SUMMARY_NAMES = (
    "min_actor_polygon_clearance_m",
    "min_actor_center_distance_m",
    "candidate_corridor_actor_count",
    "nearest_actor_relative_x_m",
    "nearest_actor_relative_y_m",
)
D_STATE_ACTOR_NAMES = (
    "object_type_id",
    "relative_x_m",
    "relative_y_m",
    "relative_vx_mps",
    "relative_vy_mps",
    "relative_heading_rad",
    "length_m",
    "width_m",
    "polygon_clearance_m",
    "in_candidate_corridor",
)
D_RISK_NAMES = (
    "any_actor_collision",
    "instantaneous_min_ttc_s",
    "soft_collision_probability",
    "min_dynamic_clearance_m",
    "risk_onset_time_s",
)
D_SIGNAL_NAMES = (
    "red_light_polygon_clearance_m",
    "red_light_polygon_intersection",
    "future_signal_valid_mask",
)
SHARED_ACTOR_NAMES = (
    "object_type_id",
    "current_ego_x_m",
    "current_ego_y_m",
    "current_ego_vx_mps",
    "current_ego_vy_mps",
    "current_ego_heading_rad",
    "length_m",
    "width_m",
)
CURRENT_ACTOR_NAMES = SHARED_ACTOR_NAMES


def _poses(group: pd.DataFrame) -> np.ndarray:
    ordered = group.sort_values("candidate_index")
    return np.stack(
        [
            np.column_stack([row.pose_x_m, row.pose_y_m, row.pose_heading_rad])
            for row in ordered.itertuples(index=False)
        ],
        axis=0,
    ).astype(np.float32)


def score_manifest(manifest: pd.DataFrame, paths: Any, sanity_scenes: int = 2) -> tuple[pd.DataFrame, dict[str, Any]]:
    caches = metric_cache_loader(paths)
    rows: list[dict[str, Any]] = []
    deterministic_errors: list[float] = []
    order_ok: list[bool] = []
    failures: list[dict[str, str]] = []
    for scene_index, (token, group) in enumerate(manifest.groupby("scene_token", sort=False)):
        started = time.perf_counter()
        try:
            poses = _poses(group)
            score = score_pose_batch(caches.get_from_token(token), poses)
            runtime = time.perf_counter() - started
            scene_rows = _rows_from_score(group, score, runtime)
            rows.extend(scene_rows)
            order_ok.append(
                [row["candidate_index"] for row in scene_rows]
                == group.sort_values("candidate_index").candidate_index.tolist()
            )
            if scene_index < sanity_scenes:
                repeated = score_pose_batch(caches.get_from_token(token), poses)
                deterministic_errors.extend(
                    [
                        float(np.max(np.abs(score["score"] - repeated["score"]))),
                        float(np.max(np.abs(score["simulated_states"] - repeated["simulated_states"]))),
                    ]
                )
        except Exception as exc:
            failures.append({"scene_token": token, "error": f"{type(exc).__name__}: {exc}"})
            for row in group.sort_values("candidate_index").to_dict("records"):
                row.update(
                    traffic_policy="non_reactive",
                    scoring_success=False,
                    scoring_error=failures[-1]["error"],
                    aggregate_score=np.nan,
                )
                rows.append(row)
    metrics = pd.DataFrame(rows)
    if "scoring_success" not in metrics:
        metrics["scoring_success"] = True
    success_rate = float(metrics.scoring_success.mean()) if len(metrics) else 0.0
    summary = {
        "scene_count": int(metrics.scene_token.nunique()) if len(metrics) else 0,
        "candidate_count": int(len(metrics)),
        "success_rate": success_rate,
        "candidate_order_preserved": all(order_ok),
        "deterministic_max_abs_error": max(deterministic_errors, default=None),
        "failure_examples": failures[:10],
        "traffic_policy": "non_reactive",
    }
    if success_rate <= 0.98 or not all(order_ok) or max(deterministic_errors, default=0.0) > 1e-12:
        raise RuntimeError(f"Gate C candidate scoring failed: {summary}")
    return metrics, summary


def current_actor_slots(
    scene: Any,
    paths: Any,
    actor_slots: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode current dynamic actors in the current-ego frame."""

    from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario

    scenario = NavSimScenario(scene, str(paths.map_path), "nuplan-maps-v1.0")
    current_index = int(scene.scene_metadata.num_history_frames) - 1
    current_pose = np.asarray(scene.frames[current_index].ego_status.ego_pose, dtype=np.float64)
    current_tracks = [
        track for track in scenario.get_tracked_objects_at_iteration(0).tracked_objects
        if track.tracked_object_type in AGENT_TYPES
    ]
    current_tracks.sort(
        key=lambda track: (
            float(np.hypot(track.center.x - current_pose[0], track.center.y - current_pose[1])),
            stable_hash(str(track.track_token)),
        )
    )
    values = np.zeros((actor_slots, len(CURRENT_ACTOR_NAMES)), dtype=np.float32)
    mask = np.zeros(actor_slots, dtype=bool)
    token_hashes = np.zeros(actor_slots, dtype=np.int64)
    c, s = math.cos(current_pose[2]), math.sin(current_pose[2])
    for slot, track in enumerate(current_tracks[:actor_slots]):
        dx = float(track.center.x - current_pose[0])
        dy = float(track.center.y - current_pose[1])
        velocity = getattr(track, "velocity", None)
        vx = float(velocity.x) if velocity is not None else 0.0
        vy = float(velocity.y) if velocity is not None else 0.0
        values[slot] = [
            int(track.tracked_object_type.value),
            c * dx + s * dy,
            -s * dx + c * dy,
            c * vx + s * vy,
            -s * vx + c * vy,
            (float(track.center.heading) - current_pose[2] + np.pi) % (2 * np.pi) - np.pi,
            float(track.box.length),
            float(track.box.width),
        ]
        mask[slot] = True
        token_hashes[slot] = stable_hash(str(track.track_token))
    return values, mask, token_hashes


def _fixed_current_actor_slots(
    scene: Any,
    old: dict[str, np.ndarray],
    paths: Any,
    actor_slots: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Realign shared logged future to actor slots selected at current time."""

    current_index = int(scene.scene_metadata.num_history_frames) - 1
    current_pose = np.asarray(scene.frames[current_index].ego_status.ego_pose, dtype=np.float64)
    current_values, current_mask, slot_hashes = current_actor_slots(scene, paths, actor_slots)

    values = np.zeros((len(HORIZONS), actor_slots, len(SHARED_ACTOR_NAMES)), dtype=np.float32)
    mask = np.zeros((len(HORIZONS), actor_slots), dtype=bool)
    new_actor_count = np.zeros(len(HORIZONS), dtype=np.int16)
    c, s = math.cos(current_pose[2]), math.sin(current_pose[2])
    slot_set = {int(value) for value in slot_hashes if value != 0}
    shared = old["shared_logged_future_actor"]
    shared_mask = old["shared_logged_future_actor_mask"]
    shared_hash = old["shared_logged_future_actor_token_hash"]
    for horizon in range(len(HORIZONS)):
        lookup = {
            int(token): shared[horizon, index]
            for index, token in enumerate(shared_hash[horizon])
            if shared_mask[horizon, index]
        }
        future_set = set(lookup)
        new_actor_count[horizon] = len(future_set - slot_set)
        for slot, token in enumerate(slot_hashes):
            record = lookup.get(int(token))
            if record is None:
                continue
            dx, dy = float(record[1] - current_pose[0]), float(record[2] - current_pose[1])
            local_x = c * dx + s * dy
            local_y = -s * dx + c * dy
            local_vx = c * float(record[3]) + s * float(record[4])
            local_vy = -s * float(record[3]) + c * float(record[4])
            local_heading = (float(record[5]) - current_pose[2] + np.pi) % (2 * np.pi) - np.pi
            values[horizon, slot] = [
                record[0], local_x, local_y, local_vx, local_vy,
                local_heading, record[6], record[7],
            ]
            mask[horizon, slot] = True
    return values, mask, slot_hashes, new_actor_count, current_values, current_mask


def split_v3_targets(
    scene: Any,
    old: dict[str, np.ndarray],
    paths: Any,
    actor_slots: int = 16,
) -> dict[str, np.ndarray]:
    trajectory = old["trajectory_derived"].astype(np.float32)
    env = old["C_environment_only"].astype(np.float32)
    k_exact = trajectory[..., 3:9]
    s_static = env[..., 5:11]
    d_state_summary = env[..., [0, 1, 2, 13, 14]]
    clearance = env[..., 0]
    collision = env[..., 3]
    ttc = np.clip(env[..., 4], 0.0, 10.0)
    soft_collision = 1.0 / (1.0 + np.exp(np.clip(clearance / 0.75, -30.0, 30.0)))
    onset = np.full((len(env),), 10.0, dtype=np.float32)
    risky = (collision > 0.5) | (ttc < 3.0) | (clearance < 1.0)
    for candidate in range(len(env)):
        indices = np.flatnonzero(risky[candidate])
        if len(indices):
            onset[candidate] = HORIZONS[indices[0]]
    d_risk = np.stack(
        [
            collision,
            ttc,
            soft_collision,
            clearance,
            np.broadcast_to(onset[:, None], clearance.shape),
        ],
        axis=-1,
    ).astype(np.float32)
    signal_mask = np.ones(env.shape[:2], dtype=np.float32)
    d_signal = np.stack([env[..., 11], env[..., 12], signal_mask], axis=-1).astype(np.float32)
    shared, shared_mask, slot_hash, new_actor_count, current_actor, current_actor_mask = _fixed_current_actor_slots(
        scene, old, paths, actor_slots,
    )
    return {
        "time_s": old["time_s"].astype(np.float32),
        "candidate_index": old["candidate_index"].astype(np.int16),
        "is_gt": old["is_gt"].astype(bool),
        "K_exact": k_exact,
        "S_static": s_static,
        "D_state_summary": d_state_summary.astype(np.float32),
        "D_state_actor": old["candidate_relative_actor"].astype(np.float32),
        "D_state_actor_mask": old["candidate_relative_actor_mask"].astype(bool),
        "D_state_actor_token_hash": old["candidate_relative_actor_token_hash"].astype(np.int64),
        "D_risk": d_risk,
        "D_signal": d_signal,
        "D_all_summary": np.concatenate([d_state_summary, d_risk, d_signal], axis=-1).astype(np.float32),
        "shared_actor_future_current_ego": shared,
        "shared_actor_future_mask": shared_mask,
        "shared_actor_slot_token_hash": slot_hash,
        "shared_new_actor_count": new_actor_count,
        "current_scene_features": old["current_scene_features"].astype(np.float32),
        "current_actor_state": current_actor,
        "current_actor_mask": current_actor_mask,
    }


def target_schema_v3(actor_slots: int) -> dict[str, Any]:
    common_offline = {
        "depends_on_logged_future": True,
        "available_as_ground_truth_at_inference": False,
        "training_only_target": True,
    }
    return {
        "schema_version": "3.0.0",
        "horizons_s": HORIZONS.tolist(),
        "traffic_policy": "non_reactive logged future",
        "inference_input_contract": {
            "allowed": ["current images/history used by EpisodeDrive", "current ego state", "navigation command", "candidate trajectory"],
            "forbidden": ["future image", "future annotation", "logged future", "future GT trajectory", "official score/factors"],
        },
        "groups": {
            "K_exact": {
                "fields": K_EXACT_NAMES,
                "shape": ["K", 8, len(K_EXACT_NAMES)],
                "candidate_direct": True,
                "depends_on_static_map": False,
                "depends_on_logged_future": False,
                "inference_available": True,
                "official_metric_proxy": False,
                "training_only": False,
            },
            "current_actor_state": {
                "fields": CURRENT_ACTOR_NAMES,
                "shape": [actor_slots, len(CURRENT_ACTOR_NAMES)],
                "coordinate_frame": "current ego frame",
                "candidate_direct": False,
                "depends_on_static_map": False,
                "depends_on_logged_future": False,
                "inference_available": False,
                "oracle_only_reason": (
                    "structured current annotations are not direct EpisodeDrive inputs; "
                    "the deployable model must infer equivalent information from current images"
                ),
                "official_metric_proxy": False,
                "training_only": False,
                "valid_mask": "current_actor_mask",
            },
            "S_static": {
                "fields": S_STATIC_NAMES,
                "shape": ["K", 8, len(S_STATIC_NAMES)],
                "candidate_direct": False,
                "depends_on_static_map": True,
                "depends_on_logged_future": False,
                "inference_available": False,
                "oracle_only_reason": "EpisodeDrive baseline does not consume HD map geometry",
                "official_metric_proxy": True,
                "training_only": True,
            },
            "D_state": {
                "summary_fields": D_STATE_SUMMARY_NAMES,
                "actor_fields": D_STATE_ACTOR_NAMES,
                "summary_shape": ["K", 8, len(D_STATE_SUMMARY_NAMES)],
                "actor_shape": ["K", 8, actor_slots, len(D_STATE_ACTOR_NAMES)],
                "candidate_direct": False,
                "depends_on_static_map": False,
                **common_offline,
                "inference_available": "predicted only",
                "official_metric_proxy": False,
            },
            "D_risk": {
                "fields": D_RISK_NAMES,
                "shape": ["K", 8, len(D_RISK_NAMES)],
                "candidate_direct": False,
                "depends_on_static_map": False,
                **common_offline,
                "inference_available": "predicted/recomputed only",
                "official_metric_proxy": True,
                "note": "physical geometry/TTC labels; no official aggregate or factor score",
            },
            "D_signal": {
                "fields": D_SIGNAL_NAMES,
                "shape": ["K", 8, len(D_SIGNAL_NAMES)],
                "candidate_direct": False,
                "depends_on_static_map": True,
                **common_offline,
                "inference_available": "predicted only",
                "official_metric_proxy": True,
            },
            "shared_actor_future": {
                "fields": SHARED_ACTOR_NAMES,
                "shape": [8, actor_slots, len(SHARED_ACTOR_NAMES)],
                "coordinate_frame": "current ego frame",
                "slot_assignment": "current-time actor distance then stable token hash",
                "candidate_direct": False,
                "depends_on_static_map": False,
                **common_offline,
                "inference_available": "predicted once only",
                "new_actor_representation": "shared_new_actor_count residual",
                "official_metric_proxy": False,
            },
        },
        "explicit_exclusions": [
            "candidate future x/y/heading copies from consequence groups",
            "candidate index/type/family",
            "official PDM aggregate score",
            "official PDM factor columns",
            "GT future image for non-GT candidates",
        ],
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    require_gate(args.output_dir, "gate_c0")
    paths = navsim_paths(args.split)
    report_dir = ensure_dir(args.output_dir)
    cache_dir = ensure_dir(args.cache_dir)
    manifest_path = args.manifest or (cache_dir / "controlled_candidate_manifest.parquet")
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = pd.read_parquet(manifest_path)
    if args.num_scenes > 0:
        tokens = manifest.scene_token.drop_duplicates().head(args.num_scenes)
        manifest = manifest[manifest.scene_token.isin(set(tokens))].copy()
    metrics_path = cache_dir / "candidate_metrics.parquet"
    if metrics_path.exists() and not args.force:
        metrics = pd.read_parquet(metrics_path)
        expected = set(manifest.scene_token)
        if set(metrics.scene_token) != expected or len(metrics) != len(manifest):
            raise RuntimeError("Existing candidate metrics do not match requested manifest; use --force")
        scoring_summary = {
            "scene_count": int(metrics.scene_token.nunique()),
            "candidate_count": int(len(metrics)),
            "success_rate": float(metrics.scoring_success.mean()),
            "reused": True,
        }
    else:
        metrics, scoring_summary = score_manifest(manifest, paths, args.sanity_scenes)
        write_parquet(metrics, metrics_path)
    write_json(report_dir / "candidate_scoring_summary.json", scoring_summary)

    successful = metrics[metrics.scoring_success].copy()
    tokens = successful.scene_token.drop_duplicates().tolist()
    loader = load_scenes_for_tokens(paths, tokens)
    caches = metric_cache_loader(paths)
    target_dir = ensure_dir(cache_dir / "targets_v3")
    coverage_rows: list[dict[str, Any]] = []
    for token, group in successful.groupby("scene_token", sort=False):
        target_path = target_dir / f"{token}.npz"
        if target_path.exists() and not args.force:
            with np.load(target_path, allow_pickle=False) as existing:
                valid = existing["D_state_actor_mask"].mean()
            coverage_rows.append({"scene_token": token, "success": True, "actor_valid_rate": float(valid), "reused": True})
            continue
        try:
            scene = loader.get_scene_from_token(token)
            old = build_scene_targets(scene, caches.get_from_token(token), group, paths, 16, 64)
            arrays = split_v3_targets(scene, old, paths, args.actor_slots)
            np.savez_compressed(target_path, **arrays)
            coverage_rows.append(
                {
                    "scene_token": token,
                    "log_name": str(group.log_name.iloc[0]),
                    "success": True,
                    "actor_valid_rate": float(arrays["D_state_actor_mask"].mean()),
                    "shared_actor_valid_rate": float(arrays["shared_actor_future_mask"].mean()),
                    "new_actor_mean": float(arrays["shared_new_actor_count"].mean()),
                    "reused": False,
                }
            )
        except Exception as exc:
            coverage_rows.append(
                {
                    "scene_token": token,
                    "log_name": str(group.log_name.iloc[0]),
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "reused": False,
                }
            )
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(report_dir / "target_coverage_v3.csv", index=False)
    schema = target_schema_v3(args.actor_slots)
    schema["actual_scene_success_rate"] = float(coverage.success.mean()) if len(coverage) else 0.0
    schema["target_cache_dir"] = str(target_dir)
    write_json(report_dir / "target_schema_v3.json", schema)
    success = float(coverage.success.mean()) if len(coverage) else 0.0
    result = {
        "scoring": scoring_summary,
        "target_scene_count": int(len(coverage)),
        "target_success_rate": success,
        "target_cache_dir": str(target_dir),
        "schema_version": schema["schema_version"],
    }
    write_json(report_dir / "target_build_summary.json", result)
    write_markdown(
        report_dir / "TARGET_V3_REPORT.md",
        f"""# Gate C v3 Target Construction

- Candidate scoring: {scoring_summary.get('candidate_count', 0)} rows, {scoring_summary.get('success_rate', 0):.3%} success
- v3 targets: {int(coverage.success.sum())}/{len(coverage)} scenes ({success:.3%})
- Shared actor frame: current ego frame; {args.actor_slots} slots fixed from current-time distance ordering
- Target cache: `{target_dir}` (not committed)

`K_exact`, `S_static`, `D_state`, `D_risk` and `D_signal` are stored separately.
Official aggregate/factor scores remain only in the offline evaluation parquet and
are explicitly absent from model-input target files. All dynamic labels are
candidate-conditioned relabeling of one shared logged future under the
non-reactive assumption.
""",
    )
    update_gate(report_dir, "target_v3", {"passed": success > 0.98, **result})
    if success <= 0.98:
        raise SystemExit("v3 target coverage is below 98%")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "trainval"), default="trainval")
    parser.add_argument("--num-scenes", type=int, default=0)
    parser.add_argument("--actor-slots", type=int, default=16)
    parser.add_argument("--sanity-scenes", type=int, default=2)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    args = parser.parse_args()
    build(args)
    append_command(args.output_dir, "python -m tools.shared_future_candidate_consequence.build_gate_c_targets " + " ".join(sys.argv[1:]))


if __name__ == "__main__":
    main()
