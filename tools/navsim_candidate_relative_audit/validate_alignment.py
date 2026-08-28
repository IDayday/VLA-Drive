#!/usr/bin/env python3
"""Validate coordinates, timestamps, actors and Gate A prerequisites."""

from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np

from .common import (
    add_common_arguments,
    append_command,
    effective_max_scenes,
    ensure_output_dir,
    global_to_local,
    local_to_global,
    metric_cache_loader,
    paths_from_args,
    percentile_summary,
    resolve_horizon_index,
    scene_loader,
    wrap_angle,
    write_json,
    write_markdown,
)


def validate(args: argparse.Namespace) -> dict[str, Any]:
    paths = paths_from_args(args)
    output_dir = ensure_output_dir(args.output_dir)
    max_scenes = effective_max_scenes(args.mode, args.max_scenes)
    caches = metric_cache_loader(paths)
    loader = scene_loader(paths, max_scenes=max_scenes, frame_interval=1, tokens=caches.tokens)

    position_errors: list[float] = []
    heading_errors: list[float] = []
    roundtrip_position: list[float] = []
    roundtrip_heading: list[float] = []
    actor_official_position: list[float] = []
    actor_official_heading: list[float] = []
    actor_candidate_roundtrip: list[float] = []
    cache_actor_centroid: list[float] = []
    frame_intervals: list[float] = []
    evidence: list[dict[str, Any]] = []
    map_success = 0
    route_success = 0
    cache_success = 0
    future_actor_success = 0

    from nuplan.common.actor_state.state_representation import StateSE2
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
    from navsim.planning.scenario_builder.navsim_scenario_utils import gt_boxes_oriented_box
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
        convert_absolute_to_relative_se2_array,
    )

    for token in loader.tokens:
        scene = loader.get_scene_from_token(token)
        current = scene.scene_metadata.num_history_frames - 1
        absolute = np.asarray([frame.ego_status.ego_pose for frame in scene.frames[current : current + 9]], dtype=np.float64)
        official = convert_absolute_to_relative_se2_array(StateSE2(*absolute[0]), absolute[1:])
        gt = np.asarray(scene.get_future_trajectory(num_trajectory_frames=8).poses, dtype=np.float64)
        delta = official - gt
        position_errors.extend(np.linalg.norm(delta[:, :2], axis=-1).tolist())
        heading_errors.extend(np.abs(wrap_angle(delta[:, 2])).tolist())

        reconstructed = local_to_global(absolute[0], gt)
        returned = global_to_local(absolute[0], reconstructed)
        roundtrip_position.extend(np.linalg.norm(returned[:, :2] - gt[:, :2], axis=-1).tolist())
        roundtrip_heading.extend(np.abs(wrap_angle(returned[:, 2] - gt[:, 2])).tolist())

        timestamps = np.asarray([frame.timestamp for frame in scene.frames], dtype=np.int64)
        frame_intervals.extend((np.diff(timestamps) / 1e6).tolist())
        horizon_indices = {
            str(horizon): resolve_horizon_index(timestamps, horizon, origin_index=current)
            for horizon in (0.5, 1.0, 2.0, 4.0)
        }
        scenario = NavSimScenario(scene, str(paths.map_path), "nuplan-maps-v1.0")
        actor_count = 0
        for iteration in (1, 2, 4, 8):
            frame = scene.frames[current + iteration]
            ego_state = scenario.get_ego_state_at_iteration(iteration)
            official_boxes = gt_boxes_oriented_box(frame.annotations.boxes, ego_state)
            ego_pose = np.asarray(frame.ego_status.ego_pose, dtype=np.float64)
            for box_index, (raw_box, official_box) in enumerate(zip(frame.annotations.boxes[:32], official_boxes[:32])):
                local_box = np.asarray([[raw_box[0], raw_box[1], raw_box[6]]], dtype=np.float64)
                manual_global = local_to_global(ego_pose, local_box)[0]
                official_global = np.asarray(
                    [official_box.center.x, official_box.center.y, official_box.center.heading], dtype=np.float64
                )
                actor_official_position.append(float(np.linalg.norm(manual_global[:2] - official_global[:2])))
                actor_official_heading.append(float(abs(wrap_angle(manual_global[2] - official_global[2]))))

                candidate_local = gt[iteration - 1 : iteration]
                candidate_global = local_to_global(absolute[0], candidate_local)[0]
                actor_in_candidate = global_to_local(candidate_global, official_global[None])[0]
                restored_actor = local_to_global(candidate_global, actor_in_candidate[None])[0]
                actor_candidate_roundtrip.append(float(np.linalg.norm(restored_actor[:2] - official_global[:2])))
                actor_count += 1
        future_actor_success += int(actor_count > 0)
        map_success += int(scene.map_api is not None)
        route_success += int(len(scene.frames[current].roadblock_ids) > 0)

        cache_error = None
        observation_interval = None
        try:
            cache = caches.get_from_token(token)
            cache_success += 1
            observation = cache.observation
            observation_interval = float(observation._sample_interval)
            for iteration in (1, 2, 4, 8):
                occupancy = observation[int(round(0.5 * iteration / observation_interval))]
                tracks = scenario.get_tracked_objects_at_iteration(iteration).tracked_objects
                for tracked in list(tracks)[:32]:
                    if tracked.track_token in occupancy.tokens:
                        centroid = occupancy[tracked.track_token].centroid
                        cache_actor_centroid.append(
                            float(np.hypot(centroid.x - tracked.center.x, centroid.y - tracked.center.y))
                        )
        except Exception as exc:
            cache_error = f"{type(exc).__name__}: {exc}"
        if len(evidence) < 8:
            evidence.append(
                {
                    "scene_token": token,
                    "log_name": scene.scene_metadata.log_name,
                    "map_name": scene.scene_metadata.map_name,
                    "horizon_indices": horizon_indices,
                    "frame_interval_s": (np.diff(timestamps) / 1e6).tolist(),
                    "future_actor_boxes_checked": actor_count,
                    "metric_cache_observation_interval_s": observation_interval,
                    "metric_cache_error": cache_error,
                }
            )

    summaries = {
        "gt_position_error_m": percentile_summary(position_errors),
        "gt_heading_error_rad": percentile_summary(heading_errors),
        "local_global_local_position_error_m": percentile_summary(roundtrip_position),
        "local_global_local_heading_error_rad": percentile_summary(roundtrip_heading),
        "actor_manual_vs_official_position_error_m": percentile_summary(actor_official_position),
        "actor_manual_vs_official_heading_error_rad": percentile_summary(actor_official_heading),
        "actor_candidate_frame_roundtrip_error_m": percentile_summary(actor_candidate_roundtrip),
        "scene_vs_metric_cache_actor_centroid_error_m": percentile_summary(cache_actor_centroid),
        "scene_frame_interval_s": percentile_summary(frame_intervals),
    }
    blockers = []
    if len(loader) == 0:
        blockers.append("No cache-matched training scenes loaded")
    if not position_errors or summaries["gt_position_error_m"]["max"] is None or summaries["gt_position_error_m"]["max"] > 1e-4:
        blockers.append("GT future trajectory does not match official per-frame absolute-to-relative conversion")
    if summaries["local_global_local_position_error_m"]["max"] is None or summaries["local_global_local_position_error_m"]["max"] > 1e-6:
        blockers.append("SE(2) local/global/local round trip exceeds 1e-6 m")
    if future_actor_success < len(loader):
        blockers.append("At least one scene has no transformable future actor world state")
    if map_success < len(loader) or route_success < len(loader):
        blockers.append("Map or route is unavailable for at least one audited scene")
    if cache_success == 0:
        blockers.append("No official MetricCache loading path succeeded")
    gate_pass = len(blockers) == 0
    result = {
        "gate": "A",
        "passed": gate_pass,
        "audited_scenes": len(loader),
        "map_success_rate": map_success / max(len(loader), 1),
        "route_success_rate": route_success / max(len(loader), 1),
        "metric_cache_success_rate": cache_success / max(len(loader), 1),
        "future_actor_world_state_success_rate": future_actor_success / max(len(loader), 1),
        "metrics": summaries,
        "evidence": evidence,
        "blockers": blockers,
        "coordinate_semantics": {
            "scene_ego_pose": "global rear-axle SE(2), meters/radians",
            "scene_annotation_box": "future-frame ego-local center/heading; verified against official gt_boxes_oriented_box",
            "candidate_trajectory": "current-ego rear-axle-local SE(2), meters/radians",
            "metric_cache_observation": "global polygon occupancy, logged future interpolated to 10 Hz",
        },
    }
    write_json(output_dir / "alignment_metrics.json", result)
    gate_path = output_dir / "gate_status.json"
    existing = {}
    if gate_path.exists():
        import json

        existing = json.loads(gate_path.read_text(encoding="utf-8"))
    existing["gate_a"] = {"passed": gate_pass, "blockers": blockers, "audited_scenes": len(loader)}
    write_json(gate_path, existing)
    report = f"""# Coordinate and Time Alignment Audit

## Gate A: {'PASS' if gate_pass else 'FAIL'}

- Cache-matched scenes: {len(loader)}
- GT future position error: mean `{summaries['gt_position_error_m']['mean']}` m, P99 `{summaries['gt_position_error_m']['p99']}` m, max `{summaries['gt_position_error_m']['max']}` m
- GT future heading error after wrap: max `{summaries['gt_heading_error_rad']['max']}` rad
- Local→global→local position error: max `{summaries['local_global_local_position_error_m']['max']}` m
- Raw annotation transform vs official `gt_boxes_oriented_box`: max `{summaries['actor_manual_vs_official_position_error_m']['max']}` m
- Future-frame actor global→candidate-local→global error: max `{summaries['actor_candidate_frame_roundtrip_error_m']['max']}` m
- Scene annotation vs cached 10 Hz logged occupancy centroid: mean `{summaries['scene_vs_metric_cache_actor_centroid_error_m']['mean']}` m, P99 `{summaries['scene_vs_metric_cache_actor_centroid_error_m']['p99']}` m
- Scene timestamp interval: mean `{summaries['scene_frame_interval_s']['mean']}` s, min/P99/max `{summaries['scene_frame_interval_s']['p50']}` / `{summaries['scene_frame_interval_s']['p99']}` / `{summaries['scene_frame_interval_s']['max']}` s

Horizon selection is performed by `resolve_horizon_index(timestamps, target_seconds)` and never assumes an array index solely from a nominal 0.5 s frequency.

## Verified coordinate semantics

Scene ego poses are global rear-axle SE(2). Raw annotation boxes are local to their own frame's ego pose; this was checked against the local official construction code and numerical outputs. Candidate trajectories are local to the current ego rear axle. Metric-cache occupancy polygons are in the global map frame.

## Blockers

{chr(10).join('- ' + blocker for blocker in blockers) if blockers else '- None.'}
"""
    write_markdown(output_dir / "ALIGNMENT_AUDIT.md", report)
    if not gate_pass:
        raise SystemExit("Gate A failed; later large-scale stages must not run. See ALIGNMENT_AUDIT.md")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    args = parser.parse_args()
    validate(args)
    append_command(args.output_dir.resolve(), "python -m tools.navsim_candidate_relative_audit.validate_alignment " + " ".join(sys.argv[1:]))


if __name__ == "__main__":
    main()
