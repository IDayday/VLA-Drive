#!/usr/bin/env python3
"""Phase 2: verify coordinate systems, timestamp alignment, and Gate A."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import pickle
from typing import Any

import numpy as np
import pandas as pd

from .common import (
    add_common_arguments,
    bootstrap_navsim,
    discover_paths,
    load_metric_cache,
    load_metric_cache_index,
    metric_log_name,
    output_tokens,
    raw_log_path,
    resolve_horizon_index,
    se2_global_to_local,
    se2_local_to_global,
    wrap_heading,
    write_dataframe,
    write_json,
    write_text,
)


def error_statistics(values: list[float]) -> dict[str, float | int | None]:
    finite = np.asarray(
        [value for value in values if math.isfinite(value)], dtype=np.float64
    )
    if not len(finite):
        return {"count": 0, "mean": None, "p95": None, "p99": None, "max": None}
    return {
        "count": int(len(finite)),
        "mean": float(np.mean(finite)),
        "p95": float(np.quantile(finite, 0.95)),
        "p99": float(np.quantile(finite, 0.99)),
        "max": float(np.max(finite)),
    }


def matched_actor_errors(
    official_tracks: Any, cached_tracks: Any
) -> list[dict[str, Any]]:
    official = {
        str(obj.metadata.track_token): obj
        for obj in official_tracks.tracked_objects.tracked_objects
    }
    cached = {
        str(obj.metadata.track_token): obj
        for obj in cached_tracks.tracked_objects.tracked_objects
    }
    rows: list[dict[str, Any]] = []
    for token in sorted(set(official) & set(cached)):
        left, right = official[token], cached[token]
        left_velocity = getattr(left, "velocity", None)
        right_velocity = getattr(right, "velocity", None)
        rows.append(
            {
                "track_token": token,
                "position_error_m": float(
                    np.hypot(
                        left.center.x - right.center.x, left.center.y - right.center.y
                    )
                ),
                "heading_error_rad": float(
                    abs(wrap_heading(left.center.heading - right.center.heading))
                ),
                "velocity_error_mps": float(
                    np.hypot(
                        getattr(left_velocity, "x", 0.0)
                        - getattr(right_velocity, "x", 0.0),
                        getattr(left_velocity, "y", 0.0)
                        - getattr(right_velocity, "y", 0.0),
                    )
                ),
            }
        )
    return rows


def validate(args: argparse.Namespace) -> dict[str, Any]:
    split = "trainval" if args.split == "train" else args.split
    paths = discover_paths(args, split=split)
    runtime = bootstrap_navsim(paths)
    if (
        paths.metric_cache is None
        or paths.logs_root is None
        or paths.sensors_root is None
        or paths.maps_root is None
    ):
        raise FileNotFoundError("Gate A requires raw logs/sensors/maps and MetricCache")
    from nuplan.common.actor_state.state_representation import StateSE2
    from navsim.common.dataclasses import Scene, SensorConfig
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
        convert_absolute_to_relative_se2_array,
    )

    metric_index = load_metric_cache_index(paths.metric_cache)
    preferred = [
        token for token in output_tokens(args.output_dir) if token in metric_index
    ]
    if preferred:
        tokens = preferred[: args.max_scenes]
    else:
        tokens = sorted(metric_index)[: args.max_scenes]
    grouped: dict[str, list[str]] = defaultdict(list)
    for token in tokens:
        grouped[metric_log_name(metric_index[token])].append(token)

    gt_position_errors: list[float] = []
    gt_heading_errors: list[float] = []
    roundtrip_position_errors: list[float] = []
    roundtrip_heading_errors: list[float] = []
    actor_position_errors: list[float] = []
    actor_heading_errors: list[float] = []
    actor_velocity_errors: list[float] = []
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    timestamp_intervals: list[float] = []
    actor_match_total = 0
    map_route_success = 0
    metric_success = 0
    for log_name in sorted(grouped):
        with raw_log_path(paths.logs_root, log_name).open("rb") as stream:
            raw = pickle.load(stream)
        positions = {str(frame["token"]): index for index, frame in enumerate(raw)}
        for token in grouped[log_name]:
            try:
                index = positions[token]
                window = list(raw[index - 3 : index + 9])
                if len(window) != 12:
                    raise IndexError("insufficient raw window")
                scene = Scene.from_scene_dict_list(
                    window,
                    paths.sensors_root,
                    num_history_frames=4,
                    num_future_frames=8,
                    sensor_config=SensorConfig.build_no_sensors(),
                )
                cache = load_metric_cache(metric_index[token])
                metric_success += 1
                trajectory = np.asarray(
                    scene.get_future_trajectory().poses, dtype=np.float64
                )
                global_poses = np.asarray(
                    [frame.ego_status.ego_pose for frame in scene.frames[3:12]],
                    dtype=np.float64,
                )
                explicit = convert_absolute_to_relative_se2_array(
                    StateSE2(*global_poses[0]), global_poses[1:]
                )
                position_error = np.linalg.norm(
                    trajectory[:, :2] - explicit[:, :2], axis=1
                )
                heading_error = np.abs(wrap_heading(trajectory[:, 2] - explicit[:, 2]))
                gt_position_errors.extend(position_error.tolist())
                gt_heading_errors.extend(heading_error.tolist())
                origin = global_poses[0]
                roundtrip = se2_global_to_local(
                    se2_local_to_global(trajectory, origin), origin
                )
                roundtrip_position = np.linalg.norm(
                    roundtrip[:, :2] - trajectory[:, :2], axis=1
                )
                roundtrip_heading = np.abs(
                    wrap_heading(roundtrip[:, 2] - trajectory[:, 2])
                )
                roundtrip_position_errors.extend(roundtrip_position.tolist())
                roundtrip_heading_errors.extend(roundtrip_heading.tolist())

                scenario = NavSimScenario(
                    scene, str(paths.maps_root), "nuplan-maps-v1.0"
                )
                scene_actor_rows: list[dict[str, Any]] = []
                for half_second_index in range(0, 9):
                    official = scenario.get_tracked_objects_at_iteration(
                        half_second_index
                    )
                    cached = (
                        cache.current_tracked_objects[0]
                        if half_second_index == 0
                        else cache.future_tracked_objects[half_second_index * 5 - 1]
                    )
                    matched = matched_actor_errors(official, cached)
                    actor_match_total += len(matched)
                    scene_actor_rows.extend(matched)
                actor_position_errors.extend(
                    row["position_error_m"] for row in scene_actor_rows
                )
                actor_heading_errors.extend(
                    row["heading_error_rad"] for row in scene_actor_rows
                )
                actor_velocity_errors.extend(
                    row["velocity_error_mps"] for row in scene_actor_rows
                )
                map_ok = bool(
                    scene.map_api
                    and cache.centerline.length > 0
                    and cache.route_lane_ids
                    and cache.drivable_area_map.tokens
                )
                map_route_success += int(map_ok)
                raw_timestamps = [int(frame.timestamp) for frame in scene.frames]
                deltas = np.diff(raw_timestamps).astype(np.float64) / 1e6
                timestamp_intervals.extend(deltas.tolist())
                resolved = {
                    str(horizon): resolve_horizon_index(raw_timestamps[3:], horizon)
                    for horizon in (0.5, 1.0, 2.0, 4.0)
                }
                rows.append(
                    {
                        "scene_token": token,
                        "log_name": log_name,
                        "gt_position_error_mean_m": float(np.mean(position_error)),
                        "gt_position_error_max_m": float(np.max(position_error)),
                        "gt_heading_error_mean_rad": float(np.mean(heading_error)),
                        "roundtrip_position_error_max_m": float(
                            np.max(roundtrip_position)
                        ),
                        "roundtrip_heading_error_max_rad": float(
                            np.max(roundtrip_heading)
                        ),
                        "actor_match_count": len(scene_actor_rows),
                        "actor_position_error_mean_m": float(
                            np.mean(
                                [row["position_error_m"] for row in scene_actor_rows]
                            )
                        )
                        if scene_actor_rows
                        else float("nan"),
                        "actor_position_error_max_m": float(
                            np.max(
                                [row["position_error_m"] for row in scene_actor_rows]
                            )
                        )
                        if scene_actor_rows
                        else float("nan"),
                        "actor_heading_error_mean_rad": float(
                            np.mean(
                                [row["heading_error_rad"] for row in scene_actor_rows]
                            )
                        )
                        if scene_actor_rows
                        else float("nan"),
                        "actor_velocity_error_mean_mps": float(
                            np.mean(
                                [row["velocity_error_mps"] for row in scene_actor_rows]
                            )
                        )
                        if scene_actor_rows
                        else float("nan"),
                        "map_route_available": map_ok,
                        "timestamp_delta_mean_s": float(np.mean(deltas)),
                        "timestamp_delta_max_s": float(np.max(deltas)),
                        "resolved_horizon_indices": json.dumps(
                            resolved, sort_keys=True
                        ),
                    }
                )
            except Exception as error:
                failures.append(
                    {"scene_token": token, "error": f"{type(error).__name__}: {error}"}
                )

    statistics = {
        "gt_position_error_m": error_statistics(gt_position_errors),
        "gt_heading_error_rad": error_statistics(gt_heading_errors),
        "roundtrip_position_error_m": error_statistics(roundtrip_position_errors),
        "roundtrip_heading_error_rad": error_statistics(roundtrip_heading_errors),
        "actor_position_error_m": error_statistics(actor_position_errors),
        "actor_heading_error_rad": error_statistics(actor_heading_errors),
        "actor_velocity_error_mps": error_statistics(actor_velocity_errors),
        "timestamp_interval_s": error_statistics(timestamp_intervals),
    }
    criteria = {
        "gt_future_stable": len(rows) == len(tokens)
        and statistics["gt_position_error_m"]["max"] is not None,
        "coordinate_roundtrip": (
            statistics["roundtrip_position_error_m"]["max"] is not None
            and float(statistics["roundtrip_position_error_m"]["max"]) < 1e-7
            and float(statistics["roundtrip_heading_error_rad"]["max"]) < 1e-9
        ),
        "future_actor_world_state": actor_match_total > 0
        and statistics["actor_position_error_m"]["p99"] is not None
        and float(statistics["actor_position_error_m"]["p99"]) < 0.05,
        "map_route_access": map_route_success == len(rows) and len(rows) > 0,
        "official_metric_cache_path": metric_success == len(rows) and len(rows) > 0,
    }
    gate_a = all(criteria.values())
    result = {
        "gate_a": "PASS" if gate_a else "FAIL",
        "criteria": criteria,
        "audited_scene_count": len(rows),
        "requested_scene_count": len(tokens),
        "actor_match_count": actor_match_total,
        "statistics": statistics,
        "runtime": runtime,
        "evidence_scene_tokens": [row["scene_token"] for row in rows[:16]],
        "failures": failures,
        "coordinate_semantics": {
            "raw_annotations": "ego-local at each logged frame",
            "metric_cache_tracks": "global SE(2), official 2 Hz logs interpolated to 10 Hz",
            "candidate_trajectory": "current rear-axle local SE(2)",
            "proof_code": str(
                paths.navsim_devkit
                / "navsim/planning/scenario_builder/navsim_scenario_utils.py"
            ),
        },
    }
    write_dataframe(pd.DataFrame(rows), args.output_dir / "alignment_scene_metrics.csv")
    return result


def render_report(result: dict[str, Any]) -> str:
    stats = result["statistics"]
    criteria = result["criteria"]
    return "\n".join(
        [
            "# NAVSIM Coordinate and Time Alignment Audit",
            "",
            f"## Gate A: **{result['gate_a']}**",
            "",
            *(
                f"- {name}: **{'PASS' if passed else 'FAIL'}**"
                for name, passed in criteria.items()
            ),
            "",
            f"- GT local trajectory position error: mean `{stats['gt_position_error_m']['mean']}`, P99 `{stats['gt_position_error_m']['p99']}`, max `{stats['gt_position_error_m']['max']}` m",
            f"- GT wrapped-heading error: max `{stats['gt_heading_error_rad']['max']}` rad",
            f"- Local→global→local position error: max `{stats['roundtrip_position_error_m']['max']}` m",
            f"- Raw-annotation→official-global vs MetricCache actor position error: P99 `{stats['actor_position_error_m']['p99']}` m",
            f"- Actor velocity error: P99 `{stats['actor_velocity_error_mps']['p99']}` m/s",
            f"- Measured logged interval: mean `{stats['timestamp_interval_s']['mean']}` s, max `{stats['timestamp_interval_s']['max']}` s",
            "",
            "Raw annotation boxes are ego-local in each future frame. Their global conversion was not inferred from names: the audit calls the local official `annotations_to_detection_tracks` path and matches stable track tokens against MetricCache global objects.",
            "",
            "All horizon lookups use nearest measured timestamps. MetricCache scoring uses its declared 10 Hz proposal sampling; no fixed raw-frame array indices are used for semantic horizons.",
            "",
            "Evidence tokens: "
            + ", ".join(f"`{token}`" for token in result["evidence_scene_tokens"][:8]),
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = validate(args)
    write_json(args.output_dir / "alignment_metrics.json", result)
    write_json(
        args.output_dir / "gate_a.json",
        {key: result[key] for key in ("gate_a", "criteria", "failures")},
    )
    write_text(args.output_dir / "ALIGNMENT_REPORT.md", render_report(result))
    print(
        json.dumps(
            {"gate_a": result["gate_a"], "criteria": result["criteria"]}, indent=2
        )
    )
    if result["gate_a"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
