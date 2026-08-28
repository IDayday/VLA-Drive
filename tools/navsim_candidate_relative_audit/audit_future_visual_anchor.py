#!/usr/bin/env python3
"""Phase 9: verify the GT future front-camera semantic-anchor data chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from .common import (
    HORIZONS_S,
    add_common_arguments,
    discover_paths,
    load_metric_cache_index,
    load_raw_window,
    metric_log_name,
    resolve_horizon_index,
    write_dataframe,
    write_json,
    write_text,
)


def front_camera(frame: dict[str, Any]) -> dict[str, Any]:
    cameras = frame.get("cams", {})
    return cameras.get("CAM_F0", cameras.get("cam_f0", {}))


def sensor_path(sensor_root: Path, camera: dict[str, Any]) -> Path | None:
    relative = camera.get("data_path")
    return sensor_root / Path(relative) if relative else None


def image_metadata(path: Path | None) -> tuple[bool, list[int] | None]:
    if path is None or not path.is_file():
        return False, None
    with Image.open(path) as image:
        return True, [int(image.width), int(image.height)]


def draw_anchor_figure(
    output_path: Path,
    token: str,
    current_image: Path,
    future_image: Path,
    trajectories: np.ndarray,
    gt_index: int,
    current_frame: dict[str, Any],
    horizon_s: float,
    consequence: np.ndarray | None,
    environment_fields: list[str],
) -> None:
    with Image.open(current_image) as image:
        current_rgb = np.asarray(image.convert("RGB"))
    with Image.open(future_image) as image:
        future_rgb = np.asarray(image.convert("RGB"))
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.4), constrained_layout=True)
    axes[0].imshow(current_rgb)
    axes[0].set_title(f"Current CAM_F0\nscene={token}")
    axes[0].axis("off")
    axes[1].imshow(future_rgb)
    axes[1].set_title(f"Logged GT future CAM_F0, h={horizon_s:.1f}s")
    axes[1].axis("off")
    annotations = current_frame.get("anns", {})
    boxes = np.asarray(annotations.get("gt_boxes", np.zeros((0, 7))))
    names = np.asarray(annotations.get("gt_names", []))
    for candidate_index, trajectory in enumerate(trajectories):
        style = (
            {"linewidth": 3.0, "color": "black", "zorder": 6}
            if candidate_index == gt_index
            else {
                "linewidth": 1.2,
                "alpha": 0.72,
            }
        )
        axes[2].plot(
            np.r_[0.0, trajectory[:, 0]],
            np.r_[0.0, trajectory[:, 1]],
            label=f"{candidate_index}{' GT' if candidate_index == gt_index else ''}",
            **style,
        )
    dynamic = (
        np.asarray([name in {"vehicle", "pedestrian", "bicycle"} for name in names])
        if len(names)
        else np.asarray([], dtype=bool)
    )
    if len(boxes):
        axes[2].scatter(
            boxes[~dynamic, 0],
            boxes[~dynamic, 1],
            s=12,
            c="gray",
            marker="x",
            label="static",
        )
        axes[2].scatter(
            boxes[dynamic, 0],
            boxes[dynamic, 1],
            s=20,
            c="red",
            marker="o",
            label="dynamic actor",
        )
    axes[2].scatter([0], [0], s=90, c="royalblue", marker="^", label="current ego")
    axes[2].set_aspect("equal")
    axes[2].grid(alpha=0.25)
    axes[2].set_xlabel("current-ego local x [m]")
    axes[2].set_ylabel("current-ego local y [m]")
    axes[2].set_title(f"GT + candidates (current-ego frame)\nK={len(trajectories)}")
    axes[2].legend(fontsize=7, ncol=2, loc="best")
    if consequence is not None:
        keys = (
            "minimum_dynamic_clearance_m",
            "minimum_linear_ttc_s",
            "candidate_center_in_drivable_map",
            "route_progress_m",
        )
        message = []
        for key in keys:
            if key in environment_fields:
                message.append(
                    f"GT {key}: {consequence[gt_index, environment_fields.index(key)]:.3g}"
                )
        figure.text(0.665, 0.01, " | ".join(message), ha="center", fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=140)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, max_scenes=12)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = discover_paths(
        args, split="trainval" if args.split == "train" else args.split
    )
    if (
        paths.metric_cache is None
        or paths.logs_root is None
        or paths.sensors_root is None
    ):
        raise FileNotFoundError(
            "MetricCache index, raw logs, and sensor blobs are required"
        )
    manifest = pd.read_parquet(
        args.output_dir / "candidate_manifest.parquet"
    ).sort_values(["scene_index", "candidate_index"])
    with np.load(args.output_dir / "candidate_trajectories.npz") as payload:
        trajectory_array = np.asarray(payload["trajectories"], dtype=np.float32)
    schema = json.loads((args.output_dir / "target_schema.json").read_text())
    environment_fields = schema["arrays"]["C_environment_only"]["fields"]
    metric_index = load_metric_cache_index(paths.metric_cache)
    scene_tokens = (
        manifest.drop_duplicates("scene_index")["scene_token"]
        .astype(str)
        .tolist()[: args.max_scenes]
    )
    coverage_rows: list[dict[str, Any]] = []
    synchronized_scene_count = 0
    figures_written = 0
    failures: list[dict[str, str]] = []
    evidence: list[dict[str, Any]] = []
    figure_root = args.output_dir / "figures" / "visual_anchor"
    for token in scene_tokens:
        try:
            metric_path = metric_index[token]
            frames = load_raw_window(
                paths.logs_root, metric_log_name(metric_path), token
            )
            current_index = 3
            current = frames[current_index]
            relevant = frames[current_index:]
            timestamps = [frame["timestamp"] for frame in relevant]
            scene_manifest = manifest[
                manifest["scene_token"].astype(str) == token
            ].sort_values("candidate_index")
            scene_index = int(scene_manifest.iloc[0]["scene_index"])
            gt_indices = np.flatnonzero(scene_manifest["is_gt"].to_numpy(dtype=bool))
            gt_index = int(gt_indices[0])
            target_path = args.output_dir / "targets" / f"{token}.npz"
            target_exists = target_path.is_file()
            target_environment = None
            if target_exists:
                with np.load(target_path) as payload:
                    target_environment = np.asarray(payload["C_environment_only"])
            all_sync = True
            selected_future_path: Path | None = None
            selected_horizon = 4.0
            for horizon_slot, horizon_s in enumerate(HORIZONS_S):
                relative_index = resolve_horizon_index(timestamps, horizon_s)
                frame = relevant[relative_index]
                camera = front_camera(frame)
                path = sensor_path(paths.sensors_root, camera)
                exists, dimensions = image_metadata(path)
                annotations = frame.get("anns", {})
                row = {
                    "scene_token": token,
                    "log_name": metric_log_name(metric_path),
                    "horizon_s": horizon_s,
                    "timestamp_us": int(frame["timestamp"]),
                    "actual_horizon_s": (
                        int(frame["timestamp"]) - int(current["timestamp"])
                    )
                    / 1e6,
                    "front_camera_declared": bool(camera.get("data_path")),
                    "front_camera_exists": exists,
                    "front_camera_width": dimensions[0] if dimensions else None,
                    "front_camera_height": dimensions[1] if dimensions else None,
                    "camera_intrinsics_present": camera.get("cam_intrinsic")
                    is not None,
                    "camera_extrinsics_present": camera.get("sensor2lidar_rotation")
                    is not None
                    and camera.get("sensor2lidar_translation") is not None,
                    "ego_pose_present": frame.get("ego2global_translation") is not None,
                    "annotations_present": "gt_boxes" in annotations,
                    "traffic_lights_present": frame.get("traffic_lights") is not None,
                    "track_tokens_present": bool(annotations.get("track_tokens")),
                    "candidate_relative_target_present": target_exists,
                    "same_raw_frame_timestamp": True,
                }
                row_sync = all(
                    bool(row[key])
                    for key in (
                        "front_camera_exists",
                        "ego_pose_present",
                        "annotations_present",
                        "traffic_lights_present",
                        "track_tokens_present",
                        "candidate_relative_target_present",
                    )
                )
                row["synchronized_anchor_available"] = row_sync
                all_sync &= row_sync
                coverage_rows.append(row)
                if horizon_s == 4.0:
                    selected_future_path = path
            current_path = sensor_path(paths.sensors_root, front_camera(current))
            if all_sync:
                synchronized_scene_count += 1
            if (
                current_path is not None
                and current_path.is_file()
                and selected_future_path is not None
                and selected_future_path.is_file()
            ):
                consequence = (
                    target_environment[:, -1]
                    if target_environment is not None
                    else None
                )
                draw_anchor_figure(
                    figure_root / f"{token}.png",
                    token,
                    current_path,
                    selected_future_path,
                    trajectory_array[scene_index],
                    gt_index,
                    current,
                    selected_horizon,
                    consequence,
                    environment_fields,
                )
                figures_written += 1
            evidence.append(
                {
                    "scene_token": token,
                    "log_name": metric_log_name(metric_path),
                    "all_four_horizons_synchronized": all_sync,
                    "figure": str(figure_root / f"{token}.png"),
                }
            )
        except Exception as error:
            failures.append(
                {"scene_token": token, "error": f"{type(error).__name__}: {error}"}
            )
    coverage = pd.DataFrame(coverage_rows)
    write_dataframe(coverage, args.output_dir / "future_visual_anchor_coverage.csv")
    summary = {
        "scene_count": len(scene_tokens),
        "horizon_row_count": len(coverage),
        "future_front_camera_coverage": float(coverage["front_camera_exists"].mean())
        if len(coverage)
        else 0.0,
        "gt_structural_image_synchrony_coverage": float(
            coverage["synchronized_anchor_available"].mean()
        )
        if len(coverage)
        else 0.0,
        "all_horizon_synchronized_scene_coverage": synchronized_scene_count
        / len(scene_tokens)
        if scene_tokens
        else 0.0,
        "figures_written": figures_written,
        "failures": failures,
        "scene_evidence": evidence,
        "supported": "I_GT(t+h) <-> C_GT,h, using the same logged-future frame timestamp",
        "unsupported": "I_candidate_i(t+h) for every non-GT candidate; no such observed image exists in the log",
        "training_use": "GT-only visual semantic anchor, conditional on the measured camera coverage",
        "inference_use": "future image/target is not directly available; a learned predictor would be required",
    }
    write_json(args.output_dir / "future_visual_anchor_summary.json", summary)
    write_text(
        args.output_dir / "FUTURE_VISUAL_ANCHOR_REPORT.md",
        "\n".join(
            [
                "# GT Future Visual Anchor Audit",
                "",
                f"- Scenes audited: **{summary['scene_count']}**",
                f"- Future `cam_f0` file coverage: **{summary['future_front_camera_coverage']:.3%}**",
                f"- Same-frame GT image/pose/annotation/traffic-light/track/structural-target coverage: **{summary['gt_structural_image_synchrony_coverage']:.3%}**",
                f"- Visual anchor figures written: **{summary['figures_written']}**",
                "",
                "## Supported",
                "",
                "`I_GT(t+h) <-> C_GT,h`: the front-camera image and GT candidate-relative structured future are resolved from the same logged-future timestamp.",
                "",
                "## Not supported",
                "",
                "The log does not provide a candidate-specific ground-truth future image for any non-GT candidate. Candidate-conditioned relabeling changes structured relationships, not the recorded pixels.",
                "",
                "This audit validates a GT-only visual anchor. It does not train a world model or describe replayed candidate futures as real counterfactual images.",
                "",
            ]
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
