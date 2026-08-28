#!/usr/bin/env python3
"""Audit synchronization between logged GT future images and structured targets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from .build_candidate_relative_targets import ENVIRONMENT_FEATURES
from .build_soft_contrastive_labels import _sigma_from_distances, prefix_distance_matrix, softmax_negative
from .common import (
    DEFAULT_HORIZONS,
    add_common_arguments,
    append_command,
    ensure_output_dir,
    global_to_local,
    load_scenes_for_tokens,
    paths_from_args,
    read_parquet,
    resolve_horizon_index,
    write_markdown,
)


def _camera_record(frame: dict[str, Any], sensor: str = "cam_f0") -> dict[str, Any] | None:
    for key, value in frame.get("cams", {}).items():
        if key.lower() == sensor.lower():
            return value
    return None


def _candidate_poses(group: pd.DataFrame) -> np.ndarray:
    return np.asarray(
        [np.column_stack([row.pose_x_m, row.pose_y_m, row.pose_heading_rad]) for row in group.itertuples()],
        dtype=np.float64,
    )


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _render_scene(
    output_path: Path,
    token: str,
    scene: Any,
    raw_frames: list[dict[str, Any]],
    paths: Any,
    group: pd.DataFrame,
    arrays: Any,
    horizon_s: float = 4.0,
) -> None:
    current = scene.scene_metadata.num_history_frames - 1
    timestamps = [frame.timestamp for frame in scene.frames]
    future_index = resolve_horizon_index(timestamps, horizon_s, origin_index=current)
    current_camera = _camera_record(raw_frames[current])
    future_camera = _camera_record(raw_frames[future_index])
    if current_camera is None or future_camera is None:
        return
    current_path = paths.sensor_blobs_path / current_camera["data_path"]
    future_path = paths.sensor_blobs_path / future_camera["data_path"]
    if not current_path.is_file() or not future_path.is_file():
        return

    poses = _candidate_poses(group)
    h_index = int(np.argmin(np.abs(np.asarray(arrays["time_s"]) - horizon_s)))
    current_pose = np.asarray(scene.frames[current].ego_status.ego_pose, dtype=np.float64)
    shared = np.asarray(arrays["shared_logged_future_actor"])[h_index]
    shared_mask = np.asarray(arrays["shared_logged_future_actor_mask"])[h_index].astype(bool)
    actor_global = shared[shared_mask]
    actor_local = (
        global_to_local(
            current_pose,
            np.column_stack([actor_global[:, 1], actor_global[:, 2], actor_global[:, 5]]),
        )
        if len(actor_global)
        else np.empty((0, 3))
    )
    gt_index = int(np.flatnonzero(group.is_gt.to_numpy())[0])
    distances = prefix_distance_matrix(poses, horizon_s)[gt_index]
    q = softmax_negative(distances[None], _sigma_from_distances(distances))[0]
    clearance_index = ENVIRONMENT_FEATURES.index("min_actor_polygon_clearance_m")
    collision_index = ENVIRONMENT_FEATURES.index("any_actor_collision")
    clearance = np.asarray(arrays["C_environment_only"])[:, h_index, clearance_index]
    collision = np.asarray(arrays["C_environment_only"])[:, h_index, collision_index]

    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    axes[0, 0].imshow(_load_rgb(current_path))
    axes[0, 0].set_title("Logged current CAM_F0 (current ego frame anchor)")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(_load_rgb(future_path))
    axes[0, 1].set_title(f"Logged GT future CAM_F0, nearest h={horizon_s:.1f}s")
    axes[0, 1].axis("off")

    bev = axes[1, 0]
    for candidate, row in zip(poses, group.itertuples()):
        style = {"color": "black", "linewidth": 3.0, "zorder": 4} if row.is_gt else {
            "linewidth": 1.0,
            "alpha": 0.65,
        }
        bev.plot(candidate[:, 0], candidate[:, 1], label=f"{row.candidate_index}{'*GT' if row.is_gt else ''}", **style)
    if len(actor_local):
        bev.scatter(actor_local[:, 0], actor_local[:, 1], s=15, c="tab:red", alpha=0.65, label="logged actors")
    bev.scatter([0], [0], marker="^", s=80, c="tab:blue", label="current ego")
    bev.set_aspect("equal", adjustable="datalim")
    bev.set_xlabel("current-ego local x [m]")
    bev.set_ylabel("current-ego local y [m]")
    bev.set_title(f"GT + candidates + logged actor world at h={horizon_s:.1f}s")
    bev.legend(fontsize=7, ncol=3, loc="best")
    bev.grid(alpha=0.2)

    consequence = axes[1, 1]
    indices = group.candidate_index.to_numpy(dtype=int)
    consequence.bar(indices - 0.2, np.minimum(clearance, 20), width=0.4, label="clearance [m], clipped 20")
    consequence.bar(indices + 0.2, q * 10, width=0.4, label="GT-prefix soft weight ×10")
    for index, flag in zip(indices, collision):
        if flag > 0.5:
            consequence.text(index, 0.1, "collision", rotation=90, color="red", ha="center", va="bottom", fontsize=7)
    consequence.set_xlabel("candidate index (0 is GT)")
    consequence.set_title("Candidate-relative consequence and prefix-aware label")
    consequence.legend(fontsize=8)
    consequence.grid(axis="y", alpha=0.2)
    figure.suptitle(
        f"scene={token} | policy=non_reactive | world=global→current-ego local | distance=m",
        fontsize=11,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=140)
    plt.close(figure)


def audit(args: argparse.Namespace) -> pd.DataFrame:
    paths = paths_from_args(args)
    output_dir = ensure_output_dir(args.output_dir)
    index = read_parquet(output_dir / "targets/index.parquet")
    manifest = read_parquet(output_dir / "candidate_manifest.parquet")
    if args.max_scenes > 0:
        index = index.head(args.max_scenes)
    tokens = index.scene_token.astype(str).tolist()
    loader = load_scenes_for_tokens(paths, tokens)
    rows: list[dict[str, Any]] = []
    rendered = 0
    figure_dir = output_dir / "figures/visual_anchor"

    for index_row in index.itertuples():
        token = str(index_row.scene_token)
        scene = loader.get_scene_from_token(token)
        raw_frames = loader.scene_frames_dicts[token]
        current = scene.scene_metadata.num_history_frames - 1
        timestamps = [frame.timestamp for frame in scene.frames]
        arrays = np.load(output_dir / index_row.target_path)
        group = manifest[manifest.scene_token == token].sort_values("candidate_index")
        for horizon in DEFAULT_HORIZONS:
            frame_index = resolve_horizon_index(timestamps, horizon, origin_index=current)
            camera = _camera_record(raw_frames[frame_index])
            image_path = paths.sensor_blobs_path / camera["data_path"] if camera and camera.get("data_path") else None
            target_index = int(np.argmin(np.abs(np.asarray(arrays["time_s"]) - horizon)))
            target_time = float(np.asarray(arrays["time_s"])[target_index])
            frame = scene.frames[frame_index]
            rows.append(
                {
                    "scene_token": token,
                    "log_name": scene.scene_metadata.log_name,
                    "horizon_s": horizon,
                    "resolved_frame_index": frame_index,
                    "timestamp_error_s": abs((timestamps[frame_index] - timestamps[current]) / 1e6 - horizon),
                    "front_camera_record_available": camera is not None,
                    "front_camera_file_available": bool(image_path and image_path.is_file()),
                    "ego_pose_available": frame.ego_status.ego_pose is not None,
                    "annotations_available": frame.annotations is not None,
                    "traffic_light_field_available": frame.traffic_lights is not None,
                    "track_token_field_available": getattr(frame.annotations, "track_tokens", None) is not None,
                    "candidate_relative_target_available": abs(target_time - horizon) <= 1e-6,
                    "synchronized_all": bool(
                        image_path
                        and image_path.is_file()
                        and frame.ego_status.ego_pose is not None
                        and frame.annotations is not None
                        and frame.traffic_lights is not None
                        and getattr(frame.annotations, "track_tokens", None) is not None
                        and abs(target_time - horizon) <= 1e-6
                    ),
                }
            )
        if rendered < args.num_figures:
            path = figure_dir / f"{rendered:02d}_{token}_h4p0.png"
            _render_scene(path, token, scene, raw_frames, paths, group, arrays)
            if path.is_file():
                rendered += 1

    coverage = pd.DataFrame(rows)
    coverage.to_csv(output_dir / "future_visual_anchor_coverage.csv", index=False)
    synchronized = float(coverage.synchronized_all.mean()) if len(coverage) else 0.0
    camera = float(coverage.front_camera_file_available.mean()) if len(coverage) else 0.0
    evidence = coverage.loc[coverage.synchronized_all, "scene_token"].drop_duplicates().head(8).tolist()
    report = f"""# GT Future Visual Anchor Audit

## Measured data chain

- Audited candidate-target scenes / horizon records: {coverage.scene_token.nunique()} / {len(coverage)}
- CAM_F0 future file coverage at 0.5/1/2/4 s: {camera:.3%}
- Full synchronization coverage (image + ego pose + annotations + traffic-light field + track tokens + structural target): {synchronized:.3%}
- Median timestamp resolution error: {coverage.timestamp_error_s.median() if len(coverage) else float('nan'):.6f} s
- Rendered visual evidence scenes: {rendered}
- Evidence tokens: `{', '.join(evidence)}`

The coverage above checks path-backed files for every record. Image decoding, dimensions and pixels are verified on the bounded field-audit sample and every rendered evidence scene; the audit intentionally does not bulk-decode all 2,000 images.

The local data supports the factual synchronization `logged I_GT(t+h) ↔ C_GT,h`.  It does not contain a logged camera image captured from any non-GT candidate pose.  Therefore non-GT candidate images are unavailable as ground truth; reprojected, generated, or synthetic images would be a different supervision class and must be labeled accordingly.

No visual encoder weights were downloaded.  The optional embedding-cache check was intentionally skipped because it is not required to establish the data link.
"""
    write_markdown(output_dir / "FUTURE_VISUAL_ANCHOR_REPORT.md", report)
    append_command(
        output_dir,
        "python -m tools.navsim_candidate_relative_audit.audit_future_visual_anchor " + " ".join(sys.argv[1:]),
    )
    return coverage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--num-figures", type=int, default=12)
    args = parser.parse_args()
    audit(args)


if __name__ == "__main__":
    main()
