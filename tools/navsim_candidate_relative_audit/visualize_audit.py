#!/usr/bin/env python3
"""Create the required quantitative audit visualizations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd

from .common import add_common_arguments, write_json


def save(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def ego_polygon(
    x: float, y: float, heading: float, length: float = 4.8, width: float = 2.0
) -> np.ndarray:
    local = np.asarray(
        [
            [length / 2, width / 2],
            [length / 2, -width / 2],
            [-length / 2, -width / 2],
            [-length / 2, width / 2],
        ]
    )
    rotation = np.asarray(
        [[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]]
    )
    return local @ rotation.T + np.asarray([x, y])


def heatmap(
    values: np.ndarray,
    xlabels: list[str],
    ylabels: list[str],
    title: str,
    colorbar: str,
    path: Path,
) -> None:
    figure, axis = plt.subplots(
        figsize=(max(7, len(xlabels) * 0.65), max(4.5, len(ylabels) * 0.42))
    )
    image = axis.imshow(values, aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(xlabels)), xlabels, rotation=45, ha="right", fontsize=8)
    axis.set_yticks(range(len(ylabels)), ylabels, fontsize=8)
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label=colorbar)
    save(figure, path)


def read_optional_csv(path: Path) -> pd.DataFrame:
    try:
        return (
            pd.read_csv(path)
            if path.is_file() and path.stat().st_size
            else pd.DataFrame()
        )
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    args = parser.parse_args()
    output = args.output_dir
    figure_root = output / "figures" / "audit"
    manifest = pd.read_parquet(output / "candidate_manifest.parquet").sort_values(
        ["scene_index", "candidate_index"]
    )
    metrics = pd.read_parquet(output / "candidate_metrics.parquet").sort_values(
        ["scene_index", "candidate_index"]
    )
    pairwise = pd.read_parquet(output / "target_pairwise_distances.parquet")
    coverage = pd.read_csv(output / "scene_coverage.csv")
    schema = json.loads((output / "target_schema.json").read_text())
    with np.load(output / "candidate_trajectories.npz") as payload:
        trajectories = np.asarray(payload["trajectories"])
    with np.load(output / "soft_labels.npz") as payload:
        gt_q = np.asarray(payload["gt_q"])
        consequence_distances = np.asarray(payload["consequence_distances"])
        horizons_s = np.asarray(payload["horizons_s"])
    first_scene = int(manifest["scene_index"].min())
    scene_manifest = manifest[manifest["scene_index"] == first_scene].sort_values(
        "candidate_index"
    )
    token = str(scene_manifest.iloc[0]["scene_token"])
    gt_index = int(np.flatnonzero(scene_manifest["is_gt"].to_numpy(dtype=bool))[0])
    candidate_labels = [
        f"{int(row.candidate_index)}{' GT' if bool(row.is_gt) else ''}"
        for row in scene_manifest.itertuples()
    ]
    with np.load(output / "targets" / f"{token}.npz") as payload:
        actor = np.asarray(payload["candidate_relative_actor_tensor"])
        actor_mask = np.asarray(payload["candidate_relative_actor_mask"])
    actor_fields = schema["arrays"]["candidate_relative_actor_tensor"]["fields"]
    relative_x = actor_fields.index("relative_x_m")
    relative_y = actor_fields.index("relative_y_m")
    in_corridor = actor_fields.index("in_candidate_corridor")
    files: list[dict[str, str]] = []

    # 1. Current BEV + GT + K candidates.
    figure, axis = plt.subplots(figsize=(8, 8))
    for candidate_index, trajectory in enumerate(trajectories[first_scene]):
        axis.plot(
            np.r_[0, trajectory[:, 0]],
            np.r_[0, trajectory[:, 1]],
            linewidth=3 if candidate_index == gt_index else 1.2,
            color="black" if candidate_index == gt_index else None,
            alpha=1 if candidate_index == gt_index else 0.75,
            label=candidate_labels[candidate_index],
        )
    axis.scatter([0], [0], marker="^", s=100, c="royalblue")
    axis.set_aspect("equal")
    axis.grid(alpha=0.25)
    axis.set_xlabel("current-ego local x [m]")
    axis.set_ylabel("current-ego local y [m]")
    axis.set_title(
        f"Current BEV: scene={token}, K={len(candidate_labels)}, GT={gt_index}\nnon-reactive audit; current-ego frame"
    )
    axis.legend(fontsize=7, ncol=2)
    path = figure_root / "01_current_bev_gt_candidates.png"
    save(figure, path)
    files.append({"name": "current_bev", "path": str(path)})

    # 2. Candidate ego footprints in one logged future world.
    figure, axis = plt.subplots(figsize=(8, 8))
    for candidate_index, pose in enumerate(trajectories[first_scene, :, -1]):
        polygon = ego_polygon(*pose)
        axis.add_patch(
            Polygon(
                polygon,
                closed=True,
                fill=False,
                linewidth=2.5 if candidate_index == gt_index else 1.1,
                label=candidate_labels[candidate_index],
            )
        )
        axis.text(pose[0], pose[1], str(candidate_index), fontsize=8)
    axis.autoscale()
    axis.set_aspect("equal")
    axis.grid(alpha=0.25)
    axis.set_xlabel("current-ego local x [m]")
    axis.set_ylabel("current-ego local y [m]")
    axis.set_title(
        f"Candidate ego footprints in shared logged future\nscene={token}, h=4.0 s, GT={gt_index}, footprint [m]"
    )
    path = figure_root / "02_shared_future_candidate_footprints.png"
    save(figure, path)
    files.append({"name": "candidate_footprints", "path": str(path)})

    # 3. Candidate-relative actor positions.
    shown = min(6, len(candidate_labels))
    figure, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for candidate_index, axis in enumerate(axes.ravel()):
        if candidate_index >= shown:
            axis.axis("off")
            continue
        valid = actor_mask[candidate_index, -1]
        values = actor[candidate_index, -1, valid]
        axis.scatter(
            values[:, relative_x],
            values[:, relative_y],
            c=values[:, 0],
            s=28,
            cmap="tab10",
        )
        axis.scatter([0], [0], marker="^", s=80, c="black")
        axis.set_aspect("equal")
        axis.grid(alpha=0.2)
        axis.set_title(f"candidate={candidate_labels[candidate_index]}")
        axis.set_xlabel("candidate-ego x [m]")
        axis.set_ylabel("candidate-ego y [m]")
    figure.suptitle(
        f"Relative actors: scene={token}, h=4.0 s, non-reactive logged future"
    )
    path = figure_root / "03_candidate_relative_actor_positions.png"
    save(figure, path)
    files.append({"name": "relative_actors", "path": str(path)})

    # 4. Candidate corridor occupancy.
    figure, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for candidate_index, axis in enumerate(axes.ravel()):
        if candidate_index >= shown:
            axis.axis("off")
            continue
        valid = actor_mask[candidate_index, -1]
        values = actor[candidate_index, -1, valid]
        corridor = values[:, in_corridor] > 0.5
        axis.scatter(
            values[~corridor, relative_x],
            values[~corridor, relative_y],
            c="gray",
            s=15,
            alpha=0.5,
        )
        axis.scatter(
            values[corridor, relative_x],
            values[corridor, relative_y],
            c="red",
            s=45,
            label="in corridor",
        )
        axis.scatter([0], [0], marker="^", s=80, c="black")
        axis.set_aspect("equal")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
        axis.set_title(
            f"candidate={candidate_labels[candidate_index]}, occupied={int(corridor.sum())}"
        )
        axis.set_xlabel("candidate-ego x [m]")
        axis.set_ylabel("candidate-ego y [m]")
    figure.suptitle(
        f"Candidate corridor occupancy: scene={token}, h=4.0 s, non-reactive"
    )
    path = figure_root / "04_candidate_corridor_occupancy.png"
    save(figure, path)
    files.append({"name": "corridor_occupancy", "path": str(path)})

    # 5. Candidate x candidate consequence distance.
    path = figure_root / "05_candidate_consequence_distance_heatmap.png"
    heatmap(
        consequence_distances[first_scene, -1],
        candidate_labels,
        candidate_labels,
        f"Candidate × candidate consequence distance\nscene={token}, h=4.0 s, C_environment_only",
        "standardized consequence distance",
        path,
    )
    files.append({"name": "consequence_distance", "path": str(path)})

    # 6. Prefix-aware GT labels.
    path = figure_root / "06_prefix_aware_gt_soft_label_heatmap.png"
    heatmap(
        gt_q[first_scene],
        candidate_labels,
        [f"{h:g} s" for h in horizons_s],
        f"Prefix-aware GT soft labels\nscene={token}, GT={gt_index}; each row uses prefix only",
        "q(candidate | GT prefix)",
        path,
    )
    files.append({"name": "prefix_soft_labels", "path": str(path)})

    # 7. Official PDM factor heatmap.
    scene_metrics = metrics[metrics["scene_index"] == first_scene].sort_values(
        "candidate_index"
    )
    factors = [
        "aggregate_score",
        "no_at_fault_collision",
        "drivable_area_compliance",
        "driving_direction_compliance",
        "traffic_light_compliance",
        "time_to_collision_within_bound",
        "lane_keeping",
        "ego_progress",
    ]
    path = figure_root / "07_pdm_factor_heatmap.png"
    heatmap(
        scene_metrics[factors].to_numpy(dtype=float),
        factors,
        candidate_labels,
        f"Official PDM factors\nscene={token}, traffic_policy={scene_metrics.iloc[0]['traffic_policy']}, GT={gt_index}",
        "factor / score",
        path,
    )
    files.append({"name": "pdm_factors", "path": str(path)})

    # 8/9. Distance relationships.
    for number, xfield, yfield, title, xlabel, ylabel in (
        (
            8,
            "trajectory_distance",
            "consequence_distance",
            "Trajectory vs consequence distance",
            "trajectory distance [standardized]",
            "C_environment_only distance [standardized]",
        ),
        (
            9,
            "consequence_distance",
            "score_difference",
            "Consequence distance vs PDM score difference",
            "C_environment_only distance [standardized]",
            "absolute PDM score difference",
        ),
    ):
        figure, axis = plt.subplots(figsize=(7, 5.5))
        axis.scatter(pairwise[xfield], pairwise[yfield], s=14, alpha=0.45)
        correlation = pairwise[[xfield, yfield]].corr(method="spearman").iloc[0, 1]
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.set_title(
            f"{title}\n{len(pairwise)} candidate pairs, Spearman={correlation:.3f}, non-reactive"
        )
        path = figure_root / f"{number:02d}_{xfield}_vs_{yfield}.png"
        save(figure, path)
        files.append({"name": f"{xfield}_vs_{yfield}", "path": str(path)})

    # 10. Non-reactive vs reactive actor response (or explicit pending-track fallback).
    response = read_optional_csv(output / "reactive_actor_response.csv")
    figure, axis = plt.subplots(figsize=(7, 5.5))
    if len(response):
        vehicle = response[response["object_type"].str.contains("VEHICLE", case=False)]
        axis.scatter(
            vehicle["replay_speed_mps"], vehicle["reactive_speed_mps"], s=12, alpha=0.35
        )
        limit = max(
            vehicle[["replay_speed_mps", "reactive_speed_mps"]].max().max(), 1.0
        )
        axis.plot([0, limit], [0, limit], "k--", linewidth=1)
        axis.set_xlabel("non-reactive replay speed [m/s]")
        axis.set_ylabel("reactive IDM speed [m/s]")
        title = f"Vehicle response: non-reactive vs reactive\n{vehicle.scene_token.nunique()} scenes, h=1/2/4 s"
    else:
        scalar = read_optional_csv(output / "reactive_scalar_comparison.csv")
        axis.hist(scalar.get("pdm_score_change", pd.Series(dtype=float)), bins=30)
        axis.set_xlabel("reactive minus non-reactive PDM score")
        axis.set_ylabel("candidate count")
        title = "Reactive scalar cache comparison\nactor-track capture not run in this invocation"
    axis.grid(alpha=0.25)
    axis.set_title(title)
    path = figure_root / "10_nonreactive_reactive_actor_response.png"
    save(figure, path)
    files.append({"name": "reactive_response", "path": str(path)})

    # 11. Future sensor availability.
    figure, axis = plt.subplots(figsize=(7, 5.5))
    availability = [
        coverage["future_cam_f0_coverage"].to_numpy(),
        coverage["future_lidar_coverage"].to_numpy(),
    ]
    axis.hist(
        availability, bins=np.linspace(0, 1, 11), label=["CAM_F0", "LiDAR"], alpha=0.7
    )
    axis.set_xlabel("per-scene sparse future availability fraction")
    axis.set_ylabel("scene count")
    axis.set_title(
        f"Future sensor availability\n{len(coverage)} train scenes; horizons near 0.5/1/2/4 s"
    )
    axis.legend()
    axis.grid(alpha=0.25)
    path = figure_root / "11_future_sensor_availability_histogram.png"
    save(figure, path)
    files.append({"name": "sensor_availability", "path": str(path)})

    # 12. Track continuity.
    figure, axis = plt.subplots(figsize=(7, 5.5))
    axis.hist(
        [
            coverage["raw_track_transition_continuity"],
            coverage["metric_track_transition_continuity"],
        ],
        bins=np.linspace(0, 1, 21),
        label=["raw ~2 Hz", "MetricCache 10 Hz"],
        alpha=0.7,
    )
    axis.set_xlabel("adjacent-frame stable track-token continuity")
    axis.set_ylabel("scene count")
    axis.set_title(
        f"Future track continuity\n{len(coverage)} train scenes, token-based"
    )
    axis.legend()
    axis.grid(alpha=0.25)
    path = figure_root / "12_track_continuity_histogram.png"
    save(figure, path)
    files.append({"name": "track_continuity", "path": str(path)})

    summary = {
        "figure_count": len(files),
        "scene_token_used_for_candidate_figures": token,
        "gt_candidate_index": gt_index,
        "traffic_policy": str(scene_metrics.iloc[0]["traffic_policy"]),
        "files": files,
    }
    write_json(output / "visualization_manifest.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
