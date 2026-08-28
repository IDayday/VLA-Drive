#!/usr/bin/env python3
"""Generate the required quantitative and scene-level audit visualizations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from .analyze_target_diversity import consequence_distance_matrix
from .build_candidate_relative_targets import ACTOR_FEATURES, ENVIRONMENT_FEATURES
from .build_soft_contrastive_labels import HORIZONS, _sigma_from_distances, prefix_distance_matrix, softmax_negative
from .common import add_common_arguments, append_command, ensure_output_dir, read_parquet, write_markdown


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _poses(group: pd.DataFrame) -> np.ndarray:
    return np.asarray(
        [np.column_stack([row.pose_x_m, row.pose_y_m, row.pose_heading_rad]) for row in group.itertuples()],
        dtype=float,
    )


def _scene_title(token: str, horizon: float | str, suffix: str) -> str:
    return f"scene={token} | h={horizon}s | policy=non_reactive | {suffix}"


def generate(args: argparse.Namespace) -> list[str]:
    output_dir = ensure_output_dir(args.output_dir)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    index = read_parquet(output_dir / "targets/index.parquet")
    manifest = read_parquet(output_dir / "candidate_manifest.parquet")
    metrics = read_parquet(output_dir / "candidate_metrics.parquet")
    if len(index) == 0:
        raise RuntimeError("No candidate-relative targets available for visualization")
    first = index.iloc[0]
    token = str(first.scene_token)
    arrays = np.load(output_dir / first.target_path)
    group = manifest[manifest.scene_token == token].sort_values("candidate_index").reset_index(drop=True)
    scores = metrics[(metrics.scene_token == token) & metrics.scoring_success].sort_values("candidate_index")
    poses = _poses(group)
    gt_index = int(np.flatnonzero(group.is_gt.to_numpy())[0])
    written: list[str] = []

    # 1. Current BEV + GT + K trajectories.
    figure, axis = plt.subplots(figsize=(9, 7))
    for candidate, row in zip(poses, group.itertuples()):
        axis.plot(
            np.r_[0, candidate[:, 0]],
            np.r_[0, candidate[:, 1]],
            linewidth=3 if row.is_gt else 1.2,
            color="black" if row.is_gt else None,
            alpha=1 if row.is_gt else 0.7,
            label=f"c{row.candidate_index}{' GT' if row.is_gt else ''}",
        )
    axis.scatter([0], [0], marker="^", s=120, c="tab:blue")
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xlabel("current-ego local x [m]")
    axis.set_ylabel("current-ego local y [m]")
    axis.set_title(_scene_title(token, "0→4", f"GT=c{gt_index}; K={len(group)}; current-ego local [m]"))
    axis.legend(ncol=3, fontsize=8)
    axis.grid(alpha=0.25)
    path = figure_dir / "01_current_bev_gt_candidates.png"; _save(figure, path); written.append(str(path))

    # 2. Candidate ego footprints in one logged world, expressed in current frame.
    figure, axis = plt.subplots(figsize=(9, 7))
    h_index = 7
    for candidate, row in zip(poses, group.itertuples()):
        x, y, heading = candidate[h_index]
        rectangle = Rectangle((-2.5, -1.0), 5.0, 2.0, fill=False, linewidth=2 if row.is_gt else 1, alpha=0.8)
        transform = matplotlib.transforms.Affine2D().rotate(heading).translate(x, y) + axis.transData
        rectangle.set_transform(transform)
        rectangle.set_edgecolor("black" if row.is_gt else plt.cm.tab20(row.candidate_index % 20))
        axis.add_patch(rectangle)
        axis.text(x, y, f"c{row.candidate_index}{'*' if row.is_gt else ''}", fontsize=7)
    axis.set_aspect("equal", adjustable="datalim")
    axis.autoscale_view()
    axis.set_xlabel("current-ego local x [m]"); axis.set_ylabel("current-ego local y [m]")
    axis.set_title(_scene_title(token, 4.0, "candidate ego footprints; current-ego local; dimensions [m]"))
    axis.grid(alpha=0.25)
    path = figure_dir / "02_logged_world_candidate_footprints.png"; _save(figure, path); written.append(str(path))

    # 3. Logged actors transformed into several candidate frames.
    candidate_choices = list(dict.fromkeys([gt_index, min(1, len(group) - 1), min(2, len(group) - 1)]))
    figure, axes = plt.subplots(1, len(candidate_choices), figsize=(5 * len(candidate_choices), 5), squeeze=False)
    rx = ACTOR_FEATURES.index("relative_x_m"); ry = ACTOR_FEATURES.index("relative_y_m")
    for axis, candidate in zip(axes[0], candidate_choices):
        actor = arrays["candidate_relative_actor"][candidate, h_index]
        mask = arrays["candidate_relative_actor_mask"][candidate, h_index].astype(bool)
        axis.scatter(actor[mask, rx], actor[mask, ry], c=actor[mask, 0], cmap="tab10", s=35)
        axis.scatter([0], [0], marker="^", s=100, c="black")
        axis.set_aspect("equal", adjustable="datalim"); axis.grid(alpha=0.2)
        axis.set_xlabel("candidate-ego x [m]"); axis.set_ylabel("candidate-ego y [m]")
        axis.set_title(f"c{candidate}{' GT' if candidate == gt_index else ''}; nearest-N={int(mask.sum())}")
    figure.suptitle(_scene_title(token, 4.0, "same logged actors → candidate-ego frames; position [m]"))
    path = figure_dir / "03_candidate_relative_actor_positions.png"; _save(figure, path); written.append(str(path))

    # 4. Candidate corridor occupancy over time.
    corridor = arrays["C_environment_only"][:, :, ENVIRONMENT_FEATURES.index("candidate_corridor_actor_count")]
    figure, axis = plt.subplots(figsize=(10, 6))
    image = axis.imshow(corridor, aspect="auto", cmap="magma")
    axis.set_xticks(range(8), [f"{x:.1f}" for x in arrays["time_s"]]); axis.set_yticks(range(len(group)))
    axis.set_yticklabels([f"c{i}{'*GT' if i == gt_index else ''}" for i in range(len(group))])
    axis.set_xlabel("horizon [s]"); axis.set_ylabel("candidate index")
    axis.set_title(_scene_title(token, "0.5–4.0", "corridor actor occupancy [count]; candidate prefix corridor"))
    figure.colorbar(image, ax=axis, label="actor count")
    path = figure_dir / "04_candidate_corridor_occupancy.png"; _save(figure, path); written.append(str(path))

    # 5. Candidate x candidate consequence distance.
    consequence = consequence_distance_matrix(arrays)
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(consequence, cmap="viridis")
    labels = [f"c{i}{'*' if i == gt_index else ''}" for i in range(len(group))]
    axis.set_xticks(range(len(group)), labels, rotation=45); axis.set_yticks(range(len(group)), labels)
    axis.set_title(
        f"scene={token} | h=4.0s | policy=non_reactive\n"
        "K×K standardized consequence distance; candidates c0*=GT; symmetric"
    )
    figure.colorbar(image, ax=axis, label="normalized distance [unitless]")
    path = figure_dir / "05_consequence_distance_heatmap.png"; _save(figure, path); written.append(str(path))

    # 6. Prefix-aware factual label across horizons.
    labels_by_horizon = []
    for horizon in HORIZONS:
        distance = prefix_distance_matrix(poses, horizon)[gt_index]
        labels_by_horizon.append(softmax_negative(distance[None], _sigma_from_distances(distance))[0])
    figure, axis = plt.subplots(figsize=(10, 5))
    image = axis.imshow(np.asarray(labels_by_horizon), aspect="auto", cmap="Blues")
    axis.set_xticks(range(len(group)), labels, rotation=45); axis.set_yticks(range(len(HORIZONS)), HORIZONS)
    axis.set_xlabel("candidate index; *=GT"); axis.set_ylabel("prefix horizon [s]")
    axis.set_title(
        f"scene={token} | h=0.5/1/2/4s prefixes | policy=non_reactive\n"
        "GT-factual soft-label probability; candidates c0*=GT; each row sums to 1"
    )
    figure.colorbar(image, ax=axis, label="q(candidate | GT prefix)")
    path = figure_dir / "06_prefix_gt_soft_label_heatmap.png"; _save(figure, path); written.append(str(path))

    # 7. Official locally exposed factor heatmap.
    factors = ["no_at_fault_collision", "dac", "ddc", "progress", "ttc", "comfort", "aggregate_score"]
    matrix = scores[factors].to_numpy(dtype=float)
    figure, axis = plt.subplots(figsize=(10, 7))
    image = axis.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn")
    axis.set_xticks(range(len(factors)), factors, rotation=35, ha="right")
    axis.set_yticks(range(len(scores)), [f"c{int(x)}{'*GT' if bool(y) else ''}" for x, y in zip(scores.candidate_index, scores.is_gt)])
    axis.set_title(_scene_title(token, 4.0, "official v1 PDM factor/aggregate values [0,1]"))
    figure.colorbar(image, ax=axis, label="factor value")
    path = figure_dir / "07_pdm_factor_heatmap.png"; _save(figure, path); written.append(str(path))

    pairs_path = output_dir / "all_candidate_pairs.parquet"
    pairs = read_parquet(pairs_path) if pairs_path.is_file() else read_parquet(output_dir / "hard_negative_pairs.parquet")
    # 8. Trajectory versus consequence distance.
    figure, axis = plt.subplots(figsize=(8, 6))
    scatter = axis.scatter(pairs.trajectory_distance, pairs.consequence_distance, c=pairs.score_difference, s=10, alpha=0.45, cmap="plasma")
    axis.set_xlabel("trajectory distance [normalized]"); axis.set_ylabel("consequence distance [normalized]")
    axis.set_title(f"scene=ALL({pairs.scene_token.nunique()}) | candidates=all pairs | h=4.0s | non_reactive | normalized units")
    figure.colorbar(scatter, ax=axis, label="|PDM score difference|")
    path = figure_dir / "08_trajectory_vs_consequence_distance.png"; _save(figure, path); written.append(str(path))

    # 9. Consequence distance versus score difference.
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(pairs.consequence_distance, pairs.score_difference, c=pairs.trajectory_distance, s=10, alpha=0.45, cmap="viridis")
    axis.set_xlabel("consequence distance [normalized]"); axis.set_ylabel("|PDM score difference| [0,1]")
    axis.set_title(f"scene=ALL({pairs.scene_token.nunique()}) | candidates=all pairs | h=4.0s | policy=non_reactive")
    path = figure_dir / "09_consequence_vs_score_difference.png"; _save(figure, path); written.append(str(path))

    # 10. Reactive availability / comparison. Never fabricate a response trace.
    v2_path = output_dir / "v2_extension_audit.json"
    v2 = json.loads(v2_path.read_text(encoding="utf-8")) if v2_path.is_file() else {}
    figure, axis = plt.subplots(figsize=(10, 4))
    available = [1.0, float(bool(v2.get("reactive_empirical_run", False)))]
    axis.bar(["non_reactive measured", "reactive measured"], available, color=["tab:blue", "tab:orange"])
    axis.set_ylim(0, 1.25); axis.set_ylabel("empirical candidate-response data available")
    axis.text(1, 0.08, str(v2.get("reactive_blocker", "reactive audit not run")), ha="center", va="bottom", wrap=True, fontsize=9)
    axis.set_title("scene=ALL eligible audited tokens | candidates=ALL | h=0–4s | actor-response availability\nNo reactive trace is synthesized for visualization")
    path = figure_dir / "10_nonreactive_vs_reactive_actor_response.png"; _save(figure, path); written.append(str(path))

    coverage = pd.read_csv(output_dir / "scene_coverage.csv")
    # 11. Future sensor availability.
    sensor_fields = ["camera_0p5s_available", "camera_1p0s_available", "camera_2p0s_available", "camera_4p0s_available", "future_lidar_all_requested"]
    values = [float(coverage[field].mean()) if field in coverage else 0.0 for field in sensor_fields]
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(sensor_fields, values); axis.set_ylim(0, 1.05); axis.set_ylabel("scene coverage")
    axis.tick_params(axis="x", rotation=25)
    axis.set_title(f"scene=ALL({len(coverage)}) | candidates=N/A | h=0.5/1/2/4s | sensors=logged paths | coverage units")
    path = figure_dir / "11_future_sensor_availability_histogram.png"; _save(figure, path); written.append(str(path))

    # 12. Track continuity.
    continuity = coverage.track_span_continuity.dropna().to_numpy(dtype=float)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(continuity, bins=np.linspace(0, 1, 21), edgecolor="black")
    axis.set_xlabel("future track span continuity [0,1]"); axis.set_ylabel("scene count")
    axis.set_title(f"scene=ALL({len(continuity)}) | candidates=N/A | h=0.5–4s | global track tokens | measured ~2 Hz")
    path = figure_dir / "12_track_continuity_histogram.png"; _save(figure, path); written.append(str(path))

    write_markdown(
        output_dir / "VISUALIZATION_INDEX.md",
        "# Audit Visualization Index\n\n" + "\n".join(f"- `{Path(item).relative_to(output_dir)}`" for item in written),
    )
    append_command(output_dir, "python -m tools.navsim_candidate_relative_audit.visualize_audit " + " ".join(sys.argv[1:]))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
