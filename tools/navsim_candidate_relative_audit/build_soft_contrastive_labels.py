#!/usr/bin/env python3
"""Prototype prefix-aware factual and KxK consequence soft labels."""

from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np
import pandas as pd

from .analyze_target_diversity import consequence_distance_matrix
from .common import (
    add_common_arguments,
    append_command,
    ensure_output_dir,
    read_parquet,
    wrap_angle,
    write_json,
    write_markdown,
)


HORIZONS = (0.5, 1.0, 2.0, 4.0)


def softmax_negative(distance: np.ndarray, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-6)
    logits = -np.asarray(distance, dtype=np.float64) / sigma
    logits -= np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.maximum(exp.sum(axis=-1, keepdims=True), 1e-12)


def prefix_distance_matrix(
    poses: np.ndarray,
    horizon_s: float,
    *,
    position_scale_m: float = 2.0,
    heading_scale_rad: float = 0.25,
    speed_scale_mps: float = 3.0,
    heading_weight: float = 0.5,
    speed_weight: float = 0.25,
) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    times = np.arange(0.5, 0.5 * poses.shape[1] + 1e-8, 0.5)
    keep = times <= horizon_s + 1e-8
    if not keep.any():
        raise ValueError(f"No trajectory prefix at horizon {horizon_s}")
    prefix = poses[:, keep]
    xy0 = np.zeros((len(poses), 1, 2), dtype=np.float64)
    speed = np.linalg.norm(np.diff(np.concatenate([xy0, poses[:, :, :2]], axis=1), axis=1), axis=-1) / 0.5
    speed = speed[:, keep]
    k = len(poses)
    result = np.zeros((k, k), dtype=np.float64)
    weights = np.linspace(1.0, 1.5, prefix.shape[1])
    weights /= weights.sum()
    for i in range(k):
        for j in range(i + 1, k):
            position = np.linalg.norm(prefix[i, :, :2] - prefix[j, :, :2], axis=-1) / position_scale_m
            heading = np.abs(wrap_angle(prefix[i, :, 2] - prefix[j, :, 2])) / heading_scale_rad
            velocity = np.abs(speed[i] - speed[j]) / speed_scale_mps
            distance = float(np.sum(weights * (position + heading_weight * heading + speed_weight * velocity)))
            result[i, j] = result[j, i] = distance
    return result


def _sigma_from_distances(values: np.ndarray, floor: float = 0.05) -> float:
    positive = np.asarray(values, dtype=np.float64)
    positive = positive[np.isfinite(positive) & (positive > 1e-10)]
    return max(float(np.median(positive)) if len(positive) else floor, floor)


def build(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    output_dir = ensure_output_dir(args.output_dir)
    index = read_parquet(output_dir / "targets/index.parquet")
    manifest = read_parquet(output_dir / "candidate_manifest.parquet")
    if args.max_scenes > 0:
        index = index.head(args.max_scenes)
    rows = []
    examples: dict[str, Any] = {
        "definition": {
            "gt_factual": "softmax(-prefix trajectory distance to GT / sigma_h)",
            "candidate_consequence": "row-wise softmax(-masked consequence distance / sigma_C_h)",
        },
        "scenes": [],
        "same_prefix_different_tail": [],
    }
    short_positive_checks, long_separation_checks = [], []
    for scene_number, index_row in enumerate(index.itertuples()):
        token = index_row.scene_token
        arrays = np.load(output_dir / index_row.target_path)
        group = manifest[manifest["scene_token"] == token].sort_values("candidate_index")
        poses = np.asarray(
            [np.column_stack([row.pose_x_m, row.pose_y_m, row.pose_heading_rad]) for row in group.itertuples()],
            dtype=np.float64,
        )
        gt_indices = np.flatnonzero(group["is_gt"].to_numpy())
        if len(gt_indices) != 1:
            raise RuntimeError(f"Scene {token} has {len(gt_indices)} GT candidates")
        gt_index = int(gt_indices[0])
        scene_example: dict[str, Any] = {"scene_token": token, "gt_candidate_index": gt_index, "horizons": {}}
        for horizon in HORIZONS:
            steps = int(round(horizon / 0.5))
            trajectory_distance = prefix_distance_matrix(poses, horizon)
            gt_distance = trajectory_distance[:, gt_index]
            sigma_h = _sigma_from_distances(gt_distance)
            q = softmax_negative(gt_distance[None, :], sigma_h)[0]
            consequence_distance = consequence_distance_matrix(arrays, prefix_steps=steps)
            upper = consequence_distance[np.triu_indices(len(group), 1)]
            sigma_c = _sigma_from_distances(upper)
            q_matrix = softmax_negative(consequence_distance, sigma_c)
            entropy = float(-np.sum(q * np.log(np.maximum(q, 1e-12))))
            effective = float(np.exp(entropy))
            top = np.argsort(q)[::-1]
            false_negative_count = int(np.sum(q[np.arange(len(q)) != gt_index] >= args.soft_positive_threshold))
            rows.append(
                {
                    "scene_token": token,
                    "log_name": group.iloc[0].log_name,
                    "horizon_s": horizon,
                    "sigma_trajectory": sigma_h,
                    "sigma_consequence": sigma_c,
                    "gt_weight": float(q[gt_index]),
                    "top2_candidate": int(group.iloc[top[1]].candidate_index) if len(top) > 1 else None,
                    "top2_weight": float(q[top[1]]) if len(top) > 1 else None,
                    "top3_candidate": int(group.iloc[top[2]].candidate_index) if len(top) > 2 else None,
                    "top3_weight": float(q[top[2]]) if len(top) > 2 else None,
                    "label_entropy": entropy,
                    "effective_positive_count": effective,
                    "hard_one_hot_false_negative_count": false_negative_count,
                    "consequence_row_entropy_mean": float(
                        np.mean(-np.sum(q_matrix * np.log(np.maximum(q_matrix, 1e-12)), axis=1))
                    ),
                    "consequence_offdiagonal_mass_mean": float(np.mean(1.0 - np.diag(q_matrix))),
                    "valid": bool(np.isfinite(q).all() and np.isfinite(q_matrix).all()),
                }
            )
            if scene_number < 8:
                scene_example["horizons"][str(horizon)] = {
                    "trajectory_distance_to_gt": gt_distance.tolist(),
                    "gt_soft_label": q.tolist(),
                    "consequence_distance": consequence_distance.tolist(),
                    "candidate_consequence_soft_label": q_matrix.tolist(),
                }

        tail_positions = np.flatnonzero(group["candidate_type"].to_numpy() == "same_prefix_different_tail")
        if len(tail_positions):
            tail_index = int(tail_positions[0])
            distances = {str(h): float(prefix_distance_matrix(poses, h)[gt_index, tail_index]) for h in HORIZONS}
            q_short = softmax_negative(
                prefix_distance_matrix(poses, 1.0)[gt_index : gt_index + 1],
                _sigma_from_distances(prefix_distance_matrix(poses, 1.0)[gt_index]),
            )[0, tail_index]
            q_long = softmax_negative(
                prefix_distance_matrix(poses, 4.0)[gt_index : gt_index + 1],
                _sigma_from_distances(prefix_distance_matrix(poses, 4.0)[gt_index]),
            )[0, tail_index]
            check = {
                "scene_token": token,
                "candidate_i": gt_index,
                "candidate_j": tail_index,
                "prefix_distances": distances,
                "short_horizon_row_weight": float(q_short),
                "long_horizon_row_weight": float(q_long),
                "short_prefix_equal": distances["1.0"] < args.same_prefix_tolerance,
                "long_tail_separated": distances["4.0"] > distances["2.0"] + 1e-3,
            }
            examples["same_prefix_different_tail"].append(check)
            short_positive_checks.append(check["short_prefix_equal"])
            long_separation_checks.append(check["long_tail_separated"])
        if scene_number < 8:
            examples["scenes"].append(scene_example)

    stats = pd.DataFrame(rows)
    stats.to_csv(output_dir / "soft_label_stats.csv", index=False)
    examples["same_prefix_short_positive_rate"] = float(np.mean(short_positive_checks)) if short_positive_checks else None
    examples["same_prefix_long_separation_rate"] = float(np.mean(long_separation_checks)) if long_separation_checks else None
    write_json(output_dir / "soft_label_examples.json", examples)
    horizon_summary = stats.groupby("horizon_s").agg(
        scenes=("scene_token", "nunique"),
        mean_gt_weight=("gt_weight", "mean"),
        mean_entropy=("label_entropy", "mean"),
        mean_effective_positive_count=("effective_positive_count", "mean"),
        mean_false_negative_count=("hard_one_hot_false_negative_count", "mean"),
        mean_consequence_offdiagonal_mass=("consequence_offdiagonal_mass_mean", "mean"),
    )
    header = ["horizon_s", *horizon_summary.columns.tolist()]
    table_lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---:" for _ in header) + "|",
    ]
    for horizon, values in horizon_summary.iterrows():
        rendered = [f"{float(horizon):.1f}"] + [
            f"{float(value):.4f}" if np.isfinite(value) else "NA" for value in values
        ]
        table_lines.append("| " + " | ".join(rendered) + " |")
    table = "\n".join(table_lines)
    report = f"""# Prefix-aware Soft Contrastive Label Audit

Every horizon uses only trajectory and consequence prefixes at or before that horizon.  No later-tail waypoint enters a short-horizon label.

{table}

Same-prefix/different-tail checks: short-prefix equality rate `{examples['same_prefix_short_positive_rate']}`; longer-horizon separation rate `{examples['same_prefix_long_separation_rate']}`.  A hard one-hot label would treat every non-GT candidate as equally negative; `mean_false_negative_count` reports how many non-GT candidates retain at least {args.soft_positive_threshold:.3f} factual probability and would therefore be false negatives under that rule.

The K×K consequence labels combine standardized environment relations with actor features matched by stable track hash, valid masks, and an unmatched-set penalty. Binary risks and continuous distances are normalized separately through fixed per-feature scales.
"""
    write_markdown(output_dir / "SOFT_CONTRASTIVE_AUDIT.md", report)
    if not bool(stats.valid.all()) or not all(short_positive_checks) or not all(long_separation_checks):
        raise SystemExit("Soft-label validation failed; see soft_label_examples.json")
    return stats, examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--soft-positive-threshold", type=float, default=0.05)
    parser.add_argument(
        "--same-prefix-tolerance",
        type=float,
        default=0.05,
        help="Normalized prefix-distance tolerance; allows logged-heading versus geometric-tangent noise.",
    )
    args = parser.parse_args()
    build(args)
    append_command(args.output_dir.resolve(), "python -m tools.navsim_candidate_relative_audit.build_soft_contrastive_labels " + " ".join(sys.argv[1:]))


if __name__ == "__main__":
    main()
