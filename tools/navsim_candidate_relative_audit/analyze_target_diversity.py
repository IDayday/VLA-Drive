#!/usr/bin/env python3
"""Quantify O(K^2) diversity in candidate-relative consequences."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from .common import (
    add_common_arguments,
    append_command,
    ensure_output_dir,
    read_parquet,
    wrap_angle,
    write_json,
    write_markdown,
    write_parquet,
)
from .build_candidate_relative_targets import ACTOR_FEATURES, ENVIRONMENT_FEATURES


ENV_SCALES = np.asarray([5, 20, 5, 1, 5, 1, 1, 1, 2, 0.5, 20, 20, 1, 20, 10], dtype=np.float64)
ACTOR_SCALES = np.asarray([1, 20, 20, 10, 10, np.pi, 5, 2, 5, 1], dtype=np.float64)


def trajectory_distance_matrix(poses: np.ndarray, prefix_steps: int | None = None) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    if prefix_steps is not None:
        poses = poses[:, :prefix_steps]
    k = len(poses)
    result = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        for j in range(i + 1, k):
            position = np.linalg.norm(poses[i, :, :2] - poses[j, :, :2], axis=-1) / 2.0
            heading = np.abs(wrap_angle(poses[i, :, 2] - poses[j, :, 2])) / 0.25
            distance = float(np.mean(position + 0.5 * heading))
            result[i, j] = result[j, i] = distance
    return result


def masked_actor_distance(
    actor_values_i: np.ndarray,
    actor_mask_i: np.ndarray,
    actor_hash_i: np.ndarray,
    actor_values_j: np.ndarray,
    actor_mask_j: np.ndarray,
    actor_hash_j: np.ndarray,
) -> float:
    """Symmetric distance matched by stable track hash with unmatched-set penalty."""

    values = []
    penalties = []
    for time_index in range(actor_values_i.shape[0]):
        ids_i = actor_hash_i[time_index][actor_mask_i[time_index]].tolist()
        ids_j = actor_hash_j[time_index][actor_mask_j[time_index]].tolist()
        map_i = {
            int(token): actor_values_i[time_index, slot]
            for slot, token in enumerate(actor_hash_i[time_index])
            if actor_mask_i[time_index, slot]
        }
        map_j = {
            int(token): actor_values_j[time_index, slot]
            for slot, token in enumerate(actor_hash_j[time_index])
            if actor_mask_j[time_index, slot]
        }
        common = sorted(set(ids_i) & set(ids_j))
        for token in common:
            delta = np.abs(map_i[token] - map_j[token]) / ACTOR_SCALES
            # Object type, size and dimensions are identical for a matched actor;
            # retaining them makes accidental token mismatch visible.
            values.append(float(np.mean(delta)))
        union_size = len(set(ids_i) | set(ids_j))
        penalties.append((union_size - len(common)) / max(union_size, 1))
    return float(np.mean(values) if values else 0.0) + float(np.mean(penalties))


def consequence_distance_matrix(arrays: Any, prefix_steps: int | None = None) -> np.ndarray:
    env = np.asarray(arrays["C_environment_only"], dtype=np.float64)
    actor = np.asarray(arrays["candidate_relative_actor"], dtype=np.float64)
    mask = np.asarray(arrays["candidate_relative_actor_mask"], dtype=bool)
    hashes = np.asarray(arrays["candidate_relative_actor_token_hash"], dtype=np.int64)
    if prefix_steps is not None:
        env = env[:, :prefix_steps]
        actor = actor[:, :prefix_steps]
        mask = mask[:, :prefix_steps]
        hashes = hashes[:, :prefix_steps]
    k = len(env)
    result = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        for j in range(i + 1, k):
            env_distance = float(np.mean(np.abs(env[i] - env[j]) / ENV_SCALES))
            actor_distance = masked_actor_distance(
                actor[i], mask[i], hashes[i], actor[j], mask[j], hashes[j]
            )
            distance = env_distance + 0.5 * actor_distance
            result[i, j] = result[j, i] = distance
    return result


def _safe_corr(x: list[float], y: list[float], method: str) -> float | None:
    a, b = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3 or np.ptp(a[valid]) == 0 or np.ptp(b[valid]) == 0:
        return None
    value = pearsonr(a[valid], b[valid]).statistic if method == "pearson" else spearmanr(a[valid], b[valid]).statistic
    return float(value)


def analyze(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    output_dir = ensure_output_dir(args.output_dir)
    index = read_parquet(output_dir / "targets/index.parquet")
    manifest = read_parquet(output_dir / "candidate_manifest.parquet")
    metrics = read_parquet(output_dir / "candidate_metrics.parquet")
    valid_metrics = metrics[metrics["scoring_success"] & (metrics["traffic_policy"] == "non_reactive")]
    if args.max_scenes > 0:
        index = index.head(args.max_scenes)
    diversity_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    all_traj, all_consequence, all_score = [], [], []
    factor_names = ["no_at_fault_collision", "dac", "ddc", "progress", "ttc", "comfort"]
    all_factor_differences = {name: [] for name in factor_names}

    for index_row in index.itertuples():
        token = index_row.scene_token
        arrays = np.load(output_dir / index_row.target_path)
        group = manifest[manifest["scene_token"] == token].sort_values("candidate_index")
        score_group = valid_metrics[valid_metrics["scene_token"] == token].sort_values("candidate_index")
        if len(group) != len(score_group) or len(group) != arrays["C_environment_only"].shape[0]:
            continue
        poses = np.asarray(
            [np.column_stack([row.pose_x_m, row.pose_y_m, row.pose_heading_rad]) for row in group.itertuples()],
            dtype=np.float64,
        )
        traj = trajectory_distance_matrix(poses)
        consequence = consequence_distance_matrix(arrays)
        scores = score_group["aggregate_score"].to_numpy(dtype=np.float64)
        score_diff = np.abs(scores[:, None] - scores[None, :])
        upper = np.triu_indices(len(group), 1)
        candidate_variance = np.var(arrays["C_environment_only"], axis=0).mean(axis=0)
        row = {
            "scene_token": token,
            "log_name": group.iloc[0].log_name,
            "candidate_count": len(group),
            "pair_count": len(upper[0]),
            "mean_trajectory_distance": float(np.mean(traj[upper])),
            "mean_consequence_distance": float(np.mean(consequence[upper])),
            "nonzero_consequence_pair_ratio": float(np.mean(consequence[upper] > 1e-8)),
            "unique_consequence_count": int(
                len({tuple(np.round(item, 4).ravel()) for item in arrays["C_environment_only"]})
            ),
            "trajectory_consequence_spearman": _safe_corr(traj[upper].tolist(), consequence[upper].tolist(), "spearman"),
            "consequence_score_difference_spearman": _safe_corr(consequence[upper].tolist(), score_diff[upper].tolist(), "spearman"),
        }
        row.update({f"variance_{name}": float(value) for name, value in zip(ENVIRONMENT_FEATURES, candidate_variance)})
        diversity_rows.append(row)
        for i, j in zip(*upper):
            endpoint_distance = float(np.linalg.norm(poses[i, -1, :2] - poses[j, -1, :2]))
            prefix_2s = float(trajectory_distance_matrix(poses[[i, j]], prefix_steps=4)[0, 1])
            pair = {
                "scene_token": token,
                "log_name": group.iloc[0].log_name,
                "candidate_i": int(group.iloc[i].candidate_index),
                "candidate_j": int(group.iloc[j].candidate_index),
                "candidate_type_i": group.iloc[i].candidate_type,
                "candidate_type_j": group.iloc[j].candidate_type,
                "is_gt_i": bool(group.iloc[i].is_gt),
                "is_gt_j": bool(group.iloc[j].is_gt),
                "trajectory_distance": float(traj[i, j]),
                "trajectory_prefix_2s_distance": prefix_2s,
                "endpoint_distance_m": endpoint_distance,
                "consequence_distance": float(consequence[i, j]),
                "score_difference": float(score_diff[i, j]),
                "score_i": float(scores[i]),
                "score_j": float(scores[j]),
            }
            for name in factor_names:
                pair[f"{name}_difference"] = float(abs(score_group.iloc[i][name] - score_group.iloc[j][name]))
                all_factor_differences[name].append(pair[f"{name}_difference"])
            pair_rows.append(pair)
        all_traj.extend(traj[upper].tolist())
        all_consequence.extend(consequence[upper].tolist())
        all_score.extend(score_diff[upper].tolist())

    diversity = pd.DataFrame(diversity_rows)
    diversity.to_csv(output_dir / "target_diversity.csv", index=False)
    pairs = pd.DataFrame(pair_rows)
    # Retain the complete O(K^2) audit table for reproducible correlations and
    # scatter plots; ``hard_negative_pairs`` remains the compact mined subset.
    write_parquet(pairs, output_dir / "all_candidate_pairs.parquet")
    if len(pairs):
        close_t = float(pairs.trajectory_distance.quantile(0.30))
        far_t = float(pairs.trajectory_distance.quantile(0.75))
        high_c = float(pairs.consequence_distance.quantile(0.70))
        low_score = float(pairs.score_difference.quantile(0.25))
        high_score = float(pairs.score_difference.quantile(0.70))
        endpoint_close = float(pairs.endpoint_distance_m.quantile(0.25))
        categories = []
        for row in pairs.itertuples():
            labels = []
            if row.trajectory_distance <= close_t and (row.no_at_fault_collision_difference > 0 or row.ttc_difference > 0):
                labels.append("geometry_close_collision_or_ttc_diff")
            if row.trajectory_distance <= close_t and row.dac_difference > 0:
                labels.append("geometry_close_dac_diff")
            if row.endpoint_distance_m <= endpoint_close and row.consequence_distance >= high_c:
                labels.append("endpoint_close_intermediate_consequence_diff")
            if (row.is_gt_i or row.is_gt_j) and row.trajectory_prefix_2s_distance <= close_t and row.score_difference >= high_score:
                labels.append("gt_prefix_close_tail_outcome_diff")
            if row.trajectory_distance >= far_t and row.score_difference <= low_score:
                labels.append("geometry_far_evaluation_close")
            if row.trajectory_distance <= close_t and row.consequence_distance >= high_c:
                labels.append("trajectory_ambiguous_consequence_distinct")
            categories.append(";".join(labels))
        pairs["hard_negative_category"] = categories
        hard = pairs[pairs["hard_negative_category"] != ""].copy()
        if len(hard) == 0:
            hard = pairs.nlargest(min(32, len(pairs)), "consequence_distance").copy()
            hard["hard_negative_category"] = "high_consequence_distance_fallback"
    else:
        hard = pairs.assign(hard_negative_category=pd.Series(dtype=str))
    write_parquet(hard, output_dir / "hard_negative_pairs.parquet")
    summary = {
        "scene_count": len(diversity),
        "pair_count": len(pairs),
        "hard_negative_count": len(hard),
        "nonzero_pair_ratio": float(np.mean(np.asarray(all_consequence) > 1e-8)) if all_consequence else 0.0,
        "trajectory_consequence_pearson": _safe_corr(all_traj, all_consequence, "pearson"),
        "trajectory_consequence_spearman": _safe_corr(all_traj, all_consequence, "spearman"),
        "consequence_score_difference_pearson": _safe_corr(all_consequence, all_score, "pearson"),
        "consequence_score_difference_spearman": _safe_corr(all_consequence, all_score, "spearman"),
        "consequence_factor_difference_spearman": {
            name: _safe_corr(all_consequence, values, "spearman") for name, values in all_factor_differences.items()
        },
        "mean_unique_consequence_count": float(diversity.unique_consequence_count.mean()) if len(diversity) else 0.0,
        "evidence_scene_tokens": diversity.scene_token.head(8).tolist() if len(diversity) else [],
        "distance_definition": {
            "environment": "mean absolute standardized C_environment_only difference over the prefix",
            "actors": "track-hash matched standardized actor relation distance plus symmetric unmatched-set penalty",
            "combined": "environment + 0.5 * actor",
        },
    }
    write_json(output_dir / "target_diversity_summary.json", summary)
    report = f"""# Candidate-relative Target Diversity

- Scenes / candidate pairs: {len(diversity)} / {len(pairs)}
- Non-zero pairwise consequence distance: {summary['nonzero_pair_ratio']:.3%}
- Mean unique consequences per scene: {summary['mean_unique_consequence_count']:.2f}
- Saved hard-negative pairs: {len(hard)}
- Trajectory vs consequence Spearman: {summary['trajectory_consequence_spearman']}
- Consequence distance vs PDM-score difference Spearman: {summary['consequence_score_difference_spearman']}

The actor component matches stable track hashes and explicitly penalizes actors present in only one nearest-N set, so masks and unequal actor sets do not silently compare unrelated slots.  O(K²) supervision is non-degenerate when the non-zero ratio and per-scene unique counts exceed their trivial values; this report records both rather than inferring diversity from trajectory perturbations alone.

Hard-negative categories include close geometry with collision/TTC or DAC differences, close endpoints with different intermediate consequences, same/close GT prefixes with divergent tails, and geometrically distant candidates with similar evaluation.
"""
    write_markdown(output_dir / "TARGET_DIVERSITY_REPORT.md", report)
    return diversity, hard, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    args = parser.parse_args()
    analyze(args)
    append_command(args.output_dir.resolve(), "python -m tools.navsim_candidate_relative_audit.analyze_target_diversity " + " ".join(sys.argv[1:]))


if __name__ == "__main__":
    main()
