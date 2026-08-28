#!/usr/bin/env python3
"""Phase 6: quantify K-candidate consequence diversity and hard negatives."""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .common import (
    HORIZONS_S,
    add_common_arguments,
    robust_standardize,
    wrap_heading,
    write_dataframe,
    write_json,
    write_text,
)


def pairwise_euclidean(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    differences = data[:, None] - data[None, :]
    return np.sqrt(np.mean(differences * differences, axis=-1))


def trajectory_distance(trajectories: np.ndarray) -> np.ndarray:
    values = np.asarray(trajectories, dtype=np.float64)
    position = values[:, None, :, :2] - values[None, :, :, :2]
    position_distance = np.linalg.norm(position, axis=-1).mean(axis=-1)
    heading = np.abs(wrap_heading(values[:, None, :, 2] - values[None, :, :, 2])).mean(
        axis=-1
    )
    return position_distance + 2.0 * heading


def finite_spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return None
    value = spearmanr(left, right).statistic
    return float(value) if np.isfinite(value) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    args = parser.parse_args()
    manifest = pd.read_parquet(
        args.output_dir / "candidate_manifest.parquet"
    ).sort_values(["scene_index", "candidate_index"])
    metrics = pd.read_parquet(
        args.output_dir / "candidate_metrics.parquet"
    ).sort_values(["scene_index", "candidate_index"])
    with np.load(args.output_dir / "candidate_trajectories.npz") as payload:
        trajectories = np.asarray(payload["trajectories"], dtype=np.float32)
    scene_count = min(args.max_scenes, trajectories.shape[0])
    environment_scenes: list[np.ndarray] = []
    scene_tokens: list[str] = []
    for scene_index in range(scene_count):
        token = str(
            manifest[manifest["scene_index"] == scene_index].iloc[0]["scene_token"]
        )
        with np.load(args.output_dir / "targets" / f"{token}.npz") as payload:
            environment_scenes.append(
                np.asarray(payload["C_environment_only"], dtype=np.float32)
            )
        scene_tokens.append(token)
    environments = np.stack(environment_scenes)
    normalized, normalization = robust_standardize(environments)
    target_schema = json.loads((args.output_dir / "target_schema.json").read_text())
    fields = target_schema["arrays"]["C_environment_only"]["fields"]

    diversity_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    hard_rows: list[dict[str, Any]] = []
    all_trajectory_distance: list[float] = []
    all_consequence_distance: list[float] = []
    all_score_difference: list[float] = []
    factor_differences: dict[str, list[float]] = {
        name: []
        for name in (
            "no_at_fault_collision",
            "drivable_area_compliance",
            "driving_direction_compliance",
            "traffic_light_compliance",
            "time_to_collision_within_bound",
            "lane_keeping",
            "ego_progress",
        )
    }
    for scene_index, token in enumerate(scene_tokens):
        scene_manifest = manifest[manifest["scene_index"] == scene_index].sort_values(
            "candidate_index"
        )
        scene_metrics = metrics[metrics["scene_index"] == scene_index].sort_values(
            "candidate_index"
        )
        types = scene_manifest["candidate_type"].astype(str).tolist()
        scores = scene_metrics["aggregate_score"].to_numpy(dtype=float)
        trajectory_matrix = trajectory_distance(trajectories[scene_index])
        horizon_matrices = [
            pairwise_euclidean(normalized[scene_index, :, horizon])
            for horizon in range(len(HORIZONS_S))
        ]
        consequence_matrix = np.mean(horizon_matrices, axis=0)
        upper = np.triu_indices(len(scores), k=1)
        nonzero = consequence_matrix[upper] > 1e-6
        unique_count = len(
            np.unique(
                np.round(normalized[scene_index].reshape(len(scores), -1), 4), axis=0
            )
        )
        field_variance = np.var(environments[scene_index], axis=0).mean(axis=0)
        diversity_rows.append(
            {
                "scene_token": token,
                "horizon_s": "all",
                "pair_count": len(upper[0]),
                "nonzero_pairwise_consequence_ratio": float(np.mean(nonzero)),
                "unique_consequence_count": unique_count,
                "mean_consequence_distance": float(np.mean(consequence_matrix[upper])),
                "max_consequence_distance": float(
                    np.max(consequence_matrix[upper], initial=0.0)
                ),
                "mean_trajectory_distance": float(np.mean(trajectory_matrix[upper])),
                "mean_score_difference": float(
                    np.mean(np.abs(scores[:, None] - scores[None, :])[upper])
                ),
                "active_target_dimension_count": int(np.sum(field_variance > 1e-8)),
                "target_dimension_variance_json": json.dumps(
                    dict(zip(fields, field_variance.tolist())), sort_keys=True
                ),
            }
        )
        for horizon_index, horizon in enumerate(HORIZONS_S):
            matrix = horizon_matrices[horizon_index]
            variance = np.var(environments[scene_index, :, horizon_index], axis=0)
            diversity_rows.append(
                {
                    "scene_token": token,
                    "horizon_s": horizon,
                    "pair_count": len(upper[0]),
                    "nonzero_pairwise_consequence_ratio": float(
                        np.mean(matrix[upper] > 1e-6)
                    ),
                    "unique_consequence_count": len(
                        np.unique(
                            np.round(normalized[scene_index, :, horizon_index], 4),
                            axis=0,
                        )
                    ),
                    "mean_consequence_distance": float(np.mean(matrix[upper])),
                    "max_consequence_distance": float(
                        np.max(matrix[upper], initial=0.0)
                    ),
                    "mean_trajectory_distance": float(
                        np.mean(trajectory_matrix[upper])
                    ),
                    "mean_score_difference": float(
                        np.mean(np.abs(scores[:, None] - scores[None, :])[upper])
                    ),
                    "active_target_dimension_count": int(np.sum(variance > 1e-8)),
                    "target_dimension_variance_json": json.dumps(
                        dict(zip(fields, variance.tolist())), sort_keys=True
                    ),
                }
            )
        trajectory_threshold = float(np.quantile(trajectory_matrix[upper], 0.30))
        consequence_threshold = float(np.quantile(consequence_matrix[upper], 0.70))
        large_trajectory_threshold = float(np.quantile(trajectory_matrix[upper], 0.70))
        score_matrix = np.abs(scores[:, None] - scores[None, :])
        endpoint = trajectories[scene_index, :, -1, :2]
        endpoint_matrix = np.linalg.norm(endpoint[:, None] - endpoint[None, :], axis=-1)
        factor_arrays = {
            name: scene_metrics[name].to_numpy(dtype=float)
            for name in factor_differences
        }
        for left, right in zip(*upper):
            trajectory_value = float(trajectory_matrix[left, right])
            consequence_value = float(consequence_matrix[left, right])
            score_value = float(score_matrix[left, right])
            endpoint_value = float(endpoint_matrix[left, right])
            reasons: list[str] = []
            if (
                trajectory_value <= trajectory_threshold
                and consequence_value >= consequence_threshold
            ):
                reasons.append("trajectory_near_consequence_far")
            if trajectory_value <= trajectory_threshold and (
                scene_metrics.iloc[left]["dynamic_collision"]
                != scene_metrics.iloc[right]["dynamic_collision"]
                or scene_metrics.iloc[left]["ttc_violation_observed"]
                != scene_metrics.iloc[right]["ttc_violation_observed"]
            ):
                reasons.append("geometry_near_collision_or_ttc_diff")
            if trajectory_value <= trajectory_threshold and any(
                factor_arrays[name][left] != factor_arrays[name][right]
                for name in (
                    "drivable_area_compliance",
                    "traffic_light_compliance",
                    "lane_keeping",
                )
            ):
                reasons.append("geometry_near_map_or_tlc_diff")
            if endpoint_value <= 0.5 and consequence_value >= consequence_threshold:
                reasons.append("endpoint_near_mid_risk_diff")
            if trajectory_value >= large_trajectory_threshold and score_value <= 0.01:
                reasons.append("trajectory_far_score_near")
            pair = {
                "scene_token": token,
                "candidate_i": int(left),
                "candidate_j": int(right),
                "candidate_i_type": types[left],
                "candidate_j_type": types[right],
                "trajectory_distance": trajectory_value,
                "endpoint_distance_m": endpoint_value,
                "consequence_distance": consequence_value,
                "score_difference": score_value,
            }
            pair_rows.append(pair)
            if reasons:
                hard_rows.append({**pair, "hard_negative_reasons": ";".join(reasons)})
            all_trajectory_distance.append(trajectory_value)
            all_consequence_distance.append(consequence_value)
            all_score_difference.append(score_value)
            for name, values in factor_arrays.items():
                factor_differences[name].append(
                    float(abs(values[left] - values[right]))
                )

    summary = {
        "scene_count": scene_count,
        "candidates_per_scene": trajectories.shape[1],
        "undirected_pair_count": len(pair_rows),
        "directed_O_K2_relations": scene_count
        * trajectories.shape[1]
        * trajectories.shape[1],
        "mean_nonzero_pairwise_consequence_ratio": float(
            pd.DataFrame(diversity_rows)
            .query("horizon_s == 'all'")["nonzero_pairwise_consequence_ratio"]
            .mean()
        ),
        "trajectory_consequence_spearman": finite_spearman(
            all_trajectory_distance, all_consequence_distance
        ),
        "consequence_score_difference_spearman": finite_spearman(
            all_consequence_distance, all_score_difference
        ),
        "trajectory_score_difference_spearman": finite_spearman(
            all_trajectory_distance, all_score_difference
        ),
        "consequence_factor_difference_spearman": {
            name: finite_spearman(all_consequence_distance, values)
            for name, values in factor_differences.items()
        },
        "hard_negative_pair_count": len(hard_rows),
        "normalization": normalization,
        "nondegenerate": bool(
            len(pair_rows)
            and np.mean(np.asarray(all_consequence_distance) > 1e-6) > 0.5
            and len(hard_rows) > 0
        ),
        "evidence_scene_tokens": scene_tokens[:16],
    }
    write_dataframe(
        pd.DataFrame(diversity_rows), args.output_dir / "target_diversity.csv"
    )
    write_dataframe(
        pd.DataFrame(pair_rows), args.output_dir / "target_pairwise_distances.parquet"
    )
    hard_frame = pd.DataFrame(hard_rows)
    if hard_frame.empty:
        hard_frame = pd.DataFrame(
            columns=[
                "scene_token",
                "candidate_i",
                "candidate_j",
                "candidate_i_type",
                "candidate_j_type",
                "trajectory_distance",
                "endpoint_distance_m",
                "consequence_distance",
                "score_difference",
                "hard_negative_reasons",
            ]
        )
    write_dataframe(hard_frame, args.output_dir / "hard_negative_pairs.parquet")
    write_json(args.output_dir / "target_diversity_summary.json", summary)
    write_text(
        args.output_dir / "TARGET_DIVERSITY_REPORT.md",
        "\n".join(
            [
                "# Candidate-Relative Target Diversity",
                "",
                f"- Non-degenerate: **{summary['nondegenerate']}**",
                f"- Scenes / candidates: **{scene_count} / {trajectories.shape[1]}**",
                f"- Undirected candidate pairs: **{summary['undirected_pair_count']}**",
                f"- Directed candidate×candidate relations: **{summary['directed_O_K2_relations']}**",
                f"- Non-zero consequence-pair ratio: **{summary['mean_nonzero_pairwise_consequence_ratio']:.3%}**",
                f"- Trajectory/consequence Spearman: `{summary['trajectory_consequence_spearman']}`",
                f"- Consequence/score-difference Spearman: `{summary['consequence_score_difference_spearman']}`",
                f"- Mined hard-negative pairs: **{summary['hard_negative_pair_count']}**",
                "",
                "Distances use robustly standardized `C_environment_only`; no official PDM score or factor is inside the consequence vector. Pairwise score/factor differences are used only for post-hoc validation.",
                "",
                "Evidence tokens: "
                + ", ".join(f"`{token}`" for token in scene_tokens[:8]),
                "",
            ]
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
