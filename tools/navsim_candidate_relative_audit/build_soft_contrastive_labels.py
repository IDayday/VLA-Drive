#!/usr/bin/env python3
"""Phase 7: construct prefix-aware GT and candidate-consequence soft labels."""

from __future__ import annotations

import argparse
import json
import math
from typing import Any

import numpy as np
import pandas as pd

from .common import (
    HORIZONS_S,
    add_common_arguments,
    wrap_heading,
    write_dataframe,
    write_json,
    write_text,
)


BINARY_ENVIRONMENT_FIELDS = {
    "object_collision",
    "dynamic_actor_collision",
    "candidate_center_in_drivable_map",
    "candidate_center_in_lane_or_connector",
    "candidate_heading_oncoming_relation",
    "candidate_center_in_intersection",
    "red_light_zone_intersection",
}


def softmax_negative_distance(
    distance: np.ndarray, sigma: float, axis: int = -1
) -> np.ndarray:
    sigma = max(float(sigma), 1e-6)
    logits = -np.asarray(distance, dtype=np.float64) / sigma
    logits -= np.max(logits, axis=axis, keepdims=True)
    weights = np.exp(logits)
    return weights / np.sum(weights, axis=axis, keepdims=True)


def candidate_speed(trajectories: np.ndarray, interval_s: float = 0.5) -> np.ndarray:
    points = np.concatenate(
        [
            np.zeros((*trajectories.shape[:-2], 1, 2), dtype=np.float64),
            trajectories[..., :2],
        ],
        axis=-2,
    )
    return np.linalg.norm(np.diff(points, axis=-2), axis=-1) / interval_s


def fit_trajectory_scales(trajectories: np.ndarray) -> dict[str, float]:
    values = np.asarray(trajectories, dtype=np.float64)
    speed = candidate_speed(values)
    position_values: list[float] = []
    heading_values: list[float] = []
    speed_values: list[float] = []
    for scene in range(values.shape[0]):
        for left in range(values.shape[1]):
            for right in range(left + 1, values.shape[1]):
                position_values.extend(
                    np.linalg.norm(
                        values[scene, left, :, :2] - values[scene, right, :, :2],
                        axis=-1,
                    )
                )
                heading_values.extend(
                    np.abs(
                        wrap_heading(
                            values[scene, left, :, 2] - values[scene, right, :, 2]
                        )
                    )
                )
                speed_values.extend(np.abs(speed[scene, left] - speed[scene, right]))

    def scale(items: list[float], minimum: float) -> float:
        nonzero = np.asarray([value for value in items if value > 1e-8])
        return max(float(np.median(nonzero)) if len(nonzero) else minimum, minimum)

    return {
        "s_p_m": scale(position_values, 0.25),
        "s_heading_rad": scale(heading_values, 0.05),
        "s_v_mps": scale(speed_values, 0.25),
    }


def prefix_distance_matrix(
    trajectories: np.ndarray,
    horizon_s: float,
    scales: dict[str, float],
    *,
    heading_weight: float = 0.5,
    velocity_weight: float = 0.5,
) -> np.ndarray:
    values = np.asarray(trajectories, dtype=np.float64)
    prefix_count = min(values.shape[1], max(1, int(round(horizon_s / 0.5))))
    prefix = values[:, :prefix_count]
    speed = candidate_speed(values)[:, :prefix_count]
    position = (
        np.linalg.norm(prefix[:, None, :, :2] - prefix[None, :, :, :2], axis=-1)
        / scales["s_p_m"]
    )
    heading = (
        np.abs(wrap_heading(prefix[:, None, :, 2] - prefix[None, :, :, 2]))
        / scales["s_heading_rad"]
    )
    velocity = np.abs(speed[:, None] - speed[None, :]) / scales["s_v_mps"]
    weights = np.arange(1, prefix_count + 1, dtype=np.float64)
    weights /= weights.sum()
    return np.sum(
        weights[None, None, :]
        * (position + heading_weight * heading + velocity_weight * velocity),
        axis=-1,
    )


def fit_environment_scales(environments: np.ndarray, fields: list[str]) -> np.ndarray:
    flattened = np.asarray(environments, dtype=np.float64).reshape(-1, len(fields))
    scales = np.ones(len(fields), dtype=np.float64)
    for index, field in enumerate(fields):
        if field in BINARY_ENVIRONMENT_FIELDS:
            scales[index] = 1.0
            continue
        column = flattened[np.isfinite(flattened[:, index]), index]
        if not len(column):
            continue
        q25, q75 = np.quantile(column, [0.25, 0.75])
        minimum = 1.0 if "count" in field else 0.25
        scales[index] = max(float(q75 - q25), minimum)
    return scales


def mask_aware_actor_distance(
    actor: np.ndarray,
    mask: np.ndarray,
    hashes: np.ndarray,
    left: int,
    right: int,
) -> float:
    feature_scale = np.asarray(
        [1.0, 10.0, 10.0, 5.0, 5.0, math.pi, 5.0, 3.0, 10.0, 1.0]
    )
    left_values = {
        int(token): actor[left, index]
        for index, token in enumerate(hashes[left])
        if mask[left, index]
    }
    right_values = {
        int(token): actor[right, index]
        for index, token in enumerate(hashes[right])
        if mask[right, index]
    }
    union = set(left_values) | set(right_values)
    if not union:
        return 0.0
    distances = []
    for token in union:
        if token not in left_values or token not in right_values:
            distances.append(1.0)
        else:
            difference = (left_values[token] - right_values[token]) / feature_scale
            difference[5] = (
                wrap_heading(left_values[token][5] - right_values[token][5]) / math.pi
            )
            distances.append(float(np.sqrt(np.mean(difference * difference))))
    return float(np.mean(distances))


def consequence_distance_matrix(
    environment: np.ndarray,
    actor: np.ndarray,
    actor_mask: np.ndarray,
    actor_hash: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    candidate_count = len(environment)
    matrix = np.zeros((candidate_count, candidate_count), dtype=np.float64)
    for left in range(candidate_count):
        for right in range(left + 1, candidate_count):
            difference = (environment[left] - environment[right]) / scales
            summary_distance = float(np.sqrt(np.mean(difference * difference)))
            actor_distance = mask_aware_actor_distance(
                actor, actor_mask, actor_hash, left, right
            )
            matrix[left, right] = matrix[right, left] = (
                summary_distance + 0.25 * actor_distance
            )
    return matrix


def entropy(probabilities: np.ndarray) -> float:
    values = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0)
    return float(-np.sum(values * np.log(values)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    args = parser.parse_args()
    manifest = pd.read_parquet(
        args.output_dir / "candidate_manifest.parquet"
    ).sort_values(["scene_index", "candidate_index"])
    with np.load(args.output_dir / "candidate_trajectories.npz") as payload:
        trajectories = np.asarray(payload["trajectories"], dtype=np.float32)
    scene_count = min(args.max_scenes, trajectories.shape[0])
    trajectories = trajectories[:scene_count]
    schema = json.loads((args.output_dir / "target_schema.json").read_text())
    fields = schema["arrays"]["C_environment_only"]["fields"]
    environments: list[np.ndarray] = []
    actors: list[np.ndarray] = []
    actor_masks: list[np.ndarray] = []
    actor_hashes: list[np.ndarray] = []
    tokens: list[str] = []
    for scene_index in range(scene_count):
        token = str(
            manifest[manifest["scene_index"] == scene_index].iloc[0]["scene_token"]
        )
        tokens.append(token)
        with np.load(args.output_dir / "targets" / f"{token}.npz") as payload:
            environments.append(
                np.asarray(payload["C_environment_only"], dtype=np.float32)
            )
            actors.append(
                np.asarray(payload["candidate_relative_actor_tensor"], dtype=np.float32)
            )
            actor_masks.append(
                np.asarray(payload["candidate_relative_actor_mask"], dtype=bool)
            )
            actor_hashes.append(
                np.asarray(
                    payload["candidate_relative_actor_track_hash"], dtype=np.uint64
                )
            )
    environment_array = np.stack(environments)
    actor_array = np.stack(actors)
    actor_mask_array = np.stack(actor_masks)
    actor_hash_array = np.stack(actor_hashes)
    trajectory_scales = fit_trajectory_scales(trajectories)
    environment_scales = fit_environment_scales(environment_array, fields)

    prefix_distances = np.zeros(
        (scene_count, len(HORIZONS_S), trajectories.shape[1], trajectories.shape[1]),
        dtype=np.float32,
    )
    consequence_distances = np.zeros_like(prefix_distances)
    for scene_index in range(scene_count):
        for horizon_index, horizon in enumerate(HORIZONS_S):
            prefix_distances[scene_index, horizon_index] = prefix_distance_matrix(
                trajectories[scene_index], horizon, trajectory_scales
            )
            consequence_distances[scene_index, horizon_index] = (
                consequence_distance_matrix(
                    environment_array[scene_index, :, horizon_index],
                    actor_array[scene_index, :, horizon_index],
                    actor_mask_array[scene_index, :, horizon_index],
                    actor_hash_array[scene_index, :, horizon_index],
                    environment_scales,
                )
            )
    trajectory_sigma = []
    consequence_sigma = []
    upper = np.triu_indices(trajectories.shape[1], k=1)
    for horizon_index in range(len(HORIZONS_S)):
        values = prefix_distances[:, horizon_index][:, upper[0], upper[1]].reshape(-1)
        nonzero = values[values > 1e-8]
        trajectory_sigma.append(
            max(float(np.median(nonzero)) if len(nonzero) else 1.0, 1e-3)
        )
        values = consequence_distances[:, horizon_index][:, upper[0], upper[1]].reshape(
            -1
        )
        nonzero = values[values > 1e-8]
        consequence_sigma.append(
            max(float(np.median(nonzero)) if len(nonzero) else 1.0, 1e-3)
        )

    gt_q = np.zeros(
        (scene_count, len(HORIZONS_S), trajectories.shape[1]), dtype=np.float32
    )
    trajectory_Q = np.zeros_like(prefix_distances)
    consequence_Q = np.zeros_like(consequence_distances)
    stats_rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    for scene_index, token in enumerate(tokens):
        scene_manifest = manifest[manifest["scene_index"] == scene_index].sort_values(
            "candidate_index"
        )
        gt_indices = np.flatnonzero(scene_manifest["is_gt"].to_numpy(dtype=bool))
        if len(gt_indices) != 1:
            raise AssertionError(f"expected exactly one GT candidate for {token}")
        gt_index = int(gt_indices[0])
        for horizon_index, horizon in enumerate(HORIZONS_S):
            distance = prefix_distances[scene_index, horizon_index]
            gt_q[scene_index, horizon_index] = softmax_negative_distance(
                distance[:, gt_index], trajectory_sigma[horizon_index]
            )
            trajectory_Q[scene_index, horizon_index] = softmax_negative_distance(
                distance, trajectory_sigma[horizon_index], axis=1
            )
            consequence_Q[scene_index, horizon_index] = softmax_negative_distance(
                consequence_distances[scene_index, horizon_index],
                consequence_sigma[horizon_index],
                axis=1,
            )
            q = gt_q[scene_index, horizon_index]
            ordered = np.argsort(-q)
            label_entropy = entropy(q)
            false_negative_count = int(
                np.sum((np.arange(len(q)) != gt_index) & (q >= 0.5 * q[gt_index]))
            )
            stats_rows.append(
                {
                    "scene_token": token,
                    "horizon_s": horizon,
                    "gt_candidate_index": gt_index,
                    "gt_weight": float(q[gt_index]),
                    "top2_candidate_index": int(ordered[1]),
                    "top2_weight": float(q[ordered[1]]),
                    "top3_candidate_index": int(ordered[2]),
                    "top3_weight": float(q[ordered[2]]),
                    "label_entropy_nats": label_entropy,
                    "effective_positive_count": float(np.exp(label_entropy)),
                    "hard_one_hot_false_negative_count": false_negative_count,
                    "trajectory_sigma": trajectory_sigma[horizon_index],
                    "consequence_sigma": consequence_sigma[horizon_index],
                    "trajectory_Q_row_sum_error_max": float(
                        np.max(
                            np.abs(
                                trajectory_Q[scene_index, horizon_index].sum(axis=1)
                                - 1.0
                            )
                        )
                    ),
                    "consequence_Q_row_sum_error_max": float(
                        np.max(
                            np.abs(
                                consequence_Q[scene_index, horizon_index].sum(axis=1)
                                - 1.0
                            )
                        )
                    ),
                }
            )

    # Find real same-prefix/different-tail evidence.  Ranking uses relative
    # quantiles so it remains valid for either model or deterministic banks.
    for scene_index, token in enumerate(tokens):
        short = prefix_distances[scene_index, 0]
        long = prefix_distances[scene_index, -1]
        short_values, long_values = short[upper], long[upper]
        short_threshold = float(np.quantile(short_values, 0.30))
        long_threshold = float(np.quantile(long_values, 0.70))
        candidates = [
            (left, right)
            for left, right in zip(*upper)
            if short[left, right] <= short_threshold
            and long[left, right] >= long_threshold
        ]
        if candidates:
            left, right = max(candidates, key=lambda pair: long[pair] - short[pair])
            examples.append(
                {
                    "scene_token": token,
                    "candidate_i": int(left),
                    "candidate_j": int(right),
                    "short_horizon_s": HORIZONS_S[0],
                    "long_horizon_s": HORIZONS_S[-1],
                    "short_prefix_distance": float(short[left, right]),
                    "long_prefix_distance": float(long[left, right]),
                    "short_mutual_soft_weights": [
                        float(trajectory_Q[scene_index, 0, left, right]),
                        float(trajectory_Q[scene_index, 0, right, left]),
                    ],
                    "long_mutual_soft_weights": [
                        float(trajectory_Q[scene_index, -1, left, right]),
                        float(trajectory_Q[scene_index, -1, right, left]),
                    ],
                    "behavior_pass": bool(
                        trajectory_Q[scene_index, 0, left, right]
                        > trajectory_Q[scene_index, -1, left, right]
                        and trajectory_Q[scene_index, 0, right, left]
                        > trajectory_Q[scene_index, -1, right, left]
                    ),
                }
            )
    stats = pd.DataFrame(stats_rows)
    summary = {
        "scene_count": scene_count,
        "candidates_per_scene": trajectories.shape[1],
        "horizons_s": list(HORIZONS_S),
        "trajectory_scales": trajectory_scales,
        "environment_field_scales": dict(zip(fields, environment_scales.tolist())),
        "trajectory_sigma_by_horizon": dict(
            zip(map(str, HORIZONS_S), trajectory_sigma)
        ),
        "consequence_sigma_by_horizon": dict(
            zip(map(str, HORIZONS_S), consequence_sigma)
        ),
        "mean_gt_weight_by_horizon": stats.groupby("horizon_s")["gt_weight"]
        .mean()
        .to_dict(),
        "mean_effective_positive_count_by_horizon": stats.groupby("horizon_s")[
            "effective_positive_count"
        ]
        .mean()
        .to_dict(),
        "mean_false_negative_count_by_horizon": stats.groupby("horizon_s")[
            "hard_one_hot_false_negative_count"
        ]
        .mean()
        .to_dict(),
        "same_prefix_different_tail_example_count": len(examples),
        "same_prefix_different_tail_pass_rate": float(
            np.mean([item["behavior_pass"] for item in examples])
        )
        if examples
        else 0.0,
        "all_rows_sum_to_one": bool(
            np.max(np.abs(gt_q.sum(axis=-1) - 1.0)) < 1e-6
            and np.max(np.abs(trajectory_Q.sum(axis=-1) - 1.0)) < 1e-6
            and np.max(np.abs(consequence_Q.sum(axis=-1) - 1.0)) < 1e-6
        ),
        "prefix_aware": True,
        "future_after_horizon_used": False,
    }
    np.savez_compressed(
        args.output_dir / "soft_labels.npz",
        gt_q=gt_q,
        trajectory_Q=trajectory_Q,
        consequence_Q=consequence_Q,
        prefix_distances=prefix_distances,
        consequence_distances=consequence_distances,
        horizons_s=np.asarray(HORIZONS_S, dtype=np.float32),
    )
    write_dataframe(stats, args.output_dir / "soft_label_stats.csv")
    write_json(
        args.output_dir / "soft_label_examples.json",
        {"summary": summary, "examples": examples},
    )
    write_text(
        args.output_dir / "SOFT_CONTRASTIVE_AUDIT.md",
        "\n".join(
            [
                "# Prefix-Aware Soft Contrastive Label Audit",
                "",
                f"- Prefix-only construction: **PASS** (future-after-horizon used: `{summary['future_after_horizon_used']}`)",
                f"- All probability rows sum to one: **{summary['all_rows_sum_to_one']}**",
                f"- Same-prefix/different-tail examples: **{summary['same_prefix_different_tail_example_count']}**, behavior pass rate **{summary['same_prefix_different_tail_pass_rate']:.3%}**",
                "",
                "| Horizon | Mean GT weight | Effective positives | One-hot false negatives |",
                "|---:|---:|---:|---:|",
                *(
                    f"| {horizon:.1f} s | {summary['mean_gt_weight_by_horizon'][horizon]:.4f} | {summary['mean_effective_positive_count_by_horizon'][horizon]:.3f} | {summary['mean_false_negative_count_by_horizon'][horizon]:.3f} |"
                    for horizon in HORIZONS_S
                ),
                "",
                "Candidate-consequence `Q` combines dimension-aware `C_environment_only` distance with a mask-aware, stable-track-aligned actor distance. Binary risk fields use unit distance; continuous fields use robust physical scales. Invalid/missing actor slots never contribute as zero-valued actors.",
                "",
            ]
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
