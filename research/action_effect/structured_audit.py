"""Per-channel action-dependence diagnostics for structured world targets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from research.action_effect.probe_data import ProbeArrays


def _load_pairs(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def decode_binary_channels(prediction: np.ndarray, binary_channels: Sequence[int]) -> np.ndarray:
    """Decode declared logits without assuming that all fields are binary."""

    result = np.asarray(prediction, dtype=np.float32).copy()
    for channel in binary_channels:
        value = np.clip(result[:, :, channel], -30.0, 30.0)
        result[:, :, channel] = 1.0 / (1.0 + np.exp(-value))
    return result


def structured_channel_audit(
    *,
    arrays: ProbeArrays,
    target: np.ndarray,
    valid: np.ndarray,
    channels: Sequence[str],
    pair_path: Path,
    selected_scene_ids: Sequence[str],
    minimum_action_variance_ratio: float,
    minimum_target_action_gap: float,
    raw_prediction: np.ndarray | None = None,
    binary_channels: Sequence[int] = (),
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Measure within/between variance and candidate sensitivity per channel."""

    target = np.asarray(target, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if target.ndim != 5 or target.shape[2] != len(channels) or len(target) != len(arrays.scene_ids):
        raise ValueError("structured target must align with arrays and channel names")
    selected = set(str(value) for value in selected_scene_ids)
    accepted = arrays.accepted & valid & np.isin(arrays.scene_ids, list(selected))
    prediction = (
        decode_binary_channels(raw_prediction, binary_channels)
        if raw_prediction is not None
        else None
    )
    if prediction is not None and prediction.shape != target.shape:
        raise ValueError("structured prediction does not match target")

    within_values: list[np.ndarray] = []
    scene_means: list[np.ndarray] = []
    target_shuffle_by_channel: list[np.ndarray] = []
    prediction_shuffle_gap_by_channel: list[np.ndarray] = []
    for scene_id in sorted(selected):
        indices = np.flatnonzero(accepted & (arrays.scene_ids == scene_id))
        if not len(indices):
            continue
        values = target[indices]
        within_values.append(np.var(values, axis=0).mean(axis=(0, 2, 3)))
        scene_means.append(np.mean(values, axis=0))
        if len(indices) > 1:
            shifted = np.roll(indices, 1)
            target_shuffle_by_channel.append(
                np.mean(np.abs(target[shifted] - target[indices]), axis=(0, 1, 3, 4))
            )
            if prediction is not None:
                correct_error = np.mean(
                    np.abs(prediction[indices] - target[indices]), axis=(0, 1, 3, 4)
                )
                shuffled_error = np.mean(
                    np.abs(prediction[shifted] - target[indices]), axis=(0, 1, 3, 4)
                )
                prediction_shuffle_gap_by_channel.append(shuffled_error - correct_error)
    if not within_values or len(scene_means) < 2:
        raise ValueError("structured audit requires at least two populated scenes")
    within = np.mean(np.stack(within_values), axis=0)
    between = np.var(np.stack(scene_means), axis=0).mean(axis=(0, 2, 3))
    ratio = within / np.maximum(within + between, 1.0e-12)
    target_shuffle = np.mean(np.stack(target_shuffle_by_channel), axis=0)
    prediction_shuffle = (
        np.mean(np.stack(prediction_shuffle_gap_by_channel), axis=0)
        if prediction_shuffle_gap_by_channel
        else np.full(len(channels), np.nan, dtype=np.float64)
    )

    candidate_lookup = {candidate: index for index, candidate in enumerate(arrays.candidate_ids)}
    pair_rows = [row for row in _load_pairs(pair_path) if str(row["scene_id"]) in selected]
    left: list[int] = []
    right: list[int] = []
    pair_type: list[str] = []
    for row in pair_rows:
        left_index = candidate_lookup[str(row["candidate_i"])]
        right_index = candidate_lookup[str(row["candidate_j"])]
        if accepted[left_index] and accepted[right_index]:
            left.append(left_index)
            right.append(right_index)
            pair_type.append(str(row["pair_type"]))
    left_array = np.asarray(left, dtype=np.int64)
    right_array = np.asarray(right, dtype=np.int64)
    pair_type_array = np.asarray(pair_type, dtype=str)
    target_distance = np.empty((len(left_array), len(channels)), dtype=np.float32)
    predicted_distance = (
        np.empty_like(target_distance) if prediction is not None else None
    )
    for start in range(0, len(left_array), 256):
        stop = min(start + 256, len(left_array))
        target_distance[start:stop] = np.sqrt(
            np.mean(
                np.square(target[left_array[start:stop]] - target[right_array[start:stop]]),
                axis=(1, 3, 4),
            )
        )
        if prediction is not None and predicted_distance is not None:
            predicted_distance[start:stop] = np.sqrt(
                np.mean(
                    np.square(
                        prediction[left_array[start:stop]] - prediction[right_array[start:stop]]
                    ),
                    axis=(1, 3, 4),
                )
            )
    divergent = pair_type_array == "effect_divergent"
    target_gap = (
        np.mean(target_distance[divergent], axis=0)
        if np.any(divergent)
        else np.full(len(channels), np.nan, dtype=np.float64)
    )
    predicted_gap = (
        np.mean(predicted_distance[divergent], axis=0)
        if predicted_distance is not None and np.any(divergent)
        else np.full(len(channels), np.nan, dtype=np.float64)
    )
    sensitivity_ratio = predicted_gap / np.maximum(target_gap, 1.0e-12)
    rows = []
    for index, channel in enumerate(channels):
        action_dependent = bool(
            ratio[index] >= minimum_action_variance_ratio
            and target_gap[index] >= minimum_target_action_gap
        )
        rows.append(
            {
                "channel": str(channel),
                "within_scene_candidate_variance": float(within[index]),
                "between_scene_variance": float(between[index]),
                "action_variance_ratio": float(ratio[index]),
                "target_action_gap": float(target_gap[index]),
                "predicted_action_gap": float(predicted_gap[index]),
                "predicted_target_sensitivity_ratio": float(sensitivity_ratio[index]),
                "target_candidate_shuffle_effect": float(target_shuffle[index]),
                "prediction_candidate_shuffle_gap": float(prediction_shuffle[index]),
                "classification": (
                    "action_effect/action-dependent"
                    if action_dependent
                    else "exogenous/action-invariant"
                ),
            }
        )
    details = {
        "pair_target_distance": target_distance,
        "pair_predicted_distance": (
            predicted_distance
            if predicted_distance is not None
            else np.empty((0, len(channels)), dtype=np.float32)
        ),
        "pair_type": pair_type_array,
        "action_dependent_channel_indices": np.asarray(
            [index for index, row in enumerate(rows) if row["classification"].startswith("action_effect")],
            dtype=np.int64,
        ),
    }
    return rows, details
