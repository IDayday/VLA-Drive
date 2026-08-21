"""Action-collapse metrics with scene-clustered bootstrap intervals."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr

from research.action_effect.probe_data import HARD_TARGET_FIELDS, ProbeArrays, ProbeScale


@dataclass(frozen=True)
class MetricInterval:
    """Point estimate and percentile bootstrap confidence interval."""

    point: float
    ci_low: float
    ci_high: float
    bootstrap_samples: int
    confidence: float


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def decoded_prediction(raw_prediction: np.ndarray, hard_dim: int) -> np.ndarray:
    """Convert safety logits to probabilities while retaining normalized soft values."""

    result = np.asarray(raw_prediction, dtype=np.float64).copy()
    result[:, :hard_dim] = _sigmoid(result[:, :hard_dim])
    return result


def output_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """RMS output distance, stable across consequence-vector dimensionality."""

    return np.sqrt(np.mean(np.square(left - right), axis=-1))


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.ptp(left) <= 1.0e-12 or np.ptp(right) <= 1.0e-12:
        return 0.0
    value = spearmanr(left, right).statistic
    return 0.0 if not math.isfinite(float(value)) else float(value)


def _candidate_error(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(prediction - target), axis=-1)


def _inverse_soft(
    normalized: np.ndarray, scales: Sequence[ProbeScale], field: str
) -> np.ndarray:
    index = [scale.field for scale in scales].index(field)
    scale = scales[index]
    return normalized[:, index] * scale.scale + scale.median


def _load_pairs(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def compute_action_collapse_metrics(
    *,
    arrays: ProbeArrays,
    raw_prediction: np.ndarray,
    heldout_scene_ids: Sequence[str],
    pair_path: Path,
    scales: Sequence[ProbeScale],
    low_ttc_seconds: float,
    safe_threshold: float,
    bootstrap_samples: int,
    confidence: float,
    seed: int,
) -> tuple[dict[str, MetricInterval], dict[str, np.ndarray]]:
    """Compute required metrics and resample whole scenes for uncertainty.

    Probe-output distance is used for sensitivity/alignment because the quick
    consequence probe has no separately supervised future latent. This avoids
    treating arbitrary trajectory-encoder variation as evidence against
    action-conditioning collapse.
    """

    hard_dim = len(HARD_TARGET_FIELDS)
    if raw_prediction.shape != arrays.targets.shape:
        raise ValueError(
            f"prediction/target mismatch: {raw_prediction.shape} vs {arrays.targets.shape}"
        )
    prediction = decoded_prediction(raw_prediction, hard_dim)
    heldout = set(heldout_scene_ids)
    accepted_mask = arrays.accepted & np.isin(arrays.scene_ids, list(heldout))
    anchor_mask = accepted_mask & arrays.anchor
    if not np.any(anchor_mask):
        raise ValueError("held-out split has no accepted factual anchors")
    candidate_lookup = {candidate: index for index, candidate in enumerate(arrays.candidate_ids)}
    pairs = [row for row in _load_pairs(pair_path) if row["scene_id"] in heldout]
    pair_scene = np.asarray([row["scene_id"] for row in pairs], dtype=str)
    pair_left = np.asarray([candidate_lookup[row["candidate_i"]] for row in pairs], dtype=np.int64)
    pair_right = np.asarray([candidate_lookup[row["candidate_j"]] for row in pairs], dtype=np.int64)
    pair_type = np.asarray([row["pair_type"] for row in pairs], dtype=str)
    true_distance = np.asarray([row["consequence_distance"] for row in pairs], dtype=np.float64)
    predicted_distance = output_distance(prediction[pair_left], prediction[pair_right])

    candidate_error = _candidate_error(prediction, arrays.targets)
    shuffled_error = np.full(len(arrays.scene_ids), np.nan, dtype=np.float64)
    for scene_id in heldout:
        indices = np.flatnonzero(accepted_mask & (arrays.scene_ids == scene_id))
        indices = indices[np.argsort(arrays.candidate_ids[indices])]
        if len(indices) > 1:
            shifted = np.roll(indices, 1)
            shuffled_error[indices] = _candidate_error(prediction[shifted], arrays.targets[indices])

    hard = arrays.raw_hard_targets
    raw_ttc = arrays.raw_soft_targets[:, 0]
    true_unsafe = (
        (hard[:, 0] < 0.5)
        | (hard[:, 1] < 0.5)
        | (hard[:, 2] < 0.5)
        | (hard[:, 3] < 0.5)
        | (hard[:, 4] > 0.5)
        | (hard[:, 5] > 0.5)
        | (raw_ttc < low_ttc_seconds)
    ) & accepted_mask
    predicted_ttc = _inverse_soft(prediction[:, hard_dim:], scales, "ttc_infraction_time_s")
    predicted_safe = (
        np.all(prediction[:, :4] >= safe_threshold, axis=1)
        & np.all(prediction[:, 4:6] < safe_threshold, axis=1)
        & (predicted_ttc >= low_ttc_seconds)
    )
    false_safe = true_unsafe & predicted_safe

    scene_to_candidate = {
        scene_id: np.flatnonzero(accepted_mask & (arrays.scene_ids == scene_id))
        for scene_id in heldout_scene_ids
    }
    scene_to_anchor = {
        scene_id: np.flatnonzero(anchor_mask & (arrays.scene_ids == scene_id))
        for scene_id in heldout_scene_ids
    }
    scene_to_pair = {
        scene_id: np.flatnonzero(pair_scene == scene_id) for scene_id in heldout_scene_ids
    }

    def evaluate(selected: Sequence[str]) -> dict[str, float]:
        candidate_indices = np.concatenate([scene_to_candidate[scene] for scene in selected])
        anchor_indices = np.concatenate([scene_to_anchor[scene] for scene in selected])
        pair_indices = np.concatenate([scene_to_pair[scene] for scene in selected])
        finite_shuffle = candidate_indices[np.isfinite(shuffled_error[candidate_indices])]
        divergent = pair_indices[pair_type[pair_indices] == "effect_divergent"]
        equivalent = pair_indices[pair_type[pair_indices] == "effect_equivalent"]
        unsafe_count = int(np.sum(true_unsafe[candidate_indices]))
        return {
            "factual_prediction_error": float(np.mean(candidate_error[anchor_indices])),
            "action_shuffle_gap": float(
                np.mean(shuffled_error[finite_shuffle] - candidate_error[finite_shuffle])
            ),
            "candidate_sensitivity": float(np.mean(predicted_distance[pair_indices])),
            "effect_alignment": _safe_spearman(
                predicted_distance[pair_indices], true_distance[pair_indices]
            ),
            "action_gap": float(np.mean(predicted_distance[divergent])) if len(divergent) else float("nan"),
            "equivalence_leakage": float(np.mean(predicted_distance[equivalent])) if len(equivalent) else float("nan"),
            "risk_false_safe_rate": (
                float(np.sum(false_safe[candidate_indices]) / unsafe_count)
                if unsafe_count
                else float("nan")
            ),
        }

    point = evaluate(list(heldout_scene_ids))
    rng = np.random.default_rng(seed)
    bootstrap: dict[str, list[float]] = {name: [] for name in point}
    scene_array = np.asarray(heldout_scene_ids, dtype=str)
    for _ in range(bootstrap_samples):
        selected = rng.choice(scene_array, size=len(scene_array), replace=True).tolist()
        sampled = evaluate(selected)
        for name, value in sampled.items():
            if math.isfinite(value):
                bootstrap[name].append(value)
    alpha = (1.0 - confidence) / 2.0
    intervals: dict[str, MetricInterval] = {}
    for name, value in point.items():
        samples = np.asarray(bootstrap[name], dtype=np.float64)
        if not len(samples) or not math.isfinite(value):
            low = high = float("nan")
        else:
            low, high = np.quantile(samples, [alpha, 1.0 - alpha])
        intervals[name] = MetricInterval(
            point=float(value),
            ci_low=float(low),
            ci_high=float(high),
            bootstrap_samples=bootstrap_samples,
            confidence=confidence,
        )
    details = {
        "prediction": prediction,
        "candidate_error": candidate_error,
        "shuffled_error": shuffled_error,
        "true_unsafe": true_unsafe,
        "predicted_safe": predicted_safe,
        "false_safe": false_safe,
        "pair_predicted_distance": predicted_distance,
        "pair_true_distance": true_distance,
        "pair_scene": pair_scene,
        "pair_type": pair_type,
        "bootstrap_values": {
            name: np.asarray(values, dtype=np.float64) for name, values in bootstrap.items()
        },
    }
    return intervals, details


def decoded_structured_prediction(raw_prediction: np.ndarray) -> np.ndarray:
    """Decode the four binary structured channels and retain three regressions."""

    result = np.asarray(raw_prediction, dtype=np.float32).copy()
    result[:, :, :4] = _sigmoid(result[:, :, :4])
    return result


def compute_structured_collapse_metrics(
    *,
    arrays: ProbeArrays,
    structured_target: np.ndarray,
    structured_valid: np.ndarray,
    raw_prediction: np.ndarray,
    heldout_scene_ids: Sequence[str],
    pair_path: Path,
    safe_threshold: float,
    minimum_clearance_normalized: float,
    ego_grid_mask: np.ndarray,
    bootstrap_samples: int,
    confidence: float,
    seed: int,
) -> tuple[dict[str, MetricInterval], dict[str, Any]]:
    """Structured-future analogue of the consequence-output collapse metrics."""

    if raw_prediction.shape != structured_target.shape:
        raise ValueError("structured prediction/target shapes differ")
    prediction = decoded_structured_prediction(raw_prediction)
    target = np.asarray(structured_target, dtype=np.float32)
    heldout = set(heldout_scene_ids)
    accepted = arrays.accepted & structured_valid & np.isin(arrays.scene_ids, list(heldout))
    anchor = accepted & arrays.anchor
    candidate_error = np.full(len(arrays.scene_ids), np.nan, dtype=np.float64)
    batch = 128
    valid_indices = np.flatnonzero(accepted)
    for start in range(0, len(valid_indices), batch):
        indices = valid_indices[start : start + batch]
        candidate_error[indices] = np.mean(
            np.abs(prediction[indices] - target[indices]), axis=(1, 2, 3, 4)
        )
    shuffled_error = np.full(len(arrays.scene_ids), np.nan, dtype=np.float64)
    for scene_id in heldout_scene_ids:
        indices = np.flatnonzero(accepted & (arrays.scene_ids == scene_id))
        indices = indices[np.argsort(arrays.candidate_ids[indices])]
        if len(indices) > 1:
            shifted = np.roll(indices, 1)
            shuffled_error[indices] = np.mean(
                np.abs(prediction[shifted] - target[indices]), axis=(1, 2, 3, 4)
            )

    candidate_lookup = {candidate: index for index, candidate in enumerate(arrays.candidate_ids)}
    pairs = [row for row in _load_pairs(pair_path) if row["scene_id"] in heldout]
    pair_scene = np.asarray([row["scene_id"] for row in pairs], dtype=str)
    pair_left = np.asarray([candidate_lookup[row["candidate_i"]] for row in pairs], dtype=np.int64)
    pair_right = np.asarray([candidate_lookup[row["candidate_j"]] for row in pairs], dtype=np.int64)
    pair_type = np.asarray([row["pair_type"] for row in pairs], dtype=str)
    true_distance = np.asarray([row["consequence_distance"] for row in pairs], dtype=np.float64)
    predicted_distance = np.empty(len(pairs), dtype=np.float64)
    for start in range(0, len(pairs), 256):
        stop = min(start + 256, len(pairs))
        predicted_distance[start:stop] = np.sqrt(
            np.mean(
                np.square(prediction[pair_left[start:stop]] - prediction[pair_right[start:stop]]),
                axis=(1, 2, 3, 4),
            )
        )

    hard = arrays.raw_hard_targets
    true_unsafe = (
        (hard[:, 0] < 0.5)
        | (hard[:, 1] < 0.5)
        | (hard[:, 2] < 0.5)
        | (hard[:, 3] < 0.5)
        | (hard[:, 4] > 0.5)
        | (hard[:, 5] > 0.5)
    ) & accepted
    footprint = np.asarray(ego_grid_mask, dtype=bool)
    dynamic_at_ego = prediction[:, :, 3, footprint].max(axis=(1, 2))
    drivable_at_ego = prediction[:, :, 0, footprint].mean(axis=(1, 2))
    clearance_at_ego = prediction[:, :, 6, footprint].min(axis=(1, 2))
    predicted_safe = (
        (dynamic_at_ego < safe_threshold)
        & (drivable_at_ego >= safe_threshold)
        & (clearance_at_ego >= minimum_clearance_normalized)
    )
    false_safe = true_unsafe & predicted_safe

    scene_to_candidate = {
        scene: np.flatnonzero(accepted & (arrays.scene_ids == scene)) for scene in heldout_scene_ids
    }
    scene_to_anchor = {
        scene: np.flatnonzero(anchor & (arrays.scene_ids == scene)) for scene in heldout_scene_ids
    }
    scene_to_pair = {scene: np.flatnonzero(pair_scene == scene) for scene in heldout_scene_ids}

    def evaluate(selected: Sequence[str]) -> dict[str, float]:
        candidate_indices = np.concatenate([scene_to_candidate[scene] for scene in selected])
        anchor_indices = np.concatenate([scene_to_anchor[scene] for scene in selected])
        pair_indices = np.concatenate([scene_to_pair[scene] for scene in selected])
        finite_shuffle = candidate_indices[np.isfinite(shuffled_error[candidate_indices])]
        divergent = pair_indices[pair_type[pair_indices] == "effect_divergent"]
        equivalent = pair_indices[pair_type[pair_indices] == "effect_equivalent"]
        unsafe_count = int(np.sum(true_unsafe[candidate_indices]))
        return {
            "factual_prediction_error": float(np.mean(candidate_error[anchor_indices])),
            "action_shuffle_gap": float(
                np.mean(shuffled_error[finite_shuffle] - candidate_error[finite_shuffle])
            ),
            "candidate_sensitivity": float(np.mean(predicted_distance[pair_indices])),
            "effect_alignment": _safe_spearman(
                predicted_distance[pair_indices], true_distance[pair_indices]
            ),
            "action_gap": float(np.mean(predicted_distance[divergent])) if len(divergent) else float("nan"),
            "equivalence_leakage": float(np.mean(predicted_distance[equivalent])) if len(equivalent) else float("nan"),
            "risk_false_safe_rate": (
                float(np.sum(false_safe[candidate_indices]) / unsafe_count)
                if unsafe_count
                else float("nan")
            ),
        }

    point = evaluate(list(heldout_scene_ids))
    rng = np.random.default_rng(seed)
    bootstrapped = {name: [] for name in point}
    scenes = np.asarray(heldout_scene_ids, dtype=str)
    for _ in range(bootstrap_samples):
        result = evaluate(rng.choice(scenes, size=len(scenes), replace=True).tolist())
        for name, value in result.items():
            if math.isfinite(value):
                bootstrapped[name].append(value)
    alpha = (1.0 - confidence) / 2.0
    intervals = {}
    for name, value in point.items():
        values = np.asarray(bootstrapped[name], dtype=np.float64)
        low, high = (
            np.quantile(values, [alpha, 1.0 - alpha]) if len(values) else (float("nan"), float("nan"))
        )
        intervals[name] = MetricInterval(
            float(value), float(low), float(high), bootstrap_samples, confidence
        )
    return intervals, {
        "candidate_error": candidate_error,
        "shuffled_error": shuffled_error,
        "pair_predicted_distance": predicted_distance,
        "pair_true_distance": true_distance,
        "pair_type": pair_type,
        "pair_scene": pair_scene,
        "true_unsafe": true_unsafe,
        "predicted_safe": predicted_safe,
        "false_safe": false_safe,
        "bootstrap_values": {
            name: np.asarray(values, dtype=np.float64) for name, values in bootstrapped.items()
        },
    }
