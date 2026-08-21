"""Engineering checks and calibrated risk metrics for the Gate 2.5 audit."""

from __future__ import annotations

from dataclasses import asdict
import math
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)

from research.action_effect.metrics import MetricInterval, decoded_prediction
from research.action_effect.probe_data import HARD_TARGET_FIELDS, ProbeArrays, ProbeScale


def trajectory_summary_target(
    trajectories: np.ndarray,
    *,
    interval_s: float,
) -> np.ndarray:
    """Summarize physical trajectories for the synthetic action-fitting test.

    The five outputs are terminal ``x/y``, mean speed, signed maximum lateral
    displacement, and maximum absolute curvature. No scene or future actor
    state is involved.
    """

    value = np.asarray(trajectories, dtype=np.float64)
    if value.ndim != 3 or value.shape[-1] < 3 or interval_s <= 0:
        raise ValueError("trajectories must be [N,T,>=3] and interval_s must be positive")
    origin = np.zeros((len(value), 1, 2), dtype=np.float64)
    points = np.concatenate((origin, value[..., :2]), axis=1)
    displacement = np.diff(points, axis=1)
    distance = np.linalg.norm(displacement, axis=-1)
    mean_speed = np.mean(distance / interval_s, axis=1)
    heading = np.unwrap(value[..., 2], axis=1)
    heading_with_origin = np.concatenate((np.zeros((len(value), 1)), heading), axis=1)
    yaw_step = np.diff(heading_with_origin, axis=1)
    curvature = np.divide(
        np.abs(yaw_step),
        distance,
        out=np.zeros_like(distance),
        where=distance >= 0.05,
    )
    lateral = value[..., 1]
    lateral_index = np.argmax(np.abs(lateral), axis=1)
    signed_lateral = lateral[np.arange(len(value)), lateral_index]
    return np.stack(
        (
            value[:, -1, 0],
            value[:, -1, 1],
            mean_speed,
            signed_lateral,
            np.max(curvature, axis=1),
        ),
        axis=1,
    ).astype(np.float32)


def standardize_target(
    target: np.ndarray,
    fit_indices: np.ndarray,
    *,
    minimum_scale: float = 1.0e-3,
) -> tuple[np.ndarray, dict[str, list[float]]]:
    """Standardize a synthetic target using fit rows only."""

    target = np.asarray(target, dtype=np.float32)
    fit = target[np.asarray(fit_indices, dtype=np.int64)]
    mean = fit.mean(axis=0)
    scale = np.maximum(fit.std(axis=0), minimum_scale)
    return ((target - mean) / scale).astype(np.float32), {
        "mean": mean.astype(float).tolist(),
        "scale": scale.astype(float).tolist(),
    }


def unsafe_labels(
    arrays: ProbeArrays,
    *,
    low_ttc_seconds: float,
) -> np.ndarray:
    """Return the declared hard/low-TTC unsafe label for accepted candidates."""

    hard = arrays.raw_hard_targets
    raw_ttc = arrays.raw_soft_targets[:, 0]
    return (
        (hard[:, 0] < 0.5)
        | (hard[:, 1] < 0.5)
        | (hard[:, 2] < 0.5)
        | (hard[:, 3] < 0.5)
        | (hard[:, 4] > 0.5)
        | (hard[:, 5] > 0.5)
        | (raw_ttc < low_ttc_seconds)
    ) & arrays.accepted


def consequence_risk_score(
    raw_prediction: np.ndarray,
    scales: Sequence[ProbeScale],
    *,
    low_ttc_seconds: float,
) -> np.ndarray:
    """Collapse calibrated consequence outputs into a monotone unsafe score."""

    hard_dim = len(HARD_TARGET_FIELDS)
    prediction = decoded_prediction(raw_prediction, hard_dim)
    hard_risk = np.concatenate(
        (1.0 - prediction[:, :4], prediction[:, 4:6]),
        axis=1,
    )
    ttc_index = [scale.field for scale in scales].index("ttc_infraction_time_s")
    ttc_scale = scales[ttc_index]
    predicted_ttc = (
        prediction[:, hard_dim + ttc_index] * ttc_scale.scale + ttc_scale.median
    )
    # A smooth score avoids threshold ties while preserving the declared TTC
    # boundary. One robust scale corresponds to a logit change of one.
    ttc_risk = 1.0 / (
        1.0
        + np.exp(
            np.clip(
                (predicted_ttc - low_ttc_seconds) / max(ttc_scale.scale, 1.0e-3),
                -30.0,
                30.0,
            )
        )
    )
    return np.maximum(np.max(hard_risk, axis=1), ttc_risk)


def _classification_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    predicted_unsafe = scores >= threshold
    unsafe_count = int(labels.sum())
    safe_count = int((~labels).sum())
    if unsafe_count and safe_count:
        balanced = float(balanced_accuracy_score(labels, predicted_unsafe))
        auroc = float(roc_auc_score(labels, scores))
        auprc = float(average_precision_score(labels, scores))
    else:
        balanced = auroc = auprc = float("nan")
    false_safe = float(np.sum(labels & ~predicted_unsafe) / unsafe_count) if unsafe_count else float("nan")
    return {
        "unsafe_prevalence": float(np.mean(labels)) if len(labels) else float("nan"),
        "balanced_accuracy": balanced,
        "auroc": auroc,
        "auprc": auprc,
        "false_safe_rate": false_safe,
        "sample_count": float(len(labels)),
        "unsafe_count": float(unsafe_count),
    }


def calibrate_risk_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, dict[str, float]]:
    """Choose a fit-only threshold maximizing balanced accuracy.

    Ties prefer the lower false-safe rate, followed by the smaller threshold.
    """

    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    if len(labels) != len(scores) or not len(labels):
        raise ValueError("calibration labels/scores must be non-empty and aligned")
    if labels.all() or (~labels).all():
        raise ValueError("calibration requires both safe and unsafe candidates")
    unique = np.unique(scores)
    if len(unique) > 2048:
        unique = np.quantile(unique, np.linspace(0.0, 1.0, 2048))
    epsilon = np.finfo(np.float64).eps
    candidates = np.unique(np.concatenate(([unique[0] - epsilon], unique, [unique[-1] + epsilon])))
    ranked: list[tuple[float, float, float, dict[str, float]]] = []
    for threshold in candidates:
        metrics = _classification_metrics(labels, scores, float(threshold))
        ranked.append(
            (
                metrics["balanced_accuracy"],
                -metrics["false_safe_rate"],
                -float(threshold),
                metrics,
            )
        )
    _, _, negative_threshold, best = max(ranked, key=lambda row: row[:3])
    threshold = -negative_threshold
    return float(threshold), best


def calibrated_scene_bootstrap(
    *,
    labels: np.ndarray,
    scores: np.ndarray,
    scene_ids: np.ndarray,
    selected_scene_ids: Sequence[str],
    threshold: float,
    samples: int,
    confidence: float,
    seed: int,
) -> tuple[dict[str, MetricInterval], dict[str, float]]:
    """Evaluate a fixed fit-calibrated threshold with whole-scene bootstrap."""

    selected = list(selected_scene_ids)
    by_scene = {scene: np.flatnonzero(scene_ids == scene) for scene in selected}

    def evaluate(scenes: Sequence[str]) -> dict[str, float]:
        indices = np.concatenate([by_scene[scene] for scene in scenes])
        return _classification_metrics(labels[indices], scores[indices], threshold)

    point = evaluate(selected)
    tracked = ("unsafe_prevalence", "balanced_accuracy", "auroc", "auprc", "false_safe_rate")
    draws: dict[str, list[float]] = {name: [] for name in tracked}
    rng = np.random.default_rng(seed)
    scene_array = np.asarray(selected, dtype=str)
    for _ in range(samples):
        sampled = rng.choice(scene_array, size=len(scene_array), replace=True).tolist()
        metrics = evaluate(sampled)
        for name in tracked:
            if math.isfinite(metrics[name]):
                draws[name].append(metrics[name])
    alpha = (1.0 - confidence) / 2.0
    intervals: dict[str, MetricInterval] = {}
    for name in tracked:
        values = np.asarray(draws[name], dtype=np.float64)
        low, high = (
            np.quantile(values, [alpha, 1.0 - alpha])
            if len(values)
            else (float("nan"), float("nan"))
        )
        intervals[name] = MetricInterval(
            point=float(point[name]),
            ci_low=float(low),
            ci_high=float(high),
            bootstrap_samples=samples,
            confidence=confidence,
        )
    return intervals, {
        "threshold": float(threshold),
        "sample_count": int(point["sample_count"]),
        "unsafe_count": int(point["unsafe_count"]),
    }


def intervals_to_json(intervals: Mapping[str, MetricInterval]) -> dict[str, Any]:
    """Serialize metric intervals without leaking numpy scalar types."""

    return {name: asdict(value) for name, value in intervals.items()}
