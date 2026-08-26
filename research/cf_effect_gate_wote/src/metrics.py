"""Planning, ranking, and scene-level uncertainty metrics for the Gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import numpy.typing as npt
from scipy.stats import kendalltau


FACTOR_NAMES = ("NC", "DAC", "EP", "TTC", "Comfort")


def pdms_from_factors(
    factors: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Compute NAVSIM v1-style PDMS from `[NC,DAC,EP,TTC,Comfort]`."""

    values = np.asarray(factors, dtype=np.float64)
    if values.shape[-1] != 5:
        raise ValueError(f"factor tensor must end in five values, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("factor tensor contains NaN/Inf")
    nc, dac, ep, ttc, comfort = np.moveaxis(values, -1, 0)
    return nc * dac * ((5.0 * ep + 5.0 * ttc + 2.0 * comfort) / 12.0)


def candidate_ranks(scores: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """Return one-based descending ranks, assigning the best rank to ties."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"scores must be [scene,candidate], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("scores contain NaN/Inf")
    return 1 + (values[:, :, None] < values[:, None, :]).sum(axis=-1)


def pairwise_ranking_accuracy(
    predicted: npt.ArrayLike, target: npt.ArrayLike
) -> float:
    predicted_values = np.asarray(predicted, dtype=np.float64)
    target_values = np.asarray(target, dtype=np.float64)
    if predicted_values.shape != target_values.shape or predicted_values.ndim != 2:
        raise ValueError("predicted and target rankings must be matching [scene,candidate]")
    correct = 0
    valid = 0
    for scene_predicted, scene_target in zip(predicted_values, target_values):
        target_delta = scene_target[:, None] - scene_target[None, :]
        predicted_delta = scene_predicted[:, None] - scene_predicted[None, :]
        upper = np.triu(np.ones_like(target_delta, dtype=bool), k=1)
        comparable = upper & (target_delta != 0)
        correct += int((np.sign(target_delta[comparable]) == np.sign(predicted_delta[comparable])).sum())
        valid += int(comparable.sum())
    if valid == 0:
        raise ValueError("no non-tied candidate pairs are available")
    return correct / valid


def mean_kendall_tau(predicted: npt.ArrayLike, target: npt.ArrayLike) -> float:
    predicted_values = np.asarray(predicted, dtype=np.float64)
    target_values = np.asarray(target, dtype=np.float64)
    if predicted_values.shape != target_values.shape or predicted_values.ndim != 2:
        raise ValueError("predicted and target rankings must be matching [scene,candidate]")
    values: list[float] = []
    for scene_predicted, scene_target in zip(predicted_values, target_values):
        statistic = kendalltau(scene_predicted, scene_target, nan_policy="raise").statistic
        if np.isfinite(statistic):
            values.append(float(statistic))
    if not values:
        raise ValueError("Kendall tau is undefined for every scene")
    return float(np.mean(values))


def false_safe_mask(
    predicted_scores: npt.ArrayLike,
    factor_labels: npt.ArrayLike,
    high_score_quantile: float = 0.75,
) -> npt.NDArray[np.bool_]:
    """Mark high-predicted selected candidates with true NC/DAC/TTC failure."""

    predicted = np.asarray(predicted_scores, dtype=np.float64)
    factors = np.asarray(factor_labels, dtype=np.float64)
    if predicted.ndim != 2 or factors.shape != predicted.shape + (5,):
        raise ValueError("false-safe inputs must be [S,K] scores and [S,K,5] factors")
    if not 0.0 < high_score_quantile < 1.0:
        raise ValueError("high_score_quantile must lie strictly between zero and one")
    selected = np.argmax(predicted, axis=1)
    selected_prediction = predicted[np.arange(len(predicted)), selected]
    high_threshold = np.quantile(predicted, high_score_quantile, axis=1)
    selected_factors = factors[np.arange(len(factors)), selected]
    unsafe = (
        (selected_factors[:, 0] == 0)
        | (selected_factors[:, 1] == 0)
        | (selected_factors[:, 3] == 0)
    )
    return (selected_prediction >= high_threshold) & unsafe


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    samples: int
    unit: str = "scene"


def paired_scene_bootstrap(
    first: npt.ArrayLike,
    second: npt.ArrayLike,
    statistic: Callable[[npt.NDArray[np.float64]], float] = np.mean,
    samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 20260827,
) -> BootstrapInterval:
    """Bootstrap the paired scene-level difference `first - second`."""

    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("paired bootstrap inputs must be matching one-dimensional scenes")
    if len(left) == 0 or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("paired bootstrap requires finite non-empty inputs")
    if samples <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid bootstrap samples or confidence")
    differences = left - right
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        scene_indices = rng.integers(0, len(differences), size=len(differences))
        estimates[index] = statistic(differences[scene_indices])
    alpha = (1.0 - confidence) / 2.0
    return BootstrapInterval(
        estimate=float(statistic(differences)),
        lower=float(np.quantile(estimates, alpha)),
        upper=float(np.quantile(estimates, 1.0 - alpha)),
        confidence=confidence,
        samples=samples,
    )


def selected_planning_metrics(
    predicted_scores: npt.ArrayLike,
    true_scores: npt.ArrayLike,
    factors: npt.ArrayLike,
) -> Mapping[str, float]:
    predicted = np.asarray(predicted_scores, dtype=np.float64)
    target = np.asarray(true_scores, dtype=np.float64)
    factor_values = np.asarray(factors, dtype=np.float64)
    if predicted.shape != target.shape or factor_values.shape != target.shape + (5,):
        raise ValueError("planning metric shape mismatch")
    selected = np.argmax(predicted, axis=1)
    rows = np.arange(len(selected))
    selected_true = target[rows, selected]
    oracle = target.max(axis=1)
    ranks = candidate_ranks(target)[rows, selected]
    false_safe = false_safe_mask(predicted, factor_values)
    return {
        "selected_pdms": float(selected_true.mean()),
        "top1_regret": float((oracle - selected_true).mean()),
        "mean_candidate_rank": float(ranks.mean()),
        "pairwise_accuracy": pairwise_ranking_accuracy(predicted, target),
        "kendall_tau": mean_kendall_tau(predicted, target),
        "false_safe_rate": float(false_safe.mean()),
        "failure_recovery_rate": float((selected_true > 0).mean()),
    }
