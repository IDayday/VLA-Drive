"""Ranking, calibration, safety and paired-bootstrap metrics for Direct V3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt
from sklearn.metrics import roc_auc_score


FACTOR_ORDER = ("NC", "DAC", "DDC", "EP", "TTC", "Comfort")
HARD_SAFETY_INDICES = (0, 1, 2, 4)


def _arrays(
    predicted: npt.ArrayLike, target: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    first = np.asarray(predicted, dtype=np.float64)
    second = np.asarray(target, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError(f"ranking arrays must share [N,K], got {first.shape}/{second.shape}")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("ranking arrays contain NaN/Inf")
    return first, second


def candidate_ranks_descending(scores: npt.ArrayLike) -> npt.NDArray[np.int64]:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("candidate ranks require [N,K]")
    order = np.argsort(-values, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(len(values))[:, None]
    ranks[rows, order] = np.arange(values.shape[1], dtype=np.int64)[None]
    return ranks + 1


def pairwise_accuracy(
    predicted: npt.ArrayLike,
    target: npt.ArrayLike,
    *,
    true_top_k: int | None = None,
) -> float:
    prediction, truth = _arrays(predicted, target)
    scene_values: list[float] = []
    for predicted_scene, true_scene in zip(prediction, truth):
        indices = (
            np.arange(len(true_scene), dtype=np.int64)
            if true_top_k is None
            else np.argsort(-true_scene, kind="stable")[: int(true_top_k)]
        )
        left, right = np.triu_indices(len(indices), k=1)
        true_delta = true_scene[indices[left]] - true_scene[indices[right]]
        predicted_delta = (
            predicted_scene[indices[left]] - predicted_scene[indices[right]]
        )
        non_tie = np.abs(true_delta) > 1.0e-12
        if non_tie.any():
            scene_values.append(
                float(
                    np.mean(
                        np.sign(true_delta[non_tie])
                        == np.sign(predicted_delta[non_tie])
                    )
                )
            )
    return float(np.mean(scene_values)) if scene_values else float("nan")


def ndcg_at_k(predicted: npt.ArrayLike, target: npt.ArrayLike, k: int) -> float:
    prediction, truth = _arrays(predicted, target)
    if not 0 < k <= prediction.shape[1]:
        raise ValueError("invalid NDCG cutoff")
    discounts = 1.0 / np.log2(np.arange(k, dtype=np.float64) + 2.0)
    values = []
    for predicted_scene, true_scene in zip(prediction, truth):
        predicted_order = np.argsort(-predicted_scene, kind="stable")[:k]
        ideal_order = np.argsort(-true_scene, kind="stable")[:k]
        dcg = float(np.sum((np.power(2.0, true_scene[predicted_order]) - 1.0) * discounts))
        ideal = float(np.sum((np.power(2.0, true_scene[ideal_order]) - 1.0) * discounts))
        values.append(dcg / ideal if ideal > 0 else 1.0)
    return float(np.mean(values))


def oracle_hit_rate(predicted: npt.ArrayLike, target: npt.ArrayLike, k: int) -> float:
    prediction, truth = _arrays(predicted, target)
    oracle = np.argmax(truth, axis=1)
    top = np.argsort(-prediction, axis=1, kind="stable")[:, :k]
    return float(np.mean(np.any(top == oracle[:, None], axis=1)))


def expected_calibration_error(
    probability: npt.ArrayLike,
    target: npt.ArrayLike,
    *,
    bins: int = 15,
) -> float:
    predicted = np.asarray(probability, dtype=np.float64).reshape(-1)
    truth = np.asarray(target, dtype=np.float64).reshape(-1)
    if predicted.shape != truth.shape or not len(predicted):
        raise ValueError("ECE requires equal non-empty arrays")
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        member = (predicted >= boundaries[index]) & (
            predicted < boundaries[index + 1]
            if index + 1 < bins
            else predicted <= boundaries[index + 1]
        )
        if member.any():
            result += float(member.mean()) * abs(
                float(predicted[member].mean()) - float(truth[member].mean())
            )
    return result


def factor_calibration_rows(
    predicted_factors: npt.ArrayLike,
    true_factors: npt.ArrayLike,
    selection_indices: npt.ArrayLike,
) -> list[dict[str, Any]]:
    predicted = np.asarray(predicted_factors, dtype=np.float64)
    truth = np.asarray(true_factors, dtype=np.float64)
    selected = np.asarray(selection_indices, dtype=np.int64)
    if predicted.shape != truth.shape or predicted.ndim != 3 or predicted.shape[-1] != 6:
        raise ValueError("factor calibration requires matching [N,K,6]")
    if selected.shape != (predicted.shape[0],):
        raise ValueError("selection indices must be [N]")
    rows = np.arange(len(selected))
    true_top32 = np.argsort(
        -(
            truth[..., 0]
            * truth[..., 1]
            * truth[..., 2]
            * (5 * truth[..., 3] + 5 * truth[..., 4] + 2 * truth[..., 5])
            / 12.0
        ),
        axis=1,
        kind="stable",
    )[:, :32]
    output: list[dict[str, Any]] = []
    for factor_index, factor_name in enumerate(FACTOR_ORDER):
        probability = predicted[..., factor_index]
        target = truth[..., factor_index]
        auc = float("nan")
        if factor_name != "EP":
            auc_target = (target > 0.0).astype(np.int64)
            if len(np.unique(auc_target)) == 2:
                auc = float(roc_auc_score(auc_target.reshape(-1), probability.reshape(-1)))
        output.append(
            {
                "factor": factor_name,
                "mae": float(np.mean(np.abs(probability - target))),
                "brier": float(np.mean(np.square(probability - target))),
                "ece": expected_calibration_error(probability, target),
                "auc": auc,
                "selected_candidate_calibration_error": float(
                    np.mean(
                        np.abs(
                            probability[rows, selected] - target[rows, selected]
                        )
                    )
                ),
                "top32_candidate_calibration_error": float(
                    np.mean(
                        np.abs(
                            np.take_along_axis(probability, true_top32, axis=1)
                            - np.take_along_axis(target, true_top32, axis=1)
                        )
                    )
                ),
            }
        )
    return output


def scene_level_metrics(
    tokens: Sequence[str],
    selection_values: npt.ArrayLike,
    predicted_factors: npt.ArrayLike,
    true_factors: npt.ArrayLike,
    true_scores: npt.ArrayLike,
    *,
    predicted_hard_safety: npt.ArrayLike | None = None,
) -> list[dict[str, Any]]:
    selection = np.asarray(selection_values, dtype=np.float64)
    predicted = np.asarray(predicted_factors, dtype=np.float64)
    factors = np.asarray(true_factors, dtype=np.float64)
    scores = np.asarray(true_scores, dtype=np.float64)
    if selection.shape != scores.shape or predicted.shape != factors.shape:
        raise ValueError("scene metric prediction/label shapes differ")
    if len(tokens) != len(scores):
        raise ValueError("scene token count differs from predictions")
    selected = np.argmax(selection, axis=1)
    oracle = np.argmax(scores, axis=1)
    ranks = candidate_ranks_descending(scores)
    if predicted_hard_safety is not None:
        safety = np.asarray(predicted_hard_safety, dtype=np.float64)
        if safety.shape != scores.shape:
            raise ValueError("predicted hard-safety shape differs from scores")
    true_safe = (factors[..., HARD_SAFETY_INDICES] > 0.0).all(axis=-1)
    output: list[dict[str, Any]] = []
    for scene_index, token in enumerate(tokens):
        candidate = int(selected[scene_index])
        output.append(
            {
                "scene_token": str(token),
                "selected_index": candidate,
                "oracle_index": int(oracle[scene_index]),
                "selected_score": float(scores[scene_index, candidate]),
                "oracle_score": float(scores[scene_index, oracle[scene_index]]),
                "regret": float(
                    scores[scene_index, oracle[scene_index]]
                    - scores[scene_index, candidate]
                ),
                "selected_rank": int(ranks[scene_index, candidate]),
                "predicted_score": float(selection[scene_index, candidate]),
                "score_overestimation": float(
                    selection[scene_index, candidate] - scores[scene_index, candidate]
                ),
                "hard_false_safe": bool(
                    not true_safe[scene_index, candidate]
                ),
                "direction_non_compliance": bool(
                    factors[scene_index, candidate, 2] < 1.0
                ),
                "zero_score_selection": bool(scores[scene_index, candidate] <= 0.0),
                "oracle_capture": bool(candidate == int(oracle[scene_index])),
                "nc_failure": bool(factors[scene_index, candidate, 0] <= 0.0),
                "dac_failure": bool(factors[scene_index, candidate, 1] <= 0.0),
                "ddc_failure": bool(factors[scene_index, candidate, 2] <= 0.0),
                "ttc_failure": bool(factors[scene_index, candidate, 4] <= 0.0),
            }
        )
    return output


def aggregate_ranking_metrics(
    selection_values: npt.ArrayLike,
    true_scores: npt.ArrayLike,
    scene_rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, float]:
    selection, scores = _arrays(selection_values, true_scores)
    return {
        "selected_score": float(np.mean([row["selected_score"] for row in scene_rows])),
        "top1_regret": float(np.mean([row["regret"] for row in scene_rows])),
        "mean_selected_candidate_rank": float(
            np.mean([row["selected_rank"] for row in scene_rows])
        ),
        "ndcg_at_5": ndcg_at_k(selection, scores, 5),
        "ndcg_at_10": ndcg_at_k(selection, scores, 10),
        "ndcg_at_20": ndcg_at_k(selection, scores, 20),
        "oracle_in_top5_rate": oracle_hit_rate(selection, scores, 5),
        "oracle_in_top10_rate": oracle_hit_rate(selection, scores, 10),
        "oracle_in_top20_rate": oracle_hit_rate(selection, scores, 20),
        "all_pair_pairwise_accuracy": pairwise_accuracy(selection, scores),
        "top128_pairwise_accuracy": pairwise_accuracy(
            selection, scores, true_top_k=128
        ),
        "top64_pairwise_accuracy": pairwise_accuracy(selection, scores, true_top_k=64),
        "top32_pairwise_accuracy": pairwise_accuracy(selection, scores, true_top_k=32),
        "top16_pairwise_accuracy": pairwise_accuracy(selection, scores, true_top_k=16),
        "hard_false_safe": float(np.mean([row["hard_false_safe"] for row in scene_rows])),
        "direction_non_compliance": float(
            np.mean([row["direction_non_compliance"] for row in scene_rows])
        ),
        "zero_score_selection": float(
            np.mean([row["zero_score_selection"] for row in scene_rows])
        ),
        "oracle_capture": float(np.mean([row["oracle_capture"] for row in scene_rows])),
        "selected_score_overestimation": float(
            np.mean([row["score_overestimation"] for row in scene_rows])
        ),
    }


@dataclass(frozen=True)
class PairedBootstrapResult:
    mean_delta: float
    ci_lower: float
    ci_upper: float
    resamples: int
    seed: int


def paired_scene_bootstrap(
    first: npt.ArrayLike,
    second: npt.ArrayLike,
    *,
    resamples: int = 5000,
    seed: int = 20260827,
) -> PairedBootstrapResult:
    left = np.asarray(first, dtype=np.float64).reshape(-1)
    right = np.asarray(second, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or not len(left):
        raise ValueError("paired bootstrap requires equal non-empty arrays")
    if resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    delta = left - right
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=np.float64)
    # Chunk index generation to bound memory at larger resample counts.
    for start in range(0, resamples, 512):
        stop = min(start + 512, resamples)
        indices = rng.integers(0, len(delta), size=(stop - start, len(delta)))
        samples[start:stop] = delta[indices].mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return PairedBootstrapResult(
        mean_delta=float(delta.mean()),
        ci_lower=float(lower),
        ci_upper=float(upper),
        resamples=int(resamples),
        seed=int(seed),
    )
