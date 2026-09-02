"""Candidate quality, geometry, rank, and failure classification metrics."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
from scipy.stats import kendalltau, rankdata

from .schema import EPSILON_ORACLE_VALUES, QUALITY_THRESHOLDS, TOP_K_VALUES


def aggregate_predicted_score(factor_logits: np.ndarray, weights: Mapping[str, float]) -> np.ndarray:
    """Reproduce DrivoR's log-space ranking score; this is not true PDMS."""

    logits = np.asarray(factor_logits, dtype=np.float64)
    if logits.shape[-1] != 6:
        raise ValueError(f"Expected six factor logits, got {logits.shape}")
    prob = 1.0 / (1.0 + np.exp(-logits))
    return (
        weights["noc"] * np.log(prob[..., 0])
        + weights["dac"] * np.log(prob[..., 1])
        + weights["ddc"] * np.log(prob[..., 2])
        + np.log(
            weights["ttc"] * prob[..., 3]
            + weights["ep"] * prob[..., 4]
            + weights["comfort"] * prob[..., 5]
        )
    )


def average_rank_descending(values: np.ndarray) -> np.ndarray:
    return rankdata(-np.asarray(values), method="average")


def rank_metrics(true_scores: np.ndarray, predicted_scores: np.ndarray) -> Dict[str, float]:
    true_scores = np.asarray(true_scores, dtype=np.float64)
    predicted_scores = np.asarray(predicted_scores, dtype=np.float64)
    if true_scores.shape != predicted_scores.shape:
        raise ValueError("True/predicted candidate shapes differ")
    true_rank = average_rank_descending(true_scores)
    pred_rank = average_rank_descending(predicted_scores)
    if np.all(true_rank == true_rank[0]) or np.all(pred_rank == pred_rank[0]):
        spearman = 0.0
    else:
        spearman = float(np.corrcoef(true_rank, pred_rank)[0, 1])
    tau = kendalltau(true_scores, predicted_scores, variant="b", nan_policy="omit").statistic
    tau = 0.0 if not np.isfinite(tau) else float(tau)
    high = true_scores >= 0.95
    low = true_scores <= 0.50
    if high.any() and low.any():
        delta = predicted_scores[high, None] - predicted_scores[None, low]
        pairwise = float(np.mean((delta > 0).astype(np.float64) + 0.5 * (delta == 0)))
    else:
        pairwise = float("nan")
    return {"spearman": spearman, "kendall_tau_b": tau, "high_low_pairwise_accuracy": pairwise}


def _cluster_count(
    proposals: np.ndarray,
    *,
    ade_threshold: float = 0.50,
    fde_threshold: float = 1.00,
) -> int:
    n = len(proposals)
    if n == 0:
        return 0
    xy = np.asarray(proposals, dtype=np.float64)[..., :2]
    delta = xy[:, None] - xy[None]
    ade = np.linalg.norm(delta, axis=-1).mean(axis=-1)
    fde = np.linalg.norm(delta[:, :, -1], axis=-1)
    adjacency = (ade <= ade_threshold) & (fde <= fde_threshold)
    parent = np.arange(n)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    left, right = np.where(np.triu(adjacency, k=1))
    for i, j in zip(left.tolist(), right.tolist()):
        union(i, j)
    return len({find(index) for index in range(n)})


def geometry_metrics(proposals: np.ndarray, high_quality: np.ndarray) -> Dict[str, float]:
    proposals = np.asarray(proposals, dtype=np.float64)
    if proposals.ndim != 3 or proposals.shape[-1] != 3:
        raise ValueError(f"Expected [K,T,3] proposals, got {proposals.shape}")
    left, right = np.triu_indices(len(proposals), k=1)
    delta = proposals[left, :, :2] - proposals[right, :, :2]
    distances = np.linalg.norm(delta, axis=-1)
    endpoints = proposals[:, -1, :2]
    center = endpoints.mean(axis=0)
    return {
        "pairwise_ade": float(distances.mean()) if len(left) else 0.0,
        "pairwise_fde": float(distances[:, -1].mean()) if len(left) else 0.0,
        "endpoint_spread": float(np.sqrt(np.mean(np.sum((endpoints - center) ** 2, axis=-1)))),
        "trajectory_cluster_count": float(_cluster_count(proposals)),
        "high_quality_cluster_count": float(_cluster_count(proposals[np.asarray(high_quality, dtype=bool)])),
    }


def exclusive_failure_class(v_b: float, o_b: float, g_b: float, n095: int) -> str:
    """Return one class while retaining separate boolean flags in scene metrics."""

    if v_b >= 0.95 and g_b <= 0.01:
        return "SATURATED"
    if o_b >= 0.95 and n095 >= 8 and v_b < 0.90:
        return "DENSE_RANK_FAILURE"
    if o_b >= 0.95 and v_b < 0.90 and g_b >= 0.05:
        return "RANKER_LIMITED"
    if o_b >= 0.95 and n095 <= 2:
        return "SPARSE_GOOD"
    if o_b < 0.90:
        return "CANDIDATE_LIMITED"
    return "AMBIGUOUS"


def scene_metrics(true_scores: np.ndarray, predicted_scores: np.ndarray, proposals: np.ndarray) -> Dict[str, float]:
    true_scores = np.asarray(true_scores, dtype=np.float64)
    predicted_scores = np.asarray(predicted_scores, dtype=np.float64)
    selected = int(np.argmax(predicted_scores))
    oracle = int(np.argmax(true_scores))
    v_b = float(true_scores[selected])
    o_b = float(true_scores[oracle])
    g_b = o_b - v_b
    pred_order = np.argsort(-predicted_scores, kind="stable")
    oracle_pred_rank = int(np.flatnonzero(pred_order == oracle)[0]) + 1
    sorted_true = np.sort(true_scores)
    second = float(sorted_true[-2]) if len(sorted_true) > 1 else o_b
    sorted_pred = np.sort(predicted_scores)
    n_counts = {q: int(np.sum(true_scores >= q)) for q in QUALITY_THRESHOLDS}
    result: Dict[str, float] = {
        "V_B": v_b,
        "O_B": o_b,
        "G_B": g_b,
        "M_B": float(true_scores.mean()),
        "median_B": float(np.median(true_scores)),
        "std_B": float(true_scores.std()),
        "selected_index": selected,
        "oracle_index": oracle,
        "oracle_predicted_rank": oracle_pred_rank,
        "oracle_reciprocal_rank": 1.0 / oracle_pred_rank,
        "exact_oracle_index_hit": float(selected == oracle),
        "oracle_second_true_margin": o_b - second,
        "predicted_top1_top2_margin": float(sorted_pred[-1] - sorted_pred[-2]) if len(sorted_pred) > 1 else 0.0,
    }
    for q, count in n_counts.items():
        result[f"N_{q:.2f}"] = count
    for epsilon in EPSILON_ORACLE_VALUES:
        result[f"epsilon_oracle_hit_{epsilon:g}"] = float(v_b >= o_b - epsilon)
    for k in TOP_K_VALUES:
        result[f"top{k}_oracle"] = float(true_scores[pred_order[:k]].max())
    result.update(rank_metrics(true_scores, predicted_scores))
    result.update(geometry_metrics(proposals, true_scores >= 0.95))
    result.update(
        {
            "is_candidate_limited": float(o_b < 0.90),
            "is_sparse_good": float(o_b >= 0.95 and n_counts[0.95] <= 2),
            "is_ranker_limited": float(o_b >= 0.95 and v_b < 0.90 and g_b >= 0.05),
            "is_dense_rank_failure": float(o_b >= 0.95 and n_counts[0.95] >= 8 and v_b < 0.90),
            "is_saturated": float(v_b >= 0.95 and g_b <= 0.01),
            "failure_class": exclusive_failure_class(v_b, o_b, g_b, n_counts[0.95]),
        }
    )
    return result
