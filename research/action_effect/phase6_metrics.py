"""Scene-bootstrap representation and structured-effect metrics for Phase 6."""

from __future__ import annotations

from dataclasses import asdict
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from research.action_effect.metrics import MetricInterval


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.ptp(left) <= 1.0e-12 or np.ptp(right) <= 1.0e-12:
        return 0.0
    value = spearmanr(left, right).statistic
    return 0.0 if not math.isfinite(float(value)) else float(value)


def _safe_auc(labels: np.ndarray, scores: np.ndarray, *, precision_recall: bool) -> float:
    labels = np.asarray(labels, dtype=bool)
    if not len(labels) or labels.all() or (~labels).all():
        return float("nan")
    function = average_precision_score if precision_recall else roc_auc_score
    return float(function(labels, scores))


def _interval(point: float, values: Sequence[float], samples: int, confidence: float) -> MetricInterval:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    alpha = (1.0 - confidence) / 2.0
    low, high = (
        np.quantile(finite, [alpha, 1.0 - alpha])
        if len(finite)
        else (float("nan"), float("nan"))
    )
    return MetricInterval(float(point), float(low), float(high), samples, confidence)


def latent_diagnostics(latent: np.ndarray, *, relative_tolerance: float) -> dict[str, float | int]:
    """Detect norm inflation and representation collapse in raw/normalized latents."""

    value = np.asarray(latent, dtype=np.float64)
    if value.ndim != 2 or len(value) < 2:
        raise ValueError("latent diagnostics require at least two rank-2 samples")
    norm = np.linalg.norm(value, axis=1)
    normalized = value / np.maximum(norm[:, None], 1.0e-12)
    covariance = np.cov(normalized, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)
    maximum = max(float(eigenvalues[-1]), 1.0e-12)
    rank = int(np.sum(eigenvalues > maximum * relative_tolerance))
    return {
        "latent_norm_mean": float(np.mean(norm)),
        "latent_norm_std": float(np.std(norm)),
        "normalized_latent_variance": float(np.var(normalized, axis=0).mean()),
        "covariance_effective_rank": rank,
        "covariance_max_eigenvalue": maximum,
    }


def representation_metrics(
    *,
    latent_by_candidate: np.ndarray,
    candidate_ids: np.ndarray,
    pair_rows: Sequence[Mapping[str, Any]],
    selected_scene_ids: Sequence[str],
    candidate_valid: np.ndarray,
    perturbation: np.ndarray,
    heldout_family: str,
    bootstrap_samples: int,
    confidence: float,
    seed: int,
) -> tuple[dict[str, MetricInterval], dict[str, Any]]:
    """Evaluate normalized-latent pair structure with whole-scene resampling."""

    lookup = {candidate: index for index, candidate in enumerate(candidate_ids)}
    selected = set(str(value) for value in selected_scene_ids)
    rows: list[Mapping[str, Any]] = []
    left: list[int] = []
    right: list[int] = []
    for row in pair_rows:
        if str(row["scene_id"]) not in selected:
            continue
        left_index = lookup[str(row["candidate_i"])]
        right_index = lookup[str(row["candidate_j"])]
        if candidate_valid[left_index] and candidate_valid[right_index]:
            rows.append(row)
            left.append(left_index)
            right.append(right_index)
    left_array = np.asarray(left, dtype=np.int64)
    right_array = np.asarray(right, dtype=np.int64)
    latent = np.asarray(latent_by_candidate, dtype=np.float64)
    normalized = latent / np.maximum(np.linalg.norm(latent, axis=1, keepdims=True), 1.0e-12)
    distance = np.linalg.norm(normalized[left_array] - normalized[right_array], axis=1)
    true_distance = np.asarray([float(row["consequence_distance"]) for row in rows])
    pair_type = np.asarray([str(row["pair_type"]) for row in rows], dtype=str)
    pair_scene = np.asarray([str(row["scene_id"]) for row in rows], dtype=str)
    safety = np.asarray([bool(row.get("safety_boundary")) for row in rows], dtype=bool)
    heldout = (perturbation[left_array] == heldout_family) | (perturbation[right_array] == heldout_family)
    grouped: dict[str, list[int]] = {str(scene): [] for scene in selected_scene_ids}
    for index, scene in enumerate(pair_scene):
        grouped.setdefault(str(scene), []).append(index)
    by_scene = {
        scene: np.asarray(indices, dtype=np.int64) for scene, indices in grouped.items()
    }
    scene_alignment: dict[str, float] = {}
    heldout_scene_alignment: dict[str, float] = {}
    for scene in selected_scene_ids:
        scene_key = str(scene)
        indices = by_scene[scene_key]
        if len(indices) >= 2:
            scene_alignment[scene_key] = _safe_spearman(
                distance[indices], true_distance[indices]
            )
        heldout_indices = indices[heldout[indices]]
        if len(heldout_indices) >= 2:
            heldout_scene_alignment[scene_key] = _safe_spearman(
                distance[heldout_indices], true_distance[heldout_indices]
            )

    def evaluate(scenes: Sequence[str], *, heldout_only: bool = False) -> dict[str, float]:
        indices = np.concatenate([by_scene[str(scene)] for scene in scenes])
        if heldout_only:
            indices = indices[heldout[indices]]
        equivalent = indices[pair_type[indices] == "effect_equivalent"]
        divergent = indices[pair_type[indices] == "effect_divergent"]
        discriminative = np.concatenate((equivalent, divergent))
        labels = np.concatenate(
            (np.zeros(len(equivalent), dtype=bool), np.ones(len(divergent), dtype=bool))
        )
        scores = distance[discriminative]
        action_gap = float(np.mean(distance[divergent])) if len(divergent) else float("nan")
        leakage = float(np.mean(distance[equivalent])) if len(equivalent) else float("nan")
        alignment = heldout_scene_alignment if heldout_only else scene_alignment
        per_scene = [alignment[str(scene)] for scene in scenes if str(scene) in alignment]
        return {
            "per_scene_effect_alignment": float(np.mean(per_scene)) if per_scene else float("nan"),
            "pooled_effect_alignment": _safe_spearman(distance[indices], true_distance[indices]),
            "action_gap": action_gap,
            "equivalence_leakage": leakage,
            "separation_ratio": action_gap / max(leakage, 1.0e-12),
            "equivalent_divergent_auroc": _safe_auc(labels, scores, precision_recall=False),
            "equivalent_divergent_auprc": _safe_auc(labels, scores, precision_recall=True),
            "safety_boundary_auprc": _safe_auc(safety[indices], distance[indices], precision_recall=True),
        }

    point = evaluate(list(selected_scene_ids))
    heldout_point = evaluate(list(selected_scene_ids), heldout_only=True)
    point.update({f"heldout_family_{name}": value for name, value in heldout_point.items()})
    rng = np.random.default_rng(seed)
    scenes = np.asarray(selected_scene_ids, dtype=str)
    draws = {name: [] for name in point}
    for _ in range(bootstrap_samples):
        sampled = rng.choice(scenes, size=len(scenes), replace=True).tolist()
        values = evaluate(sampled)
        heldout_values = evaluate(sampled, heldout_only=True)
        values.update({f"heldout_family_{name}": value for name, value in heldout_values.items()})
        for name, value in values.items():
            if math.isfinite(value):
                draws[name].append(value)
    intervals = {
        name: _interval(value, draws[name], bootstrap_samples, confidence)
        for name, value in point.items()
    }
    scene_metrics = {}
    for scene in selected_scene_ids:
        values = evaluate([str(scene)])
        heldout_values = evaluate([str(scene)], heldout_only=True)
        values.update({f"heldout_family_{name}": value for name, value in heldout_values.items()})
        scene_metrics[str(scene)] = values
    return intervals, {
        "pair_distance": distance,
        "true_distance": true_distance,
        "pair_type": pair_type,
        "pair_scene": pair_scene,
        "candidate_i": np.asarray([str(row["candidate_i"]) for row in rows], dtype=str),
        "candidate_j": np.asarray([str(row["candidate_j"]) for row in rows], dtype=str),
        "safety_boundary": safety,
        "heldout_family": heldout,
        "pair_count": len(rows),
        "scene_metrics": scene_metrics,
    }


def decoded_effect_prediction(raw: np.ndarray) -> np.ndarray:
    """Decode occupancy/collision/footprint logits and retain regressions."""

    result = np.asarray(raw, dtype=np.float32).copy()
    for channel in (0, 7, 8):
        value = np.clip(result[:, :, channel], -30.0, 30.0)
        result[:, :, channel] = 1.0 / (1.0 + np.exp(-value))
    return result


def _binary_metrics(target: np.ndarray, prediction: np.ndarray, positive_weight: float) -> dict[str, float]:
    probability = np.clip(prediction.reshape(-1), 1.0e-6, 1.0 - 1.0e-6)
    label = target.reshape(-1) > 0.5
    balanced_bce = -np.mean(
        positive_weight * label * np.log(probability) + (~label) * np.log(1.0 - probability)
    )
    predicted = probability >= 0.5
    intersection = int(np.sum(predicted & label))
    union = int(np.sum(predicted | label))
    return {
        "balanced_bce": float(balanced_bce),
        "auprc": _safe_auc(label, probability, precision_recall=True),
        "iou": float(intersection / union) if union else 1.0,
    }


def channel_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    positive_weight: Sequence[float],
) -> list[dict[str, float]]:
    """Compute loss-aligned metrics for all nine effect channels."""

    target = np.asarray(target, dtype=np.float32)
    prediction = np.asarray(prediction, dtype=np.float32)
    if target.shape != prediction.shape or target.ndim != 5 or target.shape[2] != 9:
        raise ValueError("effect channel metrics require matching [N,H,9,R,R]")
    result: list[dict[str, float]] = []
    binary_weights = {0: positive_weight[0], 7: positive_weight[1], 8: positive_weight[2]}
    for channel in range(9):
        truth = target[:, :, channel]
        estimate = prediction[:, :, channel]
        if channel in binary_weights:
            metrics = _binary_metrics(truth, estimate, float(binary_weights[channel]))
            if channel == 8:
                probability = np.clip(estimate, 0.0, 1.0)
                intersection = np.sum(probability * truth, axis=(1, 2, 3))
                denominator = np.sum(probability, axis=(1, 2, 3)) + np.sum(
                    truth, axis=(1, 2, 3)
                )
                metrics["dice"] = float(np.mean((2 * intersection + 1) / (denominator + 1)))
        elif channel in {1, 2, 3, 6}:
            difference = estimate - truth
            absolute = np.abs(difference)
            huber = np.where(absolute < 1.0, 0.5 * difference**2, absolute - 0.5)
            metrics = {"huber": float(np.mean(huber)), "normalized_l1": float(np.mean(absolute))}
        else:
            mask = target[:, :, 0] > 0.5
            difference = estimate - truth
            absolute = np.abs(difference[mask])
            huber = np.where(absolute < 1.0, 0.5 * absolute**2, absolute - 0.5)
            metrics = {
                "masked_huber": float(np.mean(huber)) if len(huber) else float("nan"),
                "masked_l1": float(np.mean(absolute)) if len(absolute) else float("nan"),
            }
        result.append(metrics)
    return result


def effect_action_shuffle_gap(
    *,
    target: np.ndarray,
    prediction: np.ndarray,
    scene_ids: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """Measure candidate-conditioned effect error after within-scene shuffling."""

    target = np.asarray(target, dtype=np.float32)
    prediction = np.asarray(prediction, dtype=np.float32)
    error = np.mean(np.abs(prediction - target), axis=(1, 2, 3, 4))
    per_scene: dict[str, float] = {}
    for scene in sorted(set(scene_ids.tolist())):
        indices = np.flatnonzero(scene_ids == scene)
        if len(indices) < 2:
            continue
        shifted = np.roll(indices, 1)
        shuffled = np.mean(np.abs(prediction[shifted] - target[indices]), axis=(1, 2, 3, 4))
        per_scene[str(scene)] = float(np.mean(shuffled - error[indices]))
    return float(np.mean(list(per_scene.values()))), per_scene


def effect_channel_shuffle_gap(
    *,
    target: np.ndarray,
    prediction: np.ndarray,
    scene_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return one within-scene candidate-shuffle gap vector per scene."""

    target = np.asarray(target, dtype=np.float32)
    prediction = np.asarray(prediction, dtype=np.float32)
    result: dict[str, np.ndarray] = {}
    for scene in sorted(set(scene_ids.tolist())):
        indices = np.flatnonzero(scene_ids == scene)
        if len(indices) < 2:
            continue
        shifted = np.roll(indices, 1)
        correct = np.mean(np.abs(prediction[indices] - target[indices]), axis=(0, 1, 3, 4))
        shuffled = np.mean(np.abs(prediction[shifted] - target[indices]), axis=(0, 1, 3, 4))
        result[str(scene)] = shuffled - correct
    return result


def effect_primary_channel_shuffle_gap(
    *,
    target: np.ndarray,
    prediction: np.ndarray,
    scene_ids: np.ndarray,
    positive_weight: Sequence[float],
) -> dict[str, np.ndarray]:
    """Measure shuffle damage using each channel's declared primary metric."""

    target = np.asarray(target, dtype=np.float32)
    prediction = np.asarray(prediction, dtype=np.float32)
    result: dict[str, np.ndarray] = {}
    for scene in sorted(set(scene_ids.tolist())):
        indices = np.flatnonzero(scene_ids == scene)
        if len(indices) < 2:
            continue
        shifted = np.roll(indices, 1)
        truth = target[indices]
        correct_prediction = prediction[indices]
        shuffled_prediction = prediction[shifted]
        gap = np.zeros(9, dtype=np.float64)
        for channel in range(9):
            if channel in {0, 7}:
                weight_index = 0 if channel == 0 else 1
                label = truth[:, :, channel] > 0.5

                def balanced_bce(value: np.ndarray) -> float:
                    probability = np.clip(value, 1.0e-6, 1.0 - 1.0e-6)
                    return float(
                        -np.mean(
                            float(positive_weight[weight_index]) * label * np.log(probability)
                            + (~label) * np.log(1.0 - probability)
                        )
                    )

                gap[channel] = balanced_bce(
                    shuffled_prediction[:, :, channel]
                ) - balanced_bce(correct_prediction[:, :, channel])
            elif channel in {1, 2, 3, 6}:
                gap[channel] = float(
                    np.mean(
                        np.abs(shuffled_prediction[:, :, channel] - truth[:, :, channel])
                    )
                    - np.mean(
                        np.abs(correct_prediction[:, :, channel] - truth[:, :, channel])
                    )
                )
            elif channel in {4, 5}:
                mask = truth[:, :, 0] > 0.5
                if np.any(mask):
                    gap[channel] = float(
                        np.mean(
                            np.abs(
                                shuffled_prediction[:, :, channel][mask]
                                - truth[:, :, channel][mask]
                            )
                        )
                        - np.mean(
                            np.abs(
                                correct_prediction[:, :, channel][mask]
                                - truth[:, :, channel][mask]
                            )
                        )
                    )
                else:
                    gap[channel] = float("nan")
            else:
                label = truth[:, :, channel]

                def dice(value: np.ndarray) -> float:
                    intersection = np.sum(value * label, axis=(1, 2, 3))
                    denominator = np.sum(value, axis=(1, 2, 3)) + np.sum(
                        label, axis=(1, 2, 3)
                    )
                    return float(np.mean((2.0 * intersection + 1.0) / (denominator + 1.0)))

                gap[channel] = dice(correct_prediction[:, :, channel]) - dice(
                    shuffled_prediction[:, :, channel]
                )
        result[str(scene)] = gap
    return result


def effect_channel_pair_sensitivity(
    *,
    target: np.ndarray,
    prediction: np.ndarray,
    scene_ids: np.ndarray,
    candidate_ids: np.ndarray,
    pair_rows: Sequence[Mapping[str, Any]],
    selected_scene_ids: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Measure target/predicted action gaps for divergent pairs in each scene.

    Inputs are the compact arrays for one evaluation split. Returning one row
    per scene keeps subsequent confidence intervals scene-bootstrap based.
    """

    target = np.asarray(target, dtype=np.float32)
    prediction = np.asarray(prediction, dtype=np.float32)
    scene_ids = np.asarray(scene_ids, dtype=str)
    candidate_ids = np.asarray(candidate_ids, dtype=str)
    if target.shape != prediction.shape or target.ndim != 5 or target.shape[2] != 9:
        raise ValueError("effect sensitivity requires matching [N,H,9,R,R] arrays")
    if len(scene_ids) != len(target) or len(candidate_ids) != len(target):
        raise ValueError("effect sensitivity identifiers do not align with targets")
    lookup = {str(candidate): index for index, candidate in enumerate(candidate_ids)}
    selected = set(str(scene) for scene in selected_scene_ids)
    grouped: dict[str, list[tuple[int, int]]] = {}
    for row in pair_rows:
        scene = str(row["scene_id"])
        if scene not in selected or str(row["pair_type"]) != "effect_divergent":
            continue
        left = lookup.get(str(row["candidate_i"]))
        right = lookup.get(str(row["candidate_j"]))
        if left is None or right is None:
            continue
        if scene_ids[left] != scene or scene_ids[right] != scene:
            raise RuntimeError("pair candidate does not belong to its declared scene")
        grouped.setdefault(scene, []).append((left, right))
    result: dict[str, dict[str, float]] = {}
    for scene, pairs in grouped.items():
        left = np.asarray([pair[0] for pair in pairs], dtype=np.int64)
        right = np.asarray([pair[1] for pair in pairs], dtype=np.int64)
        target_distance = np.sqrt(
            np.mean(np.square(target[left] - target[right]), axis=(1, 3, 4))
        )
        predicted_distance = np.sqrt(
            np.mean(np.square(prediction[left] - prediction[right]), axis=(1, 3, 4))
        )
        target_gap = np.mean(target_distance, axis=0)
        predicted_gap = np.mean(predicted_distance, axis=0)
        values: dict[str, float] = {}
        for channel in range(target.shape[2]):
            values[f"channel_{channel}_target_action_gap"] = float(target_gap[channel])
            values[f"channel_{channel}_predicted_action_gap"] = float(predicted_gap[channel])
            values[f"channel_{channel}_predicted_target_sensitivity_ratio"] = float(
                predicted_gap[channel] / max(float(target_gap[channel]), 1.0e-12)
            )
        result[scene] = values
    return result


def intervals_as_json(values: Mapping[str, MetricInterval]) -> dict[str, Any]:
    """Serialize a mapping of confidence intervals."""

    return {name: asdict(value) for name, value in values.items()}


def gate3_conditions(
    *,
    paired_deltas: Mapping[str, MetricInterval],
    aee_action_gap: float,
    global_action_gap: float,
    aee_equivalence_leakage: float,
    global_equivalence_leakage: float,
    alignment_improves_each_seed: bool,
    structured_effect_success: bool,
) -> dict[str, Any]:
    """Apply the predeclared Gate-3 criteria without result-dependent tuning."""

    leakage_reduction = 1.0 - aee_equivalence_leakage / max(
        global_equivalence_leakage, 1.0e-12
    )
    action_gap_retention = aee_action_gap / max(global_action_gap, 1.0e-12)
    absolute = {
        "effect_alignment_improves_each_seed": bool(alignment_improves_each_seed),
        "effect_alignment_scene_ci_excludes_zero": paired_deltas[
            "aee_vs_absolute_alignment"
        ].ci_low
        > 0,
        "pair_or_safety_auprc_improvement": max(
            paired_deltas["aee_vs_absolute_pair_auprc"].point,
            paired_deltas["aee_vs_absolute_safety_auprc"].point,
        )
        > 0,
        "false_safe_reduction": paired_deltas["aee_vs_absolute_false_safe"].point < 0,
        "heldout_family_improvement": paired_deltas[
            "aee_vs_absolute_heldout_alignment"
        ].point
        > 0,
    }
    global_comparison = {
        "equivalence_leakage_reduction": leakage_reduction,
        "leakage_reduction_at_least_20_percent": leakage_reduction >= 0.20,
        "action_gap_retention": action_gap_retention,
        "action_gap_retention_at_least_90_percent": action_gap_retention >= 0.90,
        "separation_ratio_significant_improvement": paired_deltas[
            "aee_vs_global_separation_ratio"
        ].ci_low
        > 0,
        # A positive error delta is a significant degradation only when its
        # entire scene-bootstrap interval lies above zero.
        "structured_target_not_significantly_degraded": paired_deltas[
            "aee_vs_global_structured_error"
        ].ci_low
        <= 0,
    }
    pass_global = all(
        value
        for key, value in global_comparison.items()
        if key not in {"equivalence_leakage_reduction", "action_gap_retention"}
    )
    return {
        "decision": (
            "PASS"
            if all(absolute.values()) and pass_global and structured_effect_success
            else "FAIL"
        ),
        "aee_vs_absolute": absolute,
        "aee_vs_global": global_comparison,
        "structured_effect_success": bool(structured_effect_success),
    }
