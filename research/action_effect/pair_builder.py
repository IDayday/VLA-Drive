"""Robust consequence distances and action-effect pair construction."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


HARD_FIELDS = (
    "no_at_fault_collision",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "traffic_light_compliance",
    "static_object_collision",
    "dynamic_collision",
)


@dataclass(frozen=True)
class RobustFeatureScale:
    """Train-split-only robust statistics for one soft consequence."""

    field: str
    median: float | None
    iqr: float | None
    quantile_05: float | None
    quantile_95: float | None
    scale: float | None
    coverage: float
    active: bool


@dataclass(frozen=True)
class PairThresholds:
    """Data-derived low/high soft-distance thresholds."""

    equivalent: float
    divergent: float
    equivalent_quantile: float
    divergent_quantile: float


def hard_vector(row: Mapping[str, Any], assumption: str = "log_replay") -> np.ndarray:
    """Return the six-component hard vector without crossing provenance."""

    exact = row["exact"]
    dynamic = row[assumption]
    if not exact.get("map_metrics_available", exact.get("available", False)) or not dynamic.get("available", False):
        raise ValueError(f"hard consequence unavailable for {row.get('candidate_id')}")
    return np.asarray(
        [
            dynamic["no_at_fault_collision"],
            exact["drivable_area_compliance"],
            exact["driving_direction_compliance"],
            dynamic["traffic_light_compliance"],
            float(bool(exact["static_object_collision"])),
            float(bool(dynamic["dynamic_collision"])),
        ],
        dtype=np.float64,
    )


def soft_value(row: Mapping[str, Any], field: str, assumption: str = "log_replay") -> float | None:
    """Resolve one configured soft field from its declared namespace."""

    if field in {
        "centerline_progress_m",
        "lane_keeping",
        "history_comfort",
        "extended_comfort",
        "route_deviation_max_m",
        "max_acceleration_mps2",
        "max_deceleration_mps2",
        "max_abs_jerk_mps3",
        "max_abs_curvature_inv_m",
    }:
        value = row["exact"].get(field)
    else:
        value = row[assumption].get(field)
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def fit_robust_scales(
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
    *,
    minimum_coverage: float,
    minimum_scale: float,
) -> list[RobustFeatureScale]:
    """Fit medians/IQRs from accepted training rows only."""

    accepted = [row for row in rows if row.get("candidate_accepted") and row["log_replay"].get("available")]
    if not accepted:
        raise ValueError("no accepted training consequences available for robust scaling")
    result: list[RobustFeatureScale] = []
    for field in fields:
        values = [soft_value(row, field) for row in accepted]
        finite = np.asarray([value for value in values if value is not None], dtype=np.float64)
        coverage = len(finite) / len(accepted)
        if len(finite):
            q05, q25, median, q75, q95 = np.quantile(finite, [0.05, 0.25, 0.5, 0.75, 0.95])
            iqr = float(q75 - q25)
            # The 5--95 range divided by the Gaussian span is a stable fallback
            # when an otherwise useful metric has a zero IQR.
            scale = max(iqr, float((q95 - q05) / 3.29), minimum_scale)
            active = coverage >= minimum_coverage and float(q95 - q05) > minimum_scale
            result.append(
                RobustFeatureScale(
                    field=field,
                    median=float(median),
                    iqr=iqr,
                    quantile_05=float(q05),
                    quantile_95=float(q95),
                    scale=scale,
                    coverage=coverage,
                    active=active,
                )
            )
        else:
            result.append(RobustFeatureScale(field, None, None, None, None, None, coverage, False))
    if not any(scale.active for scale in result):
        raise ValueError("all soft consequence fields were inactive")
    return result


def normalized_soft_vector(
    row: Mapping[str, Any],
    scales: Sequence[RobustFeatureScale],
    *,
    clip: float,
    assumption: str = "log_replay",
) -> np.ndarray:
    """Normalize active fields, retaining NaN for an assumption's missing value."""

    values: list[float] = []
    for scale in scales:
        if not scale.active:
            continue
        value = soft_value(row, scale.field, assumption)
        if value is None or scale.median is None or scale.scale is None:
            values.append(np.nan)
        else:
            values.append(float(np.clip((value - scale.median) / scale.scale, -clip, clip)))
    return np.asarray(values, dtype=np.float64)


def soft_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Masked L1 distance, normalized for occasionally missing dimensions."""

    mask = np.isfinite(left) & np.isfinite(right)
    if not np.any(mask):
        return float("nan")
    observed = int(np.sum(mask))
    return float(np.sum(np.abs(left[mask] - right[mask])) * len(left) / observed)


def geometric_distance(
    left: np.ndarray,
    right: np.ndarray,
    *,
    heading_weight_m_per_rad: float,
    terminal_weight: float,
) -> float:
    """Policy-local pose distance in interpretable metre-equivalent units."""

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    mean_xy = float(np.mean(np.linalg.norm(left[:, :2] - right[:, :2], axis=1)))
    terminal_xy = float(np.linalg.norm(left[-1, :2] - right[-1, :2]))
    heading_delta = np.abs((left[:, 2] - right[:, 2] + np.pi) % (2 * np.pi) - np.pi)
    return mean_xy + terminal_weight * terminal_xy + heading_weight_m_per_rad * float(np.mean(heading_delta))


def fit_pair_thresholds(
    soft_distances: Iterable[float],
    *,
    equivalent_quantile: float,
    divergent_quantile: float,
    equivalent_floor: float,
    divergent_floor: float,
) -> PairThresholds:
    """Fit non-overlapping low/high thresholds from train-scene pairs."""

    values = np.asarray([value for value in soft_distances if math.isfinite(value)], dtype=np.float64)
    if not len(values):
        raise ValueError("no finite train-pair distances available")
    equivalent = max(float(np.quantile(values, equivalent_quantile)), equivalent_floor)
    divergent = max(float(np.quantile(values, divergent_quantile)), divergent_floor, equivalent + 1e-6)
    return PairThresholds(equivalent, divergent, equivalent_quantile, divergent_quantile)


def _order(left: float, right: float, tolerance: float) -> int:
    delta = left - right
    return 0 if abs(delta) <= tolerance else (1 if delta > 0 else -1)


def _identifiability(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    ranking_tolerance: float,
) -> dict[str, Any]:
    if not left["reactive_model"].get("available") or not right["reactive_model"].get("available"):
        return {
            "pair_confidence": "unassessed",
            "log_replay_order": _order(
                float(left["log_replay"]["pdm_score"]),
                float(right["log_replay"]["pdm_score"]),
                ranking_tolerance,
            ),
            "reactive_order": None,
            "hard_agreement": None,
            "pairwise_hard_relation_agreement": None,
            "log_replay_hard_relation": None,
            "reactive_hard_relation": None,
            "soft_rank_agreement": None,
        }
    replay_left, replay_right = hard_vector(left), hard_vector(right)
    reactive_left = hard_vector(left, "reactive_model")
    reactive_right = hard_vector(right, "reactive_model")
    hard_agreement = bool(
        np.array_equal(replay_left, reactive_left) and np.array_equal(replay_right, reactive_right)
    )
    replay_hard_relation = "different" if np.any(replay_left != replay_right) else "same"
    reactive_hard_relation = "different" if np.any(reactive_left != reactive_right) else "same"
    pairwise_hard_relation_agreement = replay_hard_relation == reactive_hard_relation
    replay_order = _order(
        float(left["log_replay"]["pdm_score"]),
        float(right["log_replay"]["pdm_score"]),
        ranking_tolerance,
    )
    reactive_order = _order(
        float(left["reactive_model"]["pdm_score"]),
        float(right["reactive_model"]["pdm_score"]),
        ranking_tolerance,
    )
    rank_agreement = replay_order == reactive_order
    if hard_agreement:
        confidence = "high"
    elif pairwise_hard_relation_agreement and rank_agreement:
        confidence = "medium"
    else:
        confidence = "low"
    return {
        "pair_confidence": confidence,
        "log_replay_order": replay_order,
        "reactive_order": reactive_order,
        "hard_agreement": hard_agreement,
        "pairwise_hard_relation_agreement": pairwise_hard_relation_agreement,
        "log_replay_hard_relation": replay_hard_relation,
        "reactive_hard_relation": reactive_hard_relation,
        "soft_rank_agreement": rank_agreement,
    }


def build_scene_pairs(
    rows: Sequence[Mapping[str, Any]],
    trajectories: Mapping[str, np.ndarray],
    scales: Sequence[RobustFeatureScale],
    thresholds: PairThresholds,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Classify all accepted candidate pairs in deterministic ID order."""

    accepted = sorted(
        [row for row in rows if row.get("candidate_accepted") and row["log_replay"].get("available")],
        key=lambda row: str(row["candidate_id"]),
    )
    distance_cfg = config["distance"]
    ident_cfg = config["identifiability"]
    normalized = {
        row["candidate_id"]: normalized_soft_vector(
            row,
            scales,
            clip=float(config["normalization"]["clip_normalized"]),
        )
        for row in accepted
    }
    pairs: list[dict[str, Any]] = []
    for left, right in combinations(accepted, 2):
        left_id, right_id = str(left["candidate_id"]), str(right["candidate_id"])
        hard_left, hard_right = hard_vector(left), hard_vector(right)
        hard_diff_mask = ~np.isclose(hard_left, hard_right, atol=1e-8)
        hard_diff_count = int(np.sum(hard_diff_mask))
        soft = soft_distance(normalized[left_id], normalized[right_id])
        active_scales = [scale for scale in scales if scale.active]
        soft_difference_by_field = {
            scale.field: float(abs(normalized[left_id][index] - normalized[right_id][index]))
            if np.isfinite(normalized[left_id][index]) and np.isfinite(normalized[right_id][index])
            else None
            for index, scale in enumerate(active_scales)
        }
        geometry = geometric_distance(
            trajectories[left_id],
            trajectories[right_id],
            heading_weight_m_per_rad=float(distance_cfg["heading_weight_m_per_rad"]),
            terminal_weight=float(distance_cfg["terminal_weight"]),
        )
        duplicate = geometry < float(distance_cfg["minimum_nonduplicate_geometry_m"])
        identifiability = _identifiability(
            left,
            right,
            ranking_tolerance=float(ident_cfg["ranking_tolerance"]),
        )
        if duplicate:
            pair_type = "ambiguous"
            reason = "geometric_duplicate"
        elif hard_diff_count:
            pair_type = "effect_divergent"
            reason = "hard_consequence_difference"
        elif math.isfinite(soft) and soft <= thresholds.equivalent:
            pair_type = "effect_equivalent"
            reason = "low_soft_consequence_distance"
        elif math.isfinite(soft) and soft >= thresholds.divergent:
            pair_type = "effect_divergent"
            reason = "high_soft_consequence_distance"
        else:
            pair_type = "ambiguous"
            reason = "between_thresholds"
        if (
            identifiability["pair_confidence"] == "low"
            and bool(ident_cfg["low_confidence_pairs_become_ambiguous"])
        ):
            pair_type = "ambiguous"
            reason = "traffic_assumption_conflict"
        safety_boundary = bool(
            hard_diff_count
            and geometry <= float(distance_cfg["safety_boundary_geometry_max_m"])
        )
        pairs.append(
            {
                "scene_id": left["scene_id"],
                "candidate_i": left_id,
                "candidate_j": right_id,
                "candidate_i_index": int(left["candidate_index"]),
                "candidate_j_index": int(right["candidate_index"]),
                "pair_type": pair_type,
                "pair_reason": reason,
                "pair_confidence": identifiability["pair_confidence"],
                "hard_difference_count": hard_diff_count,
                "hard_difference_fields": [
                    field for field, differs in zip(HARD_FIELDS, hard_diff_mask) if differs
                ],
                "soft_consequence_distance": soft,
                "soft_difference_by_field": soft_difference_by_field,
                "consequence_distance": hard_diff_count
                + float(distance_cfg["lambda_soft"]) * soft,
                "geometric_distance": geometry,
                "safety_boundary": safety_boundary,
                **{key: value for key, value in identifiability.items() if key != "pair_confidence"},
            }
        )
    return pairs
