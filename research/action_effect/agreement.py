"""Stratified log-replay versus reactive-model agreement diagnostics."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef

from research.action_effect.pair_builder import hard_vector


def replay_divergent_pair(row: Mapping[str, Any], *, soft_threshold: float) -> bool:
    """Recover replay divergence before confidence can relabel a pair ambiguous."""

    duplicate = bool(
        row.get("geometric_duplicate", row.get("pair_reason") == "geometric_duplicate")
    )
    soft = row.get("soft_consequence_distance")
    soft_divergent = (
        soft is not None
        and math.isfinite(float(soft))
        and float(soft) >= soft_threshold
    )
    return bool(
        not duplicate
        and (int(row.get("hard_difference_count", 0)) > 0 or soft_divergent)
    )


def candidate_is_unsafe(row: Mapping[str, Any], assumption: str) -> bool:
    """Return the six-hard-field unsafe conclusion under one assumption."""

    hard = hard_vector(row, assumption)
    return bool(
        hard[0] < 0.5
        or hard[1] < 0.5
        or hard[2] < 0.5
        or hard[3] < 0.5
        or hard[4] > 0.5
        or hard[5] > 0.5
    )


def scene_interaction_flags(
    rows: Sequence[Mapping[str, Any]],
    *,
    low_ttc_seconds: float,
    dynamic_clearance_m: float,
    score_tolerance: float,
) -> dict[str, set[str]]:
    """Derive deterministic scene subsets from available consequence labels."""

    flags = {"dynamic_interaction": set(), "low_ttc": set(), "idm_reacted": set()}
    for row in rows:
        if not row.get("candidate_accepted") or not row["log_replay"].get("available"):
            continue
        scene = str(row["scene_id"])
        replay = row["log_replay"]
        if (
            float(replay.get("minimum_dynamic_clearance_m", math.inf)) <= dynamic_clearance_m
            or int(replay.get("max_dynamic_agents_within_radius", 0)) > 0
        ):
            flags["dynamic_interaction"].add(scene)
        if float(replay.get("ttc_infraction_time_s", math.inf)) < low_ttc_seconds:
            flags["low_ttc"].add(scene)
        reactive = row["reactive_model"]
        if reactive.get("available"):
            replay_hard = hard_vector(row, "log_replay")
            reactive_hard = hard_vector(row, "reactive_model")
            numeric_change = any(
                abs(float(replay.get(field, 0.0)) - float(reactive.get(field, 0.0)))
                > score_tolerance
                for field in (
                    "pdm_score",
                    "ttc_infraction_time_s",
                    "minimum_dynamic_clearance_m",
                )
                if replay.get(field) is not None and reactive.get(field) is not None
            )
            if not np.array_equal(replay_hard, reactive_hard) or numeric_change:
                flags["idm_reacted"].add(scene)
    return flags


def binary_agreement_metrics(
    replay_positive: np.ndarray,
    reactive_positive: np.ndarray,
    replay_order: np.ndarray,
    reactive_order: np.ndarray,
) -> dict[str, float | int]:
    """Compute agreement metrics without hiding class imbalance or ties."""

    replay = np.asarray(replay_positive, dtype=bool)
    reactive = np.asarray(reactive_positive, dtype=bool)
    if len(replay) != len(reactive):
        raise ValueError("agreement labels must align")
    count = len(replay)
    if not count:
        return {
            "pair_count": 0,
            "disagreement_count": 0,
            "ranking_disagreement_count": 0,
            "raw_agreement": float("nan"),
            "positive_agreement": float("nan"),
            "cohen_kappa": float("nan"),
            "mcc": float("nan"),
            "tie_aware_kendall": float("nan"),
        }
    both_positive = int(np.sum(replay & reactive))
    replay_only = int(np.sum(replay & ~reactive))
    reactive_only = int(np.sum(~replay & reactive))
    positive_denominator = 2 * both_positive + replay_only + reactive_only
    positive_agreement = (
        2.0 * both_positive / positive_denominator
        if positive_denominator
        else 1.0
    )
    kappa = float(cohen_kappa_score(replay, reactive))
    mcc = float(matthews_corrcoef(replay, reactive))
    kendall = kendalltau(
        np.asarray(replay_order, dtype=np.int8),
        np.asarray(reactive_order, dtype=np.int8),
        variant="b",
    ).statistic
    return {
        "pair_count": count,
        "replay_positive_count": int(replay.sum()),
        "reactive_positive_count": int(reactive.sum()),
        "disagreement_count": int(np.sum(replay != reactive)),
        "ranking_disagreement_count": int(
            np.sum(np.asarray(replay_order) != np.asarray(reactive_order))
        ),
        "raw_agreement": float(np.mean(replay == reactive)),
        "positive_agreement": float(positive_agreement),
        "cohen_kappa": 0.0 if not math.isfinite(float(kappa)) else kappa,
        "mcc": 0.0 if not math.isfinite(float(mcc)) else mcc,
        "tie_aware_kendall": (
            0.0 if kendall is None or not math.isfinite(float(kendall)) else float(kendall)
        ),
    }
