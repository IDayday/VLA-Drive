"""Pair-label threshold, hard-priority, and determinism tests."""

from __future__ import annotations

import numpy as np

from research.action_effect.pair_builder import (
    PairThresholds,
    RobustFeatureScale,
    build_scene_pairs,
    fit_robust_scales,
)


CONFIG = {
    "normalization": {"clip_normalized": 4.0},
    "distance": {
        "lambda_soft": 0.25,
        "minimum_nonduplicate_geometry_m": 0.05,
        "safety_boundary_geometry_max_m": 0.75,
        "heading_weight_m_per_rad": 0.25,
        "terminal_weight": 0.25,
    },
    "identifiability": {
        "ranking_tolerance": 1e-4,
        "low_confidence_pairs_become_ambiguous": True,
    },
}


def _row(candidate_id: str, progress: float, *, collision: bool = False) -> dict:
    return {
        "scene_id": "scene",
        "candidate_id": candidate_id,
        "candidate_index": ord(candidate_id[0]),
        "candidate_accepted": True,
        "exact": {
            "available": True,
            "map_metrics_available": True,
            "drivable_area_compliance": 1.0,
            "driving_direction_compliance": 1.0,
            "lane_keeping": 1.0,
            "history_comfort": 1.0,
            "extended_comfort": None,
            "centerline_progress_m": progress,
            "route_deviation_max_m": 0.0,
            "max_acceleration_mps2": 0.0,
            "max_deceleration_mps2": 0.0,
            "max_abs_jerk_mps3": 0.0,
            "max_abs_curvature_inv_m": 0.0,
            "static_object_collision": False,
        },
        "log_replay": {
            "available": True,
            "no_at_fault_collision": 0.0 if collision else 1.0,
            "traffic_light_compliance": 1.0,
            "dynamic_collision": collision,
            "pdm_score": 0.0 if collision else progress / 10.0,
        },
        "reactive_model": {"available": False},
    }


def _trajectory(offset: float) -> np.ndarray:
    trajectory = np.stack([np.arange(1, 9), np.zeros(8), np.zeros(8)], axis=1).astype(float)
    trajectory[:, 1] = np.linspace(0.0, offset, 8)
    return trajectory


def test_hard_difference_overrides_soft_threshold() -> None:
    rows = [_row("a", 5.0), _row("b", 5.0, collision=True)]
    scales = [RobustFeatureScale("centerline_progress_m", 5.0, 1.0, 0.0, 10.0, 1.0, 1.0, True)]
    pairs = build_scene_pairs(
        rows,
        {"a": _trajectory(0.0), "b": _trajectory(0.2)},
        scales,
        PairThresholds(0.1, 1.0, 0.2, 0.8),
        CONFIG,
    )
    assert pairs[0]["pair_type"] == "effect_divergent"
    assert pairs[0]["safety_boundary"] is True
    assert "no_at_fault_collision" in pairs[0]["hard_difference_fields"]


def test_distinct_equal_effects_are_equivalent_but_duplicates_are_not() -> None:
    rows = [_row("a", 5.0), _row("b", 5.0), _row("c", 5.0)]
    scales = [RobustFeatureScale("centerline_progress_m", 5.0, 1.0, 0.0, 10.0, 1.0, 1.0, True)]
    trajectories = {"a": _trajectory(0.0), "b": _trajectory(0.2), "c": _trajectory(0.0)}
    pairs = build_scene_pairs(rows, trajectories, scales, PairThresholds(0.1, 1.0, 0.2, 0.8), CONFIG)
    by_ids = {(pair["candidate_i"], pair["candidate_j"]): pair for pair in pairs}
    assert by_ids[("a", "b")]["pair_type"] == "effect_equivalent"
    assert by_ids[("a", "c")]["pair_type"] == "ambiguous"


def test_pair_output_does_not_depend_on_batch_order() -> None:
    rows = [_row("c", 9.0), _row("a", 5.0), _row("b", 5.0)]
    scales = [RobustFeatureScale("centerline_progress_m", 5.0, 1.0, 0.0, 10.0, 1.0, 1.0, True)]
    trajectories = {key: _trajectory(offset) for key, offset in (("a", 0.0), ("b", 0.2), ("c", 0.4))}
    first = build_scene_pairs(rows, trajectories, scales, PairThresholds(0.1, 1.0, 0.2, 0.8), CONFIG)
    second = build_scene_pairs(list(reversed(rows)), trajectories, scales, PairThresholds(0.1, 1.0, 0.2, 0.8), CONFIG)
    assert first == second


def test_robust_scale_uses_only_passed_training_rows() -> None:
    train = [_row("a", 0.0), _row("b", 1.0), _row("c", 2.0)]
    validation_outlier = _row("z", 10_000.0)
    scale = fit_robust_scales(
        train,
        ["centerline_progress_m"],
        minimum_coverage=1.0,
        minimum_scale=1e-3,
    )[0]
    scale_with_validation = fit_robust_scales(
        train + [validation_outlier],
        ["centerline_progress_m"],
        minimum_coverage=1.0,
        minimum_scale=1e-3,
    )[0]
    assert scale.median == 1.0
    assert scale_with_validation.median != scale.median
