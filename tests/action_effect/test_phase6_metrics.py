from __future__ import annotations

import numpy as np

from research.action_effect.metrics import MetricInterval
from research.action_effect.phase6_metrics import (
    channel_metrics,
    decoded_effect_prediction,
    effect_channel_pair_sensitivity,
    effect_primary_channel_shuffle_gap,
    gate3_conditions,
    latent_diagnostics,
    representation_metrics,
)
from scripts.action_effect.train_phase6_world_probe import _binary_channel_means


def test_latent_diagnostics_and_normalized_pair_metrics() -> None:
    candidate_ids = np.asarray(["a0", "a1", "a2", "b0", "b1", "b2"])
    latent = np.asarray(
        [[1, 0], [0.99, 0.01], [-1, 0], [0, 1], [0.01, 0.99], [0, -1]],
        dtype=np.float32,
    )
    pairs = []
    for scene, offset in (("a", 0), ("b", 3)):
        pairs.extend(
            [
                {
                    "scene_id": scene,
                    "candidate_i": candidate_ids[offset],
                    "candidate_j": candidate_ids[offset + 1],
                    "pair_type": "effect_equivalent",
                    "consequence_distance": 0.1,
                    "safety_boundary": False,
                },
                {
                    "scene_id": scene,
                    "candidate_i": candidate_ids[offset],
                    "candidate_j": candidate_ids[offset + 2],
                    "pair_type": "effect_divergent",
                    "consequence_distance": 2.0,
                    "safety_boundary": True,
                },
            ]
        )
    intervals, details = representation_metrics(
        latent_by_candidate=latent,
        candidate_ids=candidate_ids,
        pair_rows=pairs,
        selected_scene_ids=("a", "b"),
        candidate_valid=np.ones(6, dtype=bool),
        perturbation=np.asarray(["anchor", "held", "x", "anchor", "held", "x"]),
        heldout_family="held",
        bootstrap_samples=20,
        confidence=0.95,
        seed=3,
    )
    assert intervals["action_gap"].point > intervals["equivalence_leakage"].point
    assert intervals["per_scene_effect_alignment"].point > 0
    assert details["pair_count"] == 4
    diagnostics = latent_diagnostics(latent, relative_tolerance=1.0e-3)
    assert diagnostics["covariance_effective_rank"] > 0


def test_channel_metrics_follow_declared_channel_types() -> None:
    target = np.zeros((2, 3, 9, 4, 4), dtype=np.float32)
    target[:, :, 0, 1, 1] = 1.0
    target[:, :, 6] = 1.0
    target[:, :, 8, 1:3, 1:3] = 1.0
    raw = np.zeros_like(target)
    prediction = decoded_effect_prediction(raw)
    metrics = channel_metrics(target, prediction, positive_weight=(2.0, 2.0, 1.0))
    assert len(metrics) == 9
    assert set(metrics[0]) == {"balanced_bce", "auprc", "iou"}
    assert set(metrics[1]) == {"huber", "normalized_l1"}
    assert set(metrics[4]) == {"masked_huber", "masked_l1"}
    assert "dice" in metrics[8]


def test_binary_channel_means_streams_selected_rows() -> None:
    target = np.zeros((5, 3, 9, 2, 2), dtype=np.float16)
    target[1, :, 0] = 1
    target[3, :, 7] = 1
    target[[1, 3], :, 8] = 0.5
    result = _binary_channel_means(target, np.asarray([1, 3]), batch_size=1)
    np.testing.assert_allclose(result, [0.5, 0.5, 0.5])


def test_effect_channel_pair_sensitivity_is_scene_local() -> None:
    target = np.zeros((4, 1, 9, 2, 2), dtype=np.float32)
    prediction = np.zeros_like(target)
    target[1, :, 0] = 1.0
    prediction[1, :, 0] = 0.5
    target[3, :, 0] = 0.5
    prediction[3, :, 0] = 0.25
    rows = [
        {"scene_id": "a", "candidate_i": "a0", "candidate_j": "a1", "pair_type": "effect_divergent"},
        {"scene_id": "b", "candidate_i": "b0", "candidate_j": "b1", "pair_type": "effect_divergent"},
    ]
    result = effect_channel_pair_sensitivity(
        target=target,
        prediction=prediction,
        scene_ids=np.asarray(["a", "a", "b", "b"]),
        candidate_ids=np.asarray(["a0", "a1", "b0", "b1"]),
        pair_rows=rows,
        selected_scene_ids=("a", "b"),
    )
    assert result["a"]["channel_0_target_action_gap"] == 1.0
    assert result["a"]["channel_0_predicted_target_sensitivity_ratio"] == 0.5


def test_primary_channel_shuffle_gap_uses_loss_aligned_direction() -> None:
    target = np.zeros((2, 1, 9, 2, 2), dtype=np.float32)
    target[0, :, 0] = 1.0
    target[0, :, 8] = 1.0
    prediction = target.copy()
    gap = effect_primary_channel_shuffle_gap(
        target=target,
        prediction=prediction,
        scene_ids=np.asarray(["a", "a"]),
        positive_weight=(1.0, 1.0, 1.0),
    )["a"]
    assert gap[0] > 0
    assert gap[8] > 0


def test_gate3_requires_joint_absolute_global_and_structured_evidence() -> None:
    def interval(point: float, low: float, high: float) -> MetricInterval:
        return MetricInterval(point, low, high, 100, 0.95)

    deltas = {
        "aee_vs_absolute_alignment": interval(0.1, 0.02, 0.2),
        "aee_vs_absolute_pair_auprc": interval(0.02, -0.01, 0.05),
        "aee_vs_absolute_safety_auprc": interval(0.03, -0.01, 0.08),
        "aee_vs_absolute_false_safe": interval(-0.04, -0.1, 0.01),
        "aee_vs_absolute_heldout_alignment": interval(0.05, -0.01, 0.1),
        "aee_vs_global_separation_ratio": interval(0.4, 0.1, 0.7),
        "aee_vs_global_structured_error": interval(0.0, -0.01, 0.01),
    }
    passed = gate3_conditions(
        paired_deltas=deltas,
        aee_action_gap=0.95,
        global_action_gap=1.0,
        aee_equivalence_leakage=0.7,
        global_equivalence_leakage=1.0,
        alignment_improves_each_seed=True,
        structured_effect_success=True,
    )
    assert passed["decision"] == "PASS"
    failed = gate3_conditions(
        paired_deltas=deltas,
        aee_action_gap=0.95,
        global_action_gap=1.0,
        aee_equivalence_leakage=0.7,
        global_equivalence_leakage=1.0,
        alignment_improves_each_seed=True,
        structured_effect_success=False,
    )
    assert failed["decision"] == "FAIL"
