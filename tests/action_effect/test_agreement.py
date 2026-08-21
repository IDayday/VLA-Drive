from __future__ import annotations

import numpy as np

from research.action_effect.agreement import binary_agreement_metrics, replay_divergent_pair


def test_agreement_metrics_retain_imbalance_and_ties() -> None:
    replay = np.asarray([False, True, True, False, True])
    reactive = np.asarray([False, True, False, False, True])
    replay_order = np.asarray([0, 1, -1, 0, 1])
    reactive_order = np.asarray([0, 1, 0, 0, 1])
    result = binary_agreement_metrics(replay, reactive, replay_order, reactive_order)
    assert result["pair_count"] == 5
    assert result["disagreement_count"] == 1
    assert result["ranking_disagreement_count"] == 1
    assert result["raw_agreement"] == 0.8
    assert 0.0 < result["positive_agreement"] < 1.0
    assert -1.0 <= result["cohen_kappa"] <= 1.0
    assert -1.0 <= result["mcc"] <= 1.0
    assert -1.0 <= result["tie_aware_kendall"] <= 1.0


def test_replay_divergence_precedes_confidence_relabeling() -> None:
    row = {
        "pair_type": "ambiguous",
        "pair_reason": "traffic_assumption_conflict",
        "geometric_duplicate": False,
        "hard_difference_count": 1,
        "soft_consequence_distance": 0.0,
    }
    assert replay_divergent_pair(row, soft_threshold=1.0)
    row["hard_difference_count"] = 0
    row["soft_consequence_distance"] = 2.0
    assert replay_divergent_pair(row, soft_threshold=1.0)
