import numpy as np

from tools.lora_value_audit.candidate_metrics import (
    aggregate_predicted_score,
    geometry_metrics,
    scene_metrics,
)
from tools.lora_value_audit.schema import VerdictInputs, choose_verdict


def test_topk_and_gap_identities() -> None:
    true = np.linspace(0.0, 1.0, 64)
    predicted = true[::-1]
    proposals = np.zeros((64, 8, 3), dtype=np.float32)
    proposals[:, :, 0] = np.arange(64)[:, None]
    result = scene_metrics(true, predicted, proposals)
    assert result["top1_oracle"] == result["V_B"]
    assert result["top64_oracle"] == result["O_B"]
    assert np.isclose((1 - result["O_B"]) + result["G_B"], 1 - result["V_B"])


def test_aggregate_predicted_score_matches_formula() -> None:
    logits = np.zeros((2, 6), dtype=np.float32)
    weights = {"noc": 1, "dac": 1, "ddc": 0, "ttc": 5, "ep": 5, "comfort": 2}
    score = aggregate_predicted_score(logits, weights)
    expected = 2 * np.log(0.5) + np.log(6.0)
    np.testing.assert_allclose(score, expected)


def test_geometry_cluster_count() -> None:
    proposals = np.zeros((3, 8, 3), dtype=np.float32)
    proposals[1, :, 1] = 0.1
    proposals[2, :, 1] = 2.0
    values = geometry_metrics(proposals, np.array([True, True, True]))
    assert values["trajectory_cluster_count"] == 2
    assert values["high_quality_cluster_count"] == 2


def test_infra_invalid_has_priority() -> None:
    values = VerdictInputs(
        infrastructure_valid=False,
        coverage_gap=0.1,
        ranking_gap=0.1,
        ideal1_oracle_gain=0.1,
        ideal1_selected_gain=0.1,
        ideal8_oracle_gain=0.1,
        ideal16_oracle_gain=0.1,
        ideal8_selected_gain=0.1,
        ideal16_selected_gain=0.1,
        ideal8_oracle_ci_low=0.1,
        ideal8_selected_ci_low=0.1,
        fixed_budget_gain=0.1,
        duplicate_shift=0.0,
        candidate_limited_share_low=1.0,
        ranker_limited_share_low=0.0,
        target_available_rate=1.0,
        saturated_false_replacement_rate=0.0,
        practical_selected_gain=0.1,
    )
    assert choose_verdict(values)["verdict"] == "INFRA_INVALID"
