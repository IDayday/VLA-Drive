"""Behavioral reporting checks; synthetic labels are not real evaluation evidence."""
import numpy as np
import pytest

from iqe.internal_baseline.evaluation import (
    argmax_base_tie, bank_statistics, paired_bootstrap, selection_metrics,
)


def metrics(scores, choices, components=None, groups=None):
    scores = np.asarray(scores, dtype=np.float64)
    if components is None:
        components = np.ones((len(scores), 2, 6), dtype=np.float64)
    if groups is None:
        groups = [f"log_{i}" for i in range(len(scores))]
    return selection_metrics(scores, components, np.asarray(choices), groups,
                             bootstrap_repetitions=41, bootstrap_seed=937)


def test_complementarity_four_identities_and_actual_regret():
    m = metrics([[.2, .9], [.9, .1], [.5, .5], [.8, .805]], [1, 1, 1, 0])
    c = m["complementarity"]
    assert c["A_positive_expert_gain"] == pytest.approx(.705/4)
    assert c["B_negative_expert_cost"] == pytest.approx(.8/4)
    assert c["R_selection_regret"] == pytest.approx(.805/4)
    assert c["min_A_B"] == pytest.approx(.705/4)
    assert c["denominator"] == m["n"] == m["requested_scenes"] == m["joint_valid_scenes"] == 4
    assert c["identities_pass"]
    assert len(c["identities"]) == 4
    for identity in c["identities"].values():
        assert identity["lhs"] == pytest.approx(identity["rhs"], abs=1e-12)
    assert m["selected_gain_vs_best_fixed_policy"] == pytest.approx(-.025)
    assert m["oracle_capture"] < 0  # No clipped or beautified capture.


def test_positive_gain_vs_base_does_not_hide_better_expert_alone():
    m = metrics([[.1, .9], [.5, .8]], [0, 1])
    assert m["selected_gain"] > 0
    assert m["best_fixed_policy"] == "expert_1"
    assert m["selected_gain_vs_best_fixed_policy"] == pytest.approx(-.4)
    assert m["selected_gain_vs_expert"] == pytest.approx(-.4)
    assert m["oracle_gain_vs_best_fixed_policy"] == 0
    assert m["complementarity"]["min_A_B"] == 0


def test_switch_ties_and_every_precision_denominator_are_explicit():
    m = metrics([[.2, .9], [.9, .1], [.5, .5], [.8, .805]], [1, 1, 1, 0])
    assert m["switch_count"] == 3
    assert m["switch_precision_numerator"] == 1
    assert m["switch_precision_denominator"] == 3
    assert m["switch_precision"] == pytest.approx(1/3)
    assert m["switch_ties"] == 1
    assert m["switch_precision_excluding_ties_denominator"] == 2
    assert m["switch_precision_excluding_ties"] == .5
    assert m["switch_tie_rate"] == pytest.approx(1/3)
    assert m["false_replacement_count"] == 1
    assert m["false_replacement_rate_switched"] == pytest.approx(1/3)
    assert m["positive_switch_gain_total"] == pytest.approx(.7)
    assert m["negative_switch_cost_total"] == pytest.approx(.8)
    assert m["positive_switch_gain_per_scene"]-m["negative_switch_cost_per_scene"] == pytest.approx(m["selected_gain"])


def test_epsilon_optimal_and_large_gap_cost_have_different_denominators():
    m = metrics([[.2, .9], [.9, .1], [.5, .5], [.8, .805]], [1, 1, 1, 0])
    assert m["epsilon_optimal"] == {"epsilon": .01, "numerical_tolerance": 1e-12,
                                      "count": 3, "denominator": 4, "rate": .75}
    for threshold in ("0.1", "0.5"):
        d = m["large_gap_errors"][threshold]
        assert d["eligible"] == 2 and d["errors"] == 1
        assert d["error_rate"] == .5 and d["error_rate_all_scenes"] == .25
        assert d["regret_total"] == pytest.approx(.8)
    # Losing 0.01 is still epsilon-optimal, up to declared FP64 comparison tolerance.
    boundary = metrics([[.8, .81]], [0])
    assert boundary["epsilon_optimal"]["count"] == 1


def test_no_switch_and_no_oracle_gain_return_na_not_fabricated_rates():
    m = metrics([[.7, .7], [.8, .8]], [0, 0])
    assert m["best_fixed_policy"] == "expert_0"
    assert m["oracle_capture"] is None
    assert m["switch_precision"] is None
    assert m["switch_precision_excluding_ties"] is None
    assert m["switch_tie_rate"] is None
    assert m["large_gap_errors"]["0.1"]["error_rate"] is None
    assert m["epsilon_optimal"]["rate"] == 1
    assert argmax_base_tie([[.7, .7], [.2, .3]]).tolist() == [0, 1]


def test_named_safety_components_handle_missing_and_full_score_tolerance():
    components = np.ones((3, 2, 6))
    components[:, :, 0] = [[1, np.nan], [np.nan, 0], [1-5e-9, 0]]
    components[:, :, 1] = [[.5, 1], [1, 0], [1, 1]]
    m = metrics([[1, 0], [1, .9], [.2, .2]], [1, 1, 1], components)
    nc, dac = m["safety"]["NC"], m["safety"]["DAC"]
    assert nc["component_name"] == "no_at_fault_collisions"
    assert nc["denominator"] == 1 and nc["requested_scenes"] == 3
    assert nc["selected_valid_denominator"] == 2 and nc["base_valid_denominator"] == 2
    assert nc["selected_violations"] == 2 and nc["base_violations"] == 0
    assert nc["new_violations"] == 1 and nc["missing_selected"] == nc["missing_base"] == 1
    assert nc["new_violation_rate_all_valid_pairs"] == 1.
    assert dac["component_name"] == "drivable_area_compliance"
    assert dac["new_violations"] == dac["resolved_base_violations"] == 1
    assert dac["denominator"] == 3
    assert m["high_to_zero"] == 1
    assert m["high_to_zero_rate_all_scenes"] == pytest.approx(1/3)
    assert m["high_to_zero_rate_base_high"] == .5
    assert m["base_high_count"] == 2


def test_bank_comparison_weights_report_cost_concentration_without_squaring():
    scores = np.array([[0., .95], [.5, .52], [.8, .8]])
    c = np.ones((3, 2, 6)); c[2, :, 3] = np.nan
    b = bank_statistics(scores, c, ["large", "small", "tie"])
    assert b["valid_pairs"] == b["valid_pair_groups"] == 2
    assert b["comparison_total_gap_weight"] == pytest.approx(.97)
    ratio = b["per_log"]["large"]["comparison_probability_mass"]/b["per_log"]["small"]["comparison_probability_mass"]
    assert ratio == pytest.approx(.95/.02)
    assert b["comparison_effective_scene_count"] == pytest.approx(.97**2/(.95**2+.02**2))
    assert b["comparison_max_probability"] == pytest.approx(.95/.97)
    assert b["component_joint_valid_denominators"][3] == 2
    assert b["component_disagreements"][3] == 0  # NaN is not an actual disagreement.
    assert b["expert_high_base_low"] == 1


def test_bank_without_comparison_signal_is_explicit():
    b = bank_statistics(np.array([[.5, .5]]), np.ones((1, 2, 6)), ["log"])
    assert b["comparison_signal"] == "SCORER_COMPARISON_SIGNAL_ABSENT"
    assert b["comparison_effective_scene_count"] is None
    assert b["comparison_max_probability"] is None
    assert b["comparison_total_gap_weight"] == 0
    assert b["per_log"]["log"]["comparison_probability_mass"] == 0


def test_bootstrap_uses_declared_seed_and_whole_log_groups():
    # Both scenes in each group move together; no scene-wise resampling.
    a = paired_bootstrap([1., 1., -1.], ["one", "one", "two"], repetitions=73, seed=987)
    b = paired_bootstrap([1., 1., -1.], ["one", "one", "two"], repetitions=73, seed=987)
    assert a == b
    assert a["mean"] == pytest.approx(1/3)
    assert a["groups"] == 2 and not a["reliable_log_count"]
    assert a["seed"] == 987 and a["repetitions"] == 73
    assert a["lower_95"] == -1 and a["upper_95"] == 1


@pytest.mark.parametrize("scores", [np.array([[np.nan, .5]]), np.array([[np.inf, .5]]), np.empty((0, 2))])
def test_incomplete_bank_does_not_become_valid_subset_success(scores):
    with pytest.raises(ValueError, match="complete finite candidate"):
        metrics(scores, np.zeros(len(scores), dtype=np.int64))


@pytest.mark.parametrize("scores", [[[-.1, .5]], [[.5, 100.]]])
def test_score_units_are_not_guessed(scores):
    with pytest.raises(ValueError, match="score scale"):
        metrics(scores, [0])


@pytest.mark.parametrize("choices", [[2], [-1], [.0], [[0]]])
def test_invalid_candidate_identity_or_shape_is_rejected(choices):
    with pytest.raises(ValueError, match="selected indices"):
        metrics([[.3, .8]], choices)


def test_bad_component_labels_and_group_shape_are_rejected():
    with pytest.raises(ValueError, match="component scale"):
        metrics([[.3, .8]], [1], np.full((1, 2, 6), 100.))
    with pytest.raises(ValueError, match="component schema"):
        metrics([[.3, .8]], [1], np.ones((1, 2, 1)))
    with pytest.raises(ValueError, match="one source log"):
        metrics([[.3, .8]], [1], groups=[])
    with pytest.raises(ValueError, match="positive bootstrap repetitions"):
        paired_bootstrap([0.], ["log"], repetitions=0)
