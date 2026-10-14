import numpy as np
import pytest

from iqe.geometric_readout.visibility_risk import selection_accounting


def test_error_cost_uses_full_population_denominator():
    bank = {'scores': np.array([[1., 0.], [.8, .9], [.6, .6], [1., .5]]),
            'valid': np.ones((4, 2), dtype=bool)}
    predicted = np.array([[0., 1.]] * 4)
    coverage = np.array([[1., 0.], [1., .25], [1., .5], [1., 1.]])
    meta = {'provenance': {'sources': ['expert_0', 'expert_1']}}
    result = selection_accounting(predicted, bank, meta, coverage)
    assert result['gain_01'] == pytest.approx(-.35)
    assert result['negative_cost_01'] == pytest.approx(.375)
    assert result['positive_gain_01'] == pytest.approx(.025)
    assert result['high_to_zero'] == 1
    assert result['wrong_replacements'] == 2
    assert [row['scenes'] for row in result['rows']] == [1, 1, 1, 1]
    assert [row['negative_cost_population_01'] for row in result['rows']] == [.25, 0., 0., .125]
    assert sum(row['negative_cost_share'] for row in result['rows']) == pytest.approx(1.)


def test_no_error_or_empty_coverage_group_is_not_a_nan_or_false_rate():
    bank = {'scores': np.array([[.8, .9]]), 'valid': np.ones((1, 2), dtype=bool)}
    result = selection_accounting(np.array([[.2, .9]]), bank,
                                 {'provenance': {'sources': ['expert_0', 'expert_1']}},
                                 np.ones((1, 2)))
    assert result['negative_cost_01'] == 0.
    assert all(row['negative_cost_share'] is None for row in result['rows'])
    assert all(row['mean_selected_01'] is None for row in result['rows'][:-1])
    assert result['rows'][-1]['mean_selected_01'] == .9
