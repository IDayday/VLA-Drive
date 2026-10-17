import numpy as np
import pytest

from scripts.analysis.fm_transfer.analyze_A_stage_composition import compose, pass_history


def row(ep, ttc=1., nc=1., dac=1.):
    return [nc*dac*(5*ep+5*ttc+2)/12, nc, dac, ttc, ep, 1., 1.]


def test_reversals_require_recomputed_total_gain_and_paired_interval():
    initial = np.array([row(.5), row(.5), row(.5), row(.5)])
    sft = np.array([row(.8), row(.3), row(.5), row(.5)])
    rl = np.array([row(.6), row(.7), row(.5), row(.5)])
    value = compose(initial, sft, rl, ['a', 'a', 'b', 'b'])
    stages = value['stages']
    assert stages['RL_minus_initial_composed']['mean_delta']['score'] == pytest.approx(
        stages['SFT_minus_initial']['mean_delta']['score'] + stages['RL_minus_own_SFT']['mean_delta']['score'])
    assert stages['RL_minus_initial_composed']['G_minus_raw'] == 0
    assert stages['SFT_minus_initial']['G_minus_raw'] > 0
    assert stages['RL_minus_own_SFT']['G_minus_raw'] > 0
    assert stages['RL_minus_initial_composed']['G_plus_raw'] < (
        stages['SFT_minus_initial']['G_plus_raw'] + stages['RL_minus_own_SFT']['G_plus_raw'])
    assert value['score_history_counts']['SFT_improved_then_RL_degraded_ge005'] == 1
    assert value['score_history_counts']['RL_improved_ge005_after_nonimproving_SFT'] == 1


def test_three_stage_history_separates_sft_failures_rl_failures_and_repair_loss():
    bits = np.array([[int(x) for x in f'{n:03b}'] for n in range(8)], dtype=bool)
    counts = pass_history(*bits.T)
    assert set(counts) == {f'{n:03b}' for n in range(8)}
    assert all(n == 1 for n in counts.values())
    assert sum(counts.values()) == 8


def test_fractional_nc_is_not_fully_safe_and_zero_score_is_separate():
    initial = np.array([row(.4), row(.4, nc=.5), row(.4, dac=0)])
    sft = np.array([row(.5), row(.5, nc=.5), row(.5)])
    rl = np.array([row(.6, nc=.5), row(.6), row(.6, dac=0)])
    result = compose(initial, sft, rl, ['a', 'b', 'c'])
    joint = result['safety_history_counts']['joint_NC_DAC_TTC']
    assert joint['110'] == joint['001'] == joint['010'] == 1
    assert result['stages']['RL_minus_own_SFT']['nonzero_to_zero'] == 1
    assert result['stages']['SFT_minus_initial']['zero_to_nonzero'] == 1


@pytest.mark.parametrize('fault', ['nan', 'shape', 'outside', 'score_formula'])
def test_invalid_metrics_cannot_disappear_from_denominator(fault):
    a = np.array([row(.5), row(.6)])
    b = a.copy()
    if fault == 'nan': b[0, 0] = np.nan
    if fault == 'shape': b = b[:1]
    if fault == 'outside': b[0, 3] = 2.
    if fault == 'score_formula': b[0, 0] = 0.
    with pytest.raises(ValueError):
        compose(a, b, a, ['a', 'b'])
