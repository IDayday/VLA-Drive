import numpy as np
from scripts.audit_planreg_scorer_head_substitution import substitution_log_score, NC, DAC, TTC, EP, C, DDC


def test_exact_formula_zero_ddc_binary_mapping_and_invalid_ttc():
    p = {n: np.full((1, 3), .7) for n in (NC, DAC, TTC, EP, C, DDC)}
    p[DDC][:] = 0
    truth = {n: v.copy() for n, v in p.items()}
    truth[NC][0] = [0.5, 1, 0]
    truth[TTC][0] = [2, 0, 1]
    original = substitution_log_score(p, truth, ())
    np.testing.assert_allclose(np.exp(original)/12, .7**3)
    replaced = substitution_log_score(p, truth, (NC, TTC))
    assert not np.isnan(replaced).any()
    assert np.isneginf(replaced[0, 0]) and np.isneginf(replaced[0, 2])
    assert replaced.argmax(1).item() == 1
    only_ttc = substitution_log_score(p, truth, (TTC,))
    assert only_ttc[0, 0] == original[0, 0]
