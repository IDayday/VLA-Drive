from __future__ import annotations

import numpy as np

from research.cf_effect_gate_wote.src.metrics import pdms_from_factors
from research.cf_effect_gate_wote.src.six_factor_metrics import pdms_from_six_factors


def test_first_scene_candidate_87_ddc_regression_fixture() -> None:
    # Minimal numeric fixture from the prior failed audit. It deliberately avoids
    # requiring NAVSIM assets in CI while retaining scene/candidate provenance.
    scene_token = "0fcede1cbfb15faa"
    candidate_index = 87
    factors = np.asarray([1.0, 1.0, 0.5, 0.37775204, 1.0, 1.0], dtype=np.float32)
    evaluator_score = np.float32(0.3703650236129761)
    old_score = float(pdms_from_factors(factors[[0, 1, 3, 4, 5]]))
    new_score = float(pdms_from_six_factors(factors))

    assert scene_token == "0fcede1cbfb15faa"
    assert candidate_index == 87
    assert factors[2] == 0.5
    assert abs(old_score - float(evaluator_score)) > 0.3
    assert abs(new_score - float(evaluator_score)) <= 1e-6
