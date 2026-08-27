from __future__ import annotations

import numpy as np
import pytest

from research.cf_effect_gate_wote.src.metrics import pdms_from_factors
from research.cf_effect_gate_wote.src.six_factor_metrics import (
    SIX_FACTOR_ORDER,
    pdms_from_six_factors,
)


def test_ddc_is_an_explicit_multiplicative_factor() -> None:
    factors = np.asarray([1.0, 1.0, 0.5, 1.0, 1.0, 1.0], dtype=np.float32)
    assert SIX_FACTOR_ORDER == ("NC", "DAC", "DDC", "EP", "TTC", "Comfort")
    assert float(pdms_from_six_factors(factors)) == 0.5
    assert float(pdms_from_factors(factors[[0, 1, 3, 4, 5]])) == 1.0


def test_six_factor_formula_supports_candidate_and_scene_batches() -> None:
    candidate = np.asarray([1, 0.8, 0.5, 0.4, 0.9, 0.7], dtype=np.float32)
    candidates = np.stack([candidate, candidate])
    scenes = np.stack([candidates, candidates])
    assert pdms_from_six_factors(candidate).shape == ()
    assert pdms_from_six_factors(candidates).shape == (2,)
    assert pdms_from_six_factors(scenes).shape == (2, 2)
    assert pdms_from_six_factors(scenes).dtype == np.float64


@pytest.mark.parametrize(
    "values",
    [
        np.ones(5, dtype=np.float32),
        np.ones((4, 5), dtype=np.float32),
        np.ones(7, dtype=np.float32),
        np.asarray(np.nan),
    ],
)
def test_six_factor_formula_has_no_five_factor_fallback(values: np.ndarray) -> None:
    with pytest.raises(ValueError, match="exactly 6"):
        pdms_from_six_factors(values)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_six_factor_formula_rejects_nonfinite_values(bad: float) -> None:
    values = np.ones(6, dtype=np.float64)
    values[2] = bad
    with pytest.raises(ValueError, match="NaN or Inf"):
        pdms_from_six_factors(values)
