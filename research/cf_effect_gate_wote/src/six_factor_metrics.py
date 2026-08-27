"""Explicit six-factor NAVSIM score contract for independent WoTE labels."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


SIX_FACTOR_ORDER = (
    "NC",
    "DAC",
    "DDC",
    "EP",
    "TTC",
    "Comfort",
)


def pdms_from_six_factors(factors: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Reassemble PDMS from ``[NC, DAC, DDC, EP, TTC, Comfort]``.

    The final dimension is deliberately strict: historical five-factor arrays
    are never accepted and DDC is never synthesized or folded into DAC.
    """

    values = np.asarray(factors, dtype=np.float64)
    if values.ndim == 0 or values.shape[-1] != len(SIX_FACTOR_ORDER):
        shape = values.shape
        raise ValueError(
            "six-factor PDMS expects a final dimension of exactly 6 "
            f"in order {SIX_FACTOR_ORDER}, got shape {shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("six-factor PDMS input contains NaN or Inf")

    nc, dac, ddc, ep, ttc, comfort = np.moveaxis(values, -1, 0)
    score = nc * dac * ddc * ((5.0 * ep + 5.0 * ttc + 2.0 * comfort) / 12.0)
    return np.asarray(score, dtype=np.float64)
