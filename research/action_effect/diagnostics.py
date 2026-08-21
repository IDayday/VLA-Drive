"""Shared analysis helpers for data feasibility and action-collapse reports."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import math
from pathlib import Path
from typing import Any

import numpy as np


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """Render a compact GitHub Markdown table."""

    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def percentage(numerator: float, denominator: float, digits: int = 1) -> str:
    """Format a fraction as a percentage without hiding an empty denominator."""

    return "n/a" if denominator == 0 else f"{100.0 * numerator / denominator:.{digits}f}%"


def bootstrap_interval(
    values: Sequence[float] | np.ndarray,
    statistic: str = "mean",
    *,
    confidence: float = 0.95,
    samples: int = 2000,
    seed: int = 20260821,
) -> tuple[float, float, float]:
    """Return point estimate and percentile bootstrap confidence interval."""

    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return float("nan"), float("nan"), float("nan")
    reducer = np.mean if statistic == "mean" else np.median
    point = float(reducer(array))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    bootstrapped = reducer(array[indices], axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(bootstrapped, [alpha, 1.0 - alpha])
    return point, float(low), float(high)


def finite_mean(values: Iterable[float | None]) -> float:
    """Mean over finite, non-null values."""

    array = np.asarray([float(value) for value in values if value is not None], dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if len(array) else float("nan")


def require_target_provenance(rows: Sequence[Mapping[str, Any]]) -> None:
    """Fail diagnostics early if a cache lost consequence provenance."""

    expected = {"exact": "exact", "log_replay": "log_replay", "reactive_model": "reactive_model"}
    for row in rows:
        for namespace, provenance in expected.items():
            if row[namespace].get("provenance") != provenance:
                raise AssertionError(f"{row.get('candidate_id')} has invalid {namespace} provenance")


def save_figure(fig: Any, path: Path) -> None:
    """Save a deterministic, tightly cropped PNG and close the figure."""

    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", metadata={"Software": "DriveDreamer-Policy"})
    plt.close(fig)
