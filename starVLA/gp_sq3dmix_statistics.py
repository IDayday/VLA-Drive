"""Paired Stage-A-v2 statistics and immutable gate definitions."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np


def _vector(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a non-empty finite vector")
    return array


def relative_ratio_of_means(other, reference) -> float:
    other_array = _vector(other, "other")
    reference_array = _vector(reference, "reference")
    if other_array.shape != reference_array.shape:
        raise ValueError("paired arrays must have identical shapes")
    denominator = float(reference_array.mean())
    if abs(denominator) < 1e-12:
        raise ValueError("ratio-of-means reference mean is zero")
    return float((other_array.mean() - denominator) / denominator)


def absolute_mean_gap(other, reference) -> float:
    other_array = _vector(other, "other")
    reference_array = _vector(reference, "reference")
    if other_array.shape != reference_array.shape:
        raise ValueError("paired arrays must have identical shapes")
    return float((other_array - reference_array).mean())


def paired_bootstrap_ci(
    arrays: Mapping[str, np.ndarray],
    statistic: Callable[[Mapping[str, np.ndarray]], float],
    *,
    seed: int,
    draws: int = 10000,
    chunk_size: int = 128,
) -> list[float]:
    if draws < 1 or chunk_size < 1:
        raise ValueError("bootstrap draws/chunk_size must be positive")
    normalized = {name: _vector(value, name) for name, value in arrays.items()}
    lengths = {len(value) for value in normalized.values()}
    if len(lengths) != 1:
        raise ValueError("paired bootstrap arrays must have identical lengths")
    sample_count = next(iter(lengths))
    generator = np.random.default_rng(int(seed))
    statistics = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, chunk_size):
        size = min(chunk_size, draws - start)
        indices = generator.integers(
            0, sample_count, size=(size, sample_count), endpoint=False
        )
        for offset, row in enumerate(indices):
            statistics[start + offset] = statistic(
                {name: value[row] for name, value in normalized.items()}
            )
    return [float(value) for value in np.quantile(statistics, (0.025, 0.975))]


def ratio_of_means_bootstrap_ci(
    other, reference, *, seed: int, draws: int = 10000
) -> list[float]:
    return paired_bootstrap_ci(
        {"other": np.asarray(other), "reference": np.asarray(reference)},
        lambda sample: relative_ratio_of_means(
            sample["other"], sample["reference"]
        ),
        seed=seed,
        draws=draws,
    )


def absolute_gap_bootstrap_ci(
    other, reference, *, seed: int, draws: int = 10000
) -> list[float]:
    return paired_bootstrap_ci(
        {"other": np.asarray(other), "reference": np.asarray(reference)},
        lambda sample: absolute_mean_gap(
            sample["other"], sample["reference"]
        ),
        seed=seed,
        draws=draws,
    )


def residual_distribution(residual_per_horizon) -> dict:
    values = np.asarray(residual_per_horizon, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 8 or not np.isfinite(values).all():
        raise ValueError("residual_per_horizon must be finite [N,8]")
    per_sample = values.mean(axis=1)
    quantiles = np.quantile(per_sample, (0.50, 0.90, 0.95, 0.99))
    horizons = []
    for horizon in range(8):
        horizons.append(
            {
                "horizon": horizon,
                "mean": float(values[:, horizon].mean()),
                "p95": float(np.quantile(values[:, horizon], 0.95)),
            }
        )
    return {
        "mean": float(per_sample.mean()),
        "p50": float(quantiles[0]),
        "p90": float(quantiles[1]),
        "p95": float(quantiles[2]),
        "p99": float(quantiles[3]),
        "max": float(per_sample.max()),
        "per_horizon": horizons,
    }

def evaluate_stage_a_v2_gates(
    *,
    variant: str,
    base_loss,
    real_loss,
    hard_loss,
    spatial_loss,
    residual_per_horizon,
    slot_mean_identity_max_abs: float,
    adapter_grad_norm: float,
    reader_grad_norm: float,
    gate_grad_norm: float | None,
    all_named_losses_finite: bool,
    alpha: float,
    retention_near_lower_fraction: float | None,
    retention_near_upper_fraction: float | None,
    seed: int,
    draws: int = 10000,
) -> dict:
    if variant not in {"projected_residual", "gated_residual"}:
        raise ValueError("unknown Stage-A-v2 variant")
    base = _vector(base_loss, "base_loss")
    real = _vector(real_loss, "real_loss")
    hard = _vector(hard_loss, "hard_loss")
    spatial = _vector(spatial_loss, "spatial_loss")
    if len({len(base), len(real), len(hard), len(spatial)}) != 1:
        raise ValueError("Stage-A losses must be paired")

    utility = relative_ratio_of_means(real, base)
    utility_ci = ratio_of_means_bootstrap_ci(
        real, base, seed=seed, draws=draws
    )
    hard_relative = relative_ratio_of_means(hard, real)
    hard_relative_ci = ratio_of_means_bootstrap_ci(
        hard, real, seed=seed + 1, draws=draws
    )
    hard_absolute = absolute_mean_gap(hard, real)
    hard_absolute_ci = absolute_gap_bootstrap_ci(
        hard, real, seed=seed + 2, draws=draws
    )
    spatial_relative = relative_ratio_of_means(spatial, real)
    spatial_relative_ci = ratio_of_means_bootstrap_ci(
        spatial, real, seed=seed + 3, draws=draws
    )
    spatial_absolute = absolute_mean_gap(spatial, real)
    spatial_absolute_ci = absolute_gap_bootstrap_ci(
        spatial, real, seed=seed + 4, draws=draws
    )
    residual = residual_distribution(residual_per_horizon)

    checks = {
        "slot_mean_identity": float(slot_mean_identity_max_abs) < 1e-6,
        "utility_point_estimate": utility <= 0.005,
        "utility_non_inferiority_ci": utility_ci[1] <= 0.02,
        "hard_causal_relative_gap": hard_relative > 0.05
        and hard_relative_ci[0] > 0.0,
        "hard_causal_absolute_gap": hard_absolute_ci[0] > 0.0,
        "spatial_causal_relative_gap": spatial_relative > 0.02
        and spatial_relative_ci[0] > 0.0,
        "spatial_causal_absolute_gap": spatial_absolute_ci[0] > 0.0,
        "residual_mean": 0.01 <= residual["mean"] <= 0.15,
        "residual_p95": residual["p95"] <= 0.25,
        "residual_p99": residual["p99"] <= 0.40,
        "residual_max": residual["max"] <= 0.50,
        "per_horizon_residual_bound": all(
            value["mean"] <= 0.20 for value in residual["per_horizon"]
        ),
        "geometry_route_active": float(adapter_grad_norm) > 0.0
        and float(reader_grad_norm) > 0.0,
        "finite_named_losses": bool(all_named_losses_finite),
        "alpha_bound": 0.05 <= float(alpha) <= 0.20,
    }
    if variant == "gated_residual":
        checks["retention_non_collapse"] = (
            gate_grad_norm is not None
            and float(gate_grad_norm) > 0.0
            and retention_near_lower_fraction is not None
            and retention_near_upper_fraction is not None
            and float(retention_near_lower_fraction) < 0.80
            and float(retention_near_upper_fraction) < 0.80
        )
    return {
        "relative_real_minus_base": utility,
        "utility_bootstrap_ci": utility_ci,
        "relative_hard_real_gap": hard_relative,
        "relative_hard_real_gap_bootstrap_ci": hard_relative_ci,
        "absolute_hard_real_gap": hard_absolute,
        "absolute_hard_real_gap_bootstrap_ci": hard_absolute_ci,
        "relative_spatial_real_gap": spatial_relative,
        "relative_spatial_real_gap_bootstrap_ci": spatial_relative_ci,
        "absolute_spatial_real_gap": spatial_absolute,
        "absolute_spatial_real_gap_bootstrap_ci": spatial_absolute_ci,
        "residual_distribution": residual,
        "checks": checks,
        "all_passed": all(checks.values()),
    }
