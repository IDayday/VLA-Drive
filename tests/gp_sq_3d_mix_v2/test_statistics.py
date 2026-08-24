import numpy as np

from starVLA.gp_sq3dmix_statistics import (
    evaluate_stage_a_v2_gates,
    ratio_of_means_bootstrap_ci,
    relative_ratio_of_means,
    residual_distribution,
)
from tools.summarize_gp_sq3dmix_stage_b_v2 import aggregate_seed_paired


def test_utility_uses_ratio_of_means_not_mean_of_ratios():
    base = np.array([1.0, 100.0])
    real = np.array([2.0, 100.0])
    expected = (real.mean() - base.mean()) / base.mean()
    assert relative_ratio_of_means(real, base) == expected
    assert expected != np.mean((real - base) / base)


def test_relative_gap_bootstrap_resamples_paired_ratio_of_means():
    reference = np.array([1.0, 2.0, 4.0, 8.0])
    other = reference + np.array([0.1, 0.2, 0.3, 0.4])
    actual = ratio_of_means_bootstrap_ci(other, reference, seed=11, draws=64)
    rng = np.random.default_rng(11)
    values = []
    for _ in range(64):
        index = rng.integers(0, 4, size=4)
        values.append(
            (other[index].mean() - reference[index].mean())
            / reference[index].mean()
        )
    np.testing.assert_allclose(actual, np.quantile(values, (0.025, 0.975)))


def test_multiseed_aggregation_pairs_the_same_scene_before_bootstrap():
    first = np.array([1.0, 3.0, 5.0])
    second = np.array([3.0, 5.0, 7.0])
    np.testing.assert_array_equal(
        aggregate_seed_paired([first, second]), np.array([2.0, 4.0, 6.0])
    )


def test_noninferiority_uses_utility_ci_upper_and_residual_quantiles():
    base = np.ones(32)
    real = np.ones(32) * 1.004
    hard = real * 1.08
    spatial = real * 1.04
    residual = np.ones((32, 8)) * 0.05
    report = evaluate_stage_a_v2_gates(
        variant="projected_residual",
        base_loss=base,
        real_loss=real,
        hard_loss=hard,
        spatial_loss=spatial,
        residual_per_horizon=residual,
        slot_mean_identity_max_abs=0.0,
        adapter_grad_norm=1.0,
        reader_grad_norm=1.0,
        gate_grad_norm=None,
        all_named_losses_finite=True,
        alpha=0.10,
        retention_near_lower_fraction=None,
        retention_near_upper_fraction=None,
        seed=3,
        draws=64,
    )
    assert report["checks"]["utility_point_estimate"]
    assert report["checks"]["utility_non_inferiority_ci"]
    assert "retention_non_collapse" not in report["checks"]
    distribution = residual_distribution(
        np.vstack([np.ones((99, 8)) * 0.01, np.ones((1, 8)) * 0.5])
    )
    assert distribution["p95"] < distribution["max"]
    assert len(distribution["per_horizon"]) == 8


def test_gated_variant_requires_gate_gradient_and_retention_noncollapse():
    report = evaluate_stage_a_v2_gates(
        variant="gated_residual",
        base_loss=np.ones(16),
        real_loss=np.ones(16),
        hard_loss=np.ones(16) * 1.1,
        spatial_loss=np.ones(16) * 1.05,
        residual_per_horizon=np.ones((16, 8)) * 0.05,
        slot_mean_identity_max_abs=0.0,
        adapter_grad_norm=1.0,
        reader_grad_norm=1.0,
        gate_grad_norm=0.0,
        all_named_losses_finite=True,
        alpha=0.1,
        retention_near_lower_fraction=0.1,
        retention_near_upper_fraction=0.1,
        seed=4,
        draws=32,
    )
    assert report["checks"]["retention_non_collapse"] is False
