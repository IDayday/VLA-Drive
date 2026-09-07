"""Audit definitions, not a new model/regularization target."""
import numpy as np
import pytest

from scripts.analyze_task_future_lite_regression import token_statistics, distribution


def test_two_repeated_groups_have_rank_one_despite_high_total_variance():
    rng = np.random.default_rng(7)
    prototypes = rng.normal(size=(5, 2, 16))
    tokens = np.repeat(prototypes, 8, axis=1)
    full = token_statistics(tokens)
    assert full['slot_centered_rms'] > .1
    assert full['centered_energy_effective_rank']['mean'] == pytest.approx(1., abs=1e-10)
    for group in (tokens[:, :8], tokens[:, 8:]):
        within = token_statistics(group)
        assert within['slot_centered_rms'] < 1e-14
        assert within['mean_pairwise_cosine'] == pytest.approx(1.)


def test_fixed_slot_identity_does_not_pass_cross_scene_slot_content_test():
    rng = np.random.default_rng(8)
    identity = rng.normal(size=(1, 16, 20))
    content = rng.normal(size=(5, 1, 20))
    result = token_statistics(identity + content)
    assert result['slot_centered_rms'] > .1
    assert result['cross_scene_content_rms'] > .1
    assert result['cross_scene_slot_content_rms'] < 1e-14


def test_centered_rank_is_scale_invariant_but_amplitude_is_not():
    x = np.random.default_rng(9).normal(size=(6, 16, 24))
    a, b = token_statistics(x), token_statistics(x * .001)
    assert a['centered_energy_effective_rank']['mean'] == pytest.approx(b['centered_energy_effective_rank']['mean'])
    assert b['slot_centered_rms'] == pytest.approx(a['slot_centered_rms'] * .001)


def test_nonfinite_diagnostic_is_not_silently_accepted():
    with pytest.raises(ValueError, match='Nonfinite'):
        distribution([0., float('nan')])
