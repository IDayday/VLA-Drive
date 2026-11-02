"""Directed cold-start/PG0/strict-state checks; no score/performance claims."""
import copy
import io

import pytest
import torch

from starVLA.rl.qwen_world_verified_quality import calibration as C
from starVLA.rl.qwen_world_safety_exploration.retention_material import digest


def actual_components(model, pg=1., IL=0.):
    p = next(model.parameters())
    return pg * p.sum(), IL * p.sum(), 2 * p.sum(), 4 * p.sum()


def model():
    return torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)


def test_actual_gradient_target_floor_and_original_beta_alpha_budgets():
    m = model(); s = C.new_calibration_state(); rng = torch.get_rng_state().clone()
    pg, IL, cfm, ode = actual_components(m, pg=.04, IL=8e-7)
    out, metrics = C.calibrate_components(m, pg, IL, cfm, ode, s)
    assert s['PG_gradient_EMA'] == pytest.approx(.04)
    assert s['last_target_norm'] == pytest.approx(.0016)
    assert s['scale_cfm'] == pytest.approx(.0008) and s['scale_ode'] == pytest.approx(.0004)
    # Same-direction fixture gives the exact target norm before the original
    # beta50, conditional alpha and CFM/ODE .75/.25 weights are applied.
    rescue_norm = torch.autograd.grad(out['rescue'], list(m.parameters()))[0].norm().item()
    assert rescue_norm == pytest.approx(.04 * .04)
    assert 50 * .05 * rescue_norm / .04 == pytest.approx(.1)
    assert 50 * .02 * rescue_norm / .04 == pytest.approx(.04)
    assert all(p.grad is None for p in m.parameters()) and torch.equal(rng, torch.get_rng_state())
    assert float(metrics['nominal_calibration_target_norm']) == pytest.approx(.0016)


def test_true_cold_start_does_not_exhaust_or_freeze_budget_then_PG0_uses_history():
    m = model(); s = C.new_calibration_state()
    for _ in range(105):
        C.calibrate_components(m, *actual_components(m, pg=0., IL=0.), s)
    assert s['effective_updates'] == s['cold_start_skips'] == s['gradient_probe_calls'] == 105
    assert s['calibration_updates'] == s['eligible_updates'] == s['measured_updates'] == 0
    assert not s['frozen'] and s['scale_cfm'] == s['scale_ode'] == 1e-4 and s['PG_gradient_EMA'] is None
    C.calibrate_components(m, *actual_components(m, pg=1., IL=0.), s)
    assert s['PG_gradient_EMA'] == 1. and s['calibration_updates'] == 1
    out, _ = C.calibrate_components(m, *actual_components(m, pg=0., IL=0.), s)
    assert s['PG_gradient_EMA'] == .9 and s['last_target_norm'] == pytest.approx(.036)
    assert s['calibration_updates'] == 2
    assert torch.autograd.grad(out['rescue'], list(m.parameters()))[0].abs().sum() > 0


def test_pending_state_roundtrip_is_bitexact_and_rejects_missing_EMA_or_fake_freeze():
    m = model(); s = C.new_calibration_state()
    for pg in (0., .1, .2):
        C.calibrate_components(m, *actual_components(m, pg=pg, IL=0.), s)
    buffer = io.BytesIO(); torch.save(s, buffer); buffer.seek(0)
    right = torch.load(buffer, weights_only=False)
    assert digest(s) == digest(right)
    for pg in (0., .3, .1):
        left_loss, lm = C.calibrate_components(m, *actual_components(m, pg=pg, IL=0.), s)
        right_loss, rm = C.calibrate_components(m, *actual_components(m, pg=pg, IL=0.), right)
        assert digest(s) == digest(right) and digest(lm) == digest(rm)
        assert all(torch.equal(left_loss[k], right_loss[k]) for k in left_loss)
    for key, value in [('PG_gradient_EMA', None), ('PG_EMA_updates', 0), ('frozen', True), ('last_target_norm', 0.)]:
        bad = copy.deepcopy(s); bad[key] = value
        with pytest.raises(ValueError): C.validate_calibration_state(bad)
    bad = copy.deepcopy(s); del bad['PG_gradient_EMA']
    with pytest.raises(ValueError): C.validate_calibration_state(bad)


def test_scales_freeze_after_100_eligible_calls_and_measurements_stop_at500():
    m = model(); s = C.new_calibration_state()
    for _ in range(100):
        C.calibrate_components(m, *actual_components(m, pg=1., IL=0.), s)
    assert s['frozen'] and s['calibration_updates'] == 100
    frozen = (s['scale_cfm'], s['scale_ode'])
    for _ in range(400):
        C.calibrate_components(m, *actual_components(m, pg=2., IL=0.), s)
    assert s['measured_updates'] == s['PG_EMA_updates'] == s['gradient_probe_calls'] == 500
    assert (s['scale_cfm'], s['scale_ode']) == frozen
    measured = (copy.deepcopy(s['last_norms']), s['PG_gradient_EMA'])
    _, metrics = C.calibrate_components(m, *actual_components(m, pg=100., IL=0.), s)
    assert not float(metrics['nominal_gradient_norm_measured'])
    assert s['eligible_updates'] == s['effective_updates'] == 501
    assert measured == (s['last_norms'], s['PG_gradient_EMA'])
