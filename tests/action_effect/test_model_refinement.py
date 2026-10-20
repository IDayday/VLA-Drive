from dataclasses import replace
import numpy as np
import pytest
import torch
from torch import nn

from starVLA.model.modules.action_effect.effect_model import ActionEffectModel
from starVLA.model.modules.action_effect.intervention_basis import BasisSpec, basis_matrix, reconstruct
from starVLA.model.modules.action_effect.refinement_policy import (
    CoefficientRefiner, RefinerConfig, reconstruct_torch, torch_basis_matrix,
)
from starVLA.model.framework.effect_guided_refiner import EffectGuidedRefiner
from test_model_effect import cfg, spec, batch, state_hash


torch.set_num_threads(2)


def refiner_config():
    return RefinerConfig(model=cfg())


def test_torch_reconstruction_matches_canonical_numpy_and_support():
    _, a, _ = batch(3); a = a.double()
    a[1, :, 2] = torch.linspace(3.10, 3.16, 8)
    coefficients = torch.tensor([[.3, -.1, .2, 0., -.15, 0., .2, .1], [0., 0., 0., 0., .1, 0., 0., 0.], [0.] * 8], dtype=torch.float64)
    out = reconstruct_torch(a, coefficients, BasisSpec())
    for i in range(3):
        np.testing.assert_allclose(torch_basis_matrix(a, BasisSpec())[i].numpy(), basis_matrix(a[i].numpy(), BasisSpec()), rtol=0, atol=1e-15)
        np.testing.assert_allclose(out[i].numpy(), reconstruct(a[i].numpy(), coefficients[i].numpy(), BasisSpec()), rtol=0, atol=3e-15)
    late = torch.zeros(3, 8, dtype=torch.float64); late[:, 6] = .3
    assert torch.equal(reconstruct_torch(a, late, BasisSpec())[:, :4], a[:, :4])


def test_zero_identity_has_finite_nonzero_learning_jacobian():
    _, a, _ = batch(1)
    coefficients = torch.zeros(1, 8, requires_grad=True)
    out = reconstruct_torch(a, coefficients, BasisSpec())
    assert torch.equal(out, a)
    out[..., 0].sum().backward()
    assert torch.isfinite(coefficients.grad).all() and coefficients.grad.abs().sum() > 0
    stationary = torch.zeros(1, 8, 3)
    c = torch.zeros(1, 8, requires_grad=True)
    reconstruct_torch(stationary, c, BasisSpec()).sum().backward()
    assert torch.isfinite(c.grad).all()


def test_finite_difference_of_basis_gradient_at_zero():
    _, a, _ = batch(1); a = a.double()
    c = torch.zeros(1, 8, dtype=torch.float64, requires_grad=True)
    def objective(x):
        y = reconstruct_torch(a, x, BasisSpec())
        return y[..., 1].sum() + y[..., 2].sin().sum()
    gradient, = torch.autograd.grad(objective(c), c)
    finite_difference = []
    for k in range(8):
        step = torch.zeros_like(c); step[0, k] = 1e-6
        finite_difference.append(((objective(c + step) - objective(c - step)) / 2e-6).item())
    torch.testing.assert_close(gradient[0], torch.tensor(finite_difference, dtype=torch.float64), rtol=1e-6, atol=1e-8)


def test_refiner_zero_initialization_then_true_update_changes_coefficients():
    context, a, target = batch()
    a.requires_grad_(); context["memory"].requires_grad_()
    model = CoefficientRefiner(refiner_config())
    before = model(context, a)
    assert torch.equal(before["trajectory"], a) and torch.equal(before["coefficients"], torch.zeros(3, 8))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    (before["trajectory"] - target).square().mean().backward()
    assert a.grad is None and context["memory"].grad is None
    assert model.coefficient_head.weight.grad.abs().sum() > 0
    optimizer.step(); after = model(context, a)
    assert after["coefficients"].abs().sum() > 0 and not torch.equal(after["trajectory"], a)


def test_a0_a1_same_refiner_initialization_and_teacher_freeze_with_real_gradient():
    context, a, target = batch()
    teacher = ActionEffectModel(cfg(), spec())
    model = EffectGuidedRefiner(CoefficientRefiner(refiner_config()), effect_teacher=teacher).train()
    teacher_hash = state_hash(teacher)
    assert not teacher.training and all(not p.requires_grad for p in teacher.parameters())
    coeff = torch.zeros(3, 8); coeff[:, 4] = .15
    arguments = dict(target_trajectory=target, target_coefficients=coeff, high_quality=torch.zeros(3, dtype=torch.bool))
    a0 = model.training_loss(context, a, **arguments)
    a1 = model.training_loss(context, a, use_effect_guidance=True, target_effect=torch.ones(3, len(spec().names)) * .05,
                            effect_valid=torch.ones(3, len(spec().names), dtype=torch.bool), label_spec=spec(), **arguments)
    assert set(a0.terms) == {"supervision", "magnitude", "preserve"}
    assert set(a1.terms) - set(a0.terms) == {"effect_consistency", "predicted_safety"}
    effect_gradient, = torch.autograd.grad(a1.terms["effect_consistency"], model.refiner.coefficient_head.weight,
                                          retain_graph=True)
    assert torch.isfinite(effect_gradient).all() and effect_gradient.abs().sum() > 0
    optimizer = torch.optim.AdamW(model.refiner.parameters(), lr=1e-3)
    a1.total.backward()
    assert model.refiner.coefficient_head.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in teacher.parameters())
    optimizer.step(); assert teacher_hash == state_hash(teacher)


class CountingBase(nn.Module):
    def __init__(self, context, trajectory):
        super().__init__(); self.calls = 0; self.context = context; self.trajectory = trajectory
        self.parameter = nn.Parameter(torch.tensor(1.))

    def encode_and_plan(self, observations, initial_noise=None):
        self.calls += 1
        return self.context, self.trajectory


def test_deploy_never_calls_teacher_and_base_runs_once():
    context, a, _ = batch()
    base = CountingBase(context, a)
    teacher = ActionEffectModel(cfg(), spec())
    model = EffectGuidedRefiner(CoefficientRefiner(refiner_config()), effect_teacher=teacher, base_policy=base).eval()
    def forbidden(*args, **kwargs):
        raise AssertionError("Training teacher was called during deployment")
    teacher.forward = forbidden
    output = model.generate([{"current": True}] * 3, initial_noise=torch.zeros(3, 8, 4))
    assert base.calls == 1 and output["teacher_forward_calls"] == 0
    assert torch.equal(output["trajectory"], a)
    state = model.deploy_state()
    assert state["effect_teacher_required"] is False
    assert not any("teacher" in name for name in state["refiner"])


def test_geometry_rejection_falls_back_without_fabricating_safety_score():
    context, a, _ = batch()
    model = EffectGuidedRefiner(CoefficientRefiner(refiner_config())).eval()
    # Actual bound check, not a learned score or zero tensor substitution.
    model.constraint_spec = replace(model.constraint_spec, max_speed_mps=.01)
    out = model.deploy(context, a)
    assert not out["geometry_accepted"].any() and torch.equal(out["trajectory"], a)
    assert all(not row["is_safety_label"] for row in out["geometry_screen"])
    with pytest.raises(ValueError, match="verified frozen"):
        model.training_loss(context, a, use_effect_guidance=True)


def test_refiner_strict_export_load_preserves_output():
    context, a, _ = batch()
    original = CoefficientRefiner(refiner_config()).eval()
    with torch.no_grad(): original.coefficient_head.bias[4] = .1
    copy = CoefficientRefiner(refiner_config()).eval(); copy.load_state_dict(original.state_dict(), strict=True)
    torch.testing.assert_close(original(context, a)["trajectory"], copy(context, a)["trajectory"], rtol=0, atol=0)


def test_recorded_grid_support_mask_blocks_unobserved_window_coefficients():
    context, a, _ = batch()
    mask = (True, False, True, True, True, False, True, True)
    model = CoefficientRefiner(replace(refiner_config(), active_coefficients=mask))
    with torch.no_grad(): model.coefficient_head.bias.fill_(.2)
    out = model(context, a)
    assert torch.equal(out["coefficients"][:, [1, 5]], torch.zeros(3, 2))
    out["trajectory"].square().mean().backward()
    assert torch.equal(model.coefficient_head.bias.grad[[1, 5]], torch.zeros(2))
    assert model.coefficient_head.bias.grad[[0, 2, 3, 4, 6, 7]].abs().sum() > 0


def test_optional_predicted_effect_has_one_explicit_deployment_call_and_export_identity():
    context, a, _ = batch()
    teacher = ActionEffectModel(cfg(), spec())
    model = EffectGuidedRefiner(CoefficientRefiner(refiner_config()), effect_teacher=teacher).eval()
    calls = []
    hook = teacher.register_forward_hook(lambda *args: calls.append(1))
    out = model.deploy(context, a, predict_effect=True)
    hook.remove()
    assert calls == [1] and out["teacher_forward_calls"] == 1
    assert torch.equal(out["predicted_effect"]["mean_delta"], torch.zeros(3, len(spec().names)))
    state = model.deploy_state(include_effect_teacher=True)
    assert state["effect_teacher_required"] and state["effect_teacher"]
    assert model.deploy_state()["effect_teacher"] is None


def test_pdms_only_teacher_cannot_claim_trained_safety_effect_guidance():
    context, a, target = batch()
    teacher = ActionEffectModel(cfg("terminal_pdms"), spec())
    model = EffectGuidedRefiner(CoefficientRefiner(refiner_config()), effect_teacher=teacher)
    with pytest.raises(ValueError, match="PDMS-only"):
        model.training_loss(context, a, use_effect_guidance=True, target_trajectory=target,
                            target_coefficients=torch.zeros(3, 8), high_quality=torch.zeros(3, dtype=torch.bool))
