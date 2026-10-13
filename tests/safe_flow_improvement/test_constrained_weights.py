import numpy as np
import pytest

from starVLA.rl.safe_flow_improvement.constrained_weights import solve_weights


def solve(reward, costs=None, reference=None, **kwargs):
    reward = np.asarray(reward, dtype=np.float64)
    n = len(reward)
    if costs is None:
        costs = np.zeros((n, 3))
    if reference is None:
        reference = np.eye(1, n)[0]
    return solve_weights(reward, costs, reference, np.ones(n) / n, eta=0.02, **kwargs)


def test_zero_budget_has_zero_violating_mass_and_preserves_original_prior():
    result = solve([0.8, 1.0, 0.9], [[0, 0, 0], [0.5, 0, 0], [0, 0, 0]])
    assert result.weights[1] == result.target[1] == 0
    assert result.prior[1] > 0
    actual_kl = sum(w * np.log(w / u) for w, u in zip(result.weights, result.prior) if w > 0)
    assert result.diagnostics["kl_to_original_u"] == pytest.approx(actual_kl)
    assert result.diagnostics["prior_mass_on_allowed_support"] < 1


@pytest.mark.parametrize("reward", [[0, 0, 0], [0.8, 0.8, 0.8]])
def test_constant_reward_has_no_manufactured_improvement(reward):
    result = solve(reward)
    assert result.diagnostics["alpha"] <= 1e-12
    assert result.target == pytest.approx(result.reference)


def test_only_one_safe_candidate_and_reward_floor():
    result = solve([0.8, 0.99, 0.95], [[0, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert result.weights.tolist() == [1, 0, 0]
    assert result.diagnostics["alpha"] == 0
    assert result.diagnostics["fully_safe_mass_target"] == 1


def test_nonzero_cost_constraints_are_separate_and_nontrivial():
    reward = [0.65, 0.8, 0.9, 0.82, 0.95, 0.99]
    costs = [[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1], [0, 0, 0], [1, 0, 0], [0, 1, 0]]
    q = [0.5, 0.5, 0, 0, 0, 0]
    result = solve(reward, costs, q)
    assert result.weights[2] == 0
    assert (result.weights @ np.asarray(costs) <= np.asarray(q) @ costs + 1e-9).all()
    assert (result.target @ np.asarray(costs) <= np.asarray(q) @ costs + 1e-9).all()
    assert result.diagnostics["gain_after_mixing"] > 0


def test_reward_floor_can_bind_without_safe_costs():
    result = solve([0.9, 0.1, 0.2, 0.91], safe=False)
    assert result.weights @ np.array([0.9, 0.1, 0.2, 0.91]) >= 0.9 - 1e-9


def test_tiny_absolute_gains_cannot_force_large_mixture():
    result = solve([0.8, 0.800001], alpha_max=0.5, delta_useful=0.005)
    assert result.diagnostics["alpha"] <= 0.0001 + 1e-12
    assert result.diagnostics["gain_after_mixing"] <= 1e-10 + 1e-12


@pytest.mark.parametrize("seed", range(8))
def test_dual_matches_independent_primal_oracle(seed):
    rng = np.random.default_rng(seed)
    reward = rng.uniform(0.4, 1, 12)
    costs = rng.choice([0, 0.5, 1], size=(12, 3))
    reference = np.zeros(12)
    reference[:4] = 0.25
    dual = solve(reward, costs, reference, method="dual")
    oracle = solve(reward, costs, reference, method="oracle")
    assert dual.diagnostics["objective"] == pytest.approx(oracle.diagnostics["objective"], abs=2e-8)
    assert dual.weights == pytest.approx(oracle.weights, abs=3e-5)


@pytest.mark.parametrize("bad", [float("nan"), -0.01, 1.01])
def test_bad_reward_cannot_become_training_zero(bad):
    with pytest.raises(ValueError):
        solve([0.8, bad])
