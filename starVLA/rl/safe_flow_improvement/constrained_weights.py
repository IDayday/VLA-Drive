"""Finite-candidate entropy-regularized extraction, not a policy safety guarantee.

Maximize w.R - eta KL(w||u), subject to a reference reward floor and optional
separate NC/DAC/TTC cost budgets. All arithmetic is float64, in raw score units.
The reference distribution is an explicitly supplied feasible point.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, xlogy

from .contracts import validate_probability


@dataclass
class WeightSolution:
    weights: np.ndarray
    target: np.ndarray
    reference: np.ndarray
    prior: np.ndarray
    diagnostics: dict


def _kl(weights, prior):
    positive = weights > 0
    return float(np.sum(xlogy(weights[positive], weights[positive] / prior[positive])))


def _primal_oracle(reward, prior, matrix, budget, feasible, eta):
    """Independent FP64 exponential-cone primal (Clarabel 0.11.1).

    Variables are (w, h), with (-h_i,w_i,u_i) in the exponential cone:
    h_i >= w_i log(w_i/u_i). This handles zero weights without a log floor.
    The original u is used, including its mass outside the feasible support.
    See https://clarabel.org/stable/examples/py/example_expcone/ .
    """
    import clarabel
    from scipy import sparse
    from types import SimpleNamespace

    n = len(reward)
    rows = [np.r_[np.ones(n), np.zeros(n)][None],
            np.c_[matrix, np.zeros((len(budget), n))],
            np.c_[-np.eye(n), np.zeros((n, n))]]
    rhs = [np.ones(1), budget, np.zeros(n)]
    cones = [clarabel.ZeroConeT(1), clarabel.NonnegativeConeT(len(budget)),
             clarabel.NonnegativeConeT(n)]
    for i in range(n):
        row = np.zeros((3, 2 * n))
        row[0, n + i] = 1  # b-Ax = -h
        row[1, i] = -1    # b-Ax = w
        rows.append(row)
        rhs.append(np.array([0., 0., prior[i]]))
        cones.append(clarabel.ExponentialConeT())
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter = 250
    settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = 1e-11
    problem = clarabel.DefaultSolver(
        sparse.csc_matrix((2 * n, 2 * n)),
        np.r_[-(reward - reward @ feasible) / eta, np.ones(n)],
        sparse.csc_matrix(np.vstack(rows)), np.concatenate(rhs), cones, settings)
    result = problem.solve()
    raw = np.asarray(result.x[:n], dtype=np.float64)
    # Remove only solver roundoff and recheck every original constraint below.
    w = np.maximum(raw, 0)
    w /= w.sum()
    correction = float(np.max(np.abs(w - raw)))
    gap = abs(result.obj_val - result.obj_val_dual) * eta
    info = SimpleNamespace(success=(str(result.status) in ("Solved", "AlmostSolved")
                                    and gap <= 1e-9 and correction <= 1e-9),
                           message=str(result.status), duality_gap=gap,
                           roundoff_correction=correction)
    return w, info


def solve_weights(reward, costs, reference, empirical, *, eta, safe=True,
                  alpha_max=0.5, delta_useful=0.005, tolerance=1e-9, method="dual"):
    reward = np.asarray(reward, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64)
    if reward.ndim != 1 or len(reward) == 0 or costs.shape != (len(reward), 3):
        raise ValueError("expected rewards [N] and separate NC/DAC/TTC costs [N,3]")
    if not np.isfinite(reward).all() or not np.isfinite(costs).all():
        raise ValueError("nonfinite candidate feedback; infrastructure errors are not reward zero")
    if ((reward < 0) | (reward > 1)).any() or ((costs < 0) | (costs > 1)).any():
        raise ValueError("official reward and costs must remain in raw 0..1 units")
    if not np.isfinite([eta, alpha_max, delta_useful, tolerance]).all() or eta <= 0 or delta_useful <= 0 or tolerance <= 0 or not 0 <= alpha_max <= 1:
        raise ValueError("invalid fixed-scale extraction settings")
    if method not in ("dual", "oracle"):
        raise ValueError("unknown weight solver")
    n = len(reward)
    reference = validate_probability(reference, n, "q_ref")
    empirical = validate_probability(empirical, n, "candidate empirical")
    prior = 0.5 * reference + 0.5 * empirical
    r_ref, c_ref = float(reference @ reward), reference @ costs
    support = prior > 0
    zero_budgets = np.zeros(3, dtype=bool)
    if safe:
        zero_budgets = c_ref <= tolerance
        if zero_budgets.any():
            support &= (costs[:, zero_budgets] <= tolerance).all(axis=1)
    # A reward floor equal to the maximum is a support constraint, not a huge
    # finite softmax multiplier. Constant-reward rows are otherwise redundant.
    if np.max(reward[support]) <= r_ref + tolerance:
        support &= reward >= r_ref - tolerance
    if not support.any() or reference[~support].sum() > tolerance:
        raise ValueError("reference not feasible after support restrictions; inspect the bank")
    r, u, c = reward[support], prior[support], costs[support]
    q = reference[support].copy()
    q /= q.sum()
    matrix, budget, constraint_names = [], [], []
    if np.min(r) < r_ref - tolerance:
        matrix.append(-r)
        budget.append(-r_ref)
        constraint_names.append("negative_reward")
    if safe:
        for j, name in enumerate(("NC", "DAC", "TTC")):
            if not zero_budgets[j] and np.max(c[:, j]) > c_ref[j] + tolerance:
                matrix.append(c[:, j])
                budget.append(c_ref[j])
                constraint_names.append(name)
    matrix = np.asarray(matrix, dtype=np.float64).reshape(-1, len(r))
    budget = np.asarray(budget, dtype=np.float64)
    used = method
    dual_info = None
    if len(r) == 1:
        weights_sub = np.ones(1)
        used = "unique_support"
    elif method == "oracle":
        weights_sub, result = _primal_oracle(r, u, matrix, budget, q, eta)
        if not result.success:
            raise RuntimeError(f"primal oracle did not converge: {result.message}")
    else:
        def dual(lam):
            logits = np.log(u) + (r - lam @ matrix) / eta
            normalizer = logsumexp(logits)
            w = np.exp(logits - normalizer)
            return float(eta * normalizer + lam @ budget), budget - matrix @ w, w

        if len(budget):
            result = minimize(lambda x: dual(x)[:2], np.zeros(len(budget)), jac=True,
                              method="L-BFGS-B", bounds=[(0, None)] * len(budget),
                              options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 2000, "maxls": 100})
            multipliers = result.x
        else:
            multipliers = np.zeros(0)
        dual_value, gradient, weights_sub = dual(multipliers)
        violation = float(np.maximum(matrix @ weights_sub - budget, 0).max(initial=0))
        projected_gradient = np.where(multipliers > 1e-10, gradient, np.minimum(gradient, 0))
        gap = dual_value - (float(weights_sub @ r) - eta * _kl(weights_sub, u))
        dual_info = {
            "constraints": constraint_names,
            "multipliers": multipliers.tolist(),
            "dual_value": dual_value,
            "duality_gap": float(gap),
            "projected_gradient_inf": float(np.abs(projected_gradient).max(initial=0)),
            "complementarity_inf": float(np.abs(multipliers * gradient).max(initial=0)),
        }
        if violation > tolerance or gap < -tolerance or abs(gap) > 1e-8:
            weights_sub, result = _primal_oracle(r, u, matrix, budget, q, eta)
            if not result.success:
                raise RuntimeError(f"dual/primal failure, no manufactured reference success: {result.message}")
            used = "primal_fallback"

    weights = np.zeros(n, dtype=np.float64)
    weights[support] = weights_sub
    r_new, c_new = float(weights @ reward), weights @ costs
    residual = max(abs(float(weights.sum()) - 1), float(max(0, -weights.min())),
                   max(0, r_ref - r_new), float(np.maximum(c_new - c_ref, 0).max()) if safe else 0)
    if residual > tolerance or not np.isfinite(weights).all():
        raise RuntimeError(f"candidate target primal residual {residual} exceeds {tolerance}")
    gain = r_new - r_ref
    alpha = float(alpha_max * np.clip(gain / delta_useful, 0, 1))
    target = (1 - alpha) * reference + alpha * weights
    mixed_cost = target @ costs
    target_residual = max(abs(float(target.sum()) - 1), float(max(0, -target.min())),
                          max(0, r_ref - float(target @ reward)),
                          float(np.maximum(mixed_cost - c_ref, 0).max()) if safe else 0)
    if target_residual > tolerance:
        raise RuntimeError("mixed target violates the original linear constraints")
    diagnostics = {
        "solver": used, "safe_constraints": bool(safe), "eta_raw_score": float(eta),
        "reward_reference": r_ref, "reward_weights": r_new, "gain_before_mixing": gain,
        "reward_target": float(target @ reward), "gain_after_mixing": float(target @ reward - r_ref),
        "alpha": alpha, "alpha_max": float(alpha_max), "delta_useful": float(delta_useful),
        "cost_reference": c_ref.tolist(), "cost_weights": c_new.tolist(), "cost_target": mixed_cost.tolist(),
        "primal_residual": float(residual), "target_primal_residual": float(target_residual),
        "kl_to_original_u": _kl(weights, prior),
        "objective": r_new - eta * _kl(weights, prior),
        "ess_weights": float(1 / np.sum(weights**2)), "ess_target": float(1 / np.sum(target**2)),
        "prior_mass_on_allowed_support": float(u.sum()), "excluded_candidates": int((~support).sum()),
        "zero_cost_budgets": zero_budgets.tolist(), "dual": dual_info,
        "fully_safe_mass_weights": float(weights[(costs <= tolerance).all(axis=1)].sum()),
        "fully_safe_mass_target": float(target[(costs <= tolerance).all(axis=1)].sum()),
    }
    return WeightSolution(weights, target, reference.copy(), prior, diagnostics)
