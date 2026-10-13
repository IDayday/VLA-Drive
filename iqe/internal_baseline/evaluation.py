import numpy as np
from iqe.contracts import require
from iqe.scoring.reference import COMPONENTS


ZERO_EPSILON = 1e-8
EPSILON_OPTIMAL = .01


def _rate(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def _candidate_labels(scores):
    scores = np.asarray(scores, dtype=np.float64)
    # The bank builder must resolve missing labels before complete-set evaluation.
    # Never silently report the surviving valid subset as the complete dataset.
    require(scores.ndim == 2 and scores.shape[1] == 2 and len(scores) > 0
            and np.isfinite(scores).all(), "complete finite candidate labels required")
    require(((scores >= 0) & (scores <= 1)).all(), "candidate score scale must be [0,1]")
    return scores


def _component_labels(components, n):
    components = np.asarray(components, dtype=np.float64)
    require(components.ndim == 3 and components.shape[:2] == (n, 2)
            and 2 <= components.shape[-1] <= len(COMPONENTS), "canonical named component schema required")
    finite = np.isfinite(components)
    require(not np.isinf(components).any(), "infinite component label")
    require(((components[finite] >= 0) & (components[finite] <= 1)).all(), "component scale must be [0,1]")
    return components


def _safety_metrics(components, indices):
    """Arrays use bank.py's locked COMPONENTS schema; NaN denotes missing data."""
    n, rows = len(indices), np.arange(len(indices))
    safety = {}
    for short, name in (("NC", "no_at_fault_collisions"), ("DAC", "drivable_area_compliance")):
        ci = COMPONENTS.index(name)
        base, expert = components[:, 0, ci], components[:, 1, ci]
        selected = components[rows, indices, ci]
        vb, ve, vs = np.isfinite(base), np.isfinite(expert), np.isfinite(selected)
        paired, paired_expert = vb & vs, ve & vs
        base_bad, expert_bad, selected_bad = base < 1-ZERO_EPSILON, expert < 1-ZERO_EPSILON, selected < 1-ZERO_EPSILON
        new = paired & ~base_bad & selected_bad
        new_expert = paired_expert & ~expert_bad & selected_bad
        safe_base = paired & ~base_bad
        safety[short] = {
            "component_name": name, "full_score": 1., "full_score_tolerance": ZERO_EPSILON,
            "selected_violations": int((vs & selected_bad).sum()), "base_violations": int((vb & base_bad).sum()),
            "expert_violations": int((ve & expert_bad).sum()), "new_violations": int(new.sum()),
            "resolved_base_violations": int((paired & base_bad & ~selected_bad).sum()),
            "new_violations_vs_expert": int(new_expert.sum()),
            "denominator": int(paired.sum()), "requested_scenes": n,
            "selected_valid_denominator": int(vs.sum()), "base_valid_denominator": int(vb.sum()),
            "expert_valid_denominator": int(ve.sum()), "vs_expert_valid_denominator": int(paired_expert.sum()),
            "missing_selected": int((~vs).sum()), "missing_base": int((~vb).sum()), "missing_expert": int((~ve).sum()),
            "base_safe_denominator": int(safe_base.sum()),
            "new_violation_rate_all_valid_pairs": _rate(new.sum(), paired.sum()),
            "new_violation_rate_base_safe": _rate(new.sum(), safe_base.sum()),
            "selected_violation_rate": _rate((vs & selected_bad).sum(), vs.sum()),
            "base_violation_rate": _rate((vb & base_bad).sum(), vb.sum()),
        }
    return safety


def argmax_base_tie(predictions):
    predictions = np.asarray(predictions)
    require(predictions.ndim == 2 and predictions.shape[1] == 2 and np.isfinite(predictions).all(), "finite two-candidate predictions")
    return (predictions[:, 1] > predictions[:, 0]).astype(np.int64)


def paired_bootstrap(differences, groups, repetitions=2000, seed=11235):
    differences = np.asarray(differences, dtype=np.float64)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    require(differences.ndim == groups.ndim == 1 and len(differences) == len(groups) > 0
            and np.isfinite(differences).all(), "paired bootstrap inputs")
    require(isinstance(repetitions, (int, np.integer)) and repetitions > 0, "positive bootstrap repetitions required")
    sums = np.array([differences[groups == g].sum() for g in unique])
    counts = np.array([(groups == g).sum() for g in unique])
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repetitions):
        sample = rng.integers(len(unique), size=len(unique))
        estimates.append(sums[sample].sum()/counts[sample].sum())
    return {"mean": float(differences.mean()), "lower_95": float(np.quantile(estimates, .025)),
            "upper_95": float(np.quantile(estimates, .975)), "groups": len(unique),
            "repetitions": repetitions, "seed": seed, "reliable_log_count": len(unique) >= 10}


def selection_metrics(scores, components, indices, groups, *, bootstrap_repetitions=2000, bootstrap_seed=11235):
    scores = _candidate_labels(scores)
    n = len(scores); rows = np.arange(n)
    components = _component_labels(components, n)
    indices, groups = np.asarray(indices), np.asarray(groups)
    require(indices.shape == (n,) and np.issubdtype(indices.dtype, np.integer)
            and np.isin(indices, (0, 1)).all(), "two-candidate selected indices required")
    require(groups.shape == (n,), "one source log/group required per scene")
    base, expert, selected, oracle = scores[:, 0], scores[:, 1], scores[rows, indices], scores.max(1)
    delta, switch = selected-base, indices == 1
    oracle_gain = float((oracle-base).mean())
    safety = _safety_metrics(components, indices)
    gap = np.abs(expert-base)
    wrong = selected < oracle-1e-8
    candidate_delta = expert-base
    a = float(np.maximum(candidate_delta, 0).mean())
    b = float(np.maximum(-candidate_delta, 0).mean())
    regret = float((oracle-selected).mean())
    best_index = int(expert.mean() > base.mean())
    best_fixed = scores[:, best_index]
    identities = {
        "expert_minus_base_equals_A_minus_B": (float(candidate_delta.mean()), a-b),
        "oracle_minus_base_equals_A": (float(oracle.mean()-base.mean()), a),
        "oracle_minus_best_fixed_equals_min_A_B": (float(oracle.mean()-best_fixed.mean()), min(a, b)),
        "selected_minus_best_fixed_equals_min_A_B_minus_R": (float(selected.mean()-best_fixed.mean()), min(a, b)-regret),
    }
    identity_errors = {name: {"lhs": lhs, "rhs": rhs, "absolute_error": abs(lhs-rhs)}
                       for name, (lhs, rhs) in identities.items()}
    identity_tolerance = 1e-12
    require(all(v["absolute_error"] <= identity_tolerance for v in identity_errors.values()), "two-candidate A/B/R identities failed")
    tied_switches = switch & (np.abs(delta) <= ZERO_EPSILON)
    positive_switches, negative_switches = delta > ZERO_EPSILON, delta < -ZERO_EPSILON
    strict_switches = positive_switches | negative_switches
    epsilon_optimal = oracle-selected <= EPSILON_OPTIMAL+1e-12
    out = {"n": n, "base": float(base.mean()), "expert": float(expert.mean()), "oracle": float(oracle.mean()),
           "selected": float(selected.mean()), "oracle_gain": oracle_gain, "selected_gain": float(delta.mean()),
           "selection_regret": float((oracle-selected).mean()),
           "oracle_capture": float(delta.mean()/oracle_gain) if abs(oracle_gain) > 1e-8 else None,
           "switch_count": int(switch.sum()), "switch_rate": float(switch.mean()), "base_fallback_rate": float((~switch).mean()),
           "positive_switch_gain_total": float(delta.clip(0).sum()), "negative_switch_cost_total": float((-delta).clip(0).sum()),
           "positive_switches": int((delta > 1e-8).sum()), "negative_switches": int((delta < -1e-8).sum()),
           "improvement_rate": float((delta > 1e-8).mean()), "degradation_rate": float((delta < -1e-8).mean()),
           "tie_rate": float((np.abs(delta) <= 1e-8).mean()),
           "positive_switch_gain_per_scene": float(delta.clip(0).mean()), "negative_switch_cost_per_scene": float((-delta).clip(0).mean()),
           "switch_precision": float((delta > 1e-8).sum()/switch.sum()) if switch.any() else None,
           "high_to_zero": int(((base >= .8) & (selected <= 1e-8)).sum()), "rate_denominator": n,
           "high_to_zero_rate_all_scenes": float(((base >= .8) & (selected <= 1e-8)).mean()),
           "base_high_count": int((base >= .8).sum()),
           "high_to_zero_rate_base_high": float(((base >= .8) & (selected <= 1e-8)).sum()/(base >= .8).sum()) if (base >= .8).any() else None,
           "safety": safety, "large_gap_errors": {str(t): {"errors": int((wrong & (gap > t)).sum()),
                      "eligible": int((gap > t).sum()), "error_rate": _rate((wrong & (gap > t)).sum(), (gap > t).sum()),
                      "error_rate_all_scenes": float((wrong & (gap > t)).mean()), "all_scene_denominator": n,
                      "regret_total": float((oracle-selected)[gap > t].sum())} for t in (.1, .5)},
           "selected_gain_log_bootstrap": paired_bootstrap(delta, groups, bootstrap_repetitions, bootstrap_seed)}
    out.update({
        "score_scale": "zero_one", "requested_scenes": n, "joint_valid_scenes": n,
        "invalid_total_score_scenes": 0, "component_schema": list(COMPONENTS[:components.shape[-1]]),
        "best_fixed_policy": "expert_1" if best_index else "expert_0", "best_fixed_mean": float(best_fixed.mean()),
        "selected_gain_vs_expert": float((selected-expert).mean()),
        "oracle_gain_vs_best_fixed_policy": float((oracle-best_fixed).mean()),
        "selected_gain_vs_best_fixed_policy": float((selected-best_fixed).mean()),
        "complementarity": {"A_positive_expert_gain": a, "B_negative_expert_cost": b, "R_selection_regret": regret,
                            "min_A_B": min(a, b), "denominator": n, "identities": identity_errors,
                            "identity_tolerance": identity_tolerance, "identities_pass": True},
        "switch_ties": int(tied_switches.sum()), "switch_precision_denominator": int(switch.sum()),
        "switch_precision_numerator": int(positive_switches.sum()),
        "switch_precision_excluding_ties": _rate(positive_switches.sum(), strict_switches.sum()),
        "switch_precision_excluding_ties_denominator": int(strict_switches.sum()),
        "switch_tie_rate": _rate(tied_switches.sum(), switch.sum()),
        "true_candidate_ties": int((gap <= ZERO_EPSILON).sum()),
        "near_candidate_ties_001": int((gap <= .01).sum()),
        "false_replacement_count": int(negative_switches.sum()),
        "false_replacement_rate_switched": _rate(negative_switches.sum(), switch.sum()),
        "false_replacement_rate_all_scenes": float(negative_switches.mean()),
        "epsilon_optimal": {"epsilon": EPSILON_OPTIMAL, "numerical_tolerance": 1e-12,
                            "count": int(epsilon_optimal.sum()), "denominator": n, "rate": float(epsilon_optimal.mean())},
        "selected_gain_vs_best_fixed_log_bootstrap": paired_bootstrap(selected-best_fixed, groups, bootstrap_repetitions, bootstrap_seed),
        "expert_gain_log_bootstrap": paired_bootstrap(candidate_delta, groups, bootstrap_repetitions, bootstrap_seed),
    })
    return out


def bank_statistics(scores, components, groups):
    scores = _candidate_labels(scores)
    components = _component_labels(components, len(scores))
    groups = np.asarray(groups)
    require(groups.shape == (len(scores),), "one source log/group required per bank scene")
    gap = scores[:, 1]-scores[:, 0]
    valid = np.abs(gap) > .01
    weights = np.where(valid, np.abs(gap), 0)
    total_weight = float(weights.sum())
    q = weights/total_weight if total_weight > 0 else np.zeros_like(weights)
    both_component_valid = np.isfinite(components).all(1)
    return {"scenes": len(scores), "joint_valid_scenes": len(scores), "invalid_total_score_scenes": 0,
            "valid_pairs": int(valid.sum()), "valid_pair_groups": len(np.unique(groups[valid])),
            "comparison_signal": "AVAILABLE" if valid.any() else "SCORER_COMPARISON_SIGNAL_ABSENT",
            "comparison_total_gap_weight": total_weight,
            "comparison_effective_scene_count": float(1/(q*q).sum()) if valid.any() else None,
            "comparison_max_probability": float(q.max()) if valid.any() else None,
            "comparison_sampling": "q=abs(gap)/sum(abs(gap)); sampled pair loss is not reweighted",
            "base_wins": int((gap < -.01).sum()), "expert_wins": int((gap > .01).sum()),
            "near_ties": int((np.abs(gap) <= .01).sum()), "both_high": int((scores.min(1) >= .8).sum()),
            "both_low": int((scores.max(1) < .6).sum()),
            "abs_gap_quantiles": {str(q): float(np.quantile(np.abs(gap), q)) for q in (0, .25, .5, .75, .9, .99, 1)},
            "gap_over_01": int((np.abs(gap) > .1).sum()), "gap_over_05": int((np.abs(gap) > .5).sum()),
            "expert_high_base_low": int(((scores[:, 1] >= .8) & (scores[:, 0] < .6)).sum()),
            "base_high_expert_low": int(((scores[:, 0] >= .8) & (scores[:, 1] < .6)).sum()),
            "component_disagreements": ((components[:, 0] != components[:, 1]) & both_component_valid).sum(0).tolist(),
            "component_joint_valid_denominators": both_component_valid.sum(0).tolist(),
            "component_schema": list(COMPONENTS[:components.shape[-1]]),
            "per_log": {g: {"n": int((np.asarray(groups) == g).sum()),
                                   "valid_pairs": int(((groups == g) & valid).sum()),
                                   "comparison_probability_mass": float(q[groups == g].sum())} for g in sorted(set(groups))}}
