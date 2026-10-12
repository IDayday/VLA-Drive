import numpy as np
from iqe.contracts import require


def argmax_base_tie(predictions):
    require(predictions.ndim == 2 and predictions.shape[1] == 2 and np.isfinite(predictions).all(), "finite two-candidate predictions")
    return (predictions[:, 1] > predictions[:, 0]).astype(np.int64)


def paired_bootstrap(differences, groups, repetitions=2000, seed=11235):
    differences = np.asarray(differences, dtype=np.float64)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    require(len(differences) == len(groups) > 0, "paired bootstrap inputs")
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
    require(scores.ndim == 2 and scores.shape[1] == 2 and np.isfinite(scores).all(), "complete finite candidate labels required")
    n = len(scores); rows = np.arange(n)
    base, expert, selected, oracle = scores[:, 0], scores[:, 1], scores[rows, indices], scores.max(1)
    delta, switch = selected-base, indices == 1
    oracle_gain = float((oracle-base).mean())
    safety = {}
    for ci, name in enumerate(("NC", "DAC")):
        original = components[:, 0, ci]
        result = components[rows, indices, ci]
        safety[name] = {"selected_violations": int((result < 1).sum()), "base_violations": int((original < 1).sum()),
                        "new_violations": int(((original == 1) & (result < 1)).sum()), "denominator": n}
    gap = np.abs(expert-base)
    wrong = selected < oracle-1e-8
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
                      "eligible": int((gap > t).sum())} for t in (.1, .5)},
           "selected_gain_log_bootstrap": paired_bootstrap(delta, groups, bootstrap_repetitions, bootstrap_seed)}
    return out


def bank_statistics(scores, components, groups):
    gap = scores[:, 1]-scores[:, 0]
    return {"scenes": len(scores), "valid_pairs": int((np.abs(gap) > .01).sum()),
            "base_wins": int((gap < -.01).sum()), "expert_wins": int((gap > .01).sum()),
            "near_ties": int((np.abs(gap) <= .01).sum()), "both_high": int((scores.min(1) >= .8).sum()),
            "both_low": int((scores.max(1) < .6).sum()),
            "abs_gap_quantiles": {str(q): float(np.quantile(np.abs(gap), q)) for q in (0, .25, .5, .75, .9, .99, 1)},
            "gap_over_01": int((np.abs(gap) > .1).sum()), "gap_over_05": int((np.abs(gap) > .5).sum()),
            "component_disagreements": (components[:, 0] != components[:, 1]).sum(0).tolist(),
            "per_log": {g: {"n": int((np.asarray(groups) == g).sum()),
                                   "valid_pairs": int(((np.asarray(groups) == g) & (np.abs(gap) > .01)).sum())} for g in sorted(set(groups))}}
