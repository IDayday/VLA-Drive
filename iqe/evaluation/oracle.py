from __future__ import annotations
import numpy as np
from ..contracts import require


def paired_group_bootstrap(values, groups, repetitions=2000, seed=42):
    require(len(values) == len(groups) and len(values) > 0, "bootstrap pairs")
    values = np.asarray(values, dtype=np.float64)
    require(np.isfinite(values).all(), "bootstrap requires finite valid labels")
    unique = sorted(set(groups))
    # Cluster resampling retains every scene and weights scenes as in the original mean.
    sums = np.array([values[np.array(groups) == g].sum() for g in unique])
    counts = np.array([sum(x == g for x in groups) for g in unique])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(repetitions, len(unique)))
    means = sums[draws].sum(1) / counts[draws].sum(1)
    return {"mean": float(values.mean()), "lower": float(np.quantile(means, .025)),
            "upper": float(np.quantile(means, .975)), "groups": len(unique), "repetitions": repetitions, "seed": seed}


def oracle_report(scores, valid, components, component_valid, expert_ids, groups, new_id, thresholds, gates, *, smoke=False):
    scores, valid = np.asarray(scores), np.asarray(valid, bool)
    require(scores.shape == valid.shape and scores.shape[1] == len(expert_ids), "Oracle bank shapes")
    require(np.isfinite(scores[valid]).all() and valid[:, 0].all(), "invalid Base scoring prevents scientific result")
    new_index = expert_ids.index(new_id)
    old_indices = [i for i, eid in enumerate(expert_ids) if int(eid.split('_')[-1]) < int(new_id.split('_')[-1])]
    old = np.where(valid[:, old_indices], scores[:, old_indices], -np.inf).max(-1)
    current = np.maximum(old, np.where(valid[:, new_index], scores[:, new_index], -np.inf))
    gains = current - old
    require((gains >= 0).all(), "Oracle should be pointwise nondecreasing under a locked reference protocol")
    quality = valid & (scores >= thresholds.high_quality)
    hard_safe = valid.copy()
    for c in ("no_at_fault_collisions", "drivable_area_compliance"):
        hard_safe &= np.asarray(component_valid[c], bool) & (np.asarray(components[c]) >= 1 - thresholds.zero_epsilon)
    quality &= hard_safe
    unsolved = ~quality[:, old_indices].any(-1)
    new_success = unsolved & quality[:, new_index]
    bootstrap = paired_group_bootstrap(gains, groups)
    gate = not smoke and bootstrap["groups"] >= gates["minimum_logs_for_ci_gate"] and \
        gains.mean() >= gates["oracle_min_mean_gain"] and bootstrap["lower"] > 0 and \
        new_success.sum() >= gates["oracle_min_new_successes"]
    return {"base_mean_01": float(scores[:, 0].mean()), "new_mean_01": float(scores[valid[:, new_index], new_index].mean()) if valid[:, new_index].any() else None,
            "old_oracle_mean_01": float(old.mean()), "new_oracle_mean_01": float(current.mean()),
            "oracle_gain_01": float(gains.mean()), "pointwise_gain_01": gains.tolist(),
            "new_successes": int(new_success.sum()), "new_success_denominator": int(unsolved.sum()),
            "new_success_rate": float(new_success.sum() / unsolved.sum()) if unsolved.any() else None,
            "severe_low_recovery": int(((old < thresholds.severe_low) & (current >= thresholds.high_quality)).sum()),
            "ordinary_unsafe_candidate_fraction": float((~unsolved & ~hard_safe[:, new_index]).sum() / (~unsolved).sum()) if (~unsolved).any() else None,
            "ordinary_candidate_denominator": int((~unsolved).sum()),
            "bootstrap": bootstrap, "science": "UNTESTED" if smoke else "PASS" if gate else "INCONCLUSIVE" if bootstrap["groups"] < 10 else "NO_COVERAGE_GAIN",
            "oracle_gate": gate}
