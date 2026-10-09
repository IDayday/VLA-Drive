from __future__ import annotations
import numpy as np
from ..contracts import require
from .oracle import paired_group_bootstrap


def safety_comparison(scores, selected, baseline, components, component_valid, high=.8, zero=1e-8):
    scores = np.asarray(scores)
    selected, baseline = np.asarray(selected), np.asarray(baseline)
    rows = np.arange(len(scores))
    actual, reference = scores[rows, selected], scores[rows, baseline]
    delta = actual - reference
    switch = selected != baseline
    out = {"denominator_scenes": len(scores), "improved": int((delta > zero).sum()), "regressed": int((delta < -zero).sum()),
           "ties": int((np.abs(delta) <= zero).sum()), "mean_gain_01": float(delta.mean()),
           "switches": int(switch.sum()), "switch_rate": float(switch.mean()),
           "false_replacements": int((switch & (delta < -zero)).sum()),
           "switch_precision": float((switch & (delta > zero)).sum() / switch.sum()) if switch.any() else None,
           "high_to_zero": int(((reference >= high) & (actual <= zero)).sum()),
           "selection_regression_scene_indices": np.flatnonzero(delta < -zero).tolist(), "case_gains_01": delta.tolist()}
    for c in ("no_at_fault_collisions", "drivable_area_compliance"):
        label, mask = np.asarray(components[c]), np.asarray(component_valid[c], bool)
        require(mask[rows, selected].all() and mask[rows, baseline].all(), "release safety comparison needs actual valid protected metrics")
        unsafe = label[rows, selected] < 1 - zero
        previous_unsafe = label[rows, baseline] < 1 - zero
        out[c] = {"true_violations": int(unsafe.sum()), "baseline_true_violations": int(previous_unsafe.sum()),
                  "new_violation_scene_indices": np.flatnonzero(unsafe & ~previous_unsafe).tolist(),
                  "new_violations": int((unsafe & ~previous_unsafe).sum())}
    return out


def selection_report(scores, valid, selected, previous, components, component_valid, groups):
    scores, valid = np.asarray(scores), np.asarray(valid, bool)
    rows = np.arange(len(scores))
    require(valid[rows, selected].all(), "selected invalid true labels cannot yield an evaluation")
    oracle = np.where(valid, scores, -np.inf).max(-1)
    actual, base = scores[rows, selected], scores[:, 0]
    denominator = float(oracle.mean() - base.mean())
    return {"selected_mean_01": float(actual.mean()), "base_mean_01": float(base.mean()), "oracle_mean_01": float(oracle.mean()),
            "selection_regret": float((oracle - actual).mean()),
            "oracle_capture": float((actual.mean() - base.mean()) / denominator) if abs(denominator) > 1e-8 else None,
            "base_fallback_rate": float((np.asarray(selected) == 0).mean()),
            "vs_s0": safety_comparison(scores, selected, np.zeros(len(scores), int), components, component_valid),
            "vs_previous": safety_comparison(scores, selected, previous, components, component_valid),
            "paired_group_bootstrap_vs_s0": paired_group_bootstrap(actual - base, groups),
            "paired_group_bootstrap_vs_previous": paired_group_bootstrap(actual - scores[rows, previous], groups),
            "selected_indices": list(map(int, selected)), "previous_indices": list(map(int, previous))}
