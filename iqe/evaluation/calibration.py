from __future__ import annotations
from itertools import product
import numpy as np
import torch
from dataclasses import asdict
from ..contracts import require
from ..selector import SelectionRule, RouterRule
from .selection import safety_comparison


def within_budget(comparison, budgets):
    n = comparison["denominator_scenes"]
    return comparison["false_replacements"] / n <= budgets["false_replacement_rate"] and \
        comparison["high_to_zero"] / n <= budgets["high_to_zero_rate"] and all(
            comparison[c]["new_violations"] / n <= budgets["new_hard_safety_violation_rate"]
            for c in ("no_at_fault_collisions", "drivable_area_compliance"))


def calibrate(candidates, prediction, true_scores, components, component_valid, previous_indices, config, *, role, dependency_hash):
    require(role == "selector_cal", "calibration strictly isolated from fit/stage_val/dev/test")
    require(config["sampling"] == "natural_scene_distribution", "result-based calibration resampling forbidden")
    ids = candidates.expert_ids
    rows = np.arange(len(true_scores))
    base = np.zeros(len(rows), int)
    metrics = config["protected_metrics"]
    rules = [SelectionRule(0, {}, {}, True, dependency_hash=dependency_hash)]
    rules.extend(SelectionRule(delta, dict(zip(metrics, eta)), {c: config["safety_relative_tolerance"] for c in metrics},
                               dependency_hash=dependency_hash)
                 for delta in config["delta_grid"] for eta in product(config["eta_grid"], repeat=len(metrics)))
    results = []
    for rule in rules:
        _, winners, _ = rule(candidates, prediction)
        selected = np.array([ids.index(e) for e in winners])
        comparisons = [safety_comparison(true_scores, selected, reference, components, component_valid)
                       for reference in (base, previous_indices)]
        feasible = all(within_budget(c, config["budgets"]) for c in comparisons)
        results.append({"rule": asdict(rule), "mean_selected_01": float(true_scores[rows, selected].mean()),
                        "feasible": feasible, "vs_s0": comparisons[0], "vs_previous": comparisons[1]})
    feasible = [r for r in results if r["feasible"]]
    if not feasible:
        # Diagnostic Base rule remains executable, but cannot replace the preceding
        # serving bundle. Continue evaluation/export diagnostics after a failed gate.
        return rules[0], {"role": role, "dependency_hash": dependency_hash, "chosen": results[0],
            "search": results, "beneficial": False, "status": "NO_FEASIBLE_IMPROVEMENT",
            "deployment_action": "RETAIN_PREVIOUS_BUNDLE"}
    # Highest actual held-out score; ties prefer always-Base then more conservative margin.
    chosen = max(feasible, key=lambda r: (r["mean_selected_01"], r["rule"]["always_base"], r["rule"]["delta"]))
    previous_score = float(true_scores[rows, previous_indices].mean())
    improved = chosen["mean_selected_01"] > max(float(true_scores[:, 0].mean()), previous_score) + 1e-8
    return SelectionRule(**chosen["rule"]), {"role": role, "dependency_hash": dependency_hash,
        "chosen": chosen, "search": results, "beneficial": improved,
        "status": "CALIBRATED_IMPROVEMENT" if improved else "CANDIDATE_GAIN_NOT_REALIZED",
        "risk_statement": "empirical risk control, not a formal safety guarantee"}


def calibrate_router(logits, expert_ids, scores, components, component_valid, previous, config, *, role, dependency_hash=""):
    require(role == "selector_cal", "Router calibrates only selector_cal")
    rows = np.arange(len(scores))
    rules = [RouterRule(always_base=True, dependency_hash=dependency_hash)] + [RouterRule(l, p, dependency_hash=dependency_hash) for l in config["logit_margin_grid"] for p in config["probability_margin_grid"]]
    feasible = []
    for rule in rules:
        indices = rule(logits, expert_ids).cpu().numpy()
        checks = [safety_comparison(scores, indices, ref, components, component_valid) for ref in (np.zeros(len(rows), int), previous)]
        if all(within_budget(c, config["budgets"]) for c in checks):
            feasible.append((float(scores[rows, indices].mean()), rule, checks))
    if not feasible:
        return rules[0], {"role": role, "beneficial": False, "status": "NO_FEASIBLE_IMPROVEMENT",
                          "deployment_action": "RETAIN_PREVIOUS_BUNDLE", "candidate_geometry_protection": False}
    score, rule, checks = max(feasible, key=lambda x: (x[0], x[1].always_base, x[1].logit_margin, x[1].probability_margin))
    beneficial = score > max(float(scores[:,0].mean()), float(scores[rows,previous].mean())) + 1e-8
    return rule, {"mean_selected_01": score, "role": role, "checks": checks, "candidate_geometry_protection": False,
                  "beneficial": beneficial, "status": "CALIBRATED_IMPROVEMENT" if beneficial else "CANDIDATE_GAIN_NOT_REALIZED"}
