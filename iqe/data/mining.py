from __future__ import annotations
from dataclasses import dataclass, asdict
import math
from ..contracts import require

NC, DAC = "no_at_fault_collisions", "drivable_area_compliance"


@dataclass(frozen=True)
class MiningThresholds:
    high_quality: float = .8
    severe_low: float = .6
    target_gain_margin: float = .05
    zero_epsilon: float = 1e-8


def safe_quality(score, thresholds):
    return score.score_valid and score.total_score_01 >= thresholds.high_quality and all(
        score.component_valid_masks.get(c, False) and score.named_components[c] >= 1 - thresholds.zero_epsilon for c in (NC, DAC))


def hard_safe(score, thresholds):
    return score.score_valid and all(score.component_valid_masks.get(c, False) and
                                    score.named_components[c] >= 1 - thresholds.zero_epsilon for c in (NC, DAC))


def diagnose_scene(scene, old_scores, target_score, selected_id, target_audit, thresholds=MiningThresholds()):
    require(old_scores and "expert_0" in old_scores, "old candidate pool required")
    valid = {k: v for k, v in old_scores.items() if v.score_valid}
    if not valid:
        return {"scene_id": scene.scene_id, "status": "INVALID_OLD_POOL", "eligible_for_expert_training": False}
    best = max(s.total_score_01 for s in valid.values())
    good = {k for k, s in valid.items() if safe_quality(s, thresholds)}
    coverage = not good
    selected_good = selected_id in good
    selection = bool(good) and not selected_good
    target_input_valid = target_audit["finite"] and target_audit["coordinates_time_yaw_legal"] and \
                         target_audit["input_context_consistent"]
    target_ok = target_input_valid and safe_quality(target_score, thresholds)
    repair_unsafe = not any(hard_safe(s, thresholds) for s in valid.values()) and hard_safe(target_score, thresholds)
    gain_ok = target_score.score_valid and (target_score.total_score_01 - best >= thresholds.target_gain_margin or repair_unsafe)
    target_gap = coverage and not (target_ok and gain_ok)
    eligible = coverage and target_ok and gain_ok and scene.split_role == "incremental_fit"
    reasons = []
    if not target_input_valid:
        reasons.append("target_geometry_or_input_context_invalid")
    if not target_score.score_valid:
        reasons.append("target_scoring_failed")
    elif not safe_quality(target_score, thresholds):
        reasons.append("target_unusable_under_current_evaluation_protocol")
    if not gain_ok:
        reasons.append("target_has_no_audited_quality_or_safety_gain")
    return {"scene_id": scene.scene_id, "coverage_gap": coverage, "selection_gap": selection, "target_gap": target_gap,
            "deployed_selected_expert_id": selected_id,
            "eligible_for_expert_training": bool(eligible), "best_old_score_01": best,
            "severe_low": best < thresholds.severe_low, "target_score_01": target_score.total_score_01,
            "target_audit_reasons": reasons, "safe_quality_expert_ids": sorted(good)}
