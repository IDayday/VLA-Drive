"""Shared schema and the pre-registered automatic verdict rules."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Sequence


FACTOR_NAMES = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
    "score",
)
PREDICTED_FACTOR_NAMES = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "time_to_collision_within_bound",
    "ego_progress",
    "comfort",
)
TOP_K_VALUES = (1, 2, 3, 4, 6, 8, 16, 32, 64)
QUALITY_THRESHOLDS = (0.50, 0.80, 0.90, 0.95, 0.99)
EPSILON_ORACLE_VALUES = (0.005, 0.01, 0.02, 0.05)


@dataclass(frozen=True)
class VerdictInputs:
    """All values are means/rates on the 0--1 PDMS scale."""

    infrastructure_valid: bool
    coverage_gap: float
    ranking_gap: float
    ideal1_oracle_gain: float
    ideal1_selected_gain: float
    ideal8_oracle_gain: float
    ideal16_oracle_gain: float
    ideal8_selected_gain: float
    ideal16_selected_gain: float
    ideal8_oracle_ci_low: float
    ideal8_selected_ci_low: float
    fixed_budget_gain: float
    duplicate_shift: float
    candidate_limited_share_low: float
    ranker_limited_share_low: float
    target_available_rate: float
    saturated_false_replacement_rate: float
    practical_selected_gain: float
    ideal64_oracle_gain: float = float("nan")
    ideal64_selected_gain: float = float("nan")


def _finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def choose_verdict(values: VerdictInputs) -> Dict[str, Any]:
    """Apply the user's verdict rules in their explicitly registered priority.

    Vague prose conditions in SCORER_FIRST/STOP_LORA are made measurable here and
    emitted in ``operational_definitions``.  Missing F3 inputs never become a
    pass: they produce INCONCLUSIVE.
    """

    payload: Dict[str, Any] = {
        "inputs": asdict(values),
        "rule_priority": [
            "INFRA_INVALID",
            "DIRECT_LORA_GENERATOR_STRONG_PASS",
            "DIRECT_LORA_GENERATOR_CONDITIONAL_PASS",
            "LORA_DENSIFIER_PASS",
            "JOINT_LORA_SCORER_ONLY",
            "SCORER_FIRST",
            "STOP_LORA",
            "INCONCLUSIVE",
        ],
        "operational_definitions": {
            "ideal8_clearly_better_than_ideal1": "ideal8_selected_gain - ideal1_selected_gain >= 0.002",
            "perfect_scorer_potential_obvious": "ranking_gap >= 0.01",
            "fixed_budget_not_fully_eliminated": "fixed_budget_gain > 0 or fixed-budget oracle gain remains positive",
            "target_available_rate_very_low": "target_available_rate < 0.10",
            "scorer_gap_small": "ranking_gap < 0.002",
        },
    }
    if not values.infrastructure_valid:
        payload.update(verdict="INFRA_INVALID", triggered_rule="F0 evaluator/forward parity gate")
        return payload

    required = (
        values.ideal1_oracle_gain,
        values.ideal1_selected_gain,
        values.ideal8_oracle_gain,
        values.ideal16_oracle_gain,
        values.ideal8_selected_gain,
        values.ideal8_oracle_ci_low,
        values.ideal8_selected_ci_low,
        values.fixed_budget_gain,
        values.duplicate_shift,
        values.target_available_rate,
        values.saturated_false_replacement_rate,
        values.practical_selected_gain,
    )
    if not all(_finite(value) for value in required):
        payload.update(
            verdict="INCONCLUSIVE",
            triggered_rule="one or more mandatory F3 statistics are unavailable",
        )
        return payload

    if (
        values.ideal8_oracle_gain >= 0.005
        and values.ideal8_oracle_ci_low > 0.002
        and values.target_available_rate >= 0.25
        and values.fixed_budget_gain > 0.0
    ):
        payload.update(
            verdict="DIRECT_LORA_GENERATOR_STRONG_PASS",
            triggered_rule="Verdict 1",
            recommended_output="Base64+LoRA8/16 or fixed-budget Base56+LoRA8",
            frozen_scorer_safe=(
                abs(values.duplicate_shift) < 0.001
                and values.saturated_false_replacement_rate < 0.02
            ),
        )
        return payload

    if (
        0.003 <= values.ideal8_oracle_gain < 0.005
        and values.ideal8_oracle_ci_low > 0.0
        and values.candidate_limited_share_low >= 0.15
        and values.target_available_rate >= 0.20
    ):
        payload.update(
            verdict="DIRECT_LORA_GENERATOR_CONDITIONAL_PASS",
            triggered_rule="Verdict 2",
            recommended_output="low-cost 8/16-candidate pilot only",
            frozen_scorer_safe=(
                abs(values.duplicate_shift) < 0.001
                and values.saturated_false_replacement_rate < 0.02
            ),
        )
        return payload

    densifier_better_than_one = (
        values.ideal8_selected_gain - values.ideal1_selected_gain >= 0.002
    )
    if (
        values.ideal8_oracle_gain < 0.003
        and values.ideal8_selected_gain >= 0.003
        and values.ideal8_selected_ci_low > 0.0
        and densifier_better_than_one
        and abs(values.duplicate_shift) < 0.001
        and values.saturated_false_replacement_rate < 0.02
    ):
        payload.update(
            verdict="LORA_DENSIFIER_PASS",
            triggered_rule="Verdict 3",
            recommended_output="8--16 scorer-friendly near-optimal candidates",
            frozen_scorer_safe=True,
        )
        return payload

    if (
        values.ideal8_oracle_gain >= 0.003
        and (
            values.ideal8_selected_gain < 0.001
            or abs(values.duplicate_shift) >= 0.001
            or values.saturated_false_replacement_rate >= 0.02
        )
    ):
        payload.update(
            verdict="JOINT_LORA_SCORER_ONLY",
            triggered_rule="Verdict 4",
            recommended_output="LoRA candidate bank only with mixed-bank scorer retraining",
            frozen_scorer_safe=False,
        )
        return payload

    if (
        values.ranking_gap >= 3.0 * values.coverage_gap
        and values.ranker_limited_share_low > values.candidate_limited_share_low
        and values.ranking_gap >= 0.01
        and values.ideal8_oracle_gain < 0.003
        and values.practical_selected_gain < 0.002
    ):
        payload.update(
            verdict="SCORER_FIRST",
            triggered_rule="Verdict 5",
            recommended_output="do not prioritize generator LoRA; improve scorer first",
            frozen_scorer_safe=(
                abs(values.duplicate_shift) < 0.001
                and values.saturated_false_replacement_rate < 0.02
            ),
        )
        return payload

    max_large_bank_oracle = max(
        values.ideal16_oracle_gain,
        values.ideal64_oracle_gain if _finite(values.ideal64_oracle_gain) else float("-inf"),
    )
    max_large_bank_selected = max(
        values.ideal16_selected_gain,
        values.ideal64_selected_gain if _finite(values.ideal64_selected_gain) else float("-inf"),
    )
    if (
        max_large_bank_oracle < 0.002
        and max_large_bank_selected < 0.001
        and values.target_available_rate < 0.10
        and values.ranking_gap < 0.002
    ):
        payload.update(
            verdict="STOP_LORA",
            triggered_rule="Verdict 6",
            recommended_output="no generator LoRA",
            frozen_scorer_safe=True,
        )
        return payload

    boundary = []
    checks = {
        "ideal8_oracle_gain_vs_0.003": values.ideal8_oracle_gain - 0.003,
        "ideal8_oracle_gain_vs_0.005": values.ideal8_oracle_gain - 0.005,
        "ideal8_oracle_ci_low_vs_0": values.ideal8_oracle_ci_low,
        "ideal8_selected_gain_vs_0.003": values.ideal8_selected_gain - 0.003,
        "duplicate_abs_vs_0.001": abs(values.duplicate_shift) - 0.001,
        "false_replacement_vs_0.02": values.saturated_false_replacement_rate - 0.02,
    }
    for name, distance in checks.items():
        if abs(distance) <= 0.001:
            boundary.append({"statistic": name, "distance": distance})
    payload.update(
        verdict="INCONCLUSIVE",
        triggered_rule="no pre-registered verdict rule was completely satisfied",
        boundary_statistics=boundary,
        minimum_followup="increase held-out physical logs for the statistic nearest its threshold",
    )
    return payload
