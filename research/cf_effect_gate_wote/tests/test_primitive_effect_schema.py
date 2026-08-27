from __future__ import annotations

from research.cf_effect_gate_wote.src.oracle_effect_verdict import (
    FORBIDDEN_PRIMITIVE_NAMES,
    primitive_schema,
)
from research.cf_effect_gate_wote.src.replay_effect_builder import (
    ENGINEERED_EFFECT_NAMES,
)


def test_primitive_keys_contain_no_evaluator_factor_or_score() -> None:
    schema = primitive_schema()
    core = {
        name.lower()
        for names in schema["groups"].values()
        for name in names
    }
    assert not (core & FORBIDDEN_PRIMITIVE_NAMES)
    assert "footprint_outside_drivable_ratio" not in core


def test_engineered_proxies_are_quarantined_in_separate_schema() -> None:
    schema = primitive_schema()
    assert schema["engineered_schema_version"] == "engineered_effect.v1"
    assert tuple(schema["engineered_only"]) == ENGINEERED_EFFECT_NAMES

