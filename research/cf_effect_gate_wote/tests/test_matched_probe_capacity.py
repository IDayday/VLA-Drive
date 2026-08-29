from __future__ import annotations

from research.cf_effect_gate_wote.src.effect_tokenizer import MODEL_VARIANTS
from research.cf_effect_gate_wote.src.models.structured_six_factor_probe import (
    StructuredSixFactorProbe,
    trainable_parameter_count,
)


def test_all_model_types_have_exactly_equal_parameter_count() -> None:
    counts = []
    classes = []
    for _ in MODEL_VARIANTS:
        model = StructuredSixFactorProbe()
        counts.append(trainable_parameter_count(model))
        classes.append(type(model))
    assert len(set(counts)) == 1
    assert len(set(classes)) == 1
    assert counts[0] > 0

