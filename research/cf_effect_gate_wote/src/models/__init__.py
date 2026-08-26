"""Small frozen-backbone models used by the counterfactual-effect Gate."""

from .probe_heads import (
    COMMON_INPUT_DIM,
    MatchedCapacityFactorProbe,
    MatchedInputComposer,
    factorized_probe_loss,
)

__all__ = [
    "COMMON_INPUT_DIM",
    "MatchedCapacityFactorProbe",
    "MatchedInputComposer",
    "factorized_probe_loss",
]
