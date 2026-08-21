"""Policy-local action-effect research utilities.

The package is deliberately separate from :mod:`starVLA`: importing it does
not alter the baseline model, dataset, checkpoint, or evaluator paths.
"""

from .candidate_generator import (
    CandidateGeneratorConfig,
    KinematicLimits,
    PolicyLocalCandidate,
    PolicyLocalCandidateGenerator,
)

__all__ = [
    "CandidateGeneratorConfig",
    "KinematicLimits",
    "PolicyLocalCandidate",
    "PolicyLocalCandidateGenerator",
]
