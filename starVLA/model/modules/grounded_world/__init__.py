"""GroundedWorld-VLA modules from the revised research plan."""

from .consequence import (
    CONSEQUENCE_NAMES,
    ConsequencePrediction,
    ConsequenceTargets,
    PlanningConsequenceHead,
    consequence_losses,
)
from .dynamics_memory import CurrentDynamicsEncoder, PredictiveMemoryForecaster
from .core import GroundedWorldCore, GroundedWorldMemoryOutput
from .geometry_memory import MultiScaleGeometryMemoryWriter
from .prior_adapter import ExternalPriorAdapter
from .perturbations import PerturbedTrajectories, build_consequence_perturbations
from .supervision import (
    FeatureAlignmentTarget,
    FutureTargetContract,
    alignment_losses,
    future_prediction_losses,
    global_alignment_losses,
)
from .trajectory_tube_reader import (
    GroundedTubeReadout,
    MultiScaleTrajectoryTubeReader,
)
from .teacher_protocols import (
    CurrentPriorTeacherAdapter,
    CurrentPriorTeacherOutput,
    PhysicalConsequenceOutput,
    PhysicalConsequenceProvider,
)
from .types import (
    CurrentDynamicsMemory,
    MultiScaleGeometryMemory,
    PredictiveWorldMemory,
)

__all__ = [
    "CONSEQUENCE_NAMES",
    "ConsequencePrediction",
    "ConsequenceTargets",
    "CurrentDynamicsEncoder",
    "CurrentDynamicsMemory",
    "CurrentPriorTeacherAdapter",
    "CurrentPriorTeacherOutput",
    "PhysicalConsequenceOutput",
    "PhysicalConsequenceProvider",
    "FeatureAlignmentTarget",
    "ExternalPriorAdapter",
    "FutureTargetContract",
    "GroundedWorldCore",
    "GroundedWorldMemoryOutput",
    "GroundedTubeReadout",
    "MultiScaleGeometryMemory",
    "MultiScaleGeometryMemoryWriter",
    "MultiScaleTrajectoryTubeReader",
    "PlanningConsequenceHead",
    "PredictiveMemoryForecaster",
    "PredictiveWorldMemory",
    "PerturbedTrajectories",
    "alignment_losses",
    "build_consequence_perturbations",
    "consequence_losses",
    "future_prediction_losses",
    "global_alignment_losses",
]
