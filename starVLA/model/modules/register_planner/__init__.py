"""Deterministic Register1/Register64 trajectory generation components."""

from .checkpoint import (
    REGISTER_CHECKPOINT_SCHEMA_VERSION,
    load_register_generator_checkpoint,
    load_stage_component_checkpoint,
    parameter_manifest_hash,
    save_register_generator_checkpoint,
    save_stage_component_checkpoint,
    sha256_file,
    stable_config_hash,
    trainable_manifest_hash,
)
from .clover_losses import (
    CloverStage1TrajectoryLoss,
    CloverStage2GeneratorLoss,
    TeacherTargetSets,
    build_teacher_target_sets,
    clover_inter_trajectory_loss,
    selected_set_enrichment,
    selected_set_enrichment_per_scene,
    set_coverage_l1,
)
from .decoder import (
    DropPath,
    LayerScale,
    RegisterAttention,
    RegisterDecoderBlock,
    RegisterMLP,
    RegisterTrajectoryDecoder,
)
from .generator import ProposalHead, RegisterTrajectoryGenerator
from .losses import RegisterTrajectoryLoss
from .outputs import RegisterGeneratorOutput, RegisterLossOutput, RegisterPlannerOutput
from .sanitization import (
    REGISTER_XY_LIMIT_METERS,
    TrajectorySanitizationStats,
    sanitize_register_trajectories,
)

__all__ = [
    "DropPath",
    "LayerScale",
    "ProposalHead",
    "REGISTER_CHECKPOINT_SCHEMA_VERSION",
    "RegisterAttention",
    "RegisterDecoderBlock",
    "RegisterGeneratorOutput",
    "RegisterLossOutput",
    "RegisterMLP",
    "RegisterPlannerOutput",
    "RegisterTrajectoryDecoder",
    "RegisterTrajectoryGenerator",
    "RegisterTrajectoryLoss",
    "REGISTER_XY_LIMIT_METERS",
    "TrajectorySanitizationStats",
    "CloverStage1TrajectoryLoss",
    "CloverStage2GeneratorLoss",
    "TeacherTargetSets",
    "build_teacher_target_sets",
    "clover_inter_trajectory_loss",
    "selected_set_enrichment",
    "selected_set_enrichment_per_scene",
    "sanitize_register_trajectories",
    "set_coverage_l1",
    "load_register_generator_checkpoint",
    "load_stage_component_checkpoint",
    "parameter_manifest_hash",
    "save_register_generator_checkpoint",
    "save_stage_component_checkpoint",
    "sha256_file",
    "stable_config_hash",
    "trainable_manifest_hash",
]
