"""Deterministic Register1/Register64 trajectory generation components."""

from .checkpoint import (
    REGISTER_CHECKPOINT_SCHEMA_VERSION,
    load_register_generator_checkpoint,
    load_stage_component_checkpoint,
    save_register_generator_checkpoint,
    save_stage_component_checkpoint,
    sha256_file,
    stable_config_hash,
    trainable_manifest_hash,
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
    "load_register_generator_checkpoint",
    "load_stage_component_checkpoint",
    "save_register_generator_checkpoint",
    "save_stage_component_checkpoint",
    "sha256_file",
    "stable_config_hash",
    "trainable_manifest_hash",
]
