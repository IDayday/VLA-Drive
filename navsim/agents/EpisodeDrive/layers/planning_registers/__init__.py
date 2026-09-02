"""Planning-register adapters for vision backbones.

V1 provides the explicit InternViT implementation. A future Qwen3-VL adapter
can implement :class:`PlanningRegisterAdapter` without changing downstream
scene-fusion or world-model interfaces.
"""

from .internvl_planning_registers import (
    InternVLPlanningOutput,
    InternVLPlanningRegisters,
    PlanningRegisterAdapter,
)
from .register_diagnostics import compute_register_diagnostics
from .asymmetric_register_attention import (
    configure_read_only_register_attention,
    set_read_only_register_sequence_length,
)
from .vision_qv_lora import (
    InternViTQVLoRALinear,
    extract_qv_lora_state_dict,
    freeze_vision_except_qv_lora,
    inject_internvit_qv_lora,
    iter_qv_lora_modules,
    load_qv_lora_state_dict,
)

__all__ = [
    "InternVLPlanningOutput",
    "InternVLPlanningRegisters",
    "InternViTQVLoRALinear",
    "PlanningRegisterAdapter",
    "compute_register_diagnostics",
    "configure_read_only_register_attention",
    "extract_qv_lora_state_dict",
    "freeze_vision_except_qv_lora",
    "inject_internvit_qv_lora",
    "iter_qv_lora_modules",
    "load_qv_lora_state_dict",
    "set_read_only_register_sequence_length",
]
