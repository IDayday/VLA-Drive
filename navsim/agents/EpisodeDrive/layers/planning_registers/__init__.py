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
from .vision_qv_lora import (
    InternViTQVLoRALinear,
    extract_qv_lora_state_dict,
    freeze_vision_except_qv_lora,
    inject_internvit_qv_lora,
    load_qv_lora_state_dict,
)

__all__ = [
    "InternVLPlanningOutput",
    "InternVLPlanningRegisters",
    "InternViTQVLoRALinear",
    "PlanningRegisterAdapter",
    "compute_register_diagnostics",
    "extract_qv_lora_state_dict",
    "freeze_vision_except_qv_lora",
    "inject_internvit_qv_lora",
    "load_qv_lora_state_dict",
]
