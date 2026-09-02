"""EMA vision/register teacher used only while training PlanReg-WM-V1."""

from __future__ import annotations

import copy
import math
from typing import List, Optional

import torch
from torch import nn
from pytorch_lightning.callbacks import Callback


def scale_ema_momentum_for_global_batch(
    reference_momentum: float,
    actual_global_batch: int,
    reference_global_batch: int = 16,
) -> float:
    """Preserve per-sample EMA decay when the optimizer batch changes."""
    if not 0.0 <= reference_momentum <= 1.0:
        raise ValueError("reference_momentum must be in [0,1]")
    if actual_global_batch <= 0 or reference_global_batch <= 0:
        raise ValueError("actual/reference global batches must be positive")
    return float(
        reference_momentum ** (actual_global_batch / reference_global_batch)
    )


def cosine_ema_momentum(
    optimizer_step: int,
    total_optimizer_steps: int,
    start: float = 0.996,
    end: float = 0.9999,
) -> float:
    if total_optimizer_steps <= 0:
        raise ValueError("total_optimizer_steps must be positive")
    progress = min(1.0, max(0.0, optimizer_step / total_optimizer_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(end - (end - start) * cosine)


class EMARegisterTarget(nn.Module):
    """Deep-copy only InternViT (including Q/V LoRA) and the register neck."""

    def __init__(self, student_backbone: nn.Module) -> None:
        super().__init__()
        if getattr(student_backbone, "planning_register_adapter", None) is None:
            raise RuntimeError("EMA register target requires a planning-register adapter")
        self.vision_model = copy.deepcopy(student_backbone.model.vision_model)
        self.planning_register_adapter = copy.deepcopy(
            student_backbone.planning_register_adapter
        )
        for parameter in self.parameters():
            parameter.requires_grad = False
        super().train(False)

    def train(self, mode: bool = True):
        # Teacher dropout/stochastic depth must remain disabled.
        return super().train(False)

    @torch.no_grad()
    def forward(
        self,
        pixel_values: torch.Tensor,
        num_patches_list: List[int],
        tile_metadata: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.planning_register_adapter.encode_registers_only(
            self.vision_model,
            pixel_values,
            num_patches_list,
            tile_metadata,
        )

    @torch.no_grad()
    def update(self, student_backbone: nn.Module, momentum: float) -> None:
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"EMA momentum must be in [0,1], got {momentum}")
        student_modules = (
            student_backbone.model.vision_model,
            student_backbone.planning_register_adapter,
        )
        teacher_modules = (self.vision_model, self.planning_register_adapter)
        for teacher_module, student_module in zip(teacher_modules, student_modules):
            teacher_parameters = dict(teacher_module.named_parameters())
            student_parameters = dict(student_module.named_parameters())
            if teacher_parameters.keys() != student_parameters.keys():
                raise RuntimeError("EMA student/teacher parameter topology differs")
            for name, teacher_parameter in teacher_parameters.items():
                student_parameter = student_parameters[name]
                teacher_parameter.mul_(momentum).add_(
                    student_parameter.detach().to(teacher_parameter.dtype),
                    alpha=1.0 - momentum,
                )
            teacher_buffers = dict(teacher_module.named_buffers())
            student_buffers = dict(student_module.named_buffers())
            if teacher_buffers.keys() != student_buffers.keys():
                raise RuntimeError("EMA student/teacher buffer topology differs")
            for name, teacher_buffer in teacher_buffers.items():
                student_buffer = student_buffers[name]
                if teacher_buffer.is_floating_point():
                    teacher_buffer.mul_(momentum).add_(
                        student_buffer.detach().to(teacher_buffer.dtype),
                        alpha=1.0 - momentum,
                    )
                else:
                    teacher_buffer.copy_(student_buffer)
        super().train(False)


class EMARegisterTargetCallback(Callback):
    """Update once after each completed optimizer step, including accumulation."""

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ) -> None:
        agent = getattr(pl_module, "agent", None)
        if agent is not None and hasattr(agent, "update_ema_after_optimizer_step"):
            agent.update_ema_after_optimizer_step(
                optimizer_step=int(trainer.global_step),
                total_optimizer_steps=int(trainer.estimated_stepping_batches),
            )
