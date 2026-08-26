"""Explicit contracts for opt-in Qwen visual-backbone fine-tuning."""

from __future__ import annotations

from contextlib import nullcontext
from functools import partial
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def configure_qwen_visual_backbone(
    visual: nn.Module,
    *,
    freeze_visual: bool,
    gradient_checkpointing: bool,
) -> int:
    """Configure the visual module and return its trainable parameter count."""

    use_checkpointing = bool(gradient_checkpointing and not freeze_visual)
    visual.requires_grad_(not freeze_visual)
    if not hasattr(visual, "gradient_checkpointing"):
        raise AttributeError(
            "Qwen visual backbone has no gradient_checkpointing attribute"
        )
    visual.gradient_checkpointing = use_checkpointing
    return sum(
        parameter.numel()
        for parameter in visual.parameters()
        if parameter.requires_grad
    )


def encode_qwen_images(
    qwen_model: nn.Module,
    *,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    freeze_visual: bool,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Encode Qwen images under an explicit frozen/trainable gradient policy."""

    if pixel_values.ndim < 2:
        raise ValueError("pixel_values must have at least two dimensions")
    if image_grid_thw.ndim != 2 or image_grid_thw.shape[-1] != 3:
        raise ValueError("image_grid_thw must have shape [N,3]")
    gradient_context = torch.no_grad() if freeze_visual else nullcontext()
    with gradient_context:
        image_parts, deepstack_embeds = qwen_model.get_image_features(
            pixel_values,
            image_grid_thw,
        )
        if not image_parts:
            raise RuntimeError("Qwen visual encoder returned no image features")
        image_embeds = torch.cat(tuple(image_parts), dim=0)

    if image_embeds.ndim != 2:
        raise RuntimeError("Qwen image embeddings must have shape [N,C]")
    if not isinstance(deepstack_embeds, (tuple, list)):
        raise TypeError("Qwen deepstack embeddings must be a list or tuple")
    deepstack = list(deepstack_embeds)
    for index, value in enumerate(deepstack):
        if value.ndim != 2:
            raise RuntimeError(f"Qwen deepstack[{index}] must have shape [N,C]")
    if not freeze_visual and torch.is_grad_enabled() and not image_embeds.requires_grad:
        raise RuntimeError(
            "Qwen visual fine-tuning is enabled, but image embeddings have no "
            "gradient graph"
        )
    return image_embeds, deepstack


def run_visual_block(
    block: nn.Module,
    hidden_states: torch.Tensor,
    *,
    checkpoint_enabled: bool,
    **kwargs: Any,
) -> torch.Tensor:
    """Run one vision block, checkpointing only during opt-in training."""

    if checkpoint_enabled and torch.is_grad_enabled():
        return checkpoint(
            partial(block, **kwargs),
            hidden_states,
            use_reentrant=False,
        )
    return block(hidden_states, **kwargs)
