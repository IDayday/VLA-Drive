"""Small contracts for opt-in Qwen visual-backbone fine-tuning."""

from __future__ import annotations

from contextlib import nullcontext
from functools import partial
from typing import Any

from omegaconf import OmegaConf
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def configure_qwen_visual_backbone(
    visual: nn.Module,
    *,
    freeze_visual: bool,
    gradient_checkpointing: bool,
) -> int:
    """Configure a Qwen visual module and return its trainable parameter count."""

    use_checkpointing = bool(gradient_checkpointing and not freeze_visual)
    visual.requires_grad_(not freeze_visual)
    if not hasattr(visual, "gradient_checkpointing"):
        raise AttributeError(
            "Qwen visual backbone has no gradient_checkpointing attribute"
        )
    visual.gradient_checkpointing = use_checkpointing
    return sum(parameter.numel() for parameter in visual.parameters() if parameter.requires_grad)


def encode_qwen_images(
    qwen_model: nn.Module,
    *,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    freeze_visual: bool,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Encode images as ``[N,C]`` tokens with an explicit gradient policy."""

    assert pixel_values.ndim >= 2, "pixel_values must have at least two dimensions"
    assert image_grid_thw.ndim == 2 and image_grid_thw.shape[-1] == 3, (
        "image_grid_thw must be [N,3]"
    )
    gradient_context = torch.no_grad() if freeze_visual else nullcontext()
    with gradient_context:
        image_parts, deepstack_embeds = qwen_model.get_image_features(
            pixel_values,
            image_grid_thw,
        )
        if not image_parts:
            raise RuntimeError("Qwen visual encoder returned no image features")
        image_embeds = torch.cat(tuple(image_parts), dim=0)

    assert image_embeds.ndim == 2, "Qwen image embeddings must be [N,C]"
    if not isinstance(deepstack_embeds, (tuple, list)):
        raise TypeError("Qwen deepstack embeddings must be a list or tuple")
    deepstack = list(deepstack_embeds)
    for index, value in enumerate(deepstack):
        assert value.ndim == 2, f"deepstack[{index}] must be [N,C]"
    if not freeze_visual and torch.is_grad_enabled() and not image_embeds.requires_grad:
        raise RuntimeError(
            "Qwen visual fine-tuning is enabled, but image embeddings have no gradient graph"
        )
    return image_embeds, deepstack


def run_visual_block(
    block: nn.Module,
    hidden_states: torch.Tensor,
    *,
    checkpoint_enabled: bool,
    **kwargs: Any,
) -> torch.Tensor:
    """Run one vision block, checkpointing only in opt-in gradient mode."""

    if checkpoint_enabled and torch.is_grad_enabled():
        return checkpoint(
            partial(block, **kwargs),
            hidden_states,
            use_reentrant=False,
        )
    return block(hidden_states, **kwargs)


def validate_visual_action_only_config(config) -> None:
    """Fail fast unless a config is a pure action-only visual-tuning run."""

    checks = {
        "framework.name must be QwenOFT": str(
            OmegaConf.select(config, "framework.name", default="")
        )
        == "QwenOFT",
        "framework.action_prompt_mode must be minimal": str(
            OmegaConf.select(config, "framework.action_prompt_mode", default="")
        )
        == "minimal",
        "framework.qwenvl.freeze_visual must be false": not bool(
            OmegaConf.select(config, "framework.qwenvl.freeze_visual", default=True)
        ),
        "datasets.vla_data.load_act_data must be 1": bool(
            OmegaConf.select(config, "datasets.vla_data.load_act_data", default=False)
        ),
        "datasets.video_data.load_2d_data must be 0": not bool(
            OmegaConf.select(config, "datasets.video_data.load_2d_data", default=False)
        ),
        "datasets.gs_data.load_3d_data must be 0": not bool(
            OmegaConf.select(config, "datasets.gs_data.load_3d_data", default=False)
        ),
        "datasets.reward_data.load_reward_data must be 0": not bool(
            OmegaConf.select(config, "datasets.reward_data.load_reward_data", default=False)
        ),
        "w_depth must be 0": not bool(OmegaConf.select(config, "w_depth", default=False)),
        "VGGT must be disabled": not bool(
            OmegaConf.select(config, "framework.vggt.enabled", default=False)
        ),
    }
    failures = [message for message, passed in checks.items() if not passed]
    freeze_modules = str(
        OmegaConf.select(config, "trainer.freeze_modules", default="")
    )
    frozen = {value.strip() for value in freeze_modules.split(",") if value.strip()}
    if "qwen_vl_interface.model.model.visual" in frozen or "qwen_vl_interface.model.visual" in frozen:
        failures.append("trainer.freeze_modules must not contain the Qwen visual backbone")
    if "qwen_vl_interface.model.lm_head" not in frozen:
        failures.append("the unused Qwen lm_head must be frozen")
    if failures:
        raise ValueError("Invalid Qwen visual action-only experiment:\n- " + "\n- ".join(failures))
