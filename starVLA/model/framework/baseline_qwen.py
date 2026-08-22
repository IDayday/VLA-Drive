"""Shared Qwen-OFT feature extraction and trainability policy.

The local baseline is the ``QwenOFT`` framework launched by ``8-train.sh``.
This module centralizes the parts that the baseline-matched hierarchical model
must reuse verbatim: special-token discovery, cached/uncached visual packing,
the language-only Qwen forward, action-token gathering, and the configured
``trainer.freeze_modules`` boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class BaselineQwenTrainability:
    """Exact named-parameter manifests after applying the baseline policy."""

    trainable_names: frozenset[str]
    frozen_names: frozenset[str]
    qwen_freeze_paths: tuple[str, ...]


def get_trainable_parameter_names(module: nn.Module) -> set[str]:
    """Return the complete names of trainable parameters (not only a count)."""

    return {
        name
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


def get_frozen_parameter_names(module: nn.Module) -> set[str]:
    """Return the complete names of frozen parameters (not only a count)."""

    return {
        name
        for name, parameter in module.named_parameters()
        if not parameter.requires_grad
    }


def _configured_freeze_paths(config) -> tuple[str, ...]:
    trainer = getattr(config, "trainer", None)
    raw = trainer.get("freeze_modules", "") if trainer is not None else ""
    if not isinstance(raw, str):
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _resolve_child(module: nn.Module, path: str) -> nn.Module:
    current = module
    for component in path.split(".") if path else ():
        current = getattr(current, component)
    if not isinstance(current, nn.Module):
        raise TypeError(f"configured Qwen freeze path is not a module: {path}")
    return current


def apply_baseline_qwen_trainability(
    qwen_vl_interface: nn.Module,
    config,
) -> BaselineQwenTrainability:
    """Apply exactly the local QwenOFT ``freeze_modules`` policy.

    This helper intentionally changes only ``requires_grad``.  It does not call
    ``eval()``, add ``no_grad()``, detach hidden states, enable checkpointing,
    or otherwise create a training-mode policy that the baseline does not use.
    Tied parameters are handled by PyTorch itself: freezing Qwen's ``lm_head``
    also freezes its tied input-embedding parameter.
    """

    for parameter in qwen_vl_interface.parameters():
        parameter.requires_grad_(True)

    qwen_paths = []
    prefix = "qwen_vl_interface"
    for full_path in _configured_freeze_paths(config):
        if full_path == prefix:
            relative_path = ""
        elif full_path.startswith(prefix + "."):
            relative_path = full_path[len(prefix) + 1 :]
        else:
            continue
        try:
            frozen_module = _resolve_child(qwen_vl_interface, relative_path)
        except AttributeError as error:
            raise AttributeError(
                f"baseline Qwen freeze path does not exist: {full_path}"
            ) from error
        for parameter in frozen_module.parameters():
            parameter.requires_grad_(False)
        qwen_paths.append(full_path)

    trainable = get_trainable_parameter_names(qwen_vl_interface)
    frozen = get_frozen_parameter_names(qwen_vl_interface)
    return BaselineQwenTrainability(
        trainable_names=frozenset(trainable),
        frozen_names=frozenset(frozen),
        qwen_freeze_paths=tuple(qwen_paths),
    )


def find_token_positions(input_ids: Tensor, token_ids: Sequence[int]) -> Tensor:
    """Find ordered special-token positions without CUDA scalar synchronizes."""

    ids = torch.as_tensor(token_ids, device=input_ids.device, dtype=input_ids.dtype)
    matches = input_ids.unsqueeze(-1).eq(ids.view(1, 1, -1))
    return matches.to(torch.int8).argmax(dim=1)


def build_baseline_qwen_batch(
    qwen_vl_interface: nn.Module,
    examples: Sequence[Mapping],
    instructions: Sequence[str],
    special_token_ids: Mapping[str, Sequence[int]],
):
    """Build the cached or ordinary Qwen input used by local ``QwenOFT``."""

    cached = [example.get("qwen_feature_cache") for example in examples]
    if any(payload is not None for payload in cached) and not all(
        payload is not None for payload in cached
    ):
        raise RuntimeError("A batch cannot mix cached and uncached Qwen samples")

    device = qwen_vl_interface.model.device
    if all(payload is not None for payload in cached):
        lengths = [int(payload["input_ids"].numel()) for payload in cached]
        max_length = max(lengths)
        batch_size = len(cached)
        tokenizer = qwen_vl_interface.processor.tokenizer
        input_ids = torch.full(
            (batch_size, max_length),
            int(tokenizer.pad_token_id),
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_length), dtype=torch.long, device=device
        )
        position_ids = torch.ones(
            (3, batch_size, max_length), dtype=torch.long, device=device
        )
        position_names = ("history", "rgb", "gs", "action", "reward")
        positions = {name: [] for name in position_names}
        for batch_index, (payload, length) in enumerate(zip(cached, lengths)):
            offset = max_length - length
            input_ids[batch_index, offset:] = payload["input_ids"].to(
                device=device, dtype=torch.long
            )
            attention_mask[batch_index, offset:] = payload["attention_mask"].to(
                device=device, dtype=torch.long
            )
            position_ids[:, batch_index, offset:] = payload["position_ids"].to(
                device=device, dtype=torch.long
            )
            for name in position_names:
                positions[name].append(
                    payload[f"{name}_positions"].to(
                        device=device, dtype=torch.long
                    )
                    + offset
                )

        deepstack_keys = sorted(
            key for key in cached[0] if key.startswith("deepstack_")
        )
        image_embeds = torch.cat(
            [payload["image_embeds"] for payload in cached], dim=0
        )
        deepstack_embeds = [
            torch.cat([payload[key] for payload in cached], dim=0)
            for key in deepstack_keys
        ]
        return (
            input_ids,
            attention_mask,
            position_ids,
            {name: torch.stack(values) for name, values in positions.items()},
            image_embeds.to(device, non_blocking=True),
            [value.to(device, non_blocking=True) for value in deepstack_embeds],
        )

    batch_images = [example["image"] for example in examples]
    qwen_inputs = qwen_vl_interface.build_qwenvl_inputs(
        images=batch_images,
        instructions=instructions,
    )
    input_ids = qwen_inputs["input_ids"]
    attention_mask = qwen_inputs["attention_mask"]
    with torch.no_grad():
        position_ids, _ = qwen_vl_interface.model.model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=qwen_inputs["image_grid_thw"],
            video_grid_thw=qwen_inputs.get("video_grid_thw", None),
            attention_mask=attention_mask,
        )
        image_parts, deepstack_embeds = (
            qwen_vl_interface.model.model.get_image_features(
                qwen_inputs["pixel_values"],
                qwen_inputs["image_grid_thw"],
            )
        )
        image_embeds = torch.cat(image_parts, dim=0)
    positions = {
        name: find_token_positions(input_ids, token_ids)
        for name, token_ids in special_token_ids.items()
    }
    return (
        input_ids,
        attention_mask,
        position_ids,
        positions,
        image_embeds,
        deepstack_embeds,
    )


def baseline_qwen_language_forward(
    qwen_vl_interface: nn.Module,
    *,
    input_ids: Tensor,
    inputs_embeds: Tensor,
    attention_mask: Tensor,
    position_ids: Tensor,
    image_embeds: Tensor,
    deepstack_embeds: Sequence[Tensor],
) -> Tensor:
    """Run the baseline trainable Qwen backbone while skipping its LM head."""

    image_mask = input_ids.eq(qwen_vl_interface.model.config.image_token_id)
    expanded_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
    image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
    inputs_embeds = inputs_embeds.masked_scatter(expanded_mask, image_embeds)
    outputs = qwen_vl_interface.model.model.language_model(
        input_ids=None,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        visual_pos_masks=image_mask,
        deepstack_visual_embeds=[
            value.to(inputs_embeds.device, inputs_embeds.dtype)
            for value in deepstack_embeds
        ],
        use_cache=False,
    )
    return outputs.last_hidden_state


def extract_baseline_action_conditions(
    last_hidden: Tensor,
    action_positions: Tensor,
) -> Tensor:
    """Gather the eight action-token states used by the local Qwen+DiT baseline."""

    if last_hidden.ndim != 3 or action_positions.ndim != 2:
        raise ValueError("action extraction expects [B,L,H] hidden and [B,T] positions")
    if last_hidden.shape[0] != action_positions.shape[0]:
        raise ValueError("action positions and Qwen hidden states have different batches")
    gather_index = action_positions.unsqueeze(-1).expand(
        -1, -1, last_hidden.shape[-1]
    )
    return last_hidden.gather(dim=1, index=gather_index)
