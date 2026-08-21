"""Leakage-safe extraction of frozen Qwen scene/action-query features.

The extractor mirrors the checkpoint's existing Qwen inference path but stops
before flow sampling.  Candidate or expert future trajectories are never
passed to Qwen; they remain separate world-probe conditions/targets.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from research.action_effect.data_contract import assert_no_future_leakage


CURRENT_OBSERVATION_FIELDS = frozenset({"image", "state", "lang", "token"})


def sanitize_current_observation(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Whitelist runtime-available fields and reject incomplete observations.

    In particular, the dataset's ``action`` field is an expert-future label and
    is deliberately discarded before the frozen backbone is called.
    """

    missing = {"image", "state", "lang", "token"} - set(sample)
    if missing:
        raise KeyError(f"current observation is missing required fields: {sorted(missing)}")
    result = {name: sample[name] for name in CURRENT_OBSERVATION_FIELDS}
    assert_no_future_leakage(result)
    if "action" in result:
        raise AssertionError("expert future action reached frozen scene input")
    return result


def _autocast_for(model: torch.nn.Module):
    device = next(model.parameters()).device
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.inference_mode()
def extract_qwen_scene_features(
    model: torch.nn.Module,
    examples: Sequence[Mapping[str, Any]],
    *,
    qwen_forward_mode: str,
) -> dict[str, torch.Tensor]:
    """Return Qwen action-query tokens without invoking DiT flow sampling.

    Args:
        model: Loaded, frozen ``Qwenvl_OFT`` baseline.
        examples: Sanitized current observations only.
        qwen_forward_mode: ``optimized`` or ``legacy`` as resolved for the
            checkpoint by the existing inference wrapper.

    Returns:
        ``scene_tokens`` with shape ``[B, 8, 2048]`` and the unchanged action
        head projection ``action_hidden`` with shape ``[B, 8, 1536]`` for the
        audited 100k baseline.
    """

    if qwen_forward_mode not in {"optimized", "legacy"}:
        raise ValueError("qwen_forward_mode must be optimized or legacy")
    safe_examples = [sanitize_current_observation(example) for example in examples]
    instructions = [example["lang"] + model._build_action_prompt_suffix() for example in safe_examples]
    (
        input_ids,
        attention_mask,
        position_ids,
        token_positions,
        image_embeds,
        deepstack_embeds,
        image_grid_thw,
    ) = model._build_qwen_batch(safe_examples, instructions)

    with _autocast_for(model):
        text_embeds = model.qwen_vl_interface.model.get_input_embeddings()(input_ids)

    state_device = next(model.action_input_model.parameters()).device
    states = torch.as_tensor(
        np.asarray([example["state"] for example in safe_examples]),
        dtype=torch.float32,
        device=state_device,
    )[:, 0, :]
    state_embeds = model.action_input_model(states).to(
        dtype=text_embeds.dtype, device=text_embeds.device
    )
    batch_indices = torch.arange(len(safe_examples), device=text_embeds.device)
    text_embeds[batch_indices, token_positions["history"][:, 0], :] = state_embeds

    with _autocast_for(model):
        if qwen_forward_mode == "optimized":
            last_hidden = model._qwen_language_forward(
                input_ids=input_ids,
                inputs_embeds=text_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                image_embeds=image_embeds,
                deepstack_embeds=deepstack_embeds,
            )
        else:
            # This branch exists for older checkpoints whose saved training
            # source used the full Qwen wrapper and post-norm hidden state.
            qwen_inputs = model.qwen_vl_interface.build_qwenvl_inputs(
                images=[example["image"] for example in safe_examples],
                instructions=instructions,
            )
            output = model.qwen_vl_interface(
                inputs_embeds=text_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                pixel_values=qwen_inputs.get("pixel_values"),
                image_grid_thw=qwen_inputs.get("image_grid_thw"),
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = output.hidden_states[-1]

    hidden_size = int(last_hidden.shape[-1])
    gather_index = token_positions["action"].unsqueeze(-1).expand(-1, -1, hidden_size)
    scene_tokens = last_hidden.gather(dim=1, index=gather_index)
    scene_tokens, action_context, _ = model._condition_inference_action_queries(
        last_hidden,
        input_ids,
        scene_tokens,
        image_grid_thw=image_grid_thw,
        examples=safe_examples,
    )
    if action_context is not None:
        raise RuntimeError(
            "the selected baseline has an auxiliary action context; use a "
            "framework-specific scene-cache adapter instead of silently dropping it"
        )
    action_hidden = model.action_model.qwen_proj(scene_tokens.float())
    if scene_tokens.ndim != 3 or action_hidden.ndim != 3:
        raise RuntimeError("scene feature tensors must have shape [batch, query, channel]")
    return {
        "scene_tokens": scene_tokens.detach().float(),
        "action_hidden": action_hidden.detach().float(),
    }
