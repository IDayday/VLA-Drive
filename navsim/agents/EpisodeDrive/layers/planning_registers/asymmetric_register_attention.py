"""Read-only planning-register attention for the local InternViT runtime.

The implementation deliberately preserves the original attention module and
state-dict topology. Only its Python ``forward`` is bound to a mask-aware
implementation; Q/V LoRA can therefore continue to wrap ``attention.qkv``.
"""

from __future__ import annotations

from types import MethodType
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


def _read_only_attention_eager(
    attention: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    register_stop: int,
) -> torch.Tensor:
    """Reference implementation with the audited explicit asymmetric mask."""
    token_count = query.shape[-2]
    attention_logits = (query * attention.scale) @ key.transpose(-2, -1)
    blocked = torch.zeros(
        token_count,
        token_count,
        dtype=torch.bool,
        device=query.device,
    )
    blocked[0, 1:register_stop] = True
    blocked[register_stop:, 1:register_stop] = True
    attention_logits = attention_logits.masked_fill(
        blocked[None, None], float("-inf")
    )
    attention_probabilities = attention_logits.softmax(dim=-1)
    attention_probabilities = attention.attn_drop(attention_probabilities)
    return attention_probabilities @ value


def _read_only_attention_split_sdpa(
    attention: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    register_stop: int,
) -> torch.Tensor:
    """Equivalent asymmetric attention without a quadratic explicit mask.

    Register query rows attend the complete sequence. CLS/patch query rows are
    evaluated against a K/V sequence with register columns physically removed.
    This preserves the read-only graph while allowing PyTorch's fused SDPA
    kernels and avoiding the large masked probability tensor.
    """
    non_register_query = torch.cat(
        (query[:, :, :1], query[:, :, register_stop:]), dim=2
    )
    non_register_key = torch.cat(
        (key[:, :, :1], key[:, :, register_stop:]), dim=2
    )
    non_register_value = torch.cat(
        (value[:, :, :1], value[:, :, register_stop:]), dim=2
    )
    dropout_p = (
        float(attention.attn_drop.p)
        if bool(getattr(attention.attn_drop, "training", False))
        else 0.0
    )
    non_register_output = F.scaled_dot_product_attention(
        non_register_query,
        non_register_key,
        non_register_value,
        dropout_p=dropout_p,
        scale=float(attention.scale),
    )
    register_output = F.scaled_dot_product_attention(
        query[:, :, 1:register_stop],
        key,
        value,
        dropout_p=dropout_p,
        scale=float(attention.scale),
    )
    return torch.cat(
        (
            non_register_output[:, :, :1],
            register_output,
            non_register_output[:, :, 1:],
        ),
        dim=2,
    )


def _read_only_attention_forward(attention: nn.Module, hidden_states: torch.Tensor):
    configured_length = getattr(
        attention, "_planreg_read_only_sequence_length", None
    )
    if configured_length is None or hidden_states.shape[1] != configured_length:
        # Direct legacy InternViT calls do not contain register tokens.
        if bool(getattr(attention, "use_flash_attn", False)):
            return attention._flash_attn(hidden_states)
        return attention._naive_attn(hidden_states)

    if bool(getattr(attention, "use_flash_attn", False)):
        raise RuntimeError(
            "planning_registers.attention_mode=read_only does not support "
            "FlashAttention; set vlm_config.use_flash_attn=false"
        )

    batch_size, token_count, hidden_dim = hidden_states.shape
    num_heads = int(attention.num_heads)
    head_dim = hidden_dim // num_heads
    qkv = (
        attention.qkv(hidden_states)
        .reshape(batch_size, token_count, 3, num_heads, head_dim)
        .permute(2, 0, 3, 1, 4)
    )
    query, key, value = qkv.unbind(0)

    if bool(getattr(attention, "qk_normalization", False)):
        query = (
            attention.q_norm(query.transpose(1, 2).flatten(-2, -1))
            .view(batch_size, token_count, num_heads, head_dim)
            .transpose(1, 2)
        )
        key = (
            attention.k_norm(key.transpose(1, 2).flatten(-2, -1))
            .view(batch_size, token_count, num_heads, head_dim)
            .transpose(1, 2)
        )

    register_count = int(attention._planreg_read_only_num_registers)
    register_stop = 1 + register_count
    if token_count <= register_stop:
        raise RuntimeError(
            "Read-only register attention received no patch tokens: "
            f"tokens={token_count}, registers={register_count}"
        )

    backend = str(getattr(attention, "_planreg_read_only_backend", "eager"))
    if backend == "eager":
        output = _read_only_attention_eager(
            attention, query, key, value, register_stop
        )
    elif backend == "split_sdpa":
        output = _read_only_attention_split_sdpa(
            attention, query, key, value, register_stop
        )
    else:  # Defensive: configuration validates this before binding.
        raise RuntimeError(f"Unknown read-only attention backend: {backend!r}")
    output = output.transpose(1, 2).reshape(
        batch_size, token_count, hidden_dim
    )
    output = attention.proj(output)
    return attention.proj_drop(output)


def configure_read_only_register_attention(
    vision_model: nn.Module,
    num_registers: int,
    backend: str = "eager",
) -> Tuple[str, ...]:
    """Bind read-only attention to every confirmed InternViT encoder layer."""
    if num_registers <= 0:
        raise ValueError("num_registers must be positive")
    if backend not in {"eager", "split_sdpa"}:
        raise ValueError(
            "read-only attention backend must be eager or split_sdpa, "
            f"got {backend!r}"
        )
    config = getattr(vision_model, "config", None)
    if bool(getattr(config, "use_flash_attn", False)):
        raise RuntimeError(
            "planning_registers.attention_mode=read_only requires "
            "vlm_config.use_flash_attn=false"
        )
    layers = getattr(getattr(vision_model, "encoder", None), "layers", None)
    if layers is None or len(layers) == 0:
        raise RuntimeError(
            "Read-only registers require vision_model.encoder.layers"
        )

    configured = []
    required = (
        "qkv",
        "num_heads",
        "scale",
        "attn_drop",
        "proj",
        "proj_drop",
        "_naive_attn",
    )
    for index, block in enumerate(layers):
        attention = getattr(block, "attn", None)
        missing = [name for name in required if not hasattr(attention, name)]
        if missing:
            raise RuntimeError(
                "Unsupported InternViT attention structure at "
                f"encoder.layers.{index}.attn; missing={missing}"
            )
        if bool(getattr(attention, "use_flash_attn", False)):
            raise RuntimeError(
                "Read-only registers found an active FlashAttention block at "
                f"encoder.layers.{index}.attn"
            )
        attention._planreg_read_only_num_registers = int(num_registers)
        attention._planreg_read_only_backend = str(backend)
        attention._planreg_read_only_sequence_length = None
        if not bool(getattr(attention, "_planreg_read_only_installed", False)):
            attention.forward = MethodType(_read_only_attention_forward, attention)
            attention._planreg_read_only_installed = True
        configured.append(f"encoder.layers.{index}.attn")
    return tuple(configured)


def set_read_only_register_sequence_length(
    vision_model: nn.Module,
    sequence_length: int,
) -> None:
    """Declare the register-bearing length used by checkpoint recomputation."""
    layers = vision_model.encoder.layers
    for index, block in enumerate(layers):
        attention = getattr(block, "attn", None)
        if not bool(getattr(attention, "_planreg_read_only_installed", False)):
            raise RuntimeError(
                "Read-only attention was not configured for "
                f"encoder.layers.{index}.attn"
            )
        attention._planreg_read_only_sequence_length = int(sequence_length)
