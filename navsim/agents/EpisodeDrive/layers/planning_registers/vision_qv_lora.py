"""Q/V-only LoRA surgery for InternViT fused QKV projections.

The Q/V update follows the implementation style used by the original DrivoR
authors in ``dinov2_lora.py``, pinned to DrivoR commit
fc6e5aa144bbcb5a046e22c18f1bd5cf3af8634a. The key branch is deliberately
untouched.
"""

from __future__ import annotations

import math
from typing import Dict, Iterator, List, Mapping, Tuple

import torch
from torch import nn


_LORA_PARAMETER_PARTS = (
    ".q_lora_a.",
    ".q_lora_b.",
    ".v_lora_a.",
    ".v_lora_b.",
)


class InternViTQVLoRALinear(nn.Module):
    """Wrap one fused ``Linear(C, 3C)`` and add low-rank Q/V residuals."""

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(
                "InternViT Q/V LoRA requires attention.qkv to be nn.Linear; "
                f"got {type(base_layer).__module__}.{type(base_layer).__name__}"
            )
        if rank <= 0:
            raise ValueError(f"Q/V LoRA rank must be positive, got {rank}")
        if base_layer.out_features != 3 * base_layer.in_features:
            raise ValueError(
                "InternViT fused qkv must have out_features == 3 * in_features; "
                f"got {base_layer.in_features} -> {base_layer.out_features}"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"Q/V LoRA dropout must be in [0,1), got {dropout}")

        self.base_layer = base_layer
        self.rank = int(rank)
        self.dim = int(base_layer.in_features)
        self.dropout = nn.Dropout(float(dropout))

        factory_kwargs = {
            "device": base_layer.weight.device,
            "dtype": base_layer.weight.dtype,
        }
        self.q_lora_a = nn.Linear(
            self.dim, self.rank, bias=False, **factory_kwargs
        )
        self.q_lora_b = nn.Linear(
            self.rank, self.dim, bias=False, **factory_kwargs
        )
        self.v_lora_a = nn.Linear(
            self.dim, self.rank, bias=False, **factory_kwargs
        )
        self.v_lora_b = nn.Linear(
            self.rank, self.dim, bias=False, **factory_kwargs
        )
        self.reset_parameters()

        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False

    @property
    def in_features(self) -> int:
        return self.dim

    @property
    def out_features(self) -> int:
        return 3 * self.dim

    @property
    def weight(self) -> nn.Parameter:
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.q_lora_a.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.v_lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.q_lora_b.weight)
        nn.init.zeros_(self.v_lora_b.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_qkv = self.base_layer(inputs)
        base_q, base_k, base_v = base_qkv.split(self.dim, dim=-1)
        lora_inputs = self.dropout(inputs)
        q_update = self.q_lora_b(self.q_lora_a(lora_inputs))
        v_update = self.v_lora_b(self.v_lora_a(lora_inputs))
        return torch.cat(
            (base_q + q_update, base_k, base_v + v_update), dim=-1
        )


def iter_qv_lora_modules(
    module: nn.Module,
) -> Iterator[Tuple[str, InternViTQVLoRALinear]]:
    for name, child in module.named_modules():
        if isinstance(child, InternViTQVLoRALinear):
            yield name, child


def _validated_internvit_qkv_layers(vision_model: nn.Module) -> List[Tuple[int, nn.Module, nn.Linear]]:
    encoder = getattr(vision_model, "encoder", None)
    layers = getattr(encoder, "layers", None)
    if layers is None:
        raise RuntimeError(
            "InternViT Q/V LoRA expected vision_model.encoder.layers, but the "
            "loaded trust_remote_code model does not expose it."
        )

    matches: List[Tuple[int, nn.Module, nn.Linear]] = []
    errors: List[str] = []
    for index, block in enumerate(layers):
        attention = getattr(block, "attn", None)
        qkv = getattr(attention, "qkv", None)
        if attention is None or qkv is None:
            errors.append(f"encoder.layers.{index} has no attn.qkv")
        elif isinstance(qkv, InternViTQVLoRALinear):
            errors.append(f"encoder.layers.{index}.attn.qkv is already Q/V-LoRA wrapped")
        elif not isinstance(qkv, nn.Linear):
            errors.append(
                f"encoder.layers.{index}.attn.qkv is "
                f"{type(qkv).__module__}.{type(qkv).__name__}, not nn.Linear"
            )
        elif qkv.out_features != 3 * qkv.in_features:
            errors.append(
                f"encoder.layers.{index}.attn.qkv has shape "
                f"{qkv.in_features}->{qkv.out_features}, expected C->3C"
            )
        else:
            matches.append((index, attention, qkv))

    if errors:
        raise RuntimeError(
            "InternViT Q/V LoRA structure validation failed before injection: "
            + "; ".join(errors)
        )
    if not matches:
        raise RuntimeError(
            "InternViT Q/V LoRA matched zero encoder attention.qkv layers"
        )
    return matches


def inject_internvit_qv_lora(
    vision_model: nn.Module,
    rank: int = 32,
    dropout: float = 0.0,
) -> Tuple[str, ...]:
    """Inject every validated InternViT encoder block atomically."""
    matches = _validated_internvit_qkv_layers(vision_model)
    names: List[str] = []
    for index, attention, qkv in matches:
        attention.qkv = InternViTQVLoRALinear(
            qkv,
            rank=rank,
            dropout=dropout,
        )
        names.append(f"encoder.layers.{index}.attn.qkv")
    print(
        f"Injected InternViT Q/V LoRA (rank={rank}, dropout={dropout}) "
        f"into {len(names)} visual layers."
    )
    return tuple(names)


def freeze_vision_except_qv_lora(vision_model: nn.Module) -> int:
    """Freeze the complete vision model and re-enable only Q/V LoRA weights."""
    for parameter in vision_model.parameters():
        parameter.requires_grad = False
    trainable = 0
    module_count = 0
    for _, module in iter_qv_lora_modules(vision_model):
        module_count += 1
        for branch in (
            module.q_lora_a,
            module.q_lora_b,
            module.v_lora_a,
            module.v_lora_b,
        ):
            for parameter in branch.parameters():
                parameter.requires_grad = True
                trainable += parameter.numel()
    if module_count == 0:
        raise RuntimeError("Cannot enable Q/V LoRA training: no injected layers found")
    return trainable


def extract_qv_lora_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    """Return a standalone, clone-safe Q/V LoRA state dictionary."""
    return {
        key: value.detach().clone()
        for key, value in module.state_dict().items()
        if any(part in f".{key}." for part in _LORA_PARAMETER_PARTS)
    }


def load_qv_lora_state_dict(
    module: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
) -> None:
    """Strictly restore only Q/V LoRA tensors while leaving base weights intact."""
    expected = set(extract_qv_lora_state_dict(module))
    provided = set(state_dict)
    missing = sorted(expected - provided)
    unexpected = sorted(provided - expected)
    if missing or unexpected:
        raise RuntimeError(
            "Invalid Q/V LoRA state_dict: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    incompatible = module.load_state_dict(dict(state_dict), strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"Unexpected Q/V LoRA keys: {incompatible.unexpected_keys}"
        )
