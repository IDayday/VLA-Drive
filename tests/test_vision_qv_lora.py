from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from navsim.agents.EpisodeDrive.layers.planning_registers import (
    InternViTQVLoRALinear,
    extract_qv_lora_state_dict,
    freeze_vision_except_qv_lora,
    inject_internvit_qv_lora,
    load_qv_lora_state_dict,
)


class _Attention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3)


class _Block(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.attn = _Attention(dim)
        self.mlp = nn.Linear(dim, dim)


class _Vision(nn.Module):
    def __init__(self, dim: int = 8, layers: int = 4) -> None:
        super().__init__()
        self.encoder = SimpleNamespace()
        self.encoder.layers = nn.ModuleList([_Block(dim) for _ in range(layers)])
        # Register the layers as a real child module while retaining the exact
        # InternViT ``encoder.layers`` access contract.
        encoder = nn.Module()
        encoder.layers = self.encoder.layers
        self.encoder = encoder
        self.other_weight = nn.Parameter(torch.randn(dim))


def test_qv_lora_initial_output_is_exact_and_all_layers_are_injected(capsys) -> None:
    torch.manual_seed(3)
    vision = _Vision()
    inputs = torch.randn(2, 5, 8)
    expected = [block.attn.qkv(inputs).detach() for block in vision.encoder.layers]

    names = inject_internvit_qv_lora(vision, rank=32, dropout=0.0)
    assert len(names) == len(vision.encoder.layers) == 4
    assert "into 4 visual layers" in capsys.readouterr().out
    for block, reference in zip(vision.encoder.layers, expected):
        assert isinstance(block.attn.qkv, InternViTQVLoRALinear)
        torch.testing.assert_close(block.attn.qkv(inputs), reference, rtol=0.0, atol=0.0)
        assert not block.attn.qkv.base_layer.weight.requires_grad


def test_only_q_and_v_change_and_k_has_no_adapter() -> None:
    torch.manual_seed(5)
    vision = _Vision(layers=1)
    inputs = torch.randn(2, 3, 8)
    base_output = vision.encoder.layers[0].attn.qkv(inputs).detach()
    inject_internvit_qv_lora(vision, rank=32)
    qkv = vision.encoder.layers[0].attn.qkv
    nn.init.constant_(qkv.q_lora_b.weight, 0.1)
    nn.init.constant_(qkv.v_lora_b.weight, -0.1)
    adapted = qkv(inputs)
    base_q, base_k, base_v = base_output.chunk(3, dim=-1)
    new_q, new_k, new_v = adapted.chunk(3, dim=-1)

    assert not torch.equal(new_q, base_q)
    torch.testing.assert_close(new_k, base_k, rtol=0.0, atol=0.0)
    assert not torch.equal(new_v, base_v)
    assert not any("k_lora" in name for name, _ in qkv.named_parameters())


def test_freeze_and_qv_lora_state_round_trip() -> None:
    torch.manual_seed(9)
    original = _Vision(layers=2)
    restored = copy.deepcopy(original)
    inject_internvit_qv_lora(original, rank=32)
    inject_internvit_qv_lora(restored, rank=32)
    for block in original.encoder.layers:
        nn.init.normal_(block.attn.qkv.q_lora_b.weight)
        nn.init.normal_(block.attn.qkv.v_lora_b.weight)

    saved = extract_qv_lora_state_dict(original)
    assert len(saved) == 2 * 4
    load_qv_lora_state_dict(restored, saved)
    inputs = torch.randn(2, 4, 8)
    for source_block, restored_block in zip(
        original.encoder.layers, restored.encoder.layers
    ):
        torch.testing.assert_close(
            source_block.attn.qkv(inputs),
            restored_block.attn.qkv(inputs),
        )

    trainable_count = freeze_vision_except_qv_lora(original)
    trainable = {
        name for name, parameter in original.named_parameters() if parameter.requires_grad
    }
    assert trainable_count == sum(
        parameter.numel()
        for parameter in original.parameters()
        if parameter.requires_grad
    )
    assert trainable
    assert all(any(part in name for part in ("q_lora", "v_lora")) for name in trainable)


def test_zero_matching_layers_fails_immediately() -> None:
    vision = nn.Module()
    vision.encoder = nn.Module()
    vision.encoder.layers = nn.ModuleList()
    with pytest.raises(RuntimeError, match="zero"):
        inject_internvit_qv_lora(vision, rank=32)


def test_all_24_formal_blocks_receive_q_and_v_adapter_gradients() -> None:
    torch.manual_seed(17)
    vision = _Vision(dim=8, layers=24)
    inject_internvit_qv_lora(vision, rank=32, dropout=0.0)
    inputs = torch.randn(2, 5, 8)
    loss = sum(block.attn.qkv(inputs).square().mean() for block in vision.encoder.layers)
    loss.backward()
    for block in vision.encoder.layers:
        qkv = block.attn.qkv
        assert qkv.q_lora_b.weight.grad is not None
        assert qkv.v_lora_b.weight.grad is not None
        assert torch.isfinite(qkv.q_lora_b.weight.grad).all()
        assert torch.isfinite(qkv.v_lora_b.weight.grad).all()
