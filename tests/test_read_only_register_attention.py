from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from navsim.agents.EpisodeDrive.layers.planning_registers import (
    InternVLPlanningRegisters,
    inject_internvit_qv_lora,
)
from navsim.agents.EpisodeDrive.layers.planning_registers.asymmetric_register_attention import (
    configure_read_only_register_attention,
    set_read_only_register_sequence_length,
)


class _Attention(nn.Module):
    def __init__(self, dim: int = 8, heads: int = 2) -> None:
        super().__init__()
        self.num_heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.0)
        self.qk_normalization = False
        self.use_flash_attn = False

    def _naive_attn(self, hidden_states):
        batch, tokens, dim = hidden_states.shape
        qkv = (
            self.qkv(hidden_states)
            .reshape(batch, tokens, 3, self.num_heads, dim // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv.unbind(0)
        probabilities = ((query * self.scale) @ key.transpose(-2, -1)).softmax(-1)
        output = (probabilities @ value).transpose(1, 2).reshape(batch, tokens, dim)
        return self.proj_drop(self.proj(output))

    def forward(self, hidden_states):
        return self._naive_attn(hidden_states)


class _Block(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _Attention(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 16), nn.GELU(), nn.Linear(16, dim))

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states))
        return hidden_states + self.mlp(self.norm2(hidden_states))


class _Encoder(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Block(dim), _Block(dim)])

    def forward(self, inputs_embeds, output_hidden_states=False, return_dict=True):
        hidden = inputs_embeds
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


class _Embeddings(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.cls = nn.Parameter(torch.randn(1, 1, dim))
        self.patch = nn.Linear(3, dim)

    def forward(self, pixels):
        patches = pixels.permute(0, 2, 3, 1).reshape(pixels.shape[0], -1, 3)
        return torch.cat((self.cls.expand(pixels.shape[0], -1, -1), self.patch(patches)), 1)


class _Vision(nn.Module):
    def __init__(self, *, use_flash_attn: bool = False, dim: int = 8) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=dim, use_flash_attn=use_flash_attn)
        self.embeddings = _Embeddings(dim)
        self.encoder = _Encoder(dim)


def test_read_only_patch_rows_match_legacy_with_zero_init_qv_lora() -> None:
    torch.manual_seed(301)
    vision = _Vision()
    legacy = copy.deepcopy(vision).eval()
    inject_internvit_qv_lora(vision, rank=2, dropout=0.0)
    adapter = InternVLPlanningRegisters(
        8,
        num_registers=4,
        register_dim=8,
        attention_mode="read_only",
    )
    vision.eval()
    pixels = torch.randn(2, 3, 2, 2)

    with torch.no_grad():
        legacy_embedded = legacy.embeddings(pixels)
        legacy_encoded = legacy.encoder(
            inputs_embeds=legacy_embedded,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state
    encoded_registers, encoded_patches = adapter._encode_with_registers(
        vision, pixels
    )
    max_abs_diff = (encoded_patches - legacy_encoded[:, 1:]).abs().max()
    assert max_abs_diff.item() <= 1e-5
    assert torch.isfinite(encoded_registers).all()

    encoded_registers.square().mean().backward()
    assert adapter.planning_registers.grad is not None
    assert adapter.planning_registers.grad.norm().item() > 0
    qv_grad = sum(
        module.q_lora_b.weight.grad.norm().item()
        + module.v_lora_b.weight.grad.norm().item()
        for block in vision.encoder.layers
        for module in [block.attn.qkv]
    )
    assert qv_grad > 0


def test_read_only_register_rows_can_read_original_tokens() -> None:
    torch.manual_seed(302)
    vision = _Vision()
    adapter = InternVLPlanningRegisters(
        8, num_registers=4, register_dim=8, attention_mode="read_only"
    )
    first, _ = adapter._encode_with_registers(
        vision, torch.zeros(1, 3, 2, 2)
    )
    second, _ = adapter._encode_with_registers(
        vision, torch.ones(1, 3, 2, 2)
    )
    assert not torch.allclose(first, second)


def test_read_only_flash_attention_fails_explicitly() -> None:
    adapter = InternVLPlanningRegisters(
        8, num_registers=4, register_dim=8, attention_mode="read_only"
    )
    with pytest.raises(RuntimeError, match="use_flash_attn=false"):
        adapter.configure_vision_attention(_Vision(use_flash_attn=True))


def test_bidirectional_mode_remains_available_as_ablation() -> None:
    adapter = InternVLPlanningRegisters(
        8, num_registers=4, register_dim=8, attention_mode="bidirectional"
    )
    registers, patches = adapter._encode_with_registers(
        _Vision(), torch.randn(1, 3, 2, 2)
    )
    assert registers.shape == (1, 4, 8)
    assert patches.shape == (1, 4, 8)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_split_sdpa_forward_backward_matches_eager(dtype: torch.dtype) -> None:
    torch.manual_seed(304)
    eager = _Vision().to(dtype=dtype).eval()
    split = copy.deepcopy(eager).eval()
    configure_read_only_register_attention(eager, 4, backend="eager")
    configure_read_only_register_attention(split, 4, backend="split_sdpa")
    set_read_only_register_sequence_length(eager, 9)
    set_read_only_register_sequence_length(split, 9)
    eager_input = torch.randn(2, 9, 8, dtype=dtype, requires_grad=True)
    split_input = eager_input.detach().clone().requires_grad_(True)

    eager_output = eager.encoder.layers[0].attn(eager_input)
    split_output = split.encoder.layers[0].attn(split_input)
    atol = 1e-6 if dtype == torch.float32 else 1e-2
    rtol = 1e-5 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(split_output, eager_output, atol=atol, rtol=rtol)

    gradient = torch.randn_like(eager_output)
    eager_output.backward(gradient)
    split_output.backward(gradient)
    torch.testing.assert_close(
        split_input.grad, eager_input.grad, atol=atol, rtol=rtol
    )
    for eager_parameter, split_parameter in zip(
        eager.parameters(), split.parameters()
    ):
        if eager_parameter.grad is None and split_parameter.grad is None:
            continue
        torch.testing.assert_close(
            split_parameter.grad, eager_parameter.grad, atol=atol, rtol=rtol
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA parity audit")
def test_split_sdpa_cuda_bf16_realistic_attention_shape() -> None:
    """Bound fused-kernel rounding at the formal InternViT width/token count."""
    torch.manual_seed(20260903)
    eager = _Vision(dim=1024).cuda().to(dtype=torch.bfloat16).eval()
    split = copy.deepcopy(eager).eval()
    configure_read_only_register_attention(eager, 16, backend="eager")
    configure_read_only_register_attention(split, 16, backend="split_sdpa")
    set_read_only_register_sequence_length(eager, 1041)
    set_read_only_register_sequence_length(split, 1041)
    eager_input = torch.randn(
        1, 1041, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    split_input = eager_input.detach().clone().requires_grad_(True)

    eager_output = eager.encoder.layers[0].attn(eager_input)
    split_output = split.encoder.layers[0].attn(split_input)
    max_forward_difference = (
        split_output.float() - eager_output.float()
    ).abs().max()
    assert max_forward_difference.item() <= 1e-3

    gradient = torch.randn_like(eager_output)
    eager_output.backward(gradient)
    split_output.backward(gradient)
    max_input_gradient_difference = (
        split_input.grad.float() - eager_input.grad.float()
    ).abs().max()
    assert max_input_gradient_difference.item() <= 1e-3
    assert torch.isfinite(split_input.grad).all()
