from __future__ import annotations

import copy

import torch

from navsim.agents.EpisodeDrive.layers.planning_registers import (
    InternVLPlanningRegisters,
    inject_internvit_qv_lora,
)
from test_read_only_register_attention import _Vision


def test_read_only_cls_and_patch_semantics_are_legacy_exact() -> None:
    torch.manual_seed(303)
    vision = _Vision()
    reference = copy.deepcopy(vision).eval()
    inject_internvit_qv_lora(vision, rank=2)
    adapter = InternVLPlanningRegisters(
        8, num_registers=4, register_dim=8, attention_mode="read_only"
    )
    pixels = torch.randn(3, 3, 2, 2)
    legacy_tokens = reference.encoder(
        inputs_embeds=reference.embeddings(pixels), return_dict=True
    ).last_hidden_state
    _, patch_tokens = adapter._encode_with_registers(vision, pixels)
    torch.testing.assert_close(
        patch_tokens,
        legacy_tokens[:, 1:],
        atol=1e-5,
        rtol=1e-5,
    )
    assert patch_tokens.shape[1] == 4


def test_read_only_direct_legacy_call_is_not_masked_as_register_input() -> None:
    torch.manual_seed(304)
    vision = _Vision()
    reference = copy.deepcopy(vision).eval()
    adapter = InternVLPlanningRegisters(
        8, num_registers=4, register_dim=8, attention_mode="read_only"
    )
    adapter.configure_vision_attention(vision)
    pixels = torch.randn(1, 3, 2, 2)
    actual = vision.encoder(
        inputs_embeds=vision.embeddings(pixels), return_dict=True
    ).last_hidden_state
    expected = reference.encoder(
        inputs_embeds=reference.embeddings(pixels), return_dict=True
    ).last_hidden_state
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

