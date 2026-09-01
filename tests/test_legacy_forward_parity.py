from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from navsim.agents.EpisodeDrive.drivevla_backbone import (
    DriveVLABackbone,
    load_legacy_checkpoint_with_planreg_audit,
)
from navsim.agents.EpisodeDrive.layers.planning_registers import (
    InternViTQVLoRALinear,
)


class _FakeInternVL(nn.Module):
    def forward(
        self,
        pixel_values,
        input_ids,
        attention_mask,
        position_ids,
        image_flags,
        output_hidden_states,
        return_dict,
    ):
        assert output_hidden_states and return_dict
        vision_scalar = pixel_values.float().mean().to(input_ids.device)
        hidden = input_ids.float().unsqueeze(-1) + vision_scalar
        hidden = hidden + attention_mask.unsqueeze(-1) + position_ids.unsqueeze(-1)
        hidden = hidden + image_flags.float().sum()
        return SimpleNamespace(hidden_states=(hidden, hidden + 1.0))


def _make_disabled_backbone() -> DriveVLABackbone:
    backbone = DriveVLABackbone.__new__(DriveVLABackbone)
    nn.Module.__init__(backbone)
    backbone.model = _FakeInternVL()
    backbone.model_type = "internvl"
    backbone.device = "cpu"
    backbone.skip_lm_head = False
    backbone.planning_registers_enabled = False
    backbone.vision_qv_lora_enabled = False
    backbone.num_image_token = 2
    return backbone


def test_planning_disabled_calls_exact_legacy_forward() -> None:
    backbone = _make_disabled_backbone()
    pixels = torch.randn(2, 3, 2, 2)
    input_ids = torch.tensor([[4, 5, 6, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 0]])
    model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    reference = backbone.model(
        pixel_values=pixels.bfloat16(),
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        image_flags=torch.ones(2, dtype=torch.long),
        output_hidden_states=True,
        return_dict=True,
    )
    actual = backbone(
        pixels,
        ["unused"],
        [2],
        model_inputs=model_inputs,
    )
    max_abs_diff = (reference.hidden_states[-1] - actual.hidden_states[-1]).abs().max()
    assert max_abs_diff.item() <= 1e-5


class _NewPlanningState(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.planning_registers = nn.Parameter(torch.randn(1, 2, 3))
        self.register_norm = nn.LayerNorm(3)
        self.register_projection = nn.Linear(3, 4)


class _TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.planning_register_adapter = _NewPlanningState()


class _TinyAgent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _TinyBackbone()
        self.action_head = nn.Linear(2, 1)


def test_legacy_checkpoint_whitelists_only_new_planreg_keys() -> None:
    torch.manual_seed(11)
    source_agent = _TinyAgent()
    source = {
        "agent.backbone.linear.weight": source_agent.backbone.linear.weight.detach().clone(),
        "agent.backbone.linear.bias": source_agent.backbone.linear.bias.detach().clone(),
        "agent.action_head.weight": source_agent.action_head.weight.detach().clone(),
        "agent.action_head.bias": source_agent.action_head.bias.detach().clone(),
    }
    target = _TinyAgent()
    audit = load_legacy_checkpoint_with_planreg_audit(target, source)
    assert audit.invalid_missing_keys == []
    assert audit.unexpected_source_keys == []
    assert audit.allowed_missing_keys
    torch.testing.assert_close(target.backbone.linear.weight, source_agent.backbone.linear.weight)


def test_legacy_checkpoint_rejects_unexpected_and_nonwhitelisted_missing() -> None:
    target = _TinyAgent()
    bad_source = {
        "agent.backbone.linear.weight": target.backbone.linear.weight.detach().clone(),
        "agent.backbone.linear.bias": target.backbone.linear.bias.detach().clone(),
        "agent.action_head.weight": target.action_head.weight.detach().clone(),
        "agent.unexpected.weight": torch.ones(1),
    }
    with pytest.raises(RuntimeError, match="audit failed"):
        load_legacy_checkpoint_with_planreg_audit(target, bad_source)


class _RawBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.proj = nn.Linear(2, 3, bias=False)


class _RawAgent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _RawBackbone()


def test_legacy_peft_delta_is_folded_into_frozen_base_weight() -> None:
    target = _RawAgent()
    base = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    lora_a = torch.tensor([[1.0, 2.0]])
    lora_b = torch.tensor([[1.0], [2.0], [3.0]])
    source = {
        "agent.backbone.base_model.model.model.proj.base_layer.weight": base,
        "agent.backbone.base_model.model.model.proj.lora_A.default.weight": lora_a,
        "agent.backbone.base_model.model.model.proj.lora_B.default.weight": lora_b,
    }
    audit = load_legacy_checkpoint_with_planreg_audit(
        target, source, legacy_lora_scale=2.0
    )
    assert audit.merged_lora_module_count == 1
    torch.testing.assert_close(
        target.backbone.model.proj.weight,
        base + 2.0 * (lora_b @ lora_a),
    )


def test_legacy_raw_qkv_maps_to_new_qv_wrapper() -> None:
    class QKVBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.qkv = InternViTQVLoRALinear(nn.Linear(2, 6), rank=2)

    class QKVAgent(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = QKVBackbone()

    target = QKVAgent()
    source_weight = torch.randn(6, 2)
    source_bias = torch.randn(6)
    audit = load_legacy_checkpoint_with_planreg_audit(
        target,
        {
            "agent.backbone.model.qkv.weight": source_weight,
            "agent.backbone.model.qkv.bias": source_bias,
        },
    )
    assert len(audit.allowed_missing_keys) == 4
    torch.testing.assert_close(target.backbone.model.qkv.base_layer.weight, source_weight)
    torch.testing.assert_close(target.backbone.model.qkv.base_layer.bias, source_bias)
