from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest
import torch
from torch import nn
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from navsim.agents.EpisodeDrive.action_decoder import ActionDecoder
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


def _resolved_e0_config(monkeypatch):
    monkeypatch.setenv("NAVSIM_EXP_ROOT", "/tmp/navsim-exp")
    config_dir = (
        Path(__file__).resolve().parents[1]
        / "navsim/planning/script/config/training"
    )
    overrides = [
        "agent=episode_drive_planreg_wm_v1",
        "agent.vlm_config.planning_registers_enabled=false",
        "agent.vlm_config.vision_qv_lora_enabled=false",
        "agent.vision_adaptation.mode=none",
        "agent.scene_fusion.mode=semantic_only",
        "agent.world_model.enabled=false",
        "agent.ema.enabled=false",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="default_training", overrides=overrides)
    # Resolve the actual E0 agent subtree; unrelated data-path interpolations
    # intentionally remain outside this architecture parity audit.
    OmegaConf.resolve(cfg.agent)
    return cfg


def test_real_e0_resolved_config_is_exact_semantic_legacy_bypass(monkeypatch) -> None:
    cfg = _resolved_e0_config(monkeypatch)
    assert cfg.agent.vlm_config.planning_registers_enabled is False
    assert cfg.agent.vlm_config.vision_qv_lora_enabled is False
    assert cfg.agent.vision_adaptation.mode == "none"
    assert cfg.agent.world_model.enabled is False
    assert cfg.agent.ema.enabled is False
    assert cfg.agent.scene_fusion.mode == "semantic_only"

    torch.manual_seed(20260902)
    legacy = ActionDecoder(cfg.agent.action_head_config, scene_fusion_config=None)
    e0 = ActionDecoder(
        cfg.agent.action_head_config,
        scene_fusion_config=cfg.agent.scene_fusion,
    )
    # semantic_only adds no trainable parameter, normalization, gate, or buffer.
    e0.load_state_dict(legacy.state_dict(), strict=True)
    assert set(e0.state_dict()) == set(legacy.state_dict())
    assert not hasattr(e0, "scene_norm")
    assert not hasattr(e0, "semantic_gate")
    legacy.eval()
    e0.eval()

    inputs = {
        "last_hidden_state": torch.randn(2, 12, 1536),
        "status_feature": torch.randn(2, 8),
    }
    with torch.no_grad():
        reference = legacy(inputs)
        actual = e0(inputs)
    compared = ["trajectory", "proposals", "pdm_score"]
    max_abs_diff = max(
        (reference[name] - actual[name]).abs().max().item() for name in compared
    )
    for head_name in reference["pred_logit"]:
        max_abs_diff = max(
            max_abs_diff,
            (
                reference["pred_logit"][head_name]
                - actual["pred_logit"][head_name]
            )
            .abs()
            .max()
            .item(),
        )
    assert max_abs_diff <= 1e-5


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


def test_frozen_vlm_checkpoint_wrappers_train_without_enabling_dropout() -> None:
    class CheckpointLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gradient_checkpointing = True
            self.dropout = nn.Dropout(0.5)

    class Container(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([CheckpointLayer(), CheckpointLayer()])

    backbone = DriveVLABackbone.__new__(DriveVLABackbone)
    nn.Module.__init__(backbone)
    backbone.model_type = "internvl"
    backbone.gradient_checkpointing_enabled = True
    backbone.model = nn.Module()
    backbone.model.vision_model = nn.Module()
    backbone.model.vision_model.encoder = Container()
    backbone.model.language_model = nn.Module()
    backbone.model.language_model.model = Container()
    backbone.eval()

    backbone.activate_gradient_checkpointing_train_mode()

    assert backbone.model.vision_model.encoder.training
    for layer in backbone.model.language_model.model.layers:
        assert layer.training
        assert not layer.dropout.training
    for layer in backbone.model.vision_model.encoder.layers:
        assert not layer.training
        assert not layer.dropout.training
