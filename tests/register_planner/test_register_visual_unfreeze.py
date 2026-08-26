"""CPU contracts for the Register64 visual-unfrozen Stage-G variant."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from starVLA.model.framework.QwenRegisterGenerator import QwenRegisterGenerator
from starVLA.model.framework.baseline_qwen import build_baseline_qwen_batch
from starVLA.model.modules.register_planner.checkpoint import (
    load_register_generator_checkpoint,
    save_register_generator_checkpoint,
)
from starVLA.model.modules.vlm.visual_training import (
    encode_qwen_images,
    run_visual_block,
)
from starVLA.training.config_loader import load_training_config
from starVLA.training.train_register_generator import build_generator_optimizer


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "starVLA/config/training"


class _FakeImageEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 5, bias=False)

    def get_image_features(self, pixel_values, image_grid_thw):
        del image_grid_thw
        features = self.projection(pixel_values)
        return [features], [features * 0.5]


def test_visual_encoder_gradient_switch_is_strict():
    grid = torch.tensor([[1, 1, 2]], dtype=torch.long)
    pixels = torch.randn(2, 3)
    frozen = _FakeImageEncoder()
    frozen_images, _ = encode_qwen_images(
        frozen,
        pixel_values=pixels,
        image_grid_thw=grid,
        freeze_visual=True,
    )
    assert not frozen_images.requires_grad

    trainable = _FakeImageEncoder()
    trainable_images, deepstack = encode_qwen_images(
        trainable,
        pixel_values=pixels,
        image_grid_thw=grid,
        freeze_visual=False,
    )
    assert trainable_images.requires_grad
    (trainable_images.sum() + deepstack[0].sum()).backward()
    assert trainable.projection.weight.grad is not None
    assert torch.count_nonzero(trainable.projection.weight.grad)


def test_visual_block_checkpoint_preserves_output_and_gradient():
    torch.manual_seed(7)
    plain = nn.Linear(4, 4)
    checkpointed = nn.Linear(4, 4)
    checkpointed.load_state_dict(plain.state_dict())
    plain_input = torch.randn(3, 4, requires_grad=True)
    checkpointed_input = plain_input.detach().clone().requires_grad_(True)
    plain_output = run_visual_block(plain, plain_input, checkpoint_enabled=False)
    checkpointed_output = run_visual_block(
        checkpointed, checkpointed_input, checkpoint_enabled=True
    )
    plain_output.square().mean().backward()
    checkpointed_output.square().mean().backward()
    torch.testing.assert_close(checkpointed_output, plain_output)
    torch.testing.assert_close(checkpointed_input.grad, plain_input.grad)
    torch.testing.assert_close(checkpointed.weight.grad, plain.weight.grad)


def test_qwen_vision_model_checkpointed_backward_reaches_all_parameters():
    from starVLA.model.modules.vlm.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLVisionConfig,
        Qwen3VLVisionModel,
    )

    config = Qwen3VLVisionConfig(
        depth=2,
        hidden_size=16,
        hidden_act="gelu",
        intermediate_size=32,
        num_heads=4,
        in_channels=3,
        patch_size=2,
        spatial_merge_size=1,
        temporal_patch_size=2,
        out_hidden_size=12,
        num_position_embeddings=16,
        deepstack_visual_indexes=[0],
        _attn_implementation="eager",
    )
    model = Qwen3VLVisionModel(config).train()
    model.gradient_checkpointing = True
    image_tokens, deepstack = model(
        torch.randn(4, 24),
        torch.tensor([[1, 2, 2]], dtype=torch.long),
    )
    (image_tokens.sum() + sum(value.sum() for value in deepstack)).backward()
    assert image_tokens.shape == (4, 12)
    assert len(deepstack) == 1
    assert all(parameter.grad is not None for parameter in model.parameters())


class _CachedInterface(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.visual = nn.Linear(2, 2)
        self.model.device = torch.device("cpu")


def test_trainable_visual_rejects_cached_qwen_features():
    interface = _CachedInterface()
    with pytest.raises(RuntimeError, match="requires raw images"):
        build_baseline_qwen_batch(
            interface,
            [{"qwen_feature_cache": {}}],
            ["instruction"],
            {},
        )


def test_visual_unfrozen_configs_share_checkpoint_manifest_boundary():
    names = (
        "qwen_register64_generator_visual_unfrozen.yaml",
        "register64_candidate_bank_visual_unfrozen.yaml",
        "register64_inference_visual_unfrozen.yaml",
    )
    configs = [load_training_config(CONFIG_ROOT / name) for name in names]
    for config in configs:
        frozen = {
            value.strip()
            for value in str(config.trainer.freeze_modules).split(",")
            if value.strip()
        }
        assert config.framework.qwenvl.freeze_visual is False
        assert config.framework.qwenvl.visual_gradient_checkpointing is True
        assert "qwen_vl_interface.model.visual" not in frozen
        assert "qwen_vl_interface.model.lm_head" in frozen
        assert config.optimizer.learning_rates.qwen_visual == pytest.approx(2.0e-6)
    assert configs[0].trainer.global_batch_size == 32
    assert configs[0].datasets.vla_data.per_device_batch_size == 2


def test_visual_optimizer_group_is_unique_and_uses_lower_lr(tiny_factory):
    config = tiny_factory.config(4)
    config.framework.qwenvl.freeze_visual = False
    config.framework.qwenvl.visual_gradient_checkpointing = True
    config.trainer.freeze_modules = "qwen_vl_interface.model.lm_head"
    config.optimizer = OmegaConf.create(
        {
            "learning_rates": {
                "qwen_visual": 2.0e-6,
                "qwen_vl_interface": 1.0e-5,
                "scene_encoder": 2.0e-4,
                "register_generator": 2.0e-4,
                "action_input_model": 2.0e-4,
            },
            "betas": [0.9, 0.95],
            "weight_decay": 1.0e-3,
            "eps": 1.0e-8,
            "fused": False,
        }
    )
    qwen = tiny_factory.qwen()
    qwen.model.visual.gradient_checkpointing = False
    model = QwenRegisterGenerator(
        config,
        qwen_vl_interface=qwen,
        qwen_hidden_extractor=tiny_factory.extractor,
    )
    optimizer = build_generator_optimizer(model, config)
    groups = {group["name"]: group for group in optimizer.param_groups}
    visual_ids = {id(parameter) for parameter in model.qwen_visual.parameters()}
    grouped_ids = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    assert model.qwen_visual_frozen is False
    assert model.qwen_visual.gradient_checkpointing is True
    assert groups["qwen_visual"]["lr"] == pytest.approx(2.0e-6)
    assert groups["qwen_vl_interface"]["lr"] == pytest.approx(1.0e-5)
    assert {
        id(parameter) for parameter in groups["qwen_visual"]["params"]
    } == visual_ids
    assert len(grouped_ids) == len(set(grouped_ids))


def test_visual_weights_roundtrip_in_generator_component_checkpoint(
    tiny_factory, tmp_path
):
    config = tiny_factory.config(4)
    config.framework.qwenvl.freeze_visual = False
    config.framework.qwenvl.visual_gradient_checkpointing = True
    config.trainer.freeze_modules = "qwen_vl_interface.model.lm_head"

    def build_model():
        qwen = tiny_factory.qwen()
        qwen.model.visual.gradient_checkpointing = False
        return QwenRegisterGenerator(
            config,
            qwen_vl_interface=qwen,
            qwen_hidden_extractor=tiny_factory.extractor,
        )

    source = build_model()
    with torch.no_grad():
        source.qwen_visual.weight.fill_(0.125)
    path = tmp_path / "visual-generator.pt"
    metadata = {
        "schema_version": 1,
        "stage": "register_generator",
        "qwen_base_model": "tiny-qwen",
        "proposal_num": 4,
        "num_poses": 8,
        "state_dim": 3,
        "scene_queries": 4,
        "scene_dim": 32,
        "decoder_layers": 2,
        "decoder_heads": 1,
        "proposal_head_style": "donor_mlp_v1",
        "stage_loss_mode": "final_only",
        "proposal_head_count": 1,
        "commit": "test",
        "config_hash": "test",
    }
    save_register_generator_checkpoint(
        path,
        qwen_vl_interface=source.qwen_vl_interface,
        action_input_model=source.action_input_model,
        scene_encoder=source.scene_encoder,
        register_generator=source.register_generator,
        metadata=metadata,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert "model.visual.weight" in payload["state_dict"]["qwen_trainable"]

    target = build_model()
    load_register_generator_checkpoint(
        path,
        qwen_vl_interface=target.qwen_vl_interface,
        action_input_model=target.action_input_model,
        scene_encoder=target.scene_encoder,
        register_generator=target.register_generator,
    )
    torch.testing.assert_close(target.qwen_visual.weight, source.qwen_visual.weight)
