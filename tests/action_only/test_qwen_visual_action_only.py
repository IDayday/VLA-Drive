"""CPU contracts for the action-only Qwen visual fine-tuning experiment."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class _FakeQwenBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 5, bias=False)

    def get_image_features(self, pixel_values, image_grid_thw):
        del image_grid_thw
        features = self.projection(pixel_values)
        return [features], [features * 0.5]


def test_visual_encoder_gradient_switch_is_strict():
    from starVLA.model.modules.vlm.visual_training import encode_qwen_images

    grid = torch.tensor([[1, 1, 2]], dtype=torch.long)
    pixels = torch.randn(2, 3)

    frozen = _FakeQwenBackbone()
    frozen_images, frozen_deepstack = encode_qwen_images(
        frozen,
        pixel_values=pixels,
        image_grid_thw=grid,
        freeze_visual=True,
    )
    assert frozen_images.shape == (2, 5)
    assert len(frozen_deepstack) == 1
    assert not frozen_images.requires_grad

    trainable = _FakeQwenBackbone()
    trainable_images, trainable_deepstack = encode_qwen_images(
        trainable,
        pixel_values=pixels,
        image_grid_thw=grid,
        freeze_visual=False,
    )
    assert trainable_images.requires_grad
    (trainable_images.sum() + trainable_deepstack[0].sum()).backward()
    assert trainable.projection.weight.grad is not None
    assert trainable.projection.weight.grad.abs().sum() > 0


def test_visual_block_checkpoint_preserves_output_and_gradient():
    from starVLA.model.modules.vlm.visual_training import run_visual_block

    torch.manual_seed(7)
    plain_block = nn.Linear(4, 4)
    checkpointed_block = nn.Linear(4, 4)
    checkpointed_block.load_state_dict(plain_block.state_dict())
    plain_input = torch.randn(3, 4, requires_grad=True)
    checkpointed_input = plain_input.detach().clone().requires_grad_(True)

    plain_output = run_visual_block(
        plain_block,
        plain_input,
        checkpoint_enabled=False,
    )
    checkpointed_output = run_visual_block(
        checkpointed_block,
        checkpointed_input,
        checkpoint_enabled=True,
    )
    plain_output.square().mean().backward()
    checkpointed_output.square().mean().backward()

    torch.testing.assert_close(checkpointed_output, plain_output)
    torch.testing.assert_close(checkpointed_input.grad, plain_input.grad)
    torch.testing.assert_close(
        checkpointed_block.weight.grad,
        plain_block.weight.grad,
    )


class _NestedLearningRateModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qwen_vl_interface = nn.Module()
        self.qwen_vl_interface.language = nn.Linear(4, 4)
        self.qwen_vl_interface.model = nn.Module()
        self.qwen_vl_interface.model.model = nn.Module()
        self.qwen_vl_interface.model.model.visual = nn.Linear(4, 4)
        self.action_model = nn.Linear(4, 2)


def test_nested_learning_rate_group_assigns_visual_parameters_once():
    from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups

    model = _NestedLearningRateModel()
    cfg = OmegaConf.create(
        {
            "trainer": {
                "freeze_modules": "",
                "learning_rate": {
                    "base": 1.0e-5,
                    "qwen_vl_interface": 1.0e-5,
                    "qwen_vl_interface.model.model.visual": 2.0e-6,
                    "action_model": 1.0e-5,
                },
            }
        }
    )
    groups = build_param_lr_groups(model, cfg)
    by_name = {group["name"]: group for group in groups}
    visual_ids = {
        id(parameter)
        for parameter in model.qwen_vl_interface.model.model.visual.parameters()
    }
    grouped_ids = [id(parameter) for group in groups for parameter in group["params"]]

    assert len(grouped_ids) == len(set(grouped_ids))
    assert {
        id(parameter)
        for parameter in by_name["qwen_vl_interface.model.model.visual"]["params"]
    } == visual_ids
    assert by_name["qwen_vl_interface.model.model.visual"]["lr"] == 2.0e-6
    assert visual_ids.isdisjoint(
        id(parameter) for parameter in by_name["qwen_vl_interface"]["params"]
    )


def test_learning_rate_metrics_report_every_optimizer_group():
    from starVLA.training.trainer_utils.trainer_tools import (
        collect_learning_rate_metrics,
    )

    parameter_a = nn.Parameter(torch.ones(()))
    parameter_b = nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW(
        [
            {"params": [parameter_a], "lr": 2.0e-6, "name": "qwen_visual"},
            {"params": [parameter_b], "lr": 1.0e-5, "name": "action_model"},
        ]
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[lambda _: 1.0, lambda _: 1.0],
    )

    metrics = collect_learning_rate_metrics(scheduler)

    assert metrics["learning_rate"] == 2.0e-6
    assert metrics["learning_rate/qwen_visual"] == 2.0e-6
    assert metrics["learning_rate/action_model"] == 1.0e-5


def test_visual_action_only_overlay_has_no_auxiliary_teacher_or_world_head():
    from starVLA.model.modules.vlm.visual_training import (
        validate_visual_action_only_config,
    )

    base = OmegaConf.load(REPO_ROOT / "starVLA/config/training/cfg_yaw_1225.yaml")
    overlay = OmegaConf.load(
        REPO_ROOT / "starVLA/config/training/qwen_visual_action_only.yaml"
    )
    cfg = OmegaConf.merge(base, overlay)
    validate_visual_action_only_config(cfg)

    assert cfg.framework.name == "QwenOFT"
    assert cfg.framework.action_prompt_mode == "minimal"
    assert cfg.framework.qwenvl.freeze_visual is False
    assert cfg.framework.qwenvl.visual_gradient_checkpointing is True
    assert cfg.datasets.vla_data.load_act_data == 1
    assert cfg.datasets.video_data.load_2d_data == 0
    assert cfg.datasets.gs_data.load_3d_data == 0
    assert cfg.datasets.reward_data.load_reward_data == 0
    assert cfg.w_depth == 0
    assert "qwen_vl_interface.model.model.visual" not in cfg.trainer.freeze_modules
    assert "qwen_vl_interface.model.lm_head" in cfg.trainer.freeze_modules


def test_visual_action_launcher_disables_all_external_feature_caches():
    launcher = (
        REPO_ROOT / "8-train_action-only-qwen-visual.sh"
    ).read_text(encoding="utf-8")

    assert "NAVSIM_USE_FEATURE_CACHE=0" in launcher
    assert "unset NAVSIM_AGENT_DINO_CACHE_ROOT" in launcher
    assert "unset NAVSIM_VGGT_CACHE_ROOT" in launcher
    assert "qwen_visual_action_only.yaml" in launcher
    assert "TARGET_EFFECTIVE_BATCH_SIZE" in launcher
    assert "VISUAL_LEARNING_RATE" in launcher
    assert "QWEN_VISUAL_RUN_SMOKE_BEFORE_FORMAL" in launcher
    assert "MAX_TRAIN_STEPS=2" in launcher


def test_framework_registry_does_not_eagerly_import_optional_world_heads():
    code = """
import sys
import starVLA.model.framework
for name in (
    'starVLA.model.modules.video_model.wan_i2v_header',
    'starVLA.model.modules.depth_model.models.ppd_train',
    'starVLA.model.modules.gs_model.storm_gs_header',
):
    if name in sys.modules:
        raise SystemExit(f'eager optional import: {name}')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_inference_loads_visual_weights_without_building_a_gradient_graph():
    inference_source = (REPO_ROOT / "infer.py").read_text(encoding="utf-8")

    assert '"framework.qwenvl.freeze_visual"' in inference_source
    assert '"framework.qwenvl.visual_gradient_checkpointing"' in inference_source
