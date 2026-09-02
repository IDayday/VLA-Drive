from __future__ import annotations

from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import torch
from torch import nn

from navsim.agents.EpisodeDrive.action_decoder import ActionDecoder
from navsim.agents.EpisodeDrive.drivevla_base_agent import DriveVLABaseAgent
from navsim.agents.EpisodeDrive.layers.planning_registers import (
    InternViTQVLoRALinear,
)
from navsim.agents.EpisodeDrive.layers.world_model import FutureRegisterPredictor


def _action_config():
    return OmegaConf.create(
        {
            "b2d": False,
            "num_poses": 8,
            "tf_d_model": 256,
            "tf_d_ffn": 64,
            "num_scene_tokens": 16,
            "proposal_num": 8,
            "ref_num": 1,
            "scorer_ref_num": 4,
            "one_token_per_traj": True,
            "full_history_status": False,
            "cam_f0": [3],
            "cam_l0": [],
            "cam_l1": [],
            "cam_l2": [],
            "cam_r0": [],
            "cam_r1": [],
            "cam_r2": [],
            "cam_b0": [],
            "lidar_pc": [],
            "double_score": False,
            "agent_pred": False,
            "area_pred": False,
            "bev_map": False,
            "bev_agent": False,
            "refiner_num_heads": 1,
            "refiner_ls_values": 0.0,
            "noc": 1.0,
            "dac": 1.0,
            "ddc": 0.0,
            "ttc": 5.0,
            "ep": 5.0,
            "comfort": 2.0,
        }
    )


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.planning_register_adapter = nn.Sequential(nn.Linear(4, 4))
        self.model = nn.Module()
        self.model.vision_model = nn.Module()
        self.model.vision_model.qkv = InternViTQVLoRALinear(
            nn.Linear(4, 12), rank=2
        )
        self.model.language_model = nn.Linear(4, 4)
        for parameter in self.model.language_model.parameters():
            parameter.requires_grad = False


def _agent():
    agent = DriveVLABaseAgent.__new__(DriveVLABaseAgent)
    nn.Module.__init__(agent)
    agent.scene_fusion = SimpleNamespace(mode="planning_plus_semantic")
    agent.vlm_config = SimpleNamespace(planning_registers_enabled=True)
    agent.world_model_enabled = True
    agent.batch_size = 2
    agent.num_gpus = 8
    agent.scheduler_args = None
    agent._lr_args = {
        "name": "AdamW",
        "base_batch_size": 16,
        "scale_with_batch_size": False,
        "planning_adapter_lr": 2e-4,
        "future_predictor_lr": 2e-4,
        "fusion_lr": 1e-4,
        "action_head_lr": 1e-4,
        "scorer_lr": 1e-4,
        "vision_qv_lora_lr": 5e-5,
        "semantic_qformer_lr": 1e-5,
        "language_model_lr": 0.0,
        "decay_weight_decay": 0.01,
        "no_decay_weight_decay": 0.0,
        "betas": (0.9, 0.999),
    }
    agent.backbone = _Backbone()
    agent.action_head = ActionDecoder(
        _action_config(),
        scene_fusion_config=agent.scene_fusion,
        total_optimizer_steps=100,
    )
    agent.future_register_predictor = FutureRegisterPredictor(
        hidden_dim=32, predictor_layers=2, num_heads=4
    )
    agent.ema_register_target = None
    return agent


def test_exact_configured_optimizer_groups_and_lrs() -> None:
    agent = _agent()
    optimizer = agent.get_optimizers()[0]
    groups = {group["name"]: group for group in optimizer.param_groups}
    assert {group["logical_name"] for group in groups.values()} == {
        "planning_adapter",
        "future_predictor",
        "fusion",
        "action_head",
        "scorer",
        "vision_qv_lora",
        "semantic_qformer",
    }
    expected_lrs = {
        "planning_adapter": 2e-4,
        "future_predictor": 2e-4,
        "fusion": 1e-4,
        "action_head": 1e-4,
        "scorer": 1e-4,
        "vision_qv_lora": 5e-5,
        "semantic_qformer": 1e-5,
    }
    for group in groups.values():
        assert group["lr"] == pytest.approx(expected_lrs[group["logical_name"]])
        if group["name"].endswith("_no_decay"):
            assert group["weight_decay"] == 0.0
        else:
            assert group["weight_decay"] == pytest.approx(0.01)
    assert groups["vision_qv_lora_no_decay"]["weight_decay"] == 0.0
    assert groups["fusion_no_decay"]["weight_decay"] == 0.0
    assert all(
        not parameter.requires_grad
        for parameter in agent.backbone.model.language_model.parameters()
    )


def test_unclassified_trainable_parameter_fails() -> None:
    agent = _agent()
    agent.rogue_parameter = nn.Parameter(torch.ones(1))
    with pytest.raises(RuntimeError, match="Unclassified"):
        agent.get_optimizers()


def test_planreg_hydra_config_contract() -> None:
    path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "navsim/planning/script/config/common/agent/episode_drive_planreg_wm_v1.yaml"
    )
    config = OmegaConf.load(path)
    assert config.vlm_config.vlm_type == "internvl"
    assert config.vlm_config.cache_hidden_state is False
    assert config.vlm_config.cache_mode is False
    assert config.vlm_config.planning_registers_enabled is True
    assert config.vlm_config.num_planning_registers == 16
    assert config.vlm_config.planning_register_dim == 256
    assert config.vision_adaptation.mode == "qv_lora"
    assert config.vision_adaptation.rank == 32
    assert config.vision_adaptation.train_k is False
    assert config.lora_config.use_lora is False
    assert config.world_model.horizons_sec == [0.5, 1.5, 3.0]
    assert config.scheduler_args.warmup_ratio == pytest.approx(0.03)
    assert config.scheduler_args.start_lr_ratio == pytest.approx(0.01)
    assert config.action_head_config.b2d is False
    assert config.action_head_config.proposal_num == 64
    assert config.action_head_config.scorer_ref_num == 4
