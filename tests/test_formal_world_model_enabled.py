from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest
import torch
from torch import nn

from navsim.agents.EpisodeDrive.drivevla_base_agent import DriveVLABaseAgent
from navsim.agents.EpisodeDrive.formal_initialization import (
    validate_formal_scientific_contract,
)


CONFIG_DIR = Path(
    "navsim/planning/script/config/common/agent"
).resolve()


def _formal_agent_config(monkeypatch):
    monkeypatch.setenv("PLANREG_BASE_VLM_PATH", "/tmp/base")
    monkeypatch.setenv("PLANREG_SHARED_INIT", "/tmp/shared.pt")
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name="episode_drive_planreg_wm_formal_base")


def test_both_formal_variants_have_world_model_from_step_zero(monkeypatch):
    monkeypatch.setenv("PLANREG_BASE_VLM_PATH", "/tmp/base")
    monkeypatch.setenv("PLANREG_VQA_VLM_PATH", "/tmp/vqa")
    monkeypatch.setenv("PLANREG_SHARED_INIT", "/tmp/shared.pt")
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        base = compose(config_name="episode_drive_planreg_wm_formal_base")
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        vqa = compose(config_name="episode_drive_planreg_wm_formal_vqa")
    for config in (base, vqa):
        assert config.world_model.enabled is True
        assert config.world_model.future_mode == "correct"
        assert config.world_model.predictor_only is False
        assert config.world_model.min_weight == pytest.approx(0.01)
        assert config.world_model.start_fraction == 0.0
        assert config.world_model.candidate_count == 1
        assert config.world_model.trajectory_source == "gt"


def test_formal_world_model_schedule_is_positive_on_first_step():
    agent = DriveVLABaseAgent.__new__(DriveVLABaseAgent)
    nn.Module.__init__(agent)
    agent.world_model_enabled = True
    agent.world_model_config = OmegaConf.create(
        {
            "min_weight": 0.01,
            "max_weight": 0.10,
            "start_fraction": 0.0,
            "ramp_fraction": 0.10,
        }
    )
    agent.register_buffer("_world_model_optimizer_step", torch.tensor(0))
    agent.register_buffer("_world_model_total_optimizer_steps", torch.tensor(100))
    assert agent.current_world_model_weight() == pytest.approx(0.01)
    agent._world_model_optimizer_step.fill_(10)
    assert agent.current_world_model_weight() == pytest.approx(0.10)


def test_formal_contract_rejects_no_world_model_ablation(monkeypatch):
    config = _formal_agent_config(monkeypatch)
    config.world_model.enabled = False
    with pytest.raises(ValueError, match="world model must be enabled"):
        validate_formal_scientific_contract(
            vlm_config=config.vlm_config,
            vision_adaptation=config.vision_adaptation,
            planning_registers=config.planning_registers,
            scene_fusion=config.scene_fusion,
            semantic_path=config.semantic_path,
            world_model=config.world_model,
            ema=config.ema,
            action_head_config=config.action_head_config,
        )
