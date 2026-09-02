from __future__ import annotations

import inspect

from omegaconf import OmegaConf
import pytest
import torch

from navsim.agents.EpisodeDrive.action_decoder import ActionDecoder
from navsim.agents.EpisodeDrive.drivevla_base_agent import DriveVLABaseAgent


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


def _fusion_config(mode: str):
    return OmegaConf.create(
        {
            "mode": mode,
            "transition_fraction": 0.20,
            "semantic_gate_init": 0.549306,
        }
    )


def test_planning_plus_semantic_rho_schedule_and_formula() -> None:
    decoder = ActionDecoder(
        _action_config(),
        scene_fusion_config=_fusion_config("planning_plus_semantic"),
        total_optimizer_steps=100,
    )
    decoder.train()
    semantic = torch.randn(2, 16, 256)
    planning = torch.randn(2, 16, 256)
    assert torch.tanh(decoder.semantic_gate).mean().item() == pytest.approx(0.5)

    decoder.set_optimizer_step(0)
    scene0, rho0 = decoder.fuse_scene_features(semantic, planning)
    torch.testing.assert_close(scene0, semantic)
    assert rho0.item() == 0.0

    decoder.set_optimizer_step(10)
    scene_half, rho_half = decoder.fuse_scene_features(semantic, planning)
    planning_target = decoder.scene_norm(
        planning + torch.tanh(decoder.semantic_gate) * semantic
    )
    expected_half = 0.5 * semantic + 0.5 * planning_target
    torch.testing.assert_close(scene_half, expected_half)
    assert rho_half.item() == 0.5

    decoder.set_optimizer_step(20)
    scene1, rho1 = decoder.fuse_scene_features(semantic, planning)
    torch.testing.assert_close(scene1, planning_target)
    assert rho1.item() == 1.0
    assert scene1.shape == (2, 16, 256)

    decoder.eval()
    decoder.set_optimizer_step(0)
    _, eval_rho = decoder.fuse_scene_features(semantic, planning)
    assert eval_rho.item() == 1.0


def test_semantic_only_and_planning_only_modes() -> None:
    semantic = torch.randn(2, 16, 256)
    planning = torch.randn(2, 16, 256)

    semantic_decoder = ActionDecoder(
        _action_config(),
        scene_fusion_config=_fusion_config("semantic_only"),
        total_optimizer_steps=100,
    )
    semantic_scene, semantic_rho = semantic_decoder.fuse_scene_features(
        semantic, None
    )
    torch.testing.assert_close(semantic_scene, semantic)
    assert not hasattr(semantic_decoder, "scene_norm")
    assert not hasattr(semantic_decoder, "semantic_gate")
    assert semantic_rho.item() == 0.0

    planning_decoder = ActionDecoder(
        _action_config(),
        scene_fusion_config=_fusion_config("planning_only"),
        total_optimizer_steps=100,
    )
    scene_a, planning_rho = planning_decoder.fuse_scene_features(
        semantic, planning
    )
    scene_b, _ = planning_decoder.fuse_scene_features(
        semantic + 1000.0, planning
    )
    torch.testing.assert_close(scene_a, scene_b)
    torch.testing.assert_close(scene_a, planning_decoder.scene_norm(planning))
    assert planning_rho.item() == 1.0


def test_scorer_gradient_reaches_planning_but_not_proposals() -> None:
    torch.manual_seed(21)
    decoder = ActionDecoder(
        _action_config(),
        scene_fusion_config=_fusion_config("planning_plus_semantic"),
        total_optimizer_steps=100,
    )
    decoder.train()
    decoder.set_optimizer_step(100)
    planning = torch.randn(2, 16, 256, requires_grad=True)
    captured_cross_features = {}

    def capture_trajectory(_module, args):
        captured_cross_features["trajectory"] = args[1]

    def capture_scorer(_module, args):
        captured_cross_features["scorer"] = args[1]

    trajectory_hook = decoder.trajectory_decoder.register_forward_pre_hook(
        capture_trajectory
    )
    scorer_hook = decoder.scorer_attention.register_forward_pre_hook(capture_scorer)
    output = decoder(
        {
            "last_hidden_state": torch.randn(2, 12, 1536),
            "status_feature": torch.randn(2, 8),
            "planning_registers": planning,
        }
    )
    trajectory_hook.remove()
    scorer_hook.remove()

    assert output["semantic_scene_features"].shape == (2, 16, 256)
    assert output["planning_scene_features"].shape == (2, 16, 256)
    assert output["scene_mix_ratio"].item() == 1.0
    assert captured_cross_features["trajectory"] is captured_cross_features["scorer"]

    output["proposals"].retain_grad()
    scorer_loss = sum(value.sum() for value in output["pred_logit"].values())
    scorer_loss.backward()
    assert output["proposals"].grad is None or torch.count_nonzero(
        output["proposals"].grad
    ).item() == 0
    assert planning.grad is not None
    assert planning.grad.norm().item() > 0.0


def test_optimizer_step_is_checkpointed_for_resume() -> None:
    first = ActionDecoder(
        _action_config(),
        scene_fusion_config=_fusion_config("planning_plus_semantic"),
        total_optimizer_steps=100,
    )
    first.set_optimizer_step(7)
    restored = ActionDecoder(
        _action_config(),
        scene_fusion_config=_fusion_config("planning_plus_semantic"),
        total_optimizer_steps=100,
    )
    restored.load_state_dict(first.state_dict(), strict=True)
    restored.train()
    assert restored.scene_mix_ratio(torch.zeros(())).item() == pytest.approx(0.35)


def test_agent_forward_does_not_pop_shared_feature_dictionary() -> None:
    source = inspect.getsource(DriveVLABaseAgent.forward)
    assert "features.pop(" not in source
    assert "runtime_features = dict(features)" in source
