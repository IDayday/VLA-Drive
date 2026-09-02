import torch
from omegaconf import OmegaConf

from navsim.agents.EpisodeDrive.action_decoder import ActionDecoder


def _decoder() -> ActionDecoder:
    config = OmegaConf.load(
        "navsim/planning/script/config/common/agent/"
        "episode_drive_planreg_wm_formal_common.yaml"
    )
    return ActionDecoder(config.action_head_config, config.scene_fusion)


def test_planning_primary_cross_attention_shape_and_gate():
    torch.manual_seed(0)
    decoder = _decoder()
    planning = torch.randn(2, 16, 256, requires_grad=True)
    semantic = torch.randn(2, 16, 256, requires_grad=True)
    scene, gate = decoder.fuse_scene_features(semantic, planning)
    assert scene.shape == (2, 16, 256)
    assert torch.allclose(gate, torch.tensor(0.20), atol=1e-6)

    loss = (scene * torch.randn_like(scene)).sum()
    loss.backward()
    assert planning.grad is not None and planning.grad.abs().sum() > 0
    assert semantic.grad is not None and semantic.grad.abs().sum() > 0
    assert decoder.semantic_gate.grad is not None
    assert decoder.semantic_cross_attention.in_proj_weight.grad is not None


def test_semantic_tokens_are_key_value_context_not_slotwise_addition():
    torch.manual_seed(1)
    decoder = _decoder().eval()
    planning = torch.randn(1, 16, 256)
    semantic = torch.randn(1, 16, 256)
    original, _ = decoder.fuse_scene_features(semantic, planning)
    permuted, _ = decoder.fuse_scene_features(
        semantic[:, torch.randperm(semantic.shape[1])], planning
    )
    # Cross-attention without semantic positional encodings is invariant to a
    # joint K/V permutation. A direct slot-wise semantic addition is not.
    torch.testing.assert_close(original, permuted, atol=2e-6, rtol=2e-6)


def test_formal_fusion_has_no_rho_transition_buffers():
    decoder = _decoder()
    assert decoder.scene_feature_mode == "planning_primary_semantic_xattn"
    assert not hasattr(decoder, "_optimizer_step")
    assert not hasattr(decoder, "_total_optimizer_steps")
    assert not hasattr(decoder, "scene_norm")
