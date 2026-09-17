from pathlib import Path
import pytest
from omegaconf import OmegaConf
from starVLA.rl.flow_grpo.config import resolve_config


def test_inactive_sft_losses_keep_source_fp32_with_bf16_condition():
    """Exercise real framework forward without constructing another CPU Qwen."""
    from types import SimpleNamespace
    import torch
    from starVLA.model.framework.QwenOFT import Qwenvl_OFT

    hidden = torch.ones(1, 8, 4, dtype=torch.bfloat16, requires_grad=True)
    action_loss = torch.tensor(0.25, dtype=torch.float32, requires_grad=True)
    fake = SimpleNamespace(
        w_depth=False,
        w_video_latent=False,
        mlp_head=0,
        config=OmegaConf.create(
            {
                "datasets": {
                    "video_data": {"load_2d_data": False},
                    "vla_data": {"load_act_data": 1},
                    "gs_data": {"load_3d_data": False},
                    "reward_data": {"load_reward_data": False},
                },
                "framework": {"action_model": {"flow_train_repeats": 1}},
            }
        ),
        encode_policy_features=lambda examples: (
            hidden,
            {"action": torch.arange(8).unsqueeze(0)},
        ),
        action_model=lambda *args, **kwargs: action_loss,
        agent_dino_head=torch.nn.Linear(4, 4),
    )
    fake.forward = lambda **kwargs: Qwenvl_OFT.forward(fake, **kwargs)
    losses = Qwenvl_OFT.compute_sft_losses(fake, [{"action": [[0.0] * 4] * 8}])
    assert losses["action_loss"] is action_loss
    for name in ("rgb_loss", "gs_loss", "reward_loss", "agent_dino_loss"):
        assert losses[name].dtype == torch.float32
        assert losses[name].item() == 0 and not losses[name].requires_grad
    sum(losses.values()).backward()
    assert hidden.grad is None
    assert all(p.grad is None for p in fake.agent_dino_head.parameters())


@pytest.mark.parametrize("variant", ["frozen_visual", "unfrozen_visual"])
def test_exact_action_source_config_and_forbidden_freeze_changes(tmp_path, variant):
    path = f"configs/flow_grpo/action_only_{variant}.yaml"
    raw = OmegaConf.load(path)
    checkpoint_cfg = OmegaConf.load(Path(raw.sft_checkpoint) / "config.yaml")
    OmegaConf.save(checkpoint_cfg, tmp_path / "config.yaml")
    cfg, _ = resolve_config(path, sft_checkpoint=str(tmp_path))
    assert cfg["runtime"]["noise_seed_schedule"] == "global_scene_v1"
    checkpoint_cfg.trainer.freeze_modules += (
        ",qwen_vl_interface.model.model.language_model"
    )
    OmegaConf.save(checkpoint_cfg, tmp_path / "config.yaml")
    with pytest.raises(ValueError, match="exact audited SFT config"):
        resolve_config(path, sft_checkpoint=str(tmp_path))
    with pytest.raises(ValueError, match="source SFT configuration SHA256"):
        resolve_config(
            path, overrides=["checkpoint_contract.source_config_sha256=invalid"]
        )
    with pytest.raises(ValueError, match="visual freeze is mandatory"):
        resolve_config(path, overrides=["rl_freeze_modules=[]"])
