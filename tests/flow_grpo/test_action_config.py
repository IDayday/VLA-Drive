from pathlib import Path
import pytest
from omegaconf import OmegaConf
from starVLA.rl.flow_grpo.config import resolve_config


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
