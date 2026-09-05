import os
from pathlib import Path
import pytest
from hydra import initialize_config_dir,compose
from omegaconf import OmegaConf
from navsim.agents.EpisodeDrive.layers.world_model.task_future_training import validate_lite_config
from navsim.agents.EpisodeDrive.shared_planreg_initialization import classify_shared_trainable_parameter
from scripts.export_planreg_student_checkpoint import is_training_only_state_key


def test_formal_lite_config_and_export_classification():
    root=Path(__file__).resolve().parents[1]
    with initialize_config_dir(version_base=None,config_dir=str(root/'navsim/planning/script/config/common/agent')):
        cfg=compose(config_name='episode_drive_task_future_lite_base')
    validate_lite_config(cfg.world_model,cfg.consequence,cfg.scorer_variant)
    assert cfg.world_model.mode=='task_future_lite' and cfg.world_model.enabled
    assert cfg.world_model.min_weight>0 and cfg.world_model.candidate_count==8
    assert cfg.checkpoint_path is None and cfg.stage1_checkpoint_path is None
    assert cfg.vlm_config.tile_register_aggregation=='global_local_8_8'
    assert cfg.action_head_config.proposal_num==64
    assert classify_shared_trainable_parameter('physical_query_decoder.gap_head.weight')=='future_predictor'
    assert is_training_only_state_key('agent.physical_query_decoder.gap_head.weight')
    assert not is_training_only_state_key('agent.action_head.scorer.pred_layers.foo.weight')
    cfg.consequence.inference_use=True
    with pytest.raises(ValueError):validate_lite_config(cfg.world_model,cfg.consequence,cfg.scorer_variant)
