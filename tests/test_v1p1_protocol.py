from pathlib import Path
import copy
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from scripts.audit_formal_config_pair import audit_formal_config_pair
from navsim.agents.EpisodeDrive.layers.world_model import FutureRegisterPredictor
from navsim.agents.EpisodeDrive.layers.world_model.frozen_task_diagnostics import frozen_predictor_controls


def test_resolved_v1p1_pair_and_current_only_student(monkeypatch):
    for key,value in {'OPENSCENE_DATA_ROOT':'/data','NAVSIM_EXP_ROOT':'/experiments',
                      'PLANREG_BASE_VLM_PATH':'/base','PLANREG_VQA_VLM_PATH':'/vqa',
                      'PLANREG_SHARED_INIT':'/shared.pt','PLANREG_INPUT_CACHE':'/cache-v1p1',
                      'PLANREG_VLM_CHECKPOINT_SHA256':'weightsha','PLANREG_VLM_CONFIG_SHA256':'configsha',
                      'PLANREG_OUTPUT_DIR':'/output','PLANREG_INITIALIZATION_VARIANT':'base',
                      'PLANREG_FORMAL_VLM_PATH':'/base','PLANREG_STUDENT_CHECKPOINT':'/student.ckpt'}.items():
        monkeypatch.setenv(key,value)
    root=Path(__file__).resolve().parents[1]/'navsim/planning/script/config/training'
    with initialize_config_dir(version_base=None,config_dir=str(root)):
        base=compose(config_name='formal_planreg_wm_v1p1_training')
        vqa=compose(config_name='formal_planreg_wm_v1p1_training',overrides=['agent=episode_drive_planreg_wm_v1p1_vqa'])
        student=compose(config_name='formal_planreg_wm_v1p1_training',overrides=['agent=episode_drive_planreg_wm_v1p1_student'])
    result=audit_formal_config_pair(OmegaConf.to_container(base,resolve=True),OmegaConf.to_container(vqa,resolve=True))
    assert result['pair_equal_outside_allowlist']
    assert base.agent.vlm_config.tile_register_aggregation=='global_local_8_8'
    assert base.agent.action_head_config.semantic_query_init_std==.02
    assert base.agent.action_head_config.semantic_use_padding_mask
    assert base.agent.world_model.enabled and base.agent.world_model.min_weight>0
    assert base.agent.lr_args.reference_global_batch==64
    assert base.agent.lr_args.vision_qv_lora_lr==3e-5
    assert base.trainer.params.max_epochs==27 and base.trainer.params.limit_val_batches==0
    assert not student.agent.world_model.enabled and not student.agent.ema.enabled
    assert student.agent.initialization is None
    assert student.agent.vlm_config.exact_student_checkpoint


def test_frozen_controls_reject_training_and_accept_double_gt():
    predictor=FutureRegisterPredictor(hidden_dim=32,predictor_layers=2,num_heads=4)
    current=torch.randn(2,16,32)
    trajectory=torch.randn(2,8,3,dtype=torch.float64)
    future=torch.randn(2,3,16,32)
    with pytest.raises(ValueError,match='frozen predictor'):
        frozen_predictor_controls(predictor,current,trajectory,future,torch.ones(2))
    predictor.eval().requires_grad_(False)
    result=frozen_predictor_controls(predictor,current,trajectory,future,torch.ones(2))
    assert set(result)=={'correct','action_only','shuffle_current','copy_current','target_variance'}
    assert all(torch.isfinite(torch.tensor(v)) for v in result.values())


def test_new_readout_queries_are_no_decay():
    from test_planreg_optimizer_groups import _agent
    from navsim.agents.EpisodeDrive.layers.planning_registers.global_local_readout import GlobalLocalRegisterReadout
    agent=_agent()
    agent.backbone.planning_register_adapter=GlobalLocalRegisterReadout(256)
    optimizer=agent.get_optimizers()[0]
    no_decay={id(p) for g in optimizer.param_groups if g['weight_decay']==0 for p in g['params']}
    readout=agent.backbone.planning_register_adapter
    assert id(readout.global_queries) in no_decay and id(readout.local_queries) in no_decay
