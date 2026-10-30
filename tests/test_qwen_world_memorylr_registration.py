import copy
from pathlib import Path
import pytest
import yaml
from starVLA.rl.qwen_world_research.runtime import validate_recipe


def test_memory_lr_change_requires_separate_single_factor_registration():
    cfg=yaml.safe_load(Path('/mnt/project/qwen-world-rl-optimization-20261030/configs/PPO_FULL.yaml').read_text())
    candidate=copy.deepcopy(cfg)
    candidate['memory_learning_rate']=2e-5
    with pytest.raises(ValueError,match='Invalid bounded optimizer'):
        validate_recipe(candidate)
    candidate['research'].update(arm='PPO_MEMORYLR20',memory_lr_ablation='memory_lr2e-5_v1')
    validate_recipe(candidate)
    candidate['research']['anchor_ablation']='raw_batchstd_beta40_v1'
    with pytest.raises(ValueError):validate_recipe(candidate)
