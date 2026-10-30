import copy
from pathlib import Path

import pytest
import yaml

from starVLA.rl.qwen_world_research.runtime import validate_recipe


def test_beta40_requires_its_own_canonical_raw_std_full_data_arm():
    cfg = yaml.safe_load(Path('/mnt/project/qwen-world-rl-optimization-20261030/configs/PPO_FULL.yaml').read_text())
    validate_recipe(cfg)
    candidate = copy.deepcopy(cfg)
    candidate['grpo']['train']['kl_beta'] = 40
    with pytest.raises(ValueError, match='Unregistered inherited scientific change'):
        validate_recipe(candidate)
    candidate['research'].update(arm='PPO_ANCHOR40', anchor_ablation='raw_batchstd_beta40_v1')
    validate_recipe(candidate)
    candidate['research']['reward_transform'] = 'safe_comfort'
    with pytest.raises(ValueError, match='own RAW-batchstd'):
        validate_recipe(candidate)
