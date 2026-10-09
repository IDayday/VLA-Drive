"""Protect matched recipes and the real tail-batch epoch observations."""
from copy import deepcopy
from pathlib import Path
import yaml

from tools.structured_world.run_nuscenes_group import observation_updates, scientific_config_digest

ROOT = Path(__file__).resolve().parents[2]


def test_nuscenes_groups_share_the_entire_recipe_except_registered_objective():
    recipes = []
    for group in ('G1_FULL_UNIFORM', 'G3_EVENT_LOCAL'):
        config = yaml.safe_load((ROOT/('configs/structured_world/nuscenes_'+group+'_seed42.yaml')).read_text())
        assert config['framework']['camera_count'] == 6
        action = config['framework']['action_model']
        assert action['action_horizon'] == action['future_action_window_size'] == 6
        assert action['num_inference_timesteps'] == 10
        assert action['repeated_diffusion_steps'] == 8
        assert config['trainer']['epochs'] == 24
        assert config['trainer']['max_train_steps'] == 17424
        assert config['trainer']['num_warmup_steps'] == 726
        config.pop('run_id')
        assert config['structured_world'].pop('group') == group
        recipes.append(config)
    assert recipes[0] == recipes[1]


def test_tail_batch_keeps_all_23230_scenes_at_every_epoch():
    # 725 full batches plus one 30-scene batch: no dropped training samples.
    assert 725*32+30 == 23230
    assert observation_updates(23230) == [4356, 8712, 13068, 17424]


def test_formal_review_binds_configuration_and_allows_only_registration_status_change(tmp_path):
    original = yaml.safe_load((ROOT/'configs/structured_world/nuscenes_G1_FULL_UNIFORM_seed42.yaml').read_text())
    pending, frozen, altered = [tmp_path/name for name in ('pending.yaml', 'frozen.yaml', 'altered.yaml')]
    pending.write_text(yaml.safe_dump(original))
    ready = deepcopy(original)
    ready['structured_world']['registration_status'] = 'frozen_after_common_calibration_and_profile'
    frozen.write_text(yaml.safe_dump(ready))
    assert scientific_config_digest(pending) == scientific_config_digest(frozen)
    ready['structured_world']['loss_weights']['future'] *= 2
    altered.write_text(yaml.safe_dump(ready))
    assert scientific_config_digest(altered) != scientific_config_digest(frozen)
