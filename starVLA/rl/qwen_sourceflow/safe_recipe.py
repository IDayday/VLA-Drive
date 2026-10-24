"""Two explicitly registered safety adaptations of the effective native recipe.

The original recipe validator remains unchanged. X1 changes only the logical
longitudinal anchor. SCADV transfers the previous component/dual design for the
three official event costs; no learned noise, geometric proxy or resampling.
"""
import copy
import json
from pathlib import Path
import yaml
from .recipe import expected_recipe as base_recipe
from .safe_admission import TASK, require_dispatch

ROOT = Path('/mnt/project/qwen-sourceflow-safe-rl-20261024')
VARIANTS = ('R_CANONICAL_X1', 'R_CANONICAL_SCADV3')
COSTS = ('no_at_fault_collisions', 'drivable_area_compliance',
         'time_to_collision_within_bound')


def expected_recipe(variant):
    if variant not in VARIANTS:
        raise ValueError('Unregistered native safety variant')
    cfg = copy.deepcopy(base_recipe('R_CANONICAL'))
    cfg['variant'] = variant
    if variant == 'R_CANONICAL_X1':
        cfg['grpo']['train']['kl_dim_weights'] = [1., 1., 1.]
    else:
        cfg['grpo']['train']['adv_clip_max'] = None
        cfg['safe_objective'] = dict(
            schema='component_floor_official_events_dual_v1',
            cost_metrics=list(COSTS), cost_definition='I[official_metric < 1]',
            component_floors=[.01, 1e-8, 1e-8, 1e-8],
            component_std='scene population std, FP64 centered',
            multiplier_initial=[1., 1., 1.], dual_optimizer='Adam',
            dual_lr=.01, dual_nonnegative_projection=True,
            budget_population='fixed512 public-hard TRAIN scenes, initial SFT source SDE G8',
            scalarization='Z_PDMS - sum(lambda_snapshot_j * Z_cost_j)',
            post_scalarization_normalization=False, advantage_clip=None,
            numerical_alarm_abs_advantage=1e4, numerical_alarm_multiplier=1000.,
            dual_updates='once per fresh unfiltered global collection; before four fixed inners',
            dynamic_group_filtering=False, safety_guarantee=False)
    return cfg


def load_recipe(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg != expected_recipe(cfg.get('variant')):
        raise ValueError('Safety recipe differs from the entire preregistered contract')
    return cfg
