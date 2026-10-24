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
VARIANTS = ('R_CANONICAL_X1', 'R_CANONICAL_SCADV3', 'R_CANONICAL_WLA_ADV')
COSTS = ('no_at_fault_collisions', 'drivable_area_compliance',
         'time_to_collision_within_bound')


def expected_recipe(variant):
    if variant not in VARIANTS:
        raise ValueError('Unregistered native safety variant')
    cfg = copy.deepcopy(base_recipe('R_CANONICAL'))
    cfg['variant'] = variant
    if variant == 'R_CANONICAL_X1':
        cfg['grpo']['train']['kl_dim_weights'] = [1., 1., 1.]
    elif variant == 'R_CANONICAL_SCADV3':
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
    else:
        cfg['factorized_objective'] = dict(schema='DriveWLA_v1_equations8_9_advantage_only',
            paper='https://arxiv.org/html/2610.17903v1',
            metrics=['no_at_fault_collisions','drivable_area_compliance','ego_progress',
                     'time_to_collision_within_bound','comfort'],
            weights=[2.,2.,5.,5.,2.], epsilon=1e-8, shrinkage_tau=.1,
            moments='FP64 stable scene population moments over G8',
            combined_normalization='weighted component sum / sqrt(scene variance + epsilon)',
            shrinkage='canonical_PDMS_scene_variance / (variance + tau**2)',
            full_DriveWLA_recipe=False,
            kept_effective_SimWAM_sampler_anchor_optimizer=True,
            not_transferred=['world/language experts','ten stochastic steps','discount0.8','BC0.1','eight epochs'])
    return cfg


def load_recipe(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg != expected_recipe(cfg.get('variant')):
        raise ValueError('Safety recipe differs from the entire preregistered contract')
    return cfg
