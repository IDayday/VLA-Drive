"""One-factor FP32 final action projection; inherited fixed50k SafetyEP."""
import copy

from .credit import STRUCTURED_SCHEMA
from .group_runtime import CAMPAIGN
from .standardized_recipe import PROTOCOL as SCHEDULE, validate_standardized_recipe

ROOT = CAMPAIGN/'improvement_training_20261101'
ARM = 'IMPROVEMENT_FP32_OUTPUT_G8'
CONFIG = dict(schema='world_fp32_clean_action_projection_v1',
              output_projection_fp32=True, scope='final_action_output_linear_only',
              rollout_replay_reference_export_same_precision=True)
EVALUATE_UPDATES = [0, 2000, 5000, 10000]+list(range(12000, 50001, 2000))


def validate_precision_recipe(cfg):
    r = cfg.get('research', {})
    if (r.get('campaign_root') != str(ROOT) or r.get('arm') != ARM
            or r.get('standard_role') != 'fp32_output' or r.get('precision_training') != CONFIG
            or r.get('structured_advantage') != STRUCTURED_SCHEMA
            or cfg.get('evaluate_updates') != EVALUATE_UPDATES):
        raise ValueError('Only registered freshIL70/G8 FP32 final-projection experiment')
    for key in ('family', 'batch32', 'verified_retention', 'consolidation_training',
                'standardized_training', 'revival_training', 'curriculum', 'endpoint_policy',
                'success_memory', 'rescue_auxiliary', 'fpo'):
        if r.get(key) is not None:
            raise ValueError('FP32 projection must not bundle another modification: '+key)
    view = copy.deepcopy(cfg)
    view['research'].update(campaign_root=str(CAMPAIGN/'standard_training_20261101'),
                            arm='SAFETY_G8_CONTROL', standardized_training=copy.deepcopy(SCHEDULE))
    validate_standardized_recipe(view)
    return cfg


def build_recipe(base):
    cfg = copy.deepcopy(base)
    r = cfg['research']
    r.pop('standardized_training', None)
    r.update(campaign_root=str(ROOT), arm=ARM, standard_role='fp32_output',
             precision_training=copy.deepcopy(CONFIG),
             stage_gate_updates=[2000, 5000, 10000]+list(range(12000, 20001, 2000)))
    cfg['evaluate_updates'] = list(EVALUATE_UPDATES)
    return validate_precision_recipe(cfg)


def export_model_config(parent_config, rlmeta):
    """Execution precision is exported explicitly; sealed IL ancestry is intact."""
    config = copy.deepcopy(parent_config)
    if rlmeta is not None and rlmeta['recipe'].get('research', {}).get('precision_training') is not None:
        validate_precision_recipe(rlmeta['recipe'])
        config.setdefault('action', {})['output_projection_fp32'] = True
    return config


def validate_export_precision(meta):
    expected = export_model_config(meta['IL_parent']['identity']['model_config'], meta['RL_parent'])
    if meta['model_config'] != expected:
        raise ValueError('Export execution config differs from declared IL/RL precision')

