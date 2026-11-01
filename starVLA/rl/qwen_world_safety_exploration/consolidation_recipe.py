"""Fresh IL70 SafetyEP: fixed50k budget, cosine converged at20k then held.

The decay end is a separate scientific identity from the native budget horizon.
Legacy standardized/F/G recipes and scheduler states are never accepted here.
"""
import copy
import math
from pathlib import Path

import torch

from .group_runtime import CAMPAIGN, validate_recipe
from .credit import STRUCTURED_SCHEMA

ROOT = CAMPAIGN/'improvement_training_20261101'
ARM = 'IMPROVEMENT_EARLY_DECAY_G8'
FAMILY = 'consolidation_training'
SCHEMA = 'world_fixed50k_warmup1000_cosine_end20k_floor01_v1'
PROTOCOL = dict(schema=SCHEMA, horizon_updates=50000, warmup_updates=1000,
                cosine_decay_end_updates=20000, min_lr_ratio=.1)
EVALUATE_UPDATES = [2000, 5000, 10000]+list(range(12000, 50001, 2000))


def validate_consolidation_recipe(cfg):
    r = cfg.get('research', {})
    if (r.get('campaign_root') != str(ROOT) or r.get('arm') != ARM
            or r.get('family') != FAMILY or r.get('consolidation_training') != PROTOCOL
            or cfg.get('max_steps') != 50000 or cfg.get('lr_warmup_steps') != 1000
            or cfg.get('lr_scheduler_type') != 'cosine'
            or cfg.get('save_every_optimizer_updates') != 500
            or cfg.get('evaluate_updates') != EVALUATE_UPDATES
            or cfg.get('grpo', {}).get('sample', {}).get('group_size') != 8
            or r.get('data_policy') != 'uniform_full' or r.get('curriculum') is not None
            or r.get('structured_advantage') != STRUCTURED_SCHEMA
            or (cfg.get('learning_rate'), cfg.get('memory_learning_rate')) != (5e-5, 5e-6)):
        raise ValueError('Only fresh H early-decay SafetyEP/W3/G8/full103288/50k budget is registered')
    forbidden = ('standardized_training', 'revival_training', 'endpoint_policy',
                 'endpoint_learning_rate_scale', 'success_memory', 'success_auxiliary',
                 'rescue_auxiliary', 'rescue_manifest', 'fpo', 'fpo_auxiliary')
    if any(r.get(k) is not None for k in forbidden):
        raise ValueError('Consolidation must not bundle another schedule, sampler or auxiliary')
    # Strict validation view only. Actual cfg and identity retain H/20k-decay.
    old = copy.deepcopy(cfg)
    old['research']['campaign_root'] = str(CAMPAIGN)
    old.update(lr_warmup_steps=100, lr_scheduler_type='constant')
    validate_recipe(old)
    return cfg


def build_recipe(base_cfg):
    """Pure template helper, accepting only inherited baseline SafetyEP science.

    Copy original/F control cfg, never mutate it or write a live config/registry.
    Root-specific observation/manifest/cache bindings remain explicit in base_cfg.
    """
    cfg = copy.deepcopy(base_cfg)
    r = cfg['research']
    r.pop('standardized_training', None)
    r.pop('standard_role', None)
    r.pop('revival_training', None)
    r.update(campaign_root=str(ROOT), arm=ARM, family=FAMILY,
             consolidation_training=copy.deepcopy(PROTOCOL),
             stage_gate_updates=[2000, 5000, 10000, 12000, 14000, 16000, 18000, 20000])
    cfg.update(max_steps=50000, lr_warmup_steps=1000, lr_scheduler_type='cosine',
               save_every_optimizer_updates=500, evaluate_updates=list(EVALUATE_UPDATES))
    return validate_consolidation_recipe(cfg)


def multiplier(completed_updates):
    if type(completed_updates) is not int or not 0 <= completed_updates <= 50000:
        raise ValueError('Completed optimizer updates must be an integer in[0,50000]')
    if completed_updates <= 1000:
        return .001+.999*completed_updates/1000
    if completed_updates >= 20000:
        return .1
    return .1+.9*(1+math.cos(math.pi*(completed_updates-1000)/19000))/2


def peaks(cfg):
    return [cfg['learning_rate'], cfg['memory_learning_rate']]


def validate_scheduler_state(value, base_lrs, updates):
    if type(updates) is not int or not 0 <= updates <= 50000:
        raise ValueError('Consolidation scheduler update outside native50k identity')
    if (value.get('consolidation_protocol') != PROTOCOL
            or value.get('base_lrs') != list(base_lrs)
            or value.get('last_epoch') != updates or value.get('_step_count') != updates+1):
        raise ValueError('Consolidation schedule/peaks/native clock mismatch')
    expected = [p*multiplier(updates) for p in base_lrs]
    actual = value.get('_last_lr')
    if (not isinstance(actual, list) or len(actual) != len(expected)
            or any(not isinstance(x, (int, float)) or not math.isfinite(x)
                   or abs(x-y) > 1e-15 for x, y in zip(actual, expected))):
        raise ValueError('Persisted early-decay LR differs from completed-update clock')
    return expected


class ConsolidationCosineLR(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer):
        self.consolidation_protocol = copy.deepcopy(PROTOCOL)
        super().__init__(optimizer, last_epoch=-1)

    def get_lr(self):
        return [p*multiplier(self.last_epoch) for p in self.base_lrs]

    def load_state_dict(self, state_dict):
        validate_scheduler_state(state_dict, self.base_lrs, state_dict.get('last_epoch'))
        super().load_state_dict(copy.deepcopy(state_dict))


class ConsolidationSchedulerLossMixin:
    """Construct scheduler only; PG, anchor, sampler and optimizer are unchanged."""
    def _build_scheduler(self, scheduler_type, total_train_steps, warmup_steps=0):
        if (scheduler_type != 'cosine' or total_train_steps != 50000 or warmup_steps != 1000):
            raise ValueError('Runtime stage caps cannot replace the50k budget or1000 warmup')
        return ConsolidationCosineLR(self.optimizer)
