"""Fixed scene-exposure clock for the B64/G16 warmup/keep/cosine run."""
import copy
import math

import torch

PROTOCOL = dict(schema='world_B64_warm125_keep625_cos2500_floor01_horizon12500_v1',
                horizon_updates=12500, warmup_updates=125, keep_until_updates=625,
                cosine_decay_end_updates=2500, initial_lr_ratio=.001,
                min_lr_ratio=.1, source_scene_clock_multiplier=8)


def multiplier(updates):
    if type(updates) is not int or not 0 <= updates <= PROTOCOL['horizon_updates']:
        raise ValueError('Integrated optimizer update must be in [0,12500]')
    if updates <= 125:
        return .001 + .999 * updates / 125
    if updates <= 625:
        return 1.
    if updates >= 2500:
        return .1
    return .1 + .9 * (1 + math.cos(math.pi * (updates - 625) / 1875)) / 2


def peaks(cfg):
    return [cfg['learning_rate'], cfg['memory_learning_rate']]


def validate_scheduler_state(value, base_lrs, updates):
    expected = [p * multiplier(updates) for p in base_lrs]
    actual = value.get('_last_lr')
    if (value.get('integrated64_protocol') != PROTOCOL
            or value.get('base_lrs') != list(base_lrs)
            or value.get('last_epoch') != updates
            or value.get('_step_count') != updates + 1
            or not isinstance(actual, list) or len(actual) != 2
            or any(not isinstance(a, (int, float)) or not math.isfinite(a)
                   or abs(a - b) > 1e-15 for a, b in zip(actual, expected))):
        raise ValueError('Integrated warmup/keep/cosine state or Adam clock changed')
    return expected


class IntegratedCosineLR(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer):
        self.integrated64_protocol = copy.deepcopy(PROTOCOL)
        super().__init__(optimizer, last_epoch=-1)

    def get_lr(self):
        return [p * multiplier(self.last_epoch) for p in self.base_lrs]

    def load_state_dict(self, state_dict):
        validate_scheduler_state(state_dict, self.base_lrs, state_dict.get('last_epoch'))
        super().load_state_dict(copy.deepcopy(state_dict))


class IntegratedSchedulerLossMixin:
    def _build_scheduler(self, scheduler_type, total_train_steps, warmup_steps=0):
        if (scheduler_type, total_train_steps, warmup_steps) != ('cosine', 12500, 125):
            raise ValueError('Stage gates cannot restart the integrated scheduler')
        return IntegratedCosineLR(self.optimizer)
