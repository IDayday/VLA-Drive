"""Independent fresh study: fixed optimizer-update warmup/cosine contract."""
import copy
import math
from pathlib import Path

import torch

from .group_runtime import CAMPAIGN,validate_recipe

ROOT=CAMPAIGN/'standard_training_20261101'
SCHEMA='world_fixed50k_warmup1000_cosine_floor01_v1'
PROTOCOL=dict(schema=SCHEMA,horizon_updates=50000,warmup_updates=1000,min_lr_ratio=.1)
ARMS=('SAFETY_G8_CONTROL','STANDARD_INVERSION_G8','STANDARD_COMBINED_G8','STANDARD_ENDPOINT_LR02_G8')


def validate_standardized_recipe(cfg):
    from .credit import STRUCTURED_SCHEMA
    from .inversion_credit import SCHEMA as INVERSION
    from .combined_credit import SCHEMA as COMBINED
    from .endpoint_runtime import validate_endpoint_recipe
    r=cfg.get('research',{});arm=r.get('arm')
    if (r.get('campaign_root')!=str(ROOT) or r.get('standardized_training')!=PROTOCOL or arm not in ARMS
            or cfg.get('max_steps')!=50000 or cfg.get('lr_warmup_steps')!=1000
            or cfg.get('lr_scheduler_type')!='cosine' or cfg.get('save_every_optimizer_updates')!=500
            or cfg.get('grpo',{}).get('sample',{}).get('group_size')!=8
            or r.get('data_policy')!='uniform_full' or r.get('curriculum') is not None):
        raise ValueError('Only the independently registered fixed50k warmup1000 cosine-floor.1 fresh G8 study')
    expected={ARMS[0]:STRUCTURED_SCHEMA,ARMS[1]:INVERSION,ARMS[2]:COMBINED,ARMS[3]:STRUCTURED_SCHEMA}
    endpoint=arm==ARMS[3]
    if (r.get('structured_advantage')!=expected[arm] or bool(r.get('endpoint_policy') is not None)!=endpoint
            or r.get('endpoint_learning_rate_scale')!=(.2 if endpoint else None)):
        raise ValueError('Standardized arm/credit/policy and explicit endpoint LRscale mismatch')
    if (cfg.get('learning_rate'),cfg.get('memory_learning_rate'))!=((1e-5,1e-6) if endpoint else (5e-5,5e-6)):
        raise ValueError('Fixed peak LR: W3original; endpoint explicit x.2 algorithm-scale exception')
    # Read-only validation view. The actual F/root/LR/schedule recipe is returned
    # unchanged and seals identity, optimizer groups and all native checkpoints.
    old=copy.deepcopy(cfg);old['research']['campaign_root']=str(CAMPAIGN)
    old.update(lr_warmup_steps=100,lr_scheduler_type='constant')
    (validate_endpoint_recipe if endpoint else validate_recipe)(old)
    return cfg


def multiplier(completed_updates):
    if (isinstance(completed_updates,bool) or not isinstance(completed_updates,int)
            or not 0<=completed_updates<=50000):raise ValueError('Completed optimizer update in[0,50000] required')
    u=completed_updates
    if u<=1000:return .001+.999*u/1000
    return .1+.9*(1+math.cos(math.pi*(u-1000)/49000))/2


def peaks(cfg):return [cfg['learning_rate'],cfg['memory_learning_rate']]


def validate_scheduler_state(value,base_lrs,updates):
    if type(updates) is not int or not 0<=updates<=50000:
        raise ValueError('Native scheduler requires completed optimizer update in[0,50000]')
    if (value.get('fixed_protocol')!=PROTOCOL or value.get('base_lrs')!=list(base_lrs)
            or value.get('last_epoch')!=updates or value.get('_step_count')!=updates+1):
        raise ValueError('Fixed50k schedule identity/peak LR/native update mismatch')
    expected=[p*multiplier(updates) for p in base_lrs]
    actual=value.get('_last_lr')
    if (not isinstance(actual,list) or len(actual)!=len(expected)
            or any(not math.isfinite(x) or abs(x-y)>1e-15 for x,y in zip(actual,expected))):
        raise ValueError('Persisted cosine LR does not match global optimizer update')
    return expected


class FixedHorizonCosineLR(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self,optimizer):
        self.fixed_protocol=copy.deepcopy(PROTOCOL)
        super().__init__(optimizer,last_epoch=-1)
    def get_lr(self):return [p*multiplier(self.last_epoch) for p in self.base_lrs]
    def load_state_dict(self,state_dict):
        # Validate BEFORE accepting serialized fields; never let a staged target
        # or a polluted state silently rewrite the fixed horizon or LR floor.
        validate_scheduler_state(state_dict,self.base_lrs,state_dict.get('last_epoch'))
        super().load_state_dict(copy.deepcopy(state_dict))


class StandardSchedulerLossMixin:
    """Only scheduler construction; no _policy_loss or gradient modification."""
    def _build_scheduler(self,scheduler_type,total_train_steps,warmup_steps=0):
        if (scheduler_type!='cosine' or total_train_steps!=50000 or warmup_steps!=1000):
            raise ValueError('Stage budgets cannot replace fixed50k/warmup1000 cosine horizon')
        return FixedHorizonCosineLR(self.optimizer)
