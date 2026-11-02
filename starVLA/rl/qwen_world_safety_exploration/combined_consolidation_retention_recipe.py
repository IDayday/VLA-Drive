"""Explicit fresh combination: unchanged verified retention + early cosine end."""
import copy
from .retention_recipe import (ROOT,CONFIG as RETENTION,ARM as RETENTION_ARM,
 validate_retention_recipe)
from .consolidation_recipe import PROTOCOL as SCHEDULE,EVALUATE_UPDATES

ARM='IMPROVEMENT_RETENTION_EARLY_DECAY_G8'
ROLE='retention_early_decay'
SCHEMA='world_verified_retention_plus_early_decay_v1'
CONFIG=dict(schema=SCHEMA,retention=copy.deepcopy(RETENTION),schedule=copy.deepcopy(SCHEDULE))


def validate_combined_recipe(cfg):
 r=cfg.get('research',{})
 if (r.get('campaign_root')!=str(ROOT)or r.get('arm')!=ARM or r.get('standard_role')!=ROLE
     or r.get('combined_consolidation_retention')!=CONFIG or r.get('verified_retention')!=RETENTION
     or cfg.get('evaluate_updates')!=EVALUATE_UPDATES):
  raise ValueError('Only explicit fresh H Retention+EarlyDecay combination recipe')
 for k in ('family','consolidation_training','standardized_training','revival_training','batch32','curriculum',
           'precision_training','mixed_exploration','endpoint_policy','fpo','rescue_auxiliary','success_memory'):
  if r.get(k)is not None:raise ValueError('No extra mechanism in Retention+Early combination: '+k)
 # Read-only validation view. Actual combo root/arm/protocol/schedule stay sealed.
 view=copy.deepcopy(cfg);v=view['research'];v.pop('combined_consolidation_retention')
 v.update(arm=RETENTION_ARM,standard_role='verified_retention')
 validate_retention_recipe(view)
 return cfg


def build_recipe(retention_cfg):
 cfg=copy.deepcopy(retention_cfg);r=cfg['research']
 r.update(campaign_root=str(ROOT),arm=ARM,standard_role=ROLE,combined_consolidation_retention=copy.deepcopy(CONFIG),
  stage_gate_updates=[2000,5000,10000,12000,14000,16000,18000,20000])
 cfg['evaluate_updates']=list(EVALUATE_UPDATES)
 return validate_combined_recipe(cfg)
