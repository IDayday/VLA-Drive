"""True four-scene/rank on-policy batches, query-clock cosine and strict recipe."""
import copy
import math
import torch
from .group_runtime import CAMPAIGN,validate_recipe
from .standardized_recipe import multiplier as source_multiplier

ROOT=CAMPAIGN/'improvement_training_20261101'
ARM='IMPROVEMENT_MULTISCENE32_G8'
SCHEMA='world_multiscene32_G8_sourcePPO_query_clock_v1'
CREDIT_SCHEMA='world_SafetyEP_multiscene32_group_credit_v1'
CONFIG=dict(schema=SCHEMA,scenes_per_rank=4,group_size=8,world_size=8,inner_updates=4,
 normalization_candidates=256,optimizer_horizon_updates=12500,optimizer_warmup_updates=250,
 initial_lr_ratio=.001,min_lr_ratio=.1,source_update_clock_multiplier=4)
PROTOCOL=dict(**CONFIG,normalization='one population std of original Safety rawA over world8*scene4*G8; fixed for4inners',
 loss_reduction='equal scene mean then equal rank mean; zeroPG scenes retain unmasked IL anchor',
 behavior='all4 fresh scenes share collection-start currentLoRA+Memory and one behavior version; no optimizer between collects',
 backward='four loss/4 microbackwards; no_sync includes forward+backward of first3; oneclip/Adam/scheduler perinner',
 source_probability='unchanged source relative-clamped FP32 dimension-mean W3 proxy, NOT endpoint likelihood',
 frozen_cache='four detached frozenQwen/IL-memory trees in host LRU; only active scene ondevice; rebuild currentMemory/KV every microforward',
 schedule='factor(u)=old50k_factor(4u), u=completed optimizer updates; initial.001, warm250, floor.1 at12500',
 budget='window32scene/256queries,4inner. 1250inner2 has10016scenes/80128queries, not nominal10000/80000')


def validate_batch32_recipe(cfg):
 r=cfg.get('research',{})
 if (r.get('campaign_root')!=str(ROOT) or r.get('arm')!=ARM or r.get('standard_role')!='multiscene32'
     or r.get('batch32')!=CONFIG or r.get('structured_advantage')!=CREDIT_SCHEMA
     or r.get('data_policy')!='uniform_full' or r.get('endpoint_policy') is not None
     or r.get('curriculum') is not None or r.get('standardized_training') is not None or r.get('revival_training') is not None):
  raise ValueError('Independent strict Batch32/G8 uniform sourcePPO recipe required')
 if (cfg.get('max_steps')!=12500 or cfg.get('lr_warmup_steps')!=250 or cfg.get('lr_scheduler_type')!='cosine'
     or cfg.get('gradient_accumulation_steps')!=4 or cfg['grpo']['train'].get('rollout_buffer_batches')!=4
     or cfg['grpo']['sample']['group_size']!=8 or cfg['grpo']['train']['num_inner_epochs']!=4
     or cfg['learning_rate']!=5e-5 or cfg['memory_learning_rate']!=5e-6
     or cfg['save_every_optimizer_updates']!=500
     or cfg['logical_budget']!=dict(fresh_global_rollouts=3125,scenes=100000,candidates=800000)):
  raise ValueError('Batch32 actual optimizer/collection/query-clock values changed')
 old=copy.deepcopy(cfg);old['research']['campaign_root']=str(CAMPAIGN)
 old.update(max_steps=50000,lr_warmup_steps=100,lr_scheduler_type='constant',gradient_accumulation_steps=1,
  logical_budget=dict(fresh_global_rollouts=12500,scenes=100000,candidates=800000))
 old['grpo']['train']['rollout_buffer_batches']=1
 validate_recipe(old)
 return cfg


def multiplier(u):
 if type(u) is not int or not 0<=u<=12500:raise ValueError('Batch32 completed optimizer update must be0..12500')
 return source_multiplier(4*u)


def validate_scheduler_state(value,cfg,u):
 expected=[cfg['learning_rate']*multiplier(u),cfg['memory_learning_rate']*multiplier(u)]
 if (value.get('batch32_protocol')!=CONFIG or value.get('last_epoch')!=u or value.get('_step_count')!=u+1
     or value.get('base_lrs')!=[cfg['learning_rate'],cfg['memory_learning_rate']]
     or len(value.get('_last_lr',[]))!=2 or any(not math.isfinite(a) or abs(a-b)>1e-15 for a,b in zip(value['_last_lr'],expected))):
  raise ValueError('Batch32 scheduler identity/global clock/usedLR changed')
 return expected


class Batch32CosineLR(torch.optim.lr_scheduler.LRScheduler):
 def __init__(self,optimizer):self.batch32_protocol=copy.deepcopy(CONFIG);super().__init__(optimizer,last_epoch=-1)
 def get_lr(self):return [b*multiplier(self.last_epoch) for b in self.base_lrs]
 def load_state_dict(self,state_dict):
  cfg=dict(learning_rate=self.base_lrs[0],memory_learning_rate=self.base_lrs[1])
  validate_scheduler_state(state_dict,cfg,state_dict.get('last_epoch'))
  super().load_state_dict(copy.deepcopy(state_dict))


class Batch32SchedulerLossMixin:
 def _build_scheduler(self,scheduler_type,total_train_steps,warmup_steps=0):
  if (scheduler_type,total_train_steps,warmup_steps)!=('cosine',12500,250):raise ValueError('Batch32 query-clock horizon cannot be replaced by stage target')
  return Batch32CosineLR(self.optimizer)
