"""Composition only: original retention credit/bank/loss with early-decay clock."""
import copy
import math
from .retention_runtime import RetentionRuntime,RetentionLoss
from .consolidation_recipe import (ConsolidationSchedulerLossMixin,validate_scheduler_state,
 peaks,PROTOCOL as SCHEDULE)
from .combined_consolidation_retention_recipe import CONFIG,validate_combined_recipe


class CombinedRetentionLoss(ConsolidationSchedulerLossMixin,RetentionLoss):
 """Only scheduler precedence changes; _policy_loss is exactly RetentionLoss."""
 pass


class CombinedRetentionRuntime(RetentionRuntime):
 recipe_validator=staticmethod(validate_combined_recipe)
 def make_actor(self,cfg,args,device):
  actor,a,b=super().make_actor(cfg,args,device)
  actor._loss=CombinedRetentionLoss(cfg)
  return actor,a,b
 def validate_boundary(self,state,rollout,identity,optimizer=None,scheduler=None,*,legacy=False):
  if (optimizer is None)!=(scheduler is None):raise ValueError('Combined native scheduler/Adam validated together')
  cfg=self.recipe_validator(identity['recipe'])
  # Reuse every original retained-witness/bank/credit/Adam-step check, excluding
  # only its fixed50k *scheduler* check; never forge an optimizer or oldclock.
  summary=super().validate_boundary(state,rollout,identity,optimizer,None,legacy=legacy)
  actual=None
  if scheduler is not None:
   ss=scheduler.state_dict()if hasattr(scheduler,'state_dict')else scheduler
   expected=validate_scheduler_state(ss,peaks(cfg),state['updates']);actual=copy.deepcopy(ss)
   oo=optimizer.state_dict()if hasattr(optimizer,'state_dict')else optimizer;groups=oo['param_groups']
   if (len(groups)!=2 or[g.get('name')for g in groups]!=['action_lora','scene_predictive_memory']
       or[g.get('initial_lr')for g in groups]!=peaks(cfg)
       or any(not math.isfinite(g['lr'])or abs(g['lr']-value)>1e-15 for g,value in zip(groups,expected))):
    raise ValueError('Combined actual Adam LR differs from unchanged early-decay schedule')
  summary['verified_retention']['schedule']=copy.deepcopy(SCHEDULE)
  summary['verified_retention']['scheduler']=actual
  summary['combined_consolidation_retention']=copy.deepcopy(CONFIG)
  return summary
 def validate_ranks(self,summaries):
  super().validate_ranks(summaries)
  if any(s.get('combined_consolidation_retention')!=CONFIG
      or s['verified_retention']['schedule']!=SCHEDULE for s in summaries):
   raise ValueError('Combined ranks must preserve exact retention and early schedule protocols')
  return summaries
 def recipe_identity(self,cfg,args,condition_sha):
  value=super().recipe_identity(cfg,args,condition_sha)
  value['retention_schedule']=copy.deepcopy(SCHEDULE)
  value['combined_consolidation_retention']=dict(**copy.deepcopy(CONFIG),
    initialization='fresh same complete IL70/Adam0/RNG42; no parentRL or native transplant',
    attribution='two-factor combination; independent fresh validation versus both single factors and samebudgetControl; gains cannot be added',
    future='20k decayhorizon thenfloor.1 through50k scheduler; reviewed runtimebudget separate, no stage reset')
  return value


def combined_runtime(cfg):validate_combined_recipe(cfg);return CombinedRetentionRuntime()
