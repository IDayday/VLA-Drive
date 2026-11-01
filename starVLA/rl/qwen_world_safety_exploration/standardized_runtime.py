"""Explicit fresh fixed50k training; legacy runtimes and policy losses unchanged."""
import copy
from functools import partial
import math

from .standardized_recipe import (validate_standardized_recipe,PROTOCOL,StandardSchedulerLossMixin,
    validate_scheduler_state,peaks)
from .credit import GroupCredit,STRUCTURED_SCHEMA
from .group_runtime import GroupRuntime
from . import recovery


class StandardizedRuntimeMixin:
 recipe_validator=staticmethod(validate_standardized_recipe)
 def make_credit(self,cfg):
  self.recipe_validator(cfg)
  from .inversion_credit import InversionCredit,SCHEMA as INVERSION
  from .combined_credit import CombinedCredit,SCHEMA as COMBINED
  schema=cfg['research']['structured_advantage']
  if schema==STRUCTURED_SCHEMA:return GroupCredit(8)
  if schema==COMBINED:return CombinedCredit(8)
  if schema==INVERSION:
   class StandardInversionCredit(InversionCredit):validate_recipe=staticmethod(validate_standardized_recipe)
   return StandardInversionCredit(8)
  raise ValueError('Unknown standardized credit')
 def make_actor(self,cfg,args,device):
  actor,a,b=super().make_actor(cfg,args,device)
  loss_class=type(actor._loss)
  class ScheduledLoss(StandardSchedulerLossMixin,loss_class):pass
  actor._loss=ScheduledLoss(cfg)
  return actor,a,b
 def validate_boundary(self,state,rollout,identity,optimizer=None,scheduler=None,*,legacy=False):
  cfg=self.recipe_validator(identity['recipe'])
  if legacy:raise ValueError('New standard study requires complete native schema2')
  from .combined_credit import SCHEMA as COMBINED,validate_combined_boundary
  from .inversion_credit import SCHEMA as INVERSION,validate_inversion_boundary
  custom_boundary=getattr(self,'credit_boundary_validator',None)
  if custom_boundary is not None:
   summary=custom_boundary(state,rollout,identity,optimizer,scheduler,credit=self.credit,
       _recipe_validator=self.recipe_validator)
  elif cfg['research'].get('endpoint_policy') is not None:
   from .endpoint_runtime import validate_endpoint_structure
   structure=partial(validate_endpoint_structure,_recipe_validator=self.recipe_validator)
   summary=recovery.validate_boundary(state,rollout,identity,optimizer,scheduler,credit=self.credit,
       recipe_validator=self.recipe_validator,_structure_validator=structure)
  elif cfg['research']['structured_advantage']==COMBINED:
   summary=validate_combined_boundary(state,rollout,identity,optimizer,scheduler,credit=self.credit)
  elif cfg['research']['structured_advantage']==INVERSION:
   summary=validate_inversion_boundary(state,rollout,identity,optimizer,scheduler,credit=self.credit)
  else:
   summary=recovery.validate_boundary(state,rollout,identity,optimizer,scheduler,credit=self.credit,recipe_validator=self.recipe_validator)
  # Save and BEFORE-restore validation accept both objects and their native dicts.
  actual=None
  if scheduler is not None:
   ss=scheduler.state_dict() if hasattr(scheduler,'state_dict') else scheduler
   expected=validate_scheduler_state(ss,peaks(cfg),state['updates'])
   actual=copy.deepcopy(ss)
   if optimizer is None:raise ValueError('Native scheduler validation requires its paired Adam state')
   oo=optimizer.state_dict() if hasattr(optimizer,'state_dict') else optimizer
   groups=oo['param_groups']
   if (len(groups)!=2 or [g.get('name') for g in groups]!=['action_lora','scene_predictive_memory']
       or [g.get('initial_lr') for g in groups]!=peaks(cfg)
       or any(not math.isfinite(g['lr']) or abs(g['lr']-e)>1e-15 for g,e in zip(groups,expected))):
    raise ValueError('Native Adam LR/initialpeak differs from fixed global cosine state')
  summary['standardized_training']=dict(protocol=copy.deepcopy(PROTOCOL),peak_lrs=peaks(cfg),scheduler_state=actual)
  return summary
 def validate_ranks(self,summaries):
  if len(summaries)!=8 or any(s.get('standardized_training')!=summaries[0].get('standardized_training') for s in summaries):
   raise ValueError('Eight ranks must share exact fixed50k schedule and optimizer boundary')
  if summaries[0]['standardized_training']['protocol']!=PROTOCOL:raise ValueError('Standard schedule changed')
  return super().validate_ranks(summaries)
 def recipe_identity(self,cfg,args,condition_sha):
  v=super().recipe_identity(cfg,args,condition_sha)
  v['standardized_training_protocol']=dict(**PROTOCOL,update_unit='optimizer updates; u=completed updates before next step',
      warmup_formula='factor=.001+.999*u/1000 for0<=u<=1000',
      decay_formula='factor=.1+.9*(1+cos(pi*(u-1000)/49000))/2 for1000<u<=50000',
      peak_action_lr=cfg['learning_rate'],peak_memory_lr=cfg['memory_learning_rate'],
      native_resume='same fixed50k schedule/Adam/RNG/cursor; stage cap never resets epoch/horizon; no older-study native fork',
      phase_policy='fresh same IL70/Adam/RNG42/update0; initial10k registered budget;2k/5k observations not alternate schedule horizons')
  return v


def standardized_runtime(cfg):
 validate_standardized_recipe(cfg)
 from .quality import QualityDiagnosticsMixin
 from .endpoint_runtime import EndpointRuntimeMixin
 from .inversion_credit import InversionRuntimeMixin,SCHEMA as INVERSION
 from .combined_credit import CombinedRuntimeMixin,SCHEMA as COMBINED
 if cfg['research'].get('endpoint_policy') is not None:mixins=(StandardizedRuntimeMixin,EndpointRuntimeMixin,GroupRuntime)
 else:
  credit=cfg['research']['structured_advantage'];extra=[]
  if credit==INVERSION:extra=[InversionRuntimeMixin]
  elif credit==COMBINED:extra=[CombinedRuntimeMixin]
  mixins=tuple([StandardizedRuntimeMixin,QualityDiagnosticsMixin]+extra+[GroupRuntime])
 return type('StandardizedFreshRuntime',mixins,{})()
