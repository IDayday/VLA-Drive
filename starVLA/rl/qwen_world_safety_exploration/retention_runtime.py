"""Regression-gated historical RL mean anchor within the original beta50 budget."""
import copy
import math
import time
import torch
import torch.distributed as dist
from starVLA.rl.qwen_world_research_algorithms.policy import ResearchLoss
from starVLA.rl.simwam_recipe import checkpoint
from .group_runtime import GroupRuntime
from .credit import GroupCredit
from .quality import QualityLossMetrics,PROTOCOL as QUALITY_PROTOCOL
from .standardized_recipe import StandardSchedulerLossMixin,validate_scheduler_state,peaks,PROTOCOL as SCHEDULE
from .retention_recipe import CONFIG,SCHEMA,PROTOCOL,validate_retention_recipe
from .retention_material import FrozenPairFeatures,nominal_means,probe,digest,validate_witness,verify_seal
from .retention_bank import new_bank,selection,make_payload,transition,validate_memory
from . import recovery


class RetentionLoss(StandardSchedulerLossMixin,QualityLossMetrics,ResearchLoss):
 def _policy_loss(self,model,rollout):
  parts={};observer=getattr(self,'record_policy_components',None)
  def capture(pg,anchor):parts.update(pg=pg,anchor=anchor)
  self.record_policy_components=capture
  try:total,metrics=super()._policy_loss(model,rollout)
  finally:
   if observer is None:del self.record_policy_components
   else:self.record_policy_components=observer
  p=rollout['verified_retention'];enabled=p['enabled'];history=total.new_zeros(())
  if enabled:
   teacher=p['teacher'];audit=model._probability_audit
   try:
    chain=teacher['chain'].to(model.device);times=teacher['timesteps'].to(model.device);deltas=teacher['deltas'].to(model.device)
    current=nominal_means(model,teacher['condition'],chain,times,deltas)
    target=teacher['nominal_mean'].detach().to(current)
    w=model.kl_weight_matrix(horizon=8,action_dim=3,device=current.device,dtype=current.dtype)
    error=(current-target).square()
    raw=error.mean()if w is None else ((error*w).sum((2,3))/w.sum().clamp_min(1e-12)).mean()
    history=float(model.grpo_kl_beta)*raw
   finally:model._probability_audit=audit
   # Only on the verified gate: replace25% of beta50, never add another beta.
   total=total+CONFIG['history_share']*(history-parts['anchor'])
  effective=(1-CONFIG['history_share']*int(enabled))*parts['anchor']+CONFIG['history_share']*int(enabled)*history
  if observer is not None:observer(parts['pg'],effective)
  dev=total.device
  metrics.update(kl_loss=effective.detach(),retention_enabled=total.new_tensor(float(enabled)),
   retention_fresh_IL_weighted_anchor=parts['anchor'].detach(),retention_RL_weighted_anchor=history.detach(),
   retention_effective_IL_share=total.new_tensor(1-CONFIG['history_share']*int(enabled)),
   retention_effective_RL_share=total.new_tensor(CONFIG['history_share']*int(enabled)),
   retention_bank_size=total.new_tensor(len(p['memory_before']['entries'])),
   retention_fresh_ODE_queries=total.new_tensor(int(p['fresh_witness']is not None)),
   retention_history_ODE_queries=total.new_tensor(int(p['history_witness']is not None)))
  if hasattr(model,'conditions'):
   metrics['retention_qwen_prefill_calls_process']=total.new_tensor(model.conditions.prefill_calls)
   metrics['retention_qwen_prefill_seconds_process']=total.new_tensor(model.conditions.prefill_seconds)
  return total,metrics


class RetentionRuntime(GroupRuntime):
 recipe_validator=staticmethod(validate_retention_recipe)
 def make_credit(self,cfg):self.recipe_validator(cfg);return GroupCredit(8)
 def make_scene_stream(self,cfg,args,world):self.stream=super().make_scene_stream(cfg,args,world);return self.stream
 def make_actor(self,cfg,args,device):
  actor,a,b=super().make_actor(cfg,args,device);actor._loss=RetentionLoss(cfg)
  self.conditions=FrozenPairFeatures(getattr(self,'conditions',actor.conditions));object.__setattr__(actor,'conditions',self.conditions)
  return actor,a,b
 def bind_training_state(self,state,rollout,identity):
  super().bind_training_state(state,rollout,identity)
  rank=dist.get_rank()if dist.is_initialized()else 0
  if 'retention'not in state:
   if state['collections']or state['updates']or rollout is not None:raise ValueError('Nonfresh run cannot initialize an empty retention bank')
   state['retention']=new_bank(identity['sha256'],rank)
 def collect(self,actor,condition,token,log,service,generator,**kwargs):
  rollout=super().collect(actor,condition,token,log,service,generator,**kwargs)
  memory=self.training_state['retention'];version=rollout['behavior_version']
  index,_=selection(memory,token);fresh=history=None;audit=actor._probability_audit
  try:
   if version['update']>=CONFIG['admission_start_update']:
    fresh=probe(actor,self.conditions,token,log,service,version)
   if index is not None:
    teacher=memory['entries'][index]
    history=probe(actor,self.conditions,teacher['token'],teacher['log'],service,version)
  finally:actor._probability_audit=audit
  rollout['verified_retention']=make_payload(memory,rollout,fresh,history,index)
  return rollout
 def on_collection_committed(self,rollout,state):
  p=rollout['verified_retention'];memory=transition(state['retention'],p)
  if digest(memory)!=p['memory_after_sha256']or memory['collections']!=state['collections']:raise ValueError('Retention bank commit changed')
  state['retention']=memory
 def public_training_state(self,state):
  result=super().public_training_state(state)
  if 'retention'in result:
   m=result['retention'];result['retention']={k:v for k,v in m.items()if k not in ('entries','rng_state')}
   result['retention'].update(bank_size=len(m['entries']),bank_sha256=digest(m),extra_ODE_queries_local=m['fresh_ODE_queries_local']+m['history_ODE_queries_local'])
  return result
 def validate_boundary(self,state,rollout,identity,optimizer=None,scheduler=None,*,legacy=False):
  if legacy:raise ValueError('Verified retention requires its complete independent native state')
  cfg=self.recipe_validator(identity['recipe']);summary=recovery.validate_boundary(state,rollout,identity,optimizer,scheduler,credit=self.credit,recipe_validator=self.recipe_validator)
  rank=dist.get_rank()if dist.is_initialized()else state['retention']['rank']
  allowed={t:self.stream.logs[t]for t in self.stream.tokens};m=state.get('retention')
  validate_memory(m,run=identity['sha256'],rank=rank,collections=state['collections'],allowed=allowed,
    condition_sha=identity['projected_condition_identity'],protocol=identity['score_protocol']['sha256'],maximum_update=state['updates']-state['next_inner'])
  if summary['pending']:
   p=rollout.get('verified_retention',{});verify_seal(p,'material_sha256')
   if (p.get('schema')!=SCHEMA or p.get('config')!=CONFIG or p['source_version']!=rollout['behavior_version']
       or p['collection']!=state['collections']or p['rank']!=rank or p['fresh_token']!=rollout['token']
       or p['memory_after_sha256']!=digest(m)):raise ValueError('Pending retention material version/commit mismatch')
   before=p['memory_before'];validate_memory(before,run=identity['sha256'],rank=rank,collections=state['collections']-1,allowed=allowed,
     condition_sha=identity['projected_condition_identity'],protocol=identity['score_protocol']['sha256'],maximum_update=p['source_version']['update'])
   if (p['fresh_witness']is not None)!=(p['source_version']['update']>=CONFIG['admission_start_update']):raise ValueError('Admission/probe start clock changed')
   for name in ('fresh_witness','history_witness'):
    w=p[name]
    if w is not None:
     validate_witness(w,identity['sha256'],allowed,identity['projected_condition_identity'],identity['score_protocol']['sha256'],expected_rank=rank)
     if w['source_version']!=p['source_version']:raise ValueError('Current ODE witness comes from another version')
     expected=rollout['token']if name=='fresh_witness'else p['teacher']['token']
     if w['token']!=expected:raise ValueError('Wrong current-probe scene')
   if p['teacher']is not None:
    validate_witness(p['teacher'],identity['sha256'],allowed,identity['projected_condition_identity'],identity['score_protocol']['sha256'],require_good=True,expected_rank=rank)
    if p['teacher']['source_version']['update']>=p['source_version']['update']:raise ValueError('Retention target must be a genuinely older policy version')
   if digest(transition(before,p))!=digest(m):raise ValueError('Retention reservoir/RNG/counters were not committed exactly')
  actual=None
  if scheduler is not None:
   ss=scheduler.state_dict()if hasattr(scheduler,'state_dict')else scheduler;actual=copy.deepcopy(ss)
   lr=validate_scheduler_state(ss,peaks(cfg),state['updates'])
   oo=optimizer.state_dict()if hasattr(optimizer,'state_dict')else optimizer
   groups=oo['param_groups']
   if(len(groups)!=2 or[g.get('name')for g in groups]!=['action_lora','scene_predictive_memory']or[g.get('initial_lr')for g in groups]!=peaks(cfg)
       or any(not math.isfinite(g['lr'])or abs(g['lr']-v)>1e-15 for g,v in zip(groups,lr))):raise ValueError('Retention native LR/scheduler identity mismatch')
  summary['verified_retention']=dict(protocol=CONFIG,schedule=SCHEDULE,scheduler=actual,rank=rank,collections=m['collections'],
    bank_sha256=digest(m),bank_size=len(m['entries']),pending_material_sha256=None if rollout is None else rollout['verified_retention']['material_sha256'])
  return summary
 def validate_ranks(self,summaries):
  super().validate_ranks(summaries)
  if [s['verified_retention']['rank']for s in summaries]!=list(range(8)):raise ValueError('Retention rank ownership changed')
  common=('protocol','schedule','scheduler','collections')
  for s in summaries:
   if {k:s['verified_retention'][k]for k in common}!={k:summaries[0]['verified_retention'][k]for k in common}:raise ValueError('Retention ranks disagree on protocol/native schedule/clock')
  return summaries
 def recipe_identity(self,cfg,args,condition_sha):
  v=super().recipe_identity(cfg,args,condition_sha);v['verified_retention_protocol']=copy.deepcopy(PROTOCOL);v['retention_schedule']=copy.deepcopy(SCHEDULE);v['quality_credit_diagnostics']=QUALITY_PROTOCOL
  return v


def retention_runtime(cfg):validate_retention_recipe(cfg);return RetentionRuntime()
