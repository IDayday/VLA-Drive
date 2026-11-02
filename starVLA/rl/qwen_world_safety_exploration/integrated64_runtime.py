"""Actual64scene/G16 fresh windows and eight microbackwards, no old-run edits."""
import copy
from collections import OrderedDict
from contextlib import nullcontext
import torch
import torch.distributed as dist
from starVLA.rl.simwam_recipe import checkpoint
from .group_runtime import GroupRuntime,collect_group
from .credit import GroupCredit
from .quality import PROTOCOL as QUALITY_PROTOCOL
from .retention_runtime import RetentionLoss
from .retention_material import digest
from .integrated64_recipe import CONFIG,PROTOCOL,SCHEMA,validate_integrated_recipe
from .integrated64_schedule import IntegratedSchedulerLossMixin
from .integrated64_credit import finalize_window,protocol
from . import integrated64_retention as history

class FrozenWindowFeatures:
 def __init__(self,base):self.base=base;self.cache=OrderedDict()
 def __getattr__(self,name):return getattr(self.base,name)
 def materialize(self,descriptor,device):
  token=descriptor['token']
  if token in self.cache:self.base.token=token;self.base.cached=checkpoint.device_tree(self.cache.pop(token),device)
  value=self.base.materialize(descriptor,device);self.cache[token]=checkpoint.cpu_tree(self.base.cached)
  while len(self.cache)>16:self.cache.popitem(last=False)
  return value

class IntegratedLoss(IntegratedSchedulerLossMixin,RetentionLoss):
 """Retention per-scene beta50 split unchanged; only new schedule precedence."""
 def _policy_loss(self,model,rollout):
  total,metrics=super()._policy_loss(model,rollout);event=rollout['verified_retention']
  #A repeated token reuses one deterministic currentB1 score within thiswindow;
  #its separately rebuilt differentiable historyloss is still a distinctslot.
  metrics['retention_history_ODE_queries']=total.new_tensor(int(event['history_scored_now']))
  metrics['retention_history_selected_slots']=total.new_tensor(int(event['teacher']is not None))
  return total,metrics

class IntegratedRuntime(GroupRuntime):
 recipe_validator=staticmethod(validate_integrated_recipe)
 def make_credit(self,cfg):self.recipe_validator(cfg);c=GroupCredit(16);c.protocol=protocol();return c
 def make_scene_stream(self,cfg,args,world):self.stream=super().make_scene_stream(cfg,args,world);return self.stream
 def make_actor(self,cfg,args,device):
  actor,a,b=super().make_actor(cfg,args,device);actor._loss=IntegratedLoss(cfg)
  self.conditions=FrozenWindowFeatures(getattr(self,'conditions',actor.conditions));object.__setattr__(actor,'conditions',self.conditions)
  return actor,a,b
 def bind_training_state(self,state,rollout,identity):
  super().bind_training_state(state,rollout,identity)
  if 'integrated_retention'not in state:
   if state['updates']or state['collections']or rollout is not None:raise ValueError('Integrated history starts empty only atfreshIL70')
   state['integrated_retention']=history.new_bank(identity['sha256'],dist.get_rank()if dist.is_initialized()else 0)
 @torch.no_grad()
 def collect(self,actor,condition,token,log,service,generator,*,variant,version,progress=None):
  rank=dist.get_rank()if dist.is_initialized()else 0;start=self.training_state['scene_cursor'];samples=[];positions=[];states=[generator.get_state().clone()]
  for b in range(8):
   t,pos=self.stream.at(start+8*b,rank)
   if b==0 and(t!=token or self.stream.logs[t]!=log):raise ValueError('Integrated first stream scene mismatch')
   cond=condition if b==0 else self.conditions.get(t,actor.device)
   sample=collect_group(actor,cond,t,self.stream.logs[t],service,generator,group_size=16,variant=variant,version=version,progress=progress)
   sample['stream_position']=pos;samples.append(sample);positions.append(pos);states.append(generator.get_state().clone())
  if len({s['token']for s in samples})!=8:raise ValueError('Eight distinct current scenes/rank required')
  window=dict(schema=SCHEMA,protocol=copy.deepcopy(CONFIG),samples=samples,rank=rank,group_size=16,window_start_cursor=start,window_positions=positions,generator_states=states,
   cond_bg=dict(schema='world_integrated64_conditions_v1',observations=[s['cond_bg']for s in samples]),chain=torch.stack([s['chain']for s in samples]),old_logp=torch.stack([s['old_logp']for s in samples]),
   token=token,log=log,tokens=[s['token']for s in samples],logs=[s['log']for s in samples],behavior_version=copy.deepcopy(version),records=[r for s in samples for r in s['records']],
   rewards=torch.cat([s['rewards']for s in samples]),protocol_sha256=service.protocol['sha256'],seconds={k:sum(s['seconds'][k]for s in samples)for k in samples[0]['seconds']})
  return history.attach_window(self.training_state['integrated_retention'],window,actor,self.conditions,service)
 def validate_collection(self,window):
  if window.get('schema')!=SCHEMA or len(window['samples'])!=8 or any(s['stoch_idx'].tolist()!=[7,8,9]or s['chain'].shape!=(16,11,8,3)for s in window['samples']):raise ValueError('Real8scene/G16/W3 chains required')
 def finalize_advantages(self,window):return finalize_window(window)
 def commit_collection_accounting(self,window,state,world):
  if world!=8 or window['window_start_cursor']!=state['scene_cursor']:raise ValueError('Integrated actual exposure clock mismatch')
  state['collections']+=1;state['raw_scene_exposures']+=64;state['new_candidates']+=1024;state['scene_cursor']+=64
 def on_collection_committed(self,window,state):
  material=window['integrated_retention']
  if material['memory_before']is not state['integrated_retention']and digest(material['memory_before'])!=digest(state['integrated_retention']):raise ValueError('Integrated bank changed beforecommit')
  after=history.replay(material)
  if after['windows']!=state['collections']:raise ValueError('Integrated window/bank commit clock differs')
  state['integrated_retention']=after
 def public_training_state(self,state):
  value=super().public_training_state(state)
  if 'integrated_retention'in value:
   m=value['integrated_retention'];value['integrated_retention']={k:v for k,v in m.items()if k not in ('entries','rng_state')}
   value['integrated_retention'].update(bank_size=len(m['entries']),bank_sha256=digest(m),extra_ODE_queries_local=m['fresh_ODE_queries_local']+m['history_ODE_queries_local'])
  return value
 def validate_boundary(self,*args,**kwargs):
  from .integrated64_recovery import validate_boundary
  return validate_boundary(*args,stream=getattr(self,'stream',None),**kwargs)
 def validate_ranks(self,summaries):
  from .integrated64_recovery import validate_ranks
  return validate_ranks(summaries)
 def backward_update(self,model,actor,window,state):
  import traceback
  totals=[];values=[];fresh=0.;clamp=[];prefills=self.conditions.prefill_calls
  for b,sample in enumerate(window['samples']):
   with(model.no_sync()if b<7 and hasattr(model,'no_sync')else nullcontext()):
    error=None
    try:
     total,metrics=model(sample);current=actor._probability_audit['current'];ratio=(current['source_clamped_logprob'].mean((-1,-2))-sample['old_logp']).exp();err=float((ratio-1).abs().max())
     if not torch.isfinite(total)or not torch.isfinite(ratio).all():raise FloatingPointError('Nonfinite integrated loss/ratio')
     if state['next_inner']==0 and err>1e-4:raise ValueError('Fresh saved-state ratio differs by '+str(err))
    except Exception:error=traceback.format_exc()
    if error and checkpoint.fatal_device_error(error):raise RuntimeError(error)
    checkpoint.check_errors(error);(total/8).backward()
   totals.append(total.detach());values.append({k:v.detach()for k,v in metrics.items()});fresh=max(fresh,err);clamp.append(float(current['source_clamp_hit_rate']))
  metrics={k:torch.stack([v[k]for v in values]).mean()for k in values[0]}
  for key in ('quality_strict_high_count','quality_positive_high_count','quality_negative_high_count','quality_feasible_count','retention_enabled'):
   if key in metrics:metrics['integrated64_total_'+key]=metrics[key]*8
  metrics.update(integrated64_scenes_per_rank=total.new_tensor(8.),integrated64_candidates_per_rank=total.new_tensor(128.),integrated64_global_normalization_candidates=total.new_tensor(1024.),
   integrated64_prefill_calls_update=total.new_tensor(self.conditions.prefill_calls-prefills),integrated64_prefill_calls_process=total.new_tensor(self.conditions.prefill_calls),integrated64_host_cache_entries=total.new_tensor(len(self.conditions.cache)))
  return torch.stack(totals).mean(),metrics,dict(fresh_error=fresh,source_clamp_hit_rate=sum(clamp)/8)
 def exploration_summary(self,window):
  from starVLA.rl.simwam_recipe.collection import exploration_summary
  material=window['integrated_retention'];events=material['events']
  return dict(schema=SCHEMA,scenes_per_rank=8,candidates_per_rank=128,tokens=window['tokens'],logs=window['logs'],groups=[exploration_summary(s)for s in window['samples']],normalization_candidates=1024,
   normalization_std=float(window['normalization_std']),fresh_B1_queries=sum(e['fresh_witness']is not None for e in events),history_B1_queries=sum(e['history_scored_now']for e in events),
   history_slots_selected=sum(e['teacher']is not None for e in events),history_gates=sum(e['enabled']for e in events))
 def recipe_identity(self,cfg,args,condition_sha):
  value=super().recipe_identity(cfg,args,condition_sha);value['integrated64_protocol']=copy.deepcopy(PROTOCOL);value['quality_credit_diagnostics']=QUALITY_PROTOCOL
  value['exploration_protocol'].update(global_normalization_candidates=1024,scenes_per_rank=8,global_scenes_per_collection=64,native_resume='complete8scene/G16 window+scene-shadow history+Adam/RNG/WKC only')
  return value

def integrated_runtime(cfg):validate_integrated_recipe(cfg);return IntegratedRuntime()
