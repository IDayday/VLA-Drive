"""Four fresh scenes per rank, genuine DDP accumulation, one optimizer clock."""
import copy
from collections import OrderedDict
from contextlib import nullcontext
import torch
from .group_runtime import GroupRuntime,collect_group
from .credit import GroupCredit
from .quality import QualityLossMetrics,PROTOCOL as QUALITY_PROTOCOL
from .batch32_recipe import CONFIG,PROTOCOL,SCHEMA,validate_batch32_recipe,Batch32SchedulerLossMixin
from .batch32_credit import finalize_window,protocol
from . import batch32_recovery
from starVLA.rl.qwen_world_research_algorithms.policy import ResearchLoss
from starVLA.rl.simwam_recipe import checkpoint


class FrozenFeatureWindow:
 """LRU4 of detached Qwen/IL features only; never current memory or KV graphs."""
 def __init__(self,base):self.base=base;self.cache=OrderedDict()
 def __getattr__(self,name):return getattr(self.base,name)
 def materialize(self,descriptor,device):
  token=descriptor['token']
  if token in self.cache:
   self.base.token=token;self.base.cached=checkpoint.device_tree(self.cache.pop(token),device)
  value=self.base.materialize(descriptor,device)
  self.cache[token]=checkpoint.cpu_tree(self.base.cached)
  while len(self.cache)>4:self.cache.popitem(last=False)
  return value


class Batch32Loss(Batch32SchedulerLossMixin,QualityLossMetrics,ResearchLoss):pass


class Batch32Runtime(GroupRuntime):
 recipe_validator=staticmethod(validate_batch32_recipe)
 def make_scene_stream(self,cfg,args,world):
  self.stream=super().make_scene_stream(cfg,args,world)
  return self.stream
 def make_credit(self,cfg):
  self.recipe_validator(cfg);credit=GroupCredit(8);credit.protocol=protocol();return credit
 def make_actor(self,cfg,args,device):
  actor,a,b=super().make_actor(cfg,args,device)
  actor._loss=Batch32Loss(cfg)
  # Tests may attach the genuine provider directly without runtime.conditions.
  self.conditions=FrozenFeatureWindow(getattr(self,'conditions',actor.conditions))
  object.__setattr__(actor,'conditions',self.conditions)
  return actor,a,b
 @torch.no_grad()
 def collect(self,actor,condition,token,log,service,generator,*,variant,version,progress=None):
  import torch.distributed as dist
  rank=dist.get_rank()if dist.is_initialized()else 0
  start=self.training_state['scene_cursor'];samples=[];positions=[];states=[generator.get_state().clone()]
  for b in range(4):
   t,pos=self.stream.at(start+8*b,rank)
   if b==0 and(t!=token or self.stream.logs[t]!=log):raise ValueError('First Batch32 scene does not match committed stream slot')
   cond=condition if b==0 else self.conditions.get(t,actor.device)
   sample=collect_group(actor,cond,t,self.stream.logs[t],service,generator,group_size=8,variant=variant,version=version,progress=progress)
   sample['stream_position']=pos;samples.append(sample);positions.append(pos);states.append(generator.get_state().clone())
  if len({s['token']for s in samples})!=4:raise ValueError('Batch32 rank must use four distinct fresh scenes')
  window=dict(schema=SCHEMA,protocol=copy.deepcopy(CONFIG),samples=samples,rank=rank,group_size=8,
   window_start_cursor=start,window_positions=positions,generator_states=states,
   cond_bg=dict(schema='world_multiscene32_conditions_v1',observations=[s['cond_bg']for s in samples]),
   chain=torch.stack([s['chain']for s in samples]),old_logp=torch.stack([s['old_logp']for s in samples]),
   token=token,log=log,tokens=[s['token']for s in samples],logs=[s['log']for s in samples],
   behavior_version=copy.deepcopy(version),records=[r for s in samples for r in s['records']],
   rewards=torch.cat([s['rewards']for s in samples]),protocol_sha256=service.protocol['sha256'],
   seconds={k:sum(s['seconds'][k]for s in samples)for k in samples[0]['seconds']})
  return window
 def validate_collection(self,window):
  if window.get('schema')!=SCHEMA or len(window['samples'])!=4:raise ValueError('Four actual scored source chains required')
  if any(s['stoch_idx'].tolist()!=[7,8,9]or tuple(s['chain'].shape)!=(8,11,8,3)for s in window['samples']):raise ValueError('Batch32 retains W3/G8 per scene')
 def finalize_advantages(self,window):return finalize_window(window)
 def commit_collection_accounting(self,window,state,world):
  if world!=8 or window['window_start_cursor']!=state['scene_cursor']:raise ValueError('Window accounting clock mismatch')
  state['collections']+=1;state['raw_scene_exposures']+=32;state['new_candidates']+=256;state['scene_cursor']+=32
 def validate_boundary(self,*args,**kwargs):return batch32_recovery.validate_boundary(*args,stream=getattr(self,'stream',None),**kwargs)
 def validate_ranks(self,summaries):return batch32_recovery.validate_ranks(summaries)
 def backward_update(self,model,actor,window,state):
  """Owns ONLY forward/backward; trainer still owns oneclip/Adam/scheduler."""
  import traceback
  totals=[];values=[];fresh=0.;clamp=[]
  before_prefills=self.conditions.prefill_calls if hasattr(self,'conditions')else 0
  for b,sample in enumerate(window['samples']):
   # DDP no_sync MUST surround forward as well as backward, not backward alone.
   context=model.no_sync()if b<3 and hasattr(model,'no_sync')else nullcontext()
   with context:
    error=None
    try:
     total,metrics=model(sample)
     current=actor._probability_audit['current']
     ratio=(current['source_clamped_logprob'].mean((-1,-2))-sample['old_logp']).exp()
     local_error=float((ratio-1).abs().max())
     if not torch.isfinite(total)or not torch.isfinite(ratio).all():raise FloatingPointError('Nonfinite Batch32 source loss/ratio')
     if state['next_inner']==0 and local_error>1e-4:raise ValueError('Fresh saved-state ratio differs by '+str(local_error))
    except Exception:error=traceback.format_exc()
    if error and checkpoint.fatal_device_error(error):raise RuntimeError(error)
    checkpoint.check_errors(error)
    (total/4).backward()
   fresh=max(fresh,local_error);clamp.append(float(current['source_clamp_hit_rate']))
   totals.append(total.detach());values.append({k:v.detach()if torch.is_tensor(v)else torch.as_tensor(v,device=total.device)for k,v in metrics.items()})
  metrics={k:torch.stack([v[k]for v in values]).mean()for k in values[0]}
  # Quality source per-scene counts keep their original units; explicit totals
  # make all32 local candidates visible without silently changing old metrics.
  for key in ('quality_strict_high_count','quality_direction_high_count','quality_positive_high_count','quality_negative_high_count','quality_feasible_count'):
   if key in metrics:metrics['batch32_total_'+key]=metrics[key]*4
  metrics.update(batch32_scenes_per_rank=torch.tensor(4.,device=totals[0].device),batch32_candidates_per_rank=torch.tensor(32.,device=totals[0].device),batch32_global_normalization_candidates=torch.tensor(256.,device=totals[0].device))
  if hasattr(self,'conditions'):
   metrics.update(batch32_qwen_prefill_calls_total=torch.tensor(float(self.conditions.prefill_calls),device=totals[0].device),batch32_qwen_prefill_seconds_total=torch.tensor(self.conditions.prefill_seconds,device=totals[0].device),batch32_frozen_host_cache_entries=torch.tensor(float(len(self.conditions.cache)),device=totals[0].device))
   metrics['batch32_qwen_prefill_calls_update']=torch.tensor(float(self.conditions.prefill_calls-before_prefills),device=totals[0].device)
  return torch.stack(totals).mean(),metrics,dict(fresh_error=fresh,source_clamp_hit_rate=sum(clamp)/4)
 def exploration_summary(self,window):
  from starVLA.rl.simwam_recipe.collection import exploration_summary
  return dict(schema=SCHEMA,scenes_per_rank=4,candidates_per_rank=32,tokens=window['tokens'],
   groups=[exploration_summary(s)for s in window['samples']],normalization_candidates=256,
   normalization_std=float(window['normalization_std']),canonical_mean=float(window['rewards'].mean()),
   strict_high_candidates=sum(int(r['metrics']['score']>=.9 and all(r['metrics'][k]==1 for k in ('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','comfort')))for r in window['records']))
 def recipe_identity(self,cfg,args,condition_sha):
  v=super().recipe_identity(cfg,args,condition_sha)
  v['multiscene32_protocol']=copy.deepcopy(PROTOCOL);v['quality_credit_diagnostics']=copy.deepcopy(QUALITY_PROTOCOL)
  v['exploration_protocol'].update(global_normalization_candidates=256,scenes_per_rank=4,global_scenes_per_collection=32,
   native_resume='complete four-scene same-behavior window/Adam/RNG/scheduler only; no single-scene resume')
  v['scene_reduction']='equal mean of four source scene losses then DDP rank mean; one Adam perinner'
  return v


def batch32_runtime(cfg):validate_batch32_recipe(cfg);return Batch32Runtime()
