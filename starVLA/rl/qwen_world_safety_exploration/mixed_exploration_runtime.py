"""Independent fixed-strata mixed-noise runtime; original runtimes untouched."""
import copy
import math
import time
import torch
from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
from starVLA.rl.qwen_world_postcot.loading import load_deployment_checkpoint
from starVLA.rl.qwen_world_research_data import ResearchSceneData
from .group_runtime import GroupRuntime
from .conditions import GroupMemoryConditions
from .quality import PROTOCOL as QUALITY_PROTOCOL
from .standardized_recipe import validate_scheduler_state,peaks,PROTOCOL as SCHEDULE
from .mixed_exploration_recipe import CONFIG,PROTOCOL,validate_mixed_recipe
from .mixed_exploration_actor import MixedScopeActor,MixedLoss
from .mixed_exploration_recovery import validate_structure
from . import recovery

class MixedRuntime(GroupRuntime):
 recipe_validator=staticmethod(validate_mixed_recipe)
 def make_actor(self,cfg,args,device):
  model,info=load_deployment_checkpoint(args.il_checkpoint,device=device)
  if info['model_sha256']!=self.parent['model_sha256']:raise ValueError('Pinned IL70 changed')
  head=model.actor
  if(head.hidden_dim,head.num_layers,head.action_dim)!=(1024,20,3):raise ValueError('Use original complete World actor')
  self.parent=info;actor=MixedScopeActor(head,cfg,scene_provider=model.scene_provider,predictive_provider=model.predictive_provider)
  actor._loss=MixedLoss(cfg)
  if actor.closure['trainable_numel']!=52707840:raise ValueError('Mixed retains original52.7M closure')
  data=ResearchSceneData(args.data,role='NAVTRAIN');self.conditions=GroupMemoryConditions(model,data,info['model_sha256'],actor,group_size=8)
  object.__setattr__(actor,'conditions',self.conditions);self.actor=actor
  if args.qualification:
   self.observation_before=self._frozen_observation_hashes();self.memory_before=self._current_memory_hashes()
  return actor,None,None
 @torch.no_grad()
 def collect(self,actor,condition,token,log,service,generator,*,variant,version,progress=None):
  if variant!='R_CANONICAL':raise ValueError('Canonical unchanged scoring only')
  actor._noise_audit=[];rng=generator.get_state().clone();tick=time.monotonic()
  chain,times,deltas,indices=actor.sample_action_chain(condition,8,generator=generator)
  sample=copy.deepcopy(actor.mixed_sampling_material)
  sample.update(protocol=copy.deepcopy(CONFIG),rng_before=rng,rng_after=generator.get_state().clone())
  if chain.is_cuda:torch.cuda.synchronize()
  rollout_seconds=time.monotonic()-tick
  if progress:progress('generated',rollout_seconds,8)
  requested=ActionCodec.denorm_odo(chain[:,-1]).cpu().numpy();tick=time.monotonic();records=service.score((token,log,requested));scoring=time.monotonic()-tick
  if len(records)!=8:raise ValueError('Exactly8actual mixed candidates scored')
  if progress:progress('scored',scoring,8)
  rewards=torch.tensor([[r['readouts']['canonical_PDMS']for r in records]],device=actor.device,dtype=torch.float32)
  tick=time.monotonic();old=actor.action_chain_logprobs(condition,chain,times,deltas,stoch_idx=indices).detach();audit=copy.deepcopy(actor._probability_audit['current'])
  mean_error=float((sample['mean']-audit['mean']).abs().max())
  if mean_error>1e-6 or not torch.equal(sample['std'],audit['std']):raise ValueError('Actual mixed BF16 sampler/replayed terms differ: '+str(mean_error))
  if chain.is_cuda:torch.cuda.synchronize()
  return dict(cond_bg=condition,chain=chain,timesteps=times,deltas=deltas,stoch_idx=indices,old_logp=old,old_probability=audit,mixed_sampling=sample,
   rewards=rewards,records=records,token=token,log=log,behavior_version=version,group_size=8,noise_audit=copy.deepcopy(actor._noise_audit),
   sampled_recomputed_mean_max_abs=mean_error,protocol_sha256=service.protocol['sha256'],seconds=dict(rollout=rollout_seconds,scoring_wait=scoring,old_probability=time.monotonic()-tick,scoring_cpu=sum(r['scoring_seconds']for r in records)))
 def validate_collection(self,r):
  if r['chain'].shape!=(8,11,8,3)or r['stoch_idx'].shape!=(8,3)or not torch.isfinite(r['chain']).all():raise ValueError('Invalid actual mixed matrix rollout')
 def validate_boundary(self,state,rollout,identity,optimizer=None,scheduler=None,*,legacy=False):
  if legacy:raise ValueError('Mixed kernel requires independent complete native state')
  cfg=self.recipe_validator(identity['recipe'])
  summary=recovery.validate_boundary(state,rollout,identity,optimizer,scheduler,credit=self.credit,recipe_validator=self.recipe_validator,_structure_validator=validate_structure)
  actual=None
  if scheduler is not None:
   ss=scheduler.state_dict()if hasattr(scheduler,'state_dict')else scheduler;actual=copy.deepcopy(ss);expected=validate_scheduler_state(ss,peaks(cfg),state['updates'])
   oo=optimizer.state_dict()if hasattr(optimizer,'state_dict')else optimizer
   if(len(oo['param_groups'])!=2 or[g.get('name')for g in oo['param_groups']]!=['action_lora','scene_predictive_memory']or[g.get('initial_lr')for g in oo['param_groups']]!=peaks(cfg)
      or any(not math.isfinite(g['lr'])or abs(g['lr']-v)>1e-15 for g,v in zip(oo['param_groups'],expected))):raise ValueError('Mixed Adam/schedule mismatch')
  summary['mixed_exploration']=dict(protocol=CONFIG,schedule=SCHEDULE,scheduler=actual)
  return summary
 def validate_ranks(self,summaries):
  super().validate_ranks(summaries)
  if any(s.get('mixed_exploration')!=summaries[0].get('mixed_exploration')for s in summaries):raise ValueError('Ranks disagree on mixed kernel/schedule')
  return summaries
 def recipe_identity(self,cfg,args,condition_sha):
  v=super().recipe_identity(cfg,args,condition_sha);v['mixed_exploration_protocol']=copy.deepcopy(PROTOCOL);v['mixed_schedule']=copy.deepcopy(SCHEDULE);v['quality_credit_diagnostics']=QUALITY_PROTOCOL
  v['exploration_protocol'].update(stochastic_steps='explicit[8,3]percandidate matrix',steps_per_candidate=copy.deepcopy(PROTOCOL['steps_per_candidate']))
  return v

def mixed_runtime(cfg):validate_mixed_recipe(cfg);return MixedRuntime()
