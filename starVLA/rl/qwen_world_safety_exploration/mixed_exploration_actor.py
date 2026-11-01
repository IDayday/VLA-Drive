"""Actual per-candidate W3 kernels; ordinal slots never masquerade as timesteps."""
import math
import torch
from starVLA.rl.qwen_world_postcot.scope_actor import ScopeRecipeActor
from starVLA.rl.qwen_world_research_algorithms.policy import ResearchLoss,policy_gradient_objective,ratio_diagnostics
from .standardized_recipe import StandardSchedulerLossMixin
from .quality import QualityLossMetrics
from .mixed_exploration_recipe import ROWS

class MixedScopeActor(ScopeRecipeActor):
 @torch.no_grad()
 def sample_action_chain(self,cond,action_horizon,deterministic=False,init_actions=None,velocity_use_ref=False,generator=None):
  if deterministic:return super().sample_action_chain(cond,action_horizon,deterministic=True,init_actions=init_actions,velocity_use_ref=velocity_use_ref,generator=generator)
  if action_horizon!=8 or init_actions is not None or velocity_use_ref:raise ValueError('Formal mixed sampler requires original8independent roots/currentpolicy')
  with self._context_call(cond)as condition:
   times,deltas=self.infer_action_scheduler.build_inference_schedule(10,self.device,torch.float32,shift_override=self.grpo_infer_shift)
   x=self._randn((8,8,3),self.device,torch.float32,generator);chain=[x.clone()]
   innovations=torch.stack([self._randn((8,8,3),self.device,torch.float32,generator)for _ in range(3)])
   clipped=innovations.clamp(-self.grpo_randn_clip,self.grpo_randn_clip);means=[[None]*3 for _ in range(2)];stds=[[None]*3 for _ in range(2)]
   for k in range(10):
    v=self.action_velocity(x,times[k].reshape(1).expand(8),condition)
    nxt=self.infer_action_scheduler.step(v,deltas[k],x)
    for branch,indices in enumerate((ROWS[0],ROWS[4])):
     if k in indices:
      q=indices.index(k);sl=slice(branch*4,(branch+1)*4)
      mean,std=self._sde_mean_std(x[sl],v[sl],float(times[k])/self.infer_action_scheduler.num_train_timesteps,float(deltas[k]))
      nxt[sl]=mean+std*clipped[q,sl];means[branch][q]=mean.clone();stds[branch][q]=std
    x=nxt;chain.append(x.clone())
   self.mixed_sampling_material=dict(raw_innovations=innovations.detach(),clipped_innovations=clipped.detach(),
    mean=torch.cat([torch.stack(m,1)for m in means]),std=torch.tensor(stds,device=self.device,dtype=torch.float32).repeat_interleave(4,0))
   return torch.stack(chain,1).detach(),times.detach(),deltas.detach(),torch.tensor(ROWS,device=self.device,dtype=torch.long)

 def action_chain_logprobs(self,cond,chain,timesteps,deltas,stoch_idx=None,return_terms=False,use_ref=False):
  indices=stoch_idx
  if indices is None or indices.shape!=(8,3)or indices.dtype!=torch.long:raise ValueError('Mixed probability needs true[8,3] candidate timesteps')
  if chain.shape!=(8,11,8,3)or not torch.equal(indices.cpu(),torch.tensor(ROWS)):raise ValueError('Fixed4late4early conditional kernels changed')
  logps=[];means=[];vels=[];stds=[];raws=[];clippeds=[]
  with self._context_call(cond)as condition:
   for q in range(3):
    k=indices[:,q];row=torch.arange(8,device=chain.device);x=chain[row,k];nxt=chain[row,k+1]
    v=self.action_velocity(x,timesteps[k],condition,use_ref=use_ref);mm=[];ss=[];peaks=[]
    for b in range(2):
     step=int(indices[b*4,q]);sl=slice(b*4,(b+1)*4)
     mean,std=self._sde_mean_std(x[sl],v[sl],float(timesteps[step])/self.infer_action_scheduler.num_train_timesteps,float(deltas[step]))
     mm.append(mean);ss.extend([std]*4);peaks.extend([-math.log(std)-.5*math.log(2*math.pi)]*4)
    mean=torch.cat(mm);std=torch.tensor(ss,device=x.device,dtype=torch.float32)
    raw=torch.distributions.Normal(mean.float(),std[:,None,None]).log_prob(nxt.float())
    lo=torch.tensor([p-self.grpo_logprob_clamp_width for p in peaks],device=x.device,dtype=torch.float32)
    hi=torch.tensor(peaks,device=x.device,dtype=torch.float32)
    clipped=raw.clamp(min=lo[:,None,None],max=hi[:,None,None])
    logps.append(clipped.mean((-1,-2)));means.append(mean);vels.append(v.float());stds.append(std);raws.append(raw);clippeds.append(clipped)
  result=torch.stack(logps,1);terms=dict(mean=torch.stack(means,1),vel=torch.stack(vels,1),std=torch.stack(stds,1))
  raw=torch.stack(raws,1);clipped=torch.stack(clippeds,1)
  self._probability_audit['reference'if use_ref else'current']=dict(raw_gaussian_logprob=raw.detach(),source_clamped_logprob=clipped.detach(),
    source_clamp_hit_rate=float((raw!=clipped).float().mean()),**{k:v.detach()for k,v in terms.items()},steps_per_candidate=indices.detach())
  return(result,terms)if return_terms else result

class MixedSourceMath(ResearchLoss):
 def _policy_loss(self,model,rollout):
  chain,times,deltas,indices=(rollout[k]for k in ('chain','timesteps','deltas','stoch_idx'))
  if model.grpo_denoising_discount!=1 or self._effective_kl_beta(model)!=50:raise ValueError('Mixed experiment preserves discount1 and beta50')
  new,terms=model.action_chain_logprobs(rollout['cond_bg'],chain,times,deltas,stoch_idx=indices,return_terms=True)
  discount=model.grpo_denoising_discount**(len(times)-1-indices.float())
  adv=rollout['advantages'].detach()[:,None]*discount
  if self.adv_clip_max is not None:adv=adv.clamp(-self.adv_clip_max,self.adv_clip_max)
  mask=rollout['adv_mask'].detach().to(new)
  pg,values=policy_gradient_objective(new,rollout['old_logp'],adv,mask,algorithm='ppo',clip_range=self.ppo_clip_range,cispo_upper=self.cispo_upper)
  extra=ratio_diagnostics(values,adv,mask,['ordinal0_late7_early3','ordinal1_late8_early6','ordinal2_late9_early8'],clip_range=self.ppo_clip_range,cispo_upper=self.cispo_upper)
  with torch.no_grad():
   strict=torch.tensor([all(r['metrics'][k]==1 for k in ('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','comfort'))and r['metrics']['score']>=.9 for r in rollout['records']],device=adv.device)
   weights=values['derivative_weight']/(mask.sum().clamp_min(1)*3)
   for b,name in enumerate(('late','early')):
    sl=slice(b*4,(b+1)*4);h=strict[sl];a=adv[sl];w=weights[sl]
    extra.update({f'mixed_{name}_strict_high_count':h.sum(),f'mixed_{name}_positive_high_count':((a[:,0]>0)&h).sum(),f'mixed_{name}_negative_high_count':((a[:,0]<0)&h).sum(),
      f'mixed_{name}_positive_effective_PPO_weight':w.clamp_min(0).sum(),f'mixed_{name}_negative_effective_PPO_weight':(-w).clamp_min(0).sum(),
      f'mixed_{name}_positive_high_effective_weight':w[h].clamp_min(0).sum(),f'mixed_{name}_negative_high_effective_weight':(-w[h]).clamp_min(0).sum(),
      f'mixed_{name}_ratio_mean':values['ratio'][sl].mean()})
  with torch.no_grad():_,reference=model.action_chain_logprobs(rollout['cond_bg'],chain,times,deltas,stoch_idx=indices,use_ref=True,return_terms=True)
  dm=(terms['mean']-reference['mean']).pow(2);dv=(terms['vel']-reference['vel']).pow(2)
  w=model.kl_weight_matrix(horizon=8,action_dim=3,device=dm.device,dtype=dm.dtype)
  mx=dm.mean((2,3))if w is None else(dm*w).sum((2,3))/w.sum().clamp_min(1e-12)
  mv=dv.mean((2,3))if w is None else(dv*w).sum((2,3))/w.sum().clamp_min(1e-12)
  if model.grpo_kl_type!='x':raise ValueError('Mixed keeps source nominalmean physicalanchor')
  anchor=mx.mean();total=pg+50*anchor
  observer=getattr(self,'record_policy_components',None)
  if observer is not None:observer(pg,50*anchor)
  extra.update(approx_kl=(rollout['old_logp']-new).mean().detach(),adv_kept_frac=mask.mean().detach(),kl_div=anchor.detach(),kl_x_norm=(mx/(2*terms['std'].square())).mean().detach(),kl_x=mx.mean().detach(),kl_v=mv.mean().detach())
  for i,value in enumerate(dm.mean((0,1,2)).tolist()):extra[f'kl_x_dim{i}']=torch.as_tensor(value,device=pg.device)
  return total,dict(policy_loss=pg.detach(),bc_loss=pg.new_zeros(()),kl_loss=(50*anchor).detach(),**extra)

class MixedLoss(StandardSchedulerLossMixin,QualityLossMetrics,MixedSourceMath):pass
