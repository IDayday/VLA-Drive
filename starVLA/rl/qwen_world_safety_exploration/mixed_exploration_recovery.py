"""Native matrix-stochastic-window structure; no one-dimensional aliasing."""
import torch
from . import recovery as R
from .mixed_exploration_recipe import CONFIG,ROWS


def validate_structure(state,r,identity,optimizer=None,scheduler=None,*,legacy=False):
 if legacy:raise ValueError('New mixedW3 only accepts complete independent native state')
 #Shared counter/Adam rules are exactly inherited, with no fake rollout/index.
 summary=R.validate_structure(state,None,identity,optimizer,scheduler)if r is None else None
 if r is None:return summary
 cfg=identity['recipe'];u,c,i=(state[k]for k in ('updates','collections','next_inner'));phase=state['phase']
 if (identity['world']!=8 or not 0<=u<=50000 or i!=u%4 or not 0<=i<4 or phase not in ('ROLLOUT_COMMITTED','INNER_COMMITTED')
     or c!=u//4+1 or any(state[k]!=8*c for k in ('scene_cursor','raw_scene_exposures'))or state['new_candidates']!=64*c
     or(phase=='ROLLOUT_COMMITTED'and i!=0)or(phase=='INNER_COMMITTED'and i==0)):
  raise ValueError('Mixed native window/optimizer/candidate clock changed')
 descriptor=r['cond_bg']
 if descriptor!=dict(schema='world_postcot_trainable_memory_observation_v1',token=r['token'],identity=identity['projected_condition_identity'],repeats=8):raise ValueError('Mixed observation ancestry changed')
 for k,shape in dict(chain=(8,11,8,3),old_logp=(8,3),rewards=(1,8),advantages=(8,),adv_mask=(8,),timesteps=(10,),deltas=(10,)).items():
  x=r[k]
  if x.shape!=shape or x.dtype!=torch.float32 or x.requires_grad or not torch.isfinite(x).all():raise ValueError('Invalid mixed native detached '+k)
 indices=r['stoch_idx']
 if indices.dtype!=torch.long or not torch.equal(indices.cpu(),torch.tensor(ROWS)):raise ValueError('Actual[8,3] stochastic steps changed')
 if(r['group_size']!=8 or r['behavior_version']['run']!=identity['sha256']or r['behavior_version']['update']!=u-i):raise ValueError('Mixed collection-start policy version changed')
 old=r['old_probability']
 for k,shape in dict(mean=(8,3,8,3),vel=(8,3,8,3),std=(8,3),source_clamped_logprob=(8,3,8,3),raw_gaussian_logprob=(8,3,8,3)).items():
  x=old[k]
  if x.shape!=shape or x.dtype!=torch.float32 or x.requires_grad or not torch.isfinite(x).all():raise ValueError('Mixed old probability material changed '+k)
 if not torch.equal(old['steps_per_candidate'].cpu(),indices.cpu())or not torch.allclose(old['source_clamped_logprob'].mean((-1,-2)).cpu(),r['old_logp'].cpu(),atol=5e-7,rtol=0):raise ValueError('Old ordinal probability/actualtimesteps mismatch')
 sampling=r['mixed_sampling']
 if sampling.get('protocol')!=CONFIG:raise ValueError('Mixed kernel protocol changed')
 gen=torch.Generator();gen.set_state(sampling['rng_before'].cpu());root=torch.randn((8,8,3),generator=gen,dtype=torch.float32);innov=torch.stack([torch.randn((8,8,3),generator=gen,dtype=torch.float32)for _ in range(3)])
 if(not torch.equal(root,r['chain'][:,0].cpu())or not torch.equal(innov,sampling['raw_innovations'].cpu())
     or not torch.equal(gen.get_state(),sampling['rng_after'].cpu())or not torch.equal(innov.clamp(-5,5),sampling['clipped_innovations'].cpu())):raise ValueError('Mixed actual RNG/root/innovation contract changed')
 if(not torch.allclose(sampling['mean'].cpu(),old['mean'].cpu(),atol=1e-6,rtol=0)or not torch.equal(sampling['std'].cpu(),old['std'].cpu())):raise ValueError('Mixed actual sample/recompute nominalterms differ')
 for q in range(3):
  for b in range(2):
   sl=slice(b*4,(b+1)*4);k=ROWS[b*4][q]
   want=sampling['mean'][sl,q].cpu()+sampling['std'][sl,q,None,None].cpu()*sampling['clipped_innovations'][q,sl].cpu()
   if not torch.equal(want,r['chain'][sl,k+1].cpu()):raise ValueError('Saved transition not the actual row-specific sampled action')
 if optimizer is not None:
  opt=optimizer.state_dict()if hasattr(optimizer,'state_dict')else optimizer;steps=[float(s['step'])for s in opt['state'].values()]
  if(u and len(steps)!=sum(len(g['params'])for g in opt['param_groups']))or any(s!=u for s in steps):raise ValueError('Mixed native Adam update mismatch')
 if scheduler is not None:
  ss=scheduler.state_dict()if hasattr(scheduler,'state_dict')else scheduler
  if ss['last_epoch']!=u:raise ValueError('Mixed scheduler clock changed')
 return dict(phase=phase,updates=u,collections=c,scene_cursor=state['scene_cursor'],next_inner=i,run=identity['sha256'],pending=True,
  digests=dict(**R.rollout_digest(r),mixed_sampling=R.tensor_digest(sampling)),behavior=r['behavior_version'],rewards=r['rewards'].double().cpu().tolist(),
  advantages=r['advantages'].double().cpu().tolist(),mask=r['adv_mask'].cpu().tolist(),global_std=float(r['global_std']))
