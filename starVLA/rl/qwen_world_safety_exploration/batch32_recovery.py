"""Strict four-scene native transactions; source single-scene structure reused."""
import copy
import math
import torch
import numpy as np
from . import recovery as R
from .credit import canonical_components,raw_advantage_parts,GroupCredit
from .batch32_credit import protocol
from .batch32_recipe import CONFIG,SCHEMA,CREDIT_SCHEMA,validate_batch32_recipe,validate_scheduler_state


def material_digest(value):
 def native(x):
  if isinstance(x,np.ndarray):return torch.from_numpy(x)
  if isinstance(x,np.generic):return x.item()
  if isinstance(x,dict):return {k:native(v)for k,v in x.items()}
  if isinstance(x,(list,tuple)):return [native(v)for v in x]
  return x
 return R.tensor_digest(native(value))


def validate_boundary(state,window,identity,optimizer=None,scheduler=None,*,stream=None,legacy=False):
 if legacy:raise ValueError('Batch32 requires its own complete native state')
 cfg=validate_batch32_recipe(identity['recipe']);u,c,i=(state[k]for k in ('updates','collections','next_inner'))
 pending=state.get('phase') in ('ROLLOUT_COMMITTED','INNER_COMMITTED')
 if (identity['world']!=8 or type(u)is not int or not 0<=u<=12500 or not 0<=i<4 or i!=u%4
     or c!=u//4+int(pending) or state.get('phase')not in ('READY','ROLLOUT_COMMITTED','INNER_COMMITTED','COMPLETE')
     or (state['phase']=='COMPLETE' and u!=12500) or (state['phase']=='ROLLOUT_COMMITTED' and i!=0)
     or (state['phase']=='INNER_COMMITTED' and i==0) or pending!=(window is not None)
     or any(state[k]!=32*c for k in ('scene_cursor','raw_scene_exposures')) or state['new_candidates']!=256*c):
  raise ValueError('Batch32 actual window/optimizer/query/phase clock mismatch')
 virtual=dict(state,scene_cursor=c*8,raw_scene_exposures=c*8,new_candidates=c*64)
 summary=dict(schema=SCHEMA,protocol=CONFIG,phase=state['phase'],updates=u,collections=c,next_inner=i,
  scene_cursor=state['scene_cursor'],run=identity['sha256'],pending=pending)
 # READY also validates Adam counters, without manufacturing a rollout.
 if not pending:R.validate_structure(virtual,None,identity,optimizer,scheduler)
 if optimizer is not None and scheduler is not None:
  opt=optimizer.state_dict()if hasattr(optimizer,'state_dict')else optimizer
  sch=scheduler.state_dict()if hasattr(scheduler,'state_dict')else scheduler
  expected=validate_scheduler_state(sch,cfg,u);groups=opt['param_groups']
  if (len(groups)!=2 or [g.get('name')for g in groups]!=['action_lora','scene_predictive_memory']
      or [g.get('initial_lr')for g in groups]!=[5e-5,5e-6]
      or any(not math.isfinite(g['lr'])or abs(g['lr']-e)>1e-15 for g,e in zip(groups,expected))):
   raise ValueError('Actual Adam groups do not match inherited Batch32 global-clock schedule')
  summary['scheduler']=copy.deepcopy(sch)
 if not pending:return summary
 if (window.get('schema')!=SCHEMA or window.get('protocol')!=CONFIG or window.get('group_size')!=8
     or len(window.get('samples',[]))!=4 or window.get('window_start_cursor')!=state['scene_cursor']-32
     or window.get('rank')not in range(8) or len(set(window.get('tokens',[])))!=4):
  raise ValueError('Native four-scene window identity changed')
 samples=window['samples'];rank=window['rank'];leaf_summaries=[];components=[]
 if (window.get('cond_bg')!={'schema':'world_multiscene32_conditions_v1','observations':[s['cond_bg']for s in samples]}
     or window.get('tokens')!=[s['token']for s in samples] or window.get('logs')!=[s['log']for s in samples]
     or window.get('token')!=samples[0]['token'] or window.get('log')!=samples[0]['log']
     or window.get('records')is None or len(window['records'])!=32
     or material_digest(window['records'])!=material_digest([r for s in samples for r in s['records']])):
  raise ValueError('Four actual observations/records are not bound to the window')
 for b,s in enumerate(samples):
  leaf=R.validate_structure(virtual,s,identity,optimizer if b==0 else None,scheduler if b==0 else None)
  if (s['behavior_version']!=window['behavior_version'] or s.get('protocol_sha256')!=identity['score_protocol']['sha256']
      or s.get('stream_position')!=window['window_positions'][b]):raise ValueError('Mixed behavior/scoring/stream material')
  if stream is not None:
   token,pos=stream.at(window['window_start_cursor']+b*8,rank)
   if s['token']!=token or s['log']!=stream.logs[token] or s['stream_position']!=pos:raise ValueError('Four scene cursor slots changed')
  for r in s['records']:
   if r.get('token')!=s['token'] or r.get('log')!=s['log'] or r.get('protocol_sha256')!=s['protocol_sha256']:
    raise ValueError('Canonical record scene/protocol identity changed')
  m=canonical_components(s['records'],8)
  if not torch.equal(s['rewards'].cpu(),m['canonical_rewards'])or not torch.equal(s['canonical_rewards'].cpu(),m['canonical_rewards']):raise ValueError('Canonical feedback changed')
  expected=dict(components=m['components'],**raw_advantage_parts(m['components']));saved=s['structured_advantage']
  if saved.get('protocol')!=protocol()or set(saved)!=set(expected)|{'protocol'}:raise ValueError('Batch32 Safety parts protocol changed')
  for k,want in expected.items():
   value=saved[k]
   if (value.requires_grad or value.dtype!=want.dtype or value.shape!=want.shape or not torch.allclose(value.cpu(),want,atol=1e-14,rtol=0)):
    raise ValueError('Saved Safety material changed: '+k)
  for k,sl in [('advantages',slice(b*8,(b+1)*8)),('adv_mask',slice(b*8,(b+1)*8))]:
   if not torch.equal(window[k][sl].cpu(),s[k].cpu()):raise ValueError('Window/leaf finalized advantages differ')
  for k in ('global_std','normalization_std'):
   if not torch.equal(window[k].cpu(),s[k].cpu()):raise ValueError('A different scale was used for one microbatch')
  raw=saved['A_raw'].cpu();scale=s['normalization_std'].cpu()
  want_mask=raw.ne(0).any(1,keepdim=True).expand_as(raw).reshape(-1).float()
  want_adv=(raw/(scale+1e-8)).reshape(-1).float()
  want_std=(m['canonical_rewards'].double()-m['canonical_rewards'].double()[:,:1]).std(1,unbiased=False,keepdim=True)
  if (not torch.equal(s['adv_mask'].cpu(),want_mask)
      or not torch.allclose(s['advantages'].cpu(),want_adv,atol=1e-7,rtol=1e-6)
      or not torch.allclose(s['raw_group_std'].cpu(),want_std,atol=1e-12,rtol=0)):
   raise ValueError('Saved microbatch normalized advantages/mask/reward std changed')
  for k in expected:
   if not torch.equal(window['structured_advantage'][k][b:b+1].cpu(),saved[k].cpu()):raise ValueError('Window/leaf raw Safety parts differ')
  if s['advantage_statistics']!=window['advantage_statistics']:raise ValueError('Window/leaf population metadata differs')
  components.append(m['components']);leaf_summaries.append(leaf)
 for k in ('chain','old_logp'):
  if not torch.equal(window[k].cpu(),torch.stack([s[k].cpu()for s in samples])):raise ValueError('Native stacked '+k+' changed')
 for k in ('rewards','canonical_rewards','raw_group_std'):
  if not torch.equal(window[k].cpu(),torch.cat([s[k].cpu()for s in samples])):raise ValueError('Native stacked '+k+' changed')
 meta=window['advantage_statistics'];scale=window['normalization_std']
 if (meta.get('structured_advantage_schema')!=CREDIT_SCHEMA or meta.get('normalization_population_candidates')!=256
     or meta.get('normalization_population_groups')!=32 or scale.shape!=()or scale.dtype!=torch.float64
     or scale.requires_grad or not torch.isfinite(scale)or scale<0 or window['structured_advantage']['protocol']!=protocol()
     or 'reward_transform'in window):raise ValueError('Batch32 normalization identity changed')
 states=window.get('generator_states')
 if not isinstance(states,list)or len(states)!=5:raise ValueError('All four generator transitions required')
 for b,s in enumerate(samples):
  gen=torch.Generator(device='cpu');gen.set_state(states[b].cpu())
  root=torch.randn((8,8,3),generator=gen,dtype=torch.float32)
  if not torch.equal(root,s['chain'][:,0].cpu()):raise ValueError('Actual first latent is not the saved CPU generator draw')
  for _ in range(3):torch.randn((8,8,3),generator=gen,dtype=torch.float32)
  if not torch.equal(gen.get_state(),states[b+1].cpu()):raise ValueError('Window sampling RNG transition changed')
 if window.get('stream_position')!=window['window_positions'][0]:raise ValueError('First stream slot lost')
 summary.update(rank=rank,behavior=window['behavior_version'],rewards=window['rewards'].cpu().double().tolist(),
  components=torch.cat(components).tolist(),advantages=window['advantages'].cpu().reshape(4,8).tolist(),
  mask=window['adv_mask'].cpu().reshape(4,8).tolist(),normalization_std=float(scale),global_std=float(window['global_std']),
  original_raw_A=window['structured_advantage']['A_raw'].cpu().tolist(),digests=material_digest(window))
 return summary


def validate_ranks(summaries):
 if len(summaries)!=8:raise ValueError('Batch32 needs exactly eight native ranks')
 common=('schema','protocol','phase','updates','collections','next_inner','scene_cursor','run','pending','scheduler')
 for s in summaries:
  if {k:s.get(k)for k in common}!={k:summaries[0].get(k)for k in common}:raise ValueError('Batch32 ranks disagree on native boundary')
 if not summaries[0]['pending']:return summaries
 if [s['rank']for s in summaries]!=list(range(8))or any(s['behavior']!=summaries[0]['behavior']for s in summaries):raise ValueError('Eight actual ordered ranks/same behavior required')
 rewards=torch.tensor([s['rewards']for s in summaries],dtype=torch.float64).reshape(32,8)
 components=torch.tensor([s['components']for s in summaries],dtype=torch.float64).reshape(32,8,4)
 material=torch.cat((rewards.unsqueeze(-1),components),-1)
 expected=GroupCredit(8).advantages(rewards,components,all_material=material)
 for rank,s in enumerate(summaries):
  adv=torch.tensor(s['advantages'],dtype=torch.float32).reshape(-1);mask=torch.tensor(s['mask'],dtype=torch.float32).reshape(-1)
  sl=slice(rank*32,(rank+1)*32)
  # Exact discrete mask comes from the actual saved FP64 raw A (GPU reductions
  # can produce tiny residuals); its math was checked within1e-14 at boundary.
  raw=torch.tensor(s['original_raw_A'],dtype=torch.float64);saved_mask=raw.ne(0).any(1,keepdim=True).expand_as(raw).reshape(-1).float()
  local=(raw/(s['normalization_std']+1e-8)).reshape(-1).float()
  if (abs(s['normalization_std']-float(expected['normalization_std']))>1e-12
      or abs(s['global_std']-float(expected['global_std']))>1e-12
      or not torch.equal(mask,saved_mask)or not torch.allclose(adv,local,atol=1e-7,rtol=1e-6)
      or not torch.allclose(adv,expected['advantages'][sl],atol=1e-7,rtol=1e-6)):
   raise ValueError('Actual 32-scene/256-candidate global credit changed')
 return summaries
