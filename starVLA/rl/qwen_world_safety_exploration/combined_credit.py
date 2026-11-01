"""Disjoint DenseDAC / successful saturation / inversion credit composition.

All decisions use ORIGINAL Safety credit. No sequential trigger amplification;
original nonzero world64 scale is retained. Not an endpoint-policy loss.
"""
import copy
import torch
import torch.distributed as dist

from .credit import GroupCredit,canonical_components,raw_advantage_parts,protocol as original_protocol
from . import dense_credit as D,saturated_credit as S,inversion_credit as I,protected_credit as P
from .recovery import validate_structure,tensor_digest
from starVLA.rl.qwen_world_research_algorithms.policy import ResearchLoss

SCHEMA='world_SafetyEP_disjoint_DenseDAC_Saturated_Inversion_fixed_base_std_v1'

def protocol(g):
 p=original_protocol(g)
 p.update(schema=SCHEMA,branches=dict(dense=D.protocol(g),saturated=S.protocol(g),inversion=I.protocol(g)),
  ordering='Evaluate every eligibility against ORIGINAL Safety rawA and canonical feedback; assert <=1 activated branch per scene',
  disjointness='Dense allfailedDAConly originalA0; Sat allfeasibleEP1 originalA0; inversion originalA nonzero with qualifying opposite signs',
  normalization='Original world*G Safety rawA std retained when >0; exactzero uses one updated world*G std collective on ALL ranks',
  group_sum='Each transform conserves original group sum in real arithmetic; FP64 roundoff audited; no recenter',
  mask='Dense/Sat activated group gains original zero PG direction; inversion retains original mask even if newraw all0; unrelated masks bitexact',
  exclusions='No endpoint joint loss, reward transform, resampling, teacher, extra actor observation or independent protected gate')
 return p


def validate_combined_recipe(cfg):
 from .standardized_recipe import validate_standardized_recipe
 validate_standardized_recipe(cfg)
 if cfg['research']['structured_advantage']!=SCHEMA:raise ValueError('Independent combined credit schema required')
 return cfg


@torch.no_grad()
def decision_from_material(rewards,components,quality,original_raw,dense_feedback=None):
 g=components.shape[1]
 fake=[{} for _ in range(g)] if dense_feedback is None else [{D.FIELD:f} for f in dense_feedback]
 dense=D.decision(fake,components)
 saturated=S.decision(rewards,components)
 inversion=I.decision_from_material(components,quality,original_raw)
 active=[dense['eligible'],saturated['eligible'],inversion['activated']]
 if sum(active)>1:raise ValueError('Combined branches must be mutually exclusive on ORIGINAL Safety credit')
 branch=('dense','saturated','inversion')[active.index(True)] if any(active) else 'none'
 raw=original_raw.detach().cpu().clone();ds=torch.zeros_like(raw);de=torch.zeros_like(raw)
 if active[0]:ds=dense['delta'];updated=raw+ds
 elif active[1]:de=saturated['delta'];updated=raw+de
 elif active[2]:de=inversion['delta'];updated=inversion['updated_raw_A']
 else:updated=raw.clone()
 roundoff=float((updated.sum(1)-raw.sum(1)).abs().max())
 if roundoff>1e-12:raise ValueError('Combined group-sum conservation failed')
 return dict(schema=SCHEMA,branch=branch,activated=any(active),original_raw_A=raw,quality_material=quality.detach().cpu(),
  dense_feedback=None if dense_feedback is None else dense_feedback.detach().cpu(),
  dense=dense,saturated=saturated,inversion=inversion,delta_safe=ds,delta_EP=de,updated_raw_A=updated,group_sum_roundoff=roundoff)


@torch.no_grad()
def decision(records,rewards,components,original_raw):
 dense=D.decision(records,components);flags=None
 if dense['subtype'] and all(r.get(D.FIELD) is not None for r in records):
  flags=torch.stack([torch.as_tensor(r[D.FIELD]).detach().cpu().bool() for r in records])
 return decision_from_material(rewards,components,P.quality_material(records,components),original_raw,flags)


def updated_parts(components,info):
 old=raw_advantage_parts(components)
 return dict(old,A_safe=old['A_safe']+info['delta_safe'].to(components.device),
  A_EP=old['A_EP']+info['delta_EP'].to(components.device),A_raw=info['updated_raw_A'].to(components.device))


class CombinedCredit(GroupCredit):
 validate_recipe=staticmethod(validate_combined_recipe)
 def __init__(self,g):super().__init__(g);self.protocol=protocol(g)
 @torch.no_grad()
 def advantages(self,rewards,components,*,records=None,info=None,all_material=None,all_new_raw=None):
  base=GroupCredit(self.group_size).advantages(rewards,components,all_material=all_material)
  if info is None:
   if records is None:raise ValueError('Actual canonical records required')
   info=decision(records,rewards,components,base['A_raw'])
  parts=updated_parts(components,info);original=base['normalization_std'];scale=original;fallback=bool(original==0)
  if fallback:
   if all_new_raw is None:
    local=parts['A_raw'].contiguous()
    if dist.is_available() and dist.is_initialized():
     if dist.get_world_size()!=8:raise ValueError('Combined fallback requires all8 ranks')
     gathered=[torch.empty_like(local) for _ in range(8)];dist.all_gather(gathered,local);all_new_raw=torch.cat(gathered)
    elif all_material is None or len(all_material)==1:all_new_raw=local
    else:raise ValueError('Complete updated raw credit including actual Dense flags required for fallback')
   if all_new_raw.shape[-1]!=self.group_size or not torch.isfinite(all_new_raw).all():raise ValueError('Invalid combined fallback material')
   scale=all_new_raw.detach().to(rewards.device,torch.float64).std(unbiased=False)
  out=dict(base,**parts,original_normalization_std=original,normalization_std=scale)
  if info['activated'] or fallback:
   replacement=(parts['A_raw']/(scale+1e-8)).reshape(-1).float()
   if info['branch']=='inversion' and not fallback:
    out['advantages']=base['advantages'].clone();changed=info['inversion']['eligible'].reshape(-1).to(rewards.device)
    out['advantages'][changed]=replacement[changed]
   else:
    out['advantages']=replacement
    out['adv_mask']=parts['A_raw'].ne(0).any(1,keepdim=True).expand_as(parts['A_raw']).reshape(-1).float()
  postzero=not bool(parts['A_raw'].ne(0).any());meta=dict(base['statistics'])
  meta.update(structured_advantage_schema=SCHEMA,normalization=self.protocol['normalization'],structured_zero_groups=int(postzero),
   effective_PG_candidate_fraction=float(out['adv_mask'].mean()),combined_activated_groups=int(info['activated']),
   combined_dense_groups=int(info['branch']=='dense'),combined_saturated_groups=int(info['branch']=='saturated'),combined_inversion_groups=int(info['branch']=='inversion'),
   combined_post_raw_PG0_groups=int(postzero),combined_normalization_fallback=int(fallback),
   combined_original_normalization_std=float(original),combined_effective_normalization_std=float(scale),
   combined_changed_candidates=int(parts['A_raw'].cpu().ne(info['original_raw_A']).sum()),
   combined_group_sum_roundoff=info['group_sum_roundoff'],combined_delta_safe_abs_mean=float(info['delta_safe'].abs().mean()),
   combined_delta_EP_abs_mean=float(info['delta_EP'].abs().mean()),combined_inversion_transfers=len(info['inversion']['pairs']),
   combined_true_nonzero_adv_fraction=float(out['advantages'].ne(0).float().mean()),combined_adv_cap5_candidates=int(out['advantages'].abs().gt(5).sum()))
  out.update(statistics=meta,combined_info=info)
  return out
 @torch.no_grad()
 def finalize(self,rollout):
  if any(k in rollout for k in ('advantages','structured_advantage','canonical_rewards','reward_transform','combined_credit','protected_credit','saturated_credit','dense_dac','inversion_credit')):
   raise ValueError('Fresh combined credit finalized once; no transformation mixing')
  m=canonical_components(rollout['records'],self.group_size,device=rollout['rewards'].device)
  if not torch.equal(m['canonical_rewards'],rollout['rewards']):raise ValueError('Canonical rewards changed')
  out=self.advantages(rollout['rewards'],m['components'],records=rollout['records'])
  rollout['canonical_rewards']=rollout['rewards'].detach().clone()
  for k in ('advantages','adv_mask','global_std','raw_group_std','normalization_std','original_normalization_std'):rollout[k]=out[k]
  rollout['combined_credit']=out['combined_info'];rollout['advantage_statistics']=out['statistics']
  rollout['structured_advantage']=dict(protocol=copy.deepcopy(self.protocol),components=m['components'],
   **{k:out[k] for k in ('feasible','cost','A_safe','A_EP','A_raw')})
  return rollout


def plain(v):
 if torch.is_tensor(v):return dict(__tensor__=str(v.dtype),shape=list(v.shape),values=v.cpu().tolist())
 if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
 if isinstance(v,list):return [plain(x) for x in v]
 return v


def restore(v):
 if isinstance(v,dict):
  if '__tensor__' in v:
   if set(v)!={'__tensor__','shape','values'} or v['__tensor__'] not in ('torch.float64','torch.bool'):raise ValueError('Invalid combined typed material')
   t=torch.tensor(v['values'],dtype={'torch.float64':torch.float64,'torch.bool':torch.bool}[v['__tensor__']])
   if list(t.shape)!=v['shape']:raise ValueError('Combined typed shape changed')
   return t
  return {k:restore(x) for k,x in v.items()}
 if isinstance(v,list):return [restore(x) for x in v]
 return v


def no_grad_material(v):
 if torch.is_tensor(v) and (v.requires_grad or not torch.isfinite(v).all()):raise ValueError('Detached finite combined feedback required')
 if isinstance(v,dict):
  for x in v.values():no_grad_material(x)
 if isinstance(v,list):
  for x in v:no_grad_material(x)


def validate_combined_boundary(state,rollout,identity,optimizer=None,scheduler=None,*,credit,legacy=False):
 credit.validate_recipe(identity['recipe']);g=credit.group_size
 if credit.protocol!=protocol(g):raise ValueError('Combined credit protocol changed')
 s=validate_structure(state,rollout,identity,optimizer,scheduler,legacy=legacy)
 s.update(group_size=g,structured_world=8,structured_advantage_schema=SCHEMA,structured_protocol=credit.protocol)
 if not s['pending']:return s
 if any(k in rollout for k in ('reward_transform','protected_credit','saturated_credit','dense_dac','inversion_credit')):raise ValueError('Cannot mix combined payload with old transforms')
 m=canonical_components(rollout['records'],g)
 for k in ('rewards','canonical_rewards'):S._equal(rollout[k],m['canonical_rewards'],k)
 info=rollout['combined_credit'];no_grad_material(info)
 S._equal(info['original_raw_A'],raw_advantage_parts(m['components'])['A_raw'],'original Safety credit',1e-14)
 expected=decision(rollout['records'],rollout['rewards'],m['components'],info['original_raw_A'])
 if tensor_digest(info)!=tensor_digest(expected):raise ValueError('Combined real feedback/gates/transform changed')
 saved=rollout['structured_advantage'];parts=updated_parts(m['components'],info)
 if set(saved)!={'protocol','components','feasible','cost','A_safe','A_EP','A_raw'} or saved['protocol']!=credit.protocol:raise ValueError('Combined structured protocol changed')
 S._equal(saved['components'],m['components'],'components')
 for k,x in parts.items():S._equal(saved[k],x,k,0 if x.dtype==torch.bool else 1e-14)
 old=S._scale(rollout['original_normalization_std'],'original std');scale=S._scale(rollout['normalization_std'],'new std')
 if old>0 and old!=scale:raise ValueError('Nonzero original scale must be retained')
 offsets=m['canonical_rewards'].double()-m['canonical_rewards'].double()[:,:1]
 S._equal(rollout['raw_group_std'],offsets.std(1,unbiased=False,keepdim=True),'raw rewardstd',1e-12)
 s.update(normalization_std=scale,original_normalization_std=old,canonical_rewards=m['canonical_rewards'].double().tolist(),
  structured_material={k:saved[k].cpu().tolist() for k in ('components','feasible','cost','A_safe','A_EP','A_raw')},
  combined_material=plain(info),combined_digest=tensor_digest(info),advantage_statistics=copy.deepcopy(rollout['advantage_statistics']))
 return s


def validate_combined_ranks(summaries,*,credit):
 common=('phase','updates','collections','scene_cursor','next_inner','run','pending','group_size','structured_world','structured_advantage_schema','structured_protocol')
 if len(summaries)!=8 or any({k:s[k] for k in common}!={k:summaries[0][k] for k in common} for s in summaries):raise ValueError('Eight matching combined ranks required')
 if summaries[0]['structured_protocol']!=credit.protocol:raise ValueError('Combined rank protocol changed')
 if not summaries[0]['pending']:return summaries
 if any(s['behavior']!=summaries[0]['behavior'] for s in summaries):raise ValueError('Combined behavior versions differ')
 rewards=torch.tensor([s['rewards'][0] for s in summaries],dtype=torch.float64)
 comp=torch.tensor([s['structured_material']['components'][0] for s in summaries],dtype=torch.float64)
 material=torch.cat((rewards.unsqueeze(-1),comp),-1);infos=[restore(s['combined_material']) for s in summaries]
 for j,(s,info) in enumerate(zip(summaries,infos)):
  no_grad_material(info)
  if tensor_digest(info)!=s['combined_digest']:raise ValueError('Combined rank digest changed')
  expected=decision_from_material(rewards[j:j+1],comp[j:j+1],info['quality_material'],info['original_raw_A'],info['dense_feedback'])
  if tensor_digest(info)!=tensor_digest(expected):raise ValueError('Combined rank transform changed')
 allnew=torch.cat([i['updated_raw_A'] for i in infos])
 for j,s in enumerate(summaries):
  e=credit.advantages(rewards[j:j+1],comp[j:j+1],info=infos[j],all_material=material,all_new_raw=allnew)
  if (s['canonical_rewards']!=s['rewards'] or abs(s['global_std']-float(e['global_std']))>1e-12
      or abs(s['normalization_std']-float(e['normalization_std']))>1e-12 or abs(s['original_normalization_std']-float(e['original_normalization_std']))>1e-12
      or not torch.allclose(torch.tensor(s['advantages']),e['advantages'],atol=1e-7,rtol=1e-6) or s['mask']!=e['adv_mask'].tolist()):raise ValueError('Combined world normalization/A/mask changed')
  # Exact-zero and cap comparisons are discontinuous. CPU/GPU FP64 reduction
  # trees can differ by1e-16 while their actual FP32 advantages remain within
  # the validated numerical contract above. These are statistics OF THE SAVED
  # ACTUAL ADVANTAGES, not statistics of a substitute CPU formula. Keep the
  # original PG tensors and compare their discrete statistics exactly.
  actual_adv=torch.tensor(s['advantages'],dtype=torch.float32)
  actual_discrete=dict(combined_true_nonzero_adv_fraction=float(actual_adv.ne(0).float().mean()),
      combined_adv_cap5_candidates=int(actual_adv.abs().gt(5).sum()))
  for k in ('feasible','cost','A_safe','A_EP','A_raw'):S._equal(torch.tensor(s['structured_material'][k],dtype=e[k].dtype),e[k],k,0 if e[k].dtype==torch.bool else 1e-14)
  if set(s['advantage_statistics'])!=set(e['statistics']):raise ValueError('Combined statistics keys changed')
  for k,w in e['statistics'].items():
   if k in actual_discrete:w=actual_discrete[k]
   got=s['advantage_statistics'][k]
   if (abs(got-w)>1e-12 if isinstance(w,float) else got!=w):raise ValueError('Combined activation/statistics changed: '+k)
 return summaries


class CombinedLoss(ResearchLoss):
 def _policy_loss(self,model,rollout):
  total,metrics=super()._policy_loss(model,rollout)
  for k,v in rollout['advantage_statistics'].items():
   if k.startswith('combined_') and isinstance(v,(int,float)):metrics[k]=torch.tensor(v,device=total.device,dtype=torch.float64)
  return total,metrics


class CombinedRuntimeMixin:
 def make_credit(self,cfg):validate_combined_recipe(cfg);return CombinedCredit(cfg['grpo']['sample']['group_size'])
 def make_actor(self,cfg,args,device):
  actor,a,b=super().make_actor(cfg,args,device);actor._loss=CombinedLoss(cfg);return actor,a,b
 def finalize_advantages(self,rollout):return self.credit.finalize(rollout)
 def validate_boundary(self,*args,**kwargs):return validate_combined_boundary(*args,credit=self.credit,**kwargs)
 def validate_ranks(self,summaries):return validate_combined_ranks(summaries,credit=self.credit)
 def recipe_identity(self,cfg,args,condition_sha):
  v=super().recipe_identity(cfg,args,condition_sha);v['combined_credit_protocol']=copy.deepcopy(self.credit.protocol);return v
