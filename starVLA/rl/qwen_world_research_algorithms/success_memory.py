"""Online TRAIN verified-success memory, independently sampled historical CFM.

Empty at fresh IL70 initialization. No historical Safety teacher, no PG replay,
no main-stream RNG changes. All persistent bank/RNG/counters live in rank state;
the selected teacher/noise/descriptor live in the pending rollout for four inners.
"""
from __future__ import annotations
import copy
import hashlib
import json
from collections import OrderedDict

import torch
import torch.distributed as dist

from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
from starVLA.rl.simwam_recipe.recovery_state import tensor_digest
from .policy import ResearchLoss
from .safety_efficiency import validate_safety_efficiency_recipe,canonical_components

ARM='PPO_SAFETY_SUCCESS_MEMORY'
SCHEMA='world_online_verified_success_memory_v1'
AUX_SCHEMA='world_online_historical_success_CFM_material_v1'
CONFIG=dict(schema=SCHEMA,capacity=1024,period_collections=4,aux_scenes_per_rank=1,mc_samples=1,
            coefficient=.1,sigma_min=.2,sigma_max=.8,min_EP=.5,seed=42)
METRICS=('score','no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','ego_progress','comfort','driving_direction_compliance')


def fp(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def rank_id():return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
def private_seed(rank):return int.from_bytes(hashlib.sha256(f'{SCHEMA}:42:{rank}'.encode()).digest()[:8],'big')&((1<<63)-1)
def ep_bin(ep):return 'ge1' if ep>=1 else 'ge09_lt1' if ep>=.9 else 'ge075_lt09' if ep>=.75 else 'ge05_lt075' if ep>=.5 else 'below05'


def validate_recipe(cfg):
    validate_safety_efficiency_recipe(cfg);r=cfg.get('research',{});s=cfg['grpo']['sample']
    if (r.get('arm')!=ARM or r.get('success_memory')!=CONFIG or r.get('data_policy')!='uniform_full'
            or cfg['lr_scheduler_type']!='constant' or cfg['lr_warmup_steps']!=100 or cfg['max_steps']!=50000
            or s['noise_level']!=.1 or s['train_sde_steps']!=3 or s['sde_exclude_last_step'] is not False or s.get('sde_step_indices') is not None
            or any(r.get(k) is not None for k in ('warmstart','native_fork','rescue_manifest','rescue_auxiliary','preference_auxiliary','preference_manifest','sampler_ablation','noise_level_ablation','reward_transform','advantage_normalization','anchor_ablation','memory_lr_ablation'))):
        raise ValueError('Success memory is one freshIL70 full/W3/constant/SafetyEP auxiliary change')
    return cfg


def new_memory(run,rank):
    g=torch.Generator(device='cpu').manual_seed(private_seed(rank))
    return dict(schema=SCHEMA,run=run,rank=rank,collections=0,entries=[],rng_state=g.get_state(),
        verified_winners_seen=0,below_EP_threshold_groups=0,no_strict_success_groups=0,evictions=0,
        aux_selected_collections=0,aux_selection_counts={},winner_EP_bins={},skip_counts={})


def entry_digest(e):return tensor_digest({k:v for k,v in e.items() if k!='teacher_id'})
def payload_digest(p):return tensor_digest({k:v for k,v in p.items() if k!='material_sha256'})


@torch.no_grad()
def verified_winner(rollout,run,allowed):
    canonical_components(rollout['records'])
    token,log=rollout['token'],rollout['log'];version=rollout['behavior_version']
    if allowed.get(token)!=log or version['run']!=run:raise ValueError('Success labels must come from this run and legal TRAIN observations')
    strict=[i for i,r in enumerate(rollout['records']) if all(r['metrics'][k]==1 for k in ('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','comfort'))]
    valid=[i for i in strict if rollout['records'][i]['metrics']['ego_progress']>=CONFIG['min_EP']]
    if not valid:return None,'below_EP_threshold' if strict else 'no_strict_success'
    i=max(valid,key=lambda j:(rollout['records'][j]['metrics']['ego_progress'],rollout['records'][j]['raw_progress_meters'],-j))
    r=rollout['records'][i]
    if r['token']!=token or r['log']!=log or r['protocol_sha256']!=rollout['protocol_sha256']:raise ValueError('Verified endpoint token/log/protocol mismatch')
    clean=rollout['chain'][i,-1].detach().float().cpu().contiguous().clone()
    requested=torch.as_tensor(r['requested_trajectory'],dtype=torch.float64).cpu().contiguous()
    decoded=ActionCodec.denorm_odo(clean).double();bound=8*torch.finfo(torch.float32).eps*(decoded.abs()+torch.tensor([66.74,42.,3.53]))
    if clean.shape!=(8,3) or requested.shape!=(8,3) or not torch.isfinite(clean).all() or not torch.isfinite(requested).all() or ((decoded-requested).abs()>bound).any():
        raise ValueError('Stored teacher must be the actually scored normalized endpoint')
    e=dict(token=token,log=log,clean=clean,requested=requested,metrics=copy.deepcopy(r['metrics']),raw_progress_meters=float(r['raw_progress_meters']),
        source_run=run,source_update=int(version['update']),source_version=copy.deepcopy(version),candidate_index=i,
        protocol_sha256=r['protocol_sha256'],cache=copy.deepcopy(r['cache']),codec_max_abs_by_dim=(decoded-requested).abs().amax(0).tolist(),codec_bound_by_dim=bound.amax(0).tolist(),
        metric_verified=True,human_optimal=False,PG_advantage_or_logprob_replayed=False)
    e['teacher_id']=entry_digest(e);return e,None


def validate_entry(e,run,allowed):
    if (e.get('teacher_id')!=entry_digest(e) or e.get('source_run')!=run or allowed.get(e.get('token'))!=e.get('log')
            or e.get('metric_verified') is not True or e.get('PG_advantage_or_logprob_replayed') is not False
            or any(e['metrics'][k]!=1 for k in ('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','comfort'))
            or e['metrics']['ego_progress']<CONFIG['min_EP'] or e['clean'].shape!=(8,3) or e['requested'].shape!=(8,3)):
        raise ValueError('Success-memory teacher identity/gates/source changed')


def private_material(state,n):
    g=torch.Generator(device='cpu');g.set_state(state.cpu())
    index=int(torch.randint(n,(1,),generator=g));sigma=.2+.6*torch.rand(1,generator=g);noise=torch.randn(1,8,3,generator=g)
    return index,sigma,noise,g.get_state()


@torch.no_grad()
def attach_history(memory,rollout,conditions,device,allowed):
    if 'success_memory_aux' in rollout:raise ValueError('History material is finalized once per fresh collection')
    c=rollout['behavior_version']['update']//4+1
    if c!=memory['collections']+1:raise ValueError('Success bank and fresh collection cursor differ')
    historical=[e for e in memory['entries'] if e['token']!=rollout['token']]
    reason='period_not_due' if c%CONFIG['period_collections'] else 'empty_history' if not historical else None
    p=dict(schema=AUX_SCHEMA,collection=c,behavior_update=rollout['behavior_version']['update'],fresh_token=rollout['token'],rank=memory['rank'],
        enabled=reason is None,skip_reason=reason,coefficient=CONFIG['coefficient'],mc_samples=1)
    if reason is None:
        before=memory['rng_state'].clone();i,sigma,noise,after=private_material(before,len(historical));teacher=copy.deepcopy(historical[i])
        if teacher['source_update']>=rollout['behavior_version']['update']:raise ValueError('Auxiliary must select a genuinely historical policy endpoint')
        p.update(teacher=teacher,selection_count=len(historical),selection_index=i,private_rng_before=before,
            sigma=sigma.to(device),noise=noise.to(device),condition=conditions.get(teacher['token'],device))
        memory['rng_state']=after;memory['aux_selected_collections']+=1
        counts=memory['aux_selection_counts'];counts[teacher['token']]=counts.get(teacher['token'],0)+1
    else:memory['skip_counts'][reason]=memory['skip_counts'].get(reason,0)+1
    # Select from the OLD bank first. Fresh success is never its own aux target.
    winner,skip=verified_winner(rollout,memory['run'],allowed)
    if winner is None:
        key='below_EP_threshold_groups' if skip=='below_EP_threshold' else 'no_strict_success_groups';memory[key]+=1
    else:
        memory['verified_winners_seen']+=1;bucket=ep_bin(winner['metrics']['ego_progress']);memory['winner_EP_bins'][bucket]=memory['winner_EP_bins'].get(bucket,0)+1
        entries=memory['entries'];old=next((j for j,e in enumerate(entries) if e['token']==winner['token']),None)
        if old is None:
            if len(entries)==CONFIG['capacity']:entries.pop(0);memory['evictions']+=1
            entries.append(winner)
        elif (winner['metrics']['ego_progress'],winner['raw_progress_meters'])>(entries[old]['metrics']['ego_progress'],entries[old]['raw_progress_meters']):entries[old]=winner
    memory['collections']=c;p['material_sha256']=payload_digest(p);rollout['success_memory_aux']=p;return rollout


def validate_payload(p,memory,allowed):
    if (p.get('schema')!=AUX_SCHEMA or p.get('material_sha256')!=payload_digest(p) or p.get('rank')!=memory['rank']
            or p.get('collection')!=memory['collections'] or p.get('coefficient')!=.1 or p.get('mc_samples')!=1):raise ValueError('Pending success CFM payload changed')
    if p['enabled']:
        t=p['teacher'];validate_entry(t,memory['run'],allowed)
        i,s,n,_=private_material(p['private_rng_before'],p['selection_count'])
        if (t['token']==p['fresh_token'] or t['source_update']>=p['behavior_update'] or i!=p['selection_index']
                or not torch.equal(s,p['sigma'].cpu()) or not torch.equal(n,p['noise'].cpu())
                or p['condition']['token']!=t['token']):raise ValueError('Historical scene/RNG/CFM material changed')


def memory_summary(state,rollout,run,rank,allowed,*,full=False):
    m=state.get('success_memory',{})
    if (m.get('schema')!=SCHEMA or m.get('run')!=run or m.get('rank')!=rank or m.get('collections')!=state['collections']
            or len(m['entries'])>1024 or m['rng_state'].dtype!=torch.uint8 or len({e['token'] for e in m['entries']})!=len(m['entries'])):
        raise ValueError('Missing/corrupt native success bank/RNG/cursor')
    if full:
        for e in m['entries']:validate_entry(e,run,allowed)
        g=torch.Generator(device='cpu');g.set_state(m['rng_state'].cpu())
    p=None if rollout is None else rollout.get('success_memory_aux')
    if rollout is not None:
        if p is None:raise ValueError('Pending fresh rollout lost historical CFM materials')
        validate_payload(p,m,allowed)
        if p['enabled'] and p['condition']!=dict(rollout['cond_bg'],token=p['teacher']['token']):raise ValueError('Historical aux observation must use the same legal current-only condition schema')
    completed={t:4*n for t,n in m['aux_selection_counts'].items()}
    if p is not None and p['enabled']:
        token=p['teacher']['token'];completed[token]-=4-state['next_inner']
    if sum(m['aux_selection_counts'].values())!=m['aux_selected_collections'] or any(n<0 for n in completed.values()):raise ValueError('Auxiliary exposure counters inconsistent')
    bins={}
    for e in m['entries']:
        b=ep_bin(e['metrics']['ego_progress']);bins[b]=bins.get(b,0)+1
    return dict(schema=SCHEMA,rank=rank,collections=m['collections'],bank_size=len(m['entries']),bank_distinct_tokens=len(m['entries']),
        bank_EP_bins=bins,lifetime_winner_EP_bins=m['winner_EP_bins'],verified_winners_seen=m['verified_winners_seen'],
        below_EP_threshold_groups=m['below_EP_threshold_groups'],no_strict_success_groups=m['no_strict_success_groups'],evictions=m['evictions'],skip_counts=m['skip_counts'],selected_aux_collections=m['aux_selected_collections'],
        selected_distinct_aux_scenes=len(m['aux_selection_counts']),completed_aux_optimizer_updates=sum(completed.values()),completed_distinct_aux_scenes=sum(n>0 for n in completed.values()),
        RNG_sha256=tensor_digest(m['rng_state']),ordered_bank_ids_sha256=fp([e['teacher_id'] for e in m['entries']]),pending_aux_sha256=p['material_sha256'] if p else None,
        exposure_units='selection once percollection;actual completed CFM updates account for saved next_inner;lowEP skipped by threshold is not a driving failure')


class FrozenFeatureLRU:
    """Two frozen-Qwen/reference-only entries; never caches current Memory/KV."""
    def __init__(self,base):self.base=base;self.cache=OrderedDict()
    def __getattr__(self,k):return getattr(self.base,k)
    def materialize(self,descriptor,device):
        token=descriptor['token']
        if token in self.cache:self.base.token=token;self.base.cached=self.cache.pop(token)
        value=self.base.materialize(descriptor,device);self.cache[token]=self.base.cached
        while len(self.cache)>2:self.cache.popitem(last=False)
        return value


def cfm_material(clean,sigma,noise):
    target=clean.detach().to(noise).unsqueeze(0)
    if target.shape!=(1,8,3) or sigma.shape!=(1,) or noise.shape!=(1,8,3):raise ValueError('Success CFM is exactly1MC/native8x3 action')
    return (1-sigma[:,None,None])*target+sigma[:,None,None]*noise,noise-target


class SuccessMemoryLoss(ResearchLoss):
    def __init__(self,cfg):validate_recipe(cfg);super().__init__(cfg);self.cfg=cfg;self._live_components=None
    def record_policy_components(self,pg,anchor):self._live_components=dict(pg=pg,anchor=anchor)
    def _policy_loss(self,model,rollout):
        self._live_components=None
        try:
            total,metrics=super()._policy_loss(model,rollout);p=rollout.get('success_memory_aux',{})
            if p.get('schema')!=AUX_SCHEMA or p.get('material_sha256')!=payload_digest(p):raise ValueError('Success CFM loss requires saved,once-finalized history material')
            raw=total*0;weighted=raw
            if p['enabled']:
                state,target=cfm_material(p['teacher']['clean'],p['sigma'],p['noise'])
                with model._context_call(p['condition']) as c:
                    # Descriptor uses native G8 carrier;1MC query needs1 carrier.
                    c=dict(c,context=c['context'][:1],valid=c['valid'][:1],ego=c['ego'][:1])
                    velocity=model.action_velocity(state,p['sigma']*1000,c)
                raw=(velocity.float()-target.detach().float()).square().mean();weighted=.1*raw
                if not bool(torch.isfinite(raw)):raise FloatingPointError('Nonfinite success CFM auxiliary')
            metrics.update(success_aux_enabled=total.detach()*0+float(p['enabled']),success_aux_loss=raw.detach(),success_aux_weighted_loss=weighted.detach(),
                success_aux_period=total.detach()*0+4,success_aux_mc=total.detach()*0+float(p['enabled']),success_aux_teacher_EP=total.detach()*0+(p['teacher']['metrics']['ego_progress'] if p['enabled'] else 0))
            if self.cfg['research'].get('gradient_probe'):
                if self._live_components is None:raise RuntimeError('gradient probe requires optional base record_policy_components callback')
                from .gradient_probe import maybe_probe_components
                metrics.update(maybe_probe_components(model,rollout,self.cfg,dict(self._live_components,aux=weighted)))
            return total+weighted,metrics
        finally:self._live_components=None
