"""Candidate-metric Safety credit with canonical feasible quality in [0,1].

Only q changes: EP -> (5*EP+2*Comfort)/7 inside the existing feasible set.
This preserves the strict feasible-vs-unsafe credit ordering. Canonical rewards,
all-fail deficit fallback, collection, PG/physical anchor and observations stay.
"""
from __future__ import annotations
import copy
import torch
import torch.distributed as dist

from .safety_efficiency import canonical_components, safety_efficiency_parts
from starVLA.rl.simwam_recipe.recovery_state import (
    tensor_digest, validate_boundary as source_boundary,
)

ARM='PPO_SAFETY_FEASIBLE_SCORE'
SCHEMA='candidate_metric_safety_feasible_canonical_quality_global_std_v1'
COMPONENTS=('no_at_fault_collisions','drivable_area_compliance',
            'time_to_collision_within_bound','ego_progress','comfort')
PARTS=('feasible','cost','A_safe','A_quality','A_raw')
PROTOCOL=dict(schema=SCHEMA,feasible='f=1 iff NC=DAC=TTC=1; Comfort is quality, not feasibility',
    cost='c=mean(1-NC,1-DAC,1-TTC); indicator deficit, not accident severity',
    safety='if any f: A_safe=f-mean(f); allfail: A_safe=mean(c)-c',
    quality='q=(5*EP+2*Comfort)/7 in[0,1]; A_quality=f*(q-mean_feasible(q)); singleton/empty gives0',
    raw='A_raw=A_safe+A_quality; group sum0',
    ordering='before Acap5: mixed group with m feasible has every A_raw_feasible-A_raw_unsafe>=1/m>0; unsafe A_raw=-m/8; monotone clamp may tie but cannot invert',
    coefficients=dict(EP=5/7,Comfort=2/7),
    normalization='one population_std of all64 A_raw+1e-8; zero groups included; fixed4inners',
    canonical='canonical rewards/records/PDMS/scorer unchanged; never label A_raw as PDMS',
    mask='all-zero A_raw group has PGmask0; allfail costtie remains zero',
    unchanged='freshIL70/seed42/full103288/W3eta.1/G8/world8/4inner/PPOclip.005/Acap5/beta50 IL70 meananchor/LR',
    limitation='metric-conditioned quality credit; not scene-risk inference or safety guarantee')


def validate_recipe(cfg):
    r=cfg.get('research',{});s=cfg['grpo']['sample'];t=cfg['grpo']['train']
    if (r.get('arm')!=ARM or r.get('structured_advantage')!=SCHEMA
        or r.get('policy_objective')!='ppo' or r.get('data_policy')!='uniform_full'
        or cfg.get('seed')!=42 or cfg.get('world_size')!=8 or cfg.get('action_shape')!=[8,3]
        or cfg.get('variant')!='R_CANONICAL' or cfg.get('learning_rate')!=5e-5
        or cfg.get('memory_learning_rate')!=5e-6 or cfg.get('lr_warmup_steps')!=100
        or cfg.get('lr_scheduler_type')!='constant' or cfg.get('max_steps')!=50000
        or s.get('train_sde_steps')!=3 or s.get('sde_exclude_last_step') is not False
        or s.get('sde_step_indices') is not None or s.get('noise_level')!=.1
        or t.get('kl_beta')!=50 or t.get('adv_clip_max')!=5 or t.get('ppo_clip_range')!=.005
        or any(r.get(k) is not None for k in ('fpo','success_memory','warmstart','native_fork',
            'lr_schedule','sampler_ablation','noise_level_ablation','reward_transform',
            'advantage_normalization','anchor_ablation','memory_lr_ablation',
            'rescue_auxiliary','preference_auxiliary'))):
        raise ValueError('Register one freshIL70 canonical-feasible-quality PPO change only')
    return cfg


@torch.no_grad()
def components_from_records(records,*,device='cpu'):
    # Reuse the original canonical equation/protocol/readout validity checks.
    original=canonical_components(records,device=device)
    comfort=torch.tensor([[r['metrics']['comfort'] for r in records]],device=device,dtype=torch.float64)
    return dict(canonical_rewards=original['canonical_rewards'],
                components=torch.cat((original['components'],comfort.unsqueeze(-1)),-1))


@torch.no_grad()
def quality_parts(components):
    if (components.ndim!=3 or tuple(components.shape[1:])!=(8,5)
        or not bool(torch.isfinite(components).all())
        or not bool(((components>=0)&(components<=1)).all())):
        raise ValueError('Finite [group,8,NC/DAC/TTC/EP/Comfort] material required')
    x=components.detach().double()
    base=safety_efficiency_parts(x[:,:,:4])  # Identical feasibility/cost/A_safe.
    f=base['feasible'].double();count=f.sum(1,keepdim=True)
    q=(5*x[:,:,3]+2*x[:,:,4])/7
    first=base['feasible'].to(torch.int64).argmax(1,keepdim=True)
    offsets=(q-q.gather(1,first))*f
    aq=(offsets-offsets.sum(1,keepdim=True)/count.clamp_min(1))*f
    aq=torch.where(count>=2,aq,torch.zeros_like(aq))
    return dict(feasible=base['feasible'],cost=base['cost'],A_safe=base['A_safe'],
                A_quality=aq.detach(),A_raw=(base['A_safe']+aq).detach())


@torch.no_grad()
def advantages(rewards,components,*,all_material=None):
    if (rewards.ndim!=2 or rewards.shape[1]!=8 or tuple(components.shape)!=(*rewards.shape,5)
        or not bool(torch.isfinite(rewards).all()) or not bool(((rewards>=0)&(rewards<=1)).all())):
        raise ValueError('Canonical rewards and matching quality components required')
    local=quality_parts(components)
    material=torch.cat((rewards.detach().double().unsqueeze(-1),components.detach().to(rewards.device).double()),-1)
    if all_material is None:
        if dist.is_available() and dist.is_initialized():
            gathered=[torch.empty_like(material) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered,material);all_material=torch.cat(gathered,0)
        else:all_material=material
    all_material=all_material.detach().to(rewards.device).double()
    if (all_material.ndim!=3 or tuple(all_material.shape[1:])!=(8,6)
        or not bool(torch.isfinite(all_material).all()) or not bool(((all_material>=0)&(all_material<=1)).all())):
        raise ValueError('Finite global canonical/quality material required')
    scale=quality_parts(all_material[:,:,1:])['A_raw'].std(unbiased=False)
    active=local['A_raw'].ne(0).any(1,keepdim=True)
    raw=rewards.detach().double();offsets=raw-raw[:,:1]
    return dict(**local,advantages=(local['A_raw']/(scale+1e-8)).reshape(-1).float().detach(),
        adv_mask=active.expand_as(raw).reshape(-1).float().detach(),normalization_std=scale.detach(),
        global_std=all_material[:,:,0].std(unbiased=False).detach(),
        raw_group_std=offsets.std(1,unbiased=False,keepdim=True).detach(),
        statistics=dict(structured_advantage_schema=SCHEMA,normalization_population_candidates=int(all_material.shape[0]*8),
            normalization_population_groups=int(all_material.shape[0]),normalization=PROTOCOL['normalization'],
            feasible_candidates=int(local['feasible'].sum()),structured_zero_groups=int((~active).sum()),
            allfail_groups=int((~local['feasible'].any(1)).sum()),
            all_full=int((raw==1).all(1).sum()),all_zero=int((raw==0).all(1).sum()),
            tied=int((offsets.std(1,unbiased=False)<=1e-8).sum()),
            reduction='source local effective mean, then equal mean across ranks',
            effective_PG_candidate_fraction=float(active.float().mean())))


@torch.no_grad()
def finalize_advantages(rollout):
    if 'advantages' in rollout or 'structured_advantage' in rollout:
        raise ValueError('Finalize canonical-quality credit once before four inner updates')
    material=components_from_records(rollout['records'],device=rollout['rewards'].device)
    if not torch.equal(rollout['rewards'],material['canonical_rewards']):
        raise ValueError('Fresh canonical rewards changed')
    stats=advantages(rollout['rewards'],material['components'])
    rollout['canonical_rewards']=material['canonical_rewards']
    for k in ('advantages','adv_mask','normalization_std','global_std','raw_group_std'):rollout[k]=stats[k]
    rollout['advantage_statistics']=stats['statistics']
    rollout['structured_advantage']=dict(protocol=copy.deepcopy(PROTOCOL),components=material['components'],
                                        **{k:stats[k] for k in PARTS})
    return rollout


def validate_boundary(state,rollout,identity,optimizer=None,scheduler=None,*,legacy=False):
    validate_recipe(identity['recipe'])
    if identity['world']!=8 or identity.get('structured_advantage_protocol')!=PROTOCOL:
        raise ValueError('Canonical-quality identity/world mismatch')
    summary=source_boundary(state,rollout,identity,optimizer,scheduler,legacy=legacy)
    summary.update(quality_schema=SCHEMA,quality_protocol=copy.deepcopy(PROTOCOL))
    if not summary['pending']:return summary
    if 'reward_transform' in rollout:raise ValueError('Canonical reward transform forbidden')
    expected=components_from_records(rollout['records'])
    for k in ('canonical_rewards','rewards'):
        v=rollout.get(k)
        if not torch.is_tensor(v) or v.requires_grad or v.dtype!=torch.float32 or not torch.equal(v.cpu(),expected['canonical_rewards']):
            raise ValueError('Canonical quality reward material changed: '+k)
    expected.update(quality_parts(expected['components']))
    saved=rollout.get('structured_advantage',{})
    if set(saved)!={'protocol','components',*PARTS} or saved.get('protocol')!=PROTOCOL:
        raise ValueError('Canonical quality saved protocol changed')
    for k in ('components',*PARTS):
        v,want=saved[k],expected[k]
        if (not torch.is_tensor(v) or v.requires_grad or v.shape!=want.shape or v.dtype!=want.dtype
            or not torch.allclose(v.cpu(),want,atol=1e-14,rtol=0)):
            raise ValueError('Canonical quality component changed: '+k)
    scale=rollout.get('normalization_std');meta=rollout['advantage_statistics']
    if (not torch.is_tensor(scale) or scale.ndim!=0 or scale.requires_grad or scale.dtype!=torch.float64
        or not bool(torch.isfinite(scale)) or float(scale)<0 or meta.get('structured_advantage_schema')!=SCHEMA
        or meta.get('normalization_population_groups')!=8 or meta.get('normalization_population_candidates')!=64):
        raise ValueError('Canonical quality global normalization changed')
    groupstd=rollout['raw_group_std'];raw=expected['canonical_rewards'].double();want=(raw-raw[:,:1]).std(1,unbiased=False,keepdim=True)
    if (not torch.is_tensor(groupstd) or groupstd.requires_grad or groupstd.dtype!=torch.float64 or groupstd.shape!=want.shape
        or not torch.allclose(groupstd.cpu(),want,atol=1e-12,rtol=0)):
        raise ValueError('Canonical raw group std changed')
    summary.update(normalization_std=float(scale),quality_material={k:saved[k].cpu().tolist() for k in ('components',*PARTS)},
        quality_digests={k:tensor_digest(saved[k]) for k in ('components',*PARTS)},
        canonical_rewards=rollout['canonical_rewards'].double().cpu().tolist())
    return summary


def validate_ranks(summaries):
    if len(summaries)!=8:raise ValueError('Eight canonical-quality ranks required')
    common=('phase','updates','collections','scene_cursor','next_inner','run','pending','quality_schema','quality_protocol')
    for s in summaries:
        if (s.get('quality_schema')!=SCHEMA or s.get('quality_protocol')!=PROTOCOL
            or {k:s[k] for k in common}!={k:summaries[0][k] for k in common}):
            raise ValueError('Canonical quality rank protocol/boundary mismatch')
    if not summaries[0]['pending']:return summaries
    if any(s['behavior']!=summaries[0]['behavior'] for s in summaries):
        raise ValueError('Canonical quality rank behavior mismatch')
    rewards=torch.tensor([s['rewards'][0] for s in summaries],dtype=torch.float64)
    components=torch.tensor([s['quality_material']['components'][0] for s in summaries],dtype=torch.float64)
    expected=advantages(rewards,components,all_material=torch.cat((rewards[:,:,None],components),-1))
    for rank,s in enumerate(summaries):
        if (s['canonical_rewards']!=s['rewards'] or abs(s['global_std']-float(expected['global_std']))>1e-12
            or abs(s['normalization_std']-float(expected['normalization_std']))>1e-12
            or not torch.allclose(torch.tensor(s['advantages']),expected['advantages'].reshape(8,8)[rank],atol=1e-7,rtol=1e-6)
            or s['mask']!=expected['adv_mask'].reshape(8,8)[rank].tolist()):
            raise ValueError('Canonical quality advantage was not globally finalized')
        for k in PARTS:
            want=expected[k][rank:rank+1];v=torch.tensor(s['quality_material'][k],dtype=want.dtype)
            if not torch.allclose(v,want,atol=1e-14,rtol=0):
                raise ValueError('Canonical quality rank component mismatch: '+k)
    return summaries
