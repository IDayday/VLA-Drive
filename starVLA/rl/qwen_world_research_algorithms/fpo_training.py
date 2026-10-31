"""Registered-shape endpoint-CFM per-MC ordinary-PPO replacement, W3 intact.

Ratio is an explicitly truncated-CFM surrogate, NOT endpoint likelihood/IS/KL.
The fixed pilot changes PG only. Canonical collection/Safety A, source physical
IL70 W3 mean anchor, observation/scope/data/LR and all native clocks remain.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import os
import torch

from .fpo_prototype import cfm_energy, per_mc_ppo_objective, _fresh_evidence, _state
from .policy import ResearchLoss
from .safety_efficiency import (
    STRUCTURED_SCHEMA, validate_safety_efficiency_recipe,
    validate_safety_efficiency_boundary, validate_safety_efficiency_ranks,
)
from starVLA.rl.simwam_recipe.recovery_state import tensor_digest

SCHEMA='world_endpoint_CFM_per_MC_PPO_v1'
ARM='PPO_SAFETY_FPO'
CONFIG=dict(schema=SCHEMA,mc_samples=2,sigma_min=.2,sigma_max=.8,
            clip_range=.01,stateless_seed=42,energy_reduction='mean_xyz_then_sum_8poses')
FULLTIME_ARM='PPO_FPO_FULLTIME'
FULLTIME_CONFIG=dict(CONFIG,sigma_max=1.0)
PROTOCOL=dict(**CONFIG,
    target='state=(1-sigma)*clean_endpoint+sigma*eps; target_velocity=eps-clean_endpoint',
    energy='ell[MC,G] = sum_horizon8 mean_xyz squared_velocity_error = 8*mean_all24',
    ratio='rho[MC,G] = exp(ell_behavior_old - ell_current), surrogate not endpoint likelihood/IS/KL',
    clipping='ordinary symmetric PPO independently perMC; same finalized SafetyA clipped5; noASPO',
    sigma='independent Uniform[.2,.8] perMC/candidate; truncatedCFM, not fullpaper U[0,1] ELBO equivalence',
    old='CURRENT collection-start behavior, no_grad cached once beforeinner0, not permanently frozen IL70',
    material='all MCnoise/sigma/endpoints/oldvelocities/oldenergies fixed all4inners, detached, native sealed',
    randomness='private stateless schema/token/behavior/rank/seed42; no sourceSDE/global RNG consumption',
    anchor='unchanged source W3[7,8,9] physical mean anchor beta50 to frozenIL70, same axisweights/reduction',
    data_and_credit='samefresh full103288 uniform W3rigorouseta.1 canonicalG8/SafetyEPglobal64std',
    fresh_ratio_tolerance=1e-4,
    stabilization='finite guards; noHuber, energyclamp, logratioSTEclamp, temperature or adaptiveLR',
    interpretation='pre-registered firstcandidate scale .01, not an optimum or copiedLLM tokenclip')


def settings_for_recipe(cfg=None):
    """Default retains the first pilot exactly; only the new upper-bound arm.

    Keep the private seed namespace/draw ordering common across both arms.
    Only the affine map from the same uniform draw to sigma changes.
    """
    value=CONFIG if cfg is None else cfg['research']['fpo']
    if value!=CONFIG and value!=FULLTIME_CONFIG:
        raise ValueError('Only original FPO or explicit sigma_max1.0 settings allowed')
    return copy.deepcopy(value)


def protocol_for_recipe(cfg=None):
    settings=settings_for_recipe(cfg)
    protocol=copy.deepcopy(PROTOCOL)
    protocol.update(settings)
    if settings==FULLTIME_CONFIG:
        protocol['sigma']='independent Uniform[.2,1.0) perMC/candidate; upper-range extension, still truncated below.2, not fullpaper U[0,1] ELBO equivalence'
    return protocol


def _settings_from_protocol(protocol):
    for settings in (CONFIG,FULLTIME_CONFIG):
        if protocol==protocol_for_recipe({'research':{'fpo':settings}}):
            return settings
    raise ValueError('Unsupported FPO material protocol')


def validate_recipe(cfg):
    validate_safety_efficiency_recipe(cfg)
    r=cfg['research'];s=cfg['grpo']['sample'];t=cfg['grpo']['train']
    expected={ARM:CONFIG,FULLTIME_ARM:FULLTIME_CONFIG}.get(r.get('arm'))
    if (expected is None or r.get('policy_objective')!='ppo'
        or r.get('data_policy')!='uniform_full' or r.get('structured_advantage')!=STRUCTURED_SCHEMA
        or r.get('fpo')!=expected or cfg.get('seed')!=42 or cfg.get('world_size')!=8
        or cfg.get('max_steps')!=50000 or cfg.get('lr_scheduler_type')!='constant'
        or cfg.get('action_shape')!=[8,3] or cfg.get('learning_rate')!=5e-5
        or cfg.get('memory_learning_rate')!=5e-6
        or s.get('train_sde_steps')!=3 or s.get('sde_step_indices') is not None
        or s.get('sde_exclude_last_step') is not False or s.get('noise_level')!=.1
        or t.get('kl_beta')!=50 or t.get('adv_clip_max')!=5
        or any(r.get(k) is not None for k in ('warmstart','native_fork','lr_schedule',
            'sampler_ablation','noise_level_ablation','rescue_auxiliary','preference_auxiliary'))):
        raise ValueError('FPO pilot must be fresh IL70 with fixed perMC/W3/Safety/anchor/data/LR protocol')
    return cfg


def _draws(rollout,rank,cfg=None):
    settings=settings_for_recipe(cfg)
    key=dict(schema=SCHEMA,token=rollout['token'],behavior=rollout['behavior_version'],rank=rank,seed=settings['stateless_seed'])
    seed=int.from_bytes(hashlib.sha256(json.dumps(key,sort_keys=True,separators=(',',':')).encode()).digest()[:8],'big')&((1<<63)-1)
    gen=torch.Generator(device='cpu').manual_seed(seed)
    width=.6 if settings==CONFIG else settings['sigma_max']-settings['sigma_min']
    sigma=settings['sigma_min']+width*torch.rand(settings['mc_samples'],8,generator=gen,dtype=torch.float32)
    noise=torch.randn(settings['mc_samples'],8,8,3,generator=gen,dtype=torch.float32)
    return seed,sigma,noise


def material_digest(p):return tensor_digest({k:v for k,v in p.items() if k!='material_sha256'})


@torch.no_grad()
def attach_material(actor,rollout,*,rank,cfg=None):
    if 'fpo' in rollout:raise ValueError('Finalize FPO material exactly once per fresh collection')
    settings=settings_for_recipe(cfg);protocol=protocol_for_recipe(cfg)
    clean,evidence=_fresh_evidence(rollout);seed,sigma,noise=_draws(rollout,rank,cfg)
    clean,sigma,noise=(x.to(actor.device) for x in (clean,sigma,noise))
    velocities=[];energies=[]
    with actor._context_call(rollout['cond_bg']) as cond:
        for j in range(settings['mc_samples']):
            state,target=_state(clean,sigma[j],noise[j])
            v=actor.action_velocity(state,sigma[j]*1000,cond,use_ref=False)
            velocities.append(v.detach().float().clone());energies.append(cfm_energy(v,target).detach().clone())
    p=dict(protocol=protocol,token=rollout['token'],log=rollout['log'],rank=rank,
        behavior_version=copy.deepcopy(rollout['behavior_version']),condition=copy.deepcopy(rollout['cond_bg']),
        seed=seed,evidence=evidence,clean=clean,sigma=sigma,noise=noise,
        behavior_velocity=torch.stack(velocities),old_energy=torch.stack(energies))
    p['material_sha256']=material_digest(p);rollout['fpo']=p
    object.__setattr__(actor,'_fpo_expect_fresh_material',p['material_sha256'])
    return rollout


def validate_material(rollout,*,rank,cfg=None):
    settings=settings_for_recipe(cfg);protocol=protocol_for_recipe(cfg)
    p=rollout.get('fpo',{});clean,evidence=_fresh_evidence(rollout);seed,sigma,noise=_draws(rollout,rank,cfg)
    keys={'protocol','token','log','rank','behavior_version','condition','seed','evidence',
          'clean','sigma','noise','behavior_velocity','old_energy','material_sha256'}
    if (set(p)!=keys or p.get('protocol')!=protocol or p.get('token')!=rollout['token']
        or p.get('log')!=rollout['log'] or p.get('rank')!=rank
        or p.get('behavior_version')!=rollout['behavior_version']
        or tensor_digest(p.get('condition'))!=tensor_digest(rollout['cond_bg'])
        or p.get('seed')!=seed or p.get('evidence')!=evidence
        or p.get('material_sha256')!=material_digest(p)):
        raise ValueError('FPO native material/source identity/digest changed')
    for k,want in [('clean',clean),('sigma',sigma),('noise',noise)]:
        v=p[k]
        if (not torch.is_tensor(v) or v.requires_grad or v.dtype!=torch.float32
            or v.shape!=want.shape or not torch.equal(v.cpu(),want)):
            raise ValueError('FPO fresh endpoint/private draw changed: '+k)
    for k,shape in [('behavior_velocity',(settings['mc_samples'],8,8,3)),('old_energy',(settings['mc_samples'],8))]:
        v=p[k]
        if (not torch.is_tensor(v) or v.requires_grad or v.dtype!=torch.float32
            or tuple(v.shape)!=shape or not bool(torch.isfinite(v).all())):
            raise ValueError('FPO old behavior must be finite/detached: '+k)
    for j in range(settings['mc_samples']):
        _,target=_state(clean,sigma[j],noise[j]);want=cfm_energy(p['behavior_velocity'][j].cpu(),target)
        if not torch.allclose(p['old_energy'][j].cpu(),want,atol=1e-6,rtol=3e-6):
            raise ValueError('FPO old velocity/energy mathematical audit differs')
    return p


def scale_diagnostic(rollout,cfg=None):
    settings=settings_for_recipe(cfg);protocol=protocol_for_recipe(cfg)
    p=rollout['fpo'];e=p['old_energy'].float();v=p['behavior_velocity'].float()
    if p['protocol']!=protocol:raise ValueError('Scale diagnostic FPO config/material mismatch')
    return dict(schema='world_FPO_actual_collection_scale_v1',protocol=protocol,
        token=p['token'],rank=p['rank'],behavior=p['behavior_version'],material_sha256=p['material_sha256'],
        old_energy_min=float(e.min()),old_energy_max=float(e.max()),old_energy_mean=float(e.mean()),
        old_energy_p95=float(torch.quantile(e.flatten(),.95)),velocity_rms=float(v.square().mean().sqrt()),
        sigma_min=float(p['sigma'].min()),sigma_max=float(p['sigma'].max()),
        clip_log_band=[math.log(1-settings['clip_range']),math.log(1+settings['clip_range'])],
        diagnostic_only=True,World_gradient_scale_known_from_this=False,
        interpretation='actual existingcollection CFMcache,notextraqualification; finiteenergy isnotproofgradientorperformance')


def validate_boundary(state,rollout,identity,optimizer=None,scheduler=None,*,rank=None,legacy=False):
    validate_recipe(identity['recipe'])
    protocol=protocol_for_recipe(identity['recipe'])
    if identity.get('fpo_protocol')!=protocol:raise ValueError('Missing/changed explicit FPO identity')
    s=validate_safety_efficiency_boundary(state,rollout,identity,optimizer,scheduler,legacy=legacy)
    rank=int(os.environ.get('RANK',0)) if rank is None else rank
    s.update(fpo_schema=SCHEMA,fpo_protocol=protocol,fpo_rank=rank)
    if s['pending']:
        p=validate_material(rollout,rank=rank,cfg=identity['recipe'])
        s.update(fpo_material_sha256=p['material_sha256'],fpo_behavior_update=p['behavior_version']['update'],
                 fpo_old_energy_shape=list(p['old_energy'].shape),fpo_material_rank=p['rank'])
    return s


def validate_ranks(summaries):
    validate_safety_efficiency_ranks(summaries)
    protocol=summaries[0].get('fpo_protocol')
    settings=_settings_from_protocol(protocol)
    for rank,s in enumerate(summaries):
        if s.get('fpo_schema')!=SCHEMA or s.get('fpo_protocol')!=protocol or s.get('fpo_rank')!=rank:
            raise ValueError('FPO rank protocol mismatch')
        if s['pending'] and (s.get('fpo_material_rank')!=rank or s.get('fpo_old_energy_shape')!=[settings['mc_samples'],8]
            or s.get('fpo_behavior_update')!=s['updates']-s['next_inner']
            or len(s.get('fpo_material_sha256',''))!=64):
            raise ValueError('FPO rank/MC/behavior boundary mismatch')
    return summaries


def _physical_anchor(model,rollout,beta):
    # Deliberately identical operations/order to ResearchLoss's existing anchor.
    chain,times,deltas=(rollout[k] for k in ('chain','timesteps','deltas'));indices=rollout.get('stoch_idx')
    _,terms=model.action_chain_logprobs(rollout['cond_bg'],chain,times,deltas,stoch_idx=indices,return_terms=True)
    with torch.no_grad():
        _,reference=model.action_chain_logprobs(rollout['cond_bg'],chain,times,deltas,stoch_idx=indices,use_ref=True,return_terms=True)
    d_mean_raw=(terms['mean']-reference['mean']).pow(2);d_vel_raw=(terms['vel']-reference['vel']).pow(2)
    weights=model.kl_weight_matrix(horizon=d_mean_raw.shape[2],action_dim=d_mean_raw.shape[3],device=d_mean_raw.device,dtype=d_mean_raw.dtype)
    if weights is None:
        d_mean_sq=d_mean_raw.mean(dim=(2,3));d_vel_sq=d_vel_raw.mean(dim=(2,3))
    else:
        denom=weights.sum().clamp(min=1e-12)
        d_mean_sq=(d_mean_raw*weights).sum(dim=(2,3))/denom;d_vel_sq=(d_vel_raw*weights).sum(dim=(2,3))/denom
    kl_norm=d_mean_sq/(2.0*terms['std'].pow(2))[None,:]
    if model.grpo_kl_type=='x_kl':kl_loss=kl_norm.mean()
    elif model.grpo_kl_type=='x':kl_loss=d_mean_sq.mean()
    else:kl_loss=d_vel_sq.mean()
    extra=dict(kl_div=kl_loss.detach(),kl_x_norm=kl_norm.mean().detach(),kl_x=d_mean_sq.mean().detach(),kl_v=d_vel_sq.mean().detach())
    for i,value in enumerate(d_mean_raw.mean(dim=(0,1,2)).tolist()):extra[f'kl_x_dim{i}']=torch.as_tensor(value,device=kl_loss.device)
    return beta*kl_loss,extra


class FPOTrainingLoss(ResearchLoss):
    def __init__(self,cfg):
        validate_recipe(cfg);super().__init__(cfg);self.fpo_cfg=copy.deepcopy(cfg)
        self.fpo_settings=settings_for_recipe(cfg);self.fpo_protocol=protocol_for_recipe(cfg)

    def _policy_loss(self,model,rollout):
        p=rollout['fpo']
        if p['protocol']!=self.fpo_protocol:raise ValueError('Wrong FPO training material protocol')
        current=[]
        with model._context_call(rollout['cond_bg']) as cond:
            for j in range(self.fpo_settings['mc_samples']):
                state,target=_state(p['clean'],p['sigma'][j],p['noise'][j])
                current.append(cfm_energy(model.action_velocity(state,p['sigma'][j]*1000,cond,use_ref=False),target))
        current=torch.stack(current)
        clip=self.fpo_settings['clip_range']
        pg,metrics=per_mc_ppo_objective(current,p['old_energy'],rollout['advantages'],rollout['adv_mask'],clip_range=clip)
        delta=p['old_energy']-current
        error=torch.expm1(delta).abs().max().detach()
        fresh=getattr(model,'_fpo_expect_fresh_material',None)==p['material_sha256']
        if fresh:
            if not bool(torch.isfinite(error)) or float(error)>1e-4:
                raise FloatingPointError('Fresh behavior CFM ratio differs by '+str(float(error)))
            object.__setattr__(model,'_fpo_expect_fresh_material',None)
        ratio=delta.exp().detach().T
        adv=rollout['advantages'].detach().clamp(-5,5)[:,None].expand_as(ratio)
        eligible=rollout['adv_mask'].detach().bool()[:,None].expand_as(ratio)&adv.ne(0)
        blocked=((adv>0)&(ratio>1+clip))|((adv<0)&(ratio<1-clip))
        for name,population in [('positive',eligible&adv.gt(0)),('negative',eligible&adv.lt(0)),('eligible',eligible)]:
            metrics['fpo_'+name+'_PPO_blocked_fraction']=(blocked&population).sum().float()/population.sum().clamp_min(1)
        anchor,extra=_physical_anchor(model,rollout,self._effective_kl_beta(model))
        metrics.update(fpo_proxy_ratio_max_abs_from_one=error,fpo_fresh_checked=torch.as_tensor(float(fresh),device=pg.device),
            fpo_current_energy_max=current.max().detach(),fpo_current_energy_min=current.min().detach(),
            fpo_energy_delta_abs_mean=delta.abs().mean().detach(),fpo_energy_delta_abs_max=delta.abs().max().detach(),
            fpo_energy_delta_p95=torch.quantile(delta.detach().float().flatten(),.95),
            fpo_positive_A_fraction=rollout['advantages'].gt(0).float().mean().detach(),
            fpo_negative_A_fraction=rollout['advantages'].lt(0).float().mean().detach(),
            fpo_all_A_zero=torch.as_tensor(float(not bool(rollout['advantages'].any())),device=pg.device),
            adv_kept_frac=rollout['adv_mask'].float().mean().detach())
        # Root-owned opt-in component gradient probe. It observes within the
        # actual DDP forward; does not alter gradients or require extra rollouts.
        from .gradient_probe import maybe_probe_components
        metrics.update(maybe_probe_components(model,rollout,self.fpo_cfg,{'pg':pg,'anchor':anchor}))
        total=pg+anchor
        return total,dict(policy_loss=pg.detach(),bc_loss=torch.zeros_like(pg).detach(),kl_loss=anchor.detach(),**metrics,**extra)
