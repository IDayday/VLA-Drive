"""D action-only JaT-inspired joint-Gaussian PPO on the existing W3 kernel.

The AUGUstus paper supplies the equal-covariance quadratic identity. Its
released implementation averages coordinates; this explicitly registered
adaptation SUMS24 coordinates. It retains our rigorous eta.1 W3 kernel, not
their lambda*t kernel, smoothing, shared root, or action+CoT objective.
"""
from __future__ import annotations
import copy
import math
import torch
import torch.distributed as dist

from starVLA.rl.simwam_recipe.trainer_adapter import SourceLoss
from starVLA.rl.qwen_world_compare.recovery_state import tensor_digest

SCHEMA='abcd_D_action_JaT_joint_gaussian_v1'
ADV_SCHEMA='abcd_D_canonical_centered_global64_std_v1'
CONFIG=dict(schema=SCHEMA,coordinate_reduction='sum_24coords',step_reduction='mean_W3',
    clip_range=.2,reference_KL_coefficient=.001,
    advantage=ADV_SCHEMA,innovation_clip=None,source_kernel='rigorous_eta0.1_W3_7_8_9')
PROTOCOL=dict(**CONFIG,
    logratio='sum_H,D[delta_mean/std*innovation - .5*(delta_mean/std)^2] for each actual stochastic transition',
    reference='joint diagonal Gaussian KL=.5*sum_H,D[((mean_current-mean_ownIL)/std)^2], then meanG,W',
    physical_mean_anchor=False,CoT_objective=False,readout_objective=False,
    lifetime='oldmean/std/recorded_next/innovation and canonicalA fixed for all4inners; currentD C/P rebuilt with gradients each current call',
    credit='canonical reward group-centered / populationstd of64 group-centered rewards +1e-8; source Acap5',
    sampling='sourceW3 rigorouseta.1 with unclipped Gaussian innovations; independent source initial roots, no smoothing',
    deployment='unchanged selected ownD clean-x K10 shift5 singleODE; preserve LoRA wrappers',
    transfer='paper-inspired action-only joint_sum adaptation, not official AUGUstus jointCoT or lambda_t reproduction')


def validate_recipe(cfg):
    r=cfg.get('research',{});s=cfg['grpo']['sample'];t=cfg['grpo']['train']
    if (cfg.get('architecture_id')!='D' or r.get('d_objective')!=CONFIG
        or cfg.get('learning_rate')!=1e-6 or cfg.get('lr_scheduler_type')!='constant'
        or cfg.get('world_size')!=8 or cfg.get('seed')!=42 or cfg.get('action_shape')!=[8,3]
        or 'randn_clip' not in s or s.get('randn_clip') is not None or s.get('noise_level')!=.1 or s.get('sde_mode')!='rigorous'
        or s.get('train_sde_steps')!=3 or s.get('sde_exclude_last_step') is not False
        or s.get('sde_step_indices') is not None or s.get('merge_sde_window') is not False
        or t.get('ppo_clip_range')!=.2 or t.get('kl_beta')!=.001 or t.get('kl_type')!='x_kl'
        or t.get('kl_dim_weights') is not None or t.get('kl_horizon_weights') is not None
        or t.get('adv_clip_max')!=5 or t.get('use_bc_loss') is not False
        or cfg.get('variant')!='R_CANONICAL'):
        raise ValueError('D requires its explicit action-only jointGaussian/nativeKL/noise/credit recipe')
    return cfg


def old_terms(rollout):
    old=rollout['old_probability'].get('current',rollout['old_probability'])
    mean,std=old['mean'],old['std']
    if (tuple(mean.shape)!=(8,3,8,3) or tuple(std.shape)!=(3,)
        or mean.dtype!=torch.float32 or std.dtype!=torch.float32 or mean.requires_grad or std.requires_grad
        or not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()) or not bool((std>0).all())):
        raise ValueError('Finite detached source oldmean[8,3,8,3]/std[3] required')
    indices=rollout['stoch_idx']
    if indices.tolist()!=[7,8,9]:raise ValueError('D joint ratios only cover actual W3 stochastic steps')
    nxt=rollout['chain'].index_select(1,indices.to(rollout['chain'].device)+1).detach().float()
    innovation=(nxt-mean)/std[None,:,None,None]
    return mean,std,nxt,innovation


def material_digest(value):return tensor_digest({k:v for k,v in value.items() if k!='sha256'})


@torch.no_grad()
def centered_advantages(rewards,*,all_rewards=None):
    values=rewards.detach().double()
    if values.shape[-1]!=8 or values.ndim!=2 or not bool(torch.isfinite(values).all()) or not bool(((values>=0)&(values<=1)).all()):
        raise ValueError('Finite canonical [scene,G8] rewards required')
    if all_rewards is None:
        if dist.is_available() and dist.is_initialized():
            gathered=[torch.empty_like(values) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered,values);all_rewards=torch.cat(gathered,0)
        else:all_rewards=values
    all_rewards=all_rewards.detach().to(values).double()
    if all_rewards.ndim!=2 or all_rewards.shape[-1]!=8 or not bool(torch.isfinite(all_rewards).all()):
        raise ValueError('Invalid global canonical rewards')
    offsets=values-values[:,:1];centered=offsets-offsets.mean(1,keepdim=True)
    all_offsets=all_rewards-all_rewards[:,:1];all_centered=all_offsets-all_offsets.mean(1,keepdim=True)
    scale=all_centered.std(unbiased=False);groupstd=offsets.std(1,unbiased=False,keepdim=True)
    return dict(advantages=(centered/(scale+1e-8)).reshape(-1).float(),
        adv_mask=(groupstd>1e-8).expand_as(values).reshape(-1).float(),
        global_std=all_rewards.std(unbiased=False),raw_group_std=groupstd,normalization_std=scale,
        advantage_statistics=dict(schema=ADV_SCHEMA,normalization_population_groups=int(all_rewards.shape[0]),
            normalization_population_candidates=int(all_rewards.numel()),normalization=PROTOCOL['credit'],
            all_full=int((values==1).all(1).sum()),all_zero=int((values==0).all(1).sum()),
            tied=int((groupstd<=1e-8).sum()),effective_PG_candidate_fraction=float((groupstd>1e-8).double().mean())))


@torch.no_grad()
def finalize_advantages(rollout):
    if 'advantages' in rollout or 'd_gaussian' in rollout:raise ValueError('Finalize fresh D material once')
    rewards=rollout['rewards']
    for record in rollout['records']:
        m=record['metrics'];v=[float(m[k]) for k in ('score','no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','ego_progress','comfort')]
        if (record.get('status')!='OK' or any(not math.isfinite(x) or not 0<=x<=1 for x in v)
            or abs(v[0]-v[1]*v[2]*(5*v[4]+5*v[3]+2*v[5])/12)>5e-15):
            raise ValueError('D requires actual successful canonical feedback')
    expected=torch.tensor([[r['metrics']['score'] for r in rollout['records']]],device=rewards.device,dtype=torch.float32)
    if not torch.equal(rewards,expected):raise ValueError('D credit requires unchanged canonical scores')
    rollout.update(centered_advantages(rewards))
    mean,std,nxt,innovation=old_terms(rollout)
    payload=dict(protocol=copy.deepcopy(PROTOCOL),token=rollout['token'],log=rollout['log'],
        behavior_version=copy.deepcopy(rollout['behavior_version']),condition=copy.deepcopy(rollout['cond_bg']),
        old_mean=mean.clone(),std=std.clone(),recorded_next=nxt.clone(),innovation=innovation.clone(),
        source_sha256=tensor_digest(dict(mean=mean,std=std,next=nxt,stoch_idx=rollout['stoch_idx'])))
    payload['sha256']=material_digest(payload);rollout['d_gaussian']=payload
    return rollout


def validate_material(rollout):
    p=rollout.get('d_gaussian',{})
    if (set(p)!={'protocol','token','log','behavior_version','condition','old_mean','std','recorded_next',
                    'innovation','source_sha256','sha256'} or p.get('protocol')!=PROTOCOL
        or p.get('token')!=rollout['token'] or p.get('log')!=rollout['log']
        or p.get('behavior_version')!=rollout['behavior_version'] or p.get('condition')!=rollout['cond_bg']
        or p.get('sha256')!=material_digest(p)):
        raise ValueError('D native material/source protocol/digest changed')
    mean,std,nxt,innovation=old_terms(rollout)
    for key,want in [('old_mean',mean),('std',std),('recorded_next',nxt),('innovation',innovation)]:
        v=p[key]
        if not torch.is_tensor(v) or v.requires_grad or v.dtype!=torch.float32 or not torch.equal(v.cpu(),want.cpu()):
            raise ValueError('D saved source Gaussian material differs: '+key)
    if p['source_sha256']!=tensor_digest(dict(mean=mean,std=std,next=nxt,stoch_idx=rollout['stoch_idx'])):
        raise ValueError('D source Gaussian ancestry changed')
    return p


def gaussian_logratio_kl(current_mean,old_mean,reference_mean,std,recorded_next):
    """Exact24-dimensional Gaussian identities, evaluated in FP32, no clamp."""
    if (current_mean.shape!=(8,3,8,3) or old_mean.shape!=current_mean.shape
        or reference_mean.shape!=current_mean.shape or recorded_next.shape!=current_mean.shape
        or std.shape!=(3,) or not bool(torch.isfinite(std).all()) or not bool((std>0).all())):
        raise ValueError('D equal-covariance Gaussian transition shape/variance mismatch')
    scale=std.detach().float()[None,:,None,None]
    delta=(current_mean.float()-old_mean.detach().float())/scale
    innovation=(recorded_next.detach().float()-old_mean.detach().float())/scale
    logr=(delta*innovation-.5*delta.square()).sum((-1,-2))
    kl=(.5*((current_mean.float()-reference_mean.detach().float())/scale).square()).sum((-1,-2))
    if not bool(torch.isfinite(logr).all() and torch.isfinite(kl).all()):
        raise FloatingPointError('Nonfinite D joint Gaussian logratio/KL')
    return logr,kl


def clipped_objective(logratio,kl,advantages,mask):
    ratio=logratio.exp()
    if not bool(torch.isfinite(ratio).all()):raise FloatingPointError('Nonfinite D joint Gaussian ratio')
    a=advantages.detach().float().clamp(-5,5)[:,None].expand_as(ratio)
    keep=mask.detach().to(ratio)[:,None];den=keep.sum().clamp_min(1)*3
    pg=(torch.maximum(-a*ratio,-a*ratio.clamp(.8,1.2))*keep).sum()/den
    anchor=kl.mean()  # Native ownIL KL remains active on tied-reward groups.
    return pg+.001*anchor,dict(policy_loss=pg.detach(),kl_loss=anchor.detach(),
        bc_loss=pg.detach()*0,kl_div=anchor.detach(),d_reference_KL_joint=anchor.detach(),
        d_weighted_reference_KL=(.001*anchor).detach(),ratio_mean=ratio.mean().detach(),
        clipfrac=(ratio.sub(1).abs()>.2).float().mean().detach(),
        d_joint_ratio_max_abs_from_one=ratio.sub(1).abs().max().detach(),
        d_joint_logratio_min=logratio.min().detach(),d_joint_logratio_max=logratio.max().detach(),
        d_PPO_blocked_fraction=(((a>0)&(ratio>1.2))|((a<0)&(ratio<.8))).float().mean().detach(),
        approx_kl=(-logratio.mean()).detach(),adv_kept_frac=mask.float().mean().detach())


class DActionJaTLoss(SourceLoss):
    def __init__(self,cfg):validate_recipe(cfg);super().__init__(cfg);self.cfg=copy.deepcopy(cfg)

    def _policy_loss(self,model,rollout):
        if not math.isinf(model.grpo_randn_clip):raise ValueError('D Gaussian ratio requires genuinely unclipped innovations')
        p=rollout['d_gaussian']
        if p['protocol']!=PROTOCOL:raise ValueError('Wrong D loss protocol')
        args=(rollout['cond_bg'],rollout['chain'],rollout['timesteps'],rollout['deltas'])
        _,terms=model.action_chain_logprobs(*args,stoch_idx=rollout['stoch_idx'],return_terms=True)
        with torch.no_grad():
            _,ref=model.action_chain_logprobs(*args,stoch_idx=rollout['stoch_idx'],use_ref=True,return_terms=True)
        if not torch.equal(terms['std'],p['std']) or not torch.equal(ref['std'],p['std']):
            raise ValueError('D policy-independent Gaussian covariance changed')
        logr,kl=gaussian_logratio_kl(terms['mean'],p['old_mean'],ref['mean'],p['std'],p['recorded_next'])
        fresh=getattr(model,'_d_expect_fresh_material',None)==p['sha256']
        if fresh:
            error=logr.detach().expm1().abs().max()
            if not bool(torch.isfinite(error)) or float(error)>1e-4:
                raise FloatingPointError('D fresh joint Gaussian ratio differs from1: '+str(float(error)))
            object.__setattr__(model,'_d_expect_fresh_material',None)
        total,metrics=clipped_objective(logr,kl,rollout['advantages'],rollout['adv_mask'])
        metrics['d_fresh_joint_ratio_checked']=torch.as_tensor(float(fresh),device=total.device)
        return total,metrics


def validate_boundary_summary(summary,state,rollout,identity):
    validate_recipe(identity['recipe'])
    if identity.get('architecture_id')!='D' or identity.get('d_objective_protocol')!=PROTOCOL:
        raise ValueError('D native objective identity mismatch')
    summary.update(d_objective_schema=SCHEMA,d_objective_protocol=copy.deepcopy(PROTOCOL))
    if summary['pending']:
        p=validate_material(rollout);scale=rollout.get('normalization_std');meta=rollout['advantage_statistics']
        if (not torch.is_tensor(scale) or scale.ndim!=0 or scale.requires_grad or not bool(torch.isfinite(scale))
            or scale.dtype!=torch.float64 or float(scale)<0 or meta.get('schema')!=ADV_SCHEMA
            or meta.get('normalization_population_groups')!=identity['world']
            or meta.get('normalization_population_candidates')!=identity['world']*8):
            raise ValueError('D global centered advantage scale changed')
        rewards=rollout['rewards'].double();want_std=(rewards-rewards[:,:1]).std(1,unbiased=False,keepdim=True)
        groupstd=rollout['raw_group_std']
        if (not torch.is_tensor(groupstd) or groupstd.requires_grad or groupstd.dtype!=torch.float64
            or groupstd.shape!=want_std.shape or not torch.allclose(groupstd,want_std,atol=1e-12,rtol=0)):
            raise ValueError('D canonical raw group std changed')
        summary.update(d_material_sha256=p['sha256'],d_behavior_update=p['behavior_version']['update'],
            d_normalization_std=float(scale))
    return summary


def validate_ranks(summaries):
    if len(summaries)!=8:raise ValueError('D requires eight rank summaries')
    common=('phase','updates','collections','scene_cursor','next_inner','run','pending','d_objective_schema','d_objective_protocol')
    for s in summaries:
        if (s.get('architecture_id')!='D' or s.get('d_objective_schema')!=SCHEMA or s.get('d_objective_protocol')!=PROTOCOL
            or {k:s[k] for k in common}!={k:summaries[0][k] for k in common}):
            raise ValueError('D rank objective/boundary mismatch')
    if not summaries[0]['pending']:return summaries
    if any(s['behavior']!=summaries[0]['behavior'] for s in summaries):raise ValueError('D ranks changed collection behavior')
    rewards=torch.tensor([s['rewards'][0] for s in summaries],dtype=torch.float64)
    expected=centered_advantages(rewards,all_rewards=rewards)
    for rank,s in enumerate(summaries):
        if (abs(s['global_std']-float(expected['global_std']))>1e-12
            or abs(s['d_normalization_std']-float(expected['normalization_std']))>1e-12
            or not torch.allclose(torch.tensor(s['advantages']),expected['advantages'].reshape(8,8)[rank],atol=1e-7,rtol=1e-6)
            or s['mask']!=expected['adv_mask'].reshape(8,8)[rank].tolist()
            or s.get('d_behavior_update')!=s['updates']-s['next_inner'] or len(s.get('d_material_sha256',''))!=64):
            raise ValueError('D advantage/material was not globally finalized')
    return summaries
