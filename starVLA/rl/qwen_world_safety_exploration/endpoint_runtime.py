"""Fixed ramped-AR1 endpoint PPO with frozen-prefix last-step replay.

A new complete policy contract, not a rho-only ablation or full-flow marginal
likelihood. Historical W3 means remain only the permanent IL anchor on ODE
states; endpoint actions/logp have an independent audited native payload.
"""
import copy
import math
import time

import torch

from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
from starVLA.rl.qwen_world_postcot.scope_contracts import MEMORY_OBSERVATION
from starVLA.rl.qwen_world_research_algorithms.policy import (
    ResearchLoss, policy_gradient_objective, ratio_diagnostics,
)
from .group_runtime import validate_recipe
from .credit import STRUCTURED_SCHEMA
from . import recovery
from .endpoint_policy import endpoint_log_prob, replay_endpoint_mean

SCHEMA='world_fixed_ramp_AR1_endpoint_PPO_v1'
BASE_STD=(1.4864179440431657,.9354143489633349,.07861934885334695)
RHO=.8
ENDPOINT_CONFIG=dict(schema=SCHEMA,rho=RHO,physical_std=list(BASE_STD),
    horizon_ramp=[i/8 for i in range(1,9)],likelihood='joint_sum24_float64',
    request_dtype='float32',prefix='deterministic_ODE_K10_shift5',
    anchor='source_W3_mean_MSE_on_stored_ODE_states')
PROTOCOL=dict(**ENDPOINT_CONFIG,group_size=8,world=8,
    density='unclamped untruncated fixed full-rank conditional Gaussian given saved last latent; not full flow marginal',
    coordinates='physical x/y metres and unwrapped yaw radians',
    sigma_origin='actual rigorous eta.1 step9 std * source codec slope [33.37,21,1.765] * horizon t/4',
    sampling='8 independent ODE roots; stationary AR1 epsilon0=z0, epsilon[h]=.8*epsilon[h-1]+.6*z[h]; no clipping',
    request='FP64 mean+noise -> FP32 before official scoring; density evaluated at that same rounded FP32 action promoted to FP64',
    rounding='continuous Gaussian log-density at stored finite-precision coordinates, not exact probability mass of FP32 quantization bins',
    prefix_gradient='none; current last-step velocity/current memory rebuilt every inner',
    anchor_definition='beta50 original weighted native-coordinate nominal SDE mean-square, W3[7,8,9], weights[.2,1,1]; not LM or endpoint Gaussian KL',
    anchor_state_change='ODE prefix states replace stochastic-chain states; explicit scientific difference',
    reductions='source effective local candidate mean then rank mean; adv cap5/PPOclip.005/SafetyEP global64 unchanged',
    deployment='unchanged single-candidate ODE10 shift5 current-only LoRA+memory; no endpoint noise at deployment')


def validate_endpoint_recipe(cfg):
    validate_recipe(cfg);r=cfg['research']
    if (r.get('endpoint_policy')!=ENDPOINT_CONFIG or cfg['grpo']['sample']['group_size']!=8
            or r['data_policy']!='uniform_full' or r['structured_advantage']!=STRUCTURED_SCHEMA
            or r.get('curriculum') is not None):
        raise ValueError('Endpoint rampAR1 is a separate G8 uniform original-Safety-credit policy contract')
    return cfg


def sigma_matrix(device='cpu'):
    return torch.tensor(BASE_STD,device=device,dtype=torch.float64)[None,:]*torch.arange(1,9,device=device,dtype=torch.float64)[:,None]/8


def ar1_from_raw(raw):
    if (raw.shape!=(8,8,3) or raw.dtype!=torch.float64 or raw.requires_grad or not torch.isfinite(raw).all()):
        raise ValueError('Detached finite FP64[G8,H8,D3] raw endpoint innovations required')
    value=[raw[:,0]]
    for h in range(1,8):value.append(RHO*value[-1]+math.sqrt(1-RHO*RHO)*raw[:,h])
    return torch.stack(value,1)


def _sync(tensor):
    if tensor.is_cuda:torch.cuda.synchronize(tensor.device)


@torch.no_grad()
def collect_endpoint(actor,condition,token,log,service,generator,*,variant,version,progress=None):
    if variant!='R_CANONICAL':raise ValueError('Canonical reward only')
    before=generator.get_state().clone();actor._noise_audit=[];tick=time.monotonic()
    chain,times,deltas,indices=actor.sample_action_chain(condition,8,deterministic=True,generator=generator)
    if chain.shape!=(8,11,8,3) or indices.numel() or not torch.isfinite(chain).all():
        raise ValueError('Endpoint prefix must be the original deterministic ODE8 chain')
    after_prefix=generator.get_state().clone()
    x=chain[:,-2].detach().clone();t=times[-1].detach().clone();dt=deltas[-1].detach().clone()
    mean,norm_mean=replay_endpoint_mean(actor,condition,x,t,dt,return_normalized=True)
    mean=mean.double()
    # Exact probe-G scale; also verify the actual current source schedule.
    _,std=actor._sde_mean_std(torch.zeros_like(x),torch.zeros_like(x),float(t)/1000,float(dt))
    derived=torch.tensor([std*v for v in (33.37,21.,1.765)],dtype=torch.float64)
    if not torch.allclose(derived,torch.tensor(BASE_STD,dtype=torch.float64),atol=1e-14,rtol=0):
        raise ValueError('Actual step9/codec scale changed from sealed structured probeG')
    raw=torch.randn((8,8,3),device='cpu',dtype=torch.float64,generator=generator)
    correlated=ar1_from_raw(raw);sigma=sigma_matrix(mean.device)
    unrounded=mean+correlated.to(mean.device)*sigma
    action=unrounded.float().detach();after_endpoint=generator.get_state().clone()
    _sync(chain);rollout_seconds=time.monotonic()-tick
    if progress:progress('generated',rollout_seconds,8)
    tick=time.monotonic();records=service.score((token,log,action.cpu().numpy()))
    if len(records)!=8:raise ValueError('Exactly eight independently scored endpoint actions required')
    scoring_seconds=time.monotonic()-tick
    if progress:progress('scored',scoring_seconds,8)
    rewards=torch.tensor([[r['readouts']['canonical_PDMS'] for r in records]],device=actor.device,dtype=torch.float32)
    tick=time.monotonic();old=endpoint_log_prob(action.double(),mean,sigma,RHO)[:,None].detach()
    endpoint=dict(protocol=copy.deepcopy(PROTOCOL),terminal_x=x,terminal_time=t,terminal_delta=dt,
        old_mean=mean.detach(),old_normalized_mean=norm_mean.detach(),action=action,
        raw_innovations=raw,correlated_noise=correlated,unrounded_action=unrounded.detach(),sigma=sigma,
        generator_before=before,generator_after_prefix=after_prefix,generator_after_endpoint=after_endpoint,
        actual_step9_std=float(std),ODE_vs_replay_mean_max_abs=float((ActionCodec.denorm_odo(chain[:,-1]).double()-mean).abs().max()),
        quantization_max_abs_by_dim=(action.double()-unrounded).abs().amax((0,1)).cpu().tolist())
    return dict(cond_bg=condition,chain=chain.detach(),timesteps=times.detach(),deltas=deltas.detach(),stoch_idx=indices,
        anchor_stoch_idx=torch.tensor([7,8,9],device=chain.device,dtype=torch.long),old_logp=old,rewards=rewards,
        endpoint_policy=endpoint,group_size=8,records=records,token=token,log=log,behavior_version=version,
        noise_audit=copy.deepcopy(actor._noise_audit),qualification_sampling_check=None,qualification_sample_terms=None,
        seconds=dict(rollout=rollout_seconds,scoring_wait=scoring_seconds,old_probability=time.monotonic()-tick,
                     scoring_cpu=sum(r['scoring_seconds'] for r in records)),protocol_sha256=service.protocol['sha256'])


def source_anchor(model,rollout):
    """The old beta50 physical-action proxy only; no old PG or fake density."""
    args=(rollout['cond_bg'],rollout['chain'],rollout['timesteps'],rollout['deltas'])
    _,current=model.action_chain_logprobs(*args,stoch_idx=rollout['anchor_stoch_idx'],return_terms=True)
    nominal_audit=copy.deepcopy(model._probability_audit['current'])
    with torch.no_grad():
        _,ref=model.action_chain_logprobs(*args,stoch_idx=rollout['anchor_stoch_idx'],use_ref=True,return_terms=True)
    sq=(current['mean']-ref['mean']).pow(2)
    weights=model.kl_weight_matrix(horizon=8,action_dim=3,device=sq.device,dtype=sq.dtype)
    per=sq.mean((2,3)) if weights is None else (sq*weights).sum((2,3))/weights.sum().clamp(min=1e-12)
    return per.mean(),nominal_audit


class EndpointLoss(ResearchLoss):
    def _policy_loss(self,model,rollout):
        payload=rollout['endpoint_policy']
        if payload['protocol']!=PROTOCOL:raise ValueError('Wrong endpoint policy protocol')
        mean=replay_endpoint_mean(model,rollout['cond_bg'],payload['terminal_x'],payload['terminal_time'],payload['terminal_delta']).double()
        logp=endpoint_log_prob(payload['action'].double(),mean,payload['sigma'],RHO)[:,None]
        adv=rollout['advantages'].detach()[:,None].to(logp)
        if self.adv_clip_max is not None:adv=adv.clamp(-self.adv_clip_max,self.adv_clip_max)
        mask=rollout['adv_mask'].detach().to(logp)
        pg,values=policy_gradient_objective(logp,rollout['old_logp'],adv,mask,algorithm='ppo',clip_range=self.ppo_clip_range)
        anchor,nominal_audit=source_anchor(model,rollout);beta=self._effective_kl_beta(model)
        total=pg+beta*anchor
        extra=ratio_diagnostics(values,adv,mask,[9],clip_range=self.ppo_clip_range,cispo_upper=1.005)
        log_ratio=(logp-rollout['old_logp']).detach()
        if not torch.isfinite(log_ratio).all():raise FloatingPointError('Nonfinite endpoint log-ratio')
        model._probability_audit['endpoint_anchor_current']=nominal_audit
        model._probability_audit['current']=dict(schema=SCHEMA,endpoint_logp=logp.detach(),endpoint_mean=mean.detach(),
            source_clamp_hit_rate=None)  # N/A for this unclamped endpoint likelihood.
        observer=getattr(self,'record_policy_components',None)
        if observer is not None:observer(pg,beta*anchor)
        metrics=dict(policy_loss=pg.detach(),bc_loss=pg.detach()*0,kl_loss=(beta*anchor).detach(),
            approx_kl=(rollout['old_logp']-logp).mean().detach(),adv_kept_frac=mask.mean().detach(),
            endpoint_logp_joint_mean=logp.mean().detach(),endpoint_logp_mean24_diagnostic=(logp/24).mean().detach(),
            endpoint_ODE_prefix_decodes=8,endpoint_actual_scored_candidates=8,
            endpoint_unscored_clean_means=8,endpoint_anchor_state_is_ODE=1,
            endpoint_quantization_max_abs=max(payload['quantization_max_abs_by_dim']),
            endpoint_ODE_replay_mean_max_abs=payload['ODE_vs_replay_mean_max_abs'],
            endpoint_log_ratio_min=log_ratio.min(),endpoint_log_ratio_max=log_ratio.max(),
            endpoint_ratio_underflow_candidates=values['ratio'].eq(0).sum(),
            anchor_nominal_source_clamp_hit_rate=nominal_audit['source_clamp_hit_rate'],**extra)
        # Exact endpoint-ratio counterpart of the existing quality diagnostics.
        good=torch.tensor([all(r['metrics'][k]==1 for k in ('no_at_fault_collisions','drivable_area_compliance',
            'time_to_collision_within_bound','comfort')) and r['metrics']['score']>=.9 for r in rollout['records']],device=logp.device)
        direction=good & torch.tensor([r['metrics']['driving_direction_compliance']==1 for r in rollout['records']],device=logp.device)
        feasible=rollout['structured_advantage']['feasible'].reshape(-1).to(logp.device)
        weights=values['derivative_weight'][:,0]/mask.sum().clamp_min(1)
        positive_high=(adv[:,0]>0)&good
        metrics.update(quality_group_size=8,quality_strict_high_count=good.sum(),quality_direction_high_count=direction.sum(),
            quality_feasible_count=feasible.sum(),quality_has_strict_high=good.any(),
            quality_positive_high_count=positive_high.sum(),quality_negative_high_count=((adv[:,0]<0)&good).sum(),
            quality_active_positive_high_weight=weights.clamp_min(0)[good].sum(),
            quality_active_negative_high_weight=(-weights).clamp_min(0)[good].sum(),
            quality_active_positive_unsafe_weight=weights.clamp_min(0)[~feasible].sum(),
            quality_active_negative_unsafe_weight=(-weights).clamp_min(0)[~feasible].sum(),
            quality_high_weight_per_good=weights.clamp_min(0)[good].sum()/good.sum().clamp_min(1),
            quality_high_positive_PPO_blocked=(positive_high&values['ppo_blocked'][:,0]).sum()/positive_high.sum().clamp_min(1))
        return total,{k:torch.as_tensor(v,device=logp.device).detach() for k,v in metrics.items()}


def _tensor(value,shape,dtype,name):
    if (not torch.is_tensor(value) or value.shape!=shape or value.dtype!=dtype or value.requires_grad
            or not torch.isfinite(value).all()):raise ValueError('Invalid detached endpoint field: '+name)


def validate_endpoint_structure(state,rollout,identity,optimizer=None,scheduler=None,*,legacy=False):
    if legacy:raise ValueError('Endpoint native checkpoints require their own schema2 material')
    cfg=identity['recipe'];validate_endpoint_recipe(cfg)
    if identity['world']!=8:raise ValueError('Endpoint requires world8')
    u,c,i=(state[k] for k in ('updates','collections','next_inner'));phase=state.get('phase')
    if not (0<=u<=cfg['max_steps'] and 0<=i<4 and c>=0) or phase not in recovery.PHASES or phase=='COLLECTING':
        raise ValueError('Uncommitted endpoint recovery phase')
    pending=phase in ('ROLLOUT_COMMITTED','INNER_COMMITTED')
    if (c!=u//4+int(pending) or i!=u%4 or phase=='ROLLOUT_COMMITTED' and i!=0
            or phase=='INNER_COMMITTED' and i==0 or phase=='COMPLETE' and u!=cfg['max_steps']
            or any(state[k]!=8*c for k in ('scene_cursor','raw_scene_exposures'))
            or state['new_candidates']!=64*c or pending!=(rollout is not None)):
        raise ValueError('Endpoint collection/update/scene/candidate boundary changed')
    summary=dict(phase=phase,updates=u,collections=c,scene_cursor=state['scene_cursor'],next_inner=i,run=identity['sha256'],pending=pending,
                 endpoint_protocol=copy.deepcopy(PROTOCOL))
    if optimizer is not None:
        opt=optimizer.state_dict() if hasattr(optimizer,'state_dict') else optimizer
        steps=[float(v['step']) for v in opt['state'].values()]
        if ((u and not steps) or any(s!=u for s in steps)
                or u and len(steps)!=sum(len(g['params']) for g in opt['param_groups'])):raise ValueError('Endpoint Adam update mismatch')
    if scheduler is not None:
        sched=scheduler.state_dict() if hasattr(scheduler,'state_dict') else scheduler
        if sched['last_epoch']!=u:raise ValueError('Endpoint scheduler update mismatch')
    if not pending:return summary
    if rollout.get('group_size')!=8 or 'old_probability' in rollout:raise ValueError('No source-W3 probability payload may masquerade as endpoint likelihood')
    fields=dict(chain=((8,11,8,3),torch.float32),timesteps=((10,),torch.float32),deltas=((10,),torch.float32),
        old_logp=((8,1),torch.float64),rewards=((1,8),torch.float32),advantages=((8,),torch.float32),adv_mask=((8,),torch.float32))
    for k,(shape,dtype) in fields.items():_tensor(rollout[k],shape,dtype,k)
    if rollout['stoch_idx'].numel()!=0 or rollout['anchor_stoch_idx'].tolist()!=[7,8,9]:raise ValueError('Actual prefixODE/nominal anchor windows changed')
    d=rollout['cond_bg']
    if d!=dict(schema=MEMORY_OBSERVATION,token=rollout['token'],identity=identity['projected_condition_identity'],repeats=8):raise ValueError('Endpoint observation ancestry changed')
    v=rollout['behavior_version']
    if v['run']!=identity['sha256'] or v['update']!=u-i:raise ValueError('Endpoint behavior version changed')
    p=rollout['endpoint_policy']
    wanted={'protocol','terminal_x','terminal_time','terminal_delta','old_mean','old_normalized_mean','action',
        'raw_innovations','correlated_noise','unrounded_action','sigma','generator_before','generator_after_prefix',
        'generator_after_endpoint','actual_step9_std','ODE_vs_replay_mean_max_abs','quantization_max_abs_by_dim'}
    if set(p)!=wanted or p['protocol']!=PROTOCOL:raise ValueError('Endpoint material protocol changed')
    material=dict(terminal_x=((8,8,3),torch.float32),terminal_time=((),torch.float32),terminal_delta=((),torch.float32),
        old_mean=((8,8,3),torch.float64),old_normalized_mean=((8,8,3),torch.float32),action=((8,8,3),torch.float32),
        raw_innovations=((8,8,3),torch.float64),correlated_noise=((8,8,3),torch.float64),unrounded_action=((8,8,3),torch.float64),sigma=((8,3),torch.float64))
    for k,(shape,dtype) in material.items():_tensor(p[k],shape,dtype,k)
    if (not torch.equal(p['terminal_x'],rollout['chain'][:,-2]) or not torch.equal(p['terminal_time'],rollout['timesteps'][-1])
            or not torch.equal(p['terminal_delta'],rollout['deltas'][-1]) or float(p['terminal_time'])<=0
            or abs(float(p['terminal_time'])/1000+float(p['terminal_delta']))>1e-6):raise ValueError('Wrong saved last ODE latent/time')
    sig=float(p['terminal_time'])/1000
    actual_std=max(.1*math.sqrt(sig/(1-sig))*math.sqrt(-float(p['terminal_delta'])),.001)
    if (not math.isfinite(p['actual_step9_std']) or abs(p['actual_step9_std']-actual_std)>1e-15
            or abs(actual_std-BASE_STD[0]/33.37)>1e-15):raise ValueError('Step9 scale provenance changed')
    if not torch.equal(p['sigma'].cpu(),sigma_matrix()):raise ValueError('Endpoint covariance scale changed')
    if not torch.equal(p['correlated_noise'].cpu(),ar1_from_raw(p['raw_innovations'].cpu())):raise ValueError('Endpoint AR1 material changed')
    unrounded=p['old_mean'].cpu()+p['correlated_noise'].cpu()*sigma_matrix()
    if not torch.equal(unrounded,p['unrounded_action'].cpu()) or not torch.equal(unrounded.float(),p['action'].cpu()):raise ValueError('Scored action is not exact FP32 endpoint sample')
    quantization=(p['action'].cpu().double()-unrounded).abs().amax((0,1)).tolist()
    if p['quantization_max_abs_by_dim']!=quantization:raise ValueError('Endpoint quantization audit changed')
    decoded=ActionCodec.denorm_odo(p['old_normalized_mean'].cpu()).double()
    if not torch.allclose(decoded,p['old_mean'].cpu(),atol=5e-6,rtol=0):raise ValueError('Old physical/normalized mean mismatch')
    expected_logp=endpoint_log_prob(p['action'].cpu().double(),p['old_mean'].cpu(),sigma_matrix(),RHO)[:,None]
    if not torch.allclose(expected_logp,rollout['old_logp'].cpu(),atol=1e-10,rtol=1e-12):raise ValueError('Old joint24 conditional logp changed')
    generator=torch.Generator(device='cpu')
    for k in ('generator_before','generator_after_prefix','generator_after_endpoint'):
        if not torch.is_tensor(p[k]) or p[k].dtype!=torch.uint8 or p[k].requires_grad:raise ValueError('Invalid private generator state')
    generator.set_state(p['generator_before'].cpu());root=torch.randn((8,8,3),dtype=torch.float32,generator=generator)
    if not torch.equal(root,rollout['chain'][:,0].cpu()) or not torch.equal(generator.get_state(),p['generator_after_prefix'].cpu()):raise ValueError('Independent ODE roots/generator budget changed')
    raw=torch.randn((8,8,3),dtype=torch.float64,generator=generator)
    if not torch.equal(raw,p['raw_innovations'].cpu()) or not torch.equal(generator.get_state(),p['generator_after_endpoint'].cpu()):raise ValueError('Endpoint innovation generator changed')
    if len(rollout['records'])!=8 or rollout['protocol_sha256']!=identity['score_protocol']['sha256']:raise ValueError('Endpoint scoring count/protocol changed')
    for j,r in enumerate(rollout['records']):
        request=torch.as_tensor(r['requested_trajectory'])
        if (r.get('status')!='OK' or r.get('token')!=rollout['token'] or r.get('log')!=rollout['log']
                or r.get('protocol_sha256')!=identity['score_protocol']['sha256'] or not r.get('cache',{}).get('sha256')
                or request.dtype!=torch.float32 or not torch.equal(request.cpu(),p['action'][j].cpu())):raise ValueError('Official endpoint score/action/cache join changed')
    summary.update(digests=recovery.rollout_digest(rollout),endpoint_digests={k:recovery.tensor_digest(v) for k,v in p.items()},
        behavior=v,rewards=rollout['rewards'].double().cpu().tolist(),advantages=rollout['advantages'].double().cpu().tolist(),
        mask=rollout['adv_mask'].cpu().tolist(),global_std=float(rollout['global_std']),
        endpoint_old_logp=rollout['old_logp'].cpu().tolist())
    return summary


class EndpointRuntimeMixin:
    rng_protocol='persistent CPU generator per rank: FP32 independent ODE roots then FP64 untruncated AR1 innovations; no repeated candidate seeds'
    def load_recipe(self,args):return validate_endpoint_recipe(super().load_recipe(args))
    def make_actor(self,cfg,args,device):
        actor,a,b=super().make_actor(cfg,args,device);actor._loss=EndpointLoss(cfg);return actor,a,b
    def collect(self,*args,**kwargs):return collect_endpoint(*args,**kwargs)
    def validate_collection(self,rollout):
        if rollout['endpoint_policy']['protocol']!=PROTOCOL:raise ValueError('Wrong endpoint collection')
    def validate_boundary(self,*args,**kwargs):
        return recovery.validate_boundary(*args,credit=self.credit,recipe_validator=validate_endpoint_recipe,
            _structure_validator=validate_endpoint_structure,**kwargs)
    def validate_ranks(self,summaries):
        if any(s.get('endpoint_protocol')!=PROTOCOL for s in summaries):raise ValueError('Endpoint rank protocol changed')
        return recovery.validate_ranks(summaries,credit=self.credit)
    @staticmethod
    def policy_ratio(actor,rollout):
        current=actor._probability_audit['current']
        if current.get('schema')!=SCHEMA:raise ValueError('Missing actual endpoint density audit')
        ratio=(current['endpoint_logp']-rollout['old_logp']).exp()
        if not torch.isfinite(ratio).all():raise FloatingPointError('Nonfinite endpoint joint ratio; no clipping or old-density rewrite')
        return ratio
    def recipe_identity(self,cfg,args,condition_sha):
        v=super().recipe_identity(cfg,args,condition_sha)
        v.update(policy_ratio='unclamped float64 joint24 conditional endpoint Gaussian at actual scored FP32 action, saved late latent',
            endpoint_policy_protocol=copy.deepcopy(PROTOCOL),exploration_protocol=dict(schema=SCHEMA,group_size=8,world=8,
                stochastic_steps=[],endpoint_noise='fixed physical ramped AR1 rho.8',global_normalization_candidates=64),
            execution_changes_vs_old_native='ODE prefix + structured endpoint noise + joint24 likelihood + terminal replay; W3 permanent IL anchor evaluated on ODE states')
        return v
