"""Fixed-covariance endpoint conditional Gaussian math and terminal replay.

StructRL (arXiv:2608.15139v1, Sec4.2/4.3, AppC) motivates this kernel. It is
only a conditional density given a saved late latent, not the flow marginal.
No runtime, noise head, optimizer, clipping or truncated noise is introduced.
Physical coordinates are (x metres, y metres, unwrapped heading radians).
"""
import math

import torch

from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec

PROTOCOL = dict(schema='world_fixed_AR1_endpoint_conditional_math_v1',
    reference='https://arxiv.org/html/2608.15139v1',
    conditioning='observation and saved detached last denoising latent; NOT full flow marginal density',
    action_shape=[8,3],units=['x_metres','y_metres','heading_radians_unwrapped'],
    covariance='diag(sigma_flat) @ (R_AR1_8 kron I_3) @ diag(sigma_flat), R[i,j]=rho**abs(i-j)',
    scale='fixed positive sigma[3] or sigma[8,3]; positive ramp supported; zero ramp rejected',
    logp='joint SUM24 with complete diagonal-scale and AR1 determinant; /24 only optional diagnostic',
    targets='saved action and reference mean detached; fixed sigma/rho must not require grad',
    replay='saved_x.detach + delta.detach * current_velocity at saved positive time; current memory rebuilt in isolated context',
    exclusions='no clamp/truncation/yaw wrapping, no covariance learning, no PPO/LR/anchor changes, no runtime registration')


def _coordinates(value, name):
    if (not torch.is_tensor(value) or value.ndim < 2 or tuple(value.shape[-2:]) != (8,3)
            or value.dtype not in (torch.float32,torch.float64) or not torch.isfinite(value).all()):
        raise ValueError('Finite FP32/FP64 [...,8,3] physical coordinates required: '+name)
    return value


def _rho(rho):
    if torch.is_tensor(rho):
        if rho.requires_grad or rho.numel()!=1:
            raise ValueError('rho is a fixed scalar, not a trainable covariance')
        rho=float(rho.detach())
    if isinstance(rho,bool) or not isinstance(rho,(float,int)) or not math.isfinite(rho) or not 0<=rho<1:
        raise ValueError('Fixed finite rho in [0,1) required')
    return float(rho)


def _sigma(sigma, like):
    if torch.is_tensor(sigma) and sigma.requires_grad:
        raise ValueError('sigma must be fixed detached physical-unit scales')
    value=torch.as_tensor(sigma,device=like.device,dtype=like.dtype)
    if value.shape not in ((3,),(8,3)) or not torch.isfinite(value).all() or not (value>0).all():
        raise ValueError('Strictly positive finite sigma[3] or sigma[8,3] required; no singular ramp')
    return value.expand(8,3)


def ar1_whiten_standardized(standardized, rho):
    """Whiten an already scale-divided [...,8,3] AR1 sequence exactly once."""
    _coordinates(standardized,'standardized residual');rho=_rho(rho)
    first=standardized[...,:1,:]
    rest=(standardized[...,1:,:]-rho*standardized[...,:-1,:])/math.sqrt(1-rho*rho)
    return torch.cat((first,rest),dim=-2)


def whiten_endpoint_residual(residual, sigma, rho):
    """D_sigma^{-1} then AR1 innovation whitening; includes one scale division."""
    _coordinates(residual,'physical residual')
    return ar1_whiten_standardized(residual/_sigma(sigma,residual),rho)


def endpoint_log_prob(action, mean, sigma, rho=0., *, return_diagnostics=False):
    """Joint conditional logp of a saved physical action; action is detached.

    Fixed scales may vary with horizon (e.g. positive t/4 ramp). Their Jacobian
    is included. No endpoint root density or prefix Jacobian is claimed here.
    """
    _coordinates(action,'saved action');_coordinates(mean,'current mean')
    if action.shape!=mean.shape or action.dtype!=mean.dtype or action.device!=mean.device:
        raise ValueError('Action/mean must have identical shape, dtype and device')
    rho=_rho(rho);scale=_sigma(sigma,mean)
    z=ar1_whiten_standardized((action.detach()-mean)/scale,rho)
    # det(L_AR1 kron I_3)=(sqrt(1-rho^2))**(7*3).
    log_abs_det=scale.log().sum()+10.5*math.log1p(-rho*rho)
    joint=-.5*z.square().sum((-1,-2))-12.*math.log(2*math.pi)-log_abs_det
    if not return_diagnostics:return joint
    return joint,dict(joint_sum24=joint,mean24_diagnostic=joint/24,
        whitened=z,log_abs_det=log_abs_det,covariance_logdet=2*log_abs_det,
        normalization='joint SUM24 is the density; mean24 is a diagnostic only')


def same_covariance_mean_kl(mean, reference_mean, sigma, rho=0., *, return_diagnostics=False):
    """KL(N(mean,Sigma)||N(frozen_reference,Sigma)), one scalar per chunk."""
    _coordinates(mean,'current mean');_coordinates(reference_mean,'frozen reference mean')
    if mean.shape!=reference_mean.shape or mean.dtype!=reference_mean.dtype or mean.device!=reference_mean.device:
        raise ValueError('Current/reference physical means must align')
    z=whiten_endpoint_residual(mean-reference_mean.detach(),sigma,rho)
    joint=.5*z.square().sum((-1,-2))
    if not return_diagnostics:return joint
    return joint,dict(joint_sum24=joint,mean24_diagnostic=joint/24,whitened_mean_difference=z)


def replay_endpoint_mean(actor, condition, stored_x, timestep, delta, *, use_ref=False, return_normalized=False):
    """Replay only the actual terminal Euler jump, then apply source affine codec.

    timestep uses the actor's native sigma*num_train_timesteps units, delta uses
    decreasing sigma units. Saved x/time/delta never carry prefix gradients.
    actor._context_call rebuilds current memory/KV for this invocation only.
    """
    _coordinates(stored_x,'stored normalized last latent')
    if stored_x.ndim!=3:raise ValueError('Replay expects [candidate,8,3]')
    x=stored_x.detach();batch=x.shape[0]
    t=torch.as_tensor(timestep,device=x.device,dtype=x.dtype).detach()
    dt=torch.as_tensor(delta,device=x.device,dtype=x.dtype).detach()
    if t.numel()==1:t=t.reshape(1).expand(batch)
    if dt.numel()==1:dt=dt.reshape(1).expand(batch)
    if t.shape!=(batch,) or dt.shape!=(batch,) or not torch.isfinite(t).all() or not torch.isfinite(dt).all():
        raise ValueError('Finite scalar or per-candidate terminal time/delta required')
    if not (t>0).all() or not (dt<0).all():raise ValueError('Positive time and negative sigma delta required')
    total=float(actor.infer_action_scheduler.num_train_timesteps)
    if not math.isfinite(total) or total<=0 or not torch.allclose(t/total+dt,torch.zeros_like(dt),atol=1e-6,rtol=0):
        raise ValueError('Replay must use the saved terminal jump to sigma0, not an earlier prefix step')
    if not callable(getattr(actor,'_context_call',None)):
        raise ValueError('Actor must provide isolated current-memory context lifetime')
    with actor._context_call(condition) as current_condition:
        velocity=actor.action_velocity(x,t,current_condition,use_ref=use_ref)
        if velocity.shape!=x.shape or not torch.isfinite(velocity).all():raise ValueError('Invalid replayed velocity')
        normalized=x+velocity*dt[:,None,None]
    physical=ActionCodec.denorm_odo(normalized)
    if not torch.isfinite(physical).all():raise FloatingPointError('Nonfinite physical endpoint mean')
    return (physical,normalized) if return_normalized else physical
