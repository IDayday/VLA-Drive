"""AUGUstus official JaT action math, adapted only at the D actor boundary.

Pinned upstream math is imported, not re-derived here. State/probability FP32;
BF16 clean-x predictions cast to FP32. The coordinate-mean log ratio / KL are
upstream's convention, NOT joint-sum24 likelihood. No trajectory/noise smoothing.
This does not claim official architecture, CoT, full-system or scope reproduction.
"""
import torch
from .vendor.fm_sde_logprob import fm_sde_step_with_logprob_x0, fm_sde_logprob_from_mean_std as log_prob
from .vendor.jit_grpo import jit_grpo_step_logratio_kl,jit_grpo_policy_loss

PROTOCOL=dict(schema='d46_AUGUstus_pinned_JaT_action_lambda_t_v1',
    upstream_commit='edf3422304e4b2987e839a9212cf1176782e7380',noise_schedule='lambda_t',
    noise_scale=[.05,.05,.02],diffusion='g(t)=a*sqrt(t), g_squared=a_squared*t',
    state_dtype='float32',probability_dtype='float32',coordinate_reduction='mean_8x3',
    clip=.2,reference_KL_coefficient=.001,noise='independent standard Gaussian; no smoothing/truncation',
    training_grid=[1.,2/3,1/3,0.],deployment_grid='project original clean-x K10 shift5; explicit adaptation',
    advantages='external group-relative canonical reward; not old SafetyEP/globalStd',
    permanent_reference='same original IL46 deployment policy; no AdaptiveKL',full_system_reproduction=False)


def _inputs(x0,state,sigma,sigma_next,noise_scale):
    if state.ndim!=3 or state.shape[1:]!=(8,3)or x0.shape!=state.shape:
        raise ValueError('JaT action must have aligned [G,8,3] states/predictions')
    state=state.detach().float();x0=x0.float()
    t=torch.as_tensor(sigma,device=state.device,dtype=torch.float32)
    nxt=torch.as_tensor(sigma_next,device=state.device,dtype=torch.float32)
    if t.ndim==0:t=t.expand(len(state))
    if nxt.ndim==0:nxt=nxt.expand(len(state))
    if t.shape!=(len(state),)or nxt.shape!=t.shape:raise ValueError('Time must be scalar or [G]')
    scale=torch.as_tensor(noise_scale,device=state.device,dtype=torch.float32)
    if scale.shape not in((3,),(8,3))or not torch.isfinite(scale).all()or not(scale>0).all():raise ValueError('Positive finite [3] or per-pose [8,3] anisotropic noise scales required')
    if (not torch.isfinite(t).all()or not torch.isfinite(nxt).all()or(t<=0).any()or(t>1).any()
        or(nxt<0).any()or(nxt>=t).any()):raise ValueError('Strict denoising time 0<=next<t<=1 required')
    return x0,state,t,nxt,scale


def step(x0,state,sigma,sigma_next,noise_scale=(.05,.05,.02),noise=None):
    """Return next/mean/std[G,8,3]; noise is actual iid N(0,1), never smoothed.

    Passing explicit noise supports the caller's native private CPU generator
    followed by device transfer, without changing the upstream kernel formula.
    None uses the upstream same-device normal sampler (global torch generator).
    """
    x0,state,t,nxt,scale=_inputs(x0,state,sigma,sigma_next,noise_scale)
    kwargs=dict(noise_level=scale,schedule='lambda_t')
    if noise is None:
        result,_,mean,std=fm_sde_step_with_logprob_x0(x0,state,t,nxt,**kwargs)
    else:
        if noise.shape!=state.shape:raise ValueError('Noise must be independent Gaussian [G,8,3]')
        # A fixed placeholder asks the official function for its EXACT mean/std;
        # the caller-provided independent innovation then uses those same terms.
        _,_,mean,std=fm_sde_step_with_logprob_x0(x0,state,t,nxt,prev_sample=state,**kwargs)
        result=mean+std*noise.detach().to(device=state.device,dtype=torch.float32)
    result=result.float();mean=mean.float();std=std.float().expand_as(state)
    if not torch.isfinite(result).all()or not torch.isfinite(mean).all()or not torch.isfinite(std).all():raise FloatingPointError('Nonfinite official JaT kernel')
    return result,mean,std


def stats(new_x0,old_x0,ref_x0,state,next,sigma,sigma_next,noise_scale=(.05,.05,.02)):
    """Official coordinate-mean log ratio and KL(current || frozen reference).

    Equal covariance makes this KL symmetric. Only new_x0 carries gradients.
    The reference is not substituted into the behavior importance ratio.
    """
    new_x0,state,t,nxt,scale=_inputs(new_x0,state,sigma,sigma_next,noise_scale)
    if old_x0.shape!=state.shape or ref_x0.shape!=state.shape or next.shape!=state.shape:raise ValueError('JaT replay arrays must all be [G,8,3]')
    args=dict(sample=state,prev_sample=next.detach().float(),sigma=t,sigma_prev=nxt,noise_level=scale,schedule='lambda_t')
    logratio,_=jit_grpo_step_logratio_kl(new_x0,old_x0.detach().float(),**args)
    _,reference_kl=jit_grpo_step_logratio_kl(new_x0,ref_x0.detach().float(),**args)
    if not torch.isfinite(logratio).all()or not torch.isfinite(reference_kl).all()or(reference_kl<0).any():raise FloatingPointError('Nonfinite official JaT ratio/KL')
    return logratio,reference_kl


def policy_loss(logratio_GW,advantages_G,refKL_GW,clip=.2,beta=.001):
    """Call upstream PPO loss unchanged; add detached tensor diagnostics only.

    No advantage clipping, density clamp, global re-normalization, entropy or
    critic. The fixed reference KL is inside this total once, not extra beta50.
    """
    if logratio_GW.ndim!=2 or refKL_GW.shape!=logratio_GW.shape or advantages_G.shape!=(len(logratio_GW),):raise ValueError('Expected ratio/KL[G,W] and advantages[G]')
    loss,raw=jit_grpo_policy_loss(logratio_GW,advantages_G.detach(),kl=refKL_GW,clip_range=clip,kl_coef=beta)
    detached=logratio_GW.detach().float();ratio=detached.exp();a=advantages_G.detach().float()[:,None]
    metrics={k:detached.new_tensor(v)for k,v in raw.items()}
    metrics.update(pg=detached.new_tensor(raw['jit_grpo/policy_loss']),reference_KL=refKL_GW.detach().float().mean(),reference_KL_joint=24*refKL_GW.detach().float().mean(),
        reference_KL_per_coordinate=refKL_GW.detach().float().mean(),reference_KL_coordinate_mean=refKL_GW.detach().float().mean(),
        # Joint KL is diagnostic sum24 only; the loss above charges beta times
        # coordinate mean, exactly upstream. Ratio remains exp(coordinate mean).
        ratio_mean=ratio.mean(),ratio_max=ratio.max(),logratio_min=detached.min(),logratio_max=detached.max(),joint_logratio_min=24*detached.min(),joint_logratio_max=24*detached.max(),
        coordinate_mean_logratio_min=detached.min(),coordinate_mean_logratio_max=detached.max(),
        clip_fraction=(ratio.sub(1).abs()>clip).float().mean(),blocked_positive_fraction=((a>0)&(ratio>1+clip)).float().mean(),
        blocked_negative_fraction=((a<0)&(ratio<1-clip)).float().mean(),ESS_fraction=ratio.sum().square()/(ratio.square().sum()*ratio.numel()).clamp_min(1e-30),
        auxiliary_loss=detached.new_tensor(0.),fresh_ratio_error=ratio.sub(1).abs().max())
    return loss,metrics
