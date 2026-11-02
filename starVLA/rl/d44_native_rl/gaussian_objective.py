"""Joint24 Gaussian transitions. No innovation or density clamping."""
import torch


def gaussian_terms(current, old, reference, std, next_state):
    if current.shape != old.shape or current.shape != reference.shape or current.shape != next_state.shape:
        raise ValueError('Gaussian state shapes differ')
    if current.shape[-2:] != (8, 3) or std.shape != (current.shape[-3],):
        raise ValueError('Expected [...,W,8,3] means and [W] std')
    if not torch.isfinite(std).all() or not (std > 0).all():
        raise ValueError('Sampling std must be finite and strictly positive')
    scale = std.detach().float()[..., None, None]
    delta = (current.float()-old.detach().float())/scale
    innovation = (next_state.detach().float()-old.detach().float())/scale
    logratio = (delta*innovation-.5*delta.square()).sum((-2, -1))
    kl = (.5*((current.float()-reference.detach().float())/scale).square()).sum((-2, -1))
    if not torch.isfinite(logratio).all() or not torch.isfinite(kl).all():
        raise FloatingPointError('Nonfinite native joint Gaussian terms')
    return logratio, kl


def clipped_loss(logratio, advantages, clip=.1):
    ratio = logratio.float().exp()
    if not torch.isfinite(ratio).all():
        raise FloatingPointError('Joint ratio overflow; no hidden surrogate clamp')
    a = advantages.detach().float().clamp(-5, 5)[..., None]
    loss = torch.maximum(-a*ratio, -a*ratio.clamp(1-clip, 1+clip)).mean()
    weight = ratio.detach().flatten()
    stats = dict(ratio_mean=weight.mean(), ratio_max=weight.max(),
        joint_logratio_min=logratio.detach().min(), joint_logratio_max=logratio.detach().max(),
        clip_fraction=(ratio.detach().sub(1).abs()>clip).float().mean(),
        blocked_positive_fraction=((a>0)&(ratio.detach()>1+clip)).float().mean(),
        blocked_negative_fraction=((a<0)&(ratio.detach()<1-clip)).float().mean(),
        ESS_fraction=weight.sum().square()/(weight.square().sum()*weight.numel()).clamp_min(1e-30))
    return loss, stats
