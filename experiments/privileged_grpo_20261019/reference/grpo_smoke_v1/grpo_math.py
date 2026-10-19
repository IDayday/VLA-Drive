"""Forward-time rectified-flow SDE and GRPO objective for 8x4 continuous actions.

For x_u=(1-u)*noise+u*data, score=(u*v-x)/(1-u).
Forward-time drift = v + sigma^2*score/2. The u=0 endpoint diffusion
is bounded by using max(u,dt); this is a declared local discretization.
"""
import math
import torch


def transition(x, velocity, step, steps=10, eta=.1):
    u = step / steps; dt = 1 / steps
    assert 0 <= step < steps and eta >= 0
    mean = x.float() + dt * (velocity.float() + eta**2 / (2 * max(u, dt)) * (u * velocity.float() - x.float()))
    std = eta * math.sqrt((1-u) / max(u, dt) * dt)
    return mean, std


def log_prob(sample, mean, std):
    assert std > 0
    return (-.5 * ((sample.detach().float()-mean.float())/std).square() - math.log(std) - .5*math.log(2*math.pi)).flatten(1).sum(1)


def advantage(rewards):
    std = rewards.std(unbiased=False)
    return (rewards-rewards.mean()) / (std+1e-8), std


def objective(new_logp, old_logp, advantages, mean, ref_mean, std, clip=.2, beta=.01):
    ratio = torch.exp(new_logp-old_logp.detach())
    surrogate = torch.minimum(ratio*advantages.detach(), ratio.clamp(1-clip,1+clip)*advantages.detach())
    kl = ((mean-ref_mean.detach()).square() / (2*std**2)).flatten(1).sum(1)
    return (-surrogate + beta*kl).mean(), ratio, kl
