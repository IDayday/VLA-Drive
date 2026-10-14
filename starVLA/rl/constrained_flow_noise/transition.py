"""Exact diagonal Gaussian transitions; mean-reduced PPO remains a surrogate.

Sampling sigma is the discrete transition std, NOT a diffusion coefficient.
The unchanged C_SDE kernel retains its own correlation/whitening convention.
"""
from dataclasses import dataclass
import math
import torch

from starVLA.rl.flow_grpo.math import probability_tensor
from .vendor.rlinf_sampler import gaussian_logprob as _upstream_logprob
from .vendor.rlinf_sampler import gaussian_entropy as _upstream_entropy


def require_gaussian(mean, sigma, value=None):
    if not torch.isfinite(mean).all() or not torch.isfinite(sigma).all() or (sigma <= 0).any():
        raise FloatingPointError('Training Gaussian requires finite mean and strictly positive finite std')
    if value is not None and not torch.isfinite(value).all():
        raise FloatingPointError('Nonfinite saved chain; do not replace or remove candidates')


def gaussian_logprob(value, mean, sigma):
    value, mean, sigma = map(probability_tensor, (value, mean, sigma))
    require_gaussian(mean, sigma, value)
    # Reuse the upstream arithmetic, but reject its sigma=0 inference branch.
    return _upstream_logprob(value, mean, sigma)


def gaussian_entropy(sigma):
    sigma = probability_tensor(sigma)
    require_gaussian(torch.zeros_like(sigma), sigma)
    return _upstream_entropy(sigma)


def gaussian_kl_parts(mean, sigma, reference_mean, reference_sigma):
    mean, sigma, reference_mean, reference_sigma = map(probability_tensor, (mean, sigma, reference_mean, reference_sigma))
    reference_mean, reference_sigma = reference_mean.detach(), reference_sigma.detach()
    require_gaussian(mean, sigma); require_gaussian(reference_mean, reference_sigma)
    mean_part = (mean-reference_mean).square() / (2*reference_sigma.square())
    variance_part = torch.log(reference_sigma/sigma) + sigma.square()/(2*reference_sigma.square()) - .5
    return mean_part, variance_part


def density_views(elementwise, root=None):
    if elementwise.ndim != 5 or elementwise.shape[-3:] != (10, 8, 4):
        raise ValueError('Expected [B,G,10,8,4] saved-transition log densities')
    joint_transition = elementwise.sum((-2, -1))
    joint_chain = joint_transition.sum(-1)
    if root is not None:
        if root.shape != elementwise.shape[:2] + (8, 4):
            raise ValueError('Root shape mismatch')
        joint_chain = joint_chain + (-.5*root.square()-.5*math.log(2*math.pi)).sum((-2, -1))
    return {'elementwise_logp': elementwise, 'joint_transition_logp': joint_transition,
            'mean_transition_logp': elementwise.mean((-2, -1)), 'joint_chain_logp': joint_chain}


@dataclass
class DiagonalTransition:
    mean: torch.Tensor
    std: torch.Tensor

    def sample(self, epsilon):
        require_gaussian(self.mean, self.std, epsilon)
        return self.mean + self.std*epsilon

    def logprob(self, x_next):
        return gaussian_logprob(x_next, self.mean, self.std)


def flow_noise_transition(x, velocity, std, dt=.1):
    x, velocity, std = map(probability_tensor, (x, velocity, std))
    if x.shape != velocity.shape or x.shape != std.shape or not 0 < dt <= 1:
        raise ValueError('Flow-Noise requires equally shaped x, velocity and discrete std')
    mean = x + dt*velocity
    require_gaussian(mean, std)
    return DiagonalTransition(mean, std)
