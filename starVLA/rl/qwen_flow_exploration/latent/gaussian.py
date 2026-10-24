"""Exact diagonal-Gaussian densities for the ORIGINAL 8x3 root random variable.

These probabilities do not describe the denoising transitions, physical poses,
or a marginal trajectory likelihood. They sum, rather than average, all 24 root
coordinates. No source denoising probability clamp applies to this variable.
"""
from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class GaussianParameters:
    mean: torch.Tensor
    log_std: torch.Tensor

    def validate(self):
        if self.mean.ndim != 3 or self.mean.shape[1:] != (8, 3) or self.log_std.shape != self.mean.shape:
            raise ValueError('Root Gaussian parameters must be [scene,8,3]')
        if self.mean.dtype != self.log_std.dtype or self.mean.device != self.log_std.device:
            raise ValueError('Gaussian mean/log_std execution mismatch')
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.log_std).all():
            raise FloatingPointError('Nonfinite Gaussian parameters')
        if not torch.isfinite(self.log_std.exp()).all() or (self.log_std.exp() <= 0).any():
            raise FloatingPointError('Gaussian standard deviation is not strictly positive')
        return self

    def detached(self):
        return GaussianParameters(self.mean.detach().clone(), self.log_std.detach().clone())

    def standard_prior(self):
        return GaussianParameters(torch.zeros_like(self.mean), torch.zeros_like(self.log_std))


def _parameters_for_roots(parameters, roots):
    parameters.validate()
    if roots.ndim != 4 or roots.shape[0] != parameters.mean.shape[0] or roots.shape[2:] != (8, 3):
        raise ValueError('Saved roots must be [scene,candidate,8,3]')
    if not torch.isfinite(roots).all():
        raise FloatingPointError('Nonfinite sampled root; do not substitute an action')
    if roots.device != parameters.mean.device:
        raise ValueError('Roots and distribution parameters must be on the same device')
    return parameters.mean[:, None], parameters.log_std[:, None]


def elementwise_log_prob(parameters, roots):
    mean, log_std = _parameters_for_roots(parameters, roots)
    return -0.5*((roots-mean)*torch.exp(-log_std)).square()-log_std-0.5*math.log(2*math.pi)


def joint_log_prob(parameters, roots):
    value = elementwise_log_prob(parameters, roots).sum(dim=(-1, -2))
    if not torch.isfinite(value).all():
        raise FloatingPointError('Nonfinite joint root density')
    return value


def joint_kl(current, reference):
    current.validate()
    reference.validate()
    if current.mean.shape != reference.mean.shape or current.mean.device != reference.mean.device:
        raise ValueError('Conditional Gaussian KL requires aligned scenes and coordinates')
    # KL(N_current || N_reference), not KL in the opposite direction.
    elementwise = (reference.log_std-current.log_std
                   + 0.5*(torch.exp(2*(current.log_std-reference.log_std))
                          + (current.mean-reference.mean).square()*torch.exp(-2*reference.log_std)-1))
    value = elementwise.sum(dim=(-1, -2))
    if not torch.isfinite(value).all():
        raise FloatingPointError('Nonfinite analytic joint root KL')
    return value


def joint_entropy(parameters):
    parameters.validate()
    return (parameters.log_std+0.5*math.log(2*math.pi*math.e)).sum(dim=(-1, -2))


@torch.no_grad()
def sample_roots(parameters, candidates, generator):
    """On-policy saved samples; intentionally NOT a differentiable rsample."""
    parameters.validate()
    if not isinstance(candidates, int) or candidates <= 0:
        raise ValueError('A positive candidate count is required')
    # The original scene-bound ODE deployment draws its root on CPU. Preserve
    # that RNG contract when a CPU generator is supplied, even for CUDA policy
    # parameters; statistical N0I equivalence alone is not paired-noise parity.
    innovation = torch.randn((len(parameters.mean), candidates, 8, 3),
                             device=generator.device, dtype=parameters.mean.dtype, generator=generator)
    innovation = innovation.to(parameters.mean.device)
    roots = parameters.mean.detach()[:, None]+parameters.log_std.detach().exp()[:, None]*innovation
    if not torch.isfinite(roots).all():
        raise FloatingPointError('Nonfinite root sampling outcome')
    return roots.detach()
