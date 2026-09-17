"""Score-corrected Flow SDE; see reference_lock.json for the comparison oracle.

DDP: x(t)=(1-t)epsilon+t*action, v_t=action-epsilon, t increasing.
Upstream: sigma=1-t, v_sigma=-v_t, d_sigma=-dt.
All probability arithmetic is FP32 (FP64 inputs retain FP64 for tests).
"""
from dataclasses import dataclass
import math
import torch


def probability_tensor(x):
    return x if x.dtype == torch.float64 else x.float()


def reduce_dimensions(values, mask=None, reduction="flow_grpo_dimension_mean"):
    """Reduce only H,D. Leading scene/candidate/transition axes are preserved."""
    if reduction not in {"flow_grpo_dimension_mean", "joint_sum"}:
        raise ValueError(f"unknown logprob reduction: {reduction}")
    if mask is None:
        return (
            values.mean((-2, -1))
            if reduction == "flow_grpo_dimension_mean"
            else values.sum((-2, -1))
        )
    mask = torch.broadcast_to(mask.bool(), values.shape)
    total = torch.where(mask, values, 0).sum((-2, -1))
    count = mask.sum((-2, -1))
    if (count == 0).any():
        raise ValueError("a transition has no valid action dimensions")
    return total / count if reduction == "flow_grpo_dimension_mean" else total


def gaussian_logprob(value, mean, std):
    value, mean, std = map(probability_tensor, (value, mean, std))
    if not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError(
            "Gaussian transitions require positive finite std; ODE is evaluation-only"
        )
    return (
        -0.5 * ((value.detach() - mean) / std).square()
        - std.log()
        - 0.5 * math.log(2 * math.pi)
    )


def conditional_kl(mean, std, ref_mean, ref_std):
    mean, std, ref_mean, ref_std = map(
        probability_tensor, (mean, std, ref_mean.detach(), ref_std.detach())
    )
    if (std <= 0).any() or (ref_std <= 0).any():
        raise ValueError("KL is undefined for deterministic transitions")
    return (
        (ref_std / std).log()
        + (std.square() + (mean - ref_mean).square()) / (2 * ref_std.square())
        - 0.5
    )


@dataclass
class Transition:
    mean: torch.Tensor
    std: torch.Tensor
    diffusion: torch.Tensor

    def logprob(self, x_next):
        return gaussian_logprob(x_next, self.mean, self.std)


def transition(x, velocity, t, dt, *, noise_level, first_dt):
    x, velocity = probability_tensor(x), probability_tensor(velocity)
    t = torch.as_tensor(t, dtype=x.dtype, device=x.device)
    dt = torch.as_tensor(dt, dtype=x.dtype, device=x.device)
    first_dt = torch.as_tensor(first_dt, dtype=x.dtype, device=x.device)
    if noise_level < 0 or not math.isfinite(noise_level):
        raise ValueError("noise_level must be finite and nonnegative")
    if ((t < 0) | (t >= 1)).any() or (dt <= 0).any() or ((t + dt) > 1 + 1e-6).any():
        raise ValueError("expected 0 <= t < t+dt <= 1")
    if (first_dt <= 0).any() or (first_dt >= 1).any():
        raise ValueError("reference endpoint rule requires at least two timesteps")
    while t.ndim < x.ndim:
        t = t.unsqueeze(-1)
    while dt.ndim < x.ndim:
        dt = dt.unsqueeze(-1)
    if noise_level == 0:
        return Transition(x + velocity * dt, torch.zeros_like(t), torch.zeros_like(t))
    sigma = 1 - t
    # Exactly the upstream sigma==1 replacement: denominator 1-sigmas[1].
    denominator = torch.where(t == 0, first_dt, t)
    g = noise_level * torch.sqrt(sigma / denominator)
    mean = (
        x * (1 - g.square() / (2 * sigma) * dt)
        + velocity * (1 + g.square() * t / (2 * sigma)) * dt
    )
    return Transition(mean, g * dt.sqrt(), g)


def group_advantages(rewards, valid=None, epsilon=1e-6, clip=5.0):
    if rewards.ndim != 2:
        raise ValueError(
            "rewards must be [scene, candidate], never rank-flattened groups"
        )
    valid = (
        torch.ones_like(rewards, dtype=torch.bool) if valid is None else valid.bool()
    )
    if not torch.isfinite(rewards[valid]).all():
        raise FloatingPointError("nonfinite valid reward")
    n = valid.sum(1, keepdim=True)
    mean = torch.where(valid, rewards, 0).sum(1, keepdim=True) / n.clamp_min(1)
    delta = torch.where(valid, rewards - mean, 0)
    std = (delta.square().sum(1, keepdim=True) / n.clamp_min(1)).sqrt()
    adv = torch.where(valid & (std > 0), delta / (std + epsilon), 0)
    return adv.clamp(-clip, clip).detach()


def clipped_surrogate(current, old, advantages, clip=0.02):
    logratio = current - old.detach()
    ratio = logratio.exp()
    if not torch.isfinite(ratio).all():
        raise FloatingPointError("nonfinite importance ratio")
    a = advantages.detach()
    return -torch.minimum(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a), ratio
