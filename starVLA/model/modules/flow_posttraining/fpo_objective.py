"""Pinned PyTorch FPO arithmetic; navigation adaptations are explicit.

Reference: akanazawa/fpo@418c2554, gridworld/models/{diffusion_policy,fpo}.py.
This is a surrogate ratio, not a trajectory likelihood or KL estimator.
Physical trajectory time and the uniform flow interpolation time are unrelated.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping

import torch
from torch import distributed as dist


@dataclass(frozen=True)
class FPOConfig:
    mc_samples: int = 50
    ratio_clip: float = 0.2
    log_ratio_clip: float = 3.0
    coordinate_reduction: str = "mean"
    interpolation_time: str = "uniform_0_1"
    policy_loss_mode: str = "eval_with_grad"
    advantage: str = "within_scene_center_only"

    def __post_init__(self):
        if type(self.mc_samples) is not int or not 1 <= self.mc_samples <= 50:
            raise ValueError("Bounded FPO MC sample count must be in [1,50]")
        if not math.isfinite(self.ratio_clip) or not 0 < self.ratio_clip < 1:
            raise ValueError("FPO ratio clip must be finite and in (0,1)")
        if self.log_ratio_clip != 3.0:
            raise ValueError("Pinned FPO reference clamps each MC difference at +/-3")
        if (self.coordinate_reduction, self.interpolation_time, self.policy_loss_mode,
            self.advantage) != ("mean", "uniform_0_1", "eval_with_grad", "within_scene_center_only"):
            raise ValueError("Unsupported FPO semantics; register a separate adaptation")

    @classmethod
    def from_dict(cls, value: Mapping):
        if not isinstance(value, Mapping) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("FPO config must be fully resolved, with no unknown fields")
        return cls(**value)

    def to_dict(self):
        return asdict(self)


def _float_math(value: torch.Tensor) -> torch.Tensor:
    # Float64 is retained for the official arithmetic reference and gradcheck.
    return value if value.dtype == torch.float64 else value.float()


@dataclass(frozen=True)
class CFMSamples:
    """Saved coupled samples: [actions, MC, horizon, channels] and [actions, MC]."""
    noise: torch.Tensor
    times: torch.Tensor

    def validate(self, actions: torch.Tensor):
        if actions.ndim != 3 or self.noise.ndim != 4 or self.noise.shape[0] != len(actions):
            raise ValueError("CFM action/noise shape mismatch")
        if self.noise.shape[2:] != actions.shape[1:] or self.times.shape != self.noise.shape[:2]:
            raise ValueError("CFM noise/time shape mismatch")
        if self.noise.device != actions.device or self.times.device != actions.device:
            raise ValueError("CFM samples must share the action device")
        if not self.noise.is_floating_point() or not self.times.is_floating_point():
            raise ValueError("CFM samples must be floating point")
        if self.noise.requires_grad or self.times.requires_grad:
            raise ValueError("CFM sampling streams are not trainable")
        if not torch.isfinite(self.noise).all() or not torch.isfinite(self.times).all():
            raise ValueError("Nonfinite CFM noise/time")
        if ((self.times < 0) | (self.times >= 1)).any():
            raise ValueError("Policy time is uniform [0,1); BC Beta time is separate")

    def slice(self, index):
        return CFMSamples(self.noise[index], self.times[index])

    def state_dict(self):
        return {"noise": self.noise.detach().cpu().clone(), "times": self.times.detach().cpu().clone()}

    @classmethod
    def from_state_dict(cls, value, *, device):
        if set(value) != {"noise", "times"}:
            raise ValueError("Invalid coupled CFM checkpoint schema")
        return cls(value["noise"].to(device), value["times"].to(device))


def sample_cfm(actions: torch.Tensor, config: FPOConfig, *, noise_generator, time_generator):
    if actions.ndim != 3 or not actions.is_floating_point() or not torch.isfinite(actions).all():
        raise ValueError("CFM requires finite, actually generated trajectory endpoints")
    if noise_generator is time_generator:
        raise ValueError("Policy noise and time require independent RNG streams")
    noise = torch.randn((len(actions), config.mc_samples, *actions.shape[1:]),
                        dtype=actions.dtype, device=actions.device, generator=noise_generator)
    times = torch.rand((len(actions), config.mc_samples), dtype=actions.dtype,
                       device=actions.device, generator=time_generator)
    result = CFMSamples(noise, times)
    result.validate(actions)
    return result


def cfm_coordinate_mean(velocity: torch.Tensor, endpoint: torch.Tensor, noise: torch.Tensor):
    """OT velocity error mean over H*D, with no extra epsilon-time weighting."""
    if velocity.shape != endpoint.shape or velocity.shape != noise.shape or velocity.ndim < 3:
        raise ValueError("CFM coordinate tensors must have identical [...,H,D] shapes")
    if not torch.isfinite(velocity).all() or not torch.isfinite(endpoint).all() or not torch.isfinite(noise).all():
        raise ValueError("Nonfinite policy prediction/endpoint/noise")
    target = _float_math(endpoint.detach()) - _float_math(noise.detach())
    return (_float_math(velocity) - target).square().mean(dim=(-2, -1))


def fpo_ratio(old_cfm: torch.Tensor, current_cfm: torch.Tensor, config: FPOConfig):
    if old_cfm.shape != current_cfm.shape or current_cfm.ndim != 2 or current_cfm.shape[1] != config.mc_samples:
        raise ValueError("FPO requires [actions, registered MC samples] losses")
    if old_cfm.requires_grad:
        raise ValueError("Old policy CFM loss must be frozen/detached")
    if not torch.isfinite(old_cfm).all() or not torch.isfinite(current_cfm).all():
        raise ValueError("Nonfinite old/current CFM loss")
    difference = (_float_math(old_cfm) - _float_math(current_cfm)).clamp(-3., 3.)
    return difference.mean(dim=1).clamp(-3., 3.).exp()


def group_centered_advantages(rewards: torch.Tensor, valid: torch.Tensor, *, require_complete=True):
    """Contextual-bandit adaptation, not upstream GAE or a learned critic.

    Every registered scene remains in the batch. A constant reward group has
    exactly zero policy gradient; its original BC anchor must still be trained.
    Default rejects infrastructure failures for retry instead of dropping them.
    """
    if rewards.ndim != 2 or rewards.shape != valid.shape or valid.dtype != torch.bool:
        raise ValueError("Reward group and validity must be [scene,candidate]")
    if rewards.requires_grad:
        raise ValueError("Official feedback is detached; no reward autograd path")
    if require_complete and not valid.all():
        raise ValueError("Incomplete official feedback: retry before policy update")
    if not torch.isfinite(rewards[valid]).all():
        raise ValueError("Nonfinite valid feedback")
    clean = torch.where(valid, _float_math(rewards), 0.)
    count = valid.sum(dim=1, keepdim=True)
    mean = clean.sum(dim=1, keepdim=True) / count.clamp_min(1)
    return torch.where(valid, clean - mean, 0.).detach()


def global_action_count(valid: torch.Tensor, *, distributed=True):
    """Call once per complete optimizer batch, never per microbatch."""
    if valid.dtype != torch.bool:
        raise ValueError("Action validity must be bool")
    count = valid.sum(dtype=torch.int64).detach()
    if distributed and dist.is_available() and dist.is_initialized():
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
    return count


def fpo_surrogate(old_cfm: torch.Tensor, current_cfm: torch.Tensor, advantages: torch.Tensor,
                  valid: torch.Tensor, config: FPOConfig, *, global_count, world_size=None):
    """Sum microbatch losses; do not divide by accumulation a second time.

    DDP averages gradients, so local sum is multiplied by world/global_count.
    All-invalid local ranks retain a graph-connected zero and still participate.
    The valid population and denominator are fixed before chunking.
    """
    ratio = fpo_ratio(old_cfm, current_cfm, config)
    if advantages.shape != ratio.shape or valid.shape != ratio.shape or valid.dtype != torch.bool:
        raise ValueError("FPO advantage/mask shape mismatch")
    if advantages.requires_grad or not torch.isfinite(advantages[valid]).all():
        raise ValueError("FPO advantages must be finite detached feedback")
    world = (dist.get_world_size() if dist.is_initialized() else 1) if world_size is None else world_size
    if type(world) is not int or world < 1:
        raise ValueError("Invalid world size")
    count = torch.as_tensor(global_count, device=ratio.device)
    if (count.numel() != 1 or count.requires_grad or not torch.isfinite(count).all()
        or (count < valid.sum()).any() or (count != count.to(torch.int64)).any()):
        raise ValueError("Global denominator smaller than local valid population")
    adv = torch.where(valid, _float_math(advantages), 0.)
    per_action = -torch.minimum(ratio * adv, ratio.clamp(1-config.ratio_clip, 1+config.ratio_clip) * adv)
    numerator = torch.where(valid, per_action, 0.).sum()
    loss = numerator * world / count.clamp_min(1)
    return loss, {"ratio": ratio.detach(), "numerator": numerator.detach(),
                  "local_valid_count": valid.sum().detach(), "global_valid_count": count.detach()}
