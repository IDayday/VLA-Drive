"""Original DiT adapter and explicit policy/RNG identities; no second encoder."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .fpo_objective import CFMSamples, cfm_coordinate_mean


@contextmanager
def policy_eval_mode(module: nn.Module):
    """Deterministic mathematical policy mode, with autograd still enabled.

    Restore each submodule's exact previous mode. BC runs separately in its
    original train mode. Old/current must use this SAME dropout contract.
    """
    previous = [(child, child.training) for child in module.modules()]
    module.eval()
    try:
        yield
    finally:
        for child, training in previous:
            child.training = training


def tensor_hash(value: torch.Tensor):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(json.dumps([str(value.dtype), list(value.shape)], separators=(",", ":")).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def module_hash(module: nn.Module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor_hash(value).encode())
    return digest.hexdigest()


class FlowRandomStreams:
    """Independent rollout, policy-MC and unchanged BC streams, all resumable."""
    NAMES = ("rollout_noise", "policy_noise", "policy_time", "bc_noise", "bc_time")

    def __init__(self, seed: int, *, device="cpu"):
        if type(seed) is not int or seed < 0:
            raise ValueError("RNG seed must be a nonnegative integer")
        self.seed, self.device = seed, str(torch.device(device))
        self.generators = {}
        for name in self.NAMES:
            offset = int.from_bytes(hashlib.sha256(f"fpo_v1:{seed}:{name}".encode()).digest()[:8], "little")
            self.generators[name] = torch.Generator(device=device).manual_seed(offset % (2**63-1))

    def __getitem__(self, name):
        return self.generators[name]

    def state_dict(self):
        return {"schema": "fpo_rng_v1", "seed": self.seed, "device": self.device,
                "states": {name: gen.get_state().cpu().clone() for name, gen in self.generators.items()}}

    def load_state_dict(self, state):
        if (set(state) != {"schema", "seed", "device", "states"} or state["schema"] != "fpo_rng_v1"
            or state["seed"] != self.seed or state["device"] != self.device
            or set(state["states"]) != set(self.NAMES)):
            raise ValueError("RNG contract mismatch; cross-device restoration is not exact")
        # Validate every saved generator before mutating any live stream.
        replacements = {}
        for name in self.NAMES:
            generator = torch.Generator(device=self.device)
            generator.set_state(state["states"][name].cpu())
            replacements[name] = generator
        self.generators = replacements


class DiTFlowPolicy(nn.Module):
    """Own the existing action head only; Qwen features enter detached.

    This adapter does not initialize a new head or change its state_dict tensors.
    It uses the original qwen_proj and forward_onestep paths, including the
    original integer timestep buckets and raw x/y/sin/cos action representation.
    """
    def __init__(self, action_head: nn.Module):
        super().__init__()
        required = ("qwen_proj", "forward_onestep", "predict_action", "config", "num_timestep_buckets")
        if any(not hasattr(action_head, key) for key in required):
            raise ValueError("Original FlowmatchingActionHead contract required")
        if int(action_head.num_inference_timesteps) != 10:
            raise ValueError("This campaign preserves the original ten Euler steps")
        self.action_head = action_head
        self.horizon = int(action_head.action_horizon)
        self.channels = int(action_head.action_dim)
        if (self.horizon, self.channels) != (8, 4):
            raise ValueError("Current bound NAVSIM raw action contract is 8 x 4 (xy/sin/cos)")

    def _check_condition(self, condition):
        if condition.ndim != 3 or not condition.is_floating_point() or not torch.isfinite(condition).all():
            raise ValueError("Frozen planner condition must be finite [batch,tokens,channels]")

    def project_condition(self, condition):
        self._check_condition(condition)
        return self.action_head.qwen_proj(condition.detach())

    def velocity(self, projected, state, times):
        if state.shape != (len(projected), self.horizon, self.channels) or times.shape != (len(projected),):
            raise ValueError("DiT velocity input shape mismatch")
        buckets = (times * self.action_head.num_timestep_buckets).long()
        output = self.action_head.forward_onestep(projected, state, buckets)
        if not isinstance(output, (list, tuple)) or len(output) != 1 or output[0].shape != state.shape:
            raise ValueError("Original single-trajectory velocity output contract changed")
        return output[0]

    def _eval_velocity(self, projected, state, times):
        # This function also wraps checkpoint recomputation, after outer forward
        # has restored train-mode for the independent BC branch.
        with policy_eval_mode(self):
            return self.velocity(projected, state, times)

    def forward(self, condition, actions, samples: CFMSamples, *, mc_chunk=1, checkpoint_velocity=False):
        """Per-action [N,MC] CFM losses. No action, feature, reward or old gradient.

        Caller owns the original autocast boundary. A small MC chunk limits
        temporary expansion. checkpoint_velocity is an explicit execution option,
        not a hidden change to the training objective or MC sample population.
        """
        self._check_condition(condition)
        if actions.shape != (len(condition), self.horizon, self.channels):
            raise ValueError("Scored action does not match the bound raw contract")
        if actions.requires_grad:
            raise ValueError("Policy endpoints must be detached actual old-policy rollouts")
        if type(mc_chunk) is not int or mc_chunk < 1:
            raise ValueError("MC chunk must be a positive integer")
        samples.validate(actions)
        with policy_eval_mode(self):
            projected = self.project_condition(condition)
            losses = []
            for start in range(0, samples.noise.shape[1], mc_chunk):
                noise = samples.noise[:, start:start+mc_chunk]
                times = samples.times[:, start:start+mc_chunk]
                count = noise.shape[1]
                endpoint = actions[:, None].expand_as(noise).detach()
                state = (1-times[..., None, None]) * noise + times[..., None, None] * endpoint
                memory = projected[:, None].expand(-1, count, -1, -1).reshape(-1, *projected.shape[1:])
                flat_state = state.reshape(-1, self.horizon, self.channels)
                flat_times = times.reshape(-1)
                if checkpoint_velocity and torch.is_grad_enabled():
                    pred = checkpoint(self._eval_velocity, memory, flat_state, flat_times,
                                      use_reentrant=False, preserve_rng_state=True)
                else:
                    pred = self.velocity(memory, flat_state, flat_times)
                losses.append(cfm_coordinate_mean(pred.reshape_as(noise), endpoint, noise))
            return torch.cat(losses, dim=1)

    def bc_anchor(self, condition, targets, *, noise_generator, time_generator,
                  global_scene_count, world_size=1):
        """Original Beta(1.5,1), s=.999, repeat8 FM in current train-mode.

        This calls the original action-head forward without replacing its loss.
        No auxiliary target, reward or policy MC sample enters the BC anchor.
        Sum microbatches; no extra accumulation division is needed.
        """
        self._check_condition(condition)
        cfg = self.action_head.config
        if not self.action_head.training or not self.training:
            raise ValueError("BC anchor must retain original train/dropout mode")
        if (int(cfg.repeated_diffusion_steps), float(cfg.noise_beta_alpha), float(cfg.noise_beta_beta),
            float(cfg.noise_s)) != (8, 1.5, 1., .999):
            raise ValueError("Original bound BC noise/time/repeat contract changed")
        if targets.shape != (len(condition), self.horizon, self.channels) or not torch.isfinite(targets).all():
            raise ValueError("Invalid original BC target")
        if noise_generator is time_generator:
            raise ValueError("BC noise and time require independent RNG streams")
        y = targets.detach().float().repeat(8, 1, 1)
        noise = torch.randn(y.shape, device=y.device, dtype=y.dtype, generator=noise_generator)
        uniform = torch.rand((len(y),), device=y.device, generator=time_generator)
        times = (.999 - uniform.pow(1./1.5)) / .999
        loss = self.action_head(condition.detach().repeat(8, 1, 1), y, noise=noise, times=times)
        count = float(global_scene_count)
        if count < len(condition) or world_size < 1:
            raise ValueError("Invalid global BC scene denominator")
        return loss * (len(condition) * world_size / count)


@dataclass(frozen=True)
class PolicyIdentity:
    version: int
    optimizer_step: int
    state_hash: str
    base_binding_hash: str
    role: str

    def __post_init__(self):
        if type(self.version) is not int or self.version < 0 or type(self.optimizer_step) is not int or self.optimizer_step < 0:
            raise ValueError("Policy version/step must be nonnegative integers")
        for value in (self.state_hash, self.base_binding_hash):
            if not isinstance(value, str) or len(value) != 64 or any(x not in "0123456789abcdef" for x in value):
                raise ValueError("Policy identity requires SHA256 content hashes")
        if self.role not in ("old", "reference"):
            raise ValueError("Snapshot is a frozen old or fixed-reference policy")


class FrozenPolicySnapshot(nn.Module):
    """Deep copy of action side only, never a second Qwen/VLM."""
    def __init__(self, current: DiTFlowPolicy, *, version, optimizer_step, base_binding_hash, role="old"):
        super().__init__()
        self.policy = deepcopy(current).requires_grad_(False).eval()
        self.identity = PolicyIdentity(version, optimizer_step, module_hash(self.policy), base_binding_hash, role)
        self.train(False)
        current_ptrs = {p.untyped_storage().data_ptr() for p in current.parameters()}
        if any(p.untyped_storage().data_ptr() in current_ptrs for p in self.parameters()):
            raise ValueError("Frozen policy snapshot aliases current trainable storage")

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        return self.policy(*args, **kwargs)

    def verify(self):
        if any(p.requires_grad for p in self.parameters()) or any(m.training for m in self.modules()):
            raise ValueError("Frozen old/reference policy mode or gradients changed")
        if module_hash(self.policy) != self.identity.state_hash:
            raise ValueError("Frozen old/reference policy state changed")

    def get_extra_state(self):
        return dict(self.identity.__dict__)

    def set_extra_state(self, state):
        self.identity = PolicyIdentity(**state)
        self.requires_grad_(False).train(False)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if not strict:
            raise ValueError("Frozen old/reference snapshots require strict loading")
        result = super().load_state_dict(state_dict, strict=True, assign=assign)
        self.requires_grad_(False).train(False)
        self.verify()
        return result
