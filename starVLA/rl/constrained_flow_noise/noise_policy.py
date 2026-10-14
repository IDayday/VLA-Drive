"""One original DiT pass for velocity and RLinf noise; all ten steps are saved."""
import copy
import math
from dataclasses import dataclass

import torch
from torch import nn

from starVLA.rl.flow_grpo.math import transition as legacy_transition
from .contracts import VARIANTS, fingerprint, seed_for
from .transition import flow_noise_transition, density_views
from .vendor.rlinf_explore_noise_net import ExploreNoiseNet


@dataclass(frozen=True)
class KernelSpec:
    variant: str
    min_std: float
    max_std: float
    group_size: int = 16
    steps: int = 10
    noise_level: float = .1
    correlation: float = .8
    rollout_chunk: int = 1

    def __post_init__(self):
        if self.variant not in VARIANTS or (self.group_size, self.steps) != (16, 10):
            raise ValueError('Only the registered G16/K10 kernels are allowed')
        if not (math.isfinite(self.min_std) and math.isfinite(self.max_std) and 0 < self.min_std < self.max_std):
            raise ValueError('noise_logvar_range must contain min/max STANDARD DEVIATIONS')
        if self.noise_level != .1 or self.correlation != .8 or self.rollout_chunk != 1:
            raise ValueError('Preserve the qualified legacy noise schedule, correlation and rollout chunk')


class NoisePolicy(nn.Module):
    def __init__(self, policy, spec):
        super().__init__()
        self.policy, self.spec = policy, spec
        head = policy.action_model
        self.hidden_width = head.action_decoder.layer1.in_features
        if self.hidden_width != head.hidden_size or (head.action_horizon, head.action_dim, head.num_inference_timesteps) != (8, 4, 10):
            raise ValueError('Unexpected original action head configuration')
        self.noise_head = (ExploreNoiseNet(self.hidden_width, 4, [128, 64], 'tanh',
                           [spec.min_std, spec.max_std], 'learn') if spec.variant == 'N_FLOW_NOISE' else None)
        self.eval()

    def forward(self, **kwargs):
        from .objective import actor_loss
        return actor_loss(self, **kwargs)

    def condition(self, raw):
        if raw.ndim != 3 or raw.shape[1] != 8:
            raise ValueError('Expected pre-projection observation tokens')
        with torch.autocast(raw.device.type, dtype=torch.float32, enabled=raw.is_cuda):
            return self.policy.action_model._project_condition(raw.detach())

    def noise_std(self, hidden):
        if self.noise_head is None or hidden.shape[-2:] != (8, self.hidden_width):
            raise ValueError('Expected original pre-decoder action hidden states')
        # Uniform BF16 storage is compatible with the audited ZeRO-2 engine.
        # A differentiable FP32 functional view preserves the exact upstream
        # MLP/tanh/logvar/exp arithmetic and gradients to both weights and hidden.
        params = {name: value.float() for name, value in self.noise_head.named_parameters()}
        device = hidden.device
        buffers = getattr(self, '_fixed_logvar_fp32', None)
        if buffers is None or buffers['logvar_min'].device != device:
            # Fixed configuration constants only; no state-dependent output is
            # cached. Construct before graph capture, preserving FP32 bounds.
            buffers = {'logvar_min': torch.log(torch.tensor([self.spec.min_std**2], device=device, dtype=torch.float32)),
                       'logvar_max': torch.log(torch.tensor([self.spec.max_std**2], device=device, dtype=torch.float32))}
            self._fixed_logvar_fp32 = buffers
        with torch.autocast(device.type, enabled=False):
            return torch.func.functional_call(self.noise_head, (params, buffers), (hidden.float(),))

    def distribution(self, x, buckets, condition, steps):
        if self.training or self.policy.action_model.training or (self.noise_head is not None and self.noise_head.training):
            raise ValueError('PG rollout/recompute require deterministic eval mode with autograd allowed')
        head = self.policy.action_model
        sigma = None
        if getattr(self, '_cfn_rollout_graph_enabled', False) and not torch.is_grad_enabled():
            from .rollout_graph import outputs
            velocity, sigma = outputs(self, x, buckets, condition)
        else:
            with torch.autocast(x.device.type, dtype=torch.float32, enabled=x.is_cuda):
                if self.noise_head is not None:
                    velocity, hidden = head.predict_velocity_with_hidden(x, buckets, condition)
                    sigma = self.noise_std(hidden)
                else:
                    velocity = head.predict_velocity(x, buckets, condition)
        if self.noise_head is not None:
            return flow_noise_transition(x, velocity, sigma)
        times = torch.arange(11, device=x.device, dtype=torch.float32)/10
        return legacy_transition(x, velocity, times[steps], times[steps+1]-times[steps],
                                 first_dt=times[1], noise_level=self.spec.noise_level,
                                 temporal_correlation=self.spec.correlation, transition_mode='flow_sde')

    def saved_statistics(self, raw, chain, *, chunk=None):
        """Flat B/G/K recomputation of fixed states; never resample x_next."""
        if chain.ndim != 5 or chain.shape[1:] != (16, 11, 8, 4) or chain.requires_grad:
            raise ValueError('Expected detached real G16/K10 chain')
        b, g, kp, h, d = chain.shape; k = kp-1
        x = chain[:, :, :-1].detach().reshape(b*g*k, h, d)
        xn = chain[:, :, 1:].detach().reshape_as(x)
        condition = self.condition(raw)[:, None, None].expand(b, g, k, 8, -1).reshape(b*g*k, 8, -1)
        step = torch.arange(k, device=x.device)[None, None].expand(b, g, k).reshape(-1)
        bucket_values = torch.tensor([int(i/k*self.policy.action_model.num_timestep_buckets) for i in range(k)], device=x.device)
        buckets = bucket_values[step]
        chunk = chunk or len(x)
        mean, std, logs = [], [], []
        for start in range(0, len(x), chunk):
            sl = slice(start, start+chunk)
            distribution = self.distribution(x[sl], buckets[sl], condition[sl], step[sl])
            mean.append(distribution.mean); std.append(distribution.std.expand_as(x[sl])); logs.append(distribution.logprob(xn[sl]))
        logp = torch.cat(logs).reshape(b, g, k, h, d)
        return {'mean': torch.cat(mean).reshape(b, g, k, h, d),
                'std': torch.cat(std).reshape(b, g, k, h, d), **density_views(logp)}

    @torch.no_grad()
    def sample_chain(self, raw, tokens, raw_positions, *, version, seed=42):
        if len(tokens) != len(raw) or len(tokens) != len(raw_positions):
            raise ValueError('Scene/position mismatch')
        device = raw.device; b = len(raw); g = self.spec.group_size
        condition = self.condition(raw)
        root_keys = [seed_for('cfn_root', seed, t, int(p)) for t, p in zip(tokens, raw_positions)]
        innovation_keys = [seed_for('cfn_innovation', seed, t, int(p)) for t, p in zip(tokens, raw_positions)]
        generators = [torch.Generator(device=device).manual_seed(key) for key in innovation_keys]
        x = torch.stack([torch.randn((g, 8, 4), device=device, dtype=torch.float32,
                        generator=torch.Generator(device=device).manual_seed(key)) for key in root_keys])
        chain, means, stds, logps = [x], [], [], []
        for k in range(10):
            mu, sigma = [], []
            for candidate in range(g):
                buckets = torch.full((b,), int(k/10*self.policy.action_model.num_timestep_buckets), device=device, dtype=torch.long)
                steps = torch.full((b,), k, device=device, dtype=torch.long)
                try:
                    distribution = self.distribution(x[:, candidate], buckets, condition, steps)
                except FloatingPointError as error:
                    error.chain_evidence = dict(states=[v.detach().cpu() for v in chain],
                        completed_mean=[v.detach().cpu() for v in means],completed_std=[v.detach().cpu() for v in stds],
                        root_keys=root_keys,innovation_keys=innovation_keys,generation_step=k,candidate=candidate,
                        gaussian=getattr(error,'gaussian_evidence',None))
                    raise
                mu.append(distribution.mean); sigma.append(distribution.std.expand_as(distribution.mean))
            mean, std = torch.stack(mu, 1), torch.stack(sigma, 1)
            noise = torch.stack([torch.randn((g, 8, 4), device=device, generator=gen) for gen in generators])
            if self.spec.variant == 'C_SDE':
                from starVLA.rl.flow_grpo.math import Transition
                distribution = Transition(mean, std, std/.1**.5, self.spec.correlation)
            else:
                from .transition import DiagonalTransition
                distribution = DiagonalTransition(mean, std)
            x = distribution.sample(noise)
            means.append(mean); stds.append(std); logps.append(distribution.logprob(x)); chain.append(x)
        result = {'raw': raw.detach(), 'chain': torch.stack(chain, 2).detach(),
                  'old_mean': torch.stack(means, 2).detach(), 'old_std': torch.stack(stds, 2).detach(),
                  'old_elementwise_logp': torch.stack(logps, 2).detach(),
                  'tokens': list(tokens), 'raw_positions': list(map(int, raw_positions)),
                  'version': copy.deepcopy(version), 'kernel': vars(self.spec),
                  'root_keys': root_keys, 'innovation_keys': innovation_keys}
        if not all(torch.isfinite(result[name]).all() for name in ('chain', 'old_mean', 'old_std', 'old_elementwise_logp')):
            error=FloatingPointError('Nonfinite rollout; save failure and abort collection')
            error.chain_evidence={key:(v.detach().cpu() if isinstance(v,torch.Tensor) else v) for key,v in result.items()}
            raise error
        return result


def legacy_std_range():
    values = []
    for step in range(10):
        times = torch.arange(11, dtype=torch.float64)/10
        d = legacy_transition(torch.zeros(1, 8, 4, dtype=torch.float64), torch.zeros(1, 8, 4, dtype=torch.float64),
                              times[step], times[step+1]-times[step], first_dt=times[1], noise_level=.1, temporal_correlation=.8)
        values.append(float(d.std.reshape(-1)[0]))
    return min(v for v in values if v > 0), max(values), values
