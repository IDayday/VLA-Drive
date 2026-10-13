"""Thin adapters around the unchanged DDP action model and transition kernel."""
import numpy as np
import torch

from starVLA.rl.flow_grpo.math import transition, Transition
from starVLA.rl.flow_grpo.rollout import velocity


def encode_condition(policy, normalized_action_token_states):
    """Inputs stop BEFORE the trainable qwen_proj. No reward/map/GT inputs."""
    raw = normalized_action_token_states
    if raw.ndim != 3 or raw.shape[1] != 8:
        raise ValueError('Expected eight pre-projection normalized action token states')
    with torch.autocast(raw.device.type, dtype=torch.float32, enabled=raw.is_cuda):
        return policy.action_model._project_condition(raw.detach())


@torch.no_grad()
def sample_ode(policy, condition, initial_noise):
    if policy.action_model.num_inference_timesteps != 10:
        raise ValueError('Deployment must retain the original ten-step ODE')
    with torch.autocast(initial_noise.device.type, dtype=torch.float32, enabled=initial_noise.is_cuda):
        return policy.action_model._euler_sample(initial_noise, condition)


@torch.no_grad()
def sample_legacy_sde(policy, condition, initial_noise, *, innovation_generator, spec):
    """Same old transition, with explicit root samples and an independent RNG.

    The caller records both RNG namespaces. Candidate chunk remains one; this
    does not modify sample_chain, dimension-mean logprobs or the old kernel.
    """
    if initial_noise.ndim != 4 or initial_noise.shape[1] != spec.group_size:
        raise ValueError('Expected root noise [scene,candidate,horizon,dimension]')
    if spec.candidate_chunk_size != 1 or spec.num_steps != 10 or spec.noise_level <= 0:
        raise ValueError('Reuse the qualified legacy G16/K10/chunk1 SDE protocol')
    x = initial_noise.float().clone()
    b, g = x.shape[:2]
    times = torch.arange(spec.num_steps + 1, device=x.device, dtype=torch.float32) / spec.num_steps
    chain, logprobs = [x], []
    for step in range(spec.num_steps):
        means = []
        for candidate in range(g):
            bucket = torch.full((b,), int(step / spec.num_steps * policy.action_model.num_timestep_buckets),
                                device=x.device, dtype=torch.long)
            v = velocity(policy, x[:, candidate], bucket, condition, use_graph=False)
            dist = transition(x[:, candidate], v, times[step], times[step + 1] - times[step],
                              noise_level=spec.noise_level, first_dt=times[1],
                              temporal_correlation=spec.temporal_noise_correlation,
                              transition_mode=spec.transition_mode)
            means.append(dist.mean[:, None])
        dist = Transition(torch.cat(means, dim=1), dist.std, dist.diffusion, spec.temporal_noise_correlation)
        innovation = torch.randn(x.shape, device=x.device, dtype=torch.float32, generator=innovation_generator)
        x = dist.sample(innovation)
        logprobs.append(dist.logprob(x))
        chain.append(x)
    return {'actions': x, 'chain': torch.stack(chain, dim=2),
            'elementwise_logprob': torch.stack(logprobs, dim=2), 'times': times}


def original_fm_loss(policy, raw, normalized_target, noise, time):
    """The original forward, with fresh explicit FM randomness supplied by caller."""
    target = normalized_target.detach().float()
    with torch.autocast(raw.device.type, dtype=torch.float32, enabled=raw.is_cuda):
        return policy.action_model(raw.detach(), target,
                                   explicit_randomness={'noise': noise.detach(), 't': time.detach()})


def decode_request(actions, *, act_norm):
    from infer import deal_action_1225
    return deal_action_1225(np.asarray(actions), act_norm=act_norm)


def encode_external_request(poses, *, act_norm):
    """Inverse request encoding for smooth external proposals, never LQR states.

    Raw model candidate actions are retained as-is. Re-encoding an external
    heading uses unit sin/cos; inverse poses do not recover a model's arbitrary
    sin/cos amplitude, which the original decoder intentionally discards.
    """
    from infer import _X_MEAN, _X_STD, _Y_MEAN, _Y_STD
    poses = np.asarray(poses, dtype=np.float64)
    if poses.shape[-2:] != (8, 3) or not np.isfinite(poses).all():
        raise ValueError('Expected finite requested ego poses [...,8,3]')
    xy = poses[..., :2].copy()
    if act_norm:
        xy[..., 0] = (xy[..., 0] - _X_MEAN) / _X_STD
        xy[..., 1] = (xy[..., 1] - _Y_MEAN) / _Y_STD
    else:
        xy[..., 0] /= 4.5912
    return np.concatenate([xy, np.sin(poses[..., 2:3]), np.cos(poses[..., 2:3])], axis=-1).astype(np.float32)
