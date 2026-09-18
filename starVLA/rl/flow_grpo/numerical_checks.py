"""Real network outputs against independent high precision probability math."""
import torch
from .math import transition, reduce_dimensions
from .rollout import velocity


def precision_gate(policy, observation, rollout):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        condition = policy.encode_policy_condition(observation)
    maxima = dict(mean=0.0, std=0.0, logprob=0.0, ratio=0.0, velocity_gradient=0.0)
    for step in range(rollout.spec.num_steps):
        x = rollout.chain[:, 0, step]
        next_x = rollout.chain[:, 0, step + 1]
        bucket = torch.full(
            (x.shape[0],),
            int(
                step / rollout.spec.num_steps * policy.action_model.num_timestep_buckets
            ),
            dtype=torch.long,
            device=x.device,
        )
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = velocity(policy, x, bucket, condition)
        v32 = prediction.detach().requires_grad_(True)
        v64 = prediction.detach().double().requires_grad_(True)
        time = rollout.times[step]
        dt = rollout.times[step + 1] - time
        d32 = transition(
            x,
            v32,
            time,
            dt,
            noise_level=rollout.spec.noise_level,
            first_dt=rollout.times[1],
            temporal_correlation=getattr(rollout.spec, "temporal_noise_correlation", 0.0),
        )
        d64 = transition(
            x.double(),
            v64,
            time.double(),
            dt.double(),
            noise_level=rollout.spec.noise_level,
            first_dt=rollout.times[1].double(),
            temporal_correlation=getattr(rollout.spec, "temporal_noise_correlation", 0.0),
        )
        # Independent torch.distributions oracle; no low-precision log/exp.
        p32 = d32.logprob(next_x)
        p64 = torch.distributions.Normal(d64.mean, d64.std).log_prob(next_x.double())
        rho = getattr(rollout.spec, "temporal_noise_correlation", 0.0)
        if rho:
            # Independent AR(1) conditional-Normal oracle; no production
            # covariance construction, whitening or density function reused.
            residual = next_x.double() - d64.mean
            conditional_mean = torch.cat((d64.mean[..., :1, :],
                d64.mean[..., 1:, :] + rho * residual[..., :-1, :]), dim=-2)
            conditional_std = torch.ones_like(d64.mean) * d64.std
            conditional_std[..., 1:, :] *= (1 - rho**2)**.5
            p64 = torch.distributions.Normal(conditional_mean, conditional_std).log_prob(next_x.double())
        old = reduce_dimensions(rollout.old_elementwise_logprob[:, 0, step])
        r32 = (reduce_dimensions(p32) - old).exp()
        r64 = (reduce_dimensions(p64) - old.double()).exp()
        (g32,) = torch.autograd.grad(r32.mean(), v32)
        (g64,) = torch.autograd.grad(r64.mean(), v64)
        for name, a, b in [
            ("mean", d32.mean, d64.mean),
            ("std", d32.std, d64.std),
            ("logprob", p32, p64),
            ("ratio", r32, r64),
            ("velocity_gradient", g32, g64),
        ]:
            maxima[name] = max(maxima[name], float((a.double() - b).abs().max()))
            torch.testing.assert_close(
                a.double(),
                b,
                atol=2e-5,
                rtol=1e-4,
                msg=lambda msg: f"{name} step {step}: {msg}",
            )
    return dict(
        network="real checkpoint BF16 Qwen/DiT weights; original action-head autocast convention",
        math="production FP32 vs independent FP64 torch Normal/AR(1) conditionals; ratio gradient w.r.t. velocity",
        atol=2e-5,
        rtol=1e-4,
        maximum_absolute_difference=maxima,
    )
