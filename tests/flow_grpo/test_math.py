import importlib.util
from pathlib import Path
import math
import pytest
import torch
from starVLA.rl.flow_grpo.math import (
    transition,
    gaussian_logprob,
    reduce_dimensions,
    conditional_kl,
    group_advantages,
    clipped_surrogate,
)


def test_locked_upstream_sampler():
    spec = importlib.util.spec_from_file_location(
        "locked_sampler", Path(__file__).parent / "vendor/sd3_sde_with_logprob.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Scheduler:
        sigmas = torch.arange(10, -1, -1).float() / 10

        def index_for_timestep(self, t):
            return int(t)

    scheduler = Scheduler()
    torch.manual_seed(7)
    x = torch.randn(3, 8, 4)
    v = torch.randn_like(x)
    xn = torch.randn_like(x)
    for step in range(10):
        upstream = module.sde_step_with_logprob(
            scheduler,
            -v,
            torch.full((3,), step),
            x,
            noise_level=0.1,
            prev_sample=xn,
            return_sqrt_dt=True,
        )
        # Use the very same continuous sigma schedule mapped to t.
        t = 1 - scheduler.sigmas[step]
        dt = scheduler.sigmas[step] - scheduler.sigmas[step + 1]
        ours = transition(
            x, v, t, dt, noise_level=0.1, first_dt=1 - scheduler.sigmas[1]
        )
        torch.testing.assert_close(ours.mean, upstream[2], rtol=2e-6, atol=5e-7)
        torch.testing.assert_close(
            ours.std.expand_as(upstream[3]),
            upstream[3] * upstream[4],
            rtol=1e-6,
            atol=1e-7,
        )
        torch.testing.assert_close(
            reduce_dimensions(ours.logprob(xn)), upstream[1], rtol=3e-6, atol=3e-4
        )
        generator = torch.Generator().manual_seed(123)
        up_sample = module.sde_step_with_logprob(
            scheduler,
            -v,
            torch.full((3,), step),
            x,
            noise_level=0.1,
            generator=generator,
        )[0]
        noise = torch.randn(x.shape, generator=torch.Generator().manual_seed(123))
        torch.testing.assert_close(
            ours.mean + ours.std * noise, up_sample, rtol=2e-6, atol=6e-7
        )


def test_normal_logprob_and_reductions_fp64():
    torch.manual_seed(0)
    x, mean = torch.randn(2, 3, 8, 4, dtype=torch.float64), torch.randn(
        2, 3, 8, 4, dtype=torch.float64
    )
    std = torch.rand_like(mean) + 0.01
    actual = gaussian_logprob(x, mean, std)
    expected = torch.distributions.Normal(mean, std).log_prob(x)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(reduce_dimensions(actual), expected.mean((-1, -2)))
    torch.testing.assert_close(
        reduce_dimensions(actual, reduction="joint_sum"), expected.sum((-1, -2))
    )
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[..., 0] = False
    torch.testing.assert_close(
        reduce_dimensions(actual, mask), expected[..., 1:].mean((-1, -2))
    )


def test_kl_matches_torch_and_zero():
    torch.manual_seed(1)
    a, b = torch.randn(2, 8, 4, dtype=torch.float64), torch.randn(
        2, 8, 4, dtype=torch.float64
    )
    s, t = torch.rand_like(a) + 0.1, torch.rand_like(a) + 0.1
    expected = torch.distributions.kl_divergence(
        torch.distributions.Normal(a, s), torch.distributions.Normal(b, t)
    )
    torch.testing.assert_close(
        conditional_kl(a, s, b, t), expected, rtol=1e-12, atol=1e-12
    )
    assert conditional_kl(a, s, a, s).abs().max() == 0


@pytest.mark.parametrize("step", [0, 1, 9])
@pytest.mark.parametrize("noise", [1e-5, 0.1])
def test_boundaries_and_sqrt_dt(step, noise):
    x = torch.ones(2, 8, 4, dtype=torch.float64)
    v = x * 0.3
    d = transition(x, v, step / 10, 0.1, noise_level=noise, first_dt=0.1)
    torch.testing.assert_close(
        d.std, d.diffusion * math.sqrt(0.1), rtol=1e-12, atol=1e-12
    )
    assert torch.isfinite(d.logprob(d.mean)).all() and (d.std > 0).all()


def test_ode_has_no_fake_probability():
    x = torch.ones(2, 8, 4)
    v = x * 0.3
    d = transition(x, v, 0, 0.1, noise_level=0, first_dt=0.1)
    torch.testing.assert_close(d.mean, x + 0.1 * v)
    with pytest.raises(ValueError):
        d.logprob(d.mean)
    with pytest.raises(ValueError):
        conditional_kl(d.mean, d.std, d.mean, d.std)


def test_empirical_gaussian_statistics():
    generator = torch.Generator().manual_seed(0)
    x = torch.tensor([[[0.4]]], dtype=torch.float64)
    d = transition(x, x * 0.8, 0.4, 0.1, noise_level=0.1, first_dt=0.1)
    samples = d.mean + d.std * torch.randn(
        200000, 1, 1, dtype=torch.float64, generator=generator
    )
    assert abs(float(samples.mean() - d.mean)) < 5 * float(d.std) / math.sqrt(
        samples.numel()
    )
    torch.testing.assert_close(
        samples.var(unbiased=False), d.std.squeeze().square(), rtol=0.015, atol=0
    )


def test_rollout_fixed_sample_grad_sign_and_equal_groups():
    means = torch.zeros(2, 1, 1, requires_grad=True)
    xn = torch.ones_like(means)
    std = torch.ones_like(means)
    old = reduce_dimensions(gaussian_logprob(xn, means.detach(), std))
    cur = reduce_dimensions(gaussian_logprob(xn, means, std))
    loss, ratio = clipped_surrogate(cur, old, torch.tensor([1.0, -1.0]))
    loss.sum().backward()
    assert means.grad[0] < 0 and means.grad[1] > 0
    assert torch.equal(ratio, torch.ones_like(ratio))
    rewards = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.5, 1.0]])
    adv = group_advantages(rewards)
    assert torch.equal(adv[:2], torch.zeros_like(adv[:2]))
    assert adv[2, 0] < 0 and adv[2, -1] > 0


def test_group_population_std_and_valid_zero():
    r = torch.tensor([[0.0, 1.0, 100.0], [0.0, 0.0, 0.0]])
    valid = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    a = group_advantages(r, valid)
    torch.testing.assert_close(a[0], torch.tensor([-1.0, 1.0, 0.0]), atol=3e-6, rtol=0)
    assert not a[1].any()


def test_mean_surrogate_is_not_joint_ratio():
    old = torch.zeros(1, 8, 4)
    cur = torch.ones_like(old) * 0.01
    mean = (reduce_dimensions(cur) - reduce_dimensions(old)).exp()
    joint = (
        reduce_dimensions(cur, reduction="joint_sum")
        - reduce_dimensions(old, reduction="joint_sum")
    ).exp()
    assert not torch.allclose(mean, joint)


def test_nonfinite_and_invalid_schedules_fail():
    with pytest.raises(FloatingPointError):
        clipped_surrogate(torch.tensor([1000.0]), torch.zeros(1), torch.ones(1))
    with pytest.raises(FloatingPointError):
        group_advantages(torch.tensor([[0.0, float("nan")]]))
    with pytest.raises(ValueError):
        transition(torch.ones(1), torch.ones(1), 1, 0.1, noise_level=0.1, first_dt=0.1)


def test_fp32_probability_from_bf16():
    x = torch.ones(2, 8, 4, dtype=torch.bfloat16)
    d = transition(x, x * 0.5, 0.5, 0.1, noise_level=0.1, first_dt=0.1)
    assert d.mean.dtype == torch.float32
    assert d.logprob(d.mean).dtype == torch.float32
