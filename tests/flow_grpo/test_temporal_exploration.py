"""Production correlated flow math against independent full-Gaussian oracles."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from starVLA.rl.flow_grpo.math import transition, gaussian_logprob, conditional_kl, reduce_dimensions
from starVLA.rl.flow_grpo.temporal_noise import correlate, whiten, validate_correlation
from starVLA.rl.flow_grpo.rollout import SamplingSpec, sample_chain, evaluate_transitions
from starVLA.rl.flow_grpo.acceptance import enforce_training_budget


def independent_covariance(h, rho, std):
    q = torch.tensor([[rho**abs(i-j) for j in range(h)] for i in range(h)], dtype=torch.float64)
    return q * std**2


@pytest.mark.parametrize("rho", [.0, .3, .8, .95])
def test_joint_logprob_kl_and_gradient_match_multivariate_normal(rho):
    gen = torch.Generator().manual_seed(204)
    x, mean, ref = [torch.randn(2, 3, 8, 4, generator=gen, dtype=torch.float64) for _ in range(3)]
    mean.requires_grad_(True)
    std, ref_std = torch.tensor(.13), torch.tensor(.21)
    std, ref_std = std.double(), ref_std.double()
    current = torch.distributions.MultivariateNormal(mean.transpose(-1, -2),
        covariance_matrix=independent_covariance(8, rho, std))
    reference = torch.distributions.MultivariateNormal(ref.transpose(-1, -2),
        covariance_matrix=independent_covariance(8, rho, ref_std))
    actual = gaussian_logprob(x, mean, std, rho).sum((-1, -2))
    expected = current.log_prob(x.transpose(-1, -2)).sum(-1)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-12)
    torch.testing.assert_close(conditional_kl(mean, std, ref, ref_std, rho).sum((-1, -2)),
        torch.distributions.kl_divergence(current, reference).sum(-1), atol=1e-10, rtol=1e-12)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), mean, retain_graph=True)[0],
        torch.autograd.grad(expected.sum(), mean)[0], atol=1e-10, rtol=1e-12)
    assert conditional_kl(mean, std, mean, std, rho).abs().max() == 0


def test_anisotropic_score_correction_preserves_gaussian_path_covariance_derivative():
    # Exact Gaussian path x_t=(1-t)*epsilon+t*data; continuous covariance equation.
    t, dt, target_variance, rho = .4, .01, .7, .8
    variance = (1-t)**2+t*t*target_variance
    slope = (t*target_variance-(1-t))/variance
    x = torch.eye(8, dtype=torch.float64)
    result = transition(x, slope*x, t, dt, noise_level=.1, first_dt=.1, temporal_correlation=rho)
    drift_matrix = (result.mean-x)/dt
    q = independent_covariance(8, rho, 1.)
    derivative = variance*(drift_matrix+drift_matrix.T)+result.diffusion.square()*q
    torch.testing.assert_close(derivative, 2*variance*slope*x, atol=1e-12, rtol=1e-12)


def test_sampling_covariance_and_sqrt_dt():
    gen = torch.Generator().manual_seed(203)
    x = torch.zeros(120000, 8, 1, dtype=torch.float64)
    result = transition(x, x, .3, .1, noise_level=.1, first_dt=.1, temporal_correlation=.8)
    samples = result.sample(torch.randn(x.shape, generator=gen, dtype=torch.float64)).squeeze(-1)
    expected = independent_covariance(8, .8, float(result.std.flatten()[0]))
    torch.testing.assert_close(torch.cov(samples.T), expected, atol=1e-5, rtol=.025)


def test_zero_correlation_is_bitwise_original_and_density_is_dimension_mean():
    x, v, eps = torch.randn(3, 8, 4), torch.randn(3, 8, 4), torch.randn(3, 8, 4)
    result = transition(x, v, .3, .1, noise_level=.1, first_dt=.1)
    zero = transition(x, v, .3, .1, noise_level=.1, first_dt=.1, temporal_correlation=0)
    assert torch.equal(result.mean, zero.mean)
    assert torch.equal(result.mean+result.std*eps, result.sample(eps))
    assert torch.equal(result.logprob(x), gaussian_logprob(x, result.mean, result.std))
    correlated = transition(x, v, .3, .1, noise_level=.1, first_dt=.1, temporal_correlation=.8)
    lp = correlated.logprob(correlated.sample(eps))
    torch.testing.assert_close(reduce_dimensions(lp), reduce_dimensions(lp, reduction="joint_sum")/32)


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.qwen_vl_interface = nn.Linear(1, 1, bias=False).double()
        self.action_model = TinyHead()

    def encode_policy_condition(self, observation):
        return self.qwen_vl_interface(torch.ones(1, 1, dtype=torch.float64)).reshape(1, 1, 1)


class TinyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.3, dtype=torch.float64))
        self.action_horizon, self.action_dim, self.num_timestep_buckets = 8, 4, 1000

    def predict_velocity(self, x, bucket, condition):
        return self.weight*x+condition.mean(-1, keepdim=True)


@pytest.mark.parametrize("mode", ["flow_sde", "euler_gaussian"])
def test_actual_rollout_recompute_saved_chain_chunking_and_vlm_gradient(mode):
    policy = TinyPolicy()
    spec = SamplingSpec(group_size=16, temporal_noise_correlation=.8, transition_mode=mode)
    observation = SimpleNamespace(tokens=("fixed",))
    rollout = sample_chain(policy, observation, spec, 0, 42, {})
    before = rollout.chain.clone()
    stats = evaluate_transitions(policy, observation, rollout)
    torch.testing.assert_close(reduce_dimensions(stats["elementwise_logprob"]), rollout.old_logprob, rtol=0, atol=1e-5)
    loss = stats["elementwise_logprob"].mean()
    gradients = torch.autograd.grad(loss, tuple(policy.parameters()))
    assert all(torch.isfinite(g).all() and g.abs().max() > 0 for g in gradients)
    rollout.spec = replace(spec, candidate_chunk_size=2)
    other = evaluate_transitions(policy, observation, rollout)
    torch.testing.assert_close(stats["elementwise_logprob"], other["elementwise_logprob"], rtol=0, atol=1e-5)
    assert torch.equal(before, rollout.chain)
    rollout.dimension_mask[..., 0, 0] = False
    with pytest.raises(ValueError, match="mask"):
        rollout.validate()


@pytest.mark.parametrize("rho", [-.1, 1., float('nan'), float('inf')])
def test_invalid_correlation_rejected(rho):
    with pytest.raises(ValueError, match="correlation"):
        validate_correlation(rho)


def test_correlation_cannot_enter_formal_training_or_mask_varying_std():
    cfg = {"runtime": {"run_mode": "formal", "max_updates": 8},
           "sampling": {"temporal_noise_correlation": .8}}
    with pytest.raises(ValueError, match="correlated exploration"):
        enforce_training_budget(cfg)
    mean = torch.randn(2, 8, 4)
    with pytest.raises(ValueError, match="waypoint-independent"):
        gaussian_logprob(mean, mean, torch.rand_like(mean)+.1, .8)


def test_noisy_euler_has_explicit_diffusion_and_joint_density_at_every_step():
    x, v, noise = [torch.randn(2, 8, 4, dtype=torch.float64) for _ in range(3)]
    for step in range(10):
        result = transition(x, v, step/10, .1, noise_level=.1, first_dt=.1,
                            temporal_correlation=.8, transition_mode="euler_gaussian")
        torch.testing.assert_close(result.mean, x+.1*v, atol=0, rtol=0)
        torch.testing.assert_close(result.std, torch.full_like(result.std, .1*.1**.5), atol=0, rtol=0)
        samples = result.sample(noise)
        oracle = torch.distributions.MultivariateNormal(result.mean.transpose(-1, -2),
                   covariance_matrix=independent_covariance(8, .8, .1*.1**.5))
        torch.testing.assert_close(result.logprob(samples).sum((-1, -2)),
                                   oracle.log_prob(samples.transpose(-1, -2)).sum(-1), atol=1e-10, rtol=1e-12)
    with pytest.raises(ValueError, match="experimental"):
        enforce_training_budget({"runtime": {"run_mode": "formal", "max_updates": 8},
                                 "sampling": {"transition_mode": "euler_gaussian"}})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_outer_bf16_autocast_does_not_round_covariance_math():
    values = torch.randn(2, 8, 4, device="cuda")
    expected = correlate(values, .8)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = correlate(values, .8)
        recovered = whiten(actual, .8)
    assert actual.dtype == recovered.dtype == torch.float32
    assert torch.equal(actual, expected)
    torch.testing.assert_close(recovered, values, atol=1e-6, rtol=1e-6)
