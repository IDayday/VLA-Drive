"""CPU packing/math checks for the real probe; never GPU acceptance evidence."""
from types import SimpleNamespace
import pytest
import torch
from tests.flow_grpo.test_temporal_exploration import TinyPolicy
from starVLA.rl.flow_grpo.rollout import SamplingSpec, sample_chain, evaluate_transitions
from scripts.analysis.action_batch_probe import flat_transitions


class TimeScenePolicy(TinyPolicy):
    def encode_policy_condition(self, observation):
        inputs = torch.arange(1, len(observation.tokens)+1, dtype=torch.float64).reshape(-1, 1)
        return self.qwen_vl_interface(inputs).reshape(-1, 1, 1)


@pytest.mark.parametrize('batch', [1, 2])
@pytest.mark.parametrize('mode', ['flow_sde', 'euler_gaussian'])
@pytest.mark.parametrize('checkpoint', [False, True])
def test_saved_transition_batch_preserves_time_scene_order_and_gradient(batch, mode, checkpoint):
    torch.manual_seed(193)
    policy = TimeScenePolicy()
    native = policy.action_model.predict_velocity
    # Distinct scene conditions and time dependence expose axis/bucket mixups.
    policy.action_model.predict_velocity = lambda x, t, c: native(x, t, c) + t[:, None, None]*.0002
    obs = SimpleNamespace(tokens=tuple(f'scene{i}' for i in range(batch)))
    chain = sample_chain(policy, obs, SamplingSpec(group_size=16, num_steps=10,
                         temporal_noise_correlation=.8, transition_mode=mode), 0, 42, {})
    before = chain.chain.clone()
    original = evaluate_transitions(policy, obs, chain, checkpoint)
    a = torch.autograd.grad(original['elementwise_logprob'].mean(), tuple(policy.parameters()))
    flat = flat_transitions(policy, chain, checkpoint)
    b = torch.autograd.grad(flat['elementwise_logprob'].mean(), tuple(policy.parameters()))
    for name in original: torch.testing.assert_close(flat[name], original[name], rtol=0, atol=0)
    for left, right in zip(a, b): torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-7)
    assert torch.equal(before, chain.chain)
    assert not chain.chain.requires_grad
