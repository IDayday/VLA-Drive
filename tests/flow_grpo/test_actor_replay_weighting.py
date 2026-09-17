"""Exercise the production loss wrapper's scene weights and deterministic replay RNG."""
from types import SimpleNamespace
import torch
from torch import nn
from starVLA.rl.flow_grpo.model import FlowGRPOActor


class ReplayPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.7, dtype=torch.float64))
        self.calls = []

    def compute_sft_losses(self, examples):
        assert len(examples) == 1
        sample = examples[0]
        self.calls.append(sample["_flow_sample_seed"])
        return {"action_loss": (self.weight - torch.rand((), dtype=torch.float64)) ** 2}


def run(monkeypatch, g, k, samples):
    policy = ReplayPolicy()
    shape = (1, g, k, 2, 2)
    monkeypatch.setattr(
        "starVLA.rl.flow_grpo.model.evaluate_transitions",
        lambda *args, **kwargs: {
            "elementwise_logprob": policy.weight.expand(shape),
            "mean": policy.weight.expand(shape),
            "std": torch.ones(shape, dtype=torch.float64),
        },
    )
    cfg = {
        "runtime": {
            "activation_checkpointing": True,
            "noise_seed_schedule": "global_scene_v1",
        },
        "algorithm": {"ppo_clip_range": 0.02, "reference_kl_coefficient": 0.01},
        "retention": {"original_sft_coefficient": 0.1},
    }
    rollout = SimpleNamespace(
        observation=None,
        old_logprob=torch.zeros((1, g, k), dtype=torch.float64),
        dimension_mask=None,
        spec=SimpleNamespace(reduction="flow_grpo_dimension_mean"),
        advantages=torch.zeros((1, g), dtype=torch.float64),
        reference_mean=torch.zeros(shape, dtype=torch.float64),
        reference_std=torch.ones(shape, dtype=torch.float64),
        transition_mask=torch.ones((1, g, k), dtype=torch.bool),
    )
    actor = FlowGRPOActor(policy, cfg)
    before = torch.get_rng_state()
    result = actor(mode="update", rollout=rollout, replay=samples)
    assert torch.equal(before, torch.get_rng_state())
    result["loss"].backward()
    return result["loss"].detach(), policy.weight.grad, policy.calls


def test_real_wrapper_replay_once_and_group_step_invariance(monkeypatch):
    samples = [{"_flow_sample_seed": 42}, {"_flow_sample_seed": 1051}]
    small = run(monkeypatch, 2, 3, samples)
    large = run(monkeypatch, 8, 10, samples)
    assert small[2] == large[2] == [42, 1051]
    torch.testing.assert_close(small[0], large[0], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(small[1], large[1], rtol=1e-12, atol=1e-12)
    singles = [run(monkeypatch, 8, 10, [sample]) for sample in samples]
    torch.testing.assert_close(
        large[0], sum(x[0] for x in singles) / 2, rtol=1e-12, atol=1e-12
    )
    torch.testing.assert_close(
        large[1], sum(x[1] for x in singles) / 2, rtol=1e-12, atol=1e-12
    )
