from types import SimpleNamespace

import torch
from torch import nn

from infer import configure_inference_seed
from starVLA.inference_noise import diffusion_initial_noise
from starVLA.model.modules.action_model.GR00T_ActionHeader import (
    FlowmatchingActionHead,
)


class _ZeroModel(nn.Module):
    def forward(self, hidden_states, **_kwargs):
        return torch.zeros_like(hidden_states)


class _PassthroughEncoder(nn.Module):
    def forward(self, actions, _timesteps):
        return actions


def _head():
    head = FlowmatchingActionHead.__new__(FlowmatchingActionHead)
    nn.Module.__init__(head)
    head.config = SimpleNamespace(
        action_horizon=8,
        action_dim=4,
        add_pos_embed=False,
        noise_s=0.999,
        noise_beta_alpha=1.5,
        noise_beta_beta=1.0,
    )
    head.num_inference_timesteps = 1
    head.num_timestep_buckets = 1000
    head.beta_dist = torch.distributions.Beta(1.5, 1.0)
    head.action_encoder = _PassthroughEncoder()
    head.action_decoder = nn.Identity()
    head.qwen_proj = nn.Identity()
    head.model = _ZeroModel()
    return head


def test_per_token_noise_is_world_size_independent():
    expected = diffusion_initial_noise(42, "scene-a", 0)
    for world_size in (1, 2, 16):
        tokens = [f"scene-{index}" for index in range(31)] + ["scene-a"]
        shards = [tokens[rank::world_size] for rank in range(world_size)]
        owner = next(rank for rank, shard in enumerate(shards) if "scene-a" in shard)
        assert "scene-a" in shards[owner]
        actual = diffusion_initial_noise(42, "scene-a", 0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_token_and_sample_index_change_noise():
    baseline = diffusion_initial_noise(42, "scene-a", 0)
    assert not torch.equal(baseline, diffusion_initial_noise(42, "scene-b", 0))
    assert not torch.equal(baseline, diffusion_initial_noise(42, "scene-a", 1))


def test_legacy_rank_stream_behavior_is_unchanged():
    assert configure_inference_seed(42, 7, "legacy_rank_stream") == 49
    assert configure_inference_seed(42, 7, "per_token") == 42


def test_real_zero_shuffled_receive_identical_initial_noise():
    values = {
        mode: diffusion_initial_noise(20260824, "token-x", 0)
        for mode in ("real", "zero", "shuffled")
    }
    assert torch.equal(values["real"], values["zero"])
    assert torch.equal(values["real"], values["shuffled"])


def test_predict_action_none_preserves_historical_rng_path():
    head = _head()
    queries = torch.zeros(2, 8, 4)
    torch.manual_seed(11)
    expected = torch.randn(2, 8, 4)
    torch.manual_seed(11)
    actual = head.predict_action(queries, initial_noise=None)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_predict_action_uses_supplied_noise_verbatim():
    head = _head()
    queries = torch.zeros(2, 8, 4)
    initial = torch.randn(2, 8, 4)
    actual = head.predict_action(queries, initial_noise=initial)
    torch.testing.assert_close(actual, initial, rtol=0, atol=0)


def test_flowmatching_default_forward_regression():
    head = _head()
    actions = torch.randn(3, 8, 4)
    queries = torch.zeros(3, 8, 4)
    torch.manual_seed(19)
    old_noise = torch.randn_like(actions)
    _ = head.sample_time(3, actions.device, actions.dtype)
    expected = ((actions - old_noise) ** 2).mean()
    torch.manual_seed(19)
    actual = head(queries, actions)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_reusable_flow_state_supports_per_sample_loss():
    head = _head()
    actions = torch.randn(3, 8, 4)
    state = head.sample_flow_state(actions)
    loss = head.loss_from_flow_state(
        torch.zeros(3, 8, 4), actions, state, reduction="none"
    )
    assert loss.shape == (3,)
    assert torch.isfinite(loss).all()
