from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from starVLA.model.modules.action_model.multi_trajectory.config import (
    MultiTrajectoryConfig,
)
from starVLA.model.modules.action_model.multi_trajectory.ddp_multi_sampler import (
    DDPMultiSampler,
)


class RandomActionHead(nn.Module):
    action_horizon = 8
    action_dim = 3

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.calls = 0

    def predict_action(self, vl_embs_list, state=None):
        self.calls += 1
        batch_size = vl_embs_list[0].shape[0]
        return torch.randn(batch_size, 8, 3) * self.scale


class OrderedActionHead(RandomActionHead):
    def predict_action(self, vl_embs_list, state=None):
        self.calls += 1
        marker = vl_embs_list[0][:, 0, 0]
        return marker[:, None, None].expand(-1, 8, 3).clone()


def hidden_states(batch_size=2):
    return [torch.randn(batch_size, 5, 7), torch.randn(batch_size, 5, 7)]


def test_ddp_multi_sampler_k1_equivalence():
    action_head = RandomActionHead()
    hidden = hidden_states()
    state = torch.randn(2, 1, 3)

    torch.manual_seed(1234)
    expected = action_head.predict_action(hidden, state)
    actual = DDPMultiSampler(action_head, num_candidates=1).sample(
        hidden, state, seed=1234
    )

    torch.testing.assert_close(actual[:, 0], expected)


def test_ddp_multi_sampler_shape():
    output = DDPMultiSampler(RandomActionHead(), num_candidates=64).sample(
        hidden_states(batch_size=2), torch.randn(2, 1, 3), seed=7
    )
    assert output.shape == (2, 64, 8, 3)


def test_ddp_multi_sampler_independent_flow_noise():
    output = DDPMultiSampler(RandomActionHead(), num_candidates=64).sample(
        hidden_states(batch_size=1), torch.zeros(1, 1, 3), seed=20260821
    )
    flattened = output[0].reshape(64, -1)
    assert torch.unique(flattened, dim=0).shape[0] == 64


def test_ddp_multi_sampler_rng_isolation():
    sampler = DDPMultiSampler(RandomActionHead(), num_candidates=4)
    torch.manual_seed(99)
    hidden = hidden_states()
    state = torch.randn(2, 1, 3)
    before = torch.random.get_rng_state().clone()
    sampler.sample(hidden, state, seed=123)
    after = torch.random.get_rng_state()
    assert torch.equal(after, before)


def test_ddp_multi_sampler_does_not_recompute_qwen():
    action_head = RandomActionHead()
    hidden = hidden_states()
    snapshots = [value.clone() for value in hidden]
    DDPMultiSampler(action_head, num_candidates=8).sample(hidden, seed=2)
    assert action_head.calls == 1
    for value, snapshot in zip(hidden, snapshots):
        torch.testing.assert_close(value, snapshot)


def test_ddp_multi_sampler_candidate_order_stable():
    action_head = OrderedActionHead()
    hidden = [torch.zeros(2, 1, 1)]
    hidden[0][:, 0, 0] = torch.tensor([10.0, 20.0])
    output = DDPMultiSampler(action_head, num_candidates=3).sample(hidden)
    torch.testing.assert_close(output[:, :, 0, 0], torch.tensor([[10.0] * 3, [20.0] * 3]))


def test_ddp_multi_sampler_frozen_generator_has_no_grad():
    action_head = RandomActionHead()
    hidden = [torch.randn(2, 3, 4, requires_grad=True)]
    output = DDPMultiSampler(action_head, num_candidates=2).sample(hidden, seed=1)
    assert not output.requires_grad
    assert action_head.scale.grad is None
    assert hidden[0].grad is None


class _DummyVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.model = nn.Linear(1, 1)
        self.model.config = SimpleNamespace(hidden_size=4)
        self.forward_calls = 0
        self.last_images = None

    def build_qwenvl_inputs(self, images, instructions):
        self.last_images = images
        return {
            "marker": torch.tensor(float(len(images))),
            "attention_mask": torch.ones(len(images), 23, dtype=torch.long),
        }

    def forward(self, marker, attention_mask, **kwargs):
        self.forward_calls += 1
        batch_size = int(marker.item())
        hidden = torch.zeros(batch_size, attention_mask.shape[1], 4)
        return SimpleNamespace(hidden_states=(hidden,))


class _DummyActionHead(RandomActionHead):
    def __init__(self):
        super().__init__()
        holder = nn.Module()
        holder.transformer_blocks = nn.ModuleList([nn.Identity()])
        self.model = holder
        self.config = SimpleNamespace(state_dim=3)


def _disabled_qwen_config():
    return OmegaConf.create(
        {
            "framework": {
                "action_model": {
                    "diffusion_model_cfg": {"num_layers": 1},
                    "hidden_size": 4,
                    "future_action_window_size": 8,
                    "past_action_window_size": 0,
                }
            },
            "datasets": {"vla_data": {}},
            "trainer": {},
            "multi_trajectory": {"enabled": False},
        }
    )


def test_multi_trajectory_disabled_equivalence(monkeypatch):
    # Importing the real framework verifies the actual opt-in integration while
    # replacing only heavyweight model factories with deterministic test doubles.
    import starVLA.model.framework.QwenPI as qwenpi
    from starVLA.model.tools import FRAMEWORK_REGISTRY

    action_head = _DummyActionHead()
    vlm = _DummyVLM()
    monkeypatch.setattr(qwenpi, "get_vlm_model", lambda config: vlm)
    monkeypatch.setattr(qwenpi, "get_action_model", lambda config: action_head)
    model = qwenpi.Qwen_PI(_disabled_qwen_config())

    assert FRAMEWORK_REGISTRY["QwenFM"] is qwenpi.Qwen_PI
    assert MultiTrajectoryConfig.from_full_config(model.config).enabled is False
    assert not hasattr(model, "multi_trajectory_planner")
    assert not any(key.startswith("multi_trajectory_planner.") for key in model.state_dict())

    torch.manual_seed(314)
    expected = action_head.predict_action([torch.zeros(2, 2, 4)], torch.zeros(2, 1, 3))
    torch.manual_seed(314)
    output = model.predict_action(
        batch_images=[[object()], [object()]],
        instructions=["left", "right"],
        state=np.zeros((2, 1, 3), dtype=np.float32),
    )["normalized_actions"]
    np.testing.assert_allclose(output, expected.detach().numpy(), rtol=0, atol=0)
    assert vlm.forward_calls == 1

    examples = [
        {
            "image": [object()],
            "lang": "left",
            "state": np.zeros((1, 3), dtype=np.float32),
        },
        {
            "image": [object()],
            "lang": "right",
            "state": np.zeros((1, 3), dtype=np.float32),
        },
    ]
    torch.manual_seed(2718)
    expected_examples = action_head.predict_action(
        [torch.zeros(2, 2, 4)], torch.zeros(2, 1, 3)
    )
    torch.manual_seed(2718)
    examples_output = model.predict_action(examples=examples)["normalized_actions"]
    np.testing.assert_allclose(
        examples_output, expected_examples.detach().numpy(), rtol=0, atol=0
    )


def test_qwen_forward_called_once_and_qformer_uses_full_sequence(monkeypatch):
    import starVLA.model.framework.QwenPI as qwenpi
    import starVLA.model.modules.action_model.multi_trajectory.planner as planner_module

    class _RecordingPlanner(nn.Module):
        def __init__(self, action_head, config, qwen_hidden_dim):
            super().__init__()
            self.config = config
            self.qwen_hidden_dim = qwen_hidden_dim
            self.qformer_calls = 0
            self.full_sequence_length = None

        def forward(
            self,
            vl_embs_list,
            state,
            full_hidden_state,
            attention_mask,
        ):
            self.qformer_calls += 1
            self.full_sequence_length = full_hidden_state.shape[1]
            assert attention_mask.shape == full_hidden_state.shape[:2]
            return vl_embs_list[0].new_zeros(
                full_hidden_state.shape[0], 8, 3, dtype=torch.bfloat16
            )

    action_head = _DummyActionHead()
    vlm = _DummyVLM()
    monkeypatch.setattr(qwenpi, "get_vlm_model", lambda config: vlm)
    monkeypatch.setattr(qwenpi, "get_action_model", lambda config: action_head)
    monkeypatch.setattr(
        planner_module, "DDPDrivoRSuprimPlanner", _RecordingPlanner
    )
    config = _disabled_qwen_config()
    config.multi_trajectory = {
        "enabled": True,
        "strict_inference": False,
    }
    model = qwenpi.Qwen_PI(config)
    examples = []
    for _ in range(2):
        examples.append(
            {
                "image": [object(), object(), object()],
                "lang": "keep straight",
                "state": np.zeros((1, 3), dtype=np.float32),
            }
        )

    output = model.predict_action(examples=examples)["normalized_actions"]

    assert output.shape == (2, 8, 3)
    assert output.dtype == np.float32
    assert all(len(sample) == 3 for sample in vlm.last_images)
    assert vlm.forward_calls == 1
    assert model.multi_trajectory_planner.qformer_calls == 1
    assert model.multi_trajectory_planner.full_sequence_length == 23
    assert model.multi_trajectory_planner.qwen_hidden_dim == 4
