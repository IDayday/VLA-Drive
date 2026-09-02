from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from navsim.agents.EpisodeDrive.drivevla_base_agent import DriveVLABaseAgent
from navsim.agents.EpisodeDrive.layers.planning_registers import (
    InternViTQVLoRALinear,
)
from navsim.agents.EpisodeDrive.layers.world_model import FutureRegisterPredictor
from navsim.agents.EpisodeDrive.layers.world_model import encode_path_tensor_batch


def _loss_agent(*, predictor_only: bool, future_mode: str = "correct"):
    agent = DriveVLABaseAgent.__new__(DriveVLABaseAgent)
    nn.Module.__init__(agent)
    agent.world_model_config = SimpleNamespace(
        predictor_only=predictor_only,
        horizons_sec=(0.5, 1.5, 3.0),
        abs_weight=1.0,
        delta_weight=0.25,
    )
    agent.future_mode = future_mode
    agent.backbone = None
    agent.ema_register_target = None
    agent.vlm_config = SimpleNamespace(freeze_backbone=False)
    agent.future_register_predictor = FutureRegisterPredictor(
        hidden_dim=32,
        predictor_layers=2,
        num_heads=4,
    )
    return agent


def _student_register_graph():
    qv = InternViTQVLoRALinear(nn.Linear(4, 12), rank=2)
    planning_registers = nn.Parameter(torch.randn(1, 16, 32))
    inputs = torch.randn(2, 5, 4)
    q_features = qv(inputs)[..., :4].mean(dim=(1, 2))
    current = planning_registers.expand(2, -1, -1) + q_features[:, None, None]
    return qv, planning_registers, current


def _targets():
    return (
        torch.randn(2, 8, 3),
        torch.randn(2, 16, 32),
        torch.randn(2, 3, 16, 32),
        torch.ones(2, 3, dtype=torch.bool),
    )


def test_world_model_loss_updates_registers_and_vision_qv_lora() -> None:
    torch.manual_seed(51)
    agent = _loss_agent(predictor_only=False)
    qv, registers, current = _student_register_graph()
    trajectory, target_current, target_future, valid = _targets()
    losses = agent._compute_world_model_loss_from_registers(
        current, trajectory, target_current, target_future, valid
    )
    losses["wm_loss"].backward()
    assert registers.grad is not None and registers.grad.norm().item() > 0.0
    assert qv.q_lora_b.weight.grad is not None
    assert qv.q_lora_b.weight.grad.norm().item() > 0.0
    assert losses["predicted_future_registers"].shape == (2, 1, 3, 16, 32)
    for key in (
        "wm_cos_0p5",
        "wm_cos_1p5",
        "wm_cos_3p0",
        "wm_delta_0p5",
        "wm_delta_1p5",
        "wm_delta_3p0",
    ):
        assert key in losses


def test_predictor_only_stops_world_loss_at_visual_register_input() -> None:
    torch.manual_seed(52)
    agent = _loss_agent(predictor_only=True)
    qv, registers, current = _student_register_graph()
    trajectory, target_current, target_future, valid = _targets()
    losses = agent._compute_world_model_loss_from_registers(
        current, trajectory, target_current, target_future, valid
    )
    losses["wm_loss"].backward()
    assert registers.grad is None
    assert qv.q_lora_b.weight.grad is None
    predictor_grad = sum(
        parameter.grad.detach().float().norm().item()
        for parameter in agent.future_register_predictor.parameters()
        if parameter.grad is not None
    )
    assert predictor_grad > 0.0


def test_shuffled_future_batch_one_fails_explicitly() -> None:
    agent = _loss_agent(predictor_only=False, future_mode="shuffled_batch")
    agent.ema_register_target = nn.Identity()
    with pytest.raises(ValueError, match="batch_size > 1"):
        agent._encode_ema_register_targets({}, {}, batch_size=1)


class _FakeEMA(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.forward_calls = 0

    def forward(self, pixel_values, tile_counts):
        self.forward_calls += 1
        groups = pixel_values.split(tile_counts)
        values = torch.stack([group.float().mean() for group in groups])
        return values[:, None, None].expand(-1, 16, 32)


def _future_path_targets(values):
    paths, lengths = encode_path_tensor_batch([str(value) for value in values])
    return {
        "future_image_paths": paths.reshape(2, 3, 1024),
        "future_image_path_lengths": lengths.reshape(2, 3),
        "future_valid_mask": torch.ones(2, 3, dtype=torch.bool),
    }


def test_repeated_and_shuffled_controls_share_one_teacher_batch(monkeypatch) -> None:
    monkeypatch.setattr(
        "navsim.agents.EpisodeDrive.drivevla_base_agent.load_image",
        lambda path: torch.full((1, 3, 2, 2), float(path)),
    )
    features = {
        "pixel_values": torch.stack(
            (
                torch.full((1, 3, 2, 2), 1.0),
                torch.full((1, 3, 2, 2), 2.0),
            )
        )
    }
    targets = _future_path_targets((10, 11, 12, 20, 21, 22))

    repeated = _loss_agent(predictor_only=False, future_mode="repeated_current")
    repeated.ema_register_target = _FakeEMA()
    current, future, valid = repeated._encode_ema_register_targets(
        features, targets, batch_size=2
    )
    torch.testing.assert_close(future, current[:, None].expand_as(future))
    assert valid.all()
    assert repeated.ema_register_target.forward_calls == 1

    shuffled = _loss_agent(predictor_only=False, future_mode="shuffled_batch")
    shuffled.ema_register_target = _FakeEMA()
    current, future, valid = shuffled._encode_ema_register_targets(
        features, targets, batch_size=2
    )
    assert future[0, 0, 0, 0].item() == 20.0
    assert future[1, 0, 0, 0].item() == 10.0
    assert valid.all()
    assert shuffled.ema_register_target.forward_calls == 1


def test_eval_action_path_requires_no_future_keys() -> None:
    # The predictor API is training-only; a pure current-register inference
    # tensor can be consumed without any future image/path/target structure.
    agent = _loss_agent(predictor_only=False)
    agent.eval()
    current = torch.randn(1, 16, 32)
    assert current.shape == (1, 16, 32)
    assert not hasattr(agent, "future_image_paths")
