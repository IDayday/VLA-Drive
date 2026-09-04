from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from navsim.agents.EpisodeDrive.drivevla_base_agent import (
    planreg_continuation_cosine_multiplier,
    planreg_warmup_cosine_multiplier,
)
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from test_planreg_optimizer_groups import _agent


def _scheduler_agent():
    agent = _agent()
    agent.scheduler_args = {
        "warmup_ratio": 0.03,
        "start_lr_ratio": 0.01,
        "min_lr_ratios": {
            "planning_adapter": 0.10,
            "future_predictor": 0.10,
            "fusion": 0.10,
            "vision_qv_lora": 0.10,
            "action_head": 0.20,
            "scorer": 0.20,
            "semantic_qformer": 0.20,
        },
    }
    return agent


def test_warmup_cosine_boundary_multipliers() -> None:
    arguments = {
        "total_optimizer_steps": 101,
        "warmup_ratio": 0.03,
        "start_lr_ratio": 0.01,
        "min_lr_ratio": 0.10,
    }
    assert planreg_warmup_cosine_multiplier(0, **arguments) == pytest.approx(0.01)
    assert planreg_warmup_cosine_multiplier(3, **arguments) == pytest.approx(1.0)
    assert planreg_warmup_cosine_multiplier(100, **arguments) == pytest.approx(0.10)


def test_continuation_cosine_is_continuous_at_restored_origin() -> None:
    arguments = {
        "origin_optimizer_step": 21_789,
        "continuation_optimizer_steps": 150,
        "reheat_optimizer_steps": 20,
        "start_lr_ratio": 0.10,
        "peak_lr_ratio": 0.20,
        "min_lr_ratio": 0.08,
    }
    assert planreg_continuation_cosine_multiplier(
        21_788, **arguments
    ) == pytest.approx(0.10)
    assert planreg_continuation_cosine_multiplier(
        21_789, **arguments
    ) == pytest.approx(0.10)
    assert planreg_continuation_cosine_multiplier(
        21_809, **arguments
    ) == pytest.approx(0.20)
    assert planreg_continuation_cosine_multiplier(
        21_939, **arguments
    ) == pytest.approx(0.08)
    assert planreg_continuation_cosine_multiplier(
        30_000, **arguments
    ) == pytest.approx(0.08)


def test_formal_continuation_lr_boundaries() -> None:
    common = {
        "origin_optimizer_step": 21_789,
        "continuation_optimizer_steps": 4_842,
        "reheat_optimizer_steps": 404,
    }
    logical_ratios = {
        "planning_adapter": (0.10, 0.15, 1 / 60),
        "semantic_fusion": (0.10, 0.15, 1 / 60),
        "action_generator": (0.10, 0.15, 2 / 75),
        "scorer": (0.10, 0.15, 1 / 30),
        "future_predictor": (0.10, 0.10, 0.01),
        "semantic_qformer": (0.10, 0.10, 1 / 75),
        "vision_qv_lora": (0.10, 0.10, 0.02),
    }
    for start, peak, minimum in logical_ratios.values():
        arguments = {
            **common,
            "start_lr_ratio": start,
            "peak_lr_ratio": peak,
            "min_lr_ratio": minimum,
        }
        assert planreg_continuation_cosine_multiplier(
            21_789, **arguments
        ) == pytest.approx(start)
        assert planreg_continuation_cosine_multiplier(
            22_193, **arguments
        ) == pytest.approx(peak)
        assert planreg_continuation_cosine_multiplier(
            26_631, **arguments
        ) == pytest.approx(minimum)


def test_scheduler_uses_group_specific_minimum_ratios() -> None:
    agent = _scheduler_agent()
    optimizers, scheduler_configs = agent.get_optimizers(
        total_optimizer_steps=101
    )
    optimizer = optimizers[0]
    scheduler = scheduler_configs[0]["scheduler"]
    assert scheduler_configs[0]["interval"] == "step"
    for group, lr_lambda in zip(optimizer.param_groups, scheduler.lr_lambdas):
        expected = 0.10 if group["logical_name"] in {
            "planning_adapter",
            "future_predictor",
            "fusion",
            "vision_qv_lora",
        } else 0.20
        assert lr_lambda(100) == pytest.approx(expected)
        assert group["lr"] == pytest.approx(group["initial_lr"] * 0.01)


def test_scheduler_resume_lr_sequence_is_exact() -> None:
    total_steps = 30
    first = _scheduler_agent()
    first_optimizers, first_configs = first.get_optimizers(
        total_optimizer_steps=total_steps
    )
    first_optimizer = first_optimizers[0]
    first_scheduler = first_configs[0]["scheduler"]

    for _ in range(9):
        first_optimizer.step()
        first_scheduler.step()
    optimizer_state = copy.deepcopy(first_optimizer.state_dict())
    scheduler_state = copy.deepcopy(first_scheduler.state_dict())

    resumed = _scheduler_agent()
    resumed_optimizers, resumed_configs = resumed.get_optimizers(
        total_optimizer_steps=total_steps
    )
    resumed_optimizer = resumed_optimizers[0]
    resumed_scheduler = resumed_configs[0]["scheduler"]
    resumed_optimizer.load_state_dict(optimizer_state)
    resumed_scheduler.load_state_dict(scheduler_state)

    uninterrupted_lrs = []
    resumed_lrs = []
    for _ in range(9, total_steps):
        first_optimizer.step()
        first_scheduler.step()
        uninterrupted_lrs.append(tuple(first_scheduler.get_last_lr()))
        resumed_optimizer.step()
        resumed_scheduler.step()
        resumed_lrs.append(tuple(resumed_scheduler.get_last_lr()))
    assert resumed_lrs == uninterrupted_lrs


def test_lightning_passes_authoritative_estimated_step_count() -> None:
    class _Agent(nn.Module):
        def __init__(self):
            super().__init__()
            self.parameter = nn.Parameter(torch.ones(()))
            self.received_steps = None

        def get_optimizers(self, total_optimizer_steps=None):
            self.received_steps = total_optimizer_steps
            return torch.optim.SGD(self.parameters(), lr=0.1)

    agent = _Agent()
    module = AgentLightningModule(agent)
    module._trainer = type(
        "TrainerStub", (), {"estimated_stepping_batches": 123}
    )()
    module.configure_optimizers()
    assert agent.received_steps == 123
