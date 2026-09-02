from __future__ import annotations

import inspect
from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

from navsim.agents.EpisodeDrive.layers.planning_registers.register_diagnostics import (
    compute_register_diagnostics,
)
from navsim.planning.training.agent_lightning_module import AgentLightningModule


class _DiagnosticAgent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.gradient_calls = 0
        self.register_calls = 0

    def get_planreg_gradient_norms(self):
        self.gradient_calls += 1
        return {
            "vision_lora_grad_norm": torch.tensor(1.0),
            "register_grad_norm": torch.tensor(2.0),
            "future_predictor_grad_norm": torch.tensor(3.0),
            "action_head_grad_norm": torch.tensor(4.0),
            "scorer_grad_norm": torch.tensor(5.0),
        }

    def get_planreg_register_diagnostics(self):
        self.register_calls += 1
        return {"register_std": torch.tensor(0.5)}


def _module(*, step: int, debug: bool = False, require_finite: bool = False):
    agent = _DiagnosticAgent()
    module = AgentLightningModule(
        agent,
        diagnostics={
            "grad_log_interval": 10,
            "register_log_interval": 20,
            "debug_unused_parameters": debug,
            "require_finite_loss_and_gradients": require_finite,
        },
    )
    module._trainer = SimpleNamespace(global_step=step, is_global_zero=True)
    logged = {}
    module.log = lambda name, value, **_: logged.setdefault(name, value)
    return module, agent, logged


def test_only_one_on_after_backward_definition_exists() -> None:
    source = inspect.getsource(AgentLightningModule)
    assert source.count("def on_after_backward") == 1


def test_gradient_and_register_diagnostics_are_interval_gated() -> None:
    module, agent, logged = _module(step=20)
    module.on_after_backward()
    assert agent.gradient_calls == 1
    assert agent.register_calls == 1
    assert set(logged) == {
        "train/vision_lora_grad_norm",
        "train/register_grad_norm",
        "train/future_predictor_grad_norm",
        "train/action_head_grad_norm",
        "train/scorer_grad_norm",
        "train/register_std",
    }

    module, agent, logged = _module(step=11)
    module.on_after_backward()
    assert agent.gradient_calls == 0
    assert agent.register_calls == 0
    assert logged == {}


def test_unused_parameter_walk_is_disabled_by_default() -> None:
    module, _, _ = _module(step=20, debug=False)

    def fail_if_walked(self, *args, **kwargs):
        raise AssertionError("full named_parameters traversal must stay disabled")

    module.named_parameters = MethodType(fail_if_walked, module)
    module.on_after_backward()


def test_finite_gradient_hooks_do_not_walk_parameters_per_step() -> None:
    module, agent, _ = _module(step=1, require_finite=True)

    def fail_if_walked(self, *args, **kwargs):
        raise AssertionError("full named_parameters traversal must stay disabled")

    module.named_parameters = MethodType(fail_if_walked, module)
    loss = agent.weight * torch.tensor(float("nan"))
    module.on_before_backward(loss)
    loss.backward()
    with pytest.raises(FloatingPointError, match="agent.weight"):
        module.on_after_backward()


def test_register_svd_diagnostics_are_detached_and_finite() -> None:
    registers = torch.randn(2, 16, 32, requires_grad=True)
    diagnostics = compute_register_diagnostics(registers)
    assert set(diagnostics) == {
        "register_effective_rank",
        "register_mean_pairwise_cosine",
        "register_std",
    }
    assert all(torch.isfinite(value) for value in diagnostics.values())
    assert all(not value.requires_grad for value in diagnostics.values())
