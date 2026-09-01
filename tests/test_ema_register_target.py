from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from navsim.agents.EpisodeDrive.layers.planning_registers import (
    InternVLPlanningRegisters,
)
from navsim.agents.EpisodeDrive.layers.world_model import (
    EMARegisterTarget,
    cosine_ema_momentum,
)


class _Embeddings(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(3, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        patches = pixels.permute(0, 2, 3, 1).reshape(pixels.shape[0], -1, 3)
        return torch.cat((self.cls.expand(pixels.shape[0], -1, -1), self.proj(patches)), 1)


class _Encoder(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim), nn.Linear(dim, dim)])

    def forward(self, inputs_embeds, output_hidden_states=False, return_dict=True):
        hidden = inputs_embeds
        for layer in self.layers:
            hidden = hidden + 0.01 * layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


class _Vision(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.embeddings = _Embeddings(dim)
        self.encoder = _Encoder(dim)


class _StudentBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.vision_model = _Vision(8)
        self.model.language_model = nn.Linear(8, 8)
        self.planning_register_adapter = InternVLPlanningRegisters(8, 16, 32)


def test_ema_contains_only_vision_and_register_modules_and_has_no_grad() -> None:
    student = _StudentBackbone()
    teacher = EMARegisterTarget(student)
    names = tuple(name for name, _ in teacher.named_parameters())
    assert names
    assert not any("language" in name for name in names)
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    teacher.train(True)
    assert not teacher.training

    output = teacher(torch.randn(2, 3, 2, 2), [1, 1])
    assert output.shape == (2, 16, 32)
    assert not output.requires_grad


def test_ema_update_and_cosine_schedule_are_exact() -> None:
    torch.manual_seed(41)
    student = _StudentBackbone()
    teacher = EMARegisterTarget(student)
    name, student_parameter = next(iter(student.model.vision_model.named_parameters()))
    teacher_parameter = dict(teacher.vision_model.named_parameters())[name]
    before = teacher_parameter.detach().clone()
    with torch.no_grad():
        student_parameter.add_(2.0)
    expected = before * 0.5 + student_parameter.detach() * 0.5
    teacher.update(student, momentum=0.5)
    torch.testing.assert_close(teacher_parameter, expected)

    assert cosine_ema_momentum(0, 100) == pytest.approx(0.996)
    assert cosine_ema_momentum(100, 100) == pytest.approx(0.9999)
    middle = cosine_ema_momentum(50, 100)
    assert 0.996 < middle < 0.9999
