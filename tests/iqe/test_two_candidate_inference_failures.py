"""Selection behavior with synthetic inputs; real-image proof lives in the run audit."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import iqe.internal_baseline.inference as inference


class Adapter(nn.Module):
    def __init__(self, physical):
        super().__init__()
        self.physical = physical

    def predict_s0(self, planner):
        return SimpleNamespace(physical=self.physical)


class Expert(nn.Module):
    def __init__(self, physical):
        super().__init__()
        self.physical = physical

    def forward(self, planner):
        return SimpleNamespace(physical=self.physical)


class Scorer(nn.Module):
    context_kind = "world_action_ego"

    def __init__(self, values, logits=None):
        super().__init__()
        self.values = torch.tensor(values, dtype=torch.float32)
        self.logits = torch.full((1, 2, 2), -100.) if logits is None else logits

    def forward(self, context, trajectories):
        return {"values": self.values, "component_logits": self.logits,
                "valid": torch.isfinite(trajectories).all((-1, -2))}


def algorithm(monkeypatch, values, *, base_invalid=False, expert_invalid=False, logits=None):
    calls = []
    def shared(adapter, observations):
        calls.append(observations)
        return SimpleNamespace(), SimpleNamespace(tensors=lambda: {})
    monkeypatch.setattr(inference, "encode_joint", shared)
    base, expert = torch.zeros(1, 8, 3), torch.ones(1, 8, 3)
    if base_invalid:
        base[0, 0, 0] = float("nan")
    if expert_invalid:
        expert[0, 0, 0] = float("nan")
    return inference.InternalAlgorithm(Adapter(base), Expert(expert), Scorer(values, logits)), calls


def test_raw_argmax_has_no_hidden_component_guard_and_one_shared_forward(monkeypatch):
    model, calls = algorithm(monkeypatch, [[.1, .9]])
    result = model([{"current_observation": "synthetic"}], return_debug=True)
    assert result["selected_expert"].tolist() == [1]
    assert torch.equal(result["trajectory"], torch.ones(1, 8, 3))
    assert len(calls) == 1
    assert not model.train().training


@pytest.mark.parametrize("values,margin,expected", [([[.5, .5]], 0., 0), ([[.5, .6]], .2, 0),
                                                   ([[.5, .7]], .1, 1)])
def test_tie_and_explicit_margin(monkeypatch, values, margin, expected):
    model, _ = algorithm(monkeypatch, values)
    assert model([{}], margin=margin, return_debug=True)["selected_expert"].item() == expected


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("candidate", [0, 1])
def test_nonfinite_valid_candidate_value_is_failure_not_silent_base_fallback(monkeypatch, invalid, candidate):
    values = [[.5, .6]]
    values[0][candidate] = invalid
    model, _ = algorithm(monkeypatch, values)
    with pytest.raises(ValueError, match="nonfinite scorer value"):
        model([{}])


def test_invalid_expert_is_excluded_even_if_its_prediction_is_nonfinite(monkeypatch):
    logits = torch.zeros(1, 2, 2)
    logits[0, 1] = float("nan")
    model, _ = algorithm(monkeypatch, [[.5, float("nan")]], expert_invalid=True, logits=logits)
    result = model([{}], return_debug=True)
    assert result["selected_expert"].item() == 0
    assert torch.equal(result["trajectory"], torch.zeros(1, 8, 3))


def test_invalid_base_raises_without_inventing_a_trajectory(monkeypatch):
    model, _ = algorithm(monkeypatch, [[.5, .6]], base_invalid=True)
    with pytest.raises(ValueError, match="Base trajectory invalid"):
        model([{}])


def test_nonfinite_component_for_legal_candidate_is_not_hidden(monkeypatch):
    logits = torch.zeros(1, 2, 2)
    logits[0, 1, 0] = float("nan")
    model, _ = algorithm(monkeypatch, [[.5, .6]], logits=logits)
    with pytest.raises(ValueError, match="nonfinite scorer component"):
        model([{}])


@pytest.mark.parametrize("margin", [float("nan"), float("inf"), -.01])
def test_margin_must_be_finite_nonnegative(monkeypatch, margin):
    model, calls = algorithm(monkeypatch, [[.5, .6]])
    with pytest.raises(ValueError, match="finite nonnegative"):
        model([{}], margin=margin)
    assert not calls
