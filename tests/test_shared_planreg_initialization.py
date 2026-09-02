from pathlib import Path

import torch
from torch import nn

from navsim.agents.EpisodeDrive.shared_planreg_initialization import (
    capture_shared_trainable_state,
    load_shared_trainable_initialization,
    save_shared_trainable_initialization,
)


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.planning_register_adapter = nn.Linear(3, 4)
        self.q_lora_a = nn.Linear(4, 2, bias=False)


class _ActionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_former = nn.Linear(4, 4)
        self.hist_encoding = nn.Linear(4, 4)
        self.scorer_attention = nn.Linear(4, 4)
        self.semantic_gate = nn.Parameter(torch.tensor(-1.0))


class _FormalStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _Backbone()
        self.action_head = _ActionHead()
        self.future_register_predictor = nn.Linear(4, 4)


def test_shared_initialization_round_trip_is_bitwise(tmp_path: Path):
    torch.manual_seed(7)
    source = _FormalStack()
    state, metadata = capture_shared_trainable_state(source)
    artifact = tmp_path / "shared.pt"
    saved = save_shared_trainable_initialization(
        source,
        str(artifact),
        seed=7,
        architecture_config_sha256="architecture",
        source_git_commit="commit",
    )
    assert saved["trainable_state_sha256"] == metadata["trainable_state_sha256"]

    torch.manual_seed(99)
    target = _FormalStack()
    assert any(
        not torch.equal(value, dict(target.named_parameters())[name])
        for name, value in state.items()
    )
    restored = load_shared_trainable_initialization(target, str(artifact))
    assert restored["trainable_state_sha256"] == metadata["trainable_state_sha256"]
    for name, parameter in target.named_parameters():
        assert torch.equal(parameter.detach().cpu(), state[name])


def test_shared_initialization_rejects_topology_mismatch(tmp_path: Path):
    source = _FormalStack()
    artifact = tmp_path / "shared.pt"
    save_shared_trainable_initialization(
        source,
        str(artifact),
        seed=0,
        architecture_config_sha256="architecture",
        source_git_commit="commit",
    )
    target = _FormalStack()
    target.extra = nn.Parameter(torch.zeros(1))
    try:
        load_shared_trainable_initialization(target, str(artifact))
    except RuntimeError as error:
        assert "Unclassified trainable parameter" in str(error) or "topology mismatch" in str(error)
    else:
        raise AssertionError("Topology mismatch was accepted")
