from __future__ import annotations

import ast
from pathlib import Path

from research.cf_effect_gate_wote.src import replay_effect_builder
from research.cf_effect_gate_wote.src.effect_prediction import PREDICTOR_INPUT_SCHEMA
from research.cf_effect_gate_wote.src.models.effect_predictor import (
    CandidateEffectPredictor,
    count_trainable_parameters,
    effect_prediction_loss,
)
import pytest
import torch


def test_replay_builder_never_reads_planning_metric_labels() -> None:
    source_path = Path(replay_effect_builder.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    forbidden_exact = {
        "score",
        "scores",
        "factor",
        "factors",
        "pdms",
        "epdms",
        "selected_index",
        "candidate_index_label",
    }
    names = {
        node.id.lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
    }
    strings = {
        node.value.lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert not (forbidden_exact & names)
    assert not (forbidden_exact & strings)


def test_forward_effect_predictor_has_current_only_input_contract() -> None:
    assert PREDICTOR_INPUT_SCHEMA == (
        "current_bev_tokens",
        "ego_status_feature",
        "candidate_trajectory",
    )
    model = CandidateEffectPredictor(hidden_dim=64, decoder_layers=2, attention_heads=8)
    assert count_trainable_parameters(model) < 10_000_000
    model.eval()
    current = torch.randn(1, 64, 256)
    ego = torch.randn(1, 8)
    trajectory = torch.randn(1, 3, 8, 3)
    with torch.inference_mode():
        ordinary = model(current, ego, trajectory)
        swapped = model(current, ego, trajectory[:, [2, 0, 1]])
    for key in ordinary:
        torch.testing.assert_close(ordinary[key][:, [2, 0, 1]], swapped[key])


def test_effect_loss_rejects_planning_labels() -> None:
    model = CandidateEffectPredictor(hidden_dim=64, decoder_layers=2, attention_heads=8)
    prediction = model(
        torch.zeros(1, 64, 256), torch.zeros(1, 8), torch.zeros(1, 2, 8, 3)
    )
    target = {
        "ego_effect": torch.zeros(1, 2, 8, 16),
        "map_effect": torch.zeros(1, 2, 8, 8),
        "actor_effect": torch.zeros(1, 2, 8, 16, 13),
        "actor_mask": torch.ones(1, 2, 8, 16, dtype=torch.bool),
        "interaction_mask": torch.zeros(1, 2, 8, 16, dtype=torch.bool),
        "candidate_score": torch.zeros(1, 2),
    }
    with pytest.raises(ValueError, match="forbidden/unknown"):
        effect_prediction_loss(prediction, target)
