from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from navsim.agents.EpisodeDrive.layers.losses.episode_drive_loss import (
    EpisodeDriveLoss,
)
from navsim.agents.EpisodeDrive.score_module.scorer import (
    DRIVOR_SCORE_HEAD_NAMES,
    Scorer,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_audit_module():
    path = REPO_ROOT / "scripts" / "audit_drivor_scorer_parity.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def parity_report():
    audit = _load_audit_module()
    repo = Path(os.environ.get("DRIVOR_REPO", "/mnt/project/external/DrivoR"))
    if not (repo / ".git").exists():
        pytest.skip(
            "Set DRIVOR_REPO to a clone containing the frozen DrivoR commit"
        )
    return audit.run_audit(repo)


def test_frozen_upstream_outputs_match(parity_report) -> None:
    assert parity_report["passed"]
    assert parity_report["synthetic_shapes"]["proposals"] == [2, 64, 8, 3]
    assert parity_report["synthetic_shapes"]["scene_features"] == [2, 16, 256]
    assert all(
        difference == 0.0
        for difference in parity_report["component_max_abs_diff"].values()
    )
    assert parity_report["pdm_score_max_abs_diff"] == 0.0
    assert parity_report["selected_indices_equal"]


def test_scorer_gradient_routing(parity_report) -> None:
    assert parity_report["proposal_grad_is_none"]
    assert parity_report["proposal_grad_norm"] == 0.0
    assert parity_report["scene_feature_grad_norm"] > 0.0


def test_state_dict_shapes_and_six_independent_heads(parity_report) -> None:
    assert parity_report["scorer_decoder_layers"] == 4
    assert tuple(parity_report["head_names"]) == DRIVOR_SCORE_HEAD_NAMES
    assert parity_report["state_dict_shapes_equal"]
    assert parity_report["b2d_state_dict_shapes_equal"]
    assert parity_report["b2d_agent_output_dim"] == 2 * 6 * 9
    assert parity_report["b2d_area_output_dim"] == 8 * 2


def test_ttc_target_two_is_masked() -> None:
    logits = {
        name: torch.zeros(
            1,
            3,
            dtype=torch.float64,
            requires_grad=(name == "time_to_collision_within_bound"),
        )
        for name in DRIVOR_SCORE_HEAD_NAMES
    }
    ttc_logits = logits["time_to_collision_within_bound"]
    with torch.no_grad():
        ttc_logits[0, 2] = 100.0
    target_scores = torch.zeros(1, 3, 7)
    target_scores[..., 3] = torch.tensor([[0.0, 1.0, 2.0]])

    sub_losses = EpisodeDriveLoss().score_loss(
        logits,
        None,
        None,
        None,
        target_scores,
        None,
        None,
        None,
        None,
    )[0]
    ttc_loss = sub_losses[1]
    expected = F.binary_cross_entropy_with_logits(
        torch.zeros(2, dtype=torch.float64),
        torch.tensor([0.0, 1.0], dtype=torch.float64),
    )
    assert math.isclose(ttc_loss.item(), expected.item(), rel_tol=0.0, abs_tol=0.0)

    ttc_loss.backward()
    assert ttc_logits.grad is not None
    assert ttc_logits.grad[0, 2].item() == 0.0


def test_b2d_false_default_shapes() -> None:
    audit = _load_audit_module()
    config = audit.make_config(
        b2d=False,
        agent_pred=True,
        area_pred=True,
    )
    scorer = Scorer(config)
    assert scorer.pred_col_agent.mlp[-1].out_features == 2 * 40 * 9
    assert scorer.pred_area.mlp[-1].out_features == 8 * 5 * 2
