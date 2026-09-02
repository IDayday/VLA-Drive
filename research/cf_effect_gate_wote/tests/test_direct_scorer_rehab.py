from __future__ import annotations

import numpy as np
import pytest
import torch

from research.cf_effect_gate_wote.src.direct_current_cache import (
    selector_index_is_maximal,
)
from research.cf_effect_gate_wote.src.direct_rehab_ensemble import (
    _top1_metrics,
    policy_selection_values,
)
from research.cf_effect_gate_wote.src.direct_rehab_metrics import scene_level_metrics
from research.cf_effect_gate_wote.src.models.top_aware_direct_scorer import (
    CandidateToBEVGrid,
    TopAwareDirectScorerConfig,
    TopAwareDirectScorerV3,
)
from research.cf_effect_gate_wote.src.top_aware_losses import (
    TopAwareLossConfig,
    full_listnet_loss,
    hard_safe_targets,
    top_aware_direct_loss,
)


def _inputs(batch: int = 2, candidates: int = 256):
    generator = torch.Generator().manual_seed(20260827)
    trajectory = torch.randn(batch, candidates, 8, 3, generator=generator)
    ego = torch.randn(batch, 8, generator=generator)
    bev = torch.randn(batch, 64, 256, generator=generator)
    candidate = torch.randn(batch, candidates, 256, generator=generator)
    return trajectory, ego, bev, candidate


@pytest.mark.parametrize(
    "representation",
    [
        "trajectory_only",
        "old_spatial_xattn",
        "pretrained_candidate_query",
        "path_aligned_current",
        "hybrid_current",
        "wote_current_only_rollout",
    ],
)
def test_direct_v3_output_contract(representation: str) -> None:
    trajectory, ego, bev, candidate = _inputs(batch=1, candidates=7)
    model = TopAwareDirectScorerV3(
        TopAwareDirectScorerConfig(representation=representation)
    ).eval()
    with torch.inference_mode():
        output = model(trajectory, ego, bev, candidate, candidate_chunk=3)
    assert output["factor_logits"].shape == (1, 7, 6)
    assert output["factors"].shape == (1, 7, 6)
    for key in ("factor_score", "utility_logit", "utility_score", "hard_safety_logit"):
        assert output[key].shape == (1, 7)
        assert torch.isfinite(output[key]).all()


def test_direct_v3_candidate_chunk_is_exact() -> None:
    trajectory, ego, bev, candidate = _inputs(batch=1, candidates=17)
    model = TopAwareDirectScorerV3(
        TopAwareDirectScorerConfig(representation="hybrid_current")
    ).eval()
    with torch.inference_mode():
        whole = model(trajectory, ego, bev, candidate)
        chunked = model(trajectory, ego, bev, candidate, candidate_chunk=5)
    for key in whole:
        torch.testing.assert_close(whole[key], chunked[key], rtol=1.0e-6, atol=2.0e-7)


def test_candidate_grid_matches_registered_wote_axes() -> None:
    grid = CandidateToBEVGrid()
    xy = torch.tensor([[[[0.0, 0.0], [4.0, -8.0], [28.0, 24.0]]]])
    row, column = grid.continuous_indices(xy)
    torch.testing.assert_close(row, torch.tensor([[[0.0, 1.0, 7.0]]]))
    torch.testing.assert_close(column, torch.tensor([[[4.0, 3.0, 7.0]]]))


def test_full_list_objective_rejects_candidate_subsets() -> None:
    with pytest.raises(ValueError, match="all 256"):
        full_listnet_loss(
            torch.zeros(2, 64),
            torch.zeros(2, 64),
            target_temperature=0.05,
            prediction_temperature=1.0,
        )


def test_six_factor_o3_loss_is_finite_and_uses_ddc_soft_target() -> None:
    trajectory, ego, bev, candidate = _inputs(batch=1)
    model = TopAwareDirectScorerV3(
        TopAwareDirectScorerConfig(representation="trajectory_only")
    )
    outputs = model(trajectory, ego, bev, candidate, candidate_chunk=64)
    factors = torch.ones(1, 256, 6)
    factors[:, ::3, 2] = 0.5
    factors[:, ::7, 2] = 0.0
    score = (
        factors[..., 0]
        * factors[..., 1]
        * factors[..., 2]
        * (5 * factors[..., 3] + 5 * factors[..., 4] + 2 * factors[..., 5])
        / 12.0
    )
    losses = top_aware_direct_loss(outputs, factors, score, TopAwareLossConfig())
    assert torch.isfinite(losses["total"])
    safe = hard_safe_targets(factors)
    assert safe[0, 3].item() == 1.0  # DDC=0.5 is not a hard false-safe event.
    assert safe[0, 7].item() == 0.0


def test_selector_reference_accepts_quantized_tie_but_not_lower_reward() -> None:
    rewards = np.asarray([2.0, 2.1, 2.1, 1.9], dtype=np.float16)
    assert selector_index_is_maximal(rewards, 1)
    assert selector_index_is_maximal(rewards, 2)
    assert not selector_index_is_maximal(rewards, 0)


def test_hard_false_safe_is_selected_label_event_not_prediction_event() -> None:
    selection = np.asarray([[1.0, 2.0]])
    predicted = np.ones((1, 2, 6), dtype=np.float64)
    factors = np.ones((1, 2, 6), dtype=np.float64)
    factors[0, 1, 2] = 0.0
    scores = np.asarray([[1.0, 0.0]])
    rows = scene_level_metrics(
        ["scene"],
        selection,
        predicted,
        factors,
        scores,
        predicted_hard_safety=np.asarray([[-10.0, -10.0]]),
    )
    assert rows[0]["selected_index"] == 1
    assert rows[0]["hard_false_safe"] is True


def test_safe_policy_no_eligible_row_falls_back_exactly() -> None:
    score = np.asarray([[0.9, 0.8, 0.7]], dtype=np.float32)
    factors = np.zeros((1, 3, 6), dtype=np.float32)
    values, diagnostics = policy_selection_values(
        score,
        factors,
        np.asarray([2]),
        factor_floor=1.0,
        margin=1.0,
    )
    assert np.argmax(values, axis=1).tolist() == [2]
    assert diagnostics["no_eligible_scene_fraction"] == 1.0
    assert diagnostics["override_fraction"] == 0.0


def test_safe_policy_uses_predictions_only_for_selection() -> None:
    score = np.asarray([[0.1, 0.9, 0.5]], dtype=np.float32)
    factors = np.ones((1, 3, 6), dtype=np.float32)
    values, _ = policy_selection_values(
        score,
        factors,
        np.asarray([0]),
        factor_floor=0.6,
        margin=0.3,
    )
    assert np.argmax(values, axis=1).tolist() == [1]


def test_arbitrated_top1_metrics_exclude_invalid_ranking_statistics() -> None:
    rows = [
        {
            "selected_score": 0.8,
            "regret": 0.1,
            "selected_rank": 2,
            "hard_false_safe": False,
            "direction_non_compliance": True,
            "zero_score_selection": False,
            "oracle_capture": False,
            "score_overestimation": 1.0e9,
        }
    ]
    metrics = _top1_metrics(rows)
    assert metrics["selected_score"] == 0.8
    assert "ndcg_at_5" not in metrics
    assert "all_pair_pairwise_accuracy" not in metrics
    assert "selected_score_overestimation" not in metrics
