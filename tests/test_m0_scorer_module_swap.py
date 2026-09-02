from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from local_stage2.audit_m0_scorer_module_swap import (
    ContextTensors,
    ProposalAuditData,
    aggregate_factor_logits,
    component_drift,
    evaluate_scores,
    load_aligned_context,
    load_aligned_pickle_context,
    load_navtest_proposal_audit_data,
    score_module_combinations,
)


class _ToyAttention(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.pos_embed = nn.Linear(24, 4, bias=False)
        with torch.no_grad():
            self.pos_embed.weight.fill_(scale / 24.0)

    def scorer_attention(
        self, embedded: torch.Tensor, scene: torch.Tensor
    ) -> torch.Tensor:
        return embedded + scene.mean(dim=1, keepdim=True)


class _ToyScorer(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, proposals: torch.Tensor, features: torch.Tensor):
        keys = (
            "no_at_fault_collisions",
            "drivable_area_compliance",
            "driving_direction_compliance",
            "time_to_collision_within_bound",
            "ego_progress",
            "comfort",
        )
        prediction = {
            key: features[..., index % features.shape[-1]] * self.scale
            for index, key in enumerate(keys)
        }
        return prediction, None, None, None, None, None, None


class _ToyModel(nn.Module):
    def __init__(self, attention_scale: float, head_scale: float):
        super().__init__()
        attention = _ToyAttention(attention_scale)
        self.pos_embed = attention.pos_embed
        self.scorer_attention = attention.scorer_attention
        self.scorer = _ToyScorer(head_scale)


def test_factor_aggregation_matches_released_formula() -> None:
    logits = torch.tensor([[[0.2, -0.1, 9.0, 0.4, 0.5, -0.3]]])
    probability = logits.sigmoid()
    expected = (
        probability[..., 0].log()
        + probability[..., 1].log()
        + (
            5 * probability[..., 3]
            + 5 * probability[..., 4]
            + 2 * probability[..., 5]
        ).log()
    )
    torch.testing.assert_close(aggregate_factor_logits(logits), expected)


def test_module_swap_scores_are_candidate_permutation_equivariant() -> None:
    torch.manual_seed(7)
    proposals = torch.randn(2, 5, 8, 3)
    contexts = {
        "a": ContextTensors(torch.randn(2, 3, 4), torch.randn(2, 1, 4)),
    }
    models = {"one": _ToyModel(1.0, 1.0), "two": _ToyModel(2.0, 0.5)}
    original = score_module_combinations(
        proposals, contexts, models, device=torch.device("cpu"), batch_size=2
    )
    permutation = torch.tensor([4, 1, 3, 0, 2])
    permuted = score_module_combinations(
        proposals[:, permutation],
        contexts,
        models,
        device=torch.device("cpu"),
        batch_size=2,
    )
    assert len(original) == 4
    for key in original:
        torch.testing.assert_close(permuted[key], original[key][:, permutation])


def test_context_loader_aligns_tokens_not_chunk_order(tmp_path: Path) -> None:
    shard = tmp_path / "all_shard_000-of-001"
    shard.mkdir()
    torch.save(
        {
            "tokens": ["b", "a"],
            "scene_features": torch.stack(
                [torch.full((16, 256), 2.0), torch.full((16, 256), 1.0)]
            ),
            "ego_features": torch.stack(
                [torch.full((1, 256), 20.0), torch.full((1, 256), 10.0)]
            ),
        },
        shard / "chunk_000000.pt",
    )
    result = load_aligned_context(tmp_path, ["a", "b"])
    assert float(result.scene_features[0, 0, 0]) == 1.0
    assert float(result.scene_features[1, 0, 0]) == 2.0
    assert float(result.ego_features[0, 0, 0]) == 10.0


def test_navtest_pickle_loaders_align_by_token(tmp_path: Path) -> None:
    predictions = {}
    for token, value in (("b", 2.0), ("a", 1.0)):
        predictions[token] = {
            "proposals": np.full((64, 8, 3), value, dtype=np.float32),
            "predicted_scores": np.linspace(0, 1, 64, dtype=np.float32) + value,
            "scene_features": np.full((16, 256), value, dtype=np.float32),
            "ego_features": np.full((1, 256), 10 * value, dtype=np.float32),
        }
    proposal_pickle = tmp_path / "predictions.pkl"
    with proposal_pickle.open("wb") as stream:
        pickle.dump(predictions, stream)

    factors = np.zeros((2, 64, 7), dtype=np.float32)
    factors[..., -1] = np.asarray(
        [np.linspace(0, 1, 64), np.linspace(1, 0, 64)], dtype=np.float32
    )
    predicted_scores = np.stack(
        [
            predictions["a"]["predicted_scores"],
            predictions["b"]["predicted_scores"],
        ]
    )
    candidate_npz = tmp_path / "candidate_scores.npz"
    np.savez_compressed(
        candidate_npz,
        tokens=np.asarray(["a", "b"]),
        log_names=np.asarray(["log_a_00000_00001", "log_b_00000_00001"]),
        candidate_scores=factors[..., -1],
        predicted_scores=predicted_scores,
        candidate_factors=factors,
        candidate_factor_names=np.asarray(
            [
                "no_at_fault_collisions",
                "drivable_area_compliance",
                "ego_progress",
                "time_to_collision_within_bound",
                "comfort",
                "driving_direction_compliance",
                "score",
            ]
        ),
    )

    data = load_navtest_proposal_audit_data(proposal_pickle, candidate_npz)
    assert data.tokens == ["a", "b"]
    assert float(data.proposals[0, 0, 0, 0]) == 1.0
    assert float(data.proposals[1, 0, 0, 0]) == 2.0
    assert data.physical_logs == ["log_a", "log_b"]

    context = load_aligned_pickle_context(proposal_pickle, ["a", "b"])
    assert float(context.scene_features[0, 0, 0]) == 1.0
    assert float(context.scene_features[1, 0, 0]) == 2.0
    assert float(context.ego_features[1, 0, 0]) == 20.0


def test_component_drift_separates_semantic_blocks() -> None:
    reference = {
        "q_former.weight": torch.ones(2),
        "scene_embeds": torch.ones(1),
        "hist_encoding.weight": torch.ones(2),
        "pos_embed.0.weight": torch.ones(2),
        "scorer_attention.layer.weight": torch.ones(2),
        "scorer.head.weight": torch.ones(2),
    }
    candidate = {key: value.clone() for key, value in reference.items()}
    candidate["q_former.weight"] += 1
    candidate["scorer.head.weight"] -= 0.5
    result = component_drift(reference, candidate)
    assert result["q_former"]["relative_l2"] > 0
    assert result["factor_heads"]["relative_l2"] > 0
    assert result["ego_encoder"]["relative_l2"] == pytest.approx(0.0)
    assert result["trajectory_embedding"]["cosine_similarity"] == pytest.approx(1.0)


def test_evaluation_uses_target_only_after_predicted_selection() -> None:
    target = torch.zeros(2, 3, 7)
    target[0, :, -1] = torch.tensor([0.1, 0.8, 0.4])
    target[1, :, -1] = torch.tensor([0.7, 0.2, 0.9])
    cached = torch.tensor([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]])
    predicted = torch.tensor([[0.0, 4.0, 1.0], [0.0, 1.0, 4.0]])
    data = ProposalAuditData(
        tokens=["a", "b"],
        segment_logs=["log_00000_00001", "log_00000_00001"],
        physical_logs=["log", "log"],
        proposals=torch.zeros(2, 3, 8, 3),
        cached_base_scores=cached,
        target_factors=target,
    )
    baseline_values = np.asarray([0.1, 0.7], dtype=np.float32)
    result = evaluate_scores(
        predicted,
        data,
        bootstrap_seed=2,
        bootstrap_replicates=10,
        baseline_values=baseline_values,
    )
    assert result["selected_pdms"] == pytest.approx(0.85)
    assert result["best_of_64_pdms"] == pytest.approx(0.85)
    assert result["top1_regret"] == pytest.approx(0.0)
    assert result["selected_pdms_delta_vs_cached_base"] == pytest.approx(0.45)
