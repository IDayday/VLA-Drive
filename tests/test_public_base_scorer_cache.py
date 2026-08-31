from pathlib import Path

import pytest
import torch

from local_stage2.export_public_base_scorer_cache import (
    _first_missing_chunk,
    _partition_tokens,
)
from local_stage2.score_public_base_scorer_cache import _belongs_to_worker
from local_stage2.train_public_base_residual_scorer import binary_factor_loss
from local_stage2.public_base_residual_scorer import (
    PublicBaseResidualRanker,
    ResidualScorerConfig,
    pdm_log_aggregate,
    proposal_kinematic_features,
)


def test_partition_tokens_is_disjoint_and_complete():
    tokens = [f"token-{index}" for index in range(17)]
    shards = [_partition_tokens(tokens, 3, index) for index in range(3)]
    assert sorted(value for shard in shards for value in shard) == sorted(tokens)
    assert not set(shards[0]).intersection(shards[1])
    assert not set(shards[0]).intersection(shards[2])
    assert not set(shards[1]).intersection(shards[2])


def test_first_missing_chunk_requires_contiguous_cache(tmp_path: Path):
    (tmp_path / "chunk_000000.pt").touch()
    assert _first_missing_chunk(tmp_path, 3) == 1
    (tmp_path / "chunk_000002.pt").touch()
    with pytest.raises(RuntimeError, match="Non-contiguous"):
        _first_missing_chunk(tmp_path, 3)


def test_cpu_label_worker_partition_is_deterministic():
    paths = [f"split/chunk_{index:06d}.pt" for index in range(31)]
    assignments = [
        [path for path in paths if _belongs_to_worker(path, 4, index)]
        for index in range(4)
    ]
    assert sorted(path for values in assignments for path in values) == paths
    assert assignments == [
        [path for path in paths if _belongs_to_worker(path, 4, index)]
        for index in range(4)
    ]


@pytest.mark.parametrize("mode", ["local", "set_aware"])
def test_residual_ranker_is_exact_at_zero_initialization(mode: str):
    torch.manual_seed(7)
    config = ResidualScorerConfig(mode=mode, dropout=0.0, top_k=8)
    model = PublicBaseResidualRanker(config).eval()
    proposals = torch.randn(2, 16, 8, 3)
    candidate_features = torch.randn(2, 16, 256)
    factor_logits = torch.randn(2, 16, 6)
    base_scores = torch.randn(2, 16)
    output = model(candidate_features, proposals, factor_logits, base_scores)
    assert torch.equal(output["residual"], torch.zeros_like(base_scores))
    assert torch.equal(output["refined_scores"], base_scores)
    assert torch.equal(
        output["selection_scores"].argmax(dim=1), base_scores.argmax(dim=1)
    )
    assert torch.equal(output["eligible_mask"].sum(dim=1), torch.tensor([8, 8]))
    assert torch.equal(output["top_k_mask"], output["eligible_mask"])


@pytest.mark.parametrize("mode", ["local", "set_aware"])
def test_residual_ranker_is_candidate_permutation_equivariant(mode: str):
    torch.manual_seed(11)
    model = PublicBaseResidualRanker(
        ResidualScorerConfig(mode=mode, dropout=0.0, top_k=8)
    ).eval()
    proposals = torch.randn(2, 16, 8, 3)
    features = torch.randn(2, 16, 256)
    logits = torch.randn(2, 16, 6)
    scores = torch.randn(2, 16)
    permutation = torch.randperm(16)
    direct = model(features, proposals, logits, scores)
    permuted = model(
        features[:, permutation],
        proposals[:, permutation],
        logits[:, permutation],
        scores[:, permutation],
    )
    assert torch.allclose(
        permuted["refined_scores"], direct["refined_scores"][:, permutation], atol=1e-6
    )
    assert torch.equal(
        permuted["eligible_mask"], direct["eligible_mask"][:, permutation]
    )


def test_proposal_kinematics_are_finite_across_heading_wrap():
    proposals = torch.zeros(1, 1, 8, 3)
    proposals[0, 0, :, 0] = torch.arange(1, 9)
    proposals[0, 0, :, 2] = torch.tensor(
        [3.13, -3.13, 3.12, -3.12, 3.11, -3.11, 3.10, -3.10]
    )
    features = proposal_kinematic_features(proposals)
    assert features.shape == (1, 1, 64)
    assert torch.isfinite(features).all()
    assert features.abs().max() <= 1.1


@pytest.mark.parametrize("safety_gate_mode", ["factor_all", "composite"])
def test_safety_gate_keeps_base_candidate_and_filters_predicted_unsafe(
    safety_gate_mode: str,
):
    model = PublicBaseResidualRanker(
        ResidualScorerConfig(
            dropout=0.0,
            top_k=4,
            safety_floor=0.95,
            safety_relative_tolerance=0.05,
            safety_gate_mode=safety_gate_mode,
        )
    ).eval()
    proposals = torch.zeros(1, 4, 8, 3)
    features = torch.zeros(1, 4, 256)
    logits = torch.full((1, 4, 6), -10.0)
    logits[0, 0, [0, 1, 3]] = 10.0
    scores = torch.tensor([[1.0, 0.9, 0.8, 0.7]])
    output = model(features, proposals, logits, scores)
    assert output["eligible_mask"].tolist() == [[True, False, False, False]]
    assert output["selection_scores"].argmax(dim=1).item() == 0
    assert torch.equal(
        output["refined_composite_safety_logit"],
        logits[..., [0, 1, 3]].amin(dim=-1),
    )


@pytest.mark.parametrize(
    "mode", ["scene_cross_attention", "scene_cross_attention_set"]
)
def test_scene_cross_attention_is_zero_init_and_candidate_equivariant(mode: str):
    torch.manual_seed(19)
    model = PublicBaseResidualRanker(
        ResidualScorerConfig(mode=mode, dropout=0.0, top_k=8)
    ).eval()
    proposals = torch.randn(2, 16, 8, 3)
    features = torch.randn(2, 16, 256)
    scene = torch.randn(2, 16, 256)
    ego = torch.randn(2, 1, 256)
    logits = torch.randn(2, 16, 6)
    scores = torch.randn(2, 16)
    permutation = torch.randperm(16)
    zero = model(features, proposals, logits, scores, scene, ego)
    assert torch.equal(zero["residual"], torch.zeros_like(scores))
    assert torch.equal(
        zero["selection_scores"].argmax(dim=1), scores.argmax(dim=1)
    )
    with torch.no_grad():
        model.utility_head[-1].weight.normal_(std=0.02)
    direct = model(features, proposals, logits, scores, scene, ego)
    permuted = model(
        features[:, permutation],
        proposals[:, permutation],
        logits[:, permutation],
        scores[:, permutation],
        scene,
        ego,
    )
    assert torch.allclose(
        permuted["refined_scores"], direct["refined_scores"][:, permutation], atol=1e-6
    )
    assert torch.equal(
        permuted["top_k_mask"], direct["top_k_mask"][:, permutation]
    )


def test_scene_cross_attention_requires_current_scene_inputs():
    model = PublicBaseResidualRanker(
        ResidualScorerConfig(mode="scene_cross_attention", dropout=0.0)
    ).eval()
    with pytest.raises(ValueError, match="scene_features and ego_features"):
        model(
            torch.zeros(1, 64, 256),
            torch.zeros(1, 64, 8, 3),
            torch.zeros(1, 64, 6),
            torch.zeros(1, 64),
        )


@pytest.mark.parametrize("score_mode", ["factor_aggregate", "hybrid"])
def test_factor_score_modes_preserve_base_at_zero_initialization(score_mode: str):
    torch.manual_seed(23)
    model = PublicBaseResidualRanker(
        ResidualScorerConfig(score_mode=score_mode, dropout=0.0, top_k=8)
    ).eval()
    proposals = torch.randn(2, 16, 8, 3)
    features = torch.randn(2, 16, 256)
    logits = torch.randn(2, 16, 6)
    scores = torch.randn(2, 16)
    output = model(features, proposals, logits, scores)
    assert torch.equal(output["residual"], torch.zeros_like(scores))
    assert torch.equal(output["refined_scores"], scores)


def test_pdm_log_aggregate_matches_public_formula():
    logits = torch.randn(3, 7, 6)
    probabilities = logits.sigmoid()
    expected = (
        probabilities[..., 0].log()
        + probabilities[..., 1].log()
        + (
            5.0 * probabilities[..., 3]
            + 5.0 * probabilities[..., 4]
            + 2.0 * probabilities[..., 5]
        ).log()
    )
    assert torch.allclose(pdm_log_aggregate(logits), expected, atol=1e-6)


def test_binary_factor_loss_maps_partial_noc_and_ddc_to_failure():
    logits = torch.zeros(1, 1, 6)
    partial = torch.tensor([[[0.5, 1.0, 0.5, 1.0, 0.8, 1.0]]])
    failure = torch.tensor([[[0.0, 1.0, 0.0, 1.0, 0.8, 1.0]]])
    assert torch.equal(
        binary_factor_loss(logits, partial, 10.0),
        binary_factor_loss(logits, failure, 10.0),
    )


def test_binary_factor_loss_upweights_safety_violations():
    logits = torch.full((1, 1, 6), 4.0)
    safe = torch.ones(1, 1, 6)
    unsafe = safe.clone()
    unsafe[..., 0] = 0.0
    unweighted = binary_factor_loss(logits, unsafe, 1.0)
    weighted = binary_factor_loss(logits, unsafe, 20.0)
    assert weighted > unweighted
