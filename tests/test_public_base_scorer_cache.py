from pathlib import Path

import numpy as np
import pytest
import torch

from local_stage2.export_public_base_scorer_cache import (
    _first_missing_chunk,
    _partition_tokens,
)
from local_stage2.score_public_base_scorer_cache import _belongs_to_worker
from local_stage2.run_navtest_proposal_audit import _compare_prediction_banks
from local_stage2.summarize_navtest_scorer_campaigns import (
    _build_rows,
    _status,
)
from local_stage2.train_public_base_residual_scorer import (
    EvaluationOutputs,
    base_pairwise_loss,
    binary_factor_loss,
    evaluate_collected,
    evaluate_selection_sweep,
    relative_safety_targets,
)
from local_stage2.public_base_residual_scorer import (
    PublicBaseResidualRanker,
    ResidualScorerConfig,
    base_anchored_topk_indices,
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


def test_navtest_campaign_summary_requires_every_promoted_artifact():
    with pytest.raises(RuntimeError, match="coverage mismatch"):
        _build_rows({"promoted-sha": {}}, {})


@pytest.mark.parametrize(
    ("delta", "low", "high", "expected"),
    [
        (0.01, 0.001, 0.02, "TEST_POSITIVE_SIGNIFICANT"),
        (0.01, -0.001, 0.02, "TEST_POSITIVE_INCONCLUSIVE"),
        (-0.01, -0.02, 0.001, "TEST_NEGATIVE_INCONCLUSIVE"),
        (-0.01, -0.02, -0.001, "TEST_NEGATIVE_SIGNIFICANT"),
    ],
)
def test_navtest_campaign_status(delta, low, high, expected):
    assert _status(delta, low, high) == expected


def test_prediction_bank_parity_is_strict_and_ignores_feature_extras():
    reference = {
        "token": {
            "proposals": np.zeros((64, 8, 3), dtype=np.float32),
            "predicted_scores": np.zeros(64, dtype=np.float32),
        }
    }
    candidate = {
        "token": {
            **reference["token"],
            "candidate_features": np.zeros((64, 256), dtype=np.float32),
        }
    }
    assert _compare_prediction_banks(candidate, reference)["passes_1e_8"]
    candidate["token"]["predicted_scores"] = np.full(64, 2e-8, dtype=np.float32)
    assert not _compare_prediction_banks(candidate, reference)["passes_1e_8"]


def test_prediction_bank_smoke_may_be_reference_subset_only_when_explicit():
    item = {
        "proposals": np.zeros((64, 8, 3), dtype=np.float32),
        "predicted_scores": np.zeros(64, dtype=np.float32),
    }
    reference = {"a": item, "b": item}
    with pytest.raises(RuntimeError, match="token mismatch"):
        _compare_prediction_banks({"a": item}, reference)
    result = _compare_prediction_banks(
        {"a": item}, reference, allow_reference_superset=True
    )
    assert result["passes_1e_8"]
    assert result["scene_count"] == 1
    assert result["reference_scene_count"] == 2


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
    assert torch.allclose(
        permuted["relative_safety_logits"],
        direct["relative_safety_logits"][:, permutation],
        atol=1e-6,
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


def test_base_anchored_topk_is_exact_size_and_keeps_argmax_first_under_ties():
    scores = torch.tensor([[1.0, 1.0, 1.0, 0.5], [0.1, 0.4, 0.3, 0.4]])
    indices = base_anchored_topk_indices(scores, 3)
    assert indices[:, 0].tolist() == scores.argmax(dim=1).tolist()
    assert indices.shape == (2, 3)
    assert all(len(set(row)) == 3 for row in indices.tolist())


@pytest.mark.parametrize(
    "safety_gate_mode", ["factor_all", "composite", "relative_factor"]
)
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
    assert torch.equal(
        output["relative_safety_logits"],
        torch.zeros_like(output["relative_safety_logits"]),
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


def test_vectorized_deployment_sweep_matches_complete_evaluation():
    torch.manual_seed(29)
    scenes, candidates = 7, 6
    base_scores = torch.randn(scenes, candidates)
    top_k_mask = torch.zeros(scenes, candidates, dtype=torch.bool)
    top_k_mask.scatter_(1, base_scores.topk(4, dim=1).indices, True)
    outputs = EvaluationOutputs(
        base_scores=base_scores,
        residual=torch.randn(scenes, candidates) * 0.1,
        refined_factor_logits=torch.randn(scenes, candidates, 6),
        refined_composite_safety_logits=torch.randn(scenes, candidates),
        relative_safety_logits=torch.randn(scenes, candidates, 3),
        top_k_mask=top_k_mask,
        target_factors=torch.rand(scenes, candidates, 7),
    )
    logs = [f"log-{index // 2}" for index in range(scenes)]
    scales = (0.0, 0.35)
    penalties = (0.0, 0.01)
    settings = (
        (0.0, 1.0, False, "factor_all"),
        (0.7, 0.1, False, "composite"),
        (0.7, 1.0, False, "relative_factor"),
    )
    fast = evaluate_selection_sweep(
        outputs,
        logs,
        scales=scales,
        penalties=penalties,
        safety_settings=settings,
        device=torch.device("cpu"),
    )
    keyed = {
        (
            item["residual_scale"],
            item["switch_penalty"],
            item["safety_floor"],
            item["safety_relative_tolerance"],
            item["preserve_ddc"],
            item["safety_gate_mode"],
        ): item
        for item in fast
    }
    assert len(keyed) == len(scales) * len(penalties) * len(settings)
    for scale in scales:
        for penalty in penalties:
            for floor, tolerance, preserve_ddc, gate_mode in settings:
                expected = evaluate_collected(
                    outputs,
                    logs,
                    seed=0,
                    residual_scale=scale,
                    switch_penalty=penalty,
                    safety_floor=floor,
                    safety_relative_tolerance=tolerance,
                    preserve_ddc=preserve_ddc,
                    safety_gate_mode=gate_mode,
                    bootstrap_replicates=0,
                )
                actual = keyed[
                    (scale, penalty, floor, tolerance, preserve_ddc, gate_mode)
                ]
                for key in (
                    "model_selected_pdms",
                    "selected_pdms_delta",
                    "model_top1_regret",
                    "selection_switch_rate",
                    "pairwise_accuracy_delta_ge_0_02",
                ):
                    assert actual[key] == pytest.approx(expected[key], abs=1e-7)
                assert actual["improved_scene_count_delta_gt_0_01"] == expected[
                    "improved_scene_count_delta_gt_0_01"
                ]
                assert actual["degraded_scene_count_delta_lt_minus_0_01"] == expected[
                    "degraded_scene_count_delta_lt_minus_0_01"
                ]
                for factor_name, factor_delta in expected[
                    "selected_factor_delta"
                ].items():
                    assert actual["selected_factor_delta"][factor_name] == pytest.approx(
                        factor_delta, abs=1e-7
                    )


def test_relative_safety_targets_compare_against_base_per_factor():
    target = torch.ones(1, 3, 6)
    target[0, 0, 1] = 0.0
    target[0, 2, 0] = 0.0
    target[0, 2, 1] = 0.0
    scores = torch.tensor([[1.0, 0.9, 0.8]])
    assert relative_safety_targets(target, scores).tolist() == [
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]
    ]


def test_base_pairwise_loss_prefers_correct_switch_direction():
    target = torch.tensor([[0.5, 0.9, 0.2]])
    correct = torch.tensor([[0.0, 2.0, -2.0]])
    reversed_prediction = -correct
    assert base_pairwise_loss(correct, target, 0.02) < base_pairwise_loss(
        reversed_prediction,
        target,
        0.02,
    )
