import json
from pathlib import Path

import numpy as np
import pytest
import torch

from local_stage2.export_public_base_scorer_cache import (
    _first_missing_chunk,
    _partition_tokens,
)
from local_stage2.score_public_base_scorer_cache import _belongs_to_worker
from local_stage2.score_public_base_consequence_cache import (
    _base_topk_indices,
    _candidate_relative_box_state,
    _event_by_horizon,
)
from local_stage2.analyze_scorer_domain_shift import _auc
from local_stage2.run_navtest_proposal_audit import _compare_prediction_banks
from local_stage2.summarize_navtest_scorer_campaigns import (
    _build_rows,
    _status,
)
from local_stage2.evaluate_cached_navtest_scorers import _score_artifact
from local_stage2.build_effective_scorer_manifest import _artifact_record
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
from local_stage2.temporal_consequence_scorer import (
    TemporalConsequenceConfig,
    TemporalConsequenceRanker,
    temporal_trajectory_features,
)
from local_stage2.train_temporal_consequence_scorer import (
    assign_balanced_log_folds,
    load_full_data_cv_policy,
)
from local_stage2.summarize_temporal_consequence_cv import (
    _console_summary,
    summarize_cv,
)
from local_stage2.materialize_temporal_cv_policy import (
    materialize_common_policy_artifact,
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


def test_consequence_topk_is_stable_and_keeps_deployed_base_first():
    scores = np.asarray([[1.0, 1.0, 0.5, 1.0], [0.1, 0.3, 0.2, 0.4]])
    assert _base_topk_indices(scores, 3).tolist() == [[0, 1, 3], [3, 1, 2]]


def test_pdm_event_indices_map_to_cumulative_horizon_targets():
    events = np.asarray([[5.0, np.inf], [11.0, 40.0]])
    targets = _event_by_horizon(events, [5, 10, 20, 40])
    assert targets.shape == (2, 2, 4)
    assert targets[0, 0].tolist() == [True, True, True, True]
    assert targets[0, 1].tolist() == [False, False, False, False]
    assert targets[1, 0].tolist() == [False, False, True, True]
    assert targets[1, 1].tolist() == [False, False, False, True]


def test_actor_box_transform_is_candidate_relative_and_mask_aware():
    # Axis-aligned 4 m x 2 m rectangle centered at current-ego (10, 2).
    box = np.asarray([[8.0, 1.0], [12.0, 1.0], [12.0, 3.0], [8.0, 3.0]])
    corners = np.zeros((1, 1, 2, 4, 2), dtype=np.float64)
    corners[0, 0, 0] = box
    corners[0, 0, 1] = box
    valid = np.asarray([[[True, False]]])
    proposals = np.asarray([[[8.0, 2.0, np.pi / 2.0]]])
    state = _candidate_relative_box_state(corners, valid, proposals)
    assert state.shape == (1, 1, 2, 6)
    # The actor is 2 m to the candidate's left after rotating into its frame.
    assert state[0, 0, 0, 0] == pytest.approx(0.0, abs=1e-6)
    assert state[0, 0, 0, 1] == pytest.approx(-2.0, abs=1e-6)
    assert state[0, 0, 0, 2] == pytest.approx(4.0, abs=1e-6)
    assert state[0, 0, 0, 3] == pytest.approx(2.0, abs=1e-6)
    assert np.array_equal(state[0, 0, 1], np.zeros(6, dtype=np.float32))


def test_domain_auc_handles_perfect_order_and_score_ties():
    assert _auc(np.asarray([0, 0, 1, 1]), np.asarray([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert _auc(np.asarray([0, 1]), np.asarray([0.5, 0.5])) == 0.5


def _temporal_consequence_inputs(batch: int = 2, candidates: int = 7):
    return {
        "candidate_features": torch.randn(batch, candidates, 256),
        "proposals": torch.randn(batch, candidates, 8, 3),
        "base_factor_logits": torch.randn(batch, candidates, 6),
        "base_scores": torch.randn(batch, candidates),
        "scene_features": torch.randn(batch, 16, 256),
        "ego_features": torch.randn(batch, 1, 256),
    }


def test_temporal_consequence_scorer_is_base_exact_at_initialization():
    torch.manual_seed(31)
    model = TemporalConsequenceRanker(
        TemporalConsequenceConfig(dropout=0.0, top_k=4)
    ).eval()
    inputs = _temporal_consequence_inputs()
    output = model(**inputs)
    assert torch.equal(output["residual"], torch.zeros_like(inputs["base_scores"]))
    assert torch.equal(output["refined_scores"], inputs["base_scores"])
    assert torch.equal(
        output["selection_scores"].argmax(dim=1),
        inputs["base_scores"].argmax(dim=1),
    )
    assert output["risk_logits"].shape == (2, 7, 8, 2)
    assert output["actor_state"].shape == (2, 7, 8, 2, 6)


def test_temporal_consequence_scorer_is_candidate_permutation_equivariant():
    torch.manual_seed(37)
    model = TemporalConsequenceRanker(
        TemporalConsequenceConfig(dropout=0.0, top_k=5)
    ).eval()
    with torch.no_grad():
        model.utility_head[-1].weight.normal_(std=0.02)
    inputs = _temporal_consequence_inputs(candidates=7)
    permutation = torch.randperm(7)
    direct = model(**inputs)
    permuted_inputs = dict(inputs)
    for key in ("candidate_features", "proposals", "base_factor_logits", "base_scores"):
        permuted_inputs[key] = inputs[key][:, permutation]
    permuted = model(**permuted_inputs)
    for key in (
        "selection_scores",
        "risk_logits",
        "area_logits",
        "actor_valid_logits",
        "actor_state",
    ):
        assert torch.allclose(permuted[key], direct[key][:, permutation], atol=1e-6)


def test_temporal_trajectory_features_are_finite_across_heading_wrap():
    proposals = torch.zeros(1, 1, 8, 3)
    proposals[0, 0, :, 0] = torch.arange(1, 9)
    proposals[0, 0, :, 2] = torch.tensor(
        [3.13, -3.13, 3.12, -3.12, 3.11, -3.11, 3.10, -3.10]
    )
    features = temporal_trajectory_features(proposals)
    assert features.shape == (1, 1, 8, 8)
    assert torch.isfinite(features).all()


def test_balanced_log_folds_are_deterministic_disjoint_and_scene_balanced():
    logs = ["large"] * 10 + [f"small-{index}" for index in range(10) for _ in range(2)]
    first = assign_balanced_log_folds(logs, 3, 41)
    second = assign_balanced_log_folds(logs, 3, 41)
    assert first == second
    assert set(first) == set(logs)
    fold_scenes = [sum(first[name] == fold for name in logs) for fold in range(3)]
    assert max(fold_scenes) - min(fold_scenes) <= 2


def _synthetic_temporal_fold(fold_index: int, unsafe_gain: float):
    validation = f"log-{fold_index}"
    training = f"log-{1 - fold_index}"
    factor_keys = (
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "driving_direction_compliance",
        "ego_progress",
        "time_to_collision_within_bound",
        "comfort",
        "score",
    )

    def result(delta, scale, collision_delta=0.0):
        return {
            "scene_count": 10,
            "selected_pdms_delta": delta,
            "base_top1_regret": 0.10,
            "model_top1_regret": 0.10 - delta,
            "pairwise_accuracy_delta_ge_0_02": 0.75 + delta,
            "residual_scale": scale,
            "switch_penalty": 0.0,
            "safety_floor": 0.0,
            "safety_relative_tolerance": 1.0,
            "selected_factor_delta": {
                key: collision_delta if key == "no_at_fault_collisions" else 0.0
                for key in factor_keys
            },
        }

    return {
        "metadata": {
            "fold": {
                "num_folds": 2,
                "fold_seed": 7,
                "fold_index": fold_index,
                "train_logs": [training],
                "validation_logs": [validation],
                "train_scene_count": 10,
                "validation_scene_count": 10,
            }
        },
        "history": [
            {"epoch": 0, "validation": result(0.01, 1.0)},
            {"epoch": 1, "validation": result(0.02, 1.0)},
        ],
        "deployment_sweep": [
            result(0.01, 0.5),
            result(unsafe_gain, 1.0, collision_delta=-0.01),
        ],
    }


def test_temporal_cv_summary_uses_one_safe_policy_across_disjoint_folds():
    summary = summarize_cv(
        [_synthetic_temporal_fold(0, 0.03), _synthetic_temporal_fold(1, 0.04)]
    )
    assert summary["fold_audit"]["complete"]
    assert summary["fold_audit"]["validation_log_count"] == 2
    assert summary["common_epoch"]["epoch"] == 1
    assert summary["common_deployment"]["residual_scale"] == 0.5
    assert summary["common_deployment"]["safety_nonregressing"]


def test_temporal_cv_console_summary_omits_large_search_grids():
    summary = summarize_cv(
        [_synthetic_temporal_fold(0, 0.03), _synthetic_temporal_fold(1, 0.04)]
    )
    compact = _console_summary(summary)
    assert "epoch_results" not in compact
    assert "deployment_results" not in compact
    assert compact["common_epoch"] == summary["common_epoch"]
    assert compact["common_deployment"] == summary["common_deployment"]
    assert compact["output_contains_full_grids"] is True


def test_temporal_cv_common_policy_changes_only_deployment_config(tmp_path: Path):
    config = TemporalConsequenceConfig(hidden_dim=64, trajectory_dim=32, temporal_layers=1)
    model = TemporalConsequenceRanker(config)
    source = tmp_path / "fold_0" / "best.pt"
    source.parent.mkdir()
    sweep = [
        {
            "residual_scale": 0.2,
            "switch_penalty": 0.02,
            "safety_floor": 0.95,
            "safety_relative_tolerance": 0.1,
            "selected_pdms_delta": 0.001,
            "selected_pdms_delta_log_bootstrap_95ci": [0.0001, 0.002],
            "selected_factor_delta": {
                "no_at_fault_collisions": 0.0,
                "drivable_area_compliance": 0.0,
                "time_to_collision_within_bound": 0.0,
            },
        }
    ]
    torch.save(
        {
            "artifact_type": "episode_drive_temporal_consequence_scorer_v1",
            "model_config": config.__dict__,
            "model_state_dict": model.state_dict(),
            "metadata": {
                "deployment_sweep": sweep,
                "future_inputs_used": False,
                "official_scores_used_at_inference": False,
            },
        },
        source,
    )
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "robust_deployment_available": True,
                "fold_audit": {"complete": True, "observed_fold_count": 2},
                "common_deployment": sweep[0],
            }
        )
    )
    output = tmp_path / "derived" / "common.pt"
    materialize_common_policy_artifact(source, output, summary_path)
    derived = torch.load(output, map_location="cpu")
    assert derived["model_config"]["inference_scale"] == 0.2
    assert derived["model_config"]["switch_penalty"] == 0.02
    assert derived["model_config"]["safety_floor"] == 0.95
    assert derived["model_config"]["safety_relative_tolerance"] == 0.1
    for key, value in model.state_dict().items():
        assert torch.equal(derived["model_state_dict"][key], value)
    assert derived["metadata"]["validation"]["selected_pdms_delta"] == 0.001
    assert not derived["metadata"]["common_cv_policy"][
        "navtest_used_for_policy_selection"
    ]


def test_full_data_temporal_training_uses_only_complete_cv_policy(tmp_path: Path):
    path = tmp_path / "cv.json"
    payload = {
        "fold_audit": {
            "complete": True,
            "declared_num_folds": 5,
            "fold_seed": 17,
        },
        "robust_deployment_available": True,
        "common_epoch": {"epoch": 9},
        "common_deployment": {
            "residual_scale": 0.2,
            "switch_penalty": 0.02,
            "safety_floor": 0.95,
            "safety_relative_tolerance": 0.1,
        },
    }
    path.write_text(json.dumps(payload))
    epoch, policy, loaded = load_full_data_cv_policy(path)
    assert epoch == 9
    assert policy["residual_scale"] == 0.2
    assert loaded["fold_audit"]["declared_num_folds"] == 5
    payload["fold_audit"]["complete"] = False
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="complete cross-validation"):
        load_full_data_cv_policy(path)


def test_cached_navtest_evaluator_accepts_temporal_consequence_artifact(tmp_path: Path):
    torch.manual_seed(43)
    config = TemporalConsequenceConfig(
        hidden_dim=64,
        trajectory_dim=32,
        temporal_layers=1,
        temporal_heads=8,
        scene_layers=1,
        scene_heads=8,
        dropout=0.0,
        top_k=16,
    )
    model = TemporalConsequenceRanker(config).eval()
    artifact = {
        "artifact_type": "episode_drive_temporal_consequence_scorer_v1",
        "artifact_version": 1,
        "model_config": config.__dict__,
        "model_state_dict": model.state_dict(),
        "metadata": {},
    }
    path = tmp_path / "temporal.pt"
    torch.save(artifact, path)
    tokens = ["a", "b"]
    cache = {
        token: {
            "candidate_features": np.zeros((64, 256), dtype=np.float32),
            "proposals": np.zeros((64, 8, 3), dtype=np.float32),
            "base_factor_logits": np.zeros((64, 6), dtype=np.float32),
            "predicted_scores": np.linspace(0.0, 1.0, 64, dtype=np.float32),
            "scene_features": np.zeros((16, 256), dtype=np.float32),
            "ego_features": np.zeros((1, 256), dtype=np.float32),
        }
        for token in tokens
    }
    selected, scores, payload = _score_artifact(
        path,
        cache,
        tokens,
        device=torch.device("cpu"),
        batch_size=2,
    )
    assert selected.tolist() == [63, 63]
    base = np.stack([cache[token]["predicted_scores"] for token in tokens])
    assert np.array_equal(scores[:, -16:], base[:, -16:])
    assert np.all(scores[:, :-16] < base[:, :-16])
    assert payload["artifact_type"] == artifact["artifact_type"]


def test_positive_mean_promotion_sends_inconclusive_gain_to_navtest(tmp_path: Path):
    path = tmp_path / "artifact.pt"
    torch.save(
        {
            "artifact_type": "episode_drive_temporal_consequence_scorer_v1",
            "model_config": {},
            "metadata": {
                "validation": {
                    "selected_pdms_delta": 0.001,
                    "selected_pdms_delta_log_bootstrap_95ci": [-0.001, 0.003],
                    "selected_factor_delta": {
                        "no_at_fault_collisions": 0.0,
                        "drivable_area_compliance": 0.0,
                        "time_to_collision_within_bound": 0.0,
                    },
                },
                "future_inputs_used": False,
                "official_scores_used_at_inference": False,
            },
        },
        path,
    )
    assert _artifact_record(path, 0.0, "positive_mean")["promoted"]
    assert not _artifact_record(path, 0.0, "positive_ci")["promoted"]


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
