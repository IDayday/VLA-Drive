from __future__ import annotations

from dataclasses import asdict
import inspect
import json
from types import SimpleNamespace

import pytest
import numpy as np
import torch

from navsim.agents.EpisodeDrive.score_module.independent_ranker import (
    ConservativeReferenceConfig,
    ConservativeReferenceHead,
    EGO_HALF_LENGTH_M,
    EGO_HALF_WIDTH_M,
    EGO_REAR_AXLE_TO_CENTER_M,
    FACTOR_KEYS,
    IndependentConservativeReferenceRanker,
    IndependentProposalRanker,
    IndependentRankerConfig,
    ProposalTrajectoryEncoder,
    SharedFutureCandidateRelabeler,
    assert_current_observation_only,
    candidate_relative_consequence_loss,
    conservative_reference_selection_scores,
    current_actor_auxiliary_loss,
    episode_drive_factor_loss,
    factor_prediction_loss,
    masked_pinball_quantile_loss,
    normalize_current_actor_targets,
    pdms_factor_log_utility,
    shared_future_auxiliary_loss,
    top_heavy_listwise_loss,
    top_regret_rank_loss,
    weighted_pairwise_rank_loss,
)
from navsim.agents.EpisodeDrive.score_module.m0_private_residual_ranker import (
    M0PrivateResidualConfig,
    M0PrivateResidualRanker,
)
from navsim.agents.EpisodeDrive.score_module.drivor_ranker import (
    DrivORInitializedProposalRanker,
    DrivORRankerConfig,
    DrivORReferenceGateRanker,
)
from local_stage2.export_private_visual_replay import pool_visual_tokens
from local_stage2.evaluate_external_proposal_scores import (
    _cluster_bootstrap as external_score_cluster_bootstrap,
    _pairwise_accuracy as external_score_pairwise_accuracy,
    _physical_log_name as external_score_physical_log_name,
)
from local_stage2.evaluate_independent_scorer_replay import (
    collect_base_shortlist_metrics,
    evaluate_base_shortlist_reranking,
)
from local_stage2.evaluate_m0_private_residual_navtest_cache import (
    _required_feature,
)
from local_stage2.evaluate_shared_future_prediction import (
    _baseline_states,
    _decode_predicted_state,
    _presence_metrics,
    _state_errors,
)
from local_stage2.independent_scorer_agent import (
    IndependentShortlistScorerAgent,
    build_independent_shortlist_artifact,
)
from local_stage2.m0_native_private_scorer_agent import (
    M0NativePrivateFeatureBuilder,
    M0NativePrivateScorerAgent,
    build_m0_native_private_scorer_artifact,
)
from local_stage2.train_independent_scorer import (
    ReplaySource,
    assign_balanced_physical_log_folds,
    load_current_actor_target_table,
    load_replay_sources,
    physical_log_name,
    validate_all_log_refit_provenance,
)
from local_stage2.train_m0_private_residual_scorer import (
    ResidualReplayDataset,
    build_m0_training_sampler,
    compute_residual_training_loss,
    evaluate_residual_predictions_by_source,
    load_shared_future_target_table,
    validate_m0_all_log_refit_provenance,
)
from local_stage2.calibrate_m0_private_residual_policy import (
    balanced_calibration_split,
    policy_selection_indices,
)
from local_stage2.train_drivor_reference_gate import (
    compute_gate_training_loss,
)
from local_stage2.train_conservative_reference_scorer import (
    compute_reference_training_loss,
)
from local_stage2.train_drivor_initialized_ranker import (
    direct_score_regression_loss,
)
from local_stage2.build_drivor_promotion_manifest import (
    _gate_record,
    _ranker_record,
)
from local_stage2.build_m0_native_promotion_manifest import (
    _record as m0_native_promotion_record,
)
from local_stage2.build_full_current_actor_target_cache import (
    aggregate as aggregate_full_current_actor_targets,
)
from local_stage2.build_m0_scorer_cv_folds import (
    assign_risk_stratified_log_folds,
)
from local_stage2.summarize_m0_residual_cv import (
    aggregate_fixed_epoch_folds,
)
from local_stage2.analyze_policy_shortlist_headroom import _parse_top_k
from local_stage2.audit_drivor_representation_dependence import (
    _cross_log_derangement,
)
from local_stage2.audit_independent_representation_dependence import (
    _mode_mask,
    _score_output,
)


def _small_config(**overrides) -> IndependentRankerConfig:
    values = dict(
        observation_dim=32,
        model_dim=32,
        status_dim=8,
        num_poses=8,
        num_heads=4,
        num_private_layers=1,
        num_trajectory_layers=1,
        num_candidate_layers=1,
        num_fine_layers=1,
        dynamic_queries=3,
        static_queries=2,
        signal_queries=2,
        global_queries=1,
        fine_top_k=3,
        dropout=0.0,
    )
    values.update(overrides)
    return IndependentRankerConfig(**values)


def _inputs(batch_size: int = 2, candidates: int = 6):
    generator = torch.Generator().manual_seed(17)
    observations = torch.randn(batch_size, 13, 32, generator=generator)
    status = torch.randn(batch_size, 8, generator=generator)
    increments = torch.randn(
        batch_size, candidates, 8, 2, generator=generator
    ) * 0.15
    xy = increments.cumsum(dim=-2)
    heading = torch.atan2(increments[..., 1], increments[..., 0]).unsqueeze(-1)
    proposals = torch.cat((xy, heading), dim=-1)
    return observations, status, proposals


def _small_drivor_config(**overrides) -> DrivORRankerConfig:
    values = dict(
        model_dim=32,
        feedforward_dim=64,
        status_dim=8,
        num_poses=8,
        scorer_layers=2,
        attention_heads=4,
        projection_dropout=0.0,
        drop_path=0.0,
    )
    values.update(overrides)
    return DrivORRankerConfig(**values)


def _risk_sampler_inputs() -> SimpleNamespace:
    base_scores = torch.zeros(4, 64)
    base_scores[:, 0] = 1.0
    base_scores[:, 1] = 0.9
    target_factors = torch.ones(4, 64, 7)
    target_factors[..., -1] = 0.5
    # Only the first scene contains a Base-top-2 safe/unsafe choice and useful
    # score headroom. The sampler may use this training target for sampling,
    # but it never becomes a model-forward feature.
    target_factors[0, 1, 0] = 0.0
    target_factors[0, 1, -1] = 0.8
    return SimpleNamespace(
        physical_logs=["log_a", "log_a", "log_b", "log_b"],
        base_scores_for_evaluation=base_scores,
        target_factors=target_factors,
    )


def test_risk_balanced_sampler_emphasizes_hard_scene_within_log_only() -> None:
    sampler, lineage = build_m0_training_sampler(
        _risk_sampler_inputs(),
        [0, 1, 2, 3],
        seed=19,
        mode="risk_balanced",
        top_k=2,
        risk_scene_max_multiplier=4.0,
    )
    weights = sampler.weights
    assert weights[0] > weights[1]
    assert weights[:2].sum().item() == pytest.approx(weights[2:].sum().item())
    assert lineage["training_only_targets_used_for_sampling"] is True
    assert lineage["model_forward_receives_sampling_targets"] is False
    assert lineage["per_physical_log_total_weight_equalized"] is True
    assert lineage["risk_contrast_fraction"] == pytest.approx(0.25)


def test_risk_balanced_sampler_is_deterministic() -> None:
    kwargs = dict(
        seed=23,
        mode="risk_balanced",
        top_k=2,
        risk_scene_max_multiplier=4.0,
    )
    first, _ = build_m0_training_sampler(
        _risk_sampler_inputs(), [0, 1, 2, 3], **kwargs
    )
    second, _ = build_m0_training_sampler(
        _risk_sampler_inputs(), [0, 1, 2, 3], **kwargs
    )
    assert list(iter(first)) == list(iter(second))


def test_log_balanced_sampler_remains_the_default_non_target_path() -> None:
    sampler, lineage = build_m0_training_sampler(
        _risk_sampler_inputs(),
        [0, 1, 2, 3],
        seed=29,
        mode="log_balanced",
        top_k=2,
        risk_scene_max_multiplier=4.0,
    )
    assert sampler.weights.tolist() == pytest.approx([0.5, 0.5, 0.5, 0.5])
    assert lineage["training_only_targets_used_for_sampling"] is False
    assert lineage["per_physical_log_total_weight_equalized"] is True


def test_risk_stratified_cv_folds_are_deterministic_and_log_disjoint() -> None:
    stats = {
        f"log_{index:02d}": {
            "scene_count": 80.0 + 7.0 * index,
            "risk_scene_count": 10.0 + 3.0 * (index % 5),
            "unsafe_candidate_count": 100.0 + 17.0 * (index % 7),
            "score_span_sum": 8.0 + float(index % 4),
        }
        for index in range(20)
    }
    first = assign_risk_stratified_log_folds(stats, num_folds=5, seed=31)
    second = assign_risk_stratified_log_folds(stats, num_folds=5, seed=31)
    assert first == second
    assert set(first) == set(stats)
    assert set(first.values()) == set(range(5))
    fold_scenes = [
        sum(
            stats[name]["scene_count"]
            for name, assigned_fold in first.items()
            if assigned_fold == fold
        )
        for fold in range(5)
    ]
    assert max(fold_scenes) - min(fold_scenes) <= max(
        value["scene_count"] for value in stats.values()
    )


def test_fixed_epoch_cv_gate_requires_every_fold_positive_lower_bound() -> None:
    passing = [
        {
            "scene_count": 100 + index,
            "selected_delta": 0.01 + 0.001 * index,
            "bootstrap_95ci": [0.002 + 0.001 * index, 0.02],
        }
        for index in range(5)
    ]
    accepted = aggregate_fixed_epoch_folds(passing)
    assert accepted["robust_refit_gate_passed"] is True
    assert accepted["validation_scene_count"] == 510
    failing = [dict(row) for row in passing]
    failing[3] = failing[3] | {"bootstrap_95ci": [-0.0001, 0.02]}
    rejected = aggregate_fixed_epoch_folds(failing)
    assert rejected["all_fold_point_deltas_positive"] is True
    assert rejected["all_fold_bootstrap_lowers_positive"] is False
    assert rejected["robust_refit_gate_passed"] is False


def test_cross_log_derangement_never_reuses_physical_log() -> None:
    logs = ["large"] * 7 + ["medium"] * 4 + ["small"] * 3
    permutation = _cross_log_derangement(logs, seed=19)
    assert sorted(permutation.tolist()) == list(range(len(logs)))
    assert all(logs[row] != logs[donor] for row, donor in enumerate(permutation))


def test_independent_representation_audit_uses_donor_mask_only_for_scene_shuffle() -> None:
    source = torch.tensor([[True, False], [False, True]])
    donor = ~source
    assert torch.equal(_mode_mask("correct", source, donor), source)
    assert torch.equal(_mode_mask("scene_zero", source, donor), source)
    assert torch.equal(
        _mode_mask("scene_cross_log_shuffle", source, donor), donor
    )
    assert torch.equal(
        _mode_mask("scene_and_status_cross_log_shuffle", source, donor), donor
    )


def test_independent_representation_audit_score_modes() -> None:
    factor_logits = torch.randn(2, 3, len(FACTOR_KEYS))
    output = {
        "utility": torch.randn(2, 3),
        "coarse_utility": torch.randn(2, 3),
        "factor_logits": factor_logits,
    }
    assert torch.equal(_score_output(output, "direct"), output["utility"])
    assert torch.equal(_score_output(output, "coarse"), output["coarse_utility"])
    torch.testing.assert_close(
        _score_output(output, "factor"), pdms_factor_log_utility(factor_logits)
    )


def test_forward_signature_has_no_released_or_future_score_inputs() -> None:
    parameters = set(inspect.signature(IndependentProposalRanker.forward).parameters)
    assert parameters == {
        "self",
        "observation_tokens",
        "status_feature",
        "proposals",
        "observation_valid_mask",
    }
    assert not any("base" in name or "pdm" in name or "future" in name for name in parameters)


def test_m0_private_residual_forward_is_current_inference_only() -> None:
    parameters = set(
        inspect.signature(M0PrivateResidualRanker.forward).parameters
    )
    assert parameters == {
        "self",
        "observation_tokens",
        "status_feature",
        "proposals",
        "base_factor_logits",
        "base_scores",
        "observation_valid_mask",
        "m0_scene_features",
        "m0_ego_features",
        "m0_candidate_features",
    }
    assert not any(
        "future" in name or "official" in name or "metric" in name
        for name in parameters
    )


def test_m0_private_residual_navtest_requires_exact_base_feature_shape() -> None:
    entry = {
        "predicted_scores": np.zeros(64, dtype=np.float32),
    }
    scores = _required_feature(entry, "predicted_scores", (64,))
    assert scores.dtype == np.float32
    with pytest.raises(RuntimeError, match="unexpected predicted_scores shape"):
        _required_feature(entry, "predicted_scores", (32,))
    with pytest.raises(RuntimeError, match="lacks base_factor_logits"):
        _required_feature(entry, "base_factor_logits", (64, 6))


def test_m0_native_promotion_uses_selection_source_log_ci(tmp_path) -> None:
    artifact_path = tmp_path / "best_factor_independent_scorer.pt"
    torch.save(
        {
            "architecture": "IndependentProposalRanker",
            "checkpoint_selection_source": "public_base",
            "epoch": 4,
            "validation": {
                "scene_count": 99,
                "physical_log_count": 9,
                "factor_selected_pdms": 0.99,
                "base_selected_pdms": 0.90,
                "factor_selected_delta": 0.09,
                "factor_selected_delta_log_bootstrap_95ci": [0.08, 0.10],
            },
            "validation_by_source": {
                "public_base": {
                    "scene_count": 80,
                    "physical_log_count": 8,
                    "factor_selected_pdms": 0.91,
                    "base_selected_pdms": 0.90,
                    "factor_selected_delta": 0.01,
                    "factor_selected_delta_log_bootstrap_95ci": [0.002, 0.018],
                }
            },
        },
        artifact_path,
    )
    record = m0_native_promotion_record(
        run_dir=tmp_path,
        path=artifact_path,
        architecture="IndependentProposalRanker",
        score_mode="factor",
        selected_key="factor_selected_pdms",
        delta_key="factor_selected_delta",
        interval_key="factor_selected_delta_log_bootstrap_95ci",
        minimum_ci_lower=0.0,
    )
    assert record["promoted"] is True
    assert record["validation_scene_count"] == 80
    assert record["validation_delta"] == pytest.approx(0.01)


def test_m0_native_private_feature_builder_uses_four_current_views(tmp_path) -> None:
    camera_values = {}
    for name in ("cam_f0", "cam_l0", "cam_r0", "cam_b0"):
        path = tmp_path / f"{name}.jpg"
        path.write_bytes(b"test")
        camera_values[name] = SimpleNamespace(image=path)
    current = SimpleNamespace(
        ego_pose=np.asarray([1.0, 2.0, 0.3], dtype=np.float32),
        ego_velocity=np.asarray([4.0, 5.0], dtype=np.float32),
        ego_acceleration=np.asarray([0.1, 0.2], dtype=np.float32),
        driving_command=np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    )
    agent_input = SimpleNamespace(
        cameras=[SimpleNamespace(**camera_values)],
        ego_statuses=[current, current, current, current],
    )
    features = M0NativePrivateFeatureBuilder().compute_features(agent_input)
    assert "image_path_tensor" in features
    assert tuple(features["m0_private_status_feature"].shape) == (11,)
    for name in camera_values:
        assert f"m0_private_{name}_path_tensor" in features


def test_m0_native_private_artifact_packages_exact_online_class(tmp_path) -> None:
    base_checkpoint = tmp_path / "base.ckpt"
    base_checkpoint.write_bytes(b"immutable released M0")
    import hashlib

    base_sha = hashlib.sha256(base_checkpoint.read_bytes()).hexdigest()
    cache_root = tmp_path / "cache"
    shard = cache_root / "m0_multiview_shard_000-of-001"
    shard.mkdir(parents=True)
    (shard / "manifest.json").write_text(
        json.dumps(
            {
                "shard_count": 1,
                "m0_checkpoint_sha256": base_sha,
                "camera_names": ["cam_f0", "cam_l0", "cam_r0", "cam_b0"],
                "max_dynamic_tiles": 4,
                "max_crops_per_camera": 5,
                "pool_grid": [2, 2],
                "visual_token_count": 80,
                "visual_width": 32,
                "visual_model_wrapper_chain": ["fake.M0"],
                "current_observation_only": True,
                "future_or_evaluator_input": False,
                "official_score_or_factor_input": False,
                "proposal_input": False,
            }
        )
    )
    config = _small_config(
        observation_dim=32,
        max_observation_tokens=80,
        status_dim=11,
    )
    model = IndependentProposalRanker(config)
    source = {
        "architecture": "IndependentProposalRanker",
        "selection_mode": "factor",
        "state_dict": model.state_dict(),
        "model_config": asdict(config),
        "epoch": 3,
        "validation": {"factor_selected_pdms": 0.91},
    }
    source_path = tmp_path / "ranker.pt"
    torch.save(source, source_path)
    payload = build_m0_native_private_scorer_artifact(
        source,
        source_path=source_path,
        base_checkpoint=base_checkpoint,
        private_observation_root=cache_root,
        shortlist_size=64,
    )
    assert payload["artifact_type"] == M0NativePrivateScorerAgent.ARTIFACT_TYPE
    assert payload["scorer_architecture"] == "IndependentProposalRanker"
    assert payload["score_mode"] == "factor"
    assert payload["future_or_evaluator_input"] is False
    assert payload["official_score_input"] is False


def test_m0_scene_token_residual_packages_without_second_visual_branch(
    tmp_path,
) -> None:
    base_checkpoint = tmp_path / "no_vqa.ckpt"
    base_checkpoint.write_bytes(b"immutable no-vqa M0")
    import hashlib

    base_sha = hashlib.sha256(base_checkpoint.read_bytes()).hexdigest()
    private_config = _small_config(
        observation_dim=256,
        max_observation_tokens=16,
        status_dim=256,
        fine_top_k=6,
    )
    residual_config = M0PrivateResidualConfig(
        hidden_dim=private_config.model_dim,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        top_k=6,
        score_mode="hybrid",
        m0_candidate_fusion=True,
        m0_candidate_dim=256,
        conservative_reference=True,
        reference_hidden_dim=64,
        reference_layers=1,
        gain_quantile_index=0,
    )
    model = M0PrivateResidualRanker(private_config, residual_config)
    source = {
        "architecture": "M0PrivateResidualRanker",
        "state_dict": model.state_dict(),
        "private_config": asdict(private_config),
        "residual_config": asdict(residual_config),
        "epoch": 2,
        "checkpoint_selection_source": "no_vqa_e35",
        "validation": {"selected_pdms": 0.92},
        "inference_input_schema": (
            "m0_current_scene_tokens",
            "m0_current_context_feature",
            "m0_released_candidate_features",
            "m0_proposals",
            "m0_base_factor_logits",
            "m0_base_scores",
        ),
        "fold_manifest": {
            "scorer_private_observation_source": (
                "source_checkpoint_current_scene_tokens"
            ),
            "source_lineage": [
                {
                    "name": "no_vqa_e35",
                    "checkpoint_sha256": base_sha,
                },
                {
                    "name": "training_only_current_actor_supervision",
                    "checkpoint_sha256": None,
                    "available_as_model_input_at_inference": False,
                },
            ],
        },
    }
    source_path = tmp_path / "scene_token_ranker.pt"
    torch.save(source, source_path)
    payload = build_m0_native_private_scorer_artifact(
        source,
        source_path=source_path,
        base_checkpoint=base_checkpoint,
        private_observation_root=None,
        shortlist_size=64,
    )
    assert payload["private_observation_source"] == (
        "source_checkpoint_current_scene_tokens"
    )
    assert payload["private_vision_config"] is None
    assert payload["artifact_version"] == 3
    assert payload["inference_input_schema"][:2] == (
        "m0_current_scene_tokens",
        "m0_current_context_feature",
    )
    assert "m0_released_candidate_features" in payload["inference_input_schema"]
    assert payload["residual_config"]["conservative_reference"] is True
    assert payload["residual_config"]["gain_quantile_index"] == 0
    assert payload["external_model_representation_or_weight_used"] is False


def test_m0_candidate_feature_fusion_preserves_base_and_uses_current_state() -> None:
    torch.manual_seed(141)
    private_config = _small_config(
        fine_top_k=6,
        trajectory_observation_attention=True,
    )
    model = M0PrivateResidualRanker(
        private_config,
        M0PrivateResidualConfig(
            hidden_dim=private_config.model_dim,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
            top_k=6,
            score_mode="hybrid",
            m0_candidate_fusion=True,
            m0_candidate_dim=32,
        ),
    ).eval()
    observation, status, proposals = _inputs()
    factor_logits = torch.randn(2, 6, len(FACTOR_KEYS))
    base_scores = torch.randn(2, 6)
    candidate_features = torch.randn(2, 6, 32)
    with pytest.raises(ValueError, match="requires released candidate features"):
        model(observation, status, proposals, factor_logits, base_scores)
    with torch.no_grad():
        output = model(
            observation,
            status,
            proposals,
            factor_logits,
            base_scores,
            m0_candidate_features=candidate_features,
        )
    torch.testing.assert_close(output["selection_scores"], base_scores, rtol=0, atol=0)
    assert output["trajectory_observation_token"].shape == (2, 6, 32)
    assert output["trajectory_observation_gate"].item() == 0.0


def test_m0_candidate_only_ranking_excludes_private_observation() -> None:
    torch.manual_seed(142)
    private_config = _small_config(fine_top_k=6)
    model = M0PrivateResidualRanker(
        private_config,
        M0PrivateResidualConfig(
            hidden_dim=private_config.model_dim,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
            top_k=6,
            score_mode="hybrid",
            m0_candidate_fusion=True,
            m0_candidate_dim=32,
            m0_candidate_only=True,
        ),
    ).eval()
    # Correction heads are zero-initialized for exact Base identity. Make them
    # observable so this test can prove the private stream is disconnected.
    with torch.no_grad():
        model.utility_delta_head[-1].weight.normal_()
        model.factor_delta_head[-1].weight.normal_()
    observation, status, proposals = _inputs()
    factor_logits = torch.randn(2, 6, len(FACTOR_KEYS))
    base_scores = torch.randn(2, 6)
    candidate_features = torch.randn(2, 6, 32)
    with torch.no_grad():
        first = model(
            observation,
            status,
            proposals,
            factor_logits,
            base_scores,
            m0_candidate_features=candidate_features,
        )["selection_scores"]
        second = model(
            observation + 100.0,
            status - 100.0,
            proposals,
            factor_logits,
            base_scores,
            m0_candidate_features=candidate_features,
        )["selection_scores"]
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_m0_candidate_only_requires_candidate_fusion() -> None:
    with pytest.raises(ValueError, match="requires candidate fusion"):
        M0PrivateResidualConfig(m0_candidate_only=True)


def test_m0_private_residual_zero_init_exactly_preserves_base() -> None:
    torch.manual_seed(101)
    private_config = _small_config(fine_top_k=6)
    model = M0PrivateResidualRanker(
        private_config,
        M0PrivateResidualConfig(
            hidden_dim=private_config.model_dim,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
            top_k=6,
            score_mode="hybrid",
        ),
    ).eval()
    observation, status, proposals = _inputs()
    factor_logits = torch.randn(2, 6, len(FACTOR_KEYS))
    base_scores = torch.randn(2, 6)
    with torch.no_grad():
        output = model(
            observation,
            status,
            proposals,
            factor_logits,
            base_scores,
        )
    torch.testing.assert_close(output["refined_scores"], base_scores, rtol=0, atol=0)
    torch.testing.assert_close(output["selection_scores"], base_scores, rtol=0, atol=0)
    torch.testing.assert_close(
        output["refined_factor_logits"], factor_logits, rtol=0, atol=0
    )
    assert torch.equal(
        output["selection_scores"].argmax(dim=1),
        base_scores.argmax(dim=1),
    )


def test_m0_context_fusion_zero_init_exactly_preserves_base() -> None:
    torch.manual_seed(120)
    private_config = _small_config(fine_top_k=6)
    model = M0PrivateResidualRanker(
        private_config,
        M0PrivateResidualConfig(
            hidden_dim=private_config.model_dim,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
            top_k=6,
            score_mode="hybrid",
            m0_context_fusion=True,
        ),
    ).eval()
    observation, status, proposals = _inputs()
    factor_logits = torch.randn(2, 6, len(FACTOR_KEYS))
    base_scores = torch.randn(2, 6)
    with torch.no_grad():
        output = model(
            observation,
            status,
            proposals,
            factor_logits,
            base_scores,
            m0_scene_features=torch.randn(2, 16, 256),
            m0_ego_features=torch.randn(2, 1, 256),
        )
    torch.testing.assert_close(output["selection_scores"], base_scores, rtol=0, atol=0)
    torch.testing.assert_close(
        output["refined_factor_logits"], factor_logits, rtol=0, atol=0
    )
    assert float(output["m0_context_fusion_gate"]) == 0.0


def test_m0_context_fusion_is_candidate_permutation_equivariant() -> None:
    torch.manual_seed(121)
    private_config = _small_config(fine_top_k=6)
    model = M0PrivateResidualRanker(
        private_config,
        M0PrivateResidualConfig(
            hidden_dim=private_config.model_dim,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
            top_k=6,
            m0_context_fusion=True,
        ),
    ).eval()
    with torch.no_grad():
        model.m0_context_gate.fill_(0.5)
        model.factor_delta_head[-1].weight.normal_(std=0.02)
        model.utility_delta_head[-1].weight.normal_(std=0.02)
    observation, status, proposals = _inputs()
    factor_logits = torch.randn(2, 6, len(FACTOR_KEYS))
    base_scores = torch.randn(2, 6)
    scene = torch.randn(2, 16, 256)
    ego = torch.randn(2, 1, 256)
    permutation = torch.tensor([4, 0, 5, 2, 1, 3])
    inverse = torch.argsort(permutation)
    with torch.no_grad():
        reference = model(
            observation,
            status,
            proposals,
            factor_logits,
            base_scores,
            m0_scene_features=scene,
            m0_ego_features=ego,
        )
        permuted = model(
            observation,
            status,
            proposals[:, permutation],
            factor_logits[:, permutation],
            base_scores[:, permutation],
            m0_scene_features=scene,
            m0_ego_features=ego,
        )
    for key in ("selection_scores", "refined_factor_logits"):
        torch.testing.assert_close(
            reference[key], permuted[key][:, inverse], rtol=1e-5, atol=1e-6
        )


def test_m0_private_residual_is_candidate_permutation_equivariant() -> None:
    torch.manual_seed(102)
    private_config = _small_config(fine_top_k=6)
    model = M0PrivateResidualRanker(
        private_config,
        M0PrivateResidualConfig(
            hidden_dim=private_config.model_dim,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
            top_k=6,
        ),
    ).eval()
    observation, status, proposals = _inputs()
    factor_logits = torch.randn(2, 6, len(FACTOR_KEYS))
    base_scores = torch.randn(2, 6)
    permutation = torch.tensor([4, 0, 5, 2, 1, 3])
    inverse = torch.argsort(permutation)
    with torch.no_grad():
        reference = model(
            observation,
            status,
            proposals,
            factor_logits,
            base_scores,
        )
        permuted = model(
            observation,
            status,
            proposals[:, permutation],
            factor_logits[:, permutation],
            base_scores[:, permutation],
        )
    for key in (
        "selection_scores",
        "refined_scores",
        "refined_factor_logits",
        "relative_safety_logits",
        "private_candidate_features",
    ):
        torch.testing.assert_close(
            reference[key], permuted[key][:, inverse], rtol=1e-5, atol=1e-6
        )


def test_m0_private_residual_training_loss_is_finite() -> None:
    torch.manual_seed(103)
    private_config = _small_config(fine_top_k=6)
    model = M0PrivateResidualRanker(
        private_config,
        M0PrivateResidualConfig(
            hidden_dim=private_config.model_dim,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
            top_k=6,
        ),
    )
    observation, status, proposals = _inputs(batch_size=3, candidates=6)
    base_factor_logits = torch.randn(3, 6, len(FACTOR_KEYS))
    base_scores = torch.randn(3, 6)
    target_factors = torch.rand(3, 6, 7)
    args = SimpleNamespace(
        minimum_pair_delta=0.02,
        factor_rank_minimum_delta=0.05,
        target_temperature=0.05,
        prediction_temperature=0.05,
        top_set_tolerance=0.01,
        safety_negative_weight=1.0,
        pairwise_weight=1.0,
        base_pairwise_weight=1.0,
        listwise_weight=0.1,
        top_set_weight=0.5,
        expected_regret_weight=1.0,
        top_regret_weight=1.0,
        top_regret_minimum_delta=0.01,
        factor_weight=1.0,
        private_factor_weight=0.25,
        factor_rank_weight=0.5,
        relative_safety_weight=0.5,
        residual_l2_weight=0.01,
        factor_loss_scope="topk",
    )
    loss, details = compute_residual_training_loss(
        model,
        (
            proposals,
            observation,
            torch.ones(3, observation.shape[1], dtype=torch.bool),
            status,
            base_scores,
            base_factor_logits,
            target_factors,
            torch.arange(3),
        ),
        args,
    )
    assert torch.isfinite(loss)
    assert all(np.isfinite(value) for value in details.values())
    assert details["top_regret"] >= 0
    loss.backward()
    assert model.factor_delta_head[-1].weight.grad is not None
    assert model.private_ranker.coarse_factor_heads[
        FACTOR_KEYS[0]
    ].weight.grad is not None


def test_m0_conservative_reference_head_is_equivariant_and_trainable() -> None:
    torch.manual_seed(110)
    private_config = _small_config(fine_top_k=6, dropout=0.0)
    model = M0PrivateResidualRanker(
        private_config,
        M0PrivateResidualConfig(
            hidden_dim=private_config.model_dim,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
            top_k=4,
            conservative_reference=True,
            reference_hidden_dim=64,
            reference_layers=1,
            gain_quantile_index=1,
        ),
    )
    observation, status, proposals = _inputs(batch_size=2, candidates=6)
    base_factor_logits = torch.randn(2, 6, len(FACTOR_KEYS))
    base_scores = torch.randn(2, 6)
    target_factors = torch.rand(2, 6, 7)
    valid_mask = torch.ones(2, observation.shape[1], dtype=torch.bool)
    output = model(
        observation,
        status,
        proposals,
        base_factor_logits,
        base_scores,
        observation_valid_mask=valid_mask,
    )
    reference = base_scores.argmax(dim=1)
    rows = torch.arange(2)
    torch.testing.assert_close(
        output["selection_scores"][rows, reference], torch.zeros(2)
    )
    assert output["gain_quantiles"].shape == (2, 6, 3)
    assert output["safety_worse_logits"].shape == (2, 6, 3)

    permutation = torch.tensor([4, 0, 5, 2, 1, 3])
    inverse = torch.argsort(permutation)
    permuted = model(
        observation,
        status,
        proposals[:, permutation],
        base_factor_logits[:, permutation],
        base_scores[:, permutation],
        observation_valid_mask=valid_mask,
    )
    for key in (
        "selection_scores",
        "gain_quantiles",
        "safety_worse_logits",
        "safe_improvement_logit",
    ):
        torch.testing.assert_close(
            output[key], permuted[key][:, inverse], rtol=1e-5, atol=1e-6
        )

    args = SimpleNamespace(
        minimum_pair_delta=0.02,
        factor_rank_minimum_delta=0.05,
        target_temperature=0.05,
        prediction_temperature=0.05,
        top_set_tolerance=0.01,
        safety_negative_weight=1.0,
        pairwise_weight=0.0,
        base_pairwise_weight=0.0,
        listwise_weight=0.0,
        top_set_weight=0.0,
        expected_regret_weight=0.0,
        top_regret_weight=0.0,
        factor_weight=0.0,
        private_factor_weight=0.0,
        factor_rank_weight=0.0,
        relative_safety_weight=0.0,
        residual_l2_weight=0.0,
        factor_loss_scope="topk",
        reference_weight=1.0,
        reference_minimum_improvement_target=0.005,
        reference_factor_epsilon=1e-6,
        reference_safety_worse_positive_weight=10.0,
        reference_safe_improvement_positive_weight=3.0,
        reference_switch_margin_temperature=0.05,
        reference_quantile_weight=1.0,
        reference_median_rank_weight=0.25,
        reference_safety_weight=1.0,
        reference_improvement_weight=0.5,
        reference_false_switch_weight=0.5,
        reference_missed_improvement_weight=0.0,
    )
    loss, details = compute_residual_training_loss(
        model,
        (
            proposals,
            observation,
            valid_mask,
            status,
            base_scores,
            base_factor_logits,
            target_factors,
            torch.arange(2),
        ),
        args,
    )
    assert torch.isfinite(loss)
    assert details["reference"] > 0
    loss.backward()
    assert model.conservative_reference_head.gain_quantile_head.weight.grad is not None


def test_m0_private_factor_loss_can_be_scoped_to_deployed_shortlist() -> None:
    torch.manual_seed(109)
    private_config = _small_config(fine_top_k=2, dropout=0.0)
    model = M0PrivateResidualRanker(
        private_config,
        M0PrivateResidualConfig(
            hidden_dim=private_config.model_dim,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
            top_k=2,
        ),
    ).eval()
    observation, status, proposals = _inputs(batch_size=1, candidates=6)
    base_factor_logits = torch.full((1, 6, len(FACTOR_KEYS)), 2.0)
    base_scores = torch.tensor([[6.0, 5.0, 4.0, 3.0, 2.0, 1.0]])
    target_a = torch.zeros(1, 6, 7)
    target_b = target_a.clone()
    # Only candidates outside the deployed Base-anchored Top-2 differ.
    target_b[:, 2:, :6] = 1.0

    args = SimpleNamespace(
        minimum_pair_delta=0.02,
        factor_rank_minimum_delta=0.05,
        target_temperature=0.05,
        prediction_temperature=0.05,
        top_set_tolerance=0.01,
        safety_negative_weight=1.0,
        pairwise_weight=0.0,
        base_pairwise_weight=0.0,
        listwise_weight=0.0,
        top_set_weight=0.0,
        expected_regret_weight=0.0,
        top_regret_weight=0.0,
        top_regret_minimum_delta=0.01,
        factor_weight=1.0,
        private_factor_weight=0.0,
        factor_rank_weight=0.0,
        relative_safety_weight=0.0,
        residual_l2_weight=0.0,
        factor_loss_scope="topk",
    )

    def compute(target_factors):
        return compute_residual_training_loss(
            model,
            (
                proposals,
                observation,
                torch.ones(1, observation.shape[1], dtype=torch.bool),
                status,
                base_scores,
                base_factor_logits,
                target_factors,
                torch.arange(1),
            ),
            args,
        )[1]["factor"]

    topk_a = compute(target_a)
    topk_b = compute(target_b)
    assert topk_a == pytest.approx(topk_b, abs=1e-7)
    args.factor_loss_scope = "all"
    assert compute(target_a) != pytest.approx(compute(target_b), abs=1e-7)


def test_m0_private_residual_shared_future_training_loss_is_finite() -> None:
    torch.manual_seed(104)
    private_config = _small_config(
        fine_top_k=6,
        current_actor_auxiliary=True,
        shared_future_auxiliary=True,
        shared_future_horizons=8,
        shared_future_relabeling=True,
        shared_future_constant_velocity_residual=True,
    )
    model = M0PrivateResidualRanker(
        private_config,
        M0PrivateResidualConfig(
            hidden_dim=private_config.model_dim,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
            top_k=6,
            score_mode="factor",
        ),
    )
    observation, status, proposals = _inputs(batch_size=2, candidates=6)
    base_factor_logits = torch.randn(2, 6, len(FACTOR_KEYS))
    base_scores = torch.randn(2, 6)
    target_factors = torch.rand(2, 6, 7)
    future = torch.zeros(2, 8, 3, 8)
    future[0, :, 0] = torch.tensor(
        [0.0, 8.0, -1.0, 2.0, 0.2, 0.1, 4.5, 1.9]
    )
    future[0, 1:, 1] = torch.tensor(
        [1.0, 15.0, 2.0, -1.0, 0.0, -0.2, 1.0, 0.8]
    )
    future_mask = torch.zeros(2, 8, 3, dtype=torch.bool)
    future_mask[0, :, 0] = True
    future_mask[0, 1:, 1] = True
    args = SimpleNamespace(
        minimum_pair_delta=0.02,
        factor_rank_minimum_delta=0.05,
        target_temperature=0.05,
        prediction_temperature=0.05,
        top_set_tolerance=0.01,
        safety_negative_weight=1.0,
        pairwise_weight=1.0,
        base_pairwise_weight=1.0,
        listwise_weight=0.1,
        top_set_weight=0.5,
        expected_regret_weight=1.0,
        factor_weight=1.0,
        private_factor_weight=0.25,
        factor_rank_weight=0.5,
        relative_safety_weight=0.5,
        residual_l2_weight=0.01,
        shared_future_weight=0.5,
        current_actor_weight=0.5,
        candidate_relative_weight=1.0,
    )
    loss, details = compute_residual_training_loss(
        model,
        (
            proposals,
            observation,
            torch.ones(2, observation.shape[1], dtype=torch.bool),
            status,
            base_scores,
            base_factor_logits,
            target_factors,
            torch.arange(2),
            future[:, 0],
            future_mask[:, 0],
            torch.tensor([True, False]),
            future,
            future_mask,
            torch.tensor([True, False]),
        ),
        args,
    )
    assert torch.isfinite(loss)
    assert details["shared_future"] > 0
    assert details["current_actor"] > 0
    assert details["candidate_relative"] > 0
    loss.backward()
    assert model.private_ranker.shared_future_state_head.weight.grad is not None
    assert model.private_ranker.current_actor_state_head.weight.grad is not None


def test_candidate_relative_relabeler_empty_actor_slots_are_safe() -> None:
    relabeler = SharedFutureCandidateRelabeler(
        model_dim=32,
        horizons=3,
        num_heads=4,
        dropout=0.0,
        interval_seconds=0.5,
    ).eval()
    presence = torch.full((1, 3, 4), -20.0)
    actor = torch.zeros(1, 3, 4, 8)
    proposals = torch.zeros(1, 2, 3, 3)
    consequence = relabeler.consequence_only(presence, actor, proposals)
    torch.testing.assert_close(
        consequence[..., 0], torch.full((1, 2, 3), 2.0), atol=1.0e-5, rtol=0
    )
    torch.testing.assert_close(
        consequence[..., 1], torch.zeros(1, 2, 3), atol=1.0e-6, rtol=0
    )
    torch.testing.assert_close(
        consequence[..., 2], torch.ones(1, 2, 3), atol=1.0e-5, rtol=0
    )
    torch.testing.assert_close(
        consequence[..., 3:], torch.zeros(1, 2, 3, 5), atol=1.0e-5, rtol=0
    )


def test_candidate_relative_geometry_uses_official_rear_axle_footprint() -> None:
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

    parameters = get_pacifica_parameters()
    assert EGO_HALF_LENGTH_M == pytest.approx(parameters.length / 2.0)
    assert EGO_HALF_WIDTH_M == pytest.approx(parameters.width / 2.0)
    assert EGO_REAR_AXLE_TO_CENTER_M == pytest.approx(
        (parameters.front_length - parameters.rear_length) / 2.0
    )

    proposals = torch.zeros(1, 1, 2, 3)
    proposals[..., 1, 2] = torch.pi / 2.0
    center, velocity = SharedFutureCandidateRelabeler._candidate_center_and_velocity(
        proposals, interval_seconds=0.5
    )
    torch.testing.assert_close(
        center[0, 0, 0], torch.tensor([EGO_REAR_AXLE_TO_CENTER_M, 0.0])
    )
    torch.testing.assert_close(
        center[0, 0, 1], torch.tensor([0.0, EGO_REAR_AXLE_TO_CENTER_M]), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(velocity[0, 0, 0], torch.zeros(2))
    torch.testing.assert_close(
        velocity[0, 0, 1],
        torch.tensor(
            [-2.0 * EGO_REAR_AXLE_TO_CENTER_M, 2.0 * EGO_REAR_AXLE_TO_CENTER_M]
        ),
        atol=1e-6,
        rtol=0,
    )


def test_candidate_relative_consequence_loss_masks_invalid_scenes() -> None:
    relabeler = SharedFutureCandidateRelabeler(
        model_dim=32,
        horizons=3,
        num_heads=4,
        dropout=0.0,
        interval_seconds=0.5,
    ).eval()
    future = torch.zeros(2, 3, 2, 8)
    future[0, :, 0] = torch.tensor(
        [0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 4.0, 2.0]
    )
    mask = torch.zeros(2, 3, 2, dtype=torch.bool)
    mask[0, :, 0] = True
    proposals = torch.zeros(2, 2, 3, 3)
    _, normalized = normalize_current_actor_targets(
        future.reshape(2 * 3, 2, 8)
    )
    normalized = normalized.reshape(2, 3, 2, 8)
    target = relabeler.consequence_only(
        torch.where(mask, torch.tensor(20.0), torch.tensor(-20.0)),
        normalized,
        proposals,
    )
    predicted = target.clone()
    predicted[0, ..., 1] += 0.5
    predicted[1] += 100.0
    losses = candidate_relative_consequence_loss(
        {"candidate_relative_consequence": predicted},
        relabeler,
        proposals,
        future,
        mask,
        torch.tensor([True, False]),
    )
    assert losses["total"] > 0
    assert losses["collision"] > 0
    assert losses["ttc"] == pytest.approx(0.0)


def test_shared_future_candidate_relabeler_has_physical_ordering() -> None:
    torch.manual_seed(105)
    relabeler = SharedFutureCandidateRelabeler(
        model_dim=32,
        horizons=3,
        num_heads=4,
        dropout=0.0,
        interval_seconds=0.5,
    ).eval()
    presence = torch.full((1, 3, 2), -10.0)
    presence[..., 0] = 10.0
    actor = torch.zeros(1, 3, 2, 8)
    actor[..., 0, 0] = 5.0 / 50.0
    actor[..., 0, 5] = 1.0
    actor[..., 0, 6] = 4.0 / 10.0
    actor[..., 0, 7] = 2.0 / 5.0
    proposals = torch.zeros(1, 2, 3, 3)
    proposals[:, 0, :, 0] = 5.0
    proposals[:, 1, :, 0] = -20.0

    consequence, token = relabeler(presence, actor, proposals)
    assert tuple(consequence.shape) == (1, 2, 3, 8)
    assert tuple(token.shape) == (1, 2, 32)
    # Candidate 0 overlaps the actor; candidate 1 remains far away.
    assert torch.all(consequence[0, 0, :, 0] < consequence[0, 1, :, 0])
    assert torch.all(consequence[0, 0, :, 1] > consequence[0, 1, :, 1])
    assert torch.isfinite(consequence).all()
    assert torch.isfinite(token).all()

    permutation = torch.tensor([1, 0])
    permuted_consequence, permuted_token = relabeler(
        presence, actor, proposals[:, permutation]
    )
    torch.testing.assert_close(
        consequence, permuted_consequence[:, permutation]
    )
    torch.testing.assert_close(token, permuted_token[:, permutation])


def test_factorized_shared_future_is_predicted_once_and_candidate_equivariant() -> None:
    torch.manual_seed(106)
    config = _small_config(
        shared_future_auxiliary=True,
        shared_future_horizons=8,
        shared_future_relabeling=True,
    )
    model = IndependentProposalRanker(config).eval()
    observations, status, proposals = _inputs()
    calls = []
    handle = model.shared_future_state_head.register_forward_hook(
        lambda *_: calls.append(1)
    )
    permutation = torch.tensor([4, 0, 5, 2, 1, 3])
    inverse = torch.argsort(permutation)
    with torch.no_grad():
        reference = model(observations, status, proposals)
        assert len(calls) == 1
        calls.clear()
        permuted = model(observations, status, proposals[:, permutation])
        assert len(calls) == 1
    handle.remove()
    for key in (
        "shared_future_presence_logits",
        "shared_future_type_logits",
        "shared_future_actor_state",
    ):
        torch.testing.assert_close(reference[key], permuted[key])
    for key in (
        "candidate_relative_consequence",
        "candidate_relative_consequence_token",
    ):
        torch.testing.assert_close(
            reference[key], permuted[key][:, inverse], rtol=1e-5, atol=1e-6
        )


def test_current_actor_cv_relabeling_is_shared_and_candidate_equivariant() -> None:
    torch.manual_seed(206)
    config = _small_config(
        current_actor_auxiliary=True,
        shared_future_horizons=8,
        current_actor_cv_relabeling=True,
    )
    model = IndependentProposalRanker(config).eval()
    observations, status, proposals = _inputs()
    calls = []
    handle = model.current_actor_state_head.register_forward_hook(
        lambda *_: calls.append(1)
    )
    permutation = torch.tensor([4, 0, 5, 2, 1, 3])
    inverse = torch.argsort(permutation)
    with torch.no_grad():
        reference = model(observations, status, proposals)
        assert len(calls) == 1
        calls.clear()
        permuted = model(observations, status, proposals[:, permutation])
        assert len(calls) == 1
    handle.remove()

    for key in (
        "current_actor_presence_logits",
        "current_actor_type_logits",
        "current_actor_state",
    ):
        torch.testing.assert_close(reference[key], permuted[key])
    for key in (
        "candidate_relative_consequence",
        "candidate_relative_consequence_token",
    ):
        torch.testing.assert_close(
            reference[key], permuted[key][:, inverse], rtol=1e-5, atol=1e-6
        )

    expected = model.shared_future_relabeler.consequence_only(
        reference["current_actor_presence_logits"][:, None].expand(-1, 8, -1),
        model._constant_velocity_actor_future(reference["current_actor_state"]),
        proposals,
    )
    torch.testing.assert_close(
        reference["candidate_relative_consequence"], expected
    )


def test_current_actor_cv_relabeling_zero_gate_preserves_legacy_outputs() -> None:
    torch.manual_seed(207)
    legacy_config = _small_config(current_actor_auxiliary=True)
    legacy = IndependentProposalRanker(legacy_config).eval()
    cv = IndependentProposalRanker(
        _small_config(
            current_actor_auxiliary=True,
            current_actor_cv_relabeling=True,
        )
    ).eval()
    missing, unexpected = cv.load_state_dict(legacy.state_dict(), strict=False)
    assert not unexpected
    assert missing
    assert all(
        name.startswith("shared_future_relabeler.")
        or name == "shared_future_fusion_gate"
        for name in missing
    )
    observations, status, proposals = _inputs()
    with torch.no_grad():
        reference = legacy(observations, status, proposals)
        candidate = cv(observations, status, proposals)
    assert candidate["shared_future_fusion_gate"].item() == 0.0
    for key in (
        "utility",
        "coarse_utility",
        "refined_utility",
        "factor_logits",
        "candidate_features",
    ):
        torch.testing.assert_close(reference[key], candidate[key], rtol=0, atol=0)


def test_current_actor_cv_relabeling_requires_current_actor_head() -> None:
    with pytest.raises(ValueError, match="requires current-actor"):
        _small_config(current_actor_cv_relabeling=True)


def test_current_actor_and_learned_future_relabeling_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        _small_config(
            current_actor_auxiliary=True,
            shared_future_auxiliary=True,
            shared_future_relabeling=True,
            current_actor_cv_relabeling=True,
        )


def test_shared_future_constant_velocity_residual_starts_from_predicted_current() -> None:
    torch.manual_seed(107)
    config = _small_config(
        current_actor_auxiliary=True,
        shared_future_auxiliary=True,
        shared_future_horizons=8,
        shared_future_relabeling=True,
        shared_future_constant_velocity_residual=True,
    )
    model = IndependentProposalRanker(config).eval()
    observations, status, proposals = _inputs()
    with torch.no_grad():
        output = model(observations, status, proposals)
    expected_state = model._constant_velocity_actor_future(
        output["current_actor_state"]
    )
    torch.testing.assert_close(output["shared_future_actor_state"], expected_state)
    torch.testing.assert_close(
        output["shared_future_presence_logits"],
        output["current_actor_presence_logits"][:, None].expand(-1, 8, -1),
    )
    torch.testing.assert_close(
        output["shared_future_type_logits"],
        output["current_actor_type_logits"][:, None].expand(-1, 8, -1, -1),
    )
    assert torch.count_nonzero(model.shared_future_state_head.weight) == 0


def test_shared_future_constant_velocity_normalization_is_metric_correct() -> None:
    config = _small_config(
        current_actor_auxiliary=True,
        shared_future_auxiliary=True,
        shared_future_horizons=2,
        shared_future_constant_velocity_residual=True,
    )
    model = IndependentProposalRanker(config)
    current = torch.zeros(1, 1, 8)
    current[..., 0] = 0.1
    current[..., 1] = -0.2
    current[..., 2] = 0.5
    current[..., 3] = -0.25
    future = model._constant_velocity_actor_future(current)
    torch.testing.assert_close(
        future[0, :, 0, :4],
        torch.tensor(
            [[0.2, -0.25, 0.5, -0.25], [0.3, -0.3, 0.5, -0.25]]
        ),
    )


def test_constant_velocity_future_residual_requires_current_actor_head() -> None:
    with pytest.raises(ValueError, match="require current-actor"):
        _small_config(
            shared_future_auxiliary=True,
            shared_future_constant_velocity_residual=True,
        )


def test_factorized_shared_future_requires_prediction_head() -> None:
    with pytest.raises(ValueError, match="requires the shared-future"):
        _small_config(shared_future_relabeling=True)


def test_shared_future_metrics_compare_constant_velocity_in_metric_units() -> None:
    normalized = torch.tensor(
        [[[[0.1, -0.2, 0.15, -0.1, 0.0, 1.0, 0.45, 0.4]]]]
    )
    decoded = _decode_predicted_state(normalized)
    torch.testing.assert_close(
        decoded,
        torch.tensor([[[[5.0, -10.0, 3.0, -2.0, 0.0, 4.5, 2.0]]]]),
    )

    current = torch.tensor([[[0.0, 0.0, 0.0, 2.0, -1.0, 0.0, 4.5, 2.0]]])
    constant_velocity = _baseline_states(current, horizons=2, constant_velocity=True)
    expected = torch.tensor(
        [[[[1.0, -0.5, 2.0, -1.0, 0.0, 4.5, 2.0]],
          [[2.0, -1.0, 2.0, -1.0, 0.0, 4.5, 2.0]]]]
    )
    torch.testing.assert_close(constant_velocity, expected)
    valid = torch.ones(1, 2, 1, dtype=torch.bool)
    errors = _state_errors(constant_velocity, expected, valid)
    assert errors["position_l2_mae_m"] == pytest.approx(0.0)
    assert errors["velocity_l2_mae_mps"] == pytest.approx(0.0)
    presence = _presence_metrics(
        torch.tensor([[[12.0], [-12.0]]]),
        torch.tensor([[[True], [False]]]),
    )
    assert presence["accuracy"] == pytest.approx(1.0)
    assert presence["f1"] == pytest.approx(1.0)


def test_m0_private_residual_multireplay_selects_metrics_by_source() -> None:
    target_factors = torch.ones(4, 3, 7)
    target_factors[..., -1] = torch.tensor(
        [
            [0.9, 0.5, 0.4],
            [0.8, 0.3, 0.2],
            [0.2, 0.7, 0.4],
            [0.1, 0.6, 0.5],
        ]
    )
    base_scores = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    selection_scores = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    combined, by_source = evaluate_residual_predictions_by_source(
        selection_scores,
        torch.zeros(4, 3, 6),
        base_scores,
        target_factors,
        ["public-log-a", "public-log-b", "epoch3-log-a", "epoch3-log-b"],
        ["public_base", "public_base", "epoch3", "epoch3"],
        seed=2,
        bootstrap_replicates=20,
    )
    assert combined["scene_count"] == 4
    assert set(by_source) == {"epoch3", "public_base"}
    assert by_source["public_base"]["scene_count"] == 2
    assert by_source["epoch3"]["scene_count"] == 2
    assert by_source["public_base"]["selected_delta"] == pytest.approx(0.45)
    assert by_source["epoch3"]["selected_delta"] == pytest.approx(0.0)


def test_m0_private_residual_policy_split_has_no_log_leakage() -> None:
    logs = ["log-a"] * 5 + ["log-b"] * 2 + ["log-c"] * 4 + ["log-d"]
    calibration, promotion, assignment = balanced_calibration_split(logs, seed=7)
    calibration_logs = {logs[index] for index in calibration}
    promotion_logs = {logs[index] for index in promotion}
    assert calibration_logs
    assert promotion_logs
    assert calibration_logs.isdisjoint(promotion_logs)
    assert set(assignment) == set(logs)


def test_m0_private_residual_conservative_policy_can_preserve_base() -> None:
    tensors = {
        "base_scores": torch.tensor([[0.8, 0.7, 0.1]]),
        "residual": torch.tensor([[0.0, 0.5, 0.5]]),
        "refined_factor_logits": torch.zeros(1, 3, 6),
        "relative_safety_logits": torch.zeros(1, 3, 3),
    }
    base_policy = {
        "inference_scale": 0.0,
        "switch_penalty": 0.0,
        "safety_floor": 0.0,
        "safety_relative_tolerance": 1.0,
        "preserve_ddc": False,
        "safety_gate_mode": "none",
    }
    assert policy_selection_indices(
        tensors,
        torch.tensor([0]),
        base_policy,
        torch.device("cpu"),
    ).item() == 0
    switched = dict(base_policy, inference_scale=1.0)
    assert policy_selection_indices(
        tensors,
        torch.tensor([0]),
        switched,
        torch.device("cpu"),
    ).item() == 1
    conservative = dict(switched, switch_penalty=0.5)
    assert policy_selection_indices(
        tensors,
        torch.tensor([0]),
        conservative,
        torch.device("cpu"),
    ).item() == 0


def test_drivor_ranker_forward_is_current_observation_only() -> None:
    parameters = set(
        inspect.signature(DrivORInitializedProposalRanker.forward).parameters
    )
    assert parameters == {
        "self",
        "scene_registers",
        "status_feature",
        "proposals",
        "scene_valid_mask",
    }
    assert not any(
        "base" in name or "pdm" in name or "future" in name
        for name in parameters
    )


def test_drivor_ranker_is_candidate_permutation_equivariant() -> None:
    torch.manual_seed(21)
    model = DrivORInitializedProposalRanker(_small_drivor_config()).eval()
    observations, status, proposals = _inputs()
    permutation = torch.tensor([4, 0, 5, 2, 1, 3])
    inverse = torch.argsort(permutation)
    with torch.no_grad():
        reference = model(observations, status, proposals)
        permuted = model(observations, status, proposals[:, permutation])
    for key in ("factor_logits", "direct_utility", "candidate_features"):
        torch.testing.assert_close(
            reference[key], permuted[key][:, inverse], rtol=1e-5, atol=1e-6
        )


def test_drivor_ranker_detaches_proposals_and_zero_initializes_direct_head() -> None:
    torch.manual_seed(22)
    model = DrivORInitializedProposalRanker(_small_drivor_config()).eval()
    observations, status, proposals = _inputs()
    proposals.requires_grad_(True)
    output = model(observations, status, proposals)
    assert torch.count_nonzero(output["direct_utility"]) == 0
    output["factor_logits"].sum().backward()
    assert proposals.grad is None


def test_drivor_checkpoint_mapping_is_exact(tmp_path) -> None:
    torch.manual_seed(23)
    source = DrivORInitializedProposalRanker(_small_drivor_config())
    checkpoint_state = {}
    expected = {}
    for key, value in source.state_dict().items():
        if key.split(".", 1)[0] not in source._PRETRAINED_MODULES:
            continue
        replacement = torch.randn_like(value)
        checkpoint_state[f"agent._drivor_model.{key}"] = replacement
        expected[key] = replacement
    checkpoint_state["agent._drivor_model.trajectory_decoder.unused"] = torch.ones(1)
    checkpoint = tmp_path / "drivor.ckpt"
    torch.save({"state_dict": checkpoint_state}, checkpoint)

    target = DrivORInitializedProposalRanker(_small_drivor_config())
    audit = target.load_drivor_checkpoint(checkpoint)
    assert audit["loaded_tensor_count"] == len(expected)
    assert audit["direct_head_zero_initialized"] is True
    target_state = target.state_dict()
    for key, value in expected.items():
        torch.testing.assert_close(target_state[key], value)
    assert torch.count_nonzero(target.direct_utility_head[-1].weight) == 0


def test_future_and_evaluator_inputs_are_rejected() -> None:
    assert_current_observation_only(
        {"current_image": object(), "status_feature": object(), "proposals": object()}
    )
    with pytest.raises(RuntimeError, match="official_score"):
        assert_current_observation_only({"current_image": object(), "official_score": object()})
    with pytest.raises(RuntimeError, match="future_annotations"):
        assert_current_observation_only({"future_annotations": object()})


def test_episode_drive_factor_loss_matches_released_six_head_bce() -> None:
    logits = torch.tensor(
        [[[0.2, -0.4, 0.7, -1.0, 0.5, 1.2], [-0.3, 0.8, -0.2, 0.4, -0.7, 0.1]]]
    )
    targets = torch.tensor(
        [[[1.0, 0.0, 0.5, 1.0, 0.35, 1.0], [0.5, 1.0, 1.0, 0.0, 0.82, 0.0]]]
    )
    released_targets = targets.clone()
    released_targets[..., 0] = (released_targets[..., 0] == 1.0).float()
    released_targets[..., 2] = (released_targets[..., 2] == 1.0).float()
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, released_targets
    )
    actual = episode_drive_factor_loss(
        logits, targets, safety_negative_weight=1.0
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    changed_progress = targets.clone()
    changed_progress[..., 4] = 1.0 - changed_progress[..., 4]
    assert not torch.equal(
        episode_drive_factor_loss(logits, changed_progress), actual
    )


def test_safety_weight_includes_rare_ddc_failures() -> None:
    logits = torch.zeros(1, 1, len(FACTOR_KEYS))
    safe = torch.ones_like(logits)
    ddc_failure = safe.clone()
    ddc_failure[..., 2] = 0.0
    source = episode_drive_factor_loss(
        logits, ddc_failure, safety_negative_weight=1.0
    )
    weighted = episode_drive_factor_loss(
        logits, ddc_failure, safety_negative_weight=5.0
    )
    # With zero logits every BCE element is equal, so a correctly normalized
    # weighted mean is numerically unchanged.  A nonzero DDC logit exposes the
    # increased influence of the rare negative label.
    assert weighted == pytest.approx(source)
    logits[..., 2] = 2.0
    source = episode_drive_factor_loss(
        logits, ddc_failure, safety_negative_weight=1.0
    )
    weighted = episode_drive_factor_loss(
        logits, ddc_failure, safety_negative_weight=5.0
    )
    assert weighted > source


def _all_log_refit_fixture():
    config = _small_config()
    locked = {
        "seed": 2,
        "batch_size": 32,
        "learning_rate": 3.0e-4,
        "weight_decay": 1.0e-4,
        "candidate_keep_count": 64,
        "fine_top_k": 3,
        "model_dim": 32,
        "dynamic_queries": 3,
        "private_layers": 1,
        "trajectory_layers": 1,
        "candidate_layers": 1,
        "fine_layers": 1,
        "dropout": 0.0,
        "target_temperature": 0.05,
        "prediction_temperature": 0.05,
        "top_set_tolerance": 0.01,
        "pairwise_weight": 0.0,
        "hard_pairwise_weight": 0.0,
        "listwise_weight": 0.0,
        "top_set_weight": 0.0,
        "expected_regret_weight": 0.0,
        "top_regret_weight": 0.0,
        "coarse_loss_weight": 0.0,
        "factor_weight": 1.0,
        "factor_loss_mode": "continuous_progress",
        "progress_regression_weight": 2.0,
        "factor_rank_weight": 1.0,
        "consequence_weight": 0.0,
        "confidence_weight": 0.0,
        "current_actor_weight": 0.0,
        "safety_negative_weight": 1.0,
    }
    args = SimpleNamespace(**locked, epochs=4)
    # The first sweep predates explicit factor-loss switches. Their only
    # accepted legacy resolution is encoded by the refit validator.
    selected_args = {
        key: value
        for key, value in locked.items()
        if key not in {"factor_loss_mode", "progress_regression_weight"}
    }
    selected_args["epochs"] = 8
    selected = {
        "architecture": "IndependentProposalRanker",
        "selection_mode": "factor",
        "epoch": 3,
        "checkpoint_selection_source": "public_base",
        "model_config": asdict(config),
        "validation_by_source": {
            "public_base": {
                "factor_selected_pdms": 0.96,
                "factor_selected_delta_log_bootstrap_95ci": [0.001, 0.01],
            }
        },
        "fold_manifest": {
            "args": selected_args,
            "train_physical_logs": ["train-a", "train-b"],
            "validation_physical_logs": ["validation-a"],
        },
    }
    return selected, args, config


def test_all_log_refit_requires_positive_heldout_ci_and_locks_schedule() -> None:
    selected, args, config = _all_log_refit_fixture()
    provenance = validate_all_log_refit_provenance(selected, args, config)
    assert provenance["selection_mode"] == "factor"
    assert provenance["selected_epoch"] == 3
    assert provenance["scheduler_horizon_epochs"] == 8
    assert provenance["locked_training_arguments"]["factor_loss_mode"] == (
        "continuous_progress"
    )
    assert provenance["selection_validation_physical_log_count"] == 1


def test_all_log_refit_rejects_nonpositive_heldout_ci() -> None:
    selected, args, config = _all_log_refit_fixture()
    selected["validation_by_source"]["public_base"][
        "factor_selected_delta_log_bootstrap_95ci"
    ] = [0.0, 0.01]
    with pytest.raises(RuntimeError, match="held-out CI gate"):
        validate_all_log_refit_provenance(selected, args, config)


def test_all_log_refit_rejects_training_argument_change() -> None:
    selected, args, config = _all_log_refit_fixture()
    args.learning_rate = 1.0e-3
    with pytest.raises(RuntimeError, match="arguments differ"):
        validate_all_log_refit_provenance(selected, args, config)


def _m0_all_log_refit_fixture():
    private_config = _small_config()
    residual_config = M0PrivateResidualConfig(
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        top_k=4,
        max_residual=0.5,
        score_mode="hybrid",
        m0_candidate_fusion=True,
        m0_candidate_dim=16,
    )
    locked = {
        "seed": 2,
        "batch_size": 32,
        "learning_rate": 3.0e-4,
        "weight_decay": 1.0e-4,
        "model_dim": 32,
        "dynamic_queries": 3,
        "private_layers": 1,
        "trajectory_layers": 1,
        "candidate_layers": 1,
        "fine_layers": 1,
        "private_fine_top_k": 3,
        "residual_layers": 1,
        "m0_context_fusion": False,
        "m0_candidate_fusion": True,
        "m0_candidate_only": False,
        "conservative_reference": False,
        "reference_hidden_dim": 512,
        "reference_layers": 2,
        "reference_gain_quantile_index": 1,
        "reference_minimum_lcb_gain": 0.0,
        "reference_maximum_safety_worse_probability": 0.1,
        "reference_minimum_safe_improvement_probability": 0.7,
        "residual_top_k": 4,
        "score_mode": "hybrid",
        "max_residual": 0.5,
        "dropout": 0.0,
        "minimum_pair_delta": 0.02,
        "factor_rank_minimum_delta": 0.05,
        "target_temperature": 0.05,
        "prediction_temperature": 0.05,
        "top_set_tolerance": 0.01,
        "pairwise_weight": 1.0,
        "base_pairwise_weight": 1.0,
        "listwise_weight": 0.1,
        "top_set_weight": 0.5,
        "expected_regret_weight": 1.0,
        "top_regret_weight": 0.0,
        "top_regret_minimum_delta": 0.01,
        "factor_weight": 1.0,
        "private_factor_weight": 0.25,
        "factor_rank_weight": 0.5,
        "relative_safety_weight": 0.5,
        "residual_l2_weight": 0.01,
        "reference_weight": 0.0,
        "reference_quantile_weight": 1.0,
        "reference_median_rank_weight": 0.25,
        "reference_safety_weight": 1.0,
        "reference_improvement_weight": 0.5,
        "reference_false_switch_weight": 0.5,
        "reference_missed_improvement_weight": 0.0,
        "reference_safety_worse_positive_weight": 10.0,
        "reference_safe_improvement_positive_weight": 3.0,
        "reference_switch_margin_temperature": 0.05,
        "reference_minimum_improvement_target": 0.005,
        "reference_factor_epsilon": 1e-6,
        "shared_future_weight": 0.0,
        "current_actor_weight": 0.0,
        "candidate_relative_weight": 0.0,
        "safety_negative_weight": 5.0,
        "factor_loss_scope": "all",
        "shared_future_relabeling": False,
        "shared_future_constant_velocity_residual": False,
    }
    args = SimpleNamespace(**locked, epochs=3)
    selected_args = dict(locked, epochs=8)
    # Exercise the only valid compatibility path for pre-wave-4 artifacts.
    for name in (
        "top_regret_weight",
        "top_regret_minimum_delta",
        "factor_loss_scope",
    ):
        selected_args.pop(name)
    deployment_config = asdict(residual_config)
    deployment_config.update(
        inference_scale=0.5,
        safety_floor=0.85,
        safety_gate_mode="relative_factor",
    )
    selected = {
        "architecture": "M0PrivateResidualRanker",
        "epoch": 2,
        "checkpoint_selection_source": "no_vqa_e35",
        "private_config": asdict(private_config),
        "residual_config": deployment_config,
        "validation_by_source": {
            "no_vqa_e35": {
                "selected_pdms": 0.94,
                "base_selected_pdms": 0.93,
                "selected_delta": 0.01,
                "selected_delta_log_bootstrap_95ci": [0.001, 0.01],
                "scene_count": 100,
                "physical_log_count": 10,
            }
        },
        "fold_manifest": {
            "args": selected_args,
            "train_physical_logs": ["train-a", "train-b"],
            "validation_physical_logs": ["validation-a"],
        },
        "policy_calibration": {"policy": {"inference_scale": 0.5}},
        "policy_selection_uses_navtest": False,
        "policy_selection_uses_disjoint_physical_logs": True,
    }
    return selected, args, private_config, residual_config


def test_m0_all_log_refit_locks_training_and_deployment_policy() -> None:
    selected, args, private_config, residual_config = _m0_all_log_refit_fixture()
    provenance = validate_m0_all_log_refit_provenance(
        selected, args, private_config, residual_config
    )
    assert provenance["selected_epoch"] == 2
    assert provenance["scheduler_horizon_epochs"] == 8
    assert provenance["deployment_policy_frozen_from_selection"] is True
    assert provenance["deployment_residual_config"]["inference_scale"] == 0.5
    assert provenance["locked_training_arguments"]["factor_loss_scope"] == "all"


def test_m0_all_log_refit_rejects_training_change_or_navtest_policy() -> None:
    selected, args, private_config, residual_config = _m0_all_log_refit_fixture()
    args.safety_negative_weight = 1.0
    with pytest.raises(RuntimeError, match="arguments differ"):
        validate_m0_all_log_refit_provenance(
            selected, args, private_config, residual_config
        )
    args.safety_negative_weight = 5.0
    selected["policy_selection_uses_navtest"] = True
    with pytest.raises(RuntimeError, match="must not use Navtest"):
        validate_m0_all_log_refit_provenance(
            selected, args, private_config, residual_config
        )


def test_candidate_permutation_equivariance() -> None:
    torch.manual_seed(4)
    model = IndependentProposalRanker(_small_config()).eval()
    observations, status, proposals = _inputs()
    permutation = torch.tensor([4, 0, 5, 2, 1, 3])
    inverse = torch.argsort(permutation)
    with torch.no_grad():
        reference = model(observations, status, proposals)
        permuted = model(observations, status, proposals[:, permutation])
    for key in (
        "utility",
        "coarse_utility",
        "factor_logits",
        "predicted_consequence",
        "confidence_logit",
    ):
        torch.testing.assert_close(
            reference[key], permuted[key][:, inverse], rtol=1e-5, atol=1e-6
        )
    assert torch.equal(reference["fine_mask"], permuted["fine_mask"][:, inverse])


def test_trajectory_observation_attention_is_permutation_equivariant() -> None:
    torch.manual_seed(41)
    model = IndependentProposalRanker(
        _small_config(trajectory_observation_attention=True)
    ).eval()
    with torch.no_grad():
        model.trajectory_observation_gate.fill_(1.0)
    observations, status, proposals = _inputs()
    permutation = torch.tensor([4, 0, 5, 2, 1, 3])
    inverse = torch.argsort(permutation)
    with torch.no_grad():
        reference = model(observations, status, proposals)
        permuted = model(observations, status, proposals[:, permutation])
    for key in (
        "utility",
        "coarse_utility",
        "factor_logits",
        "trajectory_observation_token",
        "candidate_features",
    ):
        torch.testing.assert_close(
            reference[key],
            permuted[key][:, inverse],
            rtol=1e-5,
            atol=1e-6,
        )
    assert torch.equal(reference["fine_mask"], permuted["fine_mask"][:, inverse])


def test_trajectory_observation_attention_zero_gate_is_exact_noop() -> None:
    torch.manual_seed(42)
    model = IndependentProposalRanker(
        _small_config(trajectory_observation_attention=True)
    ).eval()
    observations, status, proposals = _inputs()
    calls = 0

    def count_call(_module, _arguments, _output):
        nonlocal calls
        calls += 1

    handle = model.scene_encoder.register_forward_hook(count_call)
    with torch.no_grad():
        reference = model(observations, status, proposals)
        for parameter in model.trajectory_observation_attention.parameters():
            parameter.add_(torch.randn_like(parameter) * 10.0)
        perturbed = model(observations, status, proposals)
    handle.remove()
    assert calls == 2
    assert reference["trajectory_observation_gate"].item() == 0.0
    torch.testing.assert_close(
        reference["candidate_features"],
        perturbed["candidate_features"],
        rtol=0,
        atol=0,
    )


def test_current_actor_auxiliary_is_candidate_independent_and_masked() -> None:
    torch.manual_seed(46)
    model = IndependentProposalRanker(
        _small_config(current_actor_auxiliary=True)
    ).eval()
    observations, status, proposals = _inputs()
    permutation = torch.tensor([4, 0, 5, 2, 1, 3])
    with torch.no_grad():
        reference = model(observations, status, proposals)
        permuted = model(observations, status, proposals[:, permutation])
    for key in (
        "current_actor_presence_logits",
        "current_actor_type_logits",
        "current_actor_state",
    ):
        torch.testing.assert_close(reference[key], permuted[key])

    target = torch.zeros(2, 3, 8)
    target[0, 0] = torch.tensor([0.0, 10.0, -2.0, 3.0, 0.5, 3.1, 4.8, 2.0])
    target[0, 1] = torch.tensor([2.0, 4.0, 1.0, 0.0, 0.0, -3.1, 1.8, 0.7])
    # Invalid scenes must be entirely ignored, including out-of-range types.
    target[1, :, 0] = 99.0
    mask = torch.tensor([[True, True, False], [True, True, True]])
    valid = torch.tensor([True, False])
    output = model(observations, status, proposals)
    losses = current_actor_auxiliary_loss(output, target, mask, valid)
    assert set(losses) == {"total", "presence", "type", "state"}
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    assert model.current_actor_state_head.weight.grad is not None


def test_shared_future_auxiliary_is_candidate_independent_and_masked() -> None:
    torch.manual_seed(47)
    config = _small_config(
        shared_future_auxiliary=True,
        shared_future_horizons=3,
    )
    model = IndependentProposalRanker(config).eval()
    observations, status, proposals = _inputs()
    permutation = torch.tensor([4, 0, 5, 2, 1, 3])
    with torch.no_grad():
        reference = model(observations, status, proposals)
        permuted = model(observations, status, proposals[:, permutation])
    expected_shapes = {
        "shared_future_presence_logits": (2, 3, 3),
        "shared_future_type_logits": (2, 3, 3, 3),
        "shared_future_actor_state": (2, 3, 3, 8),
    }
    for key, shape in expected_shapes.items():
        assert tuple(reference[key].shape) == shape
        torch.testing.assert_close(reference[key], permuted[key])

    target = torch.zeros(2, 3, 3, 8)
    target[0, :, 0] = torch.tensor(
        [0.0, 10.0, -2.0, 3.0, 0.5, 3.1, 4.8, 2.0]
    )
    target[0, 1:, 1] = torch.tensor(
        [2.0, 4.0, 1.0, 0.0, 0.0, -3.1, 1.8, 0.7]
    )
    # An invalid scene must be ignored, including its invalid actor types.
    target[1, :, :, 0] = 99.0
    mask = torch.tensor(
        [
            [[True, False, False], [True, True, False], [True, True, False]],
            [[True, True, True], [True, True, True], [True, True, True]],
        ]
    )
    valid = torch.tensor([True, False])
    output = model(observations, status, proposals)
    losses = shared_future_auxiliary_loss(output, target, mask, valid)
    assert set(losses) == {"total", "presence", "type", "state"}
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    assert model.shared_future_presence_head.weight.grad is not None
    assert model.shared_future_type_head.weight.grad is not None
    assert model.shared_future_state_head.weight.grad is not None


def test_shared_future_target_table_is_training_only_and_ordered(tmp_path) -> None:
    import hashlib
    import pandas as pd

    actor_future = np.zeros((2, 8, 16, 8), dtype=np.float32)
    actor_future[0, :, 0, 0] = 1.0
    actor_future[0, :, 0, 1] = 7.0
    actor_future[1, :, 0, 0] = 2.0
    actor_future[1, :, 0, 1] = 11.0
    actor_mask = np.zeros((2, 8, 16), dtype=bool)
    actor_mask[:, :, 0] = True
    completed = np.array([True, False])
    np.save(tmp_path / "shared_actor_future.npy", actor_future)
    np.save(tmp_path / "shared_actor_mask.npy", actor_mask)
    np.save(tmp_path / "completed.npy", completed)
    pd.DataFrame(
        {
            "scene_token": ["row-one", "row-zero"],
            "scene_index": [1, 0],
            "target_preflight_available": [True, True],
        }
    ).to_parquet(tmp_path / "scene_metadata.parquet", index=False)

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "depends_on_logged_future": True,
                "training_only_target": True,
                "available_as_model_input_at_inference": False,
                "coordinate_frame": "current_ego",
                "valid_scene_count": 1,
                "array_sha256": {
                    "shared_actor_future.npy": digest(
                        tmp_path / "shared_actor_future.npy"
                    ),
                    "shared_actor_mask.npy": digest(
                        tmp_path / "shared_actor_mask.npy"
                    ),
                    "completed.npy": digest(tmp_path / "completed.npy"),
                },
            }
        )
        + "\n"
    )

    table = load_shared_future_target_table(tmp_path)
    assert table.tokens == ["row-zero", "row-one"]
    assert tuple(table.actor_future.shape) == (2, 8, 16, 8)
    assert tuple(table.actor_masks.shape) == (2, 8, 16)
    assert table.supervision_valid.tolist() == [True, False]
    assert table.actor_future[0, 0, 0, 1].item() == pytest.approx(7.0)
    assert table.lineage["depends_on_logged_future"] is True
    assert table.lineage["available_as_model_input_at_inference"] is False


def test_load_current_actor_target_table_respects_scene_indices(tmp_path) -> None:
    import pandas as pd

    actor_slots = 2
    current = np.zeros((2, 6 + actor_slots * 9), dtype=np.float32)
    current[0, 6 : 6 + 16] = np.arange(16, dtype=np.float32)
    current[0, 6 + 16 :] = [1.0, 0.0]
    current[1, 6 : 6 + 16] = np.arange(100, 116, dtype=np.float32)
    current[1, 6 + 16 :] = [1.0, 1.0]
    np.save(tmp_path / "current.npy", current)
    np.save(tmp_path / "completed.npy", np.array([True, False]))
    pd.DataFrame(
        {
            "scene_token": ["row-one", "row-zero"],
            "scene_index": [1, 0],
            "target_preflight_available": [True, True],
        }
    ).to_parquet(tmp_path / "scene_metadata.parquet", index=False)
    (tmp_path / "store_config.json").write_text("{}\n")

    table = load_current_actor_target_table(tmp_path)
    assert table.tokens == ["row-one", "row-zero"]
    assert table.actor_states.shape == (2, actor_slots, 8)
    assert table.actor_masks.tolist() == [[True, True], [True, False]]
    assert table.supervision_valid.tolist() == [False, True]
    torch.testing.assert_close(
        table.actor_states[0].flatten(), torch.arange(100, 116).float()
    )


def test_full_current_actor_target_aggregate_matches_training_schema(tmp_path) -> None:
    shard_root = tmp_path / "shards"
    final_root = tmp_path / "final"
    feature_root = tmp_path / "features"
    shard_root.mkdir()
    feature_root.mkdir()
    for shard in range(2):
        states = np.zeros((2, 16, 8), dtype=np.float32)
        states[:, 0, 1] = np.asarray([shard * 2 + 1, shard * 2 + 2])
        masks = np.zeros((2, 16), dtype=bool)
        masks[:, 0] = True
        np.savez_compressed(
            shard_root / f"shard_{shard:03d}-of-002.npz",
            tokens=np.asarray([f"token-{shard}-0", f"token-{shard}-1"]),
            log_names=np.asarray([f"log-{shard}", f"log-{shard}"]),
            actor_states=states,
            actor_masks=masks,
        )
    result = aggregate_full_current_actor_targets(
        SimpleNamespace(
            output_root=shard_root,
            final_root=final_root,
            feature_root=feature_root,
            num_shards=2,
            expected_scenes=4,
        )
    )
    assert result["status"] == "PASS"
    assert result["current_observation_only"] is True
    assert result["future_or_evaluator_input"] is False
    table = load_current_actor_target_table(final_root)
    assert table.tokens == [
        "token-0-0",
        "token-0-1",
        "token-1-0",
        "token-1-1",
    ]
    assert table.actor_states.shape == (4, 16, 8)
    assert table.actor_masks[:, 0].all()
    assert table.supervision_valid.all()
    torch.testing.assert_close(
        table.actor_states[:, 0, 1],
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
    )


def test_conservative_reference_head_is_permutation_equivariant() -> None:
    torch.manual_seed(41)
    config = ConservativeReferenceConfig(
        model_dim=32,
        hidden_dim=64,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
    )
    head = ConservativeReferenceHead(config).eval()
    features = torch.randn(2, 6, 32)
    references = torch.tensor([1, 4])
    permutation = torch.tensor([4, 0, 5, 2, 1, 3])
    inverse = torch.argsort(permutation)
    permuted_references = inverse[references]
    with torch.no_grad():
        reference = head(features, references)
        permuted = head(features[:, permutation], permuted_references)
    for key in (
        "gain_quantiles",
        "safety_worse_logits",
        "safe_improvement_logit",
        "reference_mask",
    ):
        torch.testing.assert_close(reference[key], permuted[key][:, inverse])
    quantiles = reference["gain_quantiles"]
    assert (quantiles[..., 0] <= quantiles[..., 1]).all()
    assert (quantiles[..., 1] <= quantiles[..., 2]).all()
    assert torch.count_nonzero(quantiles[reference["reference_mask"]]) == 0


def test_conservative_reference_selection_keeps_safe_gain_and_fallback() -> None:
    quantiles = torch.tensor(
        [[
            [0.05, 0.10, 0.20],
            [0.00, 0.00, 0.00],
            [0.20, 0.30, 0.40],
            [-0.01, 0.10, 0.20],
        ]]
    )
    safety_logits = torch.full((1, 4, 3), -5.0)
    safety_logits[:, 2] = 5.0
    improvement_logits = torch.tensor([[5.0, -20.0, 5.0, 5.0]])
    reference_indices = torch.tensor([1])
    scores = conservative_reference_selection_scores(
        quantiles,
        safety_logits,
        improvement_logits,
        reference_indices,
    )
    assert scores.argmax(dim=1).item() == 0
    assert scores[0, 1].item() == 0.0
    assert torch.isneginf(scores[0, 2])
    assert torch.isneginf(scores[0, 3])

    no_switch = conservative_reference_selection_scores(
        quantiles - 1.0,
        safety_logits,
        improvement_logits,
        reference_indices,
    )
    assert no_switch.argmax(dim=1).item() == 1


def test_conservative_reference_selection_respects_allowed_candidates() -> None:
    quantiles = torch.tensor(
        [[[0.5, 0.6, 0.7], [0.0, 0.0, 0.0], [0.2, 0.3, 0.4]]]
    )
    safety_logits = torch.full((1, 3, 3), -10.0)
    improvement_logits = torch.full((1, 3), 10.0)
    references = torch.tensor([1])
    allowed = torch.tensor([[False, True, True]])
    scores = conservative_reference_selection_scores(
        quantiles,
        safety_logits,
        improvement_logits,
        references,
        allowed_candidate_mask=allowed,
    )
    assert torch.isneginf(scores[0, 0])
    assert scores.argmax(dim=1).item() == 2
    with pytest.raises(TypeError, match="boolean"):
        conservative_reference_selection_scores(
            quantiles,
            safety_logits,
            improvement_logits,
            references,
            allowed_candidate_mask=allowed.float(),
        )


def test_drivor_reference_gate_is_binary_and_current_observation_only() -> None:
    parameters = set(inspect.signature(DrivORReferenceGateRanker.forward).parameters)
    assert parameters == {
        "self",
        "scene_registers",
        "status_feature",
        "proposals",
        "reference_indices",
        "scene_valid_mask",
        "provided_alternative_indices",
        "gain_quantile_index",
        "minimum_lcb_gain",
        "maximum_safety_worse_probability",
        "minimum_safe_improvement_probability",
    }
    assert not any(
        "score" in name or "pdm" in name or "future" in name
        for name in parameters
    )
    ranker_config = _small_drivor_config()
    reference_config = ConservativeReferenceConfig(
        model_dim=32,
        hidden_dim=64,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
    )
    model = DrivORReferenceGateRanker(
        ranker_config, reference_config, alternative_mode="factor"
    ).eval()
    observations, status, proposals = _inputs()
    references = torch.tensor([1, 4])
    with torch.no_grad():
        output = model(
            observations,
            status,
            proposals,
            references,
            minimum_lcb_gain=100.0,
        )
    assert torch.equal(output["selected_indices"], references)
    assert output["allowed_candidate_mask"].sum(dim=1).le(2).all()
    assert output["allowed_candidate_mask"].gather(
        1, references[:, None]
    ).all()
    assert output["allowed_candidate_mask"].gather(
        1, output["alternative_indices"][:, None]
    ).all()


def test_drivor_reference_all_mode_scores_every_candidate() -> None:
    ranker_config = _small_drivor_config()
    reference_config = ConservativeReferenceConfig(
        model_dim=32,
        hidden_dim=64,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
    )
    model = DrivORReferenceGateRanker(
        ranker_config, reference_config, alternative_mode="all"
    ).eval()
    observations, status, proposals = _inputs()
    references = torch.tensor([1, 4])
    with torch.no_grad():
        output = model(
            observations,
            status,
            proposals,
            references,
            minimum_lcb_gain=100.0,
        )
    assert output["allowed_candidate_mask"].all()
    assert torch.equal(output["selected_indices"], references)


def test_drivor_reference_can_use_its_frozen_factor_choice_as_fallback() -> None:
    ranker_config = _small_drivor_config()
    reference_config = ConservativeReferenceConfig(
        model_dim=32,
        hidden_dim=64,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
    )
    model = DrivORReferenceGateRanker(
        ranker_config, reference_config, alternative_mode="all"
    ).eval()
    observations, status, proposals = _inputs()
    with torch.no_grad():
        raw = model.ranker(observations, status, proposals)
        expected = pdms_factor_log_utility(raw["factor_logits"]).argmax(dim=1)
        output = model(
            observations,
            status,
            proposals,
            None,
            minimum_lcb_gain=100.0,
        )
    assert torch.equal(output["reference_indices"], expected)
    assert torch.equal(output["selected_indices"], expected)
    assert output["allowed_candidate_mask"].all()


def test_drivor_reference_provided_mode_only_allows_fallback_and_base() -> None:
    ranker_config = _small_drivor_config()
    reference_config = ConservativeReferenceConfig(
        model_dim=32,
        hidden_dim=64,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
    )
    model = DrivORReferenceGateRanker(
        ranker_config, reference_config, alternative_mode="provided"
    ).eval()
    observations, status, proposals = _inputs()
    base_alternatives = torch.tensor([2, 5])
    with torch.no_grad():
        output = model(
            observations,
            status,
            proposals,
            None,
            provided_alternative_indices=base_alternatives,
            minimum_lcb_gain=100.0,
        )
    references = output["reference_indices"]
    assert torch.equal(output["selected_indices"], references)
    assert output["allowed_candidate_mask"].sum(dim=1).le(2).all()
    assert output["allowed_candidate_mask"].gather(
        1, references[:, None]
    ).all()
    assert output["allowed_candidate_mask"].gather(
        1, base_alternatives[:, None]
    ).all()


def test_drivor_reference_gate_supports_independent_top_k_shortlist() -> None:
    ranker_config = _small_drivor_config()
    reference_config = ConservativeReferenceConfig(
        model_dim=32,
        hidden_dim=64,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
    )
    model = DrivORReferenceGateRanker(
        ranker_config,
        reference_config,
        alternative_mode="factor",
        alternative_count=3,
    ).eval()
    observations, status, proposals = _inputs()
    references = torch.tensor([1, 4])
    with torch.no_grad():
        output = model(observations, status, proposals, references)
    assert output["alternative_candidate_indices"].shape == (2, 3)
    assert output["allowed_candidate_mask"].sum(dim=1).le(4).all()
    assert output["allowed_candidate_mask"].gather(
        1, output["alternative_candidate_indices"]
    ).all()
    with pytest.raises(ValueError, match="alternative_count"):
        DrivORReferenceGateRanker(
            ranker_config,
            reference_config,
            alternative_count=0,
        )


def test_drivor_reference_gate_is_candidate_permutation_equivariant() -> None:
    torch.manual_seed(44)
    ranker_config = _small_drivor_config()
    reference_config = ConservativeReferenceConfig(
        model_dim=32,
        hidden_dim=64,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
    )
    model = DrivORReferenceGateRanker(
        ranker_config, reference_config, alternative_mode="factor"
    ).eval()
    observations, status, proposals = _inputs()
    references = torch.tensor([1, 4])
    permutation = torch.tensor([4, 0, 5, 2, 1, 3])
    inverse = torch.argsort(permutation)
    permuted_references = inverse[references]
    with torch.no_grad():
        original = model(observations, status, proposals, references)
        permuted = model(
            observations,
            status,
            proposals[:, permutation],
            permuted_references,
        )
    for key in (
        "factor_logits",
        "candidate_features",
        "gain_quantiles",
        "safety_worse_logits",
        "safe_improvement_logit",
        "allowed_candidate_mask",
        "reference_selection_scores",
    ):
        torch.testing.assert_close(
            original[key], permuted[key][:, inverse], rtol=1e-5, atol=1e-6
        )
    assert torch.equal(
        original["alternative_indices"],
        permutation[permuted["alternative_indices"]],
    )


def test_drivor_reference_gate_training_is_finite_with_frozen_ranker() -> None:
    torch.manual_seed(45)
    ranker_config = _small_drivor_config()
    reference_config = ConservativeReferenceConfig(
        model_dim=32,
        hidden_dim=64,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
    )
    model = DrivORReferenceGateRanker(
        ranker_config, reference_config, alternative_mode="factor"
    )
    model.ranker.requires_grad_(False)
    model.ranker.eval()
    observations, status, proposals = _inputs(batch_size=3, candidates=6)
    base_scores = torch.randn(3, 6)
    target_factors = torch.rand(3, 6, 7)
    target_factors[..., -1] = torch.tensor(
        [
            [0.4, 0.8, 0.2, 0.7, 0.3, 0.6],
            [0.9, 0.3, 0.1, 0.4, 0.8, 0.2],
            [0.5, 0.2, 0.7, 0.1, 0.9, 0.4],
        ]
    )
    args = SimpleNamespace(
        reference_mode="base",
        alternative_mode="factor",
        minimum_improvement_target=0.005,
        factor_epsilon=1.0e-6,
        minimum_pair_delta=0.02,
        safety_worse_positive_weight=10.0,
        safe_improvement_positive_weight=3.0,
        switch_margin_temperature=0.05,
        quantile_weight=1.0,
        median_rank_weight=0.25,
        safety_weight=1.0,
        improvement_weight=0.5,
        false_switch_weight=0.5,
        missed_improvement_weight=0.25,
    )
    batch = (
        proposals,
        observations,
        torch.ones(3, 13, dtype=torch.bool),
        status,
        base_scores,
        target_factors,
        torch.arange(3),
        torch.empty(3, 0, 8),
        torch.empty(3, 0, dtype=torch.bool),
        torch.zeros(3, dtype=torch.bool),
    )
    loss, details = compute_gate_training_loss(
        model,
        batch,
        args,
        torch.Generator().manual_seed(450),
    )
    assert torch.isfinite(loss)
    assert all(np.isfinite(value) for value in details.values())
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.reference_head.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(value).all() for value in gradients)
    assert all(parameter.grad is None for parameter in model.ranker.parameters())


def test_conservative_reference_training_accepts_training_only_batch_tail() -> None:
    torch.manual_seed(46)
    model = IndependentConservativeReferenceRanker(
        _small_config(),
        ConservativeReferenceConfig(
            model_dim=32,
            hidden_dim=64,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
        ),
    )
    model.ranker.requires_grad_(False)
    model.ranker.eval()
    observations, status, proposals = _inputs(batch_size=3, candidates=6)
    base_scores = torch.randn(3, 6)
    target_factors = torch.rand(3, 6, 7)
    args = SimpleNamespace(
        minimum_improvement_target=0.005,
        factor_epsilon=1.0e-6,
        minimum_lcb_gain=0.0,
        maximum_safety_worse_probability=0.25,
        minimum_safe_improvement_probability=0.5,
        minimum_pair_delta=0.02,
        safety_worse_positive_weight=10.0,
        safe_improvement_positive_weight=3.0,
        switch_margin_temperature=0.05,
        quantile_weight=1.0,
        median_rank_weight=0.25,
        safety_weight=1.0,
        improvement_weight=0.5,
        false_switch_weight=0.5,
        missed_improvement_weight=0.0,
    )
    batch = (
        proposals,
        observations,
        torch.ones(3, 13, dtype=torch.bool),
        status,
        base_scores,
        target_factors,
        torch.arange(3),
        torch.empty(3, 0, 8),
        torch.empty(3, 0, dtype=torch.bool),
        torch.zeros(3, dtype=torch.bool),
    )
    loss, details = compute_reference_training_loss(
        model,
        batch,
        args,
        torch.Generator().manual_seed(460),
    )
    assert torch.isfinite(loss)
    assert all(np.isfinite(value) for value in details.values())


def test_drivor_promotion_records_lock_source_and_artifact_hash(tmp_path) -> None:
    artifact_path = tmp_path / "ranker.pt"
    artifact_path.write_bytes(b"ranker")
    ranker = {
        "selection_mode": "factor",
        "training_manifest": {"checkpoint_selection_source": "public_base"},
        "validation_by_source": {
            "public_base": {
                "factor_selected_pdms": 0.96,
                "base_selected_pdms": 0.95,
                "factor_selected_delta": 0.01,
                "factor_selected_delta_log_bootstrap_95ci": [0.002, 0.018],
            }
        },
        "epoch": 3,
    }
    record = _ranker_record(ranker, artifact_path)
    assert record["selection_source"] == "public_base"
    assert record["validation_delta_log_bootstrap_95ci"][0] > 0
    assert len(record["sha256"]) == 64

    gate_path = tmp_path / "gate.pt"
    gate_path.write_bytes(b"gate")
    policy = {
        "selected_pdms": 0.955,
        "base_selected_pdms": 0.95,
        "delta": 0.005,
        "delta_log_bootstrap_95ci": [0.001, 0.009],
    }
    gate = {
        "alternative_mode": "factor",
        "training_manifest": {"selection_source": "public_base"},
        "validation_by_source": {"public_base": {"best_policy": policy}},
        "selected_policy": policy,
        "epoch": 2,
    }
    gate_record = _gate_record(gate, gate_path)
    assert gate_record["selection_mode"] == "factor"
    assert gate_record["validation_locked_policy"] == policy


def test_drivor_shortlist_top_k_parser_is_sorted_and_bounded() -> None:
    assert _parse_top_k("8,1,4,8") == (1, 4, 8)
    with pytest.raises(ValueError, match=r"\[1, 64\]"):
        _parse_top_k("0,1")
    with pytest.raises(ValueError, match=r"\[1, 64\]"):
        _parse_top_k("65")


def test_drivor_direct_score_regression_calibrates_candidate_pdms() -> None:
    target = torch.tensor([[0.2, 0.8]])
    calibrated = torch.logit(target)
    uncalibrated = torch.zeros_like(target)
    assert direct_score_regression_loss(
        calibrated, target, beta=0.1
    ) < direct_score_regression_loss(uncalibrated, target, beta=0.1)
    with pytest.raises(ValueError, match="beta"):
        direct_score_regression_loss(calibrated, target, beta=0.0)


def test_masked_pinball_quantile_loss_prefers_calibrated_predictions() -> None:
    target = torch.tensor([[0.2, -0.1, 0.0]])
    calibrated = target.unsqueeze(-1).expand(-1, -1, 3).clone()
    shifted = calibrated + 0.4
    valid = torch.tensor([[True, True, False]])
    calibrated_loss = masked_pinball_quantile_loss(
        calibrated, target, valid_mask=valid
    )
    shifted_loss = masked_pinball_quantile_loss(
        shifted, target, valid_mask=valid
    )
    assert calibrated_loss.item() == 0.0
    assert shifted_loss > calibrated_loss


def test_conservative_ranker_uses_reference_index_not_numeric_score() -> None:
    torch.manual_seed(43)
    model = IndependentConservativeReferenceRanker(
        _small_config(),
        ConservativeReferenceConfig(
            model_dim=32,
            hidden_dim=64,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
        ),
    ).eval()
    observations, status, proposals = _inputs()
    references = torch.tensor([1, 4])
    with torch.no_grad():
        output = model(
            observations,
            status,
            proposals,
            references,
            minimum_lcb_gain=100.0,
        )
    assert output["reference_selection_scores"].shape == (2, 6)
    assert torch.equal(
        output["reference_selection_scores"].argmax(dim=1), references
    )
    assert output["gain_quantiles"].shape == (2, 6, 3)


def test_shared_scene_is_computed_once_for_all_candidates() -> None:
    model = IndependentProposalRanker(_small_config()).eval()
    observations, status, proposals = _inputs(candidates=7)
    calls = 0

    def count_call(_module, _arguments, _output):
        nonlocal calls
        calls += 1

    handle = model.scene_encoder.register_forward_hook(count_call)
    try:
        with torch.no_grad():
            output = model(observations, status, proposals)
    finally:
        handle.remove()
    assert calls == 1
    assert output["utility"].shape == (2, 7)
    assert output["factor_logits"].shape == (2, 7, len(FACTOR_KEYS))


def test_final_selection_is_restricted_to_coarse_shortlist() -> None:
    model = IndependentProposalRanker(_small_config(fine_top_k=3)).eval()
    observations, status, proposals = _inputs(candidates=7)
    with torch.no_grad():
        output = model(observations, status, proposals)
    selected = output["utility"].argmax(dim=1)
    assert output["fine_mask"].gather(1, selected[:, None]).all()
    assert torch.equal(output["utility"] > -9999.0, output["fine_mask"])
    assert output["refined_utility"].shape == (2, 3)


def test_geometry_features_are_finite_at_stationary_points() -> None:
    proposals = torch.zeros(1, 2, 8, 3)
    features = ProposalTrajectoryEncoder._geometry_features(proposals)
    assert features.shape == (1, 2, 8, 11)
    assert torch.isfinite(features).all()


def test_pairwise_loss_rewards_correct_ordering() -> None:
    targets = torch.tensor([[0.9, 0.6, 0.1]])
    correct = torch.tensor([[3.0, 1.0, -2.0]], requires_grad=True)
    reversed_scores = -correct.detach()
    correct_loss = weighted_pairwise_rank_loss(correct, targets, minimum_target_delta=0.02)
    reversed_loss = weighted_pairwise_rank_loss(
        reversed_scores, targets, minimum_target_delta=0.02
    )
    assert correct_loss < reversed_loss
    correct_loss.backward()
    assert torch.isfinite(correct.grad).all()


def test_empty_pairwise_mask_is_differentiable_zero() -> None:
    prediction = torch.zeros(2, 4, requires_grad=True)
    targets = torch.zeros_like(prediction)
    loss = weighted_pairwise_rank_loss(prediction, targets)
    assert loss.item() == 0.0
    loss.backward()
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))


def test_top_regret_loss_focuses_oracle_candidate() -> None:
    targets = torch.tensor([[0.2, 0.95, 0.7, 0.1]])
    correct = torch.tensor([[0.0, 3.0, 1.0, -1.0]], requires_grad=True)
    wrong = torch.tensor([[0.0, -2.0, 1.0, 3.0]])
    correct_loss = top_regret_rank_loss(correct, targets)
    wrong_loss = top_regret_rank_loss(wrong, targets)
    assert correct_loss < wrong_loss
    correct_loss.backward()
    assert torch.isfinite(correct.grad).all()


def test_factor_and_listwise_losses_are_finite() -> None:
    generator = torch.Generator().manual_seed(3)
    logits = torch.randn(2, 5, len(FACTOR_KEYS), generator=generator)
    targets = torch.rand(2, 5, len(FACTOR_KEYS), generator=generator)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    factor_loss = factor_prediction_loss(logits, targets, mask)
    listwise_loss = top_heavy_listwise_loss(
        torch.randn(2, 5, generator=generator),
        torch.rand(2, 5, generator=generator),
    )
    assert torch.isfinite(factor_loss)
    assert torch.isfinite(listwise_loss)
    assert pdms_factor_log_utility(logits).shape == (2, 5)


def test_physical_log_split_removes_segment_suffix() -> None:
    first = "2021.07.16.18.06.21_veh-38_02197_03220"
    second = "2021.07.16.18.06.21_veh-38_03221_04000"
    assert physical_log_name(first) == "2021.07.16.18.06.21_veh-38"
    assert physical_log_name(first) == physical_log_name(second)
    assignment = assign_balanced_physical_log_folds(
        [physical_log_name(first), physical_log_name(second), "log_b", "log_c"],
        num_folds=2,
        seed=7,
    )
    assert assignment[physical_log_name(first)] in {0, 1}
    assert set(assignment) == {physical_log_name(first), "log_b", "log_c"}


def test_external_score_evaluator_uses_physical_logs() -> None:
    first = "2021.07.16.18.06.21_veh-38_02197_03220"
    second = "2021.07.16.18.06.21_veh-38_03221_04000"
    assert external_score_physical_log_name(first) == physical_log_name(first)
    assert external_score_physical_log_name(second) == physical_log_name(second)


def test_external_score_pairwise_accuracy_respects_target_delta() -> None:
    targets = np.asarray([[0.90, 0.86, 0.20], [0.10, 0.80, 0.40]], dtype=np.float32)
    predictions = np.asarray([[-1.0, 3.0, -2.0], [-3.0, 2.0, 1.0]], dtype=np.float32)
    # The first row's 0.04 pair is deliberately reversed.  It counts for the
    # non-tie audit but is excluded by the >=0.05 planning-relevance gate.
    assert external_score_pairwise_accuracy(predictions, targets, 1e-9) == pytest.approx(
        5.0 / 6.0
    )
    assert external_score_pairwise_accuracy(predictions, targets, 0.05) == 1.0


def test_external_score_cluster_bootstrap_is_deterministic_and_clustered() -> None:
    delta = np.asarray([0.01, 0.01, -0.002, 0.02], dtype=np.float32)
    logs = ["log_a", "log_a", "log_b", "log_c"]
    first = external_score_cluster_bootstrap(delta, logs, iterations=1000, seed=17)
    second = external_score_cluster_bootstrap(delta, logs, iterations=1000, seed=17)
    assert first == second
    assert first["mean"] == pytest.approx(float(delta.mean()))
    assert first["ci95_low"] < first["ci95_high"]


def test_base_shortlist_reranking_never_selects_outside_base_topk() -> None:
    base_scores = torch.tensor([[0.9, 0.8, 0.1, 0.0], [0.1, 0.7, 0.8, 0.2]])
    reranker = torch.tensor([[0.0, 1.0, 100.0, 50.0], [100.0, 0.0, 1.0, 50.0]])
    target_pdms = torch.tensor([[0.6, 0.9, 1.0, 0.1], [1.0, 0.5, 0.9, 0.2]])
    target_factors = torch.zeros(2, 4, 7)
    target_factors[..., -1] = target_pdms
    metrics = evaluate_base_shortlist_reranking(
        reranker,
        base_scores,
        target_factors,
        ["log_a", "log_b"],
        shortlist_size=2,
        seed=9,
        bootstrap_replicates=100,
    )
    # Scene 0 switches from candidate 0 to candidate 1. Scene 1 keeps
    # candidate 2. The very large utilities outside Base top-2 are ignored.
    assert metrics["selected_pdms"] == pytest.approx(0.9)
    assert metrics["base_selected_pdms"] == pytest.approx(0.75)
    assert metrics["switch_rate"] == pytest.approx(0.5)
    assert metrics["base_numeric_score_used_by_reranker"] is False
    assert metrics["base_rank_used_for_shortlist"] is True


def test_base_shortlist_top1_is_exact_base_selection() -> None:
    base_scores = torch.tensor([[0.2, 0.9, 0.3]])
    reranker = torch.tensor([[100.0, -100.0, 50.0]])
    target_factors = torch.zeros(1, 3, 7)
    target_factors[..., -1] = torch.tensor([[0.8, 0.4, 1.0]])
    metrics = evaluate_base_shortlist_reranking(
        reranker,
        base_scores,
        target_factors,
        ["log_a"],
        shortlist_size=1,
        seed=10,
        bootstrap_replicates=20,
    )
    assert metrics["selected_pdms"] == pytest.approx(0.4)
    assert metrics["selected_delta"] == pytest.approx(0.0)
    assert metrics["switch_rate"] == pytest.approx(0.0)


def test_base_shortlist_metrics_cover_both_independent_heads() -> None:
    coarse = torch.zeros(2, 16)
    factors = torch.zeros(2, 16, len(FACTOR_KEYS))
    base = torch.arange(16, dtype=torch.float32).repeat(2, 1)
    targets = torch.zeros(2, 16, 7)
    targets[..., -1] = torch.linspace(0.0, 1.0, 16).repeat(2, 1)
    result = collect_base_shortlist_metrics(
        coarse,
        factors,
        base,
        targets,
        ["log_a", "log_b"],
        seed=11,
        bootstrap_replicates=20,
    )
    assert set(result) == {
        f"{mode}_base_top{k}"
        for mode in ("independent_coarse", "independent_factor")
        for k in (2, 4, 8, 16)
    }


@pytest.mark.parametrize("shortlist_size", [0, 5])
def test_base_shortlist_rejects_invalid_k(shortlist_size: int) -> None:
    with pytest.raises(ValueError, match="shortlist_size"):
        evaluate_base_shortlist_reranking(
            torch.zeros(1, 4),
            torch.zeros(1, 4),
            torch.zeros(1, 4, 7),
            ["log_a"],
            shortlist_size=shortlist_size,
            seed=12,
            bootstrap_replicates=20,
        )


def test_independent_shortlist_artifact_records_no_numeric_base_input(tmp_path) -> None:
    ranker_path = tmp_path / "ranker.pt"
    base_path = tmp_path / "base.ckpt"
    base_path.write_bytes(b"immutable-base")
    model = IndependentProposalRanker(_small_config())
    ranker = {
        "architecture": "IndependentProposalRanker",
        "epoch": 3,
        "model_config": asdict(model.config),
        "state_dict": model.state_dict(),
        "refit_all_logs": True,
        "validation_performed": False,
        "refit_provenance": {"selection_artifact_sha256": "a" * 64},
    }
    torch.save(ranker, ranker_path)
    artifact = build_independent_shortlist_artifact(
        ranker,
        ranker_artifact_path=ranker_path,
        base_checkpoint_path=base_path,
        shortlist_size=4,
        score_mode="coarse",
    )
    assert artifact["artifact_type"] == IndependentShortlistScorerAgent.ARTIFACT_TYPE
    assert artifact["base_numeric_score_used_by_independent_ranker"] is False
    assert artifact["base_rank_used_for_shortlist"] is True
    assert artifact["shortlist_size"] == 4
    assert artifact["source_ranker_refit_all_logs"] is True
    assert artifact["source_ranker_validation_performed"] is False
    assert artifact["source_ranker_refit_provenance"] == (
        ranker["refit_provenance"]
    )
    assert "official_pdm_score" in artifact["forbidden_inputs"]


def test_private_visual_pooling_preserves_crop_mask_and_constants() -> None:
    first = torch.ones(1, 256, 8)
    second = torch.full((2, 256, 8), 3.0)
    pooled, valid = pool_visual_tokens(
        torch.cat((first, second), dim=0),
        crop_counts=[1, 2],
        pool_grid=4,
        max_crops=3,
    )
    assert pooled.shape == (2, 48, 8)
    assert valid.shape == (2, 48)
    assert valid.sum(dim=1).tolist() == [16, 32]
    torch.testing.assert_close(pooled[0, :16], torch.ones(16, 8, dtype=torch.float16))
    torch.testing.assert_close(
        pooled[1, :32], torch.full((32, 8), 3.0, dtype=torch.float16)
    )
    assert torch.count_nonzero(pooled[0, 16:]) == 0


def test_private_observation_cache_joins_replay_by_token(tmp_path) -> None:
    feature_root = tmp_path / "features"
    label_root = tmp_path / "labels"
    observation_root = tmp_path / "observations"
    relative_dir = "all_shard_000-of-001"
    for root in (feature_root, label_root, observation_root):
        (root / relative_dir).mkdir(parents=True)

    tokens = ["scene_a", "scene_b"]
    logs = ["log_a_00000_00001", "log_b_00000_00001"]
    torch.save(
        {
            "tokens": tokens,
            "log_names": logs,
            "proposals": torch.zeros(2, 64, 8, 3),
            "base_scores": torch.zeros(2, 64),
            "scene_features": torch.stack(
                (
                    torch.full((16, 256), 3.0),
                    torch.full((16, 256), 4.0),
                )
            ).half(),
            "ego_features": torch.stack(
                (
                    torch.full((1, 256), 5.0),
                    torch.full((1, 256), 6.0),
                )
            ).half(),
        },
        feature_root / relative_dir / "chunk_000000.pt",
    )
    (feature_root / relative_dir / "manifest.json").write_text(
        json.dumps({"checkpoint_sha256": "proposal-hash", "checkpoint": "proposal.ckpt"})
    )
    torch.save(
        {
            "tokens": tokens,
            "log_names": logs,
            "target_factor_keys": (
                "no_at_fault_collisions",
                "drivable_area_compliance",
                "ego_progress",
                "time_to_collision_within_bound",
                "comfort",
                "driving_direction_compliance",
                "score",
            ),
            "valid_mask": torch.ones(2, dtype=torch.bool),
            "target_factors": torch.ones(2, 64, 7),
        },
        label_root / relative_dir / "chunk_000000.pt",
    )
    # Deliberately reverse cache rows; the join must use scene token, not row.
    visual = torch.stack((torch.full((5, 48), 2.0), torch.full((5, 48), 1.0)))
    torch.save(
        {
            "tokens": ["scene_b", "scene_a"],
            "visual_tokens": visual.half(),
            "visual_valid_mask": torch.ones(2, 5, dtype=torch.bool),
            "status_feature": torch.tensor(
                [[20.0] * 8, [10.0] * 8], dtype=torch.float32
            ),
            "history_trajectory": torch.tensor(
                [[[2.0] * 3] * 4, [[1.0] * 3] * 4], dtype=torch.float32
            ),
            "high_command_one_hot": torch.tensor(
                [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
                dtype=torch.float32,
            ),
        },
        observation_root / relative_dir / "chunk_000000.pt",
    )
    (observation_root / relative_dir / "manifest.json").write_text(
        json.dumps(
            {
                "shard_count": 1,
                "shard_index": 0,
                "scene_count": 2,
                "checkpoint_sha256": "visual-hash",
                "current_observation_only": True,
                "future_or_evaluator_input": False,
            }
        )
    )

    data, lineage = load_replay_sources(
        [ReplaySource("base", feature_root, label_root)],
        private_observation_root=observation_root,
        retain_m0_context=True,
    )
    assert data.observation_row_indices.tolist() == [1, 0]
    torch.testing.assert_close(
        data.observation_tokens[data.observation_row_indices[0]],
        torch.full((5, 48), 1.0, dtype=torch.float16),
    )
    first_context = data.ego_features[data.observation_row_indices[0]].tolist()
    second_context = data.ego_features[data.observation_row_indices[1]].tolist()
    assert first_context == [10.0] * 8 + [1.0] * 12 + [1.0, 0.0, 0.0, 0.0]
    assert second_context == [20.0] * 8 + [2.0] * 12 + [0.0, 1.0, 0.0, 0.0]
    assert lineage[-1]["current_observation_only"] is True
    assert lineage[-1]["current_context_width"] == 24
    torch.testing.assert_close(
        data.m0_scene_features[0],
        torch.full((16, 256), 3.0, dtype=torch.float16),
    )
    torch.testing.assert_close(
        data.m0_ego_features[1],
        torch.full((1, 256), 6.0, dtype=torch.float16),
    )

    scene_data, _ = load_replay_sources(
        [ReplaySource("base", feature_root, label_root)],
        private_observation_root=None,
    )
    assert scene_data.observation_row_indices.tolist() == [0, 1]
    assert scene_data.observation_valid_masks.all()
    torch.testing.assert_close(
        scene_data.observation_tokens[0],
        torch.full((16, 256), 3.0, dtype=torch.float16),
    )
    torch.testing.assert_close(
        scene_data.ego_features[1],
        torch.full((256,), 6.0, dtype=torch.float16),
    )
