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
    FACTOR_KEYS,
    IndependentConservativeReferenceRanker,
    IndependentProposalRanker,
    IndependentRankerConfig,
    ProposalTrajectoryEncoder,
    assert_current_observation_only,
    conservative_reference_selection_scores,
    current_actor_auxiliary_loss,
    episode_drive_factor_loss,
    factor_prediction_loss,
    masked_pinball_quantile_loss,
    pdms_factor_log_utility,
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
from local_stage2.independent_scorer_agent import (
    IndependentShortlistScorerAgent,
    build_independent_shortlist_artifact,
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
    compute_residual_training_loss,
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
        factor_weight=1.0,
        private_factor_weight=0.25,
        factor_rank_weight=0.5,
        relative_safety_weight=0.5,
        residual_l2_weight=0.01,
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
    loss.backward()
    assert model.factor_delta_head[-1].weight.grad is not None
    assert model.private_ranker.coarse_factor_heads[
        FACTOR_KEYS[0]
    ].weight.grad is not None


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
