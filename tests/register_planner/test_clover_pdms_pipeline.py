from __future__ import annotations

import inspect
import os
import pickle
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from starVLA.model.modules.register_planner import (
    CloverStage1TrajectoryLoss,
    CloverStage2GeneratorLoss,
    TeacherTargetSets,
    build_teacher_target_sets,
    clover_inter_trajectory_loss,
    selected_set_enrichment_per_scene,
    set_coverage_l1,
)
from starVLA.model.framework.QwenRegisterClover import QwenRegisterClover
from starVLA.model.modules.trajectory_scorer.drivor_dynamic_scorer import (
    DrivoRDynamicScorer,
)
from starVLA.model.modules.trajectory_scorer.losses import (
    DRIVOR_METRICS,
    PDMSValueLoss,
)
from starVLA.training import (
    build_register_candidate_bank,
    train_register_clover_refinement,
    train_register_clover_stage1,
)
from starVLA.training.clover_pseudo_experts import CloverPseudoExpertStore
from starVLA.training.config_loader import load_training_config
from starVLA.training.navsim_metric_supervisor import (
    _v1_two_way_progress_and_score,
)
from starVLA.training.register_stage_utils import (
    validate_bank_only_training_profile,
)
from starVLA.training.train_register_generator import FirstBackwardGradientGate
from tools.select_register64_clover_checkpoint_pair import select_best_pair


def _metric_logits(values: torch.Tensor) -> dict[str, torch.Tensor]:
    eps = torch.finfo(values.dtype).eps
    logits = torch.logit(values.clamp(eps, 1.0 - eps))
    return {name: logits.clone() for name in DRIVOR_METRICS}


def test_clover_set_coverage_matches_target_to_nearest_proposal():
    proposals = torch.zeros(1, 2, 8, 3)
    proposals[:, 1] = 2.0
    targets = torch.zeros(1, 3, 8, 3)
    targets[:, 0] = 0.5
    targets[:, 1] = 1.5
    targets[:, 2] = 100.0
    mask = torch.tensor([[True, True, False]])
    # Each valid target is 0.5 away in all three state dimensions.
    torch.testing.assert_close(
        set_coverage_l1(proposals, targets, mask), torch.tensor(1.5)
    )


def test_clover_stage1_keeps_gt_and_pseudo_expert_terms_separate():
    proposals = torch.zeros(1, 2, 8, 3, requires_grad=True)
    ground_truth = torch.ones(1, 8, 3)
    pseudo = torch.full((1, 1, 8, 3), 2.0)
    output = CloverStage1TrajectoryLoss(
        gt_weight=1.0, pseudo_expert_weight=0.5
    )(proposals, ground_truth, pseudo, torch.ones(1, 1, dtype=torch.bool))
    torch.testing.assert_close(output.gt_loss, torch.tensor(3.0))
    torch.testing.assert_close(output.pseudo_expert_loss, torch.tensor(6.0))
    torch.testing.assert_close(output.loss, torch.tensor(6.0))
    output.loss.backward()
    assert proposals.grad is not None


class _ImmediateMetricFuture:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class _TinyMetricSupervisor:
    def score_async(self, tokens, proposals):
        batch, candidates = proposals.shape[:2]
        target = torch.full(
            (batch, candidates),
            0.75,
            device=proposals.device,
            dtype=torch.float32,
        )
        return _ImmediateMetricFuture(
            {
                **{name: target.clone() for name in DRIVOR_METRICS},
                "aggregate_score": target,
            }
        )


def test_real_clover_stage1_backward_reaches_every_trainable_parameter(tiny_factory):
    config = tiny_factory.config(4)
    config.framework.name = "QwenRegisterClover"
    config.framework.drivor_scorer.aggregate_head = True
    config.framework.drivor_scorer.selection_mode = "learned_aggregate"
    config.framework.clover_loss = {
        "gt_weight": 1.0,
        "pseudo_expert_weight": 0.5,
        "scorer_weight": 1.0,
        "submetric_weight": 1.0,
        "aggregate_weight": 1.0,
        "listwise_weight": 1.0,
        "pairwise_weight": 0.5,
    }
    model = QwenRegisterClover(
        config,
        qwen_vl_interface=tiny_factory.qwen(),
        qwen_hidden_extractor=tiny_factory.extractor,
    )
    gate = FirstBackwardGradientGate(model)
    examples = tiny_factory.examples()
    pseudo = torch.zeros(len(examples), 1, 8, 3)
    output = model(
        examples,
        clover_supervisor=_TinyMetricSupervisor(),
        pseudo_experts=pseudo,
        pseudo_expert_mask=torch.ones(len(examples), 1, dtype=torch.bool),
    )
    output["loss"].backward()
    assert gate.missing_local() == []
    gate.close()


def test_clover_teacher_targets_include_scalar_topk_and_vector_pareto():
    proposals = torch.arange(4.0).reshape(1, 4, 1, 1).expand(-1, -1, 8, 3)
    scalar = torch.tensor([[0.2, 0.9, 0.7, 0.1]])
    rewards = torch.tensor([[0.3, 0.8, 0.7, 0.2]])
    targets = build_teacher_target_sets(
        proposals,
        _metric_logits(rewards),
        scalar,
        topk=2,
        pareto_max_size=2,
        pareto_min_size=1,
        reward_threshold=0.0,
    )
    assert targets.topk_indices.tolist() == [[1, 2]]
    assert targets.pareto_indices[0, 0].item() == 1
    assert targets.pareto_mask.tolist() == [[True, False]]


def test_clover_inter_loss_is_released_closest_pair_objective():
    proposals = torch.zeros(1, 3, 8, 3)
    proposals[:, 1] = 2.0
    proposals[:, 2] = 5.0
    # Closest off-diagonal L1 trajectory distance is 6.0.
    torch.testing.assert_close(
        clover_inter_trajectory_loss(proposals), torch.tensor(-1.0)
    )
    # The exact donor replaces all zeros by one, including the diagonal; this
    # makes its returned closest distance one in this particular pool.


def test_clover_stability_is_aligned_across_all_registers():
    teacher = torch.zeros(1, 4, 8, 3)
    student = teacher.clone()
    student[:, 3] = 1.0
    targets = TeacherTargetSets(
        topk_trajectories=teacher[:, :1],
        topk_mask=torch.ones(1, 1, dtype=torch.bool),
        topk_indices=torch.zeros(1, 1, dtype=torch.long),
        pareto_trajectories=teacher[:, :1],
        pareto_mask=torch.ones(1, 1, dtype=torch.bool),
        pareto_indices=torch.zeros(1, 1, dtype=torch.long),
        predicted_rewards=torch.zeros(1, 4, len(DRIVOR_METRICS)),
        predicted_aggregate=torch.zeros(1, 4),
    )
    output = CloverStage2GeneratorLoss(
        trajectory_weight=0.0,
        diversity_weight=0.0,
        topk_weight=0.0,
        pareto_weight=0.0,
        stability_weight=1.0,
    )([student], torch.zeros(1, 8, 3), teacher, targets)
    # One of four corresponding registers differs by L1=3 at every pose.
    torch.testing.assert_close(output.stability_loss, torch.tensor(0.75))
    torch.testing.assert_close(output.loss, output.stability_loss)


def test_pdms_value_head_trains_direct_score_and_detaches_geometry():
    scorer = DrivoRDynamicScorer(
        scene_dim=16,
        model_dim=16,
        ffn_dim=32,
        num_layers=1,
        num_heads=1,
        decoder_style="donor_register",
        proj_drop=0.0,
        drop_path=0.0,
        aggregate_head=True,
        selection_mode="learned_aggregate",
    )
    proposals = torch.randn(2, 4, 8, 3, requires_grad=True)
    scene = torch.randn(2, 3, 16, requires_grad=True)
    output = scorer(proposals, scene, torch.randn(2, 4), topm=4)
    targets = {
        name: torch.rand(2, 4)
        for name in (*DRIVOR_METRICS, "aggregate_score")
    }
    loss, details = PDMSValueLoss()(
        output.metric_logits, output.aggregate_logit, targets
    )
    loss.backward()
    assert output.aggregate_logit is not None
    assert output.aggregate_score.shape == (2, 4)
    assert proposals.grad is None
    assert scene.grad is not None and torch.count_nonzero(scene.grad)
    assert {"aggregate", "listwise", "pairwise", "submetric"} <= set(details)


def test_calibrated_selector_retains_direct_and_structured_endpoints():
    scorer = DrivoRDynamicScorer(
        scene_dim=8,
        model_dim=8,
        ffn_dim=16,
        num_layers=1,
        num_heads=1,
        decoder_style="donor_register",
        aggregate_head=True,
        selection_mode="calibrated_hybrid",
    )
    direct = torch.tensor([[0.1, 3.0, 0.2]])
    structured = torch.tensor([[2.0, 0.0, 1.0]])
    assert scorer.calibrated_hybrid_score(
        direct, structured, alpha=0.0
    ).argmax(dim=1).item() == 1
    assert scorer.calibrated_hybrid_score(
        direct, structured, alpha=1.0
    ).argmax(dim=1).item() == 0
    scorer.set_selection_alpha(0.35)
    assert torch.isclose(scorer.state_dict()["selection_alpha"], torch.tensor(0.35))


def test_calibrated_selector_rejects_noop_aggregate_temperature():
    with pytest.raises(ValueError, match="non-tunable compatibility value"):
        DrivoRDynamicScorer(
            scene_dim=8,
            model_dim=8,
            ffn_dim=16,
            num_layers=1,
            num_heads=1,
            decoder_style="donor_register",
            aggregate_head=True,
            selection_mode="calibrated_hybrid",
            aggregate_temperature=0.5,
        )


def test_official_pseudo_expert_selection_is_not_random_perturbation(tmp_path):
    trajectories = [np.full((8, 3), value, np.float32) for value in (0.0, 1.0, 2.0)]
    payload = [
        {
            "valid": True,
            "token": "scene",
            "trajectories_relative": trajectories,
            "scores": [
                {"pdm_score": 0.95},
                {"pdm_score": 0.90},
                {"pdm_score": 0.20},
            ],
        }
    ]
    path = tmp_path / "pseudo.pkl"
    with path.open("wb") as stream:
        pickle.dump(payload, stream)
    store = CloverPseudoExpertStore(path, top_k=3, score_threshold=0.8)
    selected, mask = store.select("scene", np.full((8, 3), 4.0, np.float32))
    assert mask.tolist() == [True, True, True]
    np.testing.assert_array_equal(selected[0], trajectories[0])
    np.testing.assert_array_equal(selected[1], trajectories[1])
    # GT is appended because the two evaluator-filtered candidates do not cover it.
    np.testing.assert_array_equal(selected[2], np.full((8, 3), 4.0, np.float32))


def test_pseudo_expert_missing_scene_uses_logged_expert(tmp_path):
    payload = [
        {
            "valid": True,
            "token": "available",
            "trajectories_relative": [np.zeros((8, 3), np.float32)],
            "scores": [{"pdm_score": 0.9}],
        },
        {
            "valid": False,
            "token": "invalid",
            "trajectories_relative": [np.ones((8, 3), np.float32)],
            "scores": [{"pdm_score": 1.0}],
        },
    ]
    path = tmp_path / "pseudo.pkl"
    with path.open("wb") as stream:
        pickle.dump(payload, stream)
    store = CloverPseudoExpertStore(path, top_k=2)
    assert not store.contains("invalid")
    ground_truth = np.full((8, 3), 3.0, np.float32)
    selected, mask = store.select("missing", ground_truth)
    assert mask.tolist() == [True, False]
    np.testing.assert_array_equal(selected[0], ground_truth)


def test_v1_batch_scoring_reconstructs_exact_per_candidate_two_way_pdms():
    # expert + three candidates. Candidate 1 has more raw progress than the
    # expert; candidate 0 must still be normalized against the expert (0.5),
    # not candidate 1 as a pooled normalization would do (0.25).
    scorer = SimpleNamespace(
        _multi_metrics=np.array(
            [[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
            dtype=np.float64,
        ),
        _weighted_metrics=np.array(
            [
                [1.0, 0.25, 1.0, 0.0],
                [1.0, 0.8, 0.6, 0.9],
                [1.0, 0.5, 1.0, 0.2],
                [1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        _progress_raw=np.array([10.0, 5.0, 20.0, 100.0]),
        _config=SimpleNamespace(
            progress_distance_threshold=5.0,
            weighted_metrics_array=np.array([5.0, 5.0, 2.0, 0.0]),
        ),
    )
    progress, score = _v1_two_way_progress_and_score(scorer)
    np.testing.assert_allclose(progress, [0.5, 1.0, 0.0])
    np.testing.assert_allclose(score, [7.5 / 12.0, 10.0 / 12.0, 0.0])


def test_clover_configs_pin_v1_labels_and_donor_recipe():
    stage1 = load_training_config(
        "starVLA/config/training/qwen_register64_clover_pdms_stage1.yaml"
    )
    bank = load_training_config(
        "starVLA/config/training/register64_clover_pdms_bank.yaml"
    )
    scorer = load_training_config(
        "starVLA/config/training/register64_clover_pdms_scorer.yaml"
    )
    assert stage1.metric_supervisor.protocol == "navsim_v1_1_pdms_two_way"
    assert stage1.framework.generator_loss.stage_loss_mode == "final_only"
    assert bank.candidate_bank.label_protocol == "navsim_v1_1_pdms_two_way"
    assert bank.candidate_bank.splits.selection.dataset_split == "train"
    assert scorer.model.selection_mode == "calibrated_hybrid"
    assert "selection_root" in scorer.candidate_bank
    validate_bank_only_training_profile(
        scorer, expected_name="clover_pdms_value_bank_v1"
    )


def test_clover_stage_boundaries_do_not_restore_flow_or_suprim():
    stage1_source = inspect.getsource(train_register_clover_stage1)
    refinement_source = inspect.getsource(train_register_clover_refinement)
    bank_source = inspect.getsource(build_register_candidate_bank)
    for source in (stage1_source, refinement_source, bank_source):
        assert "FlowmatchingActionHead" not in source
        assert "DriveSuprim" not in source
    assert "score_async" in inspect.getsource(QwenRegisterClover)
    assert "CandidateBankReader" in refinement_source


def test_checkpoint_pair_selection_can_keep_a_better_earlier_cycle():
    candidates = [
        {"label": "cycle_01", "selected_true_pdms": 0.81},
        {"label": "cycle_02", "selected_true_pdms": 0.86},
        {"label": "closing_critic", "selected_true_pdms": 0.83},
    ]
    assert select_best_pair(candidates)["label"] == "cycle_02"


def test_checkpoint_pair_selection_uses_conservative_holdout_bound():
    candidates = [
        {
            "label": "noisy_high_mean",
            "selected_true_pdms": 0.87,
            "selected_true_pdms_lcb95": 0.80,
        },
        {
            "label": "stable",
            "selected_true_pdms": 0.85,
            "selected_true_pdms_lcb95": 0.83,
        },
    ]
    assert select_best_pair(candidates)["label"] == "stable"


def test_checkpoint_pair_selection_requires_paired_positive_improvement():
    candidates = [
        {
            "label": "incumbent",
            "selected_true_pdms": 0.80,
            "_scene_scores": {f"t{i}": 0.80 for i in range(8)},
        },
        {
            "label": "noisy",
            "selected_true_pdms": 0.825,
            "_scene_scores": {
                f"t{i}": value
                for i, value in enumerate(
                    [1.0, 1.0, 1.0, 1.0, 0.65, 0.65, 0.65, 0.65]
                )
            },
        },
        {
            "label": "stable_gain",
            "selected_true_pdms": 0.85,
            "_scene_scores": {f"t{i}": 0.85 for i in range(8)},
        },
    ]
    selected = select_best_pair(candidates)
    assert selected["label"] == "stable_gain"
    assert candidates[1]["paired_selection"]["accepted"] is False
    assert candidates[2]["paired_selection"]["improvement_lcb95"] > 0


def test_enrichment_reports_scene_level_high_pdms_support():
    proposals = torch.zeros(2, 4, 8, 3)
    predicted = torch.tensor([[0.9, 0.8, 0.2, 0.1], [0.8, 0.7, 0.2, 0.1]])
    targets = build_teacher_target_sets(
        proposals,
        _metric_logits(predicted),
        predicted,
        topk=2,
        pareto_max_size=2,
        pareto_min_size=1,
        reward_threshold=0.0,
    )
    true_score = torch.tensor([[1.0, 0.9, 0.0, 0.0], [0.95, 0.8, 0.0, 0.0]])
    report = selected_set_enrichment_per_scene(
        targets, true_score, high_score_threshold=0.9
    )
    assert report["topk_enrichment"].shape == (2,)
    assert torch.all(report["topk_high_score_enrichment"] > 0)


def test_clover_launcher_rejects_documentation_placeholder_as_asset_path(tmp_path):
    repository = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment.update(
        CLOVER_PSEUDO_EXPERT_PKL="/absolute/path/to/pseudo_experts.pkl",
        CLOVER_RUN_ID="asset-placeholder-contract",
    )
    result = subprocess.run(
        ["bash", str(repository / "run_register64_clover_pdms_dlc.sh"), "--dry-run"],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    expected = repository / "navsim_exp/assets/clover_stage1_pseudo_experts/CLOVER/dataset_decoupled_v2_clean.pkl"
    assert f"pseudo_experts={expected}" in result.stdout
    assert "pseudo_asset_state=MISSING" in result.stdout
    assert "/absolute/path/to/pseudo_experts.pkl" not in result.stdout


def test_clover_asset_preparation_dry_run_is_no_write(tmp_path):
    repository = Path(__file__).resolve().parents[2]
    destination = tmp_path / "path with spaces" / "pseudo.pkl"
    result = subprocess.run(
        [
            "bash",
            str(repository / "prepare_clover_pseudo_experts.sh"),
            "--output",
            str(destination),
            "--dry-run",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "dry_run=1 writes=0 imports=0" in result.stdout
    assert "state=MISSING" in result.stdout
    assert str(destination) in result.stdout
    assert not destination.parent.exists()


def test_clover_asset_preparation_normalizes_documentation_placeholder(tmp_path):
    repository = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["CLOVER_PSEUDO_EXPERT_PKL"] = "/absolute/path/to/pseudo_experts.pkl"
    result = subprocess.run(
        ["bash", str(repository / "prepare_clover_pseudo_experts.sh"), "--dry-run"],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    expected = repository / "navsim_exp/assets/clover_stage1_pseudo_experts/CLOVER/dataset_decoupled_v2_clean.pkl"
    assert f"output={expected}" in result.stdout
    assert "/absolute/path/to/pseudo_experts.pkl" not in result.stdout
