from pathlib import Path

import pytest
import torch

from starVLA.training.config_loader import load_training_config
from starVLA.training.register_stage_utils import (
    validate_bank_only_training_profile,
)
from starVLA.training.train_register_generator import (
    generator_component_checkpoint_names,
    summarize_register_usage,
    validate_sparse_oracle_config,
)


CONFIG_ROOT = (
    Path(__file__).resolve().parents[2] / "starVLA" / "config" / "training"
)


def _load(name: str):
    return load_training_config(CONFIG_ROOT / name)


def test_generator_and_candidate_bank_shuffle_contract():
    generator = _load("qwen_register64_generator.yaml")
    bank = _load("register64_candidate_bank.yaml")
    assert generator.datasets.vla_data.shuffle is True
    assert bank.datasets.vla_data.shuffle is False


def test_generator_production_gradient_and_oracle_gates():
    generator = _load("qwen_register64_generator.yaml")
    assert generator.framework.generator_loss.stage_loss_mode == "final_only"
    assert generator.trainer.gradient_gate.enabled is True
    assert generator.trainer.early_stopping.monitor == "min_ade_64"
    assert generator.validation.num_scenes == 1024
    assert generator.validation.pdm_oracle.enabled is True
    assert generator.validation.pdm_oracle.interval_epochs == 5


def test_sparse_oracle_config_fails_fast_without_metric_cache():
    generator = _load("qwen_register64_generator.yaml")
    generator.validation.pdm_oracle.metric_supervisor.metric_cache_root = None
    with pytest.raises(ValueError, match="NAVSIM_METRIC_CACHE_ROOT"):
        validate_sparse_oracle_config(generator.validation)


def test_generator_checkpoint_selection_is_independent():
    names = generator_component_checkpoint_names(
        epoch=5,
        final_epoch=25,
        save_epochs={5, 10, 15, 20, 25},
        should_stop=False,
        improved_minade=False,
        improved_oracle=True,
    )
    assert names == ["generator_epoch_05.pt", "best_oracle_generator.pt"]
    assert "best_minade_generator.pt" not in names
    assert "best_generator.pt" not in names


def test_register_collapse_gate_reports_global_winner_distribution():
    metrics = summarize_register_usage(torch.tensor([3.0, 1.0, 0.0, 0.0]))
    assert metrics["register_usage_histogram"] == [3, 1, 0, 0]
    assert metrics["active_register_ratio"] == pytest.approx(0.5)
    assert metrics["top1_register_fraction"] == pytest.approx(0.75)
    assert 0.0 < metrics["register_usage_entropy"] < 1.0


def test_drivor_uses_independent_donor_anchored_profile():
    config = _load("register64_drivor_scorer.yaml")
    validate_bank_only_training_profile(
        config, expected_name="drivor_offline_bank_v1"
    )
    assert "framework" not in config
    assert config.optimizer.name == "AdamW"
    assert config.optimizer.lr == 2.0e-4
    assert config.trainer.epochs == 5
    assert config.trainer.max_epochs == 10
    assert config.trainer.global_batch_size == 256
    assert config.model.num_layers == 4
    assert config.model.num_heads == 1
    assert config.training_profile.donor.reference_recipe.navsim_v2_epochs == 10


def test_dynamic_suprim_has_its_own_short_bank_profile():
    config = _load("register64_drivor_suprim_dynamic.yaml")
    validate_bank_only_training_profile(
        config, expected_name="drivesuprim_dynamic_bank_v1"
    )
    assert "framework" not in config
    assert config.stage_mode == "dynamic"
    assert config.optimizer.lr == 7.5e-5
    assert config.trainer.epochs == 3
    assert config.trainer.max_epochs == 5
    assert config.trainer.global_batch_size == 256
    assert config.model.dynamic_topm == 32
    assert config.model.fine.refinement_layers == 3


def test_hybrid_suprim_matches_donor_scale_without_stage_g_settings():
    config = _load("register64_drivor_suprim_hybrid.yaml")
    validate_bank_only_training_profile(
        config, expected_name="drivesuprim_hybrid_bank_v1"
    )
    assert "framework" not in config
    assert config.stage_mode == "hybrid"
    assert config.optimizer.lr == 7.5e-5
    assert config.trainer.epochs == 6
    assert config.trainer.max_epochs == 10
    assert config.trainer.global_batch_size == 64
    assert config.model.coarse.static_vocab_size == 8192
    assert config.model.coarse.joint_candidate_count == 8224
    assert config.model.coarse.coarse_topk == 256
    assert config.model.coarse.coarse_layers == 3
    assert config.model.fine.refinement_layers == 3
    assert config.training_profile.donor.reference_recipe.optimizer == "Adam"


def test_bank_profile_rejects_wrong_donor_revision():
    config = _load("register64_drivor_scorer.yaml")
    config.training_profile.donor.revision = "not-the-audited-donor"
    with pytest.raises(ValueError, match="donor revision mismatch"):
        validate_bank_only_training_profile(
            config, expected_name="drivor_offline_bank_v1"
        )


def test_bank_profile_rejects_misreported_reference_recipe():
    config = _load("register64_drivor_suprim_dynamic.yaml")
    config.training_profile.donor.reference_recipe.learning_rate = 1.0e-3
    with pytest.raises(ValueError, match="donor recipe learning_rate mismatch"):
        validate_bank_only_training_profile(
            config, expected_name="drivesuprim_dynamic_bank_v1"
        )
