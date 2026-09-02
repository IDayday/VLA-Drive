from pathlib import Path

from omegaconf import OmegaConf
import pytest

from navsim.planning.script.run_training_full import (
    compute_formal_step_budget,
    configure_formal_step_budget,
)


@pytest.mark.parametrize(
    "global_batch,steps_per_epoch,total_steps",
    [(16, 6456, 174312), (32, 3228, 87156), (64, 1614, 43578)],
)
def test_exact_103k_formal_step_budget(global_batch, steps_per_epoch, total_steps):
    budget = compute_formal_step_budget(103288, global_batch, dataset_epochs=27)
    assert budget["steps_per_epoch"] == steps_per_epoch
    assert budget["total_steps"] == total_steps
    assert budget["padded_samples_per_epoch"] % global_batch == 0
    assert budget["unpadded_sample_exposures"] == 103288 * 27


def _config(tmp_path: Path, global_batch: int = 32):
    return OmegaConf.create(
        {
            "output_dir": str(tmp_path),
            "validation_run": False,
            "auto_resume": False,
            "data_protocol": {"include_val_in_train": True},
            "formal_training": {
                "enabled": True,
                "expected_dataset_size": 103288,
                "require_exact_dataset_size": True,
                "dataset_epochs": 27,
            },
            "dataloader": {"params": {"batch_size": global_batch // 8}},
            "trainer": {
                "params": {
                    "devices": 8,
                    "num_nodes": 1,
                    "limit_val_batches": 0,
                    "max_epochs": 1,
                    "max_steps": -1,
                }
            },
        }
    )


def test_runtime_step_budget_sets_max_steps_and_writes_metadata(tmp_path: Path):
    config = _config(tmp_path)
    budget = configure_formal_step_budget(config, 103288)
    assert config.trainer.params.max_epochs == 27
    assert config.trainer.params.max_steps == 87156
    assert budget["steps_per_epoch"] == 3228
    assert (tmp_path / "run_metadata" / "formal_step_budget.json").is_file()


def test_formal_step_budget_rejects_partial_dataset(tmp_path: Path):
    with pytest.raises(RuntimeError, match="dataset size mismatch"):
        configure_formal_step_budget(_config(tmp_path), 32)
