from __future__ import annotations

import json

import pytest
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import ConcatDataset, TensorDataset
import torch

from navsim.planning.script.run_training_full import (
    combine_cached_train_data,
    configure_callbacks_for_data_protocol,
    resolve_data_protocol,
    write_data_protocol_metadata,
)


def _config(tmp_path, *, include_val=False, disjoint=True, limit_val=1.0):
    return OmegaConf.create(
        {
            "train_logs": ["train_b", "train_a"],
            "val_logs": ["val_a"],
            "output_dir": str(tmp_path),
            "validation_run": False,
            "data_protocol": {
                "include_val_in_train": include_val,
                "require_disjoint_train_val": disjoint,
            },
            "trainer": {"params": {"limit_val_batches": limit_val}},
        }
    )


def test_default_protocol_keeps_train_and_validation_disjoint(tmp_path) -> None:
    cfg = _config(tmp_path)
    audit = resolve_data_protocol(cfg)
    assert audit["train_logs"] == ["train_b", "train_a"]
    assert audit["val_logs"] == ["val_a"]
    assert audit["effective_train_logs"] == audit["train_logs"]
    assert audit["overlap_count"] == 0
    assert audit["train_logs_sha256"] != audit["val_logs_sha256"]

    train = TensorDataset(torch.arange(2))
    val = TensorDataset(torch.arange(3))
    assert combine_cached_train_data(train, val, audit) is train


def test_scene_filter_is_applied_before_split_hashing(tmp_path) -> None:
    cfg = _config(tmp_path)
    audit = resolve_data_protocol(
        cfg, scene_filter_log_names=["train_a", "val_a", "unrelated"]
    )
    assert audit["train_logs"] == ["train_a"]
    assert audit["val_logs"] == ["val_a"]


def test_overlap_fails_when_disjoint_protocol_is_required(tmp_path) -> None:
    cfg = _config(tmp_path)
    cfg.val_logs = ["train_a", "val_a"]
    with pytest.raises(RuntimeError, match="overlap.*count=1"):
        resolve_data_protocol(cfg)


def test_final_fit_requires_no_validation_and_uses_last_only(tmp_path) -> None:
    cfg = _config(tmp_path, include_val=True, limit_val=0)
    audit = resolve_data_protocol(cfg)
    assert audit["mode"] == "final_fit"
    assert audit["hyperparameter_selection_allowed"] is False
    assert audit["effective_train_logs"] == ["train_b", "train_a", "val_a"]

    train = TensorDataset(torch.arange(2))
    val = TensorDataset(torch.arange(3))
    combined = combine_cached_train_data(train, val, audit)
    assert isinstance(combined, ConcatDataset)
    assert len(combined) == 5

    best = ModelCheckpoint(monitor="val/score_epoch", save_top_k=1)
    callbacks = configure_callbacks_for_data_protocol([best], audit)
    checkpoints = [item for item in callbacks if isinstance(item, ModelCheckpoint)]
    assert len(checkpoints) == 1
    assert checkpoints[0].monitor is None
    assert checkpoints[0].save_top_k == 0
    assert checkpoints[0].save_last is True


def test_final_fit_rejects_validation_or_validation_run(tmp_path) -> None:
    cfg = _config(tmp_path, include_val=True, limit_val=1.0)
    with pytest.raises(RuntimeError, match="limit_val_batches=0"):
        resolve_data_protocol(cfg)
    cfg.trainer.params.limit_val_batches = 0
    cfg.validation_run = True
    with pytest.raises(RuntimeError, match="validation_run=true"):
        resolve_data_protocol(cfg)


def test_protocol_metadata_round_trip(tmp_path) -> None:
    cfg = _config(tmp_path)
    audit = resolve_data_protocol(cfg)
    path = write_data_protocol_metadata(cfg, audit)
    assert json.loads(path.read_text(encoding="utf-8")) == audit

