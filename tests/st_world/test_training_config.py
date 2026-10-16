"""Execution contracts reject incomplete measurements and hidden config knobs."""
from copy import deepcopy

import pytest

from starVLA.dataloader.st_world_schema import TeacherFeatureContract
from tools.st_world_policy.training_config import PRECISION, validate_config


def config():
    teacher = TeacherFeatureContract(image_hw=(288, 384), patch_hw=(16, 16),
        native_grid_hw=(18, 24), output_grid_hw=(18, 24), channels=256,
        storage_dtype="float32", compute_dtype="float32", view_names=("CAM_F0", "CAM_L0", "CAM_R0"),
        teacher_identity="a" * 64, transform_identity="b" * 64,
        pooling={"kind": "none", "kernel_hw": (1, 1), "stride_hw": (1, 1)})
    return {"schema": "st_training_v1", "execution": {"family": "spatial", "scope": "PROFILE",
        "global_batch": 32, "micro_batch": 4, "world_size": 8, "max_updates": 20000,
        "stop_after": 240, "sample_limit": 64, "seed": 42, "precision_policy": PRECISION},
        "space_time": {"cell": "S-P2", "teacher": teacher.to_dict(),
            "world_queries_per_view": 48, "sampled_positions": 96},
        "optimizer": {"lr": 1e-5, "minimum_lr": 5e-7, "warmup": 1000,
            "betas": [.9, .95], "eps": 1e-8, "weight_decay": .001, "clip": 1.},
        "runtime": {"output_root": "/test-isolated-run", "loader_workers": 2,
            "save_every": 1000, "max_seconds": 3600, "deterministic": True,
            "gradient_checkpointing": True, "profile_mode": "timing"},
        "data": {"root": "/test-data", "teacher_root": "/test-teacher", "history_root": None,
            "pca_path": "/test-pca"}}


def test_profile_keeps_original_training_horizon_and_config_unchanged():
    value = config()
    before = deepcopy(value)
    validate_config(value)
    assert value == before
    assert value["execution"]["max_updates"] == 20000


@pytest.mark.parametrize("scope,stop,mode", [
    ("PROFILE", 120, "timing"), ("SMOKE", 4, "timing"),
    ("PROFILE", 5, "trace"), ("PROFILE", 240, "silent-default"),
])
def test_partial_windows_or_oversized_traces_cannot_masquerade_as_profile(scope, stop, mode):
    value = config()
    value["execution"].update(scope=scope, stop_after=stop)
    value["runtime"]["profile_mode"] = mode
    with pytest.raises(ValueError):
        validate_config(value)


def test_explicit_trace_and_legacy_no_instrumentation_are_distinct():
    value = config()
    value["execution"].update(scope="SMOKE", stop_after=4)
    value["runtime"]["profile_mode"] = "trace"
    validate_config(value)
    del value["runtime"]["profile_mode"]
    validate_config(value)


def test_unknown_runtime_configuration_is_not_silently_ignored():
    value = config()
    value["runtime"]["ignore_labels_to_go_faster"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        validate_config(value)
