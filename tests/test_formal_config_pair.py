import pytest

from scripts.audit_formal_config_pair import audit_formal_config_pair


def _config(variant="base"):
    is_base = variant == "base"
    return {
        "agent": {
            "checkpoint_path": None,
            "stage1_checkpoint_path": None,
            "initialization": {
                "variant": variant,
                "shared_trainable_init_path": "/shared/init.pt",
                "vlm_checkpoint_sha256": "base-sha" if is_base else "vqa-sha",
                "vlm_config_sha256": "base-config" if is_base else "vqa-config",
            },
            "vlm_config": {"vlm_path": "/base" if is_base else "/vqa"},
            "world_model": {
                "enabled": True,
                "future_mode": "correct",
                "candidate_count": 1,
            },
            "action_head_config": {"proposal_num": 64},
        },
        "experiment_name": "formal_base" if is_base else "formal_vqa",
        "output_dir": "/out/base" if is_base else "/out/vqa",
        "formal_training": {"dataset_epochs": 27},
        "data_protocol": {"include_val_in_train": True},
        "trainer": {
            "params": {
                "limit_val_batches": 0,
                "devices": 8,
                "default_root_dir": "/out/base" if is_base else "/out/vqa",
            }
        },
    }


def test_formal_config_pair_allows_only_vlm_identity_fields():
    report = audit_formal_config_pair(_config("base"), _config("driving_vqa"))
    assert report["pair_equal_outside_allowlist"]
    assert report["world_model_enabled_both"]
    assert not report["multi_trajectory_consequence_modeling_implemented"]


def test_formal_config_pair_rejects_training_difference():
    base = _config("base")
    vqa = _config("driving_vqa")
    vqa["trainer"]["params"]["devices"] = 16
    with pytest.raises(RuntimeError, match="forbidden differences"):
        audit_formal_config_pair(base, vqa)


def test_formal_config_pair_rejects_disabled_world_model_even_if_equal():
    base = _config("base")
    vqa = _config("driving_vqa")
    base["agent"]["world_model"]["enabled"] = False
    vqa["agent"]["world_model"]["enabled"] = False
    with pytest.raises(RuntimeError, match="world model must be enabled"):
        audit_formal_config_pair(base, vqa)


def test_formal_config_pair_rejects_agent_checkpoint():
    base = _config("base")
    vqa = _config("driving_vqa")
    base["agent"]["checkpoint_path"] = "/m0.ckpt"
    vqa["agent"]["checkpoint_path"] = "/m0.ckpt"
    with pytest.raises(RuntimeError, match="checkpoint_path must be null"):
        audit_formal_config_pair(base, vqa)


def test_dual_formal_sequence_uses_the_two_locked_launchers():
    script = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "local_planreg_wm_v1/train_formal_dual_init_sequential.sh"
    ).read_text(encoding="utf-8")
    base_position = script.index("train_formal_base_init_wm.sh")
    vqa_position = script.index("train_formal_vqa_init_wm.sh")
    assert base_position < vqa_position
    assert "PLANREG_LAYOUT_LOCK" in script
    assert "PLANREG_SHARED_INIT" in script
