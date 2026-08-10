from pathlib import Path

from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    return OmegaConf.load(PROJECT_ROOT / "starVLA" / "config" / "training" / name)


def test_phase3_main_config_is_action_free_cached_teacher_and_effective_batch_32() -> None:
    cfg = _load("cfg_field2plan_phase3.yaml")

    assert cfg.framework.name == "QwenOFT_Field2Plan"
    assert cfg.field2plan.proposal.freeze_base_planner is True
    assert cfg.field2plan.proposal.source == "cache"
    assert cfg.field2plan.dynamics.enabled is True
    assert cfg.field2plan.dynamics.teacher.type == "vjepa2_1"
    assert cfg.field2plan.dynamics.teacher.cache_splits == ["train"]
    assert "checkpoint" not in cfg.field2plan.dynamics.teacher
    assert "repo" not in cfg.field2plan.dynamics.teacher
    assert cfg.field2plan.dynamics.horizon == 8
    assert cfg.field2plan.dynamics.channels == 192
    assert cfg.field2plan.dynamics.supervision.teacher_channels == 96
    assert cfg.datasets.vla_data.per_device_batch_size * 16 == 32
    assert cfg.trainer.gradient_accumulation_steps == 1


def test_phase3_debug_config_is_small_but_keeps_same_temporal_contract() -> None:
    cfg = _load("cfg_field2plan_phase3_debug.yaml")

    assert cfg.is_debug is True
    assert cfg.field2plan.geometry.field_size == [24, 24]
    assert cfg.field2plan.dynamics.history_frame_indices == [0, 1, 2, 3]
    assert cfg.field2plan.dynamics.future_frame_indices == list(range(4, 12))
    assert cfg.field2plan.dynamics.teacher.input_image_hw == [384, 384]
    assert cfg.trainer.max_train_steps == 1


def test_pre_phase3_configs_keep_dynamics_disabled_by_default() -> None:
    for name in ("cfg_field2plan_mvp.yaml", "cfg_field2plan_mvp_debug.yaml"):
        cfg = _load(name)
        assert cfg.field2plan.dynamics.enabled is False
