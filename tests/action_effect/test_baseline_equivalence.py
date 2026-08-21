"""Static guards that Gate-2 research remains disconnected from baseline inference."""

from __future__ import annotations

from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_qwen_action_path_does_not_import_action_effect_research() -> None:
    source = (REPOSITORY_ROOT / "starVLA/model/framework/QwenOFT.py").read_text(
        encoding="utf-8"
    )
    assert "research.action_effect" not in source
    assert "ActionEffectWorldProbe" not in source
    assert "lambda_aee" not in source
    assert "lambda_world" not in source


def test_original_training_config_has_no_world_probe_switch_enabled() -> None:
    path = REPOSITORY_ROOT / "starVLA/config/training/cfg_yaw_1225.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert "action_effect" not in config
    assert "world_probe" not in config.get("framework", {})
