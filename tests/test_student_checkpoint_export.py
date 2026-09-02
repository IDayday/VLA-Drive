from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from navsim.agents.EpisodeDrive.drivevla_backbone import (
    load_exact_student_checkpoint_with_audit,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_export_module():
    path = REPO_ROOT / "scripts/export_planreg_student_checkpoint.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _training_checkpoint():
    return {
        "state_dict": {
            "agent.backbone.model.vision_model.weight": torch.randn(2, 2),
            "agent.backbone.planning_register_adapter.planning_registers": torch.randn(1, 16, 4),
            "agent.action_head.semantic_gate": torch.zeros(1, 1, 4),
            "agent.action_head.scorer.pred_score.comfort.0.weight": torch.randn(4, 4),
            "agent.ema_register_target.vision_model.weight": torch.randn(2, 2),
            "agent.future_register_predictor.residual_output.weight": torch.randn(4, 4),
            "agent._ema_optimizer_step": torch.tensor(9),
            "agent._ema_actual_start_momentum": torch.tensor(0.996),
            "agent._ema_actual_end_momentum": torch.tensor(0.9999),
            "agent._ema_current_momentum": torch.tensor(0.997),
            "agent._world_model_optimizer_step": torch.tensor(9),
            "agent._world_model_total_optimizer_steps": torch.tensor(100),
        },
        "optimizer_states": [{"state": {}}],
        "lr_schedulers": [{"last_epoch": 9}],
        "callbacks": {"checkpoint": {"best": 0.9}},
        "epoch": 1,
        "global_step": 9,
        "hyper_parameters": {
            "agent": {
                "vlm_config": {"vlm_type": "internvl"},
                "planning_registers": {"attention_mode": "read_only"},
            }
        },
    }


def test_student_export_strips_training_state_and_writes_manifest(tmp_path) -> None:
    module = _load_export_module()
    source = tmp_path / "training.ckpt"
    output = tmp_path / "student.ckpt"
    torch.save(_training_checkpoint(), source)
    manifest = module.export_student_checkpoint(
        source, output, source_git_commit="deadbeef"
    )

    exported = torch.load(output, map_location="cpu", weights_only=False)
    module.verify_student_checkpoint_payload(exported)
    assert "optimizer_states" not in exported
    assert "lr_schedulers" not in exported
    assert "callbacks" not in exported
    assert "epoch" in exported and "global_step" in exported
    keys = set(exported["state_dict"])
    assert "agent.backbone.model.vision_model.weight" in keys
    assert "agent.backbone.planning_register_adapter.planning_registers" in keys
    assert "agent.action_head.semantic_gate" in keys
    assert "agent.action_head.scorer.pred_score.comfort.0.weight" in keys
    assert not any("ema_register_target" in key for key in keys)
    assert not any("future_register_predictor" in key for key in keys)
    assert not any("_optimizer_step" in key for key in keys)

    assert manifest["source_checkpoint_sha256"] == _sha256(source)
    assert manifest["export_checkpoint_sha256"] == _sha256(output)
    assert manifest["removed_key_count"] == 8
    assert manifest["retained_key_count"] == 4
    assert manifest["source_git_commit"] == "deadbeef"
    assert manifest["resolved_architecture_config"]["vlm_config"]["vlm_type"] == "internvl"
    manifest_path = output.with_suffix(".ckpt.manifest.json")
    assert json.loads(manifest_path.read_text())["export_checkpoint_sha256"] == _sha256(output)


def test_student_verifier_rejects_training_checkpoint() -> None:
    module = _load_export_module()
    with pytest.raises(RuntimeError, match="not student-only"):
        module.verify_student_checkpoint_payload(_training_checkpoint())


def test_exact_student_loader_requires_identical_topology_and_values() -> None:
    source = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.LayerNorm(4))
    target = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.LayerNorm(4))
    prefixed = {
        f"agent.{key}": value.clone() for key, value in source.state_dict().items()
    }
    audit = load_exact_student_checkpoint_with_audit(target, prefixed)
    assert audit["missing_keys"] == []
    assert audit["unexpected_keys"] == []
    assert audit["legacy_lora_scale_applied"] is False
    for key, value in source.state_dict().items():
        torch.testing.assert_close(value, target.state_dict()[key], rtol=0, atol=0)

    incomplete = dict(prefixed)
    incomplete.pop(next(iter(incomplete)))
    with pytest.raises(RuntimeError, match="Exact formal student"):
        load_exact_student_checkpoint_with_audit(target, incomplete)


def test_export_refuses_to_overwrite_source(tmp_path) -> None:
    module = _load_export_module()
    source = tmp_path / "training.ckpt"
    torch.save(_training_checkpoint(), source)
    with pytest.raises(ValueError, match="differ from source"):
        module.export_student_checkpoint(source, source)
