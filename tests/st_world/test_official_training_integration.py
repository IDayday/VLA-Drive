"""Backend wiring/lineage regression; fixtures do not substitute for GPU evidence."""
from copy import deepcopy
import json

import pytest

from tools.st_world_policy.official_runtime import REFERENCE_RECIPE, SCHEMA, PRESET
from tools.st_world_policy.training_config import official_runtime_policy, validate_config as validate_world
from tools.st_world_policy.flow_training_config import validate_config as validate_flow
from tools.st_world_policy.train_flow_policy import numeric_execution
from tools.st_world_policy import flow_checkpoints
from test_training_config import config as world_config
from test_flow_training import config as flow_config
import test_flow_checkpoints as fixtures


def enable_official(cfg):
    cfg = deepcopy(cfg)
    cfg["runtime"]["gradient_checkpointing"] = False
    effective = {**{k: cfg["execution"][k] for k in
        ("world_size", "micro_batch", "global_batch", "max_updates")},
        **cfg["optimizer"], "gradient_checkpointing": False}
    cfg["runtime"]["official_execution"] = {
        "schema": SCHEMA, "preset": PRESET, "scope": "research_stage",
        "adaptation_reason": "Bounded registered research stage; preserve scientific recipe and exposure.",
        "declared_recipe_overrides": sorted(k for k, v in REFERENCE_RECIPE.items() if effective[k] != v)}
    return cfg


@pytest.mark.parametrize("make,validate", [(world_config, validate_world), (flow_config, validate_flow)])
def test_official_is_explicit_and_preserves_registered_budget(make, validate):
    old = make()
    assert official_runtime_policy(old) is None
    cfg = enable_official(old)
    before = deepcopy(cfg)
    validate(cfg)
    assert cfg == before
    assert cfg["execution"] == old["execution"]
    assert cfg["optimizer"] == old["optimizer"]
    actual = official_runtime_policy(cfg)
    assert actual["backend"]["optimizer"] == "torch.optim.AdamW"
    assert actual["backend"]["attention"] == "flash_attention_2"
    assert actual["effective_recipe"]["max_updates"] == old["execution"]["max_updates"]
    assert actual["effective_recipe"]["max_updates"] != 100000
    assert not actual["gpu_qualified"]
    cfg["optimizer"]["eps"] = 1e-6
    with pytest.raises(ValueError, match="overrides"):
        validate(cfg)


def test_official_flow_execution_cannot_reuse_legacy_numeric_identity():
    cfg = flow_config()
    binding = {"binding_sha256": "a"*64, "inference_contract": dict(flow_checkpoints.INFERENCE_CONTRACT)}
    old = numeric_execution(cfg, binding)
    new = numeric_execution(enable_official(cfg), binding)
    assert old["schema"] == "st_flow_numeric_execution_v2"
    assert new["schema"] == "st_flow_numeric_execution_v3"
    assert new["encoder"] == "frozen_BF16_execution_eval_official_FA2"
    assert new["official_execution"]["backend"]["master_dtype"] == "float32"
    assert new["standalone_evaluation_contract"] == old["standalone_evaluation_contract"]


def test_official_native_reader_retains_strict_canonical_deployment(tmp_path, monkeypatch):
    cfg = enable_official(flow_config())
    monkeypatch.setattr(fixtures, "config", lambda: deepcopy(cfg))
    artifact = fixtures.native_fixture(tmp_path, monkeypatch, numeric_schema="st_flow_numeric_execution_v3")
    result_cfg, _, record = flow_checkpoints.flow_checkpoint_identity(artifact.reg, artifact.tag)
    assert result_cfg["runtime"]["official_execution"]["preset"] == PRESET
    assert record["inference_contract"] == flow_checkpoints.INFERENCE_CONTRACT
    assert record["restoration"]["training_BF16_encoder_reused"] is False
    assert record["restoration"]["optimizer_resume_conversion"] is False
    # A valid outer checksum cannot turn an official execution checkpoint into
    # a legacy checkpoint by relabeling its numeric version.
    from starVLA.model.modules.vehicle_joint.initialization import identity_hash
    path = artifact.run / "identity.json"
    body = json.loads(path.read_text())
    body["numeric_execution"]["schema"] = "st_flow_numeric_execution_v2"
    body["numeric_execution_hash"] = identity_hash(body["numeric_execution"])
    body["identity"] = identity_hash({k: v for k, v in body.items() if k != "identity"})
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError):
        flow_checkpoints.flow_checkpoint_identity(artifact.reg, artifact.tag)
