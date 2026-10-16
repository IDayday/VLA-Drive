"""Deployment reads real current inputs, never supervision/cache side channels."""
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import pytest
import torch

from starVLA.dataloader.st_world_observation import STObservationDataset
from tools.st_world_policy.checkpoints import inference_source_contract
from tools.st_world_policy.export_world import validate_cached_prediction, validate_prediction
from starVLA.model.modules.vehicle_joint.initialization import file_sha256
from test_dataset import setup_data


@pytest.mark.parametrize("schema,name", [("st_training_v1", "load_world_checkpoint"),
                                        ("st_flow_training_v1", "load_flow_checkpoint")])
def test_registered_schema_dispatches_strict_native_reader(tmp_path, schema, name):
    from tools.st_world_policy.export_world import deployment_adapter
    cfg = tmp_path / "config.json"; cfg.write_text(json.dumps({"schema": schema}))
    reg = tmp_path / "reg.json"
    reg.write_text(json.dumps({"artifacts": {"config": {"path": str(cfg), "sha256": file_sha256(cfg)}}}))
    _, reader = deployment_adapter(reg)
    assert reader.__name__ == name
    cfg.write_text(json.dumps({"schema": "different"}))
    with pytest.raises(ValueError, match="configuration changed"):
        deployment_adapter(reg)
    reg.write_text(json.dumps({"artifacts": {"config": {"path": str(cfg), "sha256": file_sha256(cfg)}}}))
    with pytest.raises(ValueError, match="Unknown"):
        deployment_adapter(reg)


def test_deployment_works_with_all_training_targets_removed(tmp_path):
    args = setup_data(tmp_path)
    dataset = STObservationDataset(args["data_root"], args["manifest_path"])
    before = dataset[0]
    for path in (args["data_root"] / "ego").glob("*"):
        path.unlink()
    (args["data_root"] / "ego_identity.json").unlink()
    # Missing teacher artifacts cannot affect this target-free input reader.
    (args["teacher_root"] / "identity.json").unlink()
    another = STObservationDataset(args["data_root"], args["manifest_path"])
    after = another[0]
    assert dataset.identity == another.identity
    assert np.array_equal(before.current["state"], after.current["state"])
    assert before.current["lang"] == after.current["lang"]
    assert all(a.tobytes() == b.tobytes() for a, b in zip(before.current["image"], after.current["image"]))
    assert before.history is None and not dataset.identity["targets_read"]


def test_deployment_observation_mutation_cannot_reuse_identity(tmp_path):
    args = setup_data(tmp_path)
    dataset = STObservationDataset(args["data_root"], args["manifest_path"])
    Image.new("RGB", (32, 18), "red").save(tmp_path / "image.png")
    with pytest.raises(ValueError, match="Image content changed"):
        dataset[0]
    replacement = STObservationDataset(args["data_root"], args["manifest_path"])
    assert replacement.identity["identity"] != dataset.identity["identity"]
    path = args["data_root"] / "current/a.json"
    record = json.loads(path.read_text()); record["navigation"] = 2
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Current observation changed"):
        STObservationDataset(args["data_root"], args["manifest_path"])


@pytest.mark.parametrize("change", ["role", "foreign_log", "duplicate"])
def test_deployment_cannot_change_fixed_role_or_population(tmp_path, change):
    args = setup_data(tmp_path)
    selection = json.loads(args["manifest_path"].read_text())
    if change == "role":
        selection["split"] = "navtest"
    elif change == "foreign_log":
        selection["scenes"][0]["log"] = "another_log"
    else:
        selection["scenes"] *= 2
    args["manifest_path"].write_text(json.dumps(selection))
    with pytest.raises(ValueError):
        STObservationDataset(args["data_root"], args["manifest_path"])


def test_prediction_receipt_checks_full_payload_and_invalid_is_not_zero(tmp_path):
    path = tmp_path / "a.npz"
    raw, trajectory = torch.zeros(8, 4), torch.zeros(8, 3)
    raw[:, 3] = 1
    np.savez(path, raw=raw.numpy(), trajectory=trajectory.numpy())
    meta = {"identity": "one", "token": "a", "log": "log_a", "status": "VALID_PREDICTION",
            "sha256": file_sha256(path)}
    validate_cached_prediction(path, meta, identity="one", token="a", log="log_a")
    with pytest.raises(ValueError):
        validate_cached_prediction(path, meta, identity="different checkpoint", token="a", log="log_a")
    raw[0, 0] = float("nan")
    with pytest.raises(ValueError, match="INVALID_PREDICTION"):
        validate_prediction(raw, trajectory)
    np.savez(path, raw=raw.numpy(), trajectory=trajectory.numpy())
    with pytest.raises(ValueError, match="checksum"):
        validate_cached_prediction(path, meta, identity="one", token="a", log="log_a")
    meta["sha256"] = file_sha256(path)
    with pytest.raises(ValueError, match="INVALID_PREDICTION"):
        validate_cached_prediction(path, meta, identity="one", token="a", log="log_a")


def test_deployment_methods_must_match_even_if_strict_state_load_would_work(tmp_path):
    from tools.st_world_policy.flow_checkpoints import INFERENCE_SOURCES
    files = ["starVLA/model/framework/st_world_model.py", "tools/st_world_policy/world_expansion.py",
             "starVLA/model/framework/ddp_action_video_foresight.py", "starVLA/model/modules/st_world/config.py",
             "starVLA/dataloader/st_world_schema.py"] + list(INFERENCE_SOURCES)
    model = "class SpaceTimeStudent:\n" + "\n".join(
        "    def " + name + "(self):\n        return 1\n" for name in
        ("__init__", "train", "encode_inputs", "predict_raw", "predict_action", "strip_auxiliary_heads", "forward"))
    for name in files:
        path = tmp_path / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(model if name.endswith("st_world_model.py") else "CONTRACT = 1\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                    "commit", "-qm", "bounded deployment contract fixture"], cwd=tmp_path, check=True)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    first = inference_source_contract(tmp_path, sha, history=False)
    path = tmp_path / files[0]
    # A loss-only validation repair does not change the learned inference function.
    path.write_text(model.replace("def forward(self):\n        return 1", "def forward(self):\n        return 2"))
    assert inference_source_contract(tmp_path, sha, history=False) == first
    path.write_text(model.replace("def predict_raw(self):\n        return 1", "def predict_raw(self):\n        return 2"))
    with pytest.raises(ValueError, match="compatibility review"):
        inference_source_contract(tmp_path, sha, history=False)
