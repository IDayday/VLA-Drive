"""Phase expansion never reinitializes training or rewrites the parent run."""
import copy
import json
from pathlib import Path

import pytest

from starVLA.model.modules.vehicle_joint.initialization import file_sha256, identity_hash
from tools.st_world_policy.budget_extension import resolve_extension, validate_extension_config
from tools.st_world_policy.training_state import checkpoint_receipt


def setup(tmp_path, monkeypatch):
    # Real native receipt checks, tiny opaque shards. Registration's separate
    # full source/approval tests cover git and reviewer provenance.
    monkeypatch.setattr("tools.st_world_policy.check_formal_approval.verify_registration", lambda *a, **k: {})
    cfg = {"execution": {"stop_after": 2, "max_updates": 20, "world_size": 1,
                          "seed": 42, "scope": "SMOKE"},
           "runtime": {"output_root": str(tmp_path), "profile_mode": "none"},
           "optimizer": {"lr": 1e-4, "warmup": 1}, "data": {"identity": "same"}}
    config_path = tmp_path / "parent_config.json"
    config_path.write_text(json.dumps(cfg))
    reg = {"run_id": "parent", "source_fingerprint": "unchanged",
           "weights_usable_for_formal_init": False,
           "artifacts": {"config": {"path": str(config_path), "sha256": file_sha256(config_path)}}}
    registration_path = tmp_path / "parent_registration.json"
    registration_path.write_text(json.dumps(reg))
    identity = {"schema": "st_world_training_v1", "config": cfg, "data": {"observations": "same"},
                "base_binding": "base", "registration_sha256": file_sha256(registration_path),
                "weights_usable_for_formal_init": False}
    identity["identity"] = identity_hash(identity)
    folder = tmp_path / "parent/checkpoints/step_000002"
    folder.mkdir(parents=True)
    (folder.parent.parent / "identity.json").write_text(json.dumps(identity))
    for name in ("mp_rank_00_model_states.pt", "bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt", "rng_rank0.pt"):
        (folder / name).write_bytes(name.encode())
    checkpoint_receipt(folder, {"identity": identity["identity"], "tag": folder.name,
        "completed": 2, "epoch": 3, "offset": 7, "exposure": 256,
        "termination_reason": "RUNNING_CHECKPOINT"}, world_size=1)
    child = copy.deepcopy(cfg)
    child["execution"]["stop_after"] = 4
    child["continuation"] = {"parent_registration": {"path": str(registration_path),
        "sha256": file_sha256(registration_path)}, "parent_tag": folder.name,
        "parent_receipt_sha256": file_sha256(folder / "COMPLETE.json")}
    child_reg = {**reg, "run_id": "next_phase"}
    return child, child_reg, folder


def resolve(cfg, reg, tmp_path):
    return resolve_extension(cfg, reg, repo=tmp_path,
                             data_identity={"observations": "same"}, base_binding="base")


def test_extension_binds_exact_native_boundary_and_retains_original_horizon(tmp_path, monkeypatch):
    cfg, reg, folder = setup(tmp_path, monkeypatch)
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    result = resolve(cfg, reg, tmp_path)
    assert result["old_stop_after"] == 2 and result["new_stop_after"] == 4
    assert result["original_schedule_max_updates"] == 20
    assert result["selected"]["receipt"]["state"]["offset"] == 7
    assert result["selected"]["receipt"]["state"]["exposure"] == 256
    assert result["formal_approval_is_separate"]
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == before


@pytest.mark.parametrize("change", ["lr", "warmup", "max_updates", "seed", "precision", "scope", "data", "same_run"])
def test_extension_cannot_smuggle_a_new_training_definition(tmp_path, monkeypatch, change):
    cfg, reg, _ = setup(tmp_path, monkeypatch)
    if change in ("lr", "warmup"):
        cfg["optimizer"][change] *= 2
    elif change in ("max_updates", "seed"):
        cfg["execution"][change] += 1
    elif change == "precision": cfg["execution"][change] = "another backend"
    elif change == "scope": cfg["execution"][change] = "FORMAL"
    elif change == "data": cfg["data"]["identity"] = "other"
    else: reg["run_id"] = "parent"
    with pytest.raises(ValueError): resolve(cfg, reg, tmp_path)


@pytest.mark.parametrize("change", ["receipt", "unreceipted_shard", "registration", "partial_phase"])
def test_extension_rejects_changed_or_incomplete_parent(tmp_path, monkeypatch, change):
    cfg, reg, folder = setup(tmp_path, monkeypatch)
    if change == "receipt": (folder / "COMPLETE.json").write_text("{}")
    elif change == "unreceipted_shard": (folder / "extra_model_states.pt").write_bytes(b"not committed")
    elif change == "registration":
        Path(cfg["continuation"]["parent_registration"]["path"]).write_text("{}")
    else:
        receipt = json.loads((folder / "COMPLETE.json").read_text())
        receipt["state"]["completed"] = 1
        receipt["receipt_identity"] = identity_hash({k:v for k,v in receipt.items() if k != "receipt_identity"})
        (folder / "COMPLETE.json").write_text(json.dumps(receipt))
        cfg["continuation"]["parent_receipt_sha256"] = file_sha256(folder / "COMPLETE.json")
    with pytest.raises(ValueError): resolve(cfg, reg, tmp_path)


def test_extension_does_not_accept_unsafe_tag_or_implicit_latest():
    for tag in ("", "../x", ".", ".."):
        with pytest.raises(ValueError): validate_extension_config({
            "parent_registration": {"path": "/valid.json", "sha256": "a" * 64},
            "parent_tag": tag, "parent_receipt_sha256": "b" * 64})
