"""Small real checkpoint files exercise receipt/provenance resolution, not VLA inference."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import iqe.internal_baseline.binding as binding
from iqe.io import atomic_json, atomic_torch, file_hash, read_json


ORIGINAL = {"framework_contract_hash": "framework", "query_architecture": {"width": 4}}


def config(tmp_path, requested="latest_complete"):
    run = tmp_path/"declared_own_base"
    run.mkdir(exist_ok=True)
    return SimpleNamespace(base_run=str(run), root=tmp_path/"experiment", checkpoint=requested,
                           experiment_id="synthetic-resolution-test", identity="test-config")


def checkpoint(run, step, *, changes=None, receipt=True):
    path = Path(run)/f"step_{step:06d}.pt"
    state = {"kind": "iqe_query_base", "optimizer_step": step, "mode": "full",
             "framework_contract_hash": "framework", "resume_boundary": "optimizer_step",
             "dependencies": {"config": {"total_updates": 100000}, "dataset": "own-fixed-data"},
             "model": {"action_model.expert.query.weight": torch.zeros(1, 4)},
             "optimizer": {"state": {0: {"step": torch.tensor(float(step))}}},
             "scheduler": {}, "epoch": 0, "offset": step, "rng_by_rank": [{}]}
    state.update(changes or {})
    atomic_torch(path, state)
    if receipt:
        atomic_json(str(path)+".COMPLETE.json", {"checksum": file_hash(path), "optimizer_step": step,
                                               "dependencies": state["dependencies"]})
    return path


def test_latest_complete_skips_partial_corrupt_and_wrong_provenance_with_reasons(tmp_path):
    c = config(tmp_path)
    best = checkpoint(c.base_run, 200)
    old = checkpoint(c.base_run, 100)
    checkpoint(c.base_run, 700, receipt=False)
    bad = checkpoint(c.base_run, 600)
    with bad.open("ab") as f:
        f.write(b"corruption")
    checkpoint(c.base_run, 500, changes={"mode": "profile"})
    checkpoint(c.base_run, 400, changes={"optimizer": {"state": {}}})
    checkpoint(c.base_run, 300, changes={"framework_contract_hash": "other-framework"})
    before = {p.name: file_hash(p) for p in Path(c.base_run).glob("*")}
    selected, state = binding.resolve_base_checkpoint(c, ORIGINAL)
    assert selected["path"] == str(best) and state["optimizer_step"] == 200
    report = read_json(c.root/"BASE_SELECTION_RESOLUTION.json")
    assert report["status"] == "RESOLVED" and len(report["newer_rejections"]) == 5
    assert all(r["reason"] for r in report["newer_rejections"])
    assert report["candidates"][-1] == {"path": str(old), "filename_step": 100, "status": "NOT_INSPECTED_OLDER"}
    assert selected["receipt_hash"] == file_hash(str(best)+".COMPLETE.json")
    assert before == {p.name: file_hash(p) for p in Path(c.base_run).glob("*")}


def test_explicit_older_checkpoint_is_respected_without_scanning_newer_scores(tmp_path):
    c = config(tmp_path)
    old = checkpoint(c.base_run, 100)
    checkpoint(c.base_run, 200)
    c.checkpoint = str(old)
    selected, _ = binding.resolve_base_checkpoint(c, ORIGINAL)
    assert selected["path"] == str(old)
    report = read_json(c.root/"BASE_SELECTION_RESOLUTION.json")
    assert report["policy"] == "explicit_no_fallback" and len(report["candidates"]) == 1


def test_explicit_corrupt_checkpoint_never_falls_back(tmp_path):
    c = config(tmp_path)
    checkpoint(c.base_run, 100)
    missing = checkpoint(c.base_run, 200, receipt=False)
    c.checkpoint = str(missing)
    with pytest.raises(ValueError, match="NO_VALID_BASE_CHECKPOINT"):
        binding.resolve_base_checkpoint(c, ORIGINAL)
    report = read_json(c.root/"BASE_SELECTION_RESOLUTION.json")
    assert report["selected"] is None and len(report["candidates"]) == 1


@pytest.mark.parametrize("change", [
    {"kind": "external_policy"}, {"mode": "smoke"}, {"mode": "profile"},
    {"optimizer": {"state": {}}}, {"framework_contract_hash": "other-framework"},
    {"model": {"action_model.expert.query.weight": torch.zeros(64, 4)}},
    {"model": {"action_model.expert.query.weight": torch.zeros(1, 8)}},
])
def test_incompatible_snapshot_cannot_become_selected_base(tmp_path, change):
    c = config(tmp_path)
    checkpoint(c.base_run, 100, changes=change)
    with pytest.raises(ValueError, match="NO_VALID_BASE_CHECKPOINT"):
        binding.resolve_base_checkpoint(c, ORIGINAL)
    assert read_json(c.root/"BASE_SELECTION_RESOLUTION.json")["status"] == "REJECTED"


def test_one_step_diagnostic_is_rejected_even_if_marked_full(tmp_path):
    c = config(tmp_path)
    checkpoint(c.base_run, 1)
    with pytest.raises(ValueError, match="learned own Query Base"):
        binding.resolve_base_checkpoint(c, ORIGINAL)


def test_external_lineage_symlink_and_explicit_path_are_rejected(tmp_path):
    c = config(tmp_path)
    foreign = tmp_path/"foreign"
    foreign.mkdir()
    external = checkpoint(foreign, 900)
    (Path(c.base_run)/external.name).symlink_to(external)
    safe = checkpoint(c.base_run, 100)
    selected, _ = binding.resolve_base_checkpoint(c, ORIGINAL)
    assert selected["path"] == str(safe)
    assert "declared own Base run" in read_json(c.root/"BASE_SELECTION_RESOLUTION.json")["newer_rejections"][0]["reason"]
    c.checkpoint = str(external)
    with pytest.raises(ValueError, match="declared own Base run"):
        binding.resolve_base_checkpoint(c, ORIGINAL)


def test_no_complete_checkpoint_is_failure_not_fresh_initialization(tmp_path):
    c = config(tmp_path)
    with pytest.raises(ValueError, match="NO_VALID_BASE_CHECKPOINT"):
        binding.resolve_base_checkpoint(c, ORIGINAL)
    assert read_json(c.root/"BASE_SELECTION_RESOLUTION.json")["selected"] is None


def test_existing_binding_does_not_refresh_after_new_checkpoint_appears(tmp_path, monkeypatch):
    c = config(tmp_path)
    original = checkpoint(c.base_run, 100)
    frozen = {"experiment_id": c.experiment_id, "config_hash": c.identity,
              "checkpoint": str(original), "checkpoint_hash": file_hash(original)}
    atomic_json(c.root/"BASE_BINDING.json", frozen, immutable=True)
    checkpoint(c.base_run, 200)
    def forbidden(*args, **kwargs):
        pytest.fail("a bound experiment must not resolve latest again")
    monkeypatch.setattr(binding, "resolve_base_checkpoint", forbidden)
    assert binding.bind(c) == frozen
    assert not (c.root/"BASE_SELECTION_RESOLUTION.json").exists()
