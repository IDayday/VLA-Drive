import json
from pathlib import Path

import pytest

from tools.st_world_policy.training_state import (
    checkpoint_receipt, resolve_latest, terminal, verify_checkpoint, can_reuse_just_saved_boundary,
)


def saved(root, step, reason="RUNNING_CHECKPOINT"):
    folder = root / ("step_" + str(step))
    folder.mkdir(parents=True)
    for name in ("mp_rank_00_model_states.pt", "bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt", "rng_rank0.pt"):
        (folder / name).write_bytes((name + str(step)).encode())
    state = {"identity": "test identity", "completed": step, "tag": folder.name,
             "epoch": 0, "offset": step, "termination_reason": reason}
    checkpoint_receipt(folder, state, world_size=1)
    return folder


def test_newer_native_complete_beats_stale_pointer_and_result(tmp_path):
    saved(tmp_path, 2)
    newer = saved(tmp_path, 4)
    (tmp_path / "latest").write_text("step_2\n")
    (tmp_path / "result.json").write_text('{"step":2}')
    report = resolve_latest(tmp_path, identity="test identity", world_size=1)
    assert report["selected"]["folder"] == str(newer)


def test_partial_and_corrupt_newer_are_reported_never_selected(tmp_path):
    valid = saved(tmp_path, 2)
    corrupt = saved(tmp_path, 4)
    (corrupt / "bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt").write_bytes(b"broken")
    partial = tmp_path / "step_6"
    partial.mkdir()
    (partial / "mp_rank_00_model_states.pt").write_bytes(b"partial")
    report = resolve_latest(tmp_path, identity="test identity", world_size=1)
    assert report["selected"]["folder"] == str(valid)
    assert len(report["rejected"]) == 2
    assert any("corrupt" in row["reason"] for row in report["rejected"])
    # A partial directory never acquires a usable complete receipt.
    with pytest.raises(FileNotFoundError):
        verify_checkpoint(partial, identity="test identity", world_size=1)


def test_terminal_receipt_survives_missing_result_without_extra_updates(tmp_path):
    folder = saved(tmp_path, 4, "MAX_STEPS")
    receipt = verify_checkpoint(folder, identity="test identity", world_size=1)
    assert terminal(receipt["state"], 4)
    assert terminal({**receipt["state"], "termination_reason": "RUNNING_CHECKPOINT"}, 4)
    assert not terminal({**receipt["state"], "completed": 2, "termination_reason": "OPERATOR_PAUSED"}, 4)
    assert not terminal({**receipt["state"], "completed": 2, "termination_reason": "BUDGET_PAUSED"}, 4)
    with pytest.raises(ValueError):
        terminal(receipt["state"], 3)


def test_foreign_world_or_missing_rng_cannot_resume(tmp_path):
    folder = saved(tmp_path, 4)
    with pytest.raises(ValueError, match="Foreign"):
        verify_checkpoint(folder, identity="foreign", world_size=1)
    with pytest.raises(ValueError, match="world"):
        verify_checkpoint(folder, identity="test identity", world_size=2)
    (folder / "rng_rank0.pt").unlink()
    with pytest.raises(ValueError, match="No valid"):
        resolve_latest(tmp_path, identity="test identity", world_size=1)


def test_receipt_cannot_commit_incomplete_native_state(tmp_path):
    (tmp_path / "mp_rank_00_model_states.pt").write_bytes(b"partial")
    with pytest.raises(ValueError, match="optimizer"):
        checkpoint_receipt(tmp_path, {"completed": 1, "termination_reason": "RUNNING_CHECKPOINT"}, world_size=1)
    assert not (tmp_path / "COMPLETE.json").exists()


def test_same_native_pause_boundary_needs_no_new_checkpoint(tmp_path):
    folder = saved(tmp_path, 2)
    state = json.loads((folder / "COMPLETE.json").read_text())["state"]
    assert can_reuse_just_saved_boundary(tmp_path, state, world_size=1)
    assert can_reuse_just_saved_boundary(tmp_path, {**state, "termination_reason":"BUDGET_PAUSED"}, world_size=1)
    assert not can_reuse_just_saved_boundary(tmp_path, {**state, "epoch":1, "offset":0}, world_size=1)
    assert not can_reuse_just_saved_boundary(tmp_path, {**state, "completed":3}, world_size=1)
    assert not can_reuse_just_saved_boundary(tmp_path, {**state, "pending_feedback":"unconsumed"}, world_size=1)
    (folder / "rng_rank0.pt").write_bytes(b"corrupt-size")
    with pytest.raises(ValueError, match="size changed"):
        can_reuse_just_saved_boundary(tmp_path, state, world_size=1)
