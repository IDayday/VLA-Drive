"""Checkpoint retention and finite queue behavior; no fabricated model scores."""
from pathlib import Path

import pytest

from tools.planning_interface_transfer import watch_navtest as queue


@pytest.fixture
def registered(tmp_path, monkeypatch):
    models = []
    for arm in sorted(queue.fixed.ARMS):
        run = tmp_path/"students"/arm
        queue.atomic(run/"identity.json", {"identity":arm})
        queue.atomic(run/"status.json", {"identity":arm, "completed":76000, "status":"RUNNING"})
        models.append(dict(arm=arm, run_identity=arm, training_run=str(run)))
    r = dict(models=models, root=str(tmp_path/"queue"), protocol=dict(root=str(tmp_path/"75k")),
             identity="registered", deadline_unix=queue.time.time()+60,
             worker_python="python3", source_worktree=str(tmp_path), poll_seconds=.01)
    path = tmp_path/"registration.json"
    monkeypatch.setattr(queue, "load", lambda _:r)
    return r,path


def complete_checkpoint(model, update):
    folder = Path(model["training_run"])/"checkpoints"/f"periodic_{update:06d}"
    queue.atomic(folder/"COMPLETE.json", dict(completed=update, tag=folder.name,
        identity=model["run_identity"], exposure=update*32))
    (folder/"mp_rank_00_model_states.pt").write_bytes(b"model")
    for rank in range(8):
        (folder/f"zero_pp_rank_{rank}_optim_states.pt").write_bytes(b"optimizer")
        (folder/f"rng_rank{rank}.pt").write_bytes(b"rng")
    return folder


def test_retains_exact_checkpoint_while_75k_evaluation_runs(registered):
    r,path = registered
    m = r["models"][0]; source = complete_checkpoint(m,80000)
    queue.watch(path, once=True)
    job = Path(r["root"])/"jobs"/f"{m['arm']}_080000"
    snapshot = queue.read(job/"snapshot.json")
    frozen = Path(snapshot["training_run"])/"checkpoints"/snapshot["tag"]
    assert queue.read(Path(r["root"])/"status.json")["awaiting75k"]
    assert (source/"mp_rank_00_model_states.pt").stat().st_ino == (frozen/"mp_rank_00_model_states.pt").stat().st_ino
    # Simulate the trainer's rolling GC: preserved files survive without mutation/copy.
    import shutil
    shutil.rmtree(source)
    assert (frozen/"mp_rank_00_model_states.pt").read_bytes() == b"model"
    assert snapshot["completed"] == 80000
    assert queue.read(job/"status.json")["status"] == "QUEUED"


def test_missing_exact_step_never_uses_neighbor(registered):
    r,path = registered; m = r["models"][0]
    complete_checkpoint(m,80200)
    queue.atomic(Path(m["training_run"])/"status.json", dict(identity=m["run_identity"], completed=80200))
    queue.watch(path, once=True)
    states = queue.read(Path(r["root"])/"status.json")["tasks"]
    assert states[f"{m['arm']}_080000"] == "MISSED_CHECKPOINT"
    assert states[f"{m['arm']}_090000"] == "WAITING"
    assert len(states) == 9


def test_corrupt_checkpoint_does_not_disable_other_preservation(registered):
    r,path = registered
    bad = complete_checkpoint(r["models"][0],80000)
    queue.atomic(bad/"COMPLETE.json", dict(identity="foreign", tag=bad.name, completed=80000))
    complete_checkpoint(r["models"][1],80000)
    queue.watch(path, once=True)
    states = queue.read(Path(r["root"])/"status.json")["tasks"]
    assert states[f"{r['models'][0]['arm']}_080000"] == "FAILED"
    assert states[f"{r['models'][1]['arm']}_080000"] == "QUEUED"


def test_stop_preserves_checkpoints_without_launching(registered):
    r,path = registered; root = Path(r["root"]);root.mkdir()
    (root/"STOP_SCHEDULING").touch()
    complete_checkpoint(r["models"][0],80000)
    queue.watch(path, once=True)
    assert queue.read(root/"status.json")["status"] == "PAUSED"
    assert (root/"jobs"/f"{r['models'][0]['arm']}_080000"/"snapshot.json").exists()


def test_duplicate_observer_is_rejected(registered):
    r,path = registered
    with queue.lease(Path(r["root"])/"observer.lock"):
        with pytest.raises(BlockingIOError): queue.watch(path, once=True)


def test_foreign_training_progress_is_rejected(registered):
    r,path = registered
    queue.atomic(Path(r["models"][0]["training_run"])/"status.json", dict(identity="foreign", completed=81000))
    with pytest.raises(ValueError, match="Training status identity"):
        queue.watch(path, once=True)


def test_single_arm_completed_evaluation_resume_does_not_restart_exports(tmp_path, monkeypatch):
    fixed = queue.fixed
    root = tmp_path/"evaluation"
    fixed.atomic(root/"A_ACTION/result.json", dict(status="COMPLETE", arm="A_ACTION"))
    fixed.atomic(root/"status.json", dict(status="FAILED"))
    config = dict(root=str(root), canonical_hostname=fixed.socket.gethostname(),
                  models=[dict(arm="A_ACTION")], gpus=[7,4,5,6])
    monkeypatch.setattr(fixed,"load",lambda _:dict(config=config,identity="registered"))
    def no_launch(*args,**kwargs): raise AssertionError("Already completed export")
    monkeypatch.setattr(fixed,"launch",no_launch)
    fixed._run(tmp_path/"registration.json",[])
    assert fixed.read(root/"status.json")["status"] == "COMPLETE"
