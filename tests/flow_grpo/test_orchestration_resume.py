"""Real restart planner + checkpoint reader with small actual torch state files."""

import json
import torch
import pytest
from starVLA.rl.flow_grpo.checkpoint import (
    validate_checkpoint,
    directory_seal,
    export_checkpoint,
)
from starVLA.rl.flow_grpo.config import config_hash
from starVLA.rl.flow_grpo.transactions import atomic_json
from scripts.flow_grpo.paired_experiment import advance_target


@pytest.fixture
def checkpoint_factory(tmp_path):
    cfg = {"runtime": {"deepspeed_stage": 0}, "checkpoint_contract": {"sha256": "F"}}
    provenance = {
        "sft_sha256": "F",
        "reference_sha256": "F",
        "assets": "locked-fixture",
    }
    run = tmp_path / "run"

    def create(update, incomplete=False, source="F"):
        path = run / "checkpoints" / f"update_{update:06d}"
        path.mkdir(parents=True)
        atomic_json(
            path / "trainer_state.json",
            {
                "schema": 2,
                "update": update,
                "world_size": 1,
                "config_hash": config_hash(cfg),
                "boundary": "complete_rollout_update",
                "policy_version": update,
                "provenance": {**provenance, "sft_sha256": source},
            },
        )
        atomic_json(path / "rl_config.json", cfg)
        atomic_json(
            path / "sft_parameter_manifest.json",
            {"parameters": [{"name": "weight", "shape": [1, 2]}]},
        )
        (path / "config.yaml").write_text("fixture: true")
        atomic_json(path / "normalization.json", {"fixture": True})
        (path / "processor").mkdir()
        (path / "processor/config.json").write_text("{}")
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        torch.save(
            {
                "policy.weight": model.weight.detach(),
                "policy.bias": model.bias.detach(),
            },
            path / "pytorch_model.bin",
        )
        torch.save(optimizer.state_dict(), path / "optimizer.bin")
        torch.save({"last_epoch": update}, path / "scheduler.bin")
        torch.save(
            {"rng": {"torch": torch.get_rng_state()}, "streams": [], "pending": None},
            path / "rank_0.pt",
        )
        atomic_json(path / "checkpoint_files.json", directory_seal(path))
        if not incomplete:
            (path / "COMPLETE").write_text("complete fixture optimizer boundary")
        return path

    def validate(path):
        return validate_checkpoint(path, cfg, provenance, 1)

    return run, create, validate


def actions(factory, calls):
    run, create, validate = factory

    def train(resume, target):
        start = validate(resume)["update"] if resume else 0
        calls.append((start, target))
        for step in range(start + 100, target + 1, 100):
            create(step)

    return dict(
        validate=validate,
        train=train,
        export=lambda checkpoint, step: export_checkpoint(
            checkpoint, run / f"export_update{step}"
        ),
        evaluate=lambda export, step: {
            "update": step,
            "fixture_control_flow_only": True,
        },
    )


def test_resume400_uses300_not_previous_eval_boundary(checkpoint_factory):
    run, create, _ = checkpoint_factory
    for n in (100, 200, 300):
        create(n)
    calls = []
    advance_target(run, 400, **actions(checkpoint_factory, calls))
    assert calls == [(300, 400)]


def test_incomplete300_retained_and_resume_uses200(checkpoint_factory):
    run, create, _ = checkpoint_factory
    create(100)
    create(200)
    partial = create(300, incomplete=True)
    calls = []
    advance_target(run, 400, **actions(checkpoint_factory, calls))
    assert calls == [(200, 400)]
    assert len(list(partial.parent.glob(partial.name + ".attempt-*"))) == 1


def test_export_failure_does_not_retrain_saved200(checkpoint_factory, monkeypatch):
    run, create, _ = checkpoint_factory
    create(100)
    create(200)
    calls = []
    operations = actions(checkpoint_factory, calls)
    original = torch.save

    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError("injected interrupted export")

    with monkeypatch.context() as patch:
        patch.setattr(torch, "save", fail)
        with pytest.raises(OSError, match="interrupted export"):
            advance_target(run, 200, **operations)
    advance_target(run, 200, **operations)
    advance_target(
        run, 200, **operations
    )  # completed checkpoint/artifact path is idempotent
    assert calls == []
    assert (run / "export_update200/COMPLETE").exists()
    assert len(list(run.glob("export_update200.incomplete.attempt-*"))) == 1


@pytest.mark.parametrize("damage", ["name", "source", "optimizer", "rank", "world"])
def test_conflicting_or_corrupt_later_checkpoint_is_not_ignored(
    checkpoint_factory, damage
):
    run, create, _ = checkpoint_factory
    create(100)
    create(200)
    path = create(300, source="U" if damage == "source" else "F")
    if damage == "name":
        path.rename(path.with_name("update_000301"))
    elif damage == "optimizer":
        (path / "optimizer.bin").unlink()
    elif damage == "rank":
        (path / "rank_0.pt").unlink()
    elif damage == "world":
        state = json.loads((path / "trainer_state.json").read_text())
        state["world_size"] = 2
        atomic_json(path / "trainer_state.json", state)
    calls = []
    with pytest.raises((ValueError, FileNotFoundError)):
        advance_target(run, 200, **actions(checkpoint_factory, calls))
    assert not calls


def test_interrupted_schedule_matches_continuous_committed_updates(checkpoint_factory):
    run, create, validate = checkpoint_factory
    operations = actions(checkpoint_factory, [])
    committed = []
    interrupted = {200, 300}

    def train(resume, target):
        start = validate(resume)["update"] if resume else 0
        for step in range(start + 100, target + 1, 100):
            create(step)
            committed.append(step)
            if step in interrupted:
                interrupted.remove(step)
                raise RuntimeError("process exited after checkpoint")

    operations["train"] = train
    for target in (100, 200, 400):
        while True:
            try:
                advance_target(run, target, **operations)
                break
            except RuntimeError as exc:
                assert "process exited" in str(exc)
    assert committed == [100, 200, 300, 400]
    progress = json.loads((run / "orchestration_progress.json").read_text())
    assert progress["training_latest_update"] == 400
    assert (
        set(progress["exports"])
        == set(progress["dev_evaluations"])
        == {"100", "200", "400"}
    )
