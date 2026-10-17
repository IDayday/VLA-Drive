"""Independent CPU-only adapter tests; no real model or checkpoint is read.

Tiny injected objects exercise the new wrapper/rank dispatch, not real native
training. The existing comparator is separately exercised on CPU RNG values.
"""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

HERE = Path(__file__).parent
OWNER = HERE.parent / "official_T1_native4_execution_v1"
spec = importlib.util.spec_from_file_location("independent_t1_adapter", OWNER / "run_once.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
EXPECTED_ADAPTER = "fe7b418dadb683c143f61d51508df32dae0c9d92f2726a4aade1950f9d112f93"


def test_exact_reviewed_adapter_and_observer_source():
    assert hashlib.sha256((OWNER / "run_once.py").read_bytes()).hexdigest() == EXPECTED_ADAPTER
    assert hashlib.sha256(m.OBSERVER.read_bytes()).hexdigest() == m.OBSERVER_SHA
    assert m.SOURCE == "e85d11810bdf0240e7295d0994534f1df4ed21a4"
    assert m.LIMITS["max_invocations"] == 2
    assert m.LIMITS["total_updates"] == 4
    assert m.LIMITS["total_scene_exposures"] == 16


def harness(monkeypatch, tmp_path, phase=4, failure=None, missing=None):
    """Inject only CPU control objects, never import/run DeepSpeed or trainer."""
    events = []

    class Engine:
        def load_checkpoint(self, load_dir, tag=None, load_module_strict=True,
                            load_optimizer_states=True, load_lr_scheduler_states=True,
                            load_module_only=False, custom_load_fn=None):
            events.append("native_load")
            return str(Path(load_dir) / tag), {"completed": 2}

        def backward(self):
            events.append("backward")

        def step(self):
            events.append("step")

    class Audit:
        def __init__(self, root, **kwargs):
            self.load_calls = self.restore_calls = 0
            self.report = {"status": "STARTING", "sections": {}}
            self.kwargs, self.root = kwargs, root
            events.append("audit_created")
            instance.append(self)

        def after_load(self, engine, folder, result):
            events.append("load_comparison")
            self.load_calls += 1
            assert folder == tmp_path / "parent" / "step_2"
            assert result[1] == {"completed": 2}
            if failure == "load":
                raise ValueError("synthetic live master mismatch")

        def after_restore(self, state, generators, capture_rng):
            events.append("rng_comparison")
            self.restore_calls += 1
            if failure == "rng":
                raise ValueError("synthetic RNG mismatch")
            self.report["status"] = "LIVE_MODEL_OPTIMIZER_AND_RNG_MATCH"

        def persist(self):
            events.append("persist")

        def finish(self, *_):
            raise AssertionError("Old zero-update finish must never run")

    trainer = ModuleType("tools.st_world_policy.train_world")
    trainer.__file__ = str(m.RUNTIME / "tools/st_world_policy/train_world.py")
    trainer.restore_rng = lambda *_: events.append("native_rng_restore")
    original_restore = trainer.restore_rng

    def run(call):
        assert call.resume is False and call.approval is None
        assert call.registration == "/synthetic_control/REGISTRATION.json"
        if phase == 4:
            engine = Engine()
            if missing != "load":
                engine.load_checkpoint(str(tmp_path / "parent"), tag="step_2",
                    load_module_strict=True, load_optimizer_states=True, load_lr_scheduler_states=True)
            if missing != "rng":
                trainer.restore_rng({"fixture": True}, {})
            for _ in range(2):
                engine.backward(); engine.step()
            trainer.restore_rng({"save_rng_fixture": True}, {})
        else:
            events.append("phase2_original_trainer")

    trainer.run = run
    instance = []
    observer = SimpleNamespace(LiveAudit=Audit,
        collective_audit=lambda fn, description: fn(),
        main=lambda *_: (_ for _ in ()).throw(AssertionError("Old main must never run")))
    modules = {}
    for name in ("tools", "tools.st_world_policy", "tools.foresight", "deepspeed", "deepspeed.runtime"):
        module = ModuleType(name); module.__path__ = []
        modules[name] = module
    modules["tools.st_world_policy.train_world"] = trainer
    modules["tools.st_world_policy"].train_world = trainer
    modules["deepspeed.runtime.engine"] = ModuleType("deepspeed.runtime.engine")
    modules["deepspeed.runtime.engine"].DeepSpeedEngine = Engine
    modules["tools.foresight.student_state"] = ModuleType("tools.foresight.student_state")
    modules["tools.foresight.student_state"].capture_rng = lambda *_: {"fixture": True}
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    a = {"phase": phase, "registration": {"path": "/synthetic_control/REGISTRATION.json"}}
    reg = {"run_id": f"T1_official_real64_phase{phase}_v1"}
    monkeypatch.setattr(m, "controls", lambda *_, **__: (a, reg, {}, 0, {"state": {"identity": "parent_identity"}}, "GPU-A,GPU-B"))
    monkeypatch.setattr(m, "load_module", lambda *_: observer)
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-A,GPU-B")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    return events, trainer, Engine, instance, original_restore


def test_actual_rank_main_orders_exact_two_update_continuation(monkeypatch, tmp_path):
    events, trainer, engine, audit, original = harness(monkeypatch, tmp_path)
    original_load = engine.load_checkpoint
    assert m.rank_main(SimpleNamespace()) == 0
    assert events == ["audit_created", "native_load", "load_comparison", "native_rng_restore",
        "rng_comparison", "backward", "step", "backward", "step", "native_rng_restore", "persist"]
    assert audit[0].kwargs["world_size"] == 2 and audit[0].kwargs["expected_step"] == 2
    assert audit[0].kwargs["identity"] == "parent_identity"
    assert audit[0].kwargs["rank"] == 1
    assert audit[0].load_calls == audit[0].restore_calls == 1
    assert audit[0].report["zero_update_exit_audit"] == "NOT_RUN_OUTSIDE_TWO_INVOCATION_BUDGET"
    assert trainer.restore_rng is original and engine.load_checkpoint is original_load


def test_phase2_calls_original_trainer_without_old_observer(monkeypatch, tmp_path):
    events, *_ = harness(monkeypatch, tmp_path, phase=2)
    monkeypatch.setattr(m, "load_module", lambda *_: (_ for _ in ()).throw(AssertionError("No phase2 observer")))
    assert m.rank_main(SimpleNamespace()) == 0
    assert events == ["phase2_original_trainer"]


@pytest.mark.parametrize("failure", ["load", "rng"])
def test_comparison_failure_prevents_update3_and_restores_hooks(monkeypatch, tmp_path, failure):
    events, trainer, engine, audit, original = harness(monkeypatch, tmp_path, failure=failure)
    original_load = engine.load_checkpoint
    with pytest.raises(ValueError, match="synthetic"):
        m.rank_main(SimpleNamespace())
    assert "backward" not in events and "step" not in events
    assert trainer.restore_rng is original and engine.load_checkpoint is original_load
    assert audit[0].report["status"] == "FAILED"


@pytest.mark.parametrize("field,value", [
    ("load_module_strict", False), ("load_optimizer_states", False),
    ("load_lr_scheduler_states", False), ("load_module_only", True),
    ("custom_load_fn", lambda *_: None),
])
def test_every_partial_restore_flag_rejected_before_loading(field, value):
    events = []
    class Engine:
        def load_checkpoint(self, load_dir, tag=None, load_module_strict=True,
                            load_optimizer_states=True, load_lr_scheduler_states=True,
                            load_module_only=False, custom_load_fn=None):
            events.append("load")
    trainer = SimpleNamespace(restore_rng=lambda *_: None)
    original = Engine.load_checkpoint, trainer.restore_rng
    with pytest.raises(ValueError, match="Incomplete native restore"):
        with m.observe_restore(trainer, Engine, None, None, None):
            Engine().load_checkpoint("/synthetic", "step2", **{field: value})
    assert events == []
    assert original == (Engine.load_checkpoint, trainer.restore_rng)


def test_actual_comparator_rng_mismatch_rejected_without_checkpoint_payload(tmp_path, monkeypatch):
    observer = m.load_module("independent_existing_observer", m.OBSERVER)
    import torch
    import numpy as np
    # Explicit synthetic values only: torch.load is replaced before after_restore.
    saved = {"python": (3, (1, 2, 3), None), "numpy": np.array([1, 2], dtype=np.uint32),
             "torch": torch.tensor([1, 2], dtype=torch.uint8), "cuda": torch.tensor([3], dtype=torch.uint8),
             "generators": {"noise": torch.tensor([4], dtype=torch.uint8)}}
    monkeypatch.setattr(observer.torch, "load", lambda *_, **__: copy.deepcopy(saved))
    audit = observer.LiveAudit(tmp_path, rank=0, run=tmp_path, expected_step=2, identity="synthetic", world_size=2)
    audit.engine = object(); audit.folder = tmp_path / "synthetic_not_a_checkpoint"
    actual = copy.deepcopy(saved); actual["generators"]["noise"][0] = 5
    with pytest.raises(ValueError, match="Actual restored RNG differs"):
        audit.after_restore(copy.deepcopy(saved), {}, lambda _: actual)
    assert audit.report["sections"]["all_actual_restored_rng"]["differences"]
    assert audit.report["status"] != "LIVE_MODEL_OPTIMIZER_AND_RNG_MATCH"


def test_actual_comparator_collective_error_path_no_distributed_launch(monkeypatch):
    observer = m.load_module("independent_collective_observer", m.OBSERVER)
    monkeypatch.setattr(observer.dist, "is_initialized", lambda: False)
    with pytest.raises(RuntimeError, match="synthetic comparison failed"):
        observer.collective_audit(lambda: (_ for _ in ()).throw(ValueError("synthetic comparison failed")), "restore")


def test_adapter_has_no_optimizer_patch_or_old_observer_entrypoint():
    tree = ast.parse((OWNER / "run_once.py").read_text())
    rank = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "rank_main")
    calls = [n.func for n in ast.walk(rank) if isinstance(n, ast.Call)]
    assert not any(isinstance(x, ast.Attribute) and x.attr in ("finish", "main") for x in calls)
    wrapper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "observe_restore")
    assigned = [n.attr for n in ast.walk(wrapper) if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)]
    assert set(assigned) == {"load_checkpoint", "restore_rng"}


def test_frozen_trainer_restore_precedes_loop_and_preserves_native_parent():
    text = (m.RUNTIME / "tools/st_world_policy/train_world.py").read_text()
    assert text.index('fresh_master_initialization = not args.resume and "continuation" not in cfg') < text.index("engine.load_checkpoint(")
    assert text.index("resolve_extension(cfg, registration") < text.index("engine.load_checkpoint(")
    assert text.index('state = dict(receipt["state"])') < text.index('restore_rng(torch.load(folder / f"rng_rank{rank}.pt"')
    assert text.index('restore_rng(torch.load(folder / f"rng_rank{rank}.pt"') < text.index('while state["completed"] < execution["max_updates"]')
    proposed = json.loads((m.RUNTIME / m.REL / "official_T1_native4_preparation_v1/PROPOSED_CONFIG_STEP2.json").read_text())
    assert proposed["execution"]["max_updates"] == 20000
    assert proposed["optimizer"]["warmup"] == 1000
    assert proposed["runtime"].get("profile_mode", "none") == "none"
