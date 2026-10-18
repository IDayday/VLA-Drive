"""Independent affected-path checks; synthetic inputs, CPU and no dispatch."""
import ast
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch

from starVLA.model.modules.flow_posttraining import fpo_objective as current_fpo
from starVLA.model.modules.flow_posttraining.advantage_scale import normalize_rollout_advantages
from tools.st_world_policy import train_flow_policy as trainer
from tools.st_world_policy import flow_training_config as current_config
from tools.st_world_policy.training_state import terminal
from test_advantage_scale import identities
from test_flow_training import config, setup_rollout, scored


LEGACY = Path(os.environ["LEGACY_REVIEW_SOURCE"])


def function(source, name, namespace):
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    scope = dict(namespace)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<fixed-6714-function>", "exec"), scope)
    return scope[name]


def same(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left: same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right): same(a, b)
    else:
        assert left == right


def test_original_fpo_functions_and_default_are_unchanged():
    old_text = (LEGACY / "fpo_objective.py").read_text()
    new_text = Path(current_fpo.__file__).read_text()
    old = ast.parse(old_text); new = ast.parse(new_text)
    old_fn = {n.name: ast.dump(n) for n in old.body if isinstance(n, ast.FunctionDef)}
    new_fn = {n.name: ast.dump(n) for n in new.body if isinstance(n, ast.FunctionDef)}
    assert old_fn == new_fn  # Only the config enum may change, not original arithmetic.
    spec = importlib.util.spec_from_file_location("fixed_6714_fpo", LEGACY / "fpo_objective.py")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert asdict(module.FPOConfig()) == asdict(current_fpo.FPOConfig())
    generator = torch.Generator().manual_seed(17)
    rewards = torch.rand(24, 4, generator=generator)
    valid = torch.ones_like(rewards, dtype=torch.bool)
    a = current_fpo.group_centered_advantages(rewards, valid)
    assert torch.equal(a, module.group_centered_advantages(rewards, valid))
    old_cfm = torch.rand(96, 50, generator=generator)
    current = torch.rand(96, 50, generator=generator).requires_grad_()
    loss1, _ = module.fpo_surrogate(old_cfm, current, a.flatten(), valid.flatten(), module.FPOConfig(), global_count=96, world_size=1)
    loss2, _ = current_fpo.fpo_surrogate(old_cfm, current, a.flatten(), valid.flatten(), current_fpo.FPOConfig(), global_count=96, world_size=1)
    g1 = torch.autograd.grad(loss1, current, retain_graph=True)[0]
    g2 = torch.autograd.grad(loss2, current)[0]
    assert torch.equal(loss1, loss2) and torch.equal(g1, g2)


def test_legacy_numeric_target_config_and_feedback_payload_are_retained(tmp_path):
    trainer_text = (LEGACY / "train_flow_policy.py").read_text()
    cfg_text = (LEGACY / "flow_training_config.py").read_text()
    old_numeric = function(trainer_text, "numeric_execution", trainer.__dict__)
    old_target = function(cfg_text, "target_contract", current_config.__dict__)
    old_validate = function(cfg_text, "validate_config", current_config.__dict__)
    binding = {"binding_sha256": "a"*64, "inference_contract": {"precision": "synthetic"}}
    for cell in ("C0", "C1", "C2", "C3", "C4"):
        cfg = config(cell)
        assert old_numeric(cfg, binding) == trainer.numeric_execution(cfg, binding)
        assert old_target(cfg) == current_config.target_contract(cfg)
        assert asdict(old_validate(cfg)) == asdict(current_config.validate_config(cfg))
    _, _, old, streams, cond, rollout, requests = setup_rollout(1)
    from starVLA.model.modules.flow_posttraining import prepare_fpo_batch
    paired = prepare_fpo_batch(trainer.MatchedOldMicrobatches(old, 1), rollout, cond,
        config=current_config.validate_config(config()), streams=streams, mc_chunk=5)
    feedback = trainer.feedback_from_records(rollout, requests,
        [scored(r, score=s) for r, s in zip(requests, [0., .2, .8, 1.])], "C1", "c"*64)
    old_save = function(trainer_text, "save_update_feedback", trainer.__dict__)
    before = torch.get_rng_state().clone()
    a = old_save(tmp_path/"old", rollout, paired, feedback, numeric_hash="3"*64, rank=0)
    b = trainer.save_update_feedback(tmp_path/"new", rollout, paired, feedback, numeric_hash="3"*64, rank=0)
    assert a["identity"] == b["identity"]
    same(torch.load(a["payload"]["path"], weights_only=False), torch.load(b["payload"]["path"], weights_only=False))
    assert torch.equal(before, torch.get_rng_state())


def test_runtime_zero_receipt_does_not_report_real_amplification():
    rewards = torch.tensor([[0.]*4, [.5]*4, [1.]*4])
    common, local = identities(3)
    output, receipt = normalize_rollout_advantages(rewards, torch.ones_like(rewards, dtype=torch.bool),
        collective_device="cpu", expected_global_count=12, common_identity=common, local_identity=local, distributed=False)
    assert torch.equal(output, torch.zeros_like(output))
    assert receipt["common"]["all_zero"] and receipt["common"]["gain"] is None
    assert receipt["common"]["denominator_reciprocal"] == 1e10
    assert receipt["common"]["count"] == 12


def test_native_terminal_and_ordering_do_not_add_a_scale_or_optimizer_update():
    # Actual source ordering, not a simulation of CUDA/ZeRO resume.
    source = Path(trainer.__file__).read_text()
    tree = ast.parse(source)
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    restore = [n.lineno for n in ast.walk(run) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "verify_restored_normalization"]
    rng_restore = [n.lineno for n in ast.walk(run) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "restore_rng"]
    terminal_guards = [n for n in ast.walk(run) if isinstance(n, ast.If) and isinstance(n.test, ast.Call) and isinstance(n.test.func, ast.Name) and n.test.func.id == "terminal"]
    training_loops = [n for n in ast.walk(run) if isinstance(n, ast.While)]
    assert restore and rng_restore and max(restore) < min(rng_restore)
    assert terminal_guards and training_loops
    first_guard = min(terminal_guards, key=lambda n:n.lineno)
    assert first_guard.lineno < min(n.lineno for n in training_loops)
    assert any(isinstance(n, ast.Return) for n in ast.walk(first_guard))
    assert terminal({"completed":4096,"termination_reason":"MAX_STEPS"},4096)
    assert not terminal({"completed":3584,"termination_reason":"OPERATOR_PAUSED"},4096)
