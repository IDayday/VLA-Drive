"""Independent CPU-only checks of the exact workspace-preflight change."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

from tools.st_world_policy import calibrate_temporal_scale as cli

REPO = Path(__file__).resolve().parents[4]
OLD = Path('/mnt/project/VLA-Drive-st-loss-scale-real4-v2-20261017')
ACTUAL = Path('/mnt/project/VLA-Drive-st-loss-scale-real4-v3-20261017')
PROPOSAL = REPO / 'reports/st_world_policy/20261017/temporal_scale_cublas_preflight_fix_v1'
spec = importlib.util.spec_from_file_location('independent_launch_source', REPO /
    'reports/st_world_policy/20261017/temporal_scale_calibration_v1/run_once.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def test_preflight_does_not_advance_any_cpu_rng_or_mutate_environment(monkeypatch):
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone(), dict(os.environ))
    monkeypatch.setattr(torch.cuda, '_lazy_init', lambda: pytest.fail('No CUDA initialization'))
    output = cli.runtime_preflight({'runtime_environment': cli.RUNTIME_ENVIRONMENT},
                                   {'runtime': {'deterministic': True}})
    assert output['cuda_initialized_at_check'] is False
    assert random.getstate() == before[0]
    now = np.random.get_state()
    assert now[0] == before[1][0] and np.array_equal(now[1], before[1][1]) and now[2:] == before[1][2:]
    assert torch.equal(torch.get_rng_state(), before[2])
    assert dict(os.environ) == before[3]


def test_workspace_is_in_child_environment_at_interpreter_start(monkeypatch):
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    env = launcher.child_runtime_environment(dict(os.environ))
    value = subprocess.check_output([sys.executable, '-c',
        'import os,json,sys;print(json.dumps([os.environ.get("CUBLAS_WORKSPACE_CONFIG"),"torch" in sys.modules]))'],
        env=env, text=True, timeout=10)
    assert json.loads(value) == [':4096:8', False]
    assert 'CUBLAS_WORKSPACE_CONFIG' not in os.environ


def test_execute_computation_ast_unchanged_after_removing_only_preflight_receipts():
    def body(path):
        tree = ast.parse(path.read_text())
        return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'execute')
    old = body(OLD / 'tools/st_world_policy/calibrate_temporal_scale.py')
    new = copy.deepcopy(body(REPO / 'tools/st_world_policy/calibrate_temporal_scale.py'))
    first = new.body.pop(0)
    assert ast.unparse(first) == 'preflight = runtime_preflight(cfg, training)'
    context = next(node for node in new.body if isinstance(node, ast.With))
    receipt = [node for node in context.body if isinstance(node, ast.Assign)
               and isinstance(node.value, ast.Dict)
               and any(isinstance(k, ast.Constant) and k.value == 'calibration_preflight' for k in node.value.keys)]
    assert len(receipt) == 1
    context.body.remove(receipt[0])
    assert ast.dump(new, include_attributes=False) == ast.dump(old, include_attributes=False)


def test_configs_change_only_declared_namespace_wall_and_environment():
    old = json.loads((OLD / 'reports/st_world_policy/20261017/temporal_scale_calibration_v1/PROPOSED_CALIBRATION_CONFIG.json').read_text())
    new = json.loads((PROPOSAL / 'PROPOSED_CALIBRATION_CONFIG.json').read_text())
    assert new.pop('runtime_environment') == cli.RUNTIME_ENVIRONMENT
    assert new['output_root'].endswith('loss_scale_calibration_real4_v2/result')
    new['output_root'] = old['output_root']
    old_training = json.loads(Path(old['training_config']['path']).read_text())
    new_training = json.loads(Path(new['training_config']['path']).read_text())
    assert new_training['runtime']['max_seconds'] == 3000
    assert old_training['runtime']['max_seconds'] == 3600
    new_training['runtime']['max_seconds'] = old_training['runtime']['max_seconds']
    assert new_training['runtime']['output_root'].endswith('loss_scale_calibration_real4_v2/unused_trainer_output')
    new_training['runtime']['output_root'] = old_training['runtime']['output_root']
    assert old_training == new_training
    new['training_config'] = old['training_config']
    assert new == old


def test_actual_frozen_runtime_uses_exact_reviewed_execution_and_math_sources():
    paths = ['tools/st_world_policy/calibrate_temporal_scale.py',
             'tools/st_world_policy/calibration_measurement.py',
             'reports/st_world_policy/20261017/temporal_scale_calibration_v1/run_once.py',
             'starVLA/model/framework/st_world_model.py',
             'starVLA/model/modules/st_world/losses.py']
    for name in paths:
        a, b = (REPO / name).read_bytes(), (ACTUAL / name).read_bytes()
        assert a == b, name
    unchanged = ['tools/st_world_policy/calibration_measurement.py',
                 'starVLA/model/framework/st_world_model.py',
                 'starVLA/model/modules/st_world/losses.py']
    for name in unchanged:
        assert (REPO / name).read_bytes() == (OLD / name).read_bytes(), name


def test_failure_receipt_cannot_become_poststate_or_rng_pass():
    failure = json.loads(Path('/mnt/project/st-world-policy-20261016/audits/loss_scale_calibration_real4_v1/result/FAILED.json').read_text())
    assert failure['status'] == 'FAILED' and 'CUBLAS_WORKSPACE_CONFIG' in failure['error']
    assert not any(key in failure for key in ('model_state_before_after_equal', 'RNG_restored', 'proposal'))
    assert not Path('/mnt/project/st-world-policy-20261016/audits/loss_scale_calibration_real4_v1/result/MEASUREMENT.json').exists()
