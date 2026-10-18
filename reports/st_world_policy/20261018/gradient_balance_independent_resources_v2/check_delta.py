#!/usr/bin/env python3
"""CPU-only delta checks; no assets, host probes, CUDA, dispatch or retries."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
started = time.monotonic()
wrapper = Path(sys.argv[1])
base_commit = 'b74c1ae0bba51236fffd5fcee2ab732dce118e19'
relative = 'reports/st_world_policy/20261018/gradient_balance_systems_v1/supervise_gradient_once.py'
baseline = subprocess.check_output(['git', 'show', base_commit + ':' + relative], cwd=wrapper.parents[4], timeout=5)
current = wrapper.read_bytes()
old_ast, new_ast = ast.parse(baseline), ast.parse(current)
old_nodes = {n.name: n for n in old_ast.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
new_nodes = {n.name: n for n in new_ast.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
unchanged = ['small_json', 'write_json', 'ownership_helpers', 'ReadAdmission', 'replace_aliases',
             'directory_bytes', 'process_sample', 'supervise']
for name in unchanged:
    assert ast.dump(old_nodes[name]) == ast.dump(new_nodes[name]), name
spec = importlib.util.spec_from_file_location('gradient_resource_delta', wrapper)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert 'torch' not in sys.modules
assert (module.READ_LIMIT, module.RSS_LIMIT, module.OUTPUT_LIMIT, module.WALL) == (192 << 30, 64 << 30, 8 << 20, 1800)
assert module.SOURCE == '83679937fe9702be39f0184154fdf9ebdc4eb98a'
assert module.REG_SHA == 'fb55ca585688cd2ca19ea55af55628c60c91554c00f68ef4a6ae1821d22b06ec'
assert str(module.ATTEMPT).endswith('gradient_balance_boundary3584_v2_controls/RUNTIME_ATTEMPT')
environment = {'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
assert module.RUNTIME_ENVIRONMENT == environment
assert module.validate_runtime_environment(environment) == environment
for bad in ({}, {'CUBLAS_WORKSPACE_CONFIG': ':16:8'}, {'CUBLAS_WORKSPACE_CONFIG': 'invalid'}):
    try:
        module.validate_runtime_environment(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('invalid runtime environment admitted')
sys.modules['torch'] = object()
try:
    try:
        module.validate_runtime_environment(environment)
    except ValueError:
        pass
    else:
        raise AssertionError('late environment validation admitted')
finally:
    del sys.modules['torch']
env = module.child_environment(environment, {'gpu_uuid': 'fixture-only'}, 123)
assert env['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
assert env['CUDA_VISIBLE_DEVICES'] == 'fixture-only'
assert all(env[name] == '2' for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'))
assert env['ST_GRADIENT_LOCK_FD'] == '123'
assert 'validate_runtime_environment()' in ast.unparse(new_nodes['child_main'].body[0])
assert 'validate_runtime_environment()' in ast.unparse(new_nodes['validate_admission'].body[0])
old_reg = json.loads(Path('/mnt/project/st-world-policy-20261016/audits/consequence/gradient_balance_boundary3584_v1_controls/REGISTRATION.json').read_text())
new_reg = json.loads(module.REGISTRATION.read_text())
assert [k for k in old_reg if old_reg[k] != new_reg[k]] == ['output']
assert new_reg['output'].endswith('gradient_balance_boundary3584_v2')
assert 'ATTEMPT.mkdir(exist_ok=False)' in current.decode()
assert "if Path(r['output']).exists():raise ValueError('Diagnostic output already exists; no retry')" in current.decode()
result = {'schema': 'gradient_resource_delta_checks_v2', 'status': 'PASS',
          'wrapper_sha256': hashlib.sha256(current).hexdigest(), 'baseline_wrapper_commit': base_commit,
          'unchanged_resource_functions': unchanged, 'registration_changed_fields': ['output'],
          'fixed_namespace': str(module.ATTEMPT), 'runtime_environment': environment,
          'missing_wrong_or_late_environment_rejected': True, 'torch_imported': False,
          'old_attempt_not_opened_or_written': True, 'same_read_rss_output_wall_limits': True,
          'assets_read': False, 'host_probes': 0, 'gpu_calls': 0, 'dispatches': 0,
          'elapsed_seconds': time.monotonic() - started}
out = Path(__file__).parent / 'DELTA_CHECKS.json'
out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(json.dumps(result))
