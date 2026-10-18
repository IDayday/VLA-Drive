#!/usr/bin/env python3
"""Test the exact shared-capacity gate with mocked GPU metadata only."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

sys.dont_write_bytecode = True
started = time.monotonic()
path = Path('/mnt/project/VLA-Drive-st-world-policy-20261016/reports/st_world_policy/20261018/gradient_balance_remote_v3/supervise_gradient_once.py')
data = path.read_bytes()
old = subprocess.check_output(['git', 'show', '0792bc2853c7a01914627ad7232b27fcf01299d9:reports/st_world_policy/20261018/gradient_balance_remote_v3/supervise_gradient_once.py'], cwd=path.parents[4], timeout=5)
def rest(source):
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if not (isinstance(n, ast.FunctionDef) and n.name == 'validate_admission')
                 and not (isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'GPU_ADMISSION')]
    return ast.dump(tree)
assert rest(old) == rest(data)
spec = importlib.util.spec_from_file_location('shared_gate', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
assert 'torch' not in sys.modules
uuid = 'GPU-c0a3d651-8691-b507-5df6-6c7f0cc6c897'
host = 'training-vlawm-tsinghua-zt2-worker-0'
cpus = sorted(mod.os.sched_getaffinity(0))[:2]
review = dict(status='APPROVE_GRADIENT_BALANCE_EXECUTION', reviewer='/root/independent_fixture',
              source_sha=mod.SOURCE, registration_sha256=mod.REG_SHA,
              wrapper_sha256=hashlib.sha256(data).hexdigest(), runtime_environment=mod.RUNTIME_ENVIRONMENT,
              gpu_admission=mod.GPU_ADMISSION)
admission = dict(status='APPROVED', registration_sha256=mod.REG_SHA, source_sha=mod.SOURCE,
                 wrapper_sha256=hashlib.sha256(data).hexdigest(),
                 helper_sha256={n: hashlib.sha256((mod.HELPERS / n).read_bytes()).hexdigest() for n in ('supervise_once.py', 'prepare_bindings.py')},
                 runtime_environment=mod.RUNTIME_ENVIRONMENT, gpu_admission=mod.GPU_ADMISSION,
                 host=host, gpu_uuid=uuid, valid_until_unix=time.time()+3600,
                 cpu_affinity=cpus, dispatch_policy_sha256='fixture',
                 reviews={k: {'path': k, 'sha256': 'fixture'} for k in ('resources', 'correctness')})
def small_json(p, expected=None):
    if Path(p) == mod.REGISTRATION: return {'resource': {'host': host, 'gpu_uuid': uuid}}
    if Path(p) == mod.POLICY: return {'allowed_aliases': [host]}
    if str(p) in ('resources', 'correctness'): return dict(review, review_kind=str(p))
    return admission
results = []
for name, value, expected in (
    ('exact_threshold_accepts_shared_gpu', uuid+', 61440', True),
    ('spare_memory_accepts_shared_gpu', uuid+', 81153', True),
    ('one_mib_below_rejected', uuid+', 61439', False),
    ('wrong_uuid_rejected', 'GPU-other, 81153', False),
    ('malformed_metadata_rejected', uuid+', N/A', False),
):
    calls = []
    def output(args, **kwargs):
        calls.append(args)
        if args[0] == 'git': return mod.SOURCE+'\n'
        assert args[0] == 'nvidia-smi' and args[args.index('-i')+1] == uuid
        return value
    with patch.object(mod, 'small_json', small_json), patch.object(mod.socket, 'gethostname', lambda: host), \
         patch.object(mod.subprocess, 'check_output', output), patch.object(mod.subprocess, 'run', lambda *a, **k: type('P', (), {'returncode': 0})()), \
         patch.dict(mod.os.environ, {'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}):
        try:
            mod.validate_admission('fixture_admission')
            accepted = True
        except ValueError:
            accepted = False
    assert accepted == expected, (name, accepted)
    assert not any('--query-compute-apps=pid' in c for c in calls)
    results.append({'name': name, 'status': 'PASS', 'accepted': accepted})
result = {'status': 'PASS', 'wrapper_sha256': hashlib.sha256(data).hexdigest(),
          'only_changed_ast': ['GPU_ADMISSION', 'validate_admission'],
          'tests': results, 'GPU_queries_mocked': True, 'torch_imported': False,
          'GPU_computation': False, 'dispatches': 0, 'elapsed_seconds': time.monotonic()-started}
(Path(__file__).parent / 'SHARED_ADMISSION_CHECKS.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
print(json.dumps(result))
