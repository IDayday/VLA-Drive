"""Independent finite C4 control/receipt review; no launcher is imported/run."""
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
signal.alarm(120)
started = time.monotonic()
source = Path('/mnt/project/VLA-Drive-st-c4-dev-preparation-20261017')
commit = '0fed55bd7d1c39121117a49af4ec845b3950b8a3'
base = source / 'reports/st_world_policy/20261016/evaluation'
out = Path(__file__).resolve().parent
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip() == commit
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=source, text=True).strip()
checks, pins = [], []

def sha(path):
    path = Path(path)
    assert path.stat().st_size <= 2 * 1024 * 1024
    blob = path.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    pins.append({'path': str(path), 'sha256': digest, 'bytes': len(blob)})
    return digest

def read_pin(pin):
    assert sha(pin['path']) == pin['sha256']
    return json.loads(Path(pin['path']).read_text())

def ident(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

for directory in ('C4_HELPER_v5', 'full_dev_C4_execution_v1'):
    folder = base / directory
    manifest = json.loads((folder / 'MANIFEST.json').read_text())
    sha(folder / 'MANIFEST.json')
    for name, expected in manifest['files'].items():
        assert sha(folder / name) == expected
        rel = str((folder / name).relative_to(source))
        committed = subprocess.check_output(['git', 'show', f'{commit}:{rel}'], cwd=source)
        assert hashlib.sha256(committed).hexdigest() == expected
    checks.append({'name': directory + ': manifest bytes and exact committed files', 'status': 'PASS'})
binding = json.loads((base / 'full_dev_C4_execution_v1/TRAINING_BINDINGS.json').read_text())['bindings']['C4']
documents = {k: read_pin(binding[k]) for k in ('registration', 'config', 'approval', 'launch', 'run_identity_file')}
assert binding['runtime_identity'] == documents['run_identity_file']['identity'] == '4e023061b8ac16114cc1f9e3b5e440cb2c3a23d524d562aa1276270567f5fbbb'
assert binding['source_sha'] == documents['registration']['source_sha'] == documents['launch']['source_sha'] == '6714b1e9fb2e0c8059b90b8b4d48cd40796fe345'
assert documents['approval']['formal_launch_allowed'] is True
assert documents['approval']['fingerprint'] == binding['registration_fingerprint']
assert documents['launch']['registration_sha256'] == binding['registration']['sha256']
assert documents['launch']['approval_sha256'] == binding['approval']['sha256']
for role, ref in binding['review_receipts'].items():
    review = read_pin(ref)
    assert review['status'] == 'PASS' and review['role'] == role
    assert review['fingerprint'] == binding['registration_fingerprint']
checks.append({'name': 'actual C4 registration/approval/launch/source/runtime/review pins', 'status': 'PASS'})

folder = Path('/mnt/project/st-world-policy-20261016/formal/consequence/runs/C4_fulltrain_seed42_official_4096_v1/checkpoints/step_000512_1792223517823880689')
native_path = folder / 'COMPLETE.json'
native_sha = sha(native_path)
native = json.loads(native_path.read_text())
assert native['state']['completed'] == 512 and native['state']['identity'] == binding['runtime_identity']
assert native['state']['tag'] == folder.name and native['world_size'] == 4
assert ident({k:v for k,v in native.items() if k != 'receipt_identity'}) == native['receipt_identity']
assert set(native['files']) == {p.name for p in folder.glob('*.pt')}
assert all((folder / name).stat().st_size == value['bytes'] for name, value in native['files'].items())
checks.append({'name': 'real C4 native512 receipt self-hash/inventory/stat only', 'status': 'PASS'})

# Compile only the actual pure metadata guard and native inventory helper.
guard_path = base / 'C4_HELPER_v5/launch_completed_dev_score_v5.py'
state_path = Path('/mnt/project/VLA-Drive-st-flow-context-runtime-v3-20261017/tools/st_world_policy/training_state.py')
namespace = {'identity_hash': ident, 'TERMINATIONS': {'RUNNING_CHECKPOINT','OPERATOR_PAUSED','BUDGET_PAUSED','MAX_STEPS'}}
for path, name in ((state_path, 'require_native_inventory'), (guard_path, 'validate_native_reference')):
    sha(path)
    functions = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(functions) == 1
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
intent = {'training_world_size': 4, 'runtime_identity': binding['runtime_identity'],
          'expected_completed': 512, 'training_source_sha': binding['source_sha']}
registration = {'receipt_identity': native['receipt_identity'], 'native_files': native['files'],
    'checkpoint_tag': folder.name, 'complete_receipt': {'path': str(native_path), 'sha256': native_sha}}
checkpoint = {'receipt_identity': native['receipt_identity'], 'run_identity': binding['runtime_identity'],
    'completed': 512, 'tag': folder.name, 'native_receipt_sha256': native_sha,
    'native_checkpoint_sha256': ident({k:v['sha256'] for k,v in native['files'].items()}),
    'scope': 'FORMAL', 'weights_usable_for_formal_init': True, 'training_source_sha': binding['source_sha']}
checkpoint['sha256'] = ident(checkpoint)
guard = namespace['validate_native_reference']
guard(registration, intent, checkpoint, native)
checks.append({'name': 'positive native guard with real512 metadata (in-memory fixture only)', 'status': 'PASS'})

def reject(name, function):
    try:
        function()
    except AssertionError:
        checks.append({'name': name, 'status': 'REJECTED_AS_REQUIRED'})
        return
    raise AssertionError('Guard accepted invalid metadata: ' + name)

bad = copy.deepcopy(checkpoint)
bad['native_receipt_sha256'] = '0' * 64
bad['sha256'] = ident({k:v for k,v in bad.items() if k != 'sha256'})
reject('C4 wrong native receipt SHA despite self-consistent checkpoint hash',
       lambda: guard(registration, intent, bad, native))
wrong_native, wrong_reg, wrong_checkpoint = copy.deepcopy(native), copy.deepcopy(registration), copy.deepcopy(checkpoint)
wrong_native['state']['completed'] = 513
wrong_native['receipt_identity'] = ident({k:v for k,v in wrong_native.items() if k != 'receipt_identity'})
wrong_reg['receipt_identity'] = wrong_checkpoint['receipt_identity'] = wrong_native['receipt_identity']
wrong_checkpoint['completed'] = 513
wrong_checkpoint['sha256'] = ident({k:v for k,v in wrong_checkpoint.items() if k != 'sha256'})
reject('C4 step513 rejected against fixed512 despite re-sealed metadata',
       lambda: guard(wrong_reg, intent, wrong_checkpoint, wrong_native))

result = {'schema': 'st_C4_helper_v5_independent_native_boundary_review_v1',
    'status': 'PASS', 'reviewer_agent': '/root/st_consequence', 'candidate_commit': commit,
    'review_scope': 'C4 additive evaluation control only; not a new algorithm/training review or dispatch approval',
    'checked_utc': datetime.now(timezone.utc).isoformat(), 'checks': checks,
    'real_native_receipt': {'path': str(native_path), 'sha256': native_sha,
        'receipt_identity': native['receipt_identity'], 'state': native['state'], 'world_size': 4},
    'pins': pins, 'native_payload_hashes_recomputed': False,
    'test_fixtures_written_to_runtime': False, 'launcher_main_executed': False,
    'dispatch_calls': 0, 'GPU_calls': 0, 'scoring_calls': 0,
    'script_sha256': sha(__file__), 'elapsed_seconds': time.monotonic()-started}
(out / 'NATIVE_BOUNDARY_REVIEW.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
print(json.dumps({'status': 'PASS', 'checks': len(checks), 'native_receipt_sha256': native_sha,
                  'elapsed_seconds': result['elapsed_seconds']}))
