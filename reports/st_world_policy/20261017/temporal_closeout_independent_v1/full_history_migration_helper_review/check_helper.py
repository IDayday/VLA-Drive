"""Bounded source-only checks; never call the real migrate_policy."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import time
import types

ROOT = Path('/mnt/project/VLA-Drive-st-full-history-controller-20261018/reports/st_world_policy/20261018/full_causal_history_build_preparation_v1')
EXPECTED = {'storage_binding.py': '0e2038299e62b00190ea740914f5d83eed35436d1a1746fb6d32c9a9d17117e6', 'migrate_v3.py': 'e9eaef1db528a76cbdaf10b4b8d634fb93dc8306bd23f9e67606a4ceaee677f7'}
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

start = time.monotonic()
checks = []
for name, expected in EXPECTED.items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    checks.append('exact_source_' + name)
binding = load('storage_binding', ROOT / 'storage_binding.py')
sys.modules['storage_binding'] = binding
inherited, receipt = binding.install()
from tools.st_world_policy import metadata_storage, cache_storage_budget, spatial_w_storage
assert Path(metadata_storage.__file__) == binding.RUNTIME / 'tools/st_world_policy/metadata_storage.py'
assert Path(cache_storage_budget.__file__) == binding.RUNTIME / 'tools/st_world_policy/cache_storage_budget.py'
assert Path(spatial_w_storage.__file__) == binding.RUNTIME / 'tools/st_world_policy/spatial_w_storage.py'
assert spatial_w_storage.b is cache_storage_budget
assert receipt['policy_or_ledger_written'] is False
checks.append('original_bridge_retains_exact_v3_runtime_and_current_budget')
for value in (inherited.W_COMPAT['adapter'], inherited.W_COMPAT['proof'], inherited.W_COMPAT['current_validator']):
    assert hashlib.sha256(Path(value['path']).read_bytes()).hexdigest() == value['sha256']
checks.append('unchanged_adapter_proof_and_W_pins')
entry = load('review_migrate_entry', ROOT / 'migrate_v3.py')
assert hashlib.sha256(entry.NEW_POLICY.read_bytes()).hexdigest() == entry.NEW_POLICY_SHA
checks.append('fixed_new_package_policy_pin')
calls = []
entry.install = lambda: (None, {'stub': True})
original = metadata_storage.migrate_policy
metadata_storage.migrate_policy = lambda **kwargs: calls.append(kwargs) or {'stub': True}
args = []
values = {'old-policy': '/does/not/access/old.json', 'old-policy-sha256': entry.OLD_POLICY_SHA,
          'new-policy': str(entry.NEW_POLICY), 'new-policy-sha256': entry.NEW_POLICY_SHA,
          'expected-old-ledger-sha256': 'stub', 'quiescence-receipt': '/does/not/access/q.json',
          'quiescence-sha256': 'stub', 'pilot-inventory': '/does/not/access/p.json',
          'pilot-inventory-sha256': 'stub', 'checkpoint-receipt': '/does/not/access/c.json',
          'checkpoint-sha256': 'stub'}
for key, value in values.items():
    args.extend(['--' + key, value])
def invoke(extra, changes=None):
    argv = ['migrate_v3.py'] + args + extra
    for key, value in (changes or {}).items():
        argv[argv.index('--' + key) + 1] = value
    old = sys.argv
    sys.argv = argv
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return entry.main()
    finally:
        sys.argv = old
try:
    for extra, changed in [([], {}), (['--execute'], {'old-policy-sha256': '0' * 64}),
                           (['--execute'], {'new-policy-sha256': '0' * 64}),
                           (['--execute'], {'new-policy': '/another/policy.json'})]:
        before = len(calls)
        try:
            invoke(extra, changed)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid execution was accepted')
        assert len(calls) == before
    checks.append('four_invalid_requests_reject_before_native_call')
    assert invoke(['--execute']) == 0
    assert len(calls) == 1 and calls[0] == {k.replace('-', '_'): v for k, v in values.items()} | {
        'renew_quiescence_receipt': None, 'renew_quiescence_sha256': None}
    checks.append('one_explicit_exact_request_forwards_unmodified_to_stub_native_API')
finally:
    metadata_storage.migrate_policy = original
print(json.dumps({'status': 'PASS_SOURCE_ONLY', 'checks': checks, 'elapsed_seconds': time.monotonic() - start,
                  'actual_migration_called': False, 'ledger_read_or_written': False,
                  'bridge_receipt': receipt}, sort_keys=True, indent=2))
