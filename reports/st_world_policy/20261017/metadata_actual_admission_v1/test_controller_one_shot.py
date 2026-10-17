"""CPU-only controller admission counterexample. No real child or ledger."""
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import time

import pytest
import subprocess


CONTROLLER = Path('/mnt/project/st-world-policy-20261016/resources/run_bounded_metadata_v1.py')


@pytest.mark.parametrize('remove_atomic_check', [False, True])
def test_duplicate_started_record_is_rejected_under_storage_lock(tmp_path, monkeypatch, remove_atomic_check):
    spec = importlib.util.spec_from_file_location('review_controller', CONTROLLER)
    c = importlib.util.module_from_spec(spec)
    if remove_atomic_check:
        source = CONTROLLER.read_text()
        protected = "            if path.exists():\n                raise ValueError('Immutable invocation marker already exists: '+str(path))\n"
        assert protected in source
        exec(compile(source.replace(protected, ''), str(CONTROLLER), 'exec'), c.__dict__)
    else:
        spec.loader.exec_module(c)
    out = tmp_path / 'metadata'
    prep_path = tmp_path / 'PREPARED.json'
    prep = {'index_storage_registration': 'registration', 'index_storage_registration_sha256': 'b'*64,
            'new_policy': 'policy', 'new_policy_sha256': 'c'*64}
    prep_path.write_text(json.dumps(prep))
    review = {'status': 'PASS_BOUNDED_METADATA_ADMISSION',
              'prepared_sha256': c.sha(prep_path), 'controller_sha256': c.sha(CONTROLLER),
              'valid_until_unix': time.time() + 60}
    reg = {'output_root': str(out)}
    started = out / 'CONTROL_CURRENT_INDEX_STARTED.json'
    competing_record = {'fixture': 'another controller already admitted'}
    child_calls = []

    def fake_load(path):
        return {'allowed_hosts': ['training-vla-zt']} if str(path).endswith('server_dispatch_policy.json') else prep

    @contextmanager
    def storage_lock(**_):
        # Simulate another controller winning exactly after the outer exists
        # check but before this controller acquires the existing ledger lock.
        out.mkdir(exist_ok=True)
        started.write_text(json.dumps(competing_record))
        yield SimpleNamespace(check=lambda **_: None)

    def atomic(path, value):
        path.write_text(json.dumps(value))

    def must_not_launch(*args, **kwargs):
        child_calls.append(args)
        raise RuntimeError('fixture caught duplicate child launch')

    monkeypatch.setattr(c, 'PREP', prep_path)
    monkeypatch.setattr(c, 'load', fake_load)
    monkeypatch.setattr(c.socket, 'gethostname', lambda: 'training-vla-zt-worker-0')
    monkeypatch.setattr(c.subprocess, 'check_output', lambda cmd, **_: '1162b955\n' if 'rev-parse' in cmd else '')
    monkeypatch.setattr(c.subprocess, 'Popen', must_not_launch)
    monkeypatch.setattr(c, 'm', SimpleNamespace(
        pinned_json=lambda path, sha: (review if path == 'review' else reg, b''),
        metadata_writing=storage_lock))
    monkeypatch.setattr(c, 'b', SimpleNamespace(_atomic=atomic))
    monkeypatch.setattr(c.sys, 'argv', ['controller', '--stage', 'current_index',
        '--checkpoint-receipt', 'checkpoint', '--checkpoint-sha256', 'a'*64,
        '--review', 'review', '--review-sha256', 'd'*64])
    try:
        c.main()
    except (FileExistsError, ValueError, RuntimeError, AssertionError):
        pass
    if remove_atomic_check:
        assert len(child_calls) == 1, 'negative mutation did not reproduce the original admission race'
        assert json.loads(started.read_text()) != competing_record
    else:
        assert child_calls == [], 'a competing START receipt did not stop duplicate child launch'
        assert json.loads(started.read_text()) == competing_record, 'competing immutable START was overwritten'


@pytest.mark.parametrize('case', ['success', 'child_failure', 'rss', 'deadline', 'pipe_limit'])
def test_controller_finite_child_accounting(tmp_path, monkeypatch, case):
    spec = importlib.util.spec_from_file_location('review_controller', CONTROLLER)
    c = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c)
    out = tmp_path / 'metadata'
    prep_path = tmp_path / 'PREPARED.json'
    prep = {'future_storage_registration': 'registration', 'future_storage_registration_sha256': 'b'*64,
            'new_policy': 'policy', 'new_policy_sha256': 'c'*64}
    prep_path.write_text(json.dumps(prep))
    review = {'status': 'PASS_BOUNDED_METADATA_ADMISSION',
              'prepared_sha256': c.sha(prep_path), 'controller_sha256': c.sha(CONTROLLER),
              'valid_until_unix': time.time() + 60}
    reg = {'output_root': str(out)}
    atomic_values = {}

    @contextmanager
    def storage_lock(**_):
        out.mkdir(exist_ok=True)
        yield SimpleNamespace(check=lambda **_: None)

    def atomic(path, value):
        atomic_values[path.name] = value
        path.write_text(json.dumps(value))

    actual_popen = subprocess.Popen
    children = []
    codes = {
        'success': "print('{\"status\":\"GUARDED_METADATA_RETURN\",\"artifact\":{\"cost\":{\"source_bytes_read\":123}}}')",
        'child_failure': "raise SystemExit(2)",
        'rss': "import time; time.sleep(3)",
        'deadline': "import time; time.sleep(3)",
        'pipe_limit': "print('x'*(1024*1024+1024))",
    }

    def bounded_fixture_child(command, **kwargs):
        assert command[:3] == ['taskset', '-c', '0,1']
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
        child = actual_popen(['taskset', '-c', '0,1', c.sys.executable, '-c', codes[case]],
                             **dict(kwargs, cwd=tmp_path))
        children.append(child)
        return child

    monkeypatch.setattr(c, 'PREP', prep_path)
    monkeypatch.setattr(c, 'load', lambda path: {'allowed_hosts': ['training-vla-zt']}
                        if str(path).endswith('server_dispatch_policy.json') else prep)
    monkeypatch.setattr(c.socket, 'gethostname', lambda: 'training-vla-zt-worker-0')
    monkeypatch.setattr(c.subprocess, 'check_output', lambda cmd, **_: '1162b955\n' if 'rev-parse' in cmd else '')
    monkeypatch.setattr(c.subprocess, 'Popen', bounded_fixture_child)
    monkeypatch.setattr(c, 'm', SimpleNamespace(
        pinned_json=lambda path, sha: (review if path == 'review' else reg, b''),
        metadata_writing=storage_lock,
        process_tree_rss=lambda _: 4*2**30+1 if case == 'rss' else 1024))
    monkeypatch.setattr(c, 'b', SimpleNamespace(_atomic=atomic))
    if case == 'deadline':
        monkeypatch.setattr(c, 'time', SimpleNamespace(time=time.time, monotonic=lambda: c.START+446))
    monkeypatch.setattr(c.sys, 'argv', ['controller', '--stage', 'future_build',
        '--checkpoint-receipt', 'checkpoint', '--checkpoint-sha256', 'a'*64,
        '--review', 'review', '--review-sha256', 'd'*64])
    result = c.main()
    record = atomic_values['CONTROL_FUTURE_BUILD_RESULT.json']
    assert result == (0 if case == 'success' else 2)
    assert record['status'] == ('PASS' if case == 'success' else 'FAILED')
    assert record['charged_source_bytes'] == (123 if case == 'success' else 4*2**30)
    assert len(children) == 1 and children[0].poll() is not None
    assert len(record.get('stderr', '')) <= 2**20
    assert len(record.get('stdout_unparsed') or '') <= 2**20
