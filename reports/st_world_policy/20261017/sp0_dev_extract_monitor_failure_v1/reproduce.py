"""CPU fixture only: unchanged pinned reader and writer, controlled interleaving."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import threading
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SOURCE = Path('/mnt/project/VLA-Drive-st-spatial-probe-runtime-v2-20261017/reports/st_world_policy/20261017/spatial_probe_actual_preparation_v1/prepare_bindings.py')
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == '31d2d302d5dd441df9251762e34cd9d1762afc3463cdabfe8dbf5b1f12577c84'
spec = importlib.util.spec_from_file_location('pinned_controls', SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
fstat = os.fstat
results = {}

with tempfile.TemporaryDirectory(prefix='st-counter-race-') as temporary:
    path = Path(temporary) / 'COUNTERS.json'
    m.write_control(path, {'counter': 1}, replace=True)
    calls = [0]
    def replace_after_second_fstat(fd):
        snapshot = fstat(fd)
        calls[0] += 1
        if calls[0] == 2:
            # A real writer performs its locked atomic replacement after the
            # reader captures the open inode's final stat, before path.stat.
            m.write_control(path, {'counter': 2}, replace=True)
        return snapshot
    with patch.object(m.os, 'fstat', replace_after_second_fstat):
        try:
            m.read(path, limit=65536)
        except ValueError as exc:
            assert str(exc) == 'Control changed while reading'
            results['unlocked_read'] = {'status': 'RACE_REPRODUCED', 'exception_type': type(exc).__name__, 'message': str(exc)}
        else:
            raise AssertionError('Expected immutable-reader check to reject replacement')
    assert m.read(path)[0] == {'counter': 2}

    m.write_control(path, {'counter': 3}, replace=True)
    waiting, acquired, done = threading.Event(), threading.Event(), threading.Event()
    failures = []
    original_lock = m.control_lock
    from contextlib import contextmanager
    @contextmanager
    def instrumented_writer_lock(directory):
        if threading.current_thread().name == 'fixture-writer':
            waiting.set()
        with original_lock(directory):
            if threading.current_thread().name == 'fixture-writer':
                acquired.set()
            yield
    def write_new():
        try:
            m.write_control(path, {'counter': 4}, replace=True)
        except BaseException as exc:
            failures.append(repr(exc))
        finally:
            done.set()
    thread = threading.Thread(target=write_new, name='fixture-writer')
    calls[0] = 0
    def compete_after_second_fstat(fd):
        snapshot = fstat(fd)
        calls[0] += 1
        if calls[0] == 2:
            thread.start()
            assert waiting.wait(1), 'Writer did not reach shared lock'
            assert not acquired.is_set() and not done.is_set()
        return snapshot
    with patch.object(m, 'control_lock', instrumented_writer_lock):
        with m.control_lock(path.parent):
            with patch.object(m.os, 'fstat', compete_after_second_fstat):
                value, _ = m.read(path, limit=65536)
            assert value == {'counter': 3}
            assert not acquired.is_set()
        thread.join(3)
    assert not thread.is_alive() and done.is_set() and not failures, failures
    assert m.read(path)[0] == {'counter': 4}
    results['shared_lock_read'] = {'status': 'PASS', 'strict_reader_unchanged': True, 'writer_blocked_until_read_complete': True, 'replacement_published_after_release': True}

actual = Path('/mnt/project/st-world-policy-20261016/audits/spatial_probe_actual_SP0_5000_v1/stages/dev_extract/RESULT.json')
raw = actual.read_bytes()
assert len(raw) < 131072
receipt = json.loads(raw)
results['actual_failure'] = {'path': str(actual), 'sha256': hashlib.sha256(raw).hexdigest(), 'status': receipt['status'], 'resources': receipt['resources'], 'actual_exception_class': 'UNKNOWN_NOT_RECORDED'}
results['scope'] = {'source_changed': False, 'actual_stage_changed': False, 'gpu_or_model_execution': False, 'dispatch_or_retry': False, 'uniquely_attributed_actual_root_cause': False}
(HERE / 'EVIDENCE.json').write_text(json.dumps(results, sort_keys=True, indent=2) + '\n')
print(json.dumps(results, sort_keys=True, indent=2))
