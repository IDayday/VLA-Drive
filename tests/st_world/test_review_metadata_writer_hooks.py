"""Independent hook integration review; only synthetic temporary metadata."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from tools.st_world_policy import metadata_storage as storage
from tools.st_world_policy import prepare_future_metadata as future
from tools.st_world_policy import prepare_current_teacher as current
from starVLA.dataloader import st_world_metadata_index as index
from starVLA.dataloader.st_world_schema import identity_hash
from starVLA.dataloader.st_world_targets import atomic_json
from tests.st_world.test_metadata_writer_hooks import registered_cli, run_cli
from tests.st_world.test_metadata_storage_guard import update_registration, ledger
from tests.st_world.test_prepare_future_metadata import setup
from tests.st_world.test_ondemand_teacher import fixture as request_fixture


@pytest.mark.parametrize('kind', ['current_index', 'future_preparation'])
def test_real_subprocess_cli_completes_and_reuses_synthetic_only(tmp_path, kind):
    run = registered_cli(tmp_path, kind)
    command = [sys.executable, '-m', 'tools.st_world_policy.metadata_storage', 'run']
    for key, value in run.guard.items(): command += ['--'+key.replace('_', '-'), str(value)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    first = subprocess.run(command, cwd=Path.cwd(), env=env, text=True, capture_output=True, timeout=45)
    assert first.returncode == 0, first.stdout + first.stderr
    result = json.loads(first.stdout)
    assert result['status'] == 'GUARDED_METADATA_RETURN'
    assert result['writer_callbacks'] > 0 and result['writer_commit_callbacks'] >= 3
    complete = (run.root/'COMPLETE.json').read_bytes()
    second = subprocess.run(command, cwd=Path.cwd(), env=env, text=True, capture_output=True, timeout=45)
    assert second.returncode == 0, second.stdout + second.stderr
    assert json.loads(second.stdout)['status'] == 'GUARDED_METADATA_RETURN'
    assert (run.root/'COMPLETE.json').read_bytes() == complete
    assert storage._ACTIVE_GUARD is None


@pytest.mark.parametrize('kind', ['current_index', 'future_preparation'])
@pytest.mark.parametrize('fault', ['publication', 'rss', 'source_pin'])
def test_real_subprocess_rejects_before_complete(tmp_path, monkeypatch, kind, fault):
    # All registration/policy preparation is synthetic; no monkeypatch reaches
    # the subprocess being reviewed. Register the exact negative scope first.
    migrate = storage.migrate_policy
    with monkeypatch.context() as patch:
        patch.setattr(storage, 'migrate_policy', lambda **kwargs: None)
        run = registered_cli(tmp_path, kind)
    if fault == 'publication':
        run.reg['file_patterns'] = [pattern.replace('|COMPLETE', '')
            for pattern in run.reg['file_patterns'] if not pattern.startswith('COMPLETE')]
    elif fault == 'rss':
        run.reg['limits']['max_rss_bytes'] = index.MiB
    else:
        run.reg['source_pins'][0]['sha256'] = 'f'*64
    update_registration(run)
    migrate(**run.kwargs)
    command = [sys.executable, '-m', 'tools.st_world_policy.metadata_storage', 'run']
    for key, value in run.guard.items(): command += ['--'+key.replace('_', '-'), str(value)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    result = subprocess.run(command, cwd=Path.cwd(), env=env, text=True,
                            capture_output=True, timeout=45)
    assert result.returncode == 2, result.stdout + result.stderr
    message = json.loads(result.stdout)
    assert message['status'] == 'BLOCKED_STORAGE'
    assert not (run.root/'COMPLETE.json').exists()
    if fault == 'publication':
        assert 'unregistered metadata write destination' in message['error']
        assert not list(run.root.glob('COMPLETE.json.*'))
        assert (run.root/('metadata.sqlite' if kind == 'current_index' else 'SHARD_RECEIPTS.json')).exists()
    elif fault == 'rss':
        assert 'RSS' in message['error']
    else:
        assert not (run.root/'METADATA_ADMISSION.json').exists()


@pytest.mark.parametrize('kind', ['current_index', 'future_preparation'])
def test_bad_source_pin_prevents_parser_builder_and_admission(tmp_path, monkeypatch, capsys, kind):
    run = registered_cli(tmp_path, kind)
    before = ledger(run)
    run.reg['source_pins'][0]['sha256'] = 'f'*64
    update_registration(run)
    module = future if kind == 'future_preparation' else __import__(
        'tools.st_world_policy.index_teacher_metadata', fromlist=['run'])
    def forbidden(*args, **kwargs): raise AssertionError('Unpinned parser/builder ran')
    monkeypatch.setattr(module, 'parser', forbidden)
    monkeypatch.setattr(module, 'build' if kind == 'future_preparation' else 'run', forbidden)
    with pytest.raises(SystemExit) as error: run_cli(run, monkeypatch, capsys)
    assert error.value.code == 2
    assert not (run.root/'METADATA_ADMISSION.json').exists()
    assert not (run.root/'COMPLETE.json').exists()
    assert ledger(run) == before


@pytest.mark.parametrize('kind,target', [
    ('current_index', 'metadata.sqlite'), ('current_index', 'COMPLETE.json'),
    ('future_preparation', 'RAW_LOG_PROVENANCE.json'),
    ('future_preparation', 'TEACHER_REQUESTS.json'),
    ('future_preparation', 'SHARD_RECEIPTS.json'),
    ('future_preparation', 'COMPLETE.json'),
])
def test_publication_rejection_leaves_no_target_or_complete(tmp_path, monkeypatch, capsys, kind, target):
    run = registered_cli(tmp_path, kind)
    original = storage.MetadataWriteContext.check
    reached = []
    def reject(self, *, before_commit=False, path=None):
        if before_commit and path is not None and Path(path).name == target:
            reached.append(str(path)); assert not Path(path).exists()
            self.failed = 'independent publication fault'
            raise storage.b.StorageBudgetError(self.failed)
        return original(self, before_commit=before_commit, path=path)
    monkeypatch.setattr(storage.MetadataWriteContext, 'check', reject)
    with pytest.raises(SystemExit) as error: run_cli(run, monkeypatch, capsys)
    assert error.value.code == 2 and reached
    assert not (run.root/target).exists() and not (run.root/'COMPLETE.json').exists()
    assert str(run.root) in ledger(run)['claims']
    assert storage._ACTIVE_GUARD is None


def test_census_throttle_does_not_throttle_creation_publication(tmp_path, monkeypatch):
    now = [100.]; calls = []
    monkeypatch.setattr(index.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(index, '_LAST_STORAGE_CHECK', 0.)
    monkeypatch.setattr(index, 'check_active_metadata_storage', lambda **kw: calls.append(kw))
    index._storage_check()
    for _ in range(100): index._storage_check()
    assert len(calls) == 1
    index._storage_check(path=tmp_path/'new')
    index._storage_check(before_commit=True, path=tmp_path/'COMPLETE.json')
    index._storage_check(before_commit=True, path=tmp_path/'COMPLETE.json')
    assert len(calls) == 4
    now[0] += .99; index._storage_check(); assert len(calls) == 4
    now[0] += .02; index._storage_check(); assert len(calls) == 5


def test_multiple_threads_share_one_guard_without_reentering_ledger_flock(tmp_path, monkeypatch):
    run = registered_cli(tmp_path, 'current_index')
    with storage.metadata_writing(**run.guard) as ctx:
        def forbidden(*args, **kwargs): raise AssertionError('Writer callback reentered CLAIMS.lock')
        monkeypatch.setattr(storage.b.StorageBudget, '_locked', forbidden)
        gate = threading.Barrier(3)
        def worker(i):
            gate.wait(timeout=5)
            for _ in range(4): index._storage_check(before_commit=True, path=run.root/'COMPLETE.json')
            return i
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(worker, i) for i in range(2)]
            gate.wait(timeout=5)
            assert [x.result(timeout=10) for x in futures] == [0, 1]
        assert ctx.writer_commit_callbacks == 8
        assert ctx.failed is None
    assert storage._ACTIVE_GUARD is None


def test_optional_source_callback_stops_before_any_metadata_open(tmp_path, monkeypatch):
    args, _, _, _ = setup(tmp_path)
    opened = []; original = Path.open
    def record(path, *a, **kw): opened.append(str(path)); return original(path, *a, **kw)
    def stop(): raise RuntimeError('independent source entry fault')
    monkeypatch.setattr(Path, 'open', record)
    with pytest.raises(RuntimeError, match='source entry'):
        current.source_population(args.student_root, args.partition, args.split, resource_check=stop)
    assert opened == []


def test_optional_source_large_json_is_chunked_and_returns_identical_binding(tmp_path, monkeypatch):
    args, _, _, _ = setup(tmp_path)
    identity = Path(args.student_root)/'identity.json'
    body = json.loads(identity.read_text()); body['fixture_padding'] = 'x'*(2*index.MiB+17)
    body.pop('identity'); body['identity'] = identity_hash(body); atomic_json(identity, body)
    expected = current.source_population(args.student_root, args.partition, args.split)
    checks = []; sizes = []; original = Path.open
    class Reader:
        def __init__(self, stream): self.stream = stream
        def __enter__(self): return self
        def __exit__(self, *args): self.stream.close()
        def read(self, n):
            assert n == index.MiB; sizes.append(n)
            return self.stream.read(n)
    def intercept(path, mode='r', *a, **kw):
        stream = original(path, mode, *a, **kw)
        return Reader(stream) if path == identity and mode == 'rb' else stream
    monkeypatch.setattr(Path, 'open', intercept)
    actual = current.source_population(args.student_root, args.partition, args.split,
                                       resource_check=lambda: checks.append(True))
    assert actual == expected and len(sizes) >= 4 and len(checks) >= len(sizes)


@pytest.mark.parametrize('fail', [False, True])
def test_sqlite_progress_callback_runs_and_failure_cannot_publish(tmp_path, monkeypatch, fail):
    contract, manifest = request_fixture(tmp_path)
    one = manifest['scenes'][0]
    manifest['scenes'] = [dict(deepcopy(one), token=f's{i}', log=f'l{i}') for i in range(128)]
    manifest.pop('identity'); manifest['identity'] = identity_hash(manifest)
    request = tmp_path/'requests.json'; atomic_json(request, manifest)
    original_connect = index.sqlite3.connect; intervals = []; callback_calls = []
    class Connection:
        def __init__(self, connection): self.connection = connection
        def __getattr__(self, name): return getattr(self.connection, name)
        def set_progress_handler(self, callback, interval):
            intervals.append(interval)
            def wrapped():
                callback_calls.append(True)
                if fail: raise RuntimeError('independent VM callback failure')
                return callback()
            self.connection.set_progress_handler(wrapped, interval)
    monkeypatch.setattr(index.sqlite3, 'connect', lambda *a, **k: Connection(original_connect(*a, **k)))
    output = tmp_path/'index'
    if fail:
        with pytest.raises(index.sqlite3.OperationalError):
            index.build_index(output, requests=request, expected_request_identity=manifest['identity'],
                              contract=contract, max_scenes=128)
        assert not (output/'COMPLETE.json').exists()
    else:
        result = index.build_index(output, requests=request, expected_request_identity=manifest['identity'],
                                   contract=contract, max_scenes=128)
        assert result['scene_count'] == 128 and (output/'COMPLETE.json').exists()
    assert intervals == [1000] and callback_calls
