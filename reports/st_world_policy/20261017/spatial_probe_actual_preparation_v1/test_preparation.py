"""Metadata/owned tiny-child tests only. No native/image/model/cache/score calls."""
import copy
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prepare_bindings as b
import supervise_once as s
import observed_cli as o


def preparation():
    return dict(checkpoint_tag=b.TAG, job_root=str(b.JOB), fit_output=str(b.JOB / 'fit'),
                caches={key: {'root': str(root)} for key, root in b.ROOTS.items()},
                runtime={'path': '/unused/metadata_fixture', 'source_sha': 'a' * 40})


def admission(stage='dev_plan'):
    return dict(schema='st_SP0_probe_stage_admission_v1', status='APPROVED', reviewer='/independent-test-fixture',
                stage=stage, hostname=socket.gethostname(), valid_until_unix=time.time() + 2000,
                runtime=preparation()['runtime'], cpu_affinity=sorted(os.sched_getaffinity(0))[:2],
                helpers={name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in s.HELPERS},
                gpu=None, storage_policy=None)


@pytest.mark.parametrize('change', [
    {'status': 'PENDING'}, {'reviewer': '/root/st_evaluation'}, {'reviewer': None},
    {'stage': 'train_plan'}, {'hostname': 'foreign-host'}, {'valid_until_unix': 0}, {'valid_until_unix': float('nan')},
    {'cpu_affinity': []}, {'cpu_affinity': [1, 1]}, {'cpu_affinity': [1, 2, 3]},
    {'gpu': {'index': 0, 'uuid': 'GPU-foreign'}}, {'storage_policy': {}}, {'helpers': {}},
    {'runtime': {'path': '/different', 'source_sha': 'b' * 40}}])
def test_reject_bad_admission_before_child(change):
    a = admission(); a.update(change)
    with pytest.raises((ValueError, KeyError)):
        s.validate(a, preparation(), 'dev_plan', time.time())


@pytest.mark.parametrize('field,value', [('checkpoint_tag', 'latest'), ('job_root', '/tmp/foreign'),
                                       ('fit_output', '/tmp/foreign')])
def test_fixed_endpoint_and_roots(field, value):
    p = preparation(); p[field] = value
    with pytest.raises(ValueError):
        s.validate(admission(), p, 'dev_plan', time.time())


def test_limits_are_fixed_and_cpu_plan_accepted():
    assert s.validate(admission(), preparation(), 'dev_plan', time.time()) == (600, 64 << 30)
    assert s.LIMITS == {'plan': (600, 64 << 30), 'extract': (960, 64 << 30), 'fit': (1260, 2 << 30)}
    assert s.STAGES == ('dev_plan', 'dev_extract', 'train_plan', 'train_extract', 'fit')


@pytest.mark.parametrize('bad', ['old_ledger', 'cap', 'no_migration', 'expired_reserve', 'old_policy'])
def test_extract_rejects_unreviewed_storage_and_short_space_window(tmp_path, bad):
    a = admission('dev_extract'); a['gpu'] = {'index': 0, 'uuid': 'GPU-test'}
    policy = dict(state_dir=s.LEDGER, cache_cap_bytes=768 << 30,
                  frozen_w_package={'maximum_bytes': 1 << 30, 'members': {}})
    if bad == 'old_ledger': policy['state_dir'] = '/tmp/new_ledger'
    if bad == 'cap': policy['cache_cap_bytes'] += 1
    a['storage_policy'] = b.write_once(tmp_path / 'policy.json', policy)
    migration = dict(schema='st_metadata_storage_migration_v1', status='MIGRATED',
                     new_policy_sha256=a['storage_policy']['sha256'])
    if bad == 'no_migration': migration['status'] = 'PREPARED'
    if bad == 'old_policy': migration['new_policy_sha256'] = '0' * 64
    a['storage_migration'] = b.write_once(tmp_path / 'migration.json', migration)
    reserve = dict(schema='st_checkpoint_storage_commitments_v1', status='CURRENT',
                   checked_at_unix=time.time() - 1, valid_until_unix=time.time() + (900 if bad == 'expired_reserve' else 2000),
                   remaining_bytes=1 << 30)
    a['checkpoint_commitments'] = b.write_once(tmp_path / 'reserve.json', reserve)
    a['typed_storage_approved'] = True
    with pytest.raises(ValueError): s.validate(a, preparation(), 'dev_extract', time.time())


def test_pin_oversize_changed_and_immutable(tmp_path):
    path = tmp_path / 'control.json'; pin = b.write_once(path, {'a': 1})
    assert b.read(path, pin['sha256'])[0] == {'a': 1}
    with pytest.raises(ValueError): b.read(path, '0' * 64)
    with pytest.raises(ValueError): b.read(path, limit=1)
    with pytest.raises(ValueError): b.write_once(path, {'a': 2})
    other = tmp_path / 'link.json'; other.symlink_to(path)
    with pytest.raises(ValueError): b.read(other)
    with pytest.raises(ValueError): b.write_once(other, {'a': 1})


def test_serialized_json_limit_counts_escape_expansion(tmp_path):
    with pytest.raises(ValueError, match='Serialized'):
        b.write_once(tmp_path / 'huge.json', {'text': '\x00' * 200000})
    assert not (tmp_path / 'huge.json').exists()


def test_staging_budget_preserves_diagnostics_and_final_receipt(tmp_path):
    # Retained bytes consume the budget; a staged JSON must not be admitted from
    # its unescaped Python character count or only its eventual target size.
    old = tmp_path / 'CHILD.log'; old.write_bytes(b'x' * (4 << 20))
    with pytest.raises(ValueError, match='including staging'):
        b.write_once(tmp_path / 'SIGNATURE.json', {'value': 'x' * (600 << 10)})
    assert old.stat().st_size == 4 << 20
    assert not list(tmp_path.glob('.*.tmp'))
    b.write_once(tmp_path / 'RESULT.json', {'status': 'FAILED_NO_AUTO_RETRY', 'error': 'bounded fixture'})
    assert b.observed_bytes(tmp_path) <= b.DIRECTORY_LIMIT


def test_child_metadata_failure_terminates_only_new_child(tmp_path, monkeypatch):
    children = []
    def start(*args, **kwargs):
        child = subprocess.Popen(*args, **kwargs); children.append(child); return child
    def fail(path, value): raise ValueError('fixture metadata write failure')
    monkeypatch.setattr(s, 'write_once', fail)
    with pytest.raises(ValueError, match='fixture'):
        s.supervise([sys.executable, '-c', 'import time; time.sleep(10)'], cwd=tmp_path, env=dict(os.environ),
                    cpus=sorted(os.sched_getaffinity(0))[:2], descriptors=(), attempt=tmp_path,
                    wall=5, rss=1 << 30, rss_reader=lambda pid: 1024, popen=start)
    assert len(children) == 1 and children[0].poll() == -signal.SIGTERM


def test_global_lock_and_inherited_descriptor(tmp_path):
    path = tmp_path / 'one-loader.lock'
    with ExitStack() as stack:
        lock = s.acquire(path, stack)
        with ExitStack() as other:
            with pytest.raises(BlockingIOError): s.acquire(path, other)
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(.3)'], pass_fds=(lock.fileno(),))
    with ExitStack() as other:
        with pytest.raises(BlockingIOError): s.acquire(path, other)
    child.wait(timeout=5)
    with ExitStack() as other: s.acquire(path, other)


def test_owned_group_only_and_wall_bound(tmp_path):
    outsider = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'])
    attempt = tmp_path / 'attempt'; attempt.mkdir()
    try:
        with pytest.raises(RuntimeError, match='foreign'):
            s.own_group_signal(outsider, signal.SIGTERM)
        result = s.supervise([sys.executable, '-c', 'import time; time.sleep(10)'],
            cwd=tmp_path, env=dict(os.environ), cpus=sorted(os.sched_getaffinity(0))[:2],
            descriptors=(), attempt=attempt, wall=.15, rss=1 << 30, rss_reader=lambda pid: 1024)
        assert result['limit_reason'] == 'WALL_LIMIT'
        assert result['exit_code'] == -signal.SIGTERM and outsider.poll() is None
        assert result['wall_seconds'] < 4
    finally:
        outsider.terminate(); outsider.wait(timeout=5)  # only the test-created outsider


@pytest.mark.parametrize('kind', ['rss', 'log'])
def test_limit_stops_owned_child_and_preserves_bounded_log(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(s, 'LOG_LIMIT', 4096)
    code = ('import time; time.sleep(10)' if kind == 'rss' else
            'import os,time; os.write(1,b"x"*10000); time.sleep(10)')
    result = s.supervise([sys.executable, '-c', code], cwd=tmp_path, env=dict(os.environ),
        cpus=sorted(os.sched_getaffinity(0))[:2], descriptors=(), attempt=tmp_path,
        wall=5, rss=100, rss_reader=lambda pid: 101 if kind == 'rss' else 0)
    assert result['limit_reason'] == ('RSS_LIMIT' if kind == 'rss' else 'LOG_LIMIT')
    assert (tmp_path / 'CHILD.log').stat().st_size <= 4096


def test_hash_observer_does_not_rehash_or_change_value(tmp_path):
    class CUDA:
        @staticmethod
        def is_initialized(): return False
    class Torch: cuda = CUDA()
    path = tmp_path / 'tiny'; path.write_bytes(b'fixture')
    calls = []
    def actual(p):
        calls.append(str(p)); return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    counter = o.Counters(tmp_path / 'COUNTERS.json', Torch(), 100, o.proc_io())
    result = counter.hash_wrapper(actual)(path)
    assert result == hashlib.sha256(b'fixture').hexdigest() and len(calls) == 1
    assert counter.data['hash_calls_completed'] == 1 and counter.data['hash_logical_bytes_completed'] == 7
    counter.read_cap = 10
    with pytest.raises(RuntimeError, match='exhausted'): counter.hash_wrapper(actual)(path)
    assert len(calls) == 1  # rejected before the second real hash


def test_metadata_controls_remain_unexecuted():
    folder = HERE / 'controls_cf580501_metadata_only'
    prep, _ = b.read(folder / 'PREPARATION.json')
    assert prep['dispatch'] is False and prep['payload_read'] is False and prep['live_policy_written'] is False
    assert prep['native_bytes'] == 37366035746
    for split, count in (('dev', 16), ('train', 64)):
        binding = b.check_seal(b.read(prep['controls'][split]['binding']['path'], prep['controls'][split]['binding']['sha256'])[0])
        signature = b.check_seal(b.read(prep['controls'][split]['expected_signature']['path'], prep['controls'][split]['expected_signature']['sha256'])[0])
        assert len(signature['rows']) == count and signature['identity'] == prep['caches'][split]['identity']
        assert binding['checkpoint_tag'] == b.TAG and binding['device'] == 'cuda:0' and binding['max_seconds'] == 900
        assert signature['checkpoint']['training_source_sha'] == b.TRAIN_SOURCE
        assert binding['source_sha'] == 'cf5805016b5e23bdab9541c77b562ed5fda83dc3'
    fit = b.check_seal(b.read(prep['fit_registration']['path'], prep['fit_registration']['sha256'])[0])
    assert fit['device'] == 'cpu' and fit['max_seconds'] == 1200 and fit['max_output_bytes'] == 64 << 20
