from types import SimpleNamespace
import time
import pytest
from iqe.io import atomic_json, file_hash
from iqe.specialist_study.external_worker import ReceiptProcess


def setup_worker(tmp_path):
    config = SimpleNamespace(root=tmp_path, identity='fixed_science')
    atomic_json(tmp_path/'BASE_BINDING.json', {'base': 'immutable'})
    path = tmp_path/'remote.json'
    value = {'config_identity': config.identity, 'Base_binding_hash': file_hash(tmp_path/'BASE_BINDING.json'),
             'cell': 'S01', 'host': 'allowed_remote', 'pid': 123, 'start_ticks': '456',
             'updated_unix': time.time(), 'status': 'RUNNING'}
    atomic_json(path, value)
    return config, path, value


def test_remote_receipt_observes_completion_without_relaunch(tmp_path):
    config, path, value = setup_worker(tmp_path)
    process = ReceiptProcess(path, config, 'S01')
    assert process.poll() is None
    atomic_json(path, {**value, 'status': 'COMPLETE'})
    with pytest.raises(Exception, match='lacks complete'):
        process.poll()
    atomic_json(tmp_path/'scorers/S01/result.json', {'status': 'COMPLETE'})
    assert process.poll() == 0


def test_remote_identity_heartbeat_and_failure_are_not_relaunched(tmp_path):
    config, path, value = setup_worker(tmp_path)
    with pytest.raises(Exception, match='scientific identity'):
        ReceiptProcess(path, config, 'S11')
    process = ReceiptProcess(path, config, 'S01')
    atomic_json(path, {**value, 'start_ticks': 'reused'})
    with pytest.raises(Exception, match='process identity'):
        process.poll()
    atomic_json(path, {**value, 'updated_unix': time.time()-181})
    with pytest.raises(Exception, match='heartbeat stale'):
        process.poll()
    atomic_json(path, {**value, 'status': 'FAILED'})
    assert process.poll() == 1
