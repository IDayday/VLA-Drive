"""CPU-only resource-control negatives; not a real model calibration."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location('calibration_controller_fixture', Path(__file__).with_name('run_once.py'))
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def test_pin_and_tamper(tmp_path):
    import hashlib
    path = tmp_path / 'control.json'
    path.write_text('{"a": 1}')
    pin = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    assert c.pin_read(pin) == {'a': 1}
    path.write_text('{"a": 2}')
    with pytest.raises(ValueError, match='checksum'):
        c.pin_read(pin)


def test_symlink_rejected(tmp_path):
    path = tmp_path / 'target'; path.write_text('{}')
    link = tmp_path / 'link'; link.symlink_to(path)
    with pytest.raises(ValueError, match='Canonical'):
        c.pin_read({'path': str(link), 'sha256': 'bad'})


@pytest.mark.parametrize('amount', [2183 << 30, 2303.0 * (1 << 30), True])
def test_obligations_fail_closed(monkeypatch, tmp_path, amount):
    monkeypatch.setattr(c, 'ROOT', tmp_path)
    monkeypatch.setattr(c, 'pin_read', lambda _: dict(schema='st_checkpoint_storage_commitments_v1', status='CURRENT',
                       remaining_bytes=amount, free_floor_bytes=1 << 40, valid_until_unix=100))
    with pytest.raises(ValueError, match='commitments'):
        c.storage_guard({'checkpoint_receipt': {}, 'valid_until_unix': 90})


def test_space_and_lease(monkeypatch, tmp_path):
    monkeypatch.setattr(c, 'ROOT', tmp_path)
    cp = dict(schema='st_checkpoint_storage_commitments_v1', status='CURRENT',
              remaining_bytes=2303 << 30, free_floor_bytes=1 << 40, valid_until_unix=100)
    monkeypatch.setattr(c, 'pin_read', lambda _: cp)
    monkeypatch.setattr(c.os, 'statvfs', lambda _: SimpleNamespace(f_bavail=0, f_frsize=4096))
    with pytest.raises(ValueError, match='reserve'):
        c.storage_guard({'checkpoint_receipt': {}, 'valid_until_unix': 90})
    with pytest.raises(ValueError, match='commitments'):
        c.storage_guard({'checkpoint_receipt': {}, 'valid_until_unix': 101})


def test_no_overbudget_publication(monkeypatch, tmp_path):
    monkeypatch.setattr(c, 'ROOT', tmp_path)
    monkeypatch.setattr(c, 'output_bytes', lambda _: c.LIMITS['max_disk_bytes'])
    called = []
    with pytest.raises(ValueError, match='publication'):
        c.publish(lambda *a: called.append(a), tmp_path / 'RESULT.json', {'status': 'COMPLETE'})
    assert called == []


def test_no_tensor_or_checkpoint_budget():
    assert c.LIMITS['optimizer_updates'] == c.LIMITS['new_tensor_bytes'] == 0
    assert c.LIMITS['max_invocations'] == 1
    assert c.LIMITS['cpu_affinity'] == [2, 3]
