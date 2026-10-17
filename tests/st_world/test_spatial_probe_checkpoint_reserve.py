"""CPU-only temporary storage failures; no real cache, checkpoint or GPU."""
import hashlib
import json
from types import SimpleNamespace

import pytest

from tools.st_world_policy import spatial_probe as p, cache_storage_budget as b
from test_spatial_w_storage import fixture
from test_metadata_storage_guard import write


def commitment(tmp_path, *, end=2000, remaining=123456):
    path = tmp_path / 'space.json'
    digest = write(path, dict(schema='st_checkpoint_storage_commitments_v1', status='CURRENT',
        checked_at_unix=900, valid_until_unix=end, remaining_bytes=remaining))
    return dict(path=str(path), sha256=digest)


def test_receipt_required_for_typed_w_and_legacy_policy_unchanged():
    with pytest.raises(ValueError, match='requires pinned'):
        p.extraction_free_reserve(SimpleNamespace(core={'frozen_w_package': {}}), None, 900)
    assert p.extraction_free_reserve(SimpleNamespace(core={}), None, 900) == 0


@pytest.mark.parametrize('end', [999, 1500])
def test_reject_expired_or_expiring_during_extraction(tmp_path, monkeypatch, end):
    monkeypatch.setattr(p.time, 'time', lambda: 1000)
    with pytest.raises(ValueError, match='expired'):
        p.extraction_free_reserve(SimpleNamespace(core={'free_reserve_bytes': 100}),
                                  commitment(tmp_path, end=end), 900)


def test_pin_and_floor_are_both_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(p.time, 'time', lambda: 1000)
    ref = commitment(tmp_path)
    budget = SimpleNamespace(core={'free_reserve_bytes': 100})
    assert p.extraction_free_reserve(budget, ref, 900) == 123556
    with pytest.raises(ValueError):
        p.extraction_free_reserve(budget, {**ref, 'sha256': '0'*64}, 900)


def test_every_write_keeps_checkpoint_floor_after_successful_admission(tmp_path, monkeypatch):
    from tools.st_world_policy import metadata_storage as m
    f = fixture(tmp_path, cells=(('S-P0', 'train'),))
    m.migrate_policy(**f.kwargs)
    s = f.signatures[0]
    budget = b.StorageBudget(f.new_path)
    pin = dict(path=str(f.m0.checkpoint_path), sha256=f.m0.checkpoint_sha)
    floor = p.extraction_free_reserve(budget, pin, s.binding['max_seconds'])
    reservation = budget.reserve(s.root, s.signature['identity'],
        **{k:s.claim[k] for k in ('output_bytes','staging_bytes','metadata_bytes')}, free_reserve_bytes=floor)
    assert reservation.free_reserve_bytes == budget.core['free_reserve_bytes'] + (1 << 20)
    before = budget.state_path.read_bytes()
    # Plenty for the old bare floor, but not the separately preserved checkpoint
    # obligation. The very next write must fail before its body is entered.
    monkeypatch.setattr(b.shutil, 'disk_usage', lambda _:SimpleNamespace(free=floor-1))
    with pytest.raises(b.StorageBudgetError, match='free-space reserve'):
        with reservation.writing(1):
            raise AssertionError('write body must not run')
    assert budget.state_path.read_bytes() == before


def test_missing_commitment_rejected_before_native_or_observation_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(p, 'validate_binding', lambda _:None)
    monkeypatch.setattr(p, 'StorageBudget', lambda _:SimpleNamespace(core={'frozen_w_package':{}}))
    monkeypatch.setattr(p, 'cache_plan', lambda _:pytest.fail('No expensive plan without space pin'))
    with pytest.raises(ValueError, match='requires pinned'):
        p.extract({'max_seconds':900}, tmp_path/'output', tmp_path/'policy.json')


def test_cli_rejects_half_pin_without_extract(tmp_path, monkeypatch, capsys):
    binding = tmp_path/'binding.json'; binding.write_text('{}')
    monkeypatch.setattr(p, 'extract', lambda *a,**k:pytest.fail('Incomplete pin cannot enter extractor'))
    monkeypatch.setattr('sys.argv', ['probe','extract','--binding',str(binding),'--output',str(tmp_path/'out'),
        '--storage-policy',str(tmp_path/'policy.json'),'--checkpoint-commitments',str(tmp_path/'space.json')])
    assert p.main() == 2
    assert 'supplied together' in json.loads(capsys.readouterr().out)['error']
