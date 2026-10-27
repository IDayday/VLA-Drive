import json

import pytest

from tools.qwen_world_postcot.resume_il_owner import reconcile_journal


def row(update):
    return json.dumps(dict(updates=update, cursor=32 * update, seconds=2.)).encode() + b'\n'


def test_abandoned_suffix_preserved_and_prefix_exact(tmp_path):
    journal, archive = tmp_path / 'metrics', tmp_path / 'aborted/metrics'
    original = row(1) + row(2) + row(3) + b'{"updates":4'
    journal.write_bytes(original)
    receipt = reconcile_journal(journal, 2, archive)
    assert archive.read_bytes() == original
    assert journal.read_bytes() == row(1) + row(2)
    assert receipt['aborted_updates'] == [3]
    assert receipt['aborted_observed_worker_seconds'] == 2.
    assert receipt['corrupt_aborted_rows']
    journal.open('ab').write(row(3))
    assert archive.read_bytes() == original


@pytest.mark.parametrize('raw', [row(1) + row(1), row(2), row(1) + b'broken\n' + row(2)])
def test_bad_saved_prefix_never_replaces_history(tmp_path, raw):
    journal, archive = tmp_path / 'metrics', tmp_path / 'aborted'
    journal.write_bytes(raw)
    with pytest.raises(ValueError):
        reconcile_journal(journal, 2, archive)
    assert journal.read_bytes() == raw
    assert not archive.exists()


def test_repeated_reconciliation_requires_review(tmp_path):
    journal, archive = tmp_path / 'metrics', tmp_path / 'aborted'
    journal.write_bytes(row(1)); archive.write_bytes(b'prior attempt')
    with pytest.raises(ValueError):
        reconcile_journal(journal, 1, archive)
    assert archive.read_bytes() == b'prior attempt'


def test_corruption_within_aborted_suffix_is_preserved(tmp_path):
    journal, archive = tmp_path / 'metrics', tmp_path / 'aborted'
    raw = row(1) + b'\x00\x00broken\n' + row(3)
    journal.write_bytes(raw)
    receipt = reconcile_journal(journal, 1, archive)
    assert archive.read_bytes() == raw
    assert journal.read_bytes() == row(1)
    assert receipt['corrupt_aborted_rows'] and receipt['aborted_updates'] == [3]
    repeated = reconcile_journal(journal, 1, archive)
    assert repeated['previous_reconciliation_reused']
    assert repeated['original_sha256'] == receipt['original_sha256']
