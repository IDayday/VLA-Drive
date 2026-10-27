import json

import pytest

from tools.qwen_world_postcot.resume_il_owner import reconcile_journal, validate_adoption, validate_resume_command


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


@pytest.mark.parametrize('change', [None, 'host', 'cwd', 'command', 'gpu_uuids'])
def test_live_worker_adoption_binds_source_command_and_devices(change):
    cfg = dict(host='authorized', source='/source', command=['train_il'], gpu_uuids=['gpu'] * 8)
    job = dict(status='RUNNING', role='training', name='formal_IL', host=cfg['host'],
               command=cfg['command'], uuids=cfg['gpu_uuids'])
    worker = dict(status='RUNNING', host=cfg['host'], cwd=cfg['source'], command=cfg['command'],
                  gpu_uuids=cfg['gpu_uuids'], gpus=list(range(8)), child_pid=123)
    if change:
        worker[change] = ['changed'] if change in ('command', 'gpu_uuids') else 'changed'
        with pytest.raises(ValueError):
            validate_adoption(cfg, job, worker)
    else:
        validate_adoption(cfg, job, worker)


def test_planned_clean_exit_continuation_replaces_only_prior_resume():
    previous = dict(cwd='/frozen', returncode=0,
                    command=['python', '-m', 'tools.qwen_world_postcot.train_il', '--output', 'same', '--resume', 'old'])
    command = previous['command'][:-2] + ['--resume', 'new']
    with pytest.raises(ValueError):
        validate_resume_command(previous, command, 'new', '/frozen')
    validate_resume_command(previous, command, 'new', '/frozen', planned=True)
    with pytest.raises(ValueError):
        validate_resume_command(previous, command + ['--qualification'], 'new', '/frozen', planned=True)
