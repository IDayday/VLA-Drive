import json

import pytest

from tools.qwen_world_postcot.recover_admission_only import admission_refusal, alive, missing_rank_spec


@pytest.mark.parametrize('status,error,child,expected', [
    ('FAILED', 'fcntl.flock(lease\nBlockingIOError: [Errno 11]', False, True),
    ('FAILED', 'fcntl.flock(lease\nBlockingIOError: [Errno 11]', True, False),
    ('FAILED', 'CUDA out of memory', False, False),
    ('PROCESS_COMPLETE', 'fcntl.flock(lease\nBlockingIOError: [Errno 11]', False, False),
    ('FAILED', 'unrelated BlockingIOError: [Errno 11]', False, False),
])
def test_only_pre_child_lease_refusal(tmp_path, status, error, child, expected):
    (tmp_path/'result.json').write_text(json.dumps(dict(status=status, error=error)))
    if child:
        (tmp_path/'process.json').write_text('{}')
    assert admission_refusal(tmp_path) is expected


def test_missing_process_is_not_alive():
    assert alive(dict(pid=2147483647, start_ticks=1)) is False


def test_never_dispatched_rank_changes_only_rank(tmp_path):
    first = tmp_path/'jobs/rank1'; first.mkdir(parents=True)
    original = dict(command=['python','-m','tools.qwen_world_postcot.evaluate','--rank','1','--world','8','--seed','2101'], source='immutable')
    (first/'spec.json').write_text(json.dumps(original))
    spec, reason = missing_rank_spec(tmp_path, 6)
    assert reason == 'never_dispatched_rank6'
    index = spec['command'].index('--rank')+1
    assert spec['command'][index] == '6'
    spec['command'][index] = '1'
    assert spec == original


def test_incomplete_receipt_cannot_be_recovered_as_unstarted(tmp_path):
    (tmp_path/'jobs/rank3').mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        missing_rank_spec(tmp_path, 3)


def test_rank_zero_requires_separate_review(tmp_path):
    with pytest.raises(ValueError):
        missing_rank_spec(tmp_path, 0)
