import json

import pytest

from tools.qwen_world_postcot.recover_admission_only import admission_refusal, alive


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
