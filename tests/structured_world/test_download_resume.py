from types import SimpleNamespace
import pytest
from tools.structured_world.download_nuscenes import download_resumable


def test_each_retry_uses_new_range_and_keeps_received_bytes(tmp_path):
    partial = tmp_path/'official.tgz.part'
    partial.write_bytes(b'12')
    offsets = []
    def interrupted_request(command, check):
        offsets.append(partial.stat().st_size)
        assert command[command.index('--retry')+1] == '0'
        assert command[command.index('--continue-at')+1] == '-'
        assert command[command.index('--header')+1] == 'If-Range: "pinned-etag"'
        with partial.open('ab') as stream: stream.write(b'34' if len(offsets) == 1 else b'56')
        return SimpleNamespace(returncode=56 if len(offsets) == 1 else 0)
    download_resumable({'bytes': 6, 'url': 'https://example.invalid/fixed', 'etag': '"pinned-etag"'},
                       partial, run=interrupted_request, sleep=lambda _: None)
    assert offsets == [2, 4] and partial.read_bytes() == b'123456'


def test_resumer_rejects_a_request_that_discards_previous_data(tmp_path):
    partial = tmp_path/'official.tgz.part'
    partial.write_bytes(b'1234')
    def incorrect_request(command, check):
        partial.write_bytes(b'1')
        return SimpleNamespace(returncode=56)
    with pytest.raises(RuntimeError, match='discarded received bytes'):
        download_resumable({'bytes': 6, 'url': 'https://example.invalid/fixed'}, partial,
                           run=incorrect_request, sleep=lambda _: None)
