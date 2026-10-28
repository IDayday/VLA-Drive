import json
import pytest
from tools.qwen_world_postcot.queued_snapshot_owner import predecessor_ready


def test_in_progress_never_releases_queue(tmp_path):
    assert predecessor_ready(tmp_path)is False


def test_complete_requires_all_workers_exit(tmp_path):
    (tmp_path/'COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE',all_owned_workers_exited=False)))
    with pytest.raises(ValueError,match='workers'):predecessor_ready(tmp_path)


def test_verified_publication_releases_queue(tmp_path):
    (tmp_path/'COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE',all_owned_workers_exited=True)))
    (tmp_path/'PUBLICATION.json').write_text(json.dumps(dict(status='PUBLISHED',commit='abc',remote_SHA='abc')))
    assert predecessor_ready(tmp_path)is True


def test_failure_is_preserved_not_silently_bypassed(tmp_path):
    (tmp_path/'failure.json').write_text('{}')
    with pytest.raises(RuntimeError,match='failed'):predecessor_ready(tmp_path)


def test_remote_mismatch_does_not_release_queue(tmp_path):
    (tmp_path/'COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE',all_owned_workers_exited=True)))
    (tmp_path/'PUBLICATION.json').write_text(json.dumps(dict(status='PUBLISHED',commit='abc',remote_SHA='def')))
    with pytest.raises(ValueError,match='verified'):predecessor_ready(tmp_path)
