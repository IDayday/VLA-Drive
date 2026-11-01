import json

from tools.qwen_world_safety_exploration import job_worker


def test_explicit_finite_shared_host_is_valid_only_before_guard(tmp_path, monkeypatch):
    record = tmp_path/'grant.json'
    grant = dict(permanent_hosts=['permanent'], temporary_hosts=['finite'],
                 unbounded_temporary_hosts=[], shared_training_hosts=['finite'],
                 starts_unix=100, expires_unix=300, checkpoint_guard_seconds=20,
                 record=str(record))
    record.write_text(json.dumps(grant))
    monkeypatch.setattr(job_worker.time, 'time', lambda: 200)
    assert job_worker.shared_training_host('finite', grant)
    assert not job_worker.shared_training_host('withdrawn', grant)
    monkeypatch.setattr(job_worker.time, 'time', lambda: 280)
    assert not job_worker.shared_training_host('finite', grant)


def test_finite_host_still_requires_matching_explicit_sharing_record(tmp_path, monkeypatch):
    record = tmp_path/'grant.json'
    grant = dict(permanent_hosts=[], temporary_hosts=['finite'],
                 unbounded_temporary_hosts=[], shared_training_hosts=['finite'],
                 starts_unix=100, expires_unix=300, checkpoint_guard_seconds=20,
                 record=str(record))
    record.write_text(json.dumps(dict(grant, shared_training_hosts=[])))
    monkeypatch.setattr(job_worker.time, 'time', lambda: 200)
    try:
        job_worker.shared_training_host('finite', grant)
    except PermissionError:
        pass
    else:
        raise AssertionError('Changed sharing authorization must be rejected')
