import json
from pathlib import Path

from tools.d46_priority_rl.early_evaluate import policy_signature, pending_shards, reconcile


def test_same_parent_payload_and_scope_share_only_exact_reference():
    identity = dict(parent_model_sha256='IL46', scope=dict(trainable_names=['a']))
    sig = policy_signature(identity, {'optimizer.pt': 'same'})
    assert sig == policy_signature(dict(identity), {'optimizer.pt': 'same'})
    assert sig != policy_signature(identity, {'optimizer.pt': 'different'})
    assert sig != policy_signature(dict(identity, scope=dict(trainable_names=['b'])), {'optimizer.pt': 'same'})
    assert sig != policy_signature(dict(identity, parent_model_sha256='IL44'), {'optimizer.pt': 'same'})


def test_shards_interleave_policies():
    assert pending_shards(['D0', 'D5'], 3) == [('D0', 0), ('D5', 0), ('D0', 1), ('D5', 1), ('D0', 2), ('D5', 2)]


def make_case(tmp_path, code, shard_status=None):
    job = tmp_path / 'job'; job.mkdir()
    output = tmp_path / 'evaluation'; output.mkdir()
    if code is not None:
        (job / 'RESULT.json').write_text(json.dumps(dict(returncode=code, end_unix=1)))
    if shard_status:
        (output / 'rank0.json').write_text(json.dumps(dict(status=shard_status)))
    return dict(status='RUNNING', jobs={'0': [dict(job=str(job))]}, output=str(output))


def test_exit0_without_complete_shard_is_failure(tmp_path):
    case = make_case(tmp_path, 0, 'STOPPED_AT_SCENE_BOUNDARY')
    assert not reconcile(case, 1)
    assert case['status'] == 'FAILED_REQUIRES_REVIEW'


def test_complete_shard_requires_worker_exit0(tmp_path):
    case = make_case(tmp_path, None, 'COMPLETE')
    assert not reconcile(case, 1)
    assert case['active_shards'] == 1
    (Path(case['jobs']['0'][0]['job']) / 'RESULT.json').write_text('{"returncode":0}')
    assert reconcile(case, 1)


def test_prechild_resource_refusal_can_wait_but_failure_is_retained(tmp_path):
    case = make_case(tmp_path, 75)
    assert not reconcile(case, 1)
    assert case['status'] == 'RUNNING'
    (Path(case['jobs']['0'][0]['job']) / 'RESULT.json').write_text('{"returncode":1}')
    assert not reconcile(case, 1)
    assert case['status'] == 'FAILED_REQUIRES_REVIEW'
