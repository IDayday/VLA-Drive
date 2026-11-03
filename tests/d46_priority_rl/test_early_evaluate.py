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


def test_early_batch_keeps_training_and_defers_duplicate_evaluation(monkeypatch):
    import os
    from tools.d46_priority_rl import owner
    from tools.d44_native_rl import campaign as native
    control = dict(continue_training_during_evaluation=True,
                   defer_registered_evaluations_during_early_batch=True)
    pid = os.getpid()
    reads = {'EXECUTION_POLICY.json': control, 'STATE.json': dict(complete=False),
             'CONTROLLER_LAUNCH.json': dict(pid=pid, start_ticks=int(Path('/proc/self/stat').read_text().split()[21]))}
    monkeypatch.setattr(owner, 'read', lambda p, default=None: reads.get(Path(p).name, default))
    monkeypatch.setattr(owner, 'qualification_pass', lambda run: True)
    monkeypatch.setattr(owner, 'inspect_latest', lambda run: (None, None))
    monkeypatch.setattr(owner, 'startup_hosts', lambda campaign: set())
    dispatched = []; evaluated = []
    monkeypatch.setattr(owner, 'dispatch', lambda run, *args: dispatched.append(run['arm']) or 'RUNNING')
    monkeypatch.setattr(owner, 'status', lambda campaign: None)
    monkeypatch.setattr(owner, 'atomic_json', lambda *args: None)
    monkeypatch.setattr(native, 'pending_evaluation', lambda *args: True)
    monkeypatch.setattr(native, 'evaluation_tick', lambda *args: evaluated.append(True))
    monkeypatch.setattr(native, 'selection_index', lambda *args: {})
    value = dict(runs={arm: dict(arm=arm, status='RUNNING', jobs=[]) for arm in owner.ORDER})
    owner.tick(value)
    assert dispatched == list(owner.ORDER)
    assert not evaluated
    assert value['evaluation_operations']['registered_dispatch_deferred']
    reads['STATE.json'] = dict(complete=True)
    owner.tick(value)
    assert evaluated == [True]
