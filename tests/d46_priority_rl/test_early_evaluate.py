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
    def dispatch(run, *args, **kwargs):
        assert kwargs.get('capacity_host') is None
        dispatched.append(run['arm'])
        return 'RUNNING'
    monkeypatch.setattr(owner, 'dispatch', dispatch)
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
    dispatched.clear()
    control['execution_holds'] = {'D0': 'Registered same-run continuation'}
    owner.tick(value)
    assert dispatched == [arm for arm in owner.ORDER if arm != 'D0']
    assert value['runs']['D0']['status'] == 'SAVED_AND_HELD_FOR_REGISTERED_STAGE'


def test_native_evaluator_sharing_requires_exact_sealed_d46_parent(tmp_path, monkeypatch):
    from tools.qwen_world_compare import job_worker as worker
    from starVLA.rl.qwen_world_compare.contracts import file_sha, fingerprint
    monkeypatch.setattr(worker, 'D46_ROOT', tmp_path)
    registration = tmp_path / 'PARENT_REGISTRATION_v2.json'; registration.write_text('{}')
    digest = file_sha(registration)
    monkeypatch.setattr(worker, 'D46_CONTRACT_SHA', digest)
    cp = tmp_path / 'runs/D0/checkpoints/update001400_inner0'; cp.mkdir(parents=True)
    identity = dict(schema='d44_native_rl_v1', architecture_id='D',
                    parent_model_sha256=worker.D46_PARENT_SHA, data=dict(NAVTEST_overlap=0),
                    recipe=dict(parent='D_IL46K', arm='D0', parent_checkpoint=str(tmp_path / 'IL46_PARENT'),
                                parent_registration=dict(path=str(registration), file_sha256=digest),
                                training_uses_navtest=False))
    identity['sha256'] = fingerprint(identity)
    (cp / 'identity.json').write_text(json.dumps(identity))
    (cp / 'files.json').write_text('{}')
    (cp / 'COMPLETE').write_text(json.dumps(dict(files_sha256=file_sha(cp / 'files.json'))))
    command = ['python', '-m', 'tools.d44_native_rl.evaluate', '--checkpoint', str(cp),
               '--output', str(tmp_path / 'evaluations/EARLY/evaluation')]
    assert worker.known_owner_args(command)
    assert not worker.known_owner_args(command[:-1] + [str(tmp_path / 'foreign')])
    (cp / 'files.json').write_text('{"changed":true}')
    assert not worker.known_owner_args(command)
    (cp / 'files.json').write_text('{}')
    registration.write_text('{"changed":true}')
    assert not worker.known_owner_args(command)


def test_world16_sharing_requires_own_formal_world8_ready_ancestry(tmp_path, monkeypatch):
    import copy
    from tools.qwen_world_compare import job_worker as worker
    from starVLA.rl.qwen_world_compare.contracts import file_sha, fingerprint
    monkeypatch.setattr(worker, 'D46_ROOT', tmp_path)
    registration = tmp_path / 'PARENT_REGISTRATION_v2.json'; registration.write_text('{}')
    digest = file_sha(registration); monkeypatch.setattr(worker, 'D46_CONTRACT_SHA', digest)
    source = tmp_path / 'runs/D46_D0/checkpoints/update001400_inner0'; source.mkdir(parents=True)
    old = dict(schema='d44_native_rl_v1', architecture_id='D', world=8, qualification=False,
               parent_model_sha256=worker.D46_PARENT_SHA, scope=dict(trainable_names=['a']),
               data=dict(NAVTEST_overlap=0), recipe=dict(parent='D_IL46K', arm='D0',
               parent_checkpoint=str(tmp_path / 'IL46_PARENT'), training_uses_navtest=False,
               parent_registration=dict(path=str(registration), file_sha256=digest)))
    old['sha256'] = fingerprint(old)
    (source / 'identity.json').write_text(json.dumps(old))
    (source / 'files.json').write_text('{"optimizer.pt":"original-Adam"}')
    source_files = file_sha(source / 'files.json')
    (source / 'COMPLETE').write_text(json.dumps(dict(world=8, phase='READY', next_inner=0,
                                                  updates=1400, files_sha256=source_files)))
    tx = dict(schema='d46_exact_native_READY_world8_to16_v1', source_world=8, target_world=16,
              same_algorithm=True, old_owner_may_adopt=False, virtual_sampler_lanes=8,
              global_scenes=64, group=16, inner_updates=2, parent_policy_unchanged=True,
              source_checkpoint=str(source), origin_updates=1400, source_files_sha256=source_files,
              source_run_identity_sha256=old['sha256'], common_optimizer_sha256='original-Adam',
              source_qualification=False, target_qualification=True)
    target = tmp_path / 'qualification/WORLD16_D0'; target.mkdir(parents=True)
    current = copy.deepcopy(old); current.pop('sha256'); current.update(world=16, fresh_Adam=False,
                     qualification=True, native_world_transition=tx); current['sha256'] = fingerprint(current)
    (target / 'identity.json').write_text(json.dumps(current))
    receipt = dict(schema=tx['schema'], source_identity_sha256=old['sha256'],
                   source_checkpoint=str(source), source_files_sha256=source_files, origin_updates=1400,
                   target_identity_sha256=current['sha256'], common_optimizer_sha256='original-Adam',
                   native_weights_and_Adam_inherited=True, qualification=True, source_qualification=False)
    receipt['sha256'] = fingerprint(receipt)
    (target / 'WORLD_TRANSITION.json').write_text(json.dumps(receipt))
    command = ['python', '-m', 'tools.d46_world16.train', '--output', str(target)]
    assert worker.known_owner_args(command)
    receipt['common_optimizer_sha256'] = 'reset-Adam'
    receipt.pop('sha256'); receipt['sha256'] = fingerprint(receipt)
    (target / 'WORLD_TRANSITION.json').write_text(json.dumps(receipt))
    assert not worker.known_owner_args(command)


def test_reviewed_metadata_failure_resumes_native_and_precreates_request_directory(tmp_path, monkeypatch):
    import pytest
    from tools.d46_priority_rl import owner
    from starVLA.rl.qwen_world_compare.contracts import file_sha
    monkeypatch.setattr(owner, 'ROOT', tmp_path)
    failed=tmp_path/'oldjob';failed.mkdir();(failed/'RESULT.json').write_text('{"returncode":1}')
    config=tmp_path/'config.yaml';config.write_text('{}')
    proof=tmp_path/'recovery.json';proof.write_text('{"status":"PASS"}')
    run=dict(arm='D1',jobs=[dict(job=str(failed))],output=str(tmp_path/'formal'),qualification=str(tmp_path/'qual'),
             config=str(config),config_sha256=file_sha(config),source='immutable-science')
    with pytest.raises(RuntimeError,match='Actual child failure retained'):owner.dispatch(run)
    assert (tmp_path/'formal/evaluation_requests').is_dir()
    run['formal_boundary_recovery_review']=dict(job=str(failed),returncode=1,receipt=str(proof),receipt_sha256=file_sha(proof))
    monkeypatch.setattr(owner,'inspect_latest',lambda run:('own-native-READY',None))
    monkeypatch.setattr(owner,'allowed_hosts',lambda **kwargs:['training-vlawm-tsinghua-zt5'])
    monkeypatch.setattr(owner,'capacity',lambda *args:list(range(8)))
    commands=[]
    monkeypatch.setattr(owner,'launch',lambda host,source,job,command,*args,**kwargs:commands.append(command)or dict(job=str(job)))
    assert owner.dispatch(run)=='RUNNING_OR_ADMITTING'
    assert commands[0][-2:]==['--resume','own-native-READY']
