"""Small control fixtures only; never index, reserve storage or open request bodies."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location('index_controls', Path(__file__).with_name('prepare_index_controls.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def controls(role='dev'):
    helper = m.load_helper()
    seal = lambda value: {**value, 'identity': helper.digest(value)}
    flags = dict(gpu_authorized=False, teacher_extraction_authorized=False, training_authorized=False)
    plan = seal(dict(status='PLAN_ONLY', scope='full_population', selected_scene_count=m.COUNTS[role],
                     selected_logs=['synthetic'], target_times_s=[0., .5, 1., 2., 4.],
                     original_selection={'synthetic': 'fixture'}, **flags))
    plan_pin = {'path': '/fixture/PLAN.json', 'sha256': '1' * 64}
    reg = seal(dict(schema='st_future_metadata_registration_v1', phase='data_preparation',
                    approved_scope='full_population', plan_identity=plan['identity'],
                    plan_sha256=plan_pin['sha256'], **flags))
    storage_pin = {'path': '/fixture/STORAGE.json', 'sha256': '2' * 64}
    done = dict(schema='st_future_metadata_complete_v1', status='COMPLETE', scope='full_population',
                plan_identity=plan['identity'], registration_identity=reg['identity'], scenes=m.COUNTS[role],
                logs=1, target_times_s=plan['target_times_s'], original_selection=plan['original_selection'],
                raw_log_deserialized=True, ego_tensor_decoded=False, teacher_extractions=0,
                tensor_disk_bytes=0, training_admission=False, request_identity='3' * 64,
                files={name: '4' * 64 for name in ('TEACHER_REQUESTS.json', 'RAW_LOG_PROVENANCE.json', 'SHARD_RECEIPTS.json')})
    result = dict(schema='st_full_future_build_invocation_v1', status='BUILD_COMPLETE', role=role,
                  attempt=2 if role == 'dev' else 1, preparation=m.PREPARATION, automatic_retry=False,
                  training_authorized=False, scientific_result=False,
                  resources=dict(returncode=0, failure=None, cleanup={'live_owned_remaining': 0}),
                  cli_result=dict(status='GUARDED_METADATA_RETURN', registration_sha256=storage_pin['sha256'],
                                  artifact=copy.deepcopy(done)))
    if role == 'dev':
        result['prior_attempt'] = copy.deepcopy(m.DEV_FAILED)
    return helper, role, plan_pin, plan, reg, storage_pin, done, result


@pytest.mark.parametrize('role', ['dev', 'train'])
def test_exact_control_binding_accepts_only_fixture_metadata(role):
    values = controls(role)
    before = copy.deepcopy(values[2:])
    m.validate_completion(*values)
    assert values[2:] == before


@pytest.mark.parametrize('case', [
    'partial', 'plan_identity', 'population', 'target_times', 'request_identity',
    'extra_dependency', 'tensor_bytes', 'tensor_decode', 'supervisor_failed',
    'dev_first_attempt', 'changed_artifact', 'child_failure', 'live_child',
    'changed_storage_registration', 'dropped_failed_receipt', 'plan_gpu_scope',
])
def test_invalid_completion_never_reaches_claim_generation(case):
    values = controls()
    helper, role, plan_pin, plan, reg, storage_pin, done, result = values
    if case == 'partial': done['status'] = 'PARTIAL'
    elif case == 'plan_identity': done['plan_identity'] = '5' * 64
    elif case == 'population': done['scenes'] -= 1
    elif case == 'target_times': done['target_times_s'] = [.5]
    elif case == 'request_identity': done['request_identity'] = 'pending'
    elif case == 'extra_dependency': done['files']['unknown.pt'] = '6' * 64
    elif case == 'tensor_bytes': done['tensor_disk_bytes'] = 1
    elif case == 'tensor_decode': done['ego_tensor_decoded'] = True
    elif case == 'supervisor_failed': result['status'] = 'FAILED_NO_AUTO_RETRY'
    elif case == 'dev_first_attempt': result['attempt'] = 1
    elif case == 'changed_artifact': result['cli_result']['artifact']['request_identity'] = '7' * 64
    elif case == 'child_failure': result['resources']['returncode'] = 1
    elif case == 'live_child': result['resources']['cleanup']['live_owned_remaining'] = 1
    elif case == 'changed_storage_registration': result['cli_result']['registration_sha256'] = '8' * 64
    elif case == 'dropped_failed_receipt': result['prior_attempt'] = None
    elif case == 'plan_gpu_scope':
        plan['gpu_authorized'] = True
        plan['identity'] = helper.digest({k: v for k, v in plan.items() if k != 'identity'})
    with pytest.raises(ValueError):
        m.validate_completion(*values)


def test_waiting_train_leaves_no_partial_claim_or_output(tmp_path, monkeypatch):
    helper = m.load_helper()
    monkeypatch.setattr(m, 'load_helper', lambda: helper)
    state = {'status': 'WAITING_ACTUAL_COMPLETE', 'roles': {'train': {'status': 'WAITING_ACTUAL_COMPLETE'}}}
    monkeypatch.setattr(m, 'completion_inputs', lambda unused: ({}, state))
    monkeypatch.setattr(m, 'index_proposal', lambda *args: pytest.fail('Cannot generate claims while train is absent'))
    output = tmp_path / 'must_not_exist'
    args = SimpleNamespace(check_only=False, policy=None, policy_sha256=None, ledger_sha256=None, output=str(output))
    assert m.prepare(args) == state
    assert not output.exists()


def test_check_only_remains_read_only_even_when_ready(tmp_path, monkeypatch):
    helper = m.load_helper()
    monkeypatch.setattr(m, 'load_helper', lambda: helper)
    state = {'status': 'READY_FOR_INDEX_CONTROL_PREPARATION'}
    monkeypatch.setattr(m, 'completion_inputs', lambda unused: ({}, state))
    monkeypatch.setattr(m, 'index_proposal', lambda *args: pytest.fail('check-only may not emit controls'))
    assert m.prepare(SimpleNamespace(check_only=True)) == state
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('role,root_mib,output_mib,metadata_mib,total_mib', [
    ('dev', 112, 56, 16, 128), ('train', 2016, 1008, 32, 2048),
])
def test_cli_root_budget_matches_existing_sqlite_half_rule_and_claim(role, root_mib, output_mib, metadata_mib, total_mib):
    helper = m.load_helper()
    args, req = m.index_arguments(helper, role, '/fixture/TEACHER_REQUESTS.json', 'a' * 64, '/fixture/index')
    flag = lambda name: args[args.index(name) + 1]
    assert int(flag('--max-disk-bytes')) == root_mib * helper.MiB
    # Real indexer: max_page_count = max_disk_bytes // (2 * 4096).
    assert int(flag('--max-disk-bytes')) // (2 * 4096) * 4096 == req['output_bytes'] == output_mib * helper.MiB
    assert req['staging_bytes'] == req['output_bytes']
    assert req['metadata_bytes'] == metadata_mib * helper.MiB
    assert sum(req.values()) == total_mib * helper.MiB
    assert flag('--expected-request-identity') == 'a' * 64
    assert int(flag('--max-scenes')) == m.COUNTS[role]
    assert '--cache-root' not in args and '--expected-cache-identity' not in args


@pytest.mark.parametrize('kind', ['payload', 'oversized', 'symlink', 'hardlink', 'aggregate_budget'])
def test_actual_read_refuses_unsafe_or_overbudget_file_before_read(tmp_path, monkeypatch, kind):
    helper = m.load_helper()
    path = tmp_path / ('bad.pt' if kind == 'payload' else 'COMPLETE.json')
    if kind == 'symlink':
        path.symlink_to(tmp_path / 'missing')
    else:
        path.write_text('{}')
        if kind == 'oversized':
            with path.open('r+b') as stream: stream.truncate(4 * helper.MiB + 1)
        elif kind == 'hardlink': os.link(path, tmp_path / 'alias.json')
        elif kind == 'aggregate_budget': helper.READ_BYTES = 32 * helper.MiB
    monkeypatch.setattr(Path, 'read_bytes', lambda unused: pytest.fail('Rejected control must not be opened'))
    with pytest.raises(ValueError):
        m.actual_json(helper, path)


def test_actual_json_pin_is_derived_from_stable_bytes(tmp_path):
    helper = m.load_helper()
    path = tmp_path / 'COMPLETE.json'
    raw = b'{"status":"fixture"}\n'
    path.write_bytes(raw)
    value, pin = m.actual_json(helper, path)
    assert value == json.loads(raw)
    assert pin == {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}


def test_changed_during_read_is_rejected(tmp_path, monkeypatch):
    helper = m.load_helper()
    path = tmp_path / 'COMPLETE.json'
    path.write_text('{}')
    original = Path.read_bytes
    def changing_read(target):
        raw = original(target)
        target.write_text('{"changed":true}')
        return raw
    monkeypatch.setattr(Path, 'read_bytes', changing_read)
    with pytest.raises(ValueError, match='changed during read'):
        m.actual_json(helper, path)


@pytest.fixture
def proposal_inputs(tmp_path, monkeypatch):
    """Synthetic tiny metadata controls exercise the real original registration helpers."""
    helper = m.load_helper()
    base = tmp_path / 'campaign'
    monkeypatch.setattr(m, 'BASE', base)
    def pin(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = (json.dumps(value, sort_keys=True) + '\n').encode()
        path.write_bytes(raw)
        return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}
    old = {'schema': 'fixture', 'state_dir': str(base / 'ledger'), 'cache_cap_bytes': 768 * helper.GiB,
           'metadata_package': {'schema': 'st_metadata_storage_package_v2', 'maximum_bytes': 8 * helper.GiB},
           'frozen_w_package': {'fixture': 'unchanged'}, 'legacy_reservations': [],
           'claims': [{'root': str(base / 'existing'), 'claim_id': 'preserved'}]}
    policy = pin('POLICY.json', old)
    ledger = {'core': {k: v for k, v in old.items() if k != 'claims'},
              'claims': {old['claims'][0]['root']: {'claim': old['claims'][0]}}}
    inputs = {'old_policy': policy, 'old_ledger': pin('CLAIMS.json', ledger), 'roles': {}}
    resource = {'final_metadata_reserved_mib': 5616, 'roles': {}}
    bindings = {}
    for role in ('dev', 'train'):
        runtime = tmp_path / ('runtime_' + role)
        files = {}
        for name, text in {
            'tools/st_world_policy/index_teacher_metadata.py':
                "METADATA_STORAGE_CALLBACK_PROTOCOL='st_metadata_storage_check_v1'\n"
                "METADATA_STORAGE_CALLBACK_DEPENDENCIES=('starVLA.dataloader.st_world_metadata_index','starVLA.dataloader.st_world_future_provenance')\n",
            'starVLA/dataloader/st_world_metadata_index.py': '# synthetic metadata dependency\n',
            'starVLA/dataloader/st_world_future_provenance.py': '# synthetic provenance dependency\n',
        }.items():
            path = runtime / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            files[name] = {'path': str(path), 'sha256': hashlib.sha256(text.encode()).hexdigest()}
        identity = ('a' if role == 'dev' else 'b') * 64
        inputs['roles'][role] = {
            'source_pins': pin(role + '/SOURCE.json', {'runtime': str(runtime), 'files': files}),
            'plan': pin(role + '/PLAN.json', {'implementation': {}}),
            'future_complete': pin(role + '/COMPLETE.json', {'request_identity': identity}),
        }
        bindings[role] = {'request_identity': identity}
        resource['roles'][role] = {
            'scenes': m.COUNTS[role], 'index_claim_inclusive_mib': 128 if role == 'dev' else 2048,
            'prep_output_root': str(base / ('caches/full_future_metadata_' + role + '_v1')),
            'index_output_root': str(base / ('caches/full_future_request_index_' + role + '_v1')),
        }
    inputs['resource_plan'] = pin('RESOURCE.json', resource)
    monkeypatch.setattr(m, 'PREPARATION', pin('PREPARATION.json', {'proposed_policy': policy}))
    monkeypatch.setattr(m, 'CONTRACT', pin('CONTRACT.json', {'fixture': 'contract pin'}))
    out = tmp_path / 'new_controls'
    out.mkdir()
    return helper, inputs, out, bindings, old, ledger, pin


def test_real_helper_composition_only_adds_two_bounded_claims(proposal_inputs):
    helper, inputs, out, bindings, old, ledger, pin = proposal_inputs
    originals = {key: Path(inputs[key]['path']).read_bytes() for key in ('old_policy', 'old_ledger')}
    result = m.index_proposal(helper, inputs, out, bindings)
    policy = helper.document(result['proposed_policy'])
    assert policy['claims'][:len(old['claims'])] == old['claims']
    assert {k: v for k, v in policy.items() if k != 'claims'} == {k: v for k, v in old.items() if k != 'claims'}
    assert result['new_claim_bytes'] == (128 + 2048) * helper.MiB
    assert result['dispatch_authorized'] is False and result['live_ledger_written'] is False
    for role, mib in [('dev', 112), ('train', 2016)]:
        reg = helper.document(result['registrations'][role])
        assert reg['writer_kind'] == 'future_index' and len(reg['source_pins']) == 3
        assert reg['command'][reg['command'].index('--max-disk-bytes') + 1] == str(mib * helper.MiB)
        assert not Path(reg['output_root']).exists()
    assert all(Path(inputs[key]['path']).read_bytes() == value for key, value in originals.items())


@pytest.mark.parametrize('case', ['stale_ledger', 'changed_envelope', 'existing_index_root', 'changed_complete_pin'])
def test_proposal_failures_publish_no_registration(proposal_inputs, case):
    helper, inputs, out, bindings, old, ledger, pin = proposal_inputs
    if case == 'stale_ledger':
        ledger['core']['cache_cap_bytes'] -= 1
        inputs['old_ledger'] = pin('STALE_LEDGER.json', ledger)
    elif case == 'changed_envelope':
        old['metadata_package']['maximum_bytes'] += 1
        ledger['core']['metadata_package'] = old['metadata_package']
        inputs['old_policy'] = pin('BAD_POLICY.json', old)
        inputs['old_ledger'] = pin('BAD_LEDGER.json', ledger)
    elif case == 'existing_index_root':
        (m.BASE / 'caches/full_future_request_index_dev_v1').mkdir(parents=True)
    else:
        Path(inputs['roles']['train']['future_complete']['path']).write_text('{}')
    with pytest.raises(ValueError):
        m.index_proposal(helper, inputs, out, bindings)
    assert not list(out.iterdir())
