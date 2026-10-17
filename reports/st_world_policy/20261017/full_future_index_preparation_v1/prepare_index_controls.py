"""Bind actual full-future completions, then reuse the original index proposal helper.

No index build, migration, ledger write, image, tensor or request-body read.
Both roles must complete before any output directory is created.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import stat
from types import ModuleType

BASE = Path('/mnt/project/st-world-policy-20261016')
PREPARATION = {
    'path': str(BASE / 'audits/full_future_metadata_preparation_v1/prep_phase_v1/PREPARATION.json'),
    'sha256': 'dce5b37f391f37c3076a232510e554f382c3a61ad8d983ec381d63c59970855c',
}
HELPER_SHA = 'a478f7c3f963644d630704dd28cb02a02e65fbb3908fcfde2680820d9ae3feff'
CONTRACT = {
    'path': str(BASE / 'audits/fixed_teacher_batch64_runtime_run_v1/RESOLVED_PLAN.json'),
    'sha256': 'a1de94f3e51b3e8ed050fa7f63b55d819cfa6453af7c1268b04d20c51f5e1d32',
}
DEV_COMPLETE_SHA = '5418ed85260c449bee5df626b14e123064b883488816c6103bf4904a4e365180'
DEV_RESULT_SHA = '889a108501ddac3b269ecb5274406bb4d01a69bdd12d6e4a3b51d96321fe013f'
DEV_FAILED = {
    'path': str(BASE / 'caches/full_future_metadata_dev_v1/CONTROL_BUILD_RESULT.json'),
    'sha256': '60b7ba6177b51081fdb323b36a0821f3848e6ddf2c2bb625b97e9a8c850aefda',
}
RESULT_NAMES = {'dev': 'CONTROL_BUILD_ATTEMPT2_RESULT.json', 'train': 'CONTROL_BUILD_RESULT.json'}
COUNTS = {'dev': 1696, 'train': 101592}


def load_helper():
    path = Path(__file__).resolve().parent.parent / 'full_future_metadata_preparation_v1/prepare_controls.py'
    raw = path.read_bytes()
    if len(raw) > 1 << 20 or hashlib.sha256(raw).hexdigest() != HELPER_SHA:
        raise ValueError('Original proposal helper source changed')
    module = ModuleType('original_future_controls')
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    module.READ_BYTES = len(raw)
    return module


def actual_json(helper, path):
    """One bounded stable small-JSON read; derive its actual pin, never an identity."""
    path = helper.safe(path)
    before = path.stat()
    if (path.suffix != '.json' or not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or before.st_size > 4 * helper.MiB or helper.READ_BYTES + before.st_size > 32 * helper.MiB):
        raise ValueError('Small metadata-only completion budget exceeded')
    helper.READ_BYTES += before.st_size
    raw = path.read_bytes()
    after = path.stat()
    version = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_nlink, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if version(before) != version(after) or len(raw) != before.st_size:
        raise ValueError('Completion changed during read')
    return json.loads(raw), {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}


def validate_completion(helper, role, plan_pin, plan, registration, storage_pin, done, result):
    for value in (plan, registration):
        if value.get('identity') != helper.digest({k: v for k, v in value.items() if k != 'identity'}):
            raise ValueError('Original PLAN or preparation registration identity changed')
    if (plan.get('status') != 'PLAN_ONLY' or plan.get('scope') != 'full_population'
            or plan.get('selected_scene_count') != COUNTS[role]
            or registration.get('schema') != 'st_future_metadata_registration_v1'
            or registration.get('phase') != 'data_preparation'
            or registration.get('approved_scope') != 'full_population'
            or registration.get('plan_identity') != plan['identity']
            or registration.get('plan_sha256') != plan_pin['sha256']):
        raise ValueError('Exact original full-population preparation required')
    for value in (plan, registration):
        if any(value.get(k) is not False for k in
               ('gpu_authorized', 'teacher_extraction_authorized', 'training_authorized')):
            raise ValueError('Metadata input grants forbidden model scope')
    expected = {
        'schema': 'st_future_metadata_complete_v1', 'status': 'COMPLETE', 'scope': 'full_population',
        'plan_identity': plan['identity'], 'registration_identity': registration['identity'],
        'scenes': COUNTS[role], 'logs': len(plan['selected_logs']),
        'target_times_s': plan['target_times_s'], 'original_selection': plan['original_selection'],
    }
    if any(done.get(k) != v for k, v in expected.items()):
        raise ValueError('Actual COMPLETE differs from the original role/plan/population')
    if (any(type(done.get(k)) is not int or done[k] != 0 for k in ('teacher_extractions', 'tensor_disk_bytes'))
            or done.get('ego_tensor_decoded') is not False or done.get('training_admission') is not False
            or done.get('raw_log_deserialized') is not True):
        raise ValueError('Completion is not metadata-only')
    hashes = done.get('files')
    if (not isinstance(hashes, dict) or set(hashes) !=
            {'TEACHER_REQUESTS.json', 'RAW_LOG_PROVENANCE.json', 'SHARD_RECEIPTS.json'}):
        raise ValueError('Exact completed metadata dependency set required')
    for value in [done.get('request_identity'), *hashes.values()]:
        if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
            raise ValueError('Malformed actual request identity or dependency hash')
    resources = result.get('resources', {})
    cli = result.get('cli_result', {})
    if (result.get('schema') != 'st_full_future_build_invocation_v1' or result.get('status') != 'BUILD_COMPLETE'
            or result.get('role') != role or result.get('attempt') != (2 if role == 'dev' else 1)
            or result.get('preparation') != PREPARATION or result.get('automatic_retry') is not False
            or result.get('training_authorized') is not False or result.get('scientific_result') is not False
            or resources.get('returncode') != 0 or resources.get('failure') is not None
            or resources.get('cleanup', {}).get('live_owned_remaining') != 0
            or cli.get('status') != 'GUARDED_METADATA_RETURN' or cli.get('registration_sha256') != storage_pin['sha256']
            or cli.get('artifact') != done):
        raise ValueError('Matching successful actual bounded BUILD_COMPLETE required')
    if role == 'dev' and result.get('prior_attempt') != DEV_FAILED:
        raise ValueError('Keep the original failed dev attempt pin')


def completion_inputs(helper):
    preparation = helper.document(PREPARATION)
    inputs = copy.deepcopy(preparation['source_inputs'])
    inputs['phase'] = 'index'
    inputs['contract_receipt'] = CONTRACT
    resource = helper.document(inputs['resource_plan'])
    states = {}
    for role in ('dev', 'train'):
        role_input = inputs['roles'][role]
        root = BASE / ('caches/full_future_metadata_' + role + '_v1')
        if resource['roles'][role]['prep_output_root'] != str(root) or role_input['plan']['path'] != str(root / 'PLAN.json'):
            raise ValueError('Original preparation root changed')
        paths = [root / 'COMPLETE.json', root / RESULT_NAMES[role]]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            states[role] = {'status': 'WAITING_ACTUAL_COMPLETE', 'missing': missing}
            continue
        done, done_pin = actual_json(helper, paths[0])
        result, result_pin = actual_json(helper, paths[1])
        if role == 'dev' and (done_pin['sha256'] != DEV_COMPLETE_SHA or result_pin['sha256'] != DEV_RESULT_SHA):
            raise ValueError('Known actual dev attempt2 receipts changed')
        storage_pin = preparation['registrations'][role]
        storage = helper.document(storage_pin)
        command = storage['command']
        def flag(name):
            if command.count(name) != 1:
                raise ValueError('Original preparation command binding missing or duplicate')
            return command[command.index(name) + 1]
        reg_pin = {'path': flag('--registration'), 'sha256': flag('--registration-sha256')}
        if (flag('--plan') != role_input['plan']['path'] or storage['output_root'] != str(root)
                or storage['writer_kind'] != 'future_preparation'):
            raise ValueError('Original storage registration/root differs')
        plan = helper.document(role_input['plan'])
        registration = helper.document(reg_pin)
        validate_completion(helper, role, role_input['plan'], plan, registration, storage_pin, done, result)
        role_input['future_complete'] = done_pin
        states[role] = {'status': 'ACTUAL_COMPLETE_BOUND', 'complete': done_pin, 'build_result': result_pin,
                       'request_identity': done['request_identity'], 'request_file_sha256': done['files']['TEACHER_REQUESTS.json'],
                       'scenes': done['scenes'], 'logs': done['logs']}
    ready = all(row['status'] == 'ACTUAL_COMPLETE_BOUND' for row in states.values())
    return inputs, {'status': 'READY_FOR_INDEX_CONTROL_PREPARATION' if ready else 'WAITING_ACTUAL_COMPLETE',
                    'roles': states, 'source_preparation': PREPARATION, 'metadata_read_bytes': helper.READ_BYTES,
                    'request_body_read': False, 'index_built': False, 'ledger_written': False}


def index_arguments(helper, role, requests, request_identity, output):
    """The CLI limits the whole index root and halves it for SQLite pages."""
    size = (56 if role == 'dev' else 1008) * helper.MiB
    requirements = {'output_bytes': size, 'staging_bytes': size,
                    'metadata_bytes': (16 if role == 'dev' else 32) * helper.MiB}
    command = [helper.PYTHON, '-m', 'tools.st_world_policy.index_teacher_metadata', 'build',
               '--requests', str(requests), '--expected-request-identity', request_identity,
               '--contract-receipt', CONTRACT['path'], '--contract-receipt-sha256', CONTRACT['sha256'],
               '--output', str(output), '--max-scenes', str(COUNTS[role]),
               '--max-disk-bytes', str(requirements['output_bytes'] + requirements['staging_bytes']),
               '--max-builder-rss-bytes', str(4 * helper.GiB)]
    return command, requirements


def index_proposal(helper, inputs, out, bindings):
    """Compose the original registration/policy helpers; only index argv differs."""
    old = helper.document(inputs['old_policy'])
    ledger = helper.document(inputs['old_ledger'])
    resource = helper.document(inputs['resource_plan'])
    core_names = {'schema', 'campaign', 'state_dir', 'allowed_roots', 'cache_cap_bytes', 'free_reserve_bytes',
                  'ledger_reserve_bytes', 'legacy_reservations', 'metadata_package', 'frozen_w_package'}
    if ledger['core'] != {k: v for k, v in old.items() if k in core_names}:
        raise ValueError('Actual current policy/core mismatch')
    if any(ledger['claims'][c['root']]['claim'] != c for c in old['claims']):
        raise ValueError('Current claim mismatch')
    if (old['metadata_package']['schema'] != 'st_metadata_storage_package_v2'
            or old['metadata_package']['maximum_bytes'] != 8 * helper.GiB
            or resource['final_metadata_reserved_mib'] != 5616):
        raise ValueError('Existing exact eight-GiB metadata envelope/resource plan required')
    prepared = helper.document(PREPARATION)
    baseline = helper.document(prepared['proposed_policy'])
    if old['metadata_package'] != baseline['metadata_package']:
        raise ValueError('Original metadata-v2 package baseline changed')
    helper.document(CONTRACT)
    rows = []
    for role in ('dev', 'train'):
        item = inputs['roles'][role]
        source = helper.document(item['source_pins'])
        plan = helper.document(item['plan'])
        for name, sha in plan['implementation'].items():
            pin = source['files'][name]
            if pin['sha256'] != sha:
                raise ValueError('Index runtime changes original PLAN implementation: ' + name)
            helper.read(pin)
        # Recheck the exact small COMPLETE pin after the current ledger read.
        done = helper.document(item['future_complete'])
        if done['request_identity'] != bindings[role]['request_identity']:
            raise ValueError('Completed request identity changed')
        role_budget = resource['roles'][role]
        root = BASE / ('caches/full_future_request_index_' + role + '_v1')
        if (role_budget['index_output_root'] != str(root) or role_budget['scenes'] != COUNTS[role]
                or root.exists() or any(c['root'] == str(root) for c in old['claims'])):
            raise ValueError('Exact new full-population index root required; no overwrite')
        command, requirements = index_arguments(helper, role,
            Path(role_budget['prep_output_root']) / 'TEACHER_REQUESTS.json', done['request_identity'], root)
        if sum(requirements.values()) != role_budget['index_claim_inclusive_mib'] * helper.MiB:
            raise ValueError('Original inclusive role storage reservation changed')
        registration = helper.storage_registration(role, 'index', root, source, command, requirements)
        rows.append((role, root, registration, requirements))
    claims, registrations = [], {}
    for role, root, registration, requirements in rows:
        pin = helper.write(out, role.upper() + '_STORAGE_REGISTRATION.json', registration)
        registrations[role] = pin
        claims.append({'root': str(root), 'claim_id': 'full-future-' + role + '-index-v1',
                       'identity_kind': 'st_metadata_build_registration_v1', 'cache_identity': pin['sha256'],
                       'registration_path': pin['path'], 'registration_sha256': pin['sha256'], **requirements})
    policy_pin = helper.write(out, 'PROPOSED_POLICY.json', helper.extend_policy(old, claims))
    inventory = helper.write(out, 'NO_PILOT_ADOPTIONS.json', {'schema': 'st_metadata_pilot_inventory_v1', 'roots': []})
    return {'schema': 'st_full_future_metadata_controls_v1', 'status': 'PREPARED_NOT_MIGRATED_NOT_AUTHORIZED',
            'phase': 'index', 'source_inputs': inputs, 'proposed_policy': policy_pin, 'registrations': registrations,
            'pilot_inventory': inventory, 'index_phase': 'EXACT_REQUEST_BOUND',
            'new_claim_bytes': sum(sum(row[3].values()) for row in rows),
            'administrative_metadata_bytes_not_claims': 128 * helper.MiB,
            'final_metadata_budget_bytes': 5616 * helper.MiB, 'metadata_envelope_bytes': 8 * helper.GiB,
            'new_teacher_tensor_bytes': 0, 'live_ledger_written': False, 'build_executed': False,
            'dispatch_authorized': False, 'metadata_read_bytes': helper.READ_BYTES,
            'argv_budget_correction': 'max-disk-bytes is the root limit: output + staging; metadata remains separately reserved'}


def prepare(args):
    helper = load_helper()
    inputs, status = completion_inputs(helper)
    if args.check_only or status['status'] != 'READY_FOR_INDEX_CONTROL_PREPARATION':
        return status
    if not all((args.policy, args.policy_sha256, args.ledger_sha256, args.output)):
        raise ValueError('Fresh actual policy/ledger pins and a new output path are required')
    inputs['old_policy'] = {'path': str(helper.safe(args.policy)), 'sha256': args.policy_sha256}
    inputs['old_ledger'] = {'path': str(BASE / 'resources/cache_storage_ledger_v1/CLAIMS.json'),
                            'sha256': args.ledger_sha256}
    out = helper.safe(args.output)
    allowed = BASE / 'audits/full_future_index_preparation_v1'
    if not out.is_relative_to(allowed) or out == allowed:
        raise ValueError('Use a new child under the isolated index preparation namespace')
    # Original registration and policy helpers remain responsible for callback
    # pins and additive declarations. No request body or log_shards are visited.
    out.mkdir(parents=True, exist_ok=False)
    result = index_proposal(helper, inputs, out, status['roles'])
    result['actual_completion_bindings'] = status['roles']
    result['original_helper_sha256'] = HELPER_SHA
    # A prior additive 4 MiB build-monitor allocation remains separately charged.
    # Do not mutate the original 5616 MiB resource-plan pin to hide that addendum.
    result['separate_existing_build_monitor_addendum_bytes'] = 4 * helper.MiB
    result['final_metadata_with_existing_addendum_bytes'] = (5616 + 4) * helper.MiB
    pin = helper.write(out, 'PREPARATION.json', result)
    return {'status': result['status'], 'preparation': pin, 'index_built': False, 'ledger_written': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Read only actual small completion controls; no output')
    parser.add_argument('--policy')
    parser.add_argument('--policy-sha256')
    parser.add_argument('--ledger-sha256')
    parser.add_argument('--output')
    result = prepare(parser.parse_args())
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 3 if result['status'] == 'WAITING_ACTUAL_COMPLETE' else 0


if __name__ == '__main__':
    raise SystemExit(main())
