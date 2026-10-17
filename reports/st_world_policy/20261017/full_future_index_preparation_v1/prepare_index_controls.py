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
import sys
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
FINALIZER_RUNTIME = Path('/mnt/project/VLA-Drive-st-future-finalize-runtime-v1-20261017')
FINALIZED_TRAIN = BASE / 'caches/full_future_metadata_train_finalized_v1'
FINALIZATION_RESULT = BASE / 'audits/full_future_finalization_v1_controls/RESULT.json'


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


def strict_finalized_receipt(helper, done, done_pin):
    """Reuse the reviewed validator; do not reproduce its provenance checks here.

    This lazy import includes the writer/Torch chain. Its import IO is an outer
    process allowance, separate from helper.READ_BYTES's small-control budget.
    """
    registration = helper.document(done['finalization_registration'])
    expected = {str(FINALIZER_RUNTIME / (name.replace('.', '/') + '.py')) for name in (
        'tools.st_world_policy.finalize_future_metadata', 'tools.st_world_policy.prepare_future_metadata',
        'starVLA.dataloader.st_world_metadata_index', 'starVLA.dataloader.st_world_future_provenance',
        'tools.st_world_policy.prepare_current_teacher', 'starVLA.dataloader.st_world_schema',
        'starVLA.dataloader.st_world_targets', 'starVLA.dataloader.st_world_history',
        'tools.st_world_policy.prepare_history')}
    if {pin['path'] for pin in registration['implementation']} != expected:
        raise ValueError('Finalizer runtime implementation pin set differs')
    for pin in registration['implementation']: helper.read(pin)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(FINALIZER_RUNTIME))
    from tools.st_world_policy import finalize_future_metadata as finalizer
    if Path(finalizer.__file__).resolve() != FINALIZER_RUNTIME / 'tools/st_world_policy/finalize_future_metadata.py':
        raise ValueError('Unexpected already-imported finalizer module')
    finalizer.validate_completion(done, done_pin['path'],
        request_sha256=done['files']['TEACHER_REQUESTS.json'],
        request_identity=done['request_identity'], scene_count=COUNTS['train'])
    return registration


def bind_finalized_train(helper, plan_pin, done, done_pin, result, result_pin):
    """Only the new terminal schema can establish completion of finalization."""
    resources = result.get('resources', {})
    cli = result.get('cli_result', {})
    if (done.get('schema') != 'st_future_metadata_finalization_complete_v1'
            or done.get('status') != 'COMPLETE' or done.get('scope') != 'full_population'
            or done.get('scenes') != COUNTS['train'] or done.get('source_plan') != plan_pin
            or Path(done_pin['path']) != FINALIZED_TRAIN / 'FINALIZATION_COMPLETE.json'
            or Path(result_pin['path']) != FINALIZATION_RESULT
            or result.get('schema') != 'st_future_finalization_invocation_v1'
            or result.get('status') != 'FINALIZATION_COMPLETE'
            or result.get('automatic_retry') is not False or result.get('scientific_result') is not False
            or resources.get('returncode') != 0 or resources.get('failure') is not None
            or resources.get('cleanup', {}).get('live_owned_remaining') != 0
            or result.get('cleanup', {}).get('live_owned_remaining') != 0
            or resources.get('read_accounting', {}).get('status') != 'COMPLETE'
            or cli.get('status') != 'GUARDED_METADATA_RETURN' or cli.get('artifact') != done
            or cli.get('registration_sha256') != result.get('registration', {}).get('sha256')):
        raise ValueError('Exact successful finalization terminal/receipt pair required')
    admission = helper.document(result['admission'])
    storage = helper.document(result['registration'])
    if (admission.get('schema') != 'st_future_finalization_owned_admission_v1'
            or admission.get('status') != 'APPROVED_FUTURE_FINALIZATION_ONLY'
            or admission.get('reviewer') in (None, '', '/root/st_future_finalization_cost')
            or admission.get('automatic_retry') is not False
            or admission.get('registration') != result['registration']
            or admission.get('finalization_registration') != done['finalization_registration']
            or result.get('finalization_registration') != done['finalization_registration']
            or admission.get('source_terminal') != done['source_terminal']
            or result.get('source_terminal') != done['source_terminal']
            or admission.get('output_root') != str(FINALIZED_TRAIN)
            or admission.get('control_root') != str(FINALIZATION_RESULT.parent)
            or admission.get('runtime') != str(FINALIZER_RUNTIME)
            or result.get('ledger_result') != str(FINALIZED_TRAIN / 'CONTROL_FINALIZATION_RESULT.json')
            or type(result.get('charged_rchar_bytes')) is not int
            or not 0 <= result['charged_rchar_bytes'] <= admission['limits']['read_bytes']
            or storage.get('writer_kind') != 'future_finalization'
            or storage.get('output_root') != str(FINALIZED_TRAIN) or storage.get('cwd') != str(FINALIZER_RUNTIME)):
        raise ValueError('Finalization admission/registration/lineage differs')
    pin = done['finalization_registration']
    expected = [helper.PYTHON, '-m', 'tools.st_world_policy.finalize_future_metadata', 'finalize-existing',
        '--registration', pin['path'], '--registration-sha256', pin['sha256'], '--output', str(FINALIZED_TRAIN)]
    if storage.get('command') != expected: raise ValueError('Finalization worker argv differs')
    return strict_finalized_receipt(helper, done, done_pin)


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
        paths = ([FINALIZED_TRAIN / 'FINALIZATION_COMPLETE.json', FINALIZATION_RESULT] if role == 'train'
                 else [root / 'COMPLETE.json', root / RESULT_NAMES[role]])
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
        if role == 'train':
            finalized = bind_finalized_train(helper, role_input['plan'], done, done_pin, result, result_pin)
            if finalized['source_registration'] != reg_pin:
                raise ValueError('Finalization must retain the original preparation registration')
            role_input['original_source_code_root'] = finalized['source_code_root']
        else:
            validate_completion(helper, role, role_input['plan'], plan, registration, storage_pin, done, result)
        role_input['future_complete'] = done_pin
        states[role] = {'status': 'ACTUAL_COMPLETE_BOUND', 'complete': done_pin, 'build_result': result_pin,
                       'request_identity': done['request_identity'], 'request_file_sha256': done['files']['TEACHER_REQUESTS.json'],
                       'scenes': done['scenes'], 'logs': done['logs'],
                       'requests': str(Path(done_pin['path']).parent / 'TEACHER_REQUESTS.json')}
    ready = all(row['status'] == 'ACTUAL_COMPLETE_BOUND' for row in states.values())
    return inputs, {'status': 'READY_FOR_INDEX_CONTROL_PREPARATION' if ready else 'WAITING_ACTUAL_COMPLETE',
                    'roles': states, 'source_preparation': PREPARATION, 'metadata_read_bytes': helper.READ_BYTES,
                    'metadata_read_bytes_scope': 'helper reads only; validator internal reads/imports require outer process accounting',
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


def planned_metadata_bytes(helper, policy, new_claim_bytes):
    """Count current declarations plus the two new indexes once, retaining controls."""
    package = policy['metadata_package']
    if package['schema'] != 'st_metadata_storage_package_v2' or package['maximum_bytes'] != 8 * helper.GiB:
        raise ValueError('Exact existing eight-GiB metadata package required')
    baseline = package['baseline_claims']
    current = {row['root']: row for row in policy['claims']}
    if any(current.get(root) != row for root, row in baseline.items()):
        raise ValueError('Metadata baseline claim changed')
    old_ids = {row['claim_id'] for row in package['baseline_legacy_reservations']}
    amount = sum(row['reserved_bytes'] for row in policy['legacy_reservations'] if row['claim_id'] not in old_ids)
    amount += sum(sum(row[key] for key in ('output_bytes', 'staging_bytes', 'metadata_bytes'))
        for root, row in current.items() if root not in baseline and row.get('identity_kind') != 'st_frozen_spatial_w_v1')
    # Existing administrative128MiB, earlier build monitor4MiB, history control1MiB,
    # and new finalization terminal control1MiB remain separately planned.
    amount += new_claim_bytes + (128 + 4 + 1 + 1) * helper.MiB
    if amount > package['maximum_bytes']: raise ValueError('Planned full metadata package exceeds eight GiB')
    return amount


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
            or old['metadata_package']['maximum_bytes'] != 8 * helper.GiB):
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
            if role == 'train':
                # The original PLAN binds the original frozen writer. New index
                # callback source pins are independently validated below.
                pin = {'path': str(Path(item['original_source_code_root']) / name), 'sha256': sha}
            else:
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
            bindings[role]['requests'], done['request_identity'], root)
        if sum(requirements.values()) != role_budget['index_claim_inclusive_mib'] * helper.MiB:
            raise ValueError('Original inclusive role storage reservation changed')
        registration = helper.storage_registration(role, 'index', root, source, command, requirements)
        rows.append((role, root, registration, requirements))
    new_claim_bytes = sum(sum(row[3].values()) for row in rows)
    final_metadata_budget = planned_metadata_bytes(helper, old, new_claim_bytes)
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
            'new_claim_bytes': new_claim_bytes,
            'administrative_metadata_bytes_not_claims': 128 * helper.MiB,
            'final_metadata_budget_bytes': final_metadata_budget, 'metadata_envelope_bytes': 8 * helper.GiB,
            'new_teacher_tensor_bytes': 0, 'live_ledger_written': False, 'build_executed': False,
            'dispatch_authorized': False, 'metadata_read_bytes': helper.READ_BYTES,
            'metadata_read_bytes_scope': 'helper reads only; validator internal reads/imports require outer process accounting',
            'argv_budget_correction': 'max-disk-bytes is the root limit: output + staging; metadata remains separately reserved'}


def prepare(args):
    helper = load_helper()
    inputs, status = completion_inputs(helper)
    if args.check_only or status['status'] != 'READY_FOR_INDEX_CONTROL_PREPARATION':
        return status
    if not all((args.policy, args.policy_sha256, args.ledger_sha256, args.output)):
        raise ValueError('Fresh actual policy/ledger pins and a new output path are required')
    if not args.train_index_source_pins or not args.train_index_source_pins_sha256:
        raise ValueError('New immutable TRAIN index runtime source pins required')
    source_pin = {'path': str(helper.safe(args.train_index_source_pins)), 'sha256': args.train_index_source_pins_sha256}
    source = helper.document(source_pin)
    if source.get('runtime') != str(FINALIZER_RUNTIME):
        raise ValueError('TRAIN index must use the separately pinned finalizer-compatible runtime')
    inputs['roles']['train']['source_pins'] = source_pin
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
    result['final_metadata_with_existing_addendum_bytes'] = result['final_metadata_budget_bytes']
    pin = helper.write(out, 'PREPARATION.json', result)
    return {'status': result['status'], 'preparation': pin, 'index_built': False, 'ledger_written': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Read only actual small completion controls; no output')
    parser.add_argument('--policy')
    parser.add_argument('--policy-sha256')
    parser.add_argument('--ledger-sha256')
    parser.add_argument('--output')
    parser.add_argument('--train-index-source-pins')
    parser.add_argument('--train-index-source-pins-sha256')
    result = prepare(parser.parse_args())
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 3 if result['status'] == 'WAITING_ACTUAL_COMPLETE' else 0


if __name__ == '__main__':
    raise SystemExit(main())
