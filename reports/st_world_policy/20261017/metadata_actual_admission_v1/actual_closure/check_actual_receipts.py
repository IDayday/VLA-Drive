"""Actual metadata closure from small receipts/stat only; never hash SQLite/data."""
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time

ROOT = Path('/mnt/project/st-world-policy-20261016')
HERE = Path(__file__).parent
P = ROOT/'audits/metadata_migration_preparation_v1'
pins = []
checks = {}


def read(path, expected=None):
    path = Path(path)
    assert path.suffix == '.json' and path.stat().st_size < 65536, path
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if expected is not None:
        assert sha == expected, path
    pins.append({'path': str(path), 'sha256': sha, 'bytes': len(raw)})
    return json.loads(raw)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def stat_tree(root, registration):
    rows = []
    stack = [(root, 0)]
    while stack:
        path, depth = stack.pop()
        assert len(rows) < registration['limits']['max_entries']
        info = path.lstat()
        rel = '.' if path == root else path.relative_to(root).as_posix()
        assert info.st_dev == root.stat().st_dev and not stat.S_ISLNK(info.st_mode)
        if stat.S_ISDIR(info.st_mode):
            assert depth <= registration['limits']['max_depth']
            if path != root:
                assert any(re.fullmatch(p, rel) for p in registration['directory_patterns'])
            stack.extend((p, depth+1) for p in path.iterdir())
        else:
            assert stat.S_ISREG(info.st_mode) and info.st_nlink == 1
            assert any(re.fullmatch(p, rel) for p in registration['file_patterns']), path
        rows.append({'relative': rel, 'logical_bytes': info.st_size,
                     'allocated_bytes': info.st_blocks*512, 'mtime_ns': info.st_mtime_ns})
    return {'entries': rows, 'charged_bytes': sum(max(r['logical_bytes'], r['allocated_bytes']) for r in rows)}


approval = read(HERE.parent/'RESOURCE_APPROVAL.json', 'a759d86502d03dc32bbfa150ac64d69e7f389c57cb5a28ef8f0daa6ff6ef230a')
prep = read(P/'PREPARED_V6.json', approval['prepared_sha256'])
policy = read(prep['new_policy'], approval['new_policy_sha256'])
old_policy = read(prep['old_policy'], prep['old_policy_sha256'])
quiescence = read(P/'QUIESCENCE_V2.json', approval['quiescence_sha256'])
checkpoint = read(P/'CHECKPOINT_COMMITMENTS_V1.json', approval['checkpoint_sha256'])
future_plan = read(prep['future_plan'], prep['future_plan_sha256'])
ledger_path = Path(policy['state_dir'])/'CLAIMS.json'
ledger = read(ledger_path)
ledger_start_sha = pins[-1]['sha256']
migration_id = '6a0001e03abd39bb79d493a8928d913a6d0fabf8d994a3d64576ec02b06b234a'
prefix = ledger_path.parent/('MIGRATION_'+migration_id)
complete_path = Path(str(prefix)+'.complete.json')
migration = read(complete_path)
pending = read(str(prefix)+'.pending.json')
snapshot = read(str(prefix)+'.snapshot.json')
old_raw = bytes.fromhex(snapshot['ledger_hex'])
old_ledger = json.loads(old_raw)
checks['migration_snapshot_exact'] = hashlib.sha256(old_raw).hexdigest() == snapshot['old_sha256'] == prep['old_ledger_sha256']
checks['pending_snapshot_preserved'] = pending['old_ledger_hex'] == snapshot['ledger_hex']
checks['migration_transition_identity'] = digest(pending['transition']) == migration_id == migration['migration_id']
checks['migration_policy_binding'] = migration['new_policy_sha256'] == prep['new_policy_sha256'] and migration['old_policy_sha256'] == prep['old_policy_sha256']
checks['migration_checkpoint_quiescence'] = migration['checkpoint_sha256'] == approval['checkpoint_sha256'] and migration['completion_quiescence_sha256'] == approval['quiescence_sha256']
checks['migration_in_quiescence_window'] = quiescence['checked_at_unix'] <= complete_path.stat().st_mtime <= quiescence['valid_until_unix']
checks['single_migration'] = ledger['metadata_migrations'] == [migration_id]
checks['new_core_hash'] = digest(ledger['core']) == migration['new_core_sha256']
checks['new_core_matches_policy'] = ledger['core'] == {k: policy[k] for k in ledger['core']}
migrated_state = copy.deepcopy(old_ledger)
migrated_state['core'] = ledger['core']
migrated_state['metadata_migrations'] = [migration_id]
checks['intermediate_migrated_ledger_hash'] = hashlib.sha256((json.dumps(migrated_state,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()).hexdigest() == migration['new_ledger_sha256']
checks['old_claim_records_unchanged'] = all(ledger['claims'].get(k) == v for k,v in old_ledger['claims'].items())
checks['old_policy_still_immutable'] = True
stage_results = {}
trees = {}
runtime = Path(prep['runtime'])
for key in ('future_storage_registration', 'index_storage_registration'):
    reg = read(prep[key], prep[key+'_sha256'])
    root = Path(reg['output_root'])
    tree = stat_tree(root, reg)
    trees[str(root)] = tree
    claim = ledger['claims'][str(root)]
    maximum = sum(reg['storage_requirements'].values())
    checks[key+'_full_charge_retained'] = claim['charged_bytes'] == maximum == reg['limits']['max_disk_bytes']
    checks[key+'_final_tree_within_charge'] = tree['charged_bytes'] <= maximum
    admission = read(root/'METADATA_ADMISSION.json')
    checks[key+'_admission_identity'] = admission['claim'] == claim['claim'] and admission['policy_sha256_at_admission'] == prep['new_policy_sha256'] and admission['registration_sha256'] == prep[key+'_sha256']
    for pin in reg['source_pins']:
        source = Path(pin['path'])
        assert source.is_relative_to(runtime) and source.stat().st_size < 2**20
        assert hashlib.sha256(source.read_bytes()).hexdigest() == pin['sha256']
    checks[key+'_source_pins_unchanged'] = True
    names = ['future_build', 'future_reference'] if key.startswith('future') else ['current_index']
    for stage in names:
        started = read(root/('CONTROL_'+stage.upper()+'_STARTED.json'))
        result_path = root/('CONTROL_'+stage.upper()+'_RESULT.json')
        expected = '98120d28ce9ec254cba848856ee409608fc7f45ebffec647dccc7ae811e73c3c' if stage == 'current_index' else None
        result = read(result_path, expected)
        checks[stage+'_start_preserved'] = all(result.get(k) == v for k,v in started.items())
        checks[stage+'_pass_identity'] = (result['status'] == 'PASS' and result['returncode'] == 0 and result['failure'] is None and result['controller_sha256'] == approval['controller_sha256'] and result['prepared_sha256'] == approval['prepared_sha256'])
        checks[stage+'_cpu_only_one_shot'] = result['cpu_affinity'] == [0,1] and result['gpu_visible'] is False and result['automatic_retry'] is False
        checks[stage+'_admitted_on_time'] = approval['reviewed_at_unix'] <= result['registered_unix'] < approval['valid_until_unix']
        checks[stage+'_limits_observed'] = result['invocation_seconds'] <= result['maximum_seconds'] and result['peak_aggregate_rss_bytes'] <= 4*2**30
        checks[stage+'_checkpoint_receipt_valid'] = result['finished_unix'] < checkpoint['valid_until_unix']
        argv = result['command']
        if stage != 'future_reference':
            checks[stage+'_command_pins'] = all(argv[argv.index(flag)+1] == value for flag,value in (
                ('--policy-path',prep['new_policy']),('--policy-sha256',prep['new_policy_sha256']),
                ('--registration-path',prep[key]),('--registration-sha256',prep[key+'_sha256']),
                ('--checkpoint-sha256',approval['checkpoint_sha256'])))
        else:
            checks[stage+'_command_pins'] = argv[argv.index('--plan-sha256')+1] == prep['future_plan_sha256'] and argv[argv.index('--registration-sha256')+1] == prep['future_preparation_registration_sha256']
        stage_results[stage] = result
    done = read(root/'COMPLETE.json')
    if key.startswith('index'):
        data = root/'metadata.sqlite'
        artifact = stage_results['current_index']['stdout_json']['artifact']
        checks['index_exact_population'] = done['scene_count'] == artifact['scene_count'] == 101592 and done['record_count'] == artifact['record_count'] == 304776
        checks['index_database_size_stat_only'] = data.stat().st_size == done['database_bytes'] == artifact['database_bytes'] == 381902848
        checks['index_identity_consistent'] = done['identity'] == artifact['identity'] and done['cache_identity'] == artifact['cache_identity'] and done['request_identity'] == artifact['request_identity']
        checks['index_no_payload_or_training'] = all(done[k] == artifact[k] == 0 for k in ('image_files_opened','tensor_files_opened','tensor_disk_bytes')) and done['training_admission'] is False
        checks['index_callbacks_recorded'] = stage_results['current_index']['stdout_json']['writer_callbacks'] > 0 and stage_results['current_index']['stdout_json']['writer_commit_callbacks'] > 0
        checks['index_builder_source_matches_pin'] = done['builder_source_sha256'] == next(p['sha256'] for p in reg['source_pins'] if p['path'].endswith('/st_world_metadata_index.py'))
    else:
        checks['future_build_complete_matches_return'] = done == stage_results['future_build']['stdout_json']['artifact']
        reference = stage_results['future_reference']['stdout_json']
        checks['future_reference_receipt_match'] = reference['status'] == 'REFERENCE_MATCH' and reference['mismatch_count'] == 0 and reference['scenes'] == 32 and reference['logs'] == 2
        checks['future_complete_binding'] = hashlib.sha256((root/'COMPLETE.json').read_bytes()).hexdigest() == reference['complete_sha256']
        checks['future_plan_binding'] = done['plan_identity'] == reference['plan_identity'] == future_plan['identity'] and reference['plan_sha256'] == prep['future_plan_sha256']
        checks['future_metadata_only'] = done['teacher_extractions'] == done['tensor_disk_bytes'] == 0 and done['training_admission'] is False and done['ego_tensor_decoded'] is False
        checks['future_reference_metadata_only'] = reference['teacher_extractions'] == reference['tensor_disk_bytes'] == reference['new_cache_files_written'] == 0 and reference['training_admission'] is False
        checks['future_callbacks_recorded'] = stage_results['future_build']['stdout_json']['writer_callbacks'] > 0 and stage_results['future_build']['stdout_json']['writer_commit_callbacks'] > 0

build, reference, index = (stage_results[s] for s in ('future_build','future_reference','current_index'))
checks['serial_execution'] = build['finished_unix'] < reference['registered_unix'] and reference['finished_unix'] < index['registered_unix']
checks['future_previous_cost_exact'] = reference['prior_seconds'] == build['invocation_seconds'] and reference['prior_charged_source_bytes'] == build['charged_source_bytes']
checks['future_combined_budget'] = build['invocation_seconds']+reference['invocation_seconds'] <= 900 and build['charged_source_bytes']+reference['charged_source_bytes'] <= 8*2**30
checks['future_each_read_budget'] = all(0 <= r['charged_source_bytes'] <= 4*2**30 for r in (build,reference))
baseline = policy['metadata_package']
old_ids = {r['claim_id'] for r in baseline['baseline_legacy_reservations']}
added = sum(r['reserved_bytes'] for r in policy['legacy_reservations'] if r['claim_id'] not in old_ids)
added += sum(sum(r[k] for k in ('output_bytes','staging_bytes','metadata_bytes')) for r in policy['claims'] if r['root'] not in baseline['baseline_claims'])
aggregate = policy['ledger_reserve_bytes'] + sum(r['reserved_bytes'] for r in policy['legacy_reservations']) + sum(sum(r[k] for k in ('output_bytes','staging_bytes','metadata_bytes')) for r in policy['claims'])
checks['one_package_1072MiB'] = added == migration['metadata_package_reserved_bytes'] == 1072*2**20 <= baseline['maximum_bytes'] == 4*2**30
checks['campaign_cap_and_floor_unchanged'] = aggregate <= policy['cache_cap_bytes'] == old_policy['cache_cap_bytes'] == 768*2**30 and policy['free_reserve_bytes'] == old_policy['free_reserve_bytes'] == 1024*2**30
fs = os.statvfs(ROOT)
free = fs.f_bavail*fs.f_frsize
checks['free_now_conservative_envelope'] = free > (1024+1983+768)*2**30
checks['review_did_not_change_ledger'] = hashlib.sha256(ledger_path.read_bytes()).hexdigest() == ledger_start_sha
ledger_files = list(ledger_path.parent.iterdir())
assert len(ledger_files) <= 16 and all(p.is_file() for p in ledger_files)
ledger_stat_bytes = sum(max(p.stat().st_size,p.stat().st_blocks*512) for p in ledger_files)
checks['migration_control_inside_ledger_reserve'] = ledger_stat_bytes <= policy['ledger_reserve_bytes']
assert all(checks.values()), checks
observed = {stage:{k: r[k] for k in ('invocation_seconds','peak_aggregate_rss_bytes','charged_source_bytes','registered_unix','finished_unix')} for stage,r in stage_results.items()}
result = {'schema':'st_actual_bounded_metadata_resource_closure_v1','status':'PASS_ACTUAL_BOUNDED_METADATA_RESOURCE_CLOSURE',
    'checked_at_unix':time.time(),'checks':checks,'observed':observed,'trees':trees,'pins':pins,
    'ledger_sha256':ledger_start_sha,'metadata_reserved_bytes':added,'aggregate_reserved_bytes':aggregate,
    'ledger_control_stat_bytes':ledger_stat_bytes,
    'filesystem_available_bytes':free,'future_combined_outer_seconds':build['invocation_seconds']+reference['invocation_seconds'],
    'future_combined_source_bytes':build['charged_source_bytes']+reference['charged_source_bytes'],
    'reviewer_large_database_payload_hashes':0,'reviewer_image_tensor_raw_opens':0,
    'formal_training_admission':False,'training_throughput_qualification':False,
    'future_full_population_admission':False,'database_content_correctness_revalidated':False}
(HERE/'RESOURCE_CLOSURE.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('trees','pins','checks')},sort_keys=True))
