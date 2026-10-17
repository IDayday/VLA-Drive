"""Finite read-only closure: two small metadata DBs, exact controls, one ledger.

No project traversal, image/raw-log/tensor opens, runtime imports or live writes.
Only the sibling EVIDENCE.json in this review worktree is written.
"""
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import sqlite3
import time
import zlib

ROOT = Path('/mnt/project/st-world-policy-20261016')
HERE = Path(__file__).parent
PREP = ROOT / 'audits/indexed_composition_preparation_v1/PREPARED.json'
POLICY_SHA = 'b7062c4da878519c435db2e88b899bd476e5cd6a0e9f16dca191fe61522bb831'
BEFORE_LEDGER_SHA = 'ddeff34b7c9dc09d44dc32d14f6ba876a6f6343bb3b279d8f14fd00d78bb85dd'
EXPECTED = {
    'request_index': ('future32_request_metadata_index_v1', 32, 294,
                      '54cf472cabc61234063ab9750d4bd73d3e420560a6cccafdb4c18278ce2963e0'),
    'fit_index': ('original_pca_fit64_metadata_index_v1', 64, 957,
                  '5508a350b2777d909dc83297e997f43c02b63108c23b766f02ead5428abb5373'),
}
pins, checks, stages = [], {}, {}
started = time.monotonic()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def stamp(path):
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def read(path, expected=None, database=False):
    path = Path(path)
    assert path.resolve() == path and path.is_file() and not path.is_symlink()
    before = stamp(path)
    assert before[2] <= (2 << 20 if database else 1 << 20)
    raw = path.read_bytes()
    assert stamp(path) == before and len(raw) == before[2]
    sha = hashlib.sha256(raw).hexdigest()
    if expected is not None:
        assert sha == expected, (path, sha, expected)
    pins.append(dict(path=str(path), bytes=len(raw), sha256=sha, version=list(before)))
    return raw if database else json.loads(raw)


def decoded_rows(db, table, order):
    result = []
    for body, expected_sha in db.execute(f'SELECT body,sha FROM {table} ORDER BY {order}'):
        assert len(body) <= 65536
        decoder = zlib.decompressobj()
        raw = decoder.decompress(body, 65537)
        assert len(raw) <= 65536 and decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail
        assert hashlib.sha256(raw).hexdigest() == expected_sha
        value = json.loads(raw)
        assert canonical(value) == raw
        result.append(value)
    return result


approval = read(HERE.parent / 'RESOURCE_APPROVAL.json',
                '0da15ed84073e9e7450e07f4c691f455e1c753f682011dae4c07a5a7256e95c4')
prep = read(PREP, 'b2afa6c88c1b26764f7226efb87340f387d25fa20f7c7ba6e8bd32164042fd19')
policy = read(prep['policy'], POLICY_SHA)
old_policy = read(prep['base_policy'], prep['base_policy_sha256'])
ledger_path = Path(prep['state_dir']) / 'CLAIMS.json'
ledger = read(ledger_path)
ledger_sha = pins[-1]['sha256']
checkpoint = read(approval['checkpoint_receipt'], approval['checkpoint_sha256'])

checks['v7_exact_core_unchanged'] = all(policy.get(k) == v == old_policy.get(k) for k, v in ledger['core'].items())
checks['v7_only_two_declarations_added'] = (
    {k: v for k, v in policy.items() if k != 'claims'} ==
    {k: v for k, v in old_policy.items() if k != 'claims'} and
    policy['claims'][:len(old_policy['claims'])] == old_policy['claims'] and
    len(policy['claims']) == len(old_policy['claims']) + 2)
before = json.loads(json.dumps(ledger))
for item in prep['stages'].values():
    del before['claims'][item['output_root']]
reconstructed = (json.dumps(before, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
checks['all_old_ledger_rows_and_core_byte_exact'] = hashlib.sha256(reconstructed).hexdigest() == BEFORE_LEDGER_SHA
checks['same_ledger_lock_and_no_core_migration'] = (
    policy['state_dir'] == old_policy['state_dir'] == str(ledger_path.parent) and
    ledger_path.with_name('CLAIMS.lock').is_file() and
    before['metadata_migrations'] == ledger['metadata_migrations'])

for stage, (name, nscenes, nrecords, result_sha) in EXPECTED.items():
    root = ROOT / 'caches' / name
    entry = prep['stages'][stage]
    reg = read(entry['registration'], entry['registration_sha256'])
    result = read(root / 'CONTROL_BUILD_RESULT.json', result_sha)
    start = read(root / 'CONTROL_BUILD_STARTED.json')
    complete = read(root / 'COMPLETE.json')
    admission = read(root / 'METADATA_ADMISSION.json')
    record = ledger['claims'][str(root)]
    declaration = next(x for x in policy['claims'] if x['root'] == str(root))
    checks[stage + '_pins_and_admission'] = (
        admission['claim'] == record['claim'] == declaration and
        admission['policy_sha256_at_admission'] == POLICY_SHA and
        admission['registration_sha256'] == entry['registration_sha256'] and
        result['controller_sha256'] == approval['controller_sha256'] and
        result['prepared_sha256'] == approval['prepared_sha256'] and
        all(result[k] == v for k, v in start.items()))
    command = result['command']
    flags = dict(zip(command[4::2], command[5::2]))
    checks[stage + '_checkpoint_bound'] = (
        flags['--checkpoint-receipt'] == approval['checkpoint_receipt'] and
        flags['--checkpoint-sha256'] == approval['checkpoint_sha256'] and
        result['finished_unix'] < checkpoint['valid_until_unix'] and
        checkpoint['remaining_bytes'] == 1983 << 30)
    checks[stage + '_finite_pass'] = (
        result['status'] == 'PASS' and result['returncode'] == 0 and result['failure'] is None and
        result['automatic_retry'] is False and result['gpu_visible'] is False and
        result['cpu_affinity'] == [0, 1] and
        0 < result['invocation_seconds'] <= result['maximum_seconds'] == 180 and
        0 < result['peak_aggregate_rss_bytes'] <= result['max_aggregate_rss_bytes'] == 2 << 30 and
        result['finished_unix'] < approval['valid_until_unix'])
    guarded = result['stdout_json']
    artifact = guarded['artifact']
    checks[stage + '_callbacks_executed'] = (
        guarded['status'] == 'GUARDED_METADATA_RETURN' and guarded['checks'] > 0 and
        guarded['writer_callbacks'] > 0 and guarded['writer_commit_callbacks'] > 0 and
        guarded['registration_sha256'] == entry['registration_sha256'])
    checks[stage + '_receipt_identity'] = (
        complete['status'] == 'COMPLETE' and complete['schema'] == 'st_teacher_metadata_index_v1' and
        complete['identity'] == digest({k: v for k, v in complete.items() if k != 'identity'}) and
        complete['coverage'] == 'exact_original_manifest' and complete['training_admission'] is False and
        complete['scene_count'] == nscenes and complete['record_count'] == nrecords and
        all(artifact[k] == complete[k] for k in artifact))
    checks[stage + '_reported_zero_images_and_tensors'] = all(
        artifact[k] == complete[k] == 0 for k in ('image_files_opened', 'tensor_files_opened', 'tensor_disk_bytes'))
    for ref in reg['source_pins']:
        # Source pins are small text files, hashed without importing executable code.
        read(ref['path'], ref['sha256'], database=True)
    checks[stage + '_source_pin'] = complete['builder_source_sha256'] == next(
        x['sha256'] for x in reg['source_pins'] if x['path'].endswith('/st_world_metadata_index.py'))
    dependencies = {k: read(ref['path'], ref['sha256']) for k, ref in complete['dependencies'].items()}
    request = dependencies['requests']
    checks[stage + '_original_request_identity'] = (
        request['identity'] == complete['request_identity'] ==
        digest({k: v for k, v in request.items() if k != 'identity'}) and
        complete['manifest_header'] == {k: v for k, v in request.items() if k != 'scenes'})
    raw = read(root / 'metadata.sqlite', complete['database_sha256'], database=True)
    checks[stage + '_database_bytes'] = len(raw) == complete['database_bytes']
    db = sqlite3.connect((root / 'metadata.sqlite').as_uri() + '?mode=ro&immutable=1', uri=True)
    db.execute('PRAGMA query_only=ON')
    db.execute('PRAGMA cache_size=-2048')
    db.execute('PRAGMA mmap_size=0')
    checks[stage + '_database_integrity'] = db.execute('PRAGMA quick_check').fetchall() == [('ok',)]
    scenes = decoded_rows(db, 'scenes', 'pos')
    records = decoded_rows(db, 'records', 'pos' if stage == 'fit_index' else 'key')
    checks[stage + '_all_scene_rows_exact_original_order'] = scenes == request['scenes'] and len(scenes) == nscenes
    expected_records = {}
    contract_id = digest(complete['contract'])
    for scene in request['scenes']:
        for frame in scene['frames']:
            for ref in frame['views']:
                if ref is not None:
                    key = digest(dict(image_sha256=ref['sha256'], teacher_contract=contract_id))
                    expected_records.setdefault(key, dict(key=key, image=ref))
    expected_records = [expected_records[k] for k in sorted(expected_records)]
    checks[stage + '_all_metadata_records_exact'] = records == expected_records and len(records) == nrecords
    if stage == 'fit_index':
        original = dependencies['identity']
        checks[stage + '_original_cache_identity_and_records'] = (
            original['identity'] == complete['cache_identity'] == 'fae6ed77041cdd318e647a9f44d907ffe63c62487805b0d683e9b247b6d7714d' and
            original['identity'] == digest({k: v for k, v in original.items() if k != 'identity'}) and
            original['records'] == records and
            complete['cache_header'] == {k: v for k, v in original.items() if k != 'records'})
        checks[stage + '_original_shard_receipts_exact'] = db.execute('SELECT shard,sha FROM receipts ORDER BY shard').fetchall() == [
            (x['shard'], x['receipt_sha256']) for x in dependencies['complete']['shards']]
    else:
        checks[stage + '_no_fake_teacher_cache'] = complete['cache_identity'] is None and db.execute('SELECT count(*) FROM receipts').fetchone()[0] == 0
    db.close()
    files = []
    for path in sorted(root.iterdir()):
        assert path.is_file() and not path.is_symlink()
        st = path.stat()
        files.append(dict(name=path.name, bytes=st.st_size, allocated_bytes=st.st_blocks * 512))
    files_observed = sum(max(x['bytes'], x['allocated_bytes']) for x in files)
    root_stat = root.stat()
    root_charge = max(root_stat.st_size, root_stat.st_blocks * 512)
    observed = files_observed + root_charge
    checks[stage + '_final_files_only_metadata'] = {x['name'] for x in files} == {
        'BUILD.lock', 'COMPLETE.json', 'CONTROL_BUILD_RESULT.json', 'CONTROL_BUILD_STARTED.json',
        'METADATA_ADMISSION.json', 'metadata.sqlite'}
    checks[stage + '_claim_and_disk_caps'] = (
        record['charged_bytes'] == 32 << 20 and observed <= 32 << 20 and
        0 <= record['observed_bytes'] <= record['charged_bytes'] and
        guarded['peak_disk_bytes'] <= 32 << 20 and complete['max_disk_bytes'] == 28 << 20)
    stages[stage] = dict(root=str(root), result_sha256=result_sha, identity=complete['identity'],
                         scene_count=nscenes, record_count=nrecords, database_bytes=len(raw),
                         directory_charged_bytes=observed, files_charged_bytes=files_observed,
                         files_logical_bytes=sum(x['bytes'] for x in files),
                         files_allocated_bytes=sum(x['allocated_bytes'] for x in files),
                         root_logical_bytes=root_stat.st_size, root_allocated_bytes=root_stat.st_blocks*512,
                         root_charged_bytes=root_charge, guard_peak_disk_bytes=guarded['peak_disk_bytes'],
                         invocation_seconds=result['invocation_seconds'], peak_aggregate_rss_bytes=result['peak_aggregate_rss_bytes'],
                         guard_checks=guarded['checks'], writer_callbacks=guarded['writer_callbacks'],
                         writer_commit_callbacks=guarded['writer_commit_callbacks'], started_unix=result['started_unix'],
                         finished_unix=result['finished_unix'], files=files, ledger_record=record)

checks['serial_invocations'] = stages['request_index']['finished_unix'] < stages['fit_index']['started_unix']
total = lambda x: sum(x[k] for k in ('output_bytes', 'staging_bytes', 'metadata_bytes'))
package = policy['metadata_package']
legacy_baseline = {x['claim_id'] for x in package['baseline_legacy_reservations']}
metadata_bytes = sum(total(x) for x in policy['claims'] if x['root'] not in package['baseline_claims']) + sum(
    x['reserved_bytes'] for x in policy['legacy_reservations'] if x['claim_id'] not in legacy_baseline)
aggregate_bytes = policy['ledger_reserve_bytes'] + sum(x['reserved_bytes'] for x in policy['legacy_reservations']) + sum(total(x) for x in policy['claims'])
checks['metadata_and_campaign_caps_preserved'] = (
    metadata_bytes == 1191182336 <= package['maximum_bytes'] == 4 << 30 and
    aggregate_bytes == 694096380774 <= policy['cache_cap_bytes'] == 768 << 30 and
    policy['free_reserve_bytes'] == 1024 << 30)
free = shutil.disk_usage(ROOT).free
checks['current_conservative_free_floor'] = free >= ((768 + 1024 + 1983) << 30)
checks['ledger_unchanged_by_review'] = hashlib.sha256(ledger_path.read_bytes()).hexdigest() == ledger_sha
failed = [k for k, v in checks.items() if not v]
evidence = dict(schema='st_index_followup_actual_closure_v1', status='PASS' if not failed else 'FAIL',
                checked_at_unix=time.time(), checks=checks, failed=failed, pins=pins, stages=stages,
                ledger_path=str(ledger_path), ledger_sha256=ledger_sha, prior_ledger_sha256=BEFORE_LEDGER_SHA,
                policy_path=prep['policy'], policy_sha256=POLICY_SHA, metadata_reserved_bytes=metadata_bytes,
                all_declared_bytes=aggregate_bytes, available_bytes=free, checkpoint_remaining_bytes=checkpoint['remaining_bytes'],
                checker_seconds=time.monotonic()-started, checker_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                reviewer_image_raw_tensor_opens=0, real_builds=0, live_ledger_mutations=0,
                formal_training_admission=False, throughput_approval=False)
(HERE / 'EVIDENCE.json').write_text(json.dumps(evidence, sort_keys=True, indent=2, allow_nan=False) + '\n')
assert not failed, failed
print(json.dumps(dict(status=evidence['status'], checks=len(checks), ledger_sha256=ledger_sha,
                      stages=stages, checker_seconds=evidence['checker_seconds'],
                      checker_peak_rss_bytes=evidence['checker_peak_rss_bytes']), sort_keys=True))
