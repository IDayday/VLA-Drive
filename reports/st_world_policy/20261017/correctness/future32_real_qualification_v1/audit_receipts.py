"""Read <=2 MiB of metadata/code; never open raw logs, images or tensors."""
import hashlib
import json
from pathlib import Path
import stat
import subprocess

ROOT = Path('/mnt/project/st-world-policy-20261016/caches/future_metadata_qualification32_v2')
AUDIT = Path('/mnt/project/st-world-policy-20261016/audits/metadata_migration_preparation_v1')
OUT = Path(__file__).parent
CAP = 2 << 20
PRIOR_INSPECTION_BYTES = 36082
consumed = PRIOR_INSPECTION_BYTES
loaded = {}
evidence = []


def load(path, expected=None):
    global consumed
    path = Path(path)
    if str(path) not in loaded:
        before = path.lstat()
        assert stat.S_ISREG(before.st_mode) and before.st_nlink == 1
        assert path.suffix in ('.json', '.py')
        assert consumed + before.st_size + 1 <= CAP
        with path.open('rb') as stream: data = stream.read(before.st_size + 1)
        consumed += len(data)
        after = path.lstat()
        stamp = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        assert stamp(before) == stamp(after) and len(data) == before.st_size
        digest = hashlib.sha256(data).hexdigest()
        value = json.loads(data) if path.suffix == '.json' else None
        loaded[str(path)] = (value, digest)
        evidence.append({'path':str(path), 'bytes':len(data), 'sha256':digest})
    value, digest = loaded[str(path)]
    if expected is not None: assert digest == expected, str(path)
    return value, digest


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def signed(value):
    assert value['identity'] == identity({k:v for k,v in value.items() if k != 'identity'})


def command_flags(command):
    return dict(zip(command[3::2], command[4::2]))


plan, plan_sha = load(ROOT/'PLAN.json'); signed(plan)
done, complete_sha = load(ROOT/'COMPLETE.json')
build, build_sha = load(ROOT/'CONTROL_FUTURE_BUILD_RESULT.json')
ref, ref_sha = load(ROOT/'CONTROL_FUTURE_REFERENCE_RESULT.json')
build_start, _ = load(ROOT/'CONTROL_FUTURE_BUILD_STARTED.json')
ref_start, _ = load(ROOT/'CONTROL_FUTURE_REFERENCE_STARTED.json')
admission, _ = load(ROOT/'METADATA_ADMISSION.json')
prepared, prepared_sha = load(AUDIT/'PREPARED_V6.json', build['prepared_sha256'])
registration, registration_sha = load(prepared['future_storage_registration'], prepared['future_storage_registration_sha256'])
prep_reg, prep_reg_sha = load(prepared['future_preparation_registration'], prepared['future_preparation_registration_sha256']); signed(prep_reg)
policy, policy_sha = load(prepared['new_policy'], prepared['new_policy_sha256'])
assert build['prepared_sha256'] == ref['prepared_sha256'] == prepared_sha
assert prepared['future_plan_sha256'] == plan_sha == prep_reg['plan_sha256']
assert prep_reg['plan_identity'] == plan['identity'] == done['plan_identity']
assert prep_reg['identity'] == done['registration_identity']
assert registration['output_root'] == admission['claim']['root'] == str(ROOT)
assert admission['policy_sha256_at_admission'] == policy_sha
assert admission['registration_sha256'] == registration_sha
assert sum(registration['storage_requirements'].values()) == 32 << 20
assert registration['limits']['max_disk_bytes'] == plan['limits']['max_metadata_bytes'] == 32 << 20
assert plan['limits']['workers'] == 1
assert plan['limits']['max_source_read_bytes'] == 4 << 30
assert plan['limits']['max_seconds'] == 450
assert plan['limits']['read_mib_per_second'] == 64
assert done['status'] == 'COMPLETE' and done['scope'] == 'qualification'
assert done == build['stdout_json']['artifact']
for row, start in ((build, build_start), (ref, ref_start)):
    assert row['status'] == 'PASS' and row['returncode'] == 0 and row['failure'] is None
    assert row['command'] == start['command'] and row['controller_sha256'] == start['controller_sha256']
    assert row['automatic_retry'] is False and row['gpu_visible'] is False
    assert row['cpu_affinity'] == [0,1]
    assert row['charged_source_bytes'] <= 4 << 30
    assert row['invocation_seconds'] <= 450 and row['peak_aggregate_rss_bytes'] <= 4 << 30
assert build['prior_charged_source_bytes'] == 0 and build['prior_seconds'] == 0
assert ref['prior_charged_source_bytes'] == build['charged_source_bytes']
assert ref['prior_seconds'] == build['invocation_seconds']
assert build['finished_unix'] < ref['registered_unix']
assert build['stdout_json']['writer_callbacks'] > 0
assert build['stdout_json']['writer_commit_callbacks'] > 0

# CLI dependencies are exact receipts, not arbitrary latest paths.
build_flags = dict(zip(build['command'][4::2], build['command'][5::2]))
assert build_flags['--registration-sha256'] == registration_sha
assert build_flags['--policy-sha256'] == policy_sha
checkpoint, _ = load(build_flags['--checkpoint-receipt'], build_flags['--checkpoint-sha256'])
assert checkpoint['status'] == 'CURRENT'
assert checkpoint['checked_at_unix'] <= build['registered_unix'] < checkpoint['valid_until_unix']
reference_flags = command_flags(ref['command'])
assert reference_flags['--plan-sha256'] == plan_sha
assert reference_flags['--registration-sha256'] == prep_reg_sha
assert reference_flags['--complete-sha256'] == complete_sha

runtime = Path(prepared['runtime'])
runtime_commit = subprocess.check_output(['git','-C',str(runtime),'rev-parse','HEAD'],text=True).strip()
assert runtime_commit == '1162b955b3bf9abb6728823a84a50c38e1a19bb8'
assert not subprocess.check_output(['git','-C',str(runtime),'status','--porcelain'],text=True).strip()
for relative, digest in plan['implementation'].items(): load(runtime/relative, digest)
for source in registration['source_pins']: load(source['path'], source['sha256'])
result = ref['stdout_json']
load(runtime/'tools/st_world_policy/compare_future_metadata.py', result['comparator_source_sha256'])
assert result['comparator_source_sha256'] == '3ddc07679451d6843ff3d0f75790564127f20d37f2ea83fff64ccc84a70f6949'
_, guard_sha = load(runtime/'tools/st_world_policy/metadata_storage.py')
assert guard_sha == 'c75a8ce3b0f4424d005e1dc2caa38d9d95df11047900c3ed00bf4d45e12f6867'
for relative, digest in result['reference_sources'].items(): load(runtime/relative, digest)

manifest, requests_sha = load(ROOT/'TEACHER_REQUESTS.json', done['files']['TEACHER_REQUESTS.json']); signed(manifest)
ledger, ledger_sha = load(ROOT/'RAW_LOG_PROVENANCE.json', done['files']['RAW_LOG_PROVENANCE.json']); signed(ledger)
shards, _ = load(ROOT/'SHARD_RECEIPTS.json', done['files']['SHARD_RECEIPTS.json'])
assert manifest['identity'] == done['request_identity']
assert manifest['source_logs_ledger']['sha256'] == ledger_sha
assert manifest['source_logs_ledger']['identity'] == ledger['identity']
assert manifest['source_logs_ledger']['path'] == str(ROOT/'RAW_LOG_PROVENANCE.json')
rows = plan['selected_rows']; tokens = [row['token'] for row in rows]
assert [row['token'] for row in manifest['scenes']] == tokens
assert plan['ordered_scene_ids_hash'] == identity(tokens)
assert plan['selection_identity'] == identity(rows)
assert len(set(tokens)) == len(tokens) == 32
assert list(dict.fromkeys(row['log'] for row in rows)) == plan['selected_logs']
assert len(plan['selected_logs']) == 2
assert len(shards['shards']) == 2
scene_map = {row['token']:row for row in manifest['scenes']}
for record, coverage in zip(shards['shards'], ledger['logs']):
    assert record['log'] == coverage['log']
    shard, _ = load(record['path'], record['sha256']); signed(shard)
    selected = [row for row in rows if row['log'] == record['log']]
    assert shard['selected'] == selected
    assert shard['scenes'] == [scene_map[row['token']] for row in selected]
    assert shard['raw_log'] == coverage
    assert coverage['scenes'] == len(selected)
    assert coverage['tokens_sha256'] == identity([row['token'] for row in selected])
old, old_sha = load('/mnt/project/st-world-policy-20261016/caches/future_metadata_qualification32_v1/PLAN.json',
    'd1a0ebe7279c99edfa67a72b7f3a9fc1aca0c3a22ea10a7fe0c9c36aa3a3ae40')
assert old['selected_rows'] == rows and old['selected_logs'] == plan['selected_logs']
assert old['source'] == plan['source'] and old['original_selection'] == plan['original_selection']

assert result['status'] == 'REFERENCE_MATCH' and result['mismatch_count'] == 0
assert result['first_mismatches'] == [] and result['checked_leaf_fields'] == 3532
assert result['plan_sha256'] == plan_sha and result['complete_sha256'] == complete_sha
assert result['requests_sha256'] == requests_sha and result['ledger_sha256'] == ledger_sha
assert result['scenes'] == 32 and result['logs'] == 2 and result['time_view_slots'] == 480
assert result['missing_view_slots'] == 0
assert manifest['times_s'] == plan['target_times_s'] == [0.,.5,1.,2.,4.]
assert manifest['view_names'] == ['CAM_F0','CAM_L0','CAM_R0']
assert all(len(s['frames']) == 5 and all(len(frame['views']) == 3 for frame in s['frames']) for s in manifest['scenes'])
assert result['teacher_extractions'] == done['teacher_extractions'] == 0
assert result['tensor_disk_bytes'] == done['tensor_disk_bytes'] == 0
assert result['training_admission'] is False and done['training_admission'] is False
assert result['new_cache_files_written'] == 0
read_bytes = build['charged_source_bytes'] + ref['charged_source_bytes']
seconds = build['invocation_seconds'] + ref['invocation_seconds']
assert read_bytes == result['combined_observed_source_bytes'] == 441607148
assert read_bytes <= 8 << 30 and seconds <= 900
assert result['build_invocation_cost'] == done['cost']
assert result['reference_invocation_cost']['source_bytes_read'] == ref['charged_source_bytes']

files = [p for p in ROOT.rglob('*') if p.is_file()]
assert all(p.suffix == '.json' or p.name in ('BUILD.lock','PLAN.lock') for p in files)
root_allocated = sum(p.lstat().st_blocks*512 for p in [ROOT,*ROOT.rglob('*')])
assert root_allocated < 32 << 20
summary = {'status':'PASS_REAL_METADATA_QUALIFICATION_32_ONLY', 'runtime_commit':runtime_commit,
    'plan_sha256':plan_sha,'complete_sha256':complete_sha,'requests_sha256':requests_sha,
    'ledger_sha256':ledger_sha,'fixed_previous_plan_sha256':old_sha,'same_original_order':True,
    'scenes':32,'logs':2,'time_view_slots':480,'checked_leaf_fields':3532,'mismatches':0,'missing_view_slots':0,
    'build_outer_seconds':build['invocation_seconds'],'reference_outer_seconds':ref['invocation_seconds'],
    'combined_outer_seconds':seconds,'combined_charged_source_bytes':read_bytes,
    'build_source_bytes':build['charged_source_bytes'],'reference_source_bytes':ref['charged_source_bytes'],
    'peak_outer_rss_bytes':max(build['peak_aggregate_rss_bytes'],ref['peak_aggregate_rss_bytes']),
    'metadata_root_allocated_bytes_at_review':root_allocated,'guard_checks':build['stdout_json']['checks'],
    'writer_callbacks':build['stdout_json']['writer_callbacks'],'publication_callbacks':build['stdout_json']['writer_commit_callbacks'],
    'teacher_extractions':0,'tensor_disk_bytes':0,'training_admission':False,
    'review_metadata_and_code_bytes_read_including_prior_inspection':consumed,'review_read_limit_bytes':CAP,
    'review_raw_image_tensor_reads':0,'review_new_qualification_runs':0,
    'resource_approval':'NOT_SIGNED_BY_CORRECTNESS_REVIEWER','full_population_and_long_throughput':'UNTESTED'}
(OUT/'REVIEW.json').write_text(json.dumps(summary,indent=2)+'\n')
(OUT/'EVIDENCE_HASHES.json').write_text(json.dumps({'files':evidence,'inspection_before_script_bytes':PRIOR_INSPECTION_BYTES,
    'total_read_bytes':consumed,'cap_bytes':CAP},indent=2)+'\n')
print(json.dumps(summary,sort_keys=True))
