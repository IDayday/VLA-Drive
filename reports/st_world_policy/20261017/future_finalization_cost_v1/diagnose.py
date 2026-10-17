"""Bounded stat-only SceneStream cost diagnosis; never imports the live writer."""
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import resource
import stat
import time

MIB = 1 << 20
GIB = 1 << 30
resource.setrlimit(resource.RLIMIT_AS, (512 * MIB, 512 * MIB))
resource.setrlimit(resource.RLIMIT_CPU, (90, 90))
os.sched_setaffinity(0, {10, 11})
started = time.monotonic()
SOURCE = Path('/mnt/project/ddp-foresight-artifacts/20260928/student_train_v1/index.json')
ROOT = Path('/mnt/project/st-world-policy-20261016/caches/full_future_metadata_train_v1')
OUT = Path(__file__).resolve().parent
read_inputs = []


def read_once(path):
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError('Input must be a regular file')
    if sum(x['bytes'] for x in read_inputs) + before.st_size > 32 * MIB:
        raise RuntimeError('Diagnostic logical input read budget exceeded')
    with path.open('rb') as stream:
        data = stream.read(before.st_size + 1)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise RuntimeError('Diagnostic input changed')
    if len(data) != before.st_size:
        raise RuntimeError('Input read size changed')
    read_inputs.append({'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    return json.loads(data)


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


plan = read_once(ROOT / 'PLAN.json')
rows = read_once(SOURCE)
assert isinstance(rows, list)
logs = list(dict.fromkeys(row['log'] for row in rows))
tokens = [row['token'] for row in rows]
checks = {
    'full_population': plan['scope'] == 'full_population',
    'scene_count_matches_plan': len(rows) == plan['selected_scene_count'],
    'exact_log_first_appearance_order_matches_plan': logs == plan['selected_logs'],
    'ordered_scene_ids_hash_matches_plan': identity(tokens) == plan['ordered_scene_ids_hash'],
    'original_index_semantic_identity_matches_plan': identity(rows) == plan['source']['source_index_identity'],
    'original_index_file_sha256_matches_plan': read_inputs[1]['sha256'] == plan['source']['source_index_file_sha256'],
    'plan_identity_matches': identity({k: v for k, v in plan.items() if k != 'identity'}) == plan['identity'],
}
assert all(checks.values()), checks
scene_counts = collections.Counter(row['log'] for row in rows)
run_counts = collections.Counter()
run_logs = []
previous = None
for row in rows:
    log = row['log']
    if log != previous:
        run_logs.append(log)
        run_counts[log] += 1
        previous = log
stats = []
sizes = {}
for log in logs:
    path = ROOT / 'log_shards' / (identity({'log': log}) + '.json')
    s = path.lstat()
    if not stat.S_ISREG(s.st_mode) or s.st_nlink != 1:
        raise RuntimeError('Unexpected shard file type')
    sizes[log] = s.st_size
    stats.append({'log': log, 'file': path.name, 'bytes': s.st_size,
                  'mtime_ns': s.st_mtime_ns, 'scenes': scene_counts[log],
                  'reads_per_stream': run_counts[log],
                  'bytes_per_stream': s.st_size * run_counts[log]})
unique_bytes = sum(sizes.values())
stream_bytes = sum(sizes[log] for log in run_logs)
limit = plan['limits']['max_source_read_bytes']
rate = plan['limits']['read_mib_per_second'] * MIB
budget_crossing = None
accumulated = unique_bytes  # The known ledger prepass, excluding every earlier read.
for pass_number in range(1, 4):
    for run_number, log in enumerate(run_logs, 1):
        if accumulated + sizes[log] > limit:
            budget_crossing = {'pass_number': pass_number, 'run_number_in_pass': run_number,
                               'bytes_before_attempt': accumulated,
                               'attempted_read_bytes': sizes[log],
                               'assumption': 'Only ledger prepass + three SceneStreams; all earlier reads set to zero'}
            break
        accumulated += sizes[log]
    if budget_crossing:
        break
result = {
    'schema': 'st_future_finalization_cost_diagnosis_v1',
    'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'status': 'READ_ONLY_COST_PREDICTION_NOT_RUNTIME_FAILURE',
    'writer_source_sha': '838dd21f7a16a79086a44bc63be056d6a750570e',
    'report_base_sha': '71942071b5f0d10fccd69a8fa79395edebd73e45',
    'read_inputs': read_inputs,
    'checks': checks,
    'registered_limits': plan['limits'],
    'source': {'scenes': len(rows), 'unique_logs': len(logs),
               'adjacent_log_switches': len(run_logs) - 1,
               'read_shard_calls_per_stream': len(run_logs),
               'same_log_adjacencies': len(rows) - len(run_logs)},
    'cost_bytes': {
        'one_unique_shard_read_each': unique_bytes,
        'one_stream': stream_bytes,
        'one_stream_redundant_vs_one_unique_read_each': stream_bytes - unique_bytes,
        'three_streams': 3 * stream_bytes,
        'three_streams_redundant_vs_one_unique_read_each_total': 3 * stream_bytes - unique_bytes,
        'three_streams_redundant_vs_one_unique_read_each_per_pass': 3 * (stream_bytes - unique_bytes),
        'ledger_prepass_plus_three_streams': unique_bytes + 3 * stream_bytes,
        'earlier_read_headroom_after_ledger_and_three_streams': limit - unique_bytes - 3 * stream_bytes,
        'raw_log_file_bytes_from_plan_only': sum(row['bytes'] for row in plan['raw_log_stats']),
    },
    'amplification': {
        'one_stream_vs_one_read_each': stream_bytes / unique_bytes,
        'three_streams_vs_one_read_each': 3 * stream_bytes / unique_bytes,
        'three_streams_vs_registered_read_budget': 3 * stream_bytes / limit,
    },
    'rate_limit_time_lower_bounds_s': {
        'one_stream': stream_bytes / rate,
        'three_streams': 3 * stream_bytes / rate,
        'ledger_prepass_plus_three_streams': (unique_bytes + 3 * stream_bytes) / rate,
    },
    'budget_crossing_even_if_every_earlier_read_is_zero': budget_crossing,
    'shard_content_reads': 0,
    'raw_image_teacher_reads': 0,
    'shard_stat_count': len(stats),
    'shard_stats': stats,
    'limitations': [
        'Shard contents and validity were not checked; path, size and mtime only.',
        'Predictions assume shard sizes stay immutable and all three passes reach completion.',
        'Read allowance counts logical input bytes even on OS page-cache hits.',
        'No runtime CPU/decode/hash benchmark and no live writer failure observation.',
        'Earlier raw/image/current/ego/control reads and later manifest/receipt reads/writes are excluded from the finalization lower bound.',
    ],
}
result['diagnostic_cost'] = {
    'explicit_input_bytes_read': sum(x['bytes'] for x in read_inputs),
    'max_allowed_explicit_input_bytes': 32 * MIB,
    'wall_seconds': time.monotonic() - started,
    'cpu_seconds': resource.getrusage(resource.RUSAGE_SELF).ru_utime + resource.getrusage(resource.RUSAGE_SELF).ru_stime,
    'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    'affinity': sorted(os.sched_getaffinity(0)),
}
encoded = (json.dumps(result, sort_keys=True, indent=2) + '\n').encode()
if len(encoded) > MIB:
    raise RuntimeError('Report file would exceed 1 MiB')
(OUT / 'evidence.json').write_bytes(encoded)
print(json.dumps({k: v for k, v in result.items() if k != 'shard_stats'}, indent=2, sort_keys=True))
