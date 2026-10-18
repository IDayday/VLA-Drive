#!/usr/bin/env python3
"""Bounded CPU-only analysis of the fixed C1 full-TRAIN feedback ledger.

The numerical functions below are extracted from the immutable one-ledger
analysis without importing its top-level RLIMIT_CPU or execution side effects.
Only tie_counts is generalized from 32 scenes to the validated batch size.
"""
from __future__ import annotations
import argparse
import collections
from contextlib import contextmanager
import fcntl
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import resource
import signal
import stat
import sys
import time

np = torch = None

def load_numeric():
    global np, torch
    if torch is None:
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only CUDA visibility required before import')
        import numpy as numpy_module
        import torch as torch_module
        np, torch = numpy_module, torch_module
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    require(not torch.cuda.is_initialized(), 'CUDA initialization forbidden')

METRICS = ('score', 'no_at_fault_collisions', 'drivable_area_compliance',
           'ego_progress', 'time_to_collision_within_bound', 'comfort', 'driving_direction_compliance')


def require(value, message):
    if not value:
        raise ValueError(message)

def digest(data):
    return hashlib.sha256(data).hexdigest()

def identity(value):
    return digest(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())

def array_hash(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256(json.dumps({'dtype': a.dtype.str, 'shape': a.shape}, sort_keys=True).encode())
    h.update(a.tobytes())
    return h.hexdigest()

def bin_of(index):
    require(type(index) is int and 0 <= index <= 40, 'Invalid physical state index')
    return max(0, (index - 1) // 5)

def process_profile(record):
    """Time signatures ignore object identity and repeated contacts at one time."""
    c = record['consequences']
    road = c['road']
    require(road['valid'] is True and len(road['native_flag']) == 41, 'Invalid road trace')
    require(np.asarray(road['native_flag']).dtype == np.bool_, 'Road trace lost bool semantics')
    d_times = tuple(int(x) for x in np.flatnonzero(road['native_flag']))
    d_bins = [False] * 8
    for t in d_times:
        d_bins[bin_of(t)] = True
    require(d_bins == road['segment_any'] and bool(d_times) == road['any'] == road['event'], 'Road aggregation mismatch')
    require(road['first_time_seconds'] == (d_times[0] * .1 if d_times else None), 'Road first time mismatch')

    c_events = [e for e in c['collision_events'] if e['at_fault']]
    c_signature = tuple(sorted({(int(e['state_index']), float(e['nc_contribution'])) for e in c_events}))
    c_bins = [1.] * 8
    for t, severity in c_signature:
        require(severity in (0., .5), 'At-fault severity changed')
        c_bins[bin_of(t)] = min(c_bins[bin_of(t)], severity)
    for e in c_events:
        require(e['state_time_seconds'] == int(e['state_index']) * .1, 'Collision physical time mismatch')
    cs = c['collision_summary']
    require(cs['valid'] is True and cs['segment_nc_min'] == c_bins and cs['nc_min'] == min(c_bins), 'NC aggregation mismatch')
    require(cs['event'] == bool(c_signature), 'NC event flag mismatch')
    require(cs['first_at_fault_time_seconds'] == (min(t for t, _ in c_signature) * .1 if c_signature else None), 'NC first time mismatch')

    t_events = [e for e in c['ttc_events'] if e['violation']]
    t_times = tuple(sorted({int(e['state_index']) for e in t_events}))
    t_projection = tuple(sorted({(int(e['state_index']), int(e['projection_offset_steps']), int(e['observation_index'])) for e in t_events}))
    t_bins = [False] * 8
    for t in t_times:
        t_bins[bin_of(t)] = True
    for e in t_events:
        require(e['projection_offset_steps'] in (0, 3, 6, 9), 'TTC projection contract changed')
        require(e['observation_index'] == e['state_index'] + e['projection_offset_steps'], 'TTC observation identity mismatch')
        require(e['state_time_seconds'] == e['state_index'] * .1 and
                e['observation_time_seconds'] == e['observation_index'] * .1, 'TTC time semantics changed')
    ts = c['ttc_summary']
    require(ts['valid'] is True and ts['segment_any'] == t_bins and ts['any'] == bool(t_times) == ts['event'], 'TTC aggregation mismatch')
    require(ts['first_state_time_seconds'] == (min(t_times) * .1 if t_times else None), 'TTC first time mismatch')

    m = record['metrics']
    terminal = (1. - float(bool(d_times)), min(c_bins), 1. - float(bool(t_times)))
    require(terminal == (m['drivable_area_compliance'], m['no_at_fault_collisions'], m['time_to_collision_within_bound']), 'Event-to-metric mismatch')
    return {'terminal': terminal, 'time': (d_times, c_signature, t_times),
            'bins': (tuple(d_bins), tuple(c_bins), tuple(t_bins)), 'ttc_projection': t_projection}

def advantages(rewards):
    x = torch.tensor(np.asarray(rewards), dtype=torch.float32).reshape(-1, 4)
    return (x - x.mean(dim=1, keepdim=True)).numpy()

def decompose(metrics):
    # Names here describe algebraic products, not independent causal factors.
    nc, dac, ep, ttc, comfort = (np.asarray(metrics[k], np.float64).reshape(-1, 4) for k in
        ('no_at_fault_collisions', 'drivable_area_compliance', 'ego_progress', 'time_to_collision_within_bound', 'comfort'))
    gate = nc * dac
    terms = np.stack((5 * gate * ep / 12, 5 * gate * ttc / 12, 2 * gate * comfort / 12), axis=-1)
    A = advantages(metrics['score']).astype(np.float64)
    parts = terms - terms.mean(axis=1, keepdims=True)
    residual = A - parts.sum(axis=-1)
    all_parts = np.concatenate((parts, residual[..., None]), axis=-1)
    V = float((A * A).sum())
    sums = np.einsum('bg,bgk->k', A, all_parts)
    return A, terms, {'advantage_squared_sum': V, 'advantage_squared_mean': V / A.size,
        'signed_covariance_numerators': dict(zip(('gated_EP', 'gated_TTC', 'gated_Comfort', 'float32_rounding'), sums.tolist())),
        'signed_covariance_fractions': dict(zip(('gated_EP', 'gated_TTC', 'gated_Comfort', 'float32_rounding'), (sums / V).tolist())) if V else None,
        'max_absolute_rounding_residual': float(np.max(np.abs(residual))),
        'exact_additive_residual': float(np.max(np.abs(A - all_parts.sum(axis=-1)))),
        'interpretation': 'Descriptive centered-value algebra; not parameter-gradient share or causal attribution'}

def tie_counts(records, profiles):
    totals = {}
    conditions = ('C1_float32_tie', 'C1_float64_tie', 'safety_vector_tie', 'D_terminal_tie', 'C_terminal_tie', 'T_terminal_tie')
    for condition in conditions:
        totals[condition] = {'eligible_pairs': 0, 'eligible_scenes': set(), 'nonempty_event_pairs': 0,
            'physical_time_or_severity_diff_pairs': 0, 'physical_time_or_severity_diff_scenes': set(),
            'segment_diff_pairs': 0, 'ttc_projection_diff_pairs': 0, 'per_component_nonempty_pairs': [0] * 3,
            'per_component_time_diff_pairs': [0] * 3}
    for s in range(len(records) // 4):
        for a, b in itertools.combinations(range(4), 2):
            i, j = 4 * s + a, 4 * s + b
            r1, r2, p1, p2 = records[i], records[j], profiles[i], profiles[j]
            score1, score2 = r1['metrics']['score'], r2['metrics']['score']
            masks = (np.float32(score1) == np.float32(score2), score1 == score2,
                     p1['terminal'] == p2['terminal'], *(p1['terminal'][k] == p2['terminal'][k] for k in range(3)))
            for index, condition in enumerate(conditions):
                if not masks[index]:
                    continue
                relevant = (index - 3,) if index >= 3 else (0, 1, 2)
                out = totals[condition]
                out['eligible_pairs'] += 1
                out['eligible_scenes'].add(s)
                nonempty = [bool(p1['time'][k] or p2['time'][k]) for k in range(3)]
                different = [p1['time'][k] != p2['time'][k] for k in range(3)]
                out['nonempty_event_pairs'] += int(any(nonempty[k] for k in relevant))
                if any(different[k] for k in relevant):
                    out['physical_time_or_severity_diff_pairs'] += 1
                    out['physical_time_or_severity_diff_scenes'].add(s)
                out['segment_diff_pairs'] += int(any(p1['bins'][k] != p2['bins'][k] for k in relevant))
                if 2 in relevant:
                    out['ttc_projection_diff_pairs'] += int(p1['ttc_projection'] != p2['ttc_projection'])
                for k in relevant:
                    out['per_component_nonempty_pairs'][k] += int(nonempty[k])
                    out['per_component_time_diff_pairs'][k] += int(different[k])
    for out in totals.values():
        for key in ('eligible_scenes', 'physical_time_or_severity_diff_scenes'):
            out[key] = len(out[key])
        denominator = out['nonempty_event_pairs']
        out['conditional_time_difference_fraction_given_nonempty'] = out['physical_time_or_severity_diff_pairs'] / denominator if denominator else None
    return totals


SCHEMA = 'full_train_feedback_support_v1'
RUN = Path('/mnt/project/st-world-policy-20261016/formal/consequence/runs/C1_fulltrain_seed42_official_4096_v1')
SHARDS = 8
TOTAL_READ = 32 << 30
CONTROL_READ = 256 << 20
SHARD_READ = (TOTAL_READ - CONTROL_READ) // SHARDS
SHARD_OUTPUT = 12 << 20
TOTAL_OUTPUT = 128 << 20
WALL = 1200
SMALL = 512 << 10
MAX_PLAN = 8 << 20
READ_RATE = 4 << 20
READ_BURST = 8 << 20
BOOTSTRAP_ALLOWANCE = 64 << 10


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode() + b'\n'


def read_small(path, limit=SMALL):
    path = Path(path)
    require(not path.is_symlink() and path.is_file(), 'Regular control required: ' + str(path))
    require(path.stat().st_size <= limit, 'Oversize control: ' + str(path))
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    require(len(raw) <= limit, 'Control grew beyond limit')
    return raw


def atomic(path, raw, *, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.is_symlink(), 'Symlink output refused')
    if immutable and path.exists():
        require(read_small(path, TOTAL_OUTPUT) == raw, 'Immutable/stale output conflict: ' + str(path))
        return
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    try:
        with tmp.open('xb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        if immutable:
            try:
                os.link(tmp, path)
            except FileExistsError:
                require(read_small(path, TOTAL_OUTPUT) == raw, 'Concurrent immutable output conflict')
        else:
            os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


@contextmanager
def lock(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('a+b') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def checked_document(raw, plan_id=None):
    doc = json.loads(raw)
    claimed = doc.pop('content_sha256')
    require(identity(doc) == claimed, 'Output content checksum mismatch')
    if plan_id is not None:
        require(doc['plan_id'] == plan_id, 'Stale output plan identity')
    return doc


def sealed(value):
    return encoded({**value, 'content_sha256': identity(value)})


def create_plan(run, metadata_path, output_root):
    """Read only bounded index/run controls; no ledger payload or request scan."""
    run, output_root = Path(run).resolve(), Path(output_root).resolve()
    require(output_root != run and run not in output_root.parents, 'Outputs must be outside the immutable run')
    inputs = {}
    total = 0
    def control(path, limit=SMALL):
        nonlocal total
        raw = read_small(path, limit); total += len(raw)
        require(total <= 16 << 20, 'Planning metadata budget exceeded')
        inputs[str(path)] = {'sha256': digest(raw), 'bytes': len(raw)}
        return json.loads(raw)
    rows = control(Path(metadata_path).resolve(), MAX_PLAN)
    result = control(run / 'RESULT.json')
    run_identity = control(run / 'identity.json')
    ledger_identity = control(run / 'official_feedback/IDENTITY.json')
    require(result['status'] == 'COMPLETE' and result['completed'] == result['policy_version'], 'Run is not complete')
    require(result['identity'] == run_identity['identity'] == ledger_identity['identity'], 'Run/ledger identity mismatch')
    cfg = run_identity['config']
    require(cfg['cell'] == 'C1' and cfg['execution']['world_size'] == 4 and cfg['execution']['global_batch'] == 32, 'Only original C1 world4/global32 supported')
    expected = {'ledgers': result['completed'], 'scene_presentations': result['counts']['original_GT_scenes'],
                'candidate_presentations': result['counts']['actual_rollouts']}
    require(expected['candidate_presentations'] == 4 * expected['scene_presentations'] == 4 * result['exposure'], 'Run exposure/candidate counts inconsistent')
    if run == RUN:
        require(expected == {'ledgers': 4096, 'scene_presentations': 131064, 'candidate_presentations': 524256}, 'Fixed C1 endpoint counts changed')
    require(len(rows) == expected['ledgers'], 'Metadata inventory does not match completed updates')
    bins = [{'shard': n, 'payload_bytes': 0, 'items': []} for n in range(SHARDS)]
    seen = set()
    require(all(r.get('attempt_count') == 1 and len(r.get('complete_attempts', [])) == 1 for r in rows), 'Only one complete attempt per ledger is admitted')
    for row in sorted(rows, key=lambda r: (-r['complete_attempts'][0]['payload_bytes'], r['ledger'])):
        key = row['ledger']
        require(len(key) == 64 and set(key) <= set('0123456789abcdef') and key not in seen, 'Duplicate/malformed ledger key')
        seen.add(key)
        require(row['attempt_count'] == 1 and len(row['complete_attempts']) == 1, 'Only the verified one-attempt inventory is admitted')
        a = row['complete_attempts'][0]
        require(a['attempt'] == 'attempt_00' and 0 < a['payload_bytes'] <= 8 << 20, 'Unexpected attempt or payload size')
        item = {'ledger': key, **a}
        for k in ('payload_sha256_declared', 'complete_sha256'):
            require(len(item[k]) == 64 and set(item[k]) <= set('0123456789abcdef'), 'Invalid pinned hash')
        b = min(bins, key=lambda b: (b['payload_bytes'], b['shard']))
        b['items'].append(item); b['payload_bytes'] += a['payload_bytes']
    for b in bins:
        b['items'].sort(key=lambda x: x['ledger'])
        require(2 * b['payload_bytes'] < SHARD_READ, 'Payload alone exceeds shard logical budget')
    body = {'schema': SCHEMA, 'run': str(run), 'run_identity': run_identity,
            'expected': expected, 'source_sha256': digest(Path(__file__).read_bytes()),
            'control_pins': inputs, 'plan_metadata_read_bytes': total, 'output_root': str(output_root),
            'topology': {'hosts': ['tsinghua-zt3', 'tsinghua-zt4', 'tsinghua-zt5', 'tsinghua-zt6'],
                         'workers_per_host': 2, 'shard_host_rule': 'shard // 2 indexes hosts; Lead verifies current host policy, quota and mounts before dispatch'},
            'limits': {'shards': SHARDS, 'worker_threads': 1, 'worker_cpus': 1,
                       'total_logical_read_bytes': TOTAL_READ, 'shard_logical_read_bytes': SHARD_READ,
                       'control_logical_read_bytes': CONTROL_READ, 'shard_output_bytes': SHARD_OUTPUT,
                       'total_output_bytes': TOTAL_OUTPUT, 'wall_seconds': WALL, 'worker_actual_read_bytes_per_second': READ_RATE, 'worker_actual_read_burst_bytes': READ_BURST}, 'shards': bins,
            'scope': 'CPU-only descriptive TRAIN feedback; repeated scenes across policy versions are not independent iid observations'}
    return {**body, 'plan_id': identity(body)}


def validate_plan(raw):
    p = json.loads(raw); claimed = p.pop('plan_id')
    require(p['schema'] == SCHEMA and identity(p) == claimed, 'Invalid plan identity')
    p['plan_id'] = claimed
    require(p['source_sha256'] == digest(Path(__file__).read_bytes()), 'Analyzer source changed; stale plan refused')
    require(p['limits'] == {'shards': SHARDS, 'worker_threads': 1, 'worker_cpus': 1,
            'total_logical_read_bytes': TOTAL_READ, 'shard_logical_read_bytes': SHARD_READ,
            'control_logical_read_bytes': CONTROL_READ, 'shard_output_bytes': SHARD_OUTPUT,
            'total_output_bytes': TOTAL_OUTPUT, 'wall_seconds': WALL, 'worker_actual_read_bytes_per_second': READ_RATE, 'worker_actual_read_burst_bytes': READ_BURST}, 'Budget or topology changed')
    items = [x for b in p['shards'] for x in b['items']]
    require([b['shard'] for b in p['shards']] == list(range(SHARDS)), 'Invalid shard inventory')
    require(len(items) == len({x['ledger'] for x in items}) == p['expected']['ledgers'], 'Plan does not partition all unique ledgers')
    return p


class Budget:
    """Persist conservative admissions before IO, including resumed attempts."""
    def __init__(self, folder, plan_id, label, deadline, read_limit, output_limit, resume):
        self.folder = Path(folder); self.folder.mkdir(parents=True, exist_ok=True)
        self.path = self.folder / 'BUDGET.json'
        self.deadline, self.read_limit, self.output_limit = deadline, read_limit, output_limit
        if self.path.exists():
            require(resume, 'Incomplete state exists; explicit --resume required')
            self.state = json.loads(read_small(self.path, 65536))
            require(self.state['plan_id'] == plan_id and self.state['label'] == label and self.state['deadline_unix'] == deadline, 'Stale state or reset deadline refused')
        else:
            self.state = {'plan_id': plan_id, 'label': label, 'deadline_unix': deadline,
                          'logical_read_bytes': 0, 'actual_read_bytes': 0, 'nominal_load_bytes': 0,
                          'output_reserved_bytes': 0, 'failed_attempts': 0,
                          'io_tokens': READ_BURST, 'io_updated_unix': time.time()}
        self.save()
    def save(self):
        atomic(self.path, encoded(self.state))
    def check(self):
        require(time.time() < self.deadline, 'Shared wall deadline exhausted; no reset or automatic retry')
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= 2 << 30, 'Worker RSS exceeded 2 GiB')
    def charge(self, size, nominal=0):
        self.check()
        require(size >= 0 and self.state['logical_read_bytes'] + size + nominal <= self.read_limit, 'Logical read budget exhausted before read/load')
        self.state['logical_read_bytes'] += size + nominal
        self.state['nominal_load_bytes'] += nominal
        self.save()
    def pace(self, size):
        # Reserve bandwidth before opening; a failed/resumed attempt retains debt.
        self.check(); now = time.time()
        tokens = min(READ_BURST, self.state['io_tokens'] + (now - self.state['io_updated_unix']) * READ_RATE)
        wait = max(0., (size - tokens) / READ_RATE)
        require(now + wait < self.deadline, 'Read pacing would exceed shared deadline')
        self.state['io_tokens'] = max(0., tokens - size)
        self.state['io_updated_unix'] = now + wait
        self.save()
        if wait: time.sleep(wait)
        self.check()
    def read(self, path, expected=None, limit=SMALL, *, payload=False):
        path = Path(path)
        require(not path.is_symlink(), 'Symlink input refused')
        before = path.stat(); require(stat.S_ISREG(before.st_mode) and before.st_size <= limit, 'Input type/size refused')
        self.charge(before.st_size, before.st_size if payload else 0)
        self.pace(before.st_size)
        with path.open('rb') as stream:
            raw = stream.read(before.st_size + 1)
        after = path.stat()
        self.state['actual_read_bytes'] += len(raw); self.save()
        require((before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns) and len(raw) == before.st_size, 'Input changed during read')
        require(expected is None or digest(raw) == expected, 'Input hash mismatch: ' + str(path))
        self.check(); return raw
    def json(self, path, expected=None, limit=SMALL):
        return json.loads(self.read(path, expected, limit))
    def publish(self, path, raw, *, error=False):
        require(not Path(path).exists(), 'Output already exists; verify instead of overwriting')
        require(self.state['output_reserved_bytes'] + len(raw) + (8192 if error else (64 << 10)) <= self.output_limit, 'Output budget exhausted before write')
        self.state['output_reserved_bytes'] += len(raw); self.save()
        atomic(path, raw, immutable=True)


def verify_and_analyze(plan, item, budget):
    """One RESULTS read and BytesIO load; only small rank COMPLETE joins."""
    run = Path(plan['run']); folder = run / 'official_feedback' / item['ledger']; attempt = folder / item['attempt']
    complete_raw = budget.read(attempt / 'COMPLETE.json', item['complete_sha256'])
    complete = json.loads(complete_raw)
    require(set(complete) == {'result', 'admission_sha256'}, 'Unexpected COMPLETE schema')
    payload = attempt / 'RESULTS.pt'
    require(complete['result'] == {'path': str(payload), 'sha256': item['payload_sha256_declared']}, 'Payload pin/path mismatch')
    admitted = budget.json(attempt / 'ADMITTED.json', complete['admission_sha256'])
    request_raw = budget.read(folder / 'REQUESTS.json'); doc = json.loads(request_raw); requests = doc['requests']
    version = doc['policy_version']
    require(type(version) is str and version == str(int(version)) and 0 <= int(version) < plan['expected']['ledgers'], 'Invalid policy boundary')
    require(identity(requests) == item['ledger'], 'REQUESTS identity does not match ledger key')
    n = len(requests); require(n in (96, 128), 'Only original 24/32-scene batches supported')
    require(admitted['request_hash'] == item['ledger'] and admitted['policy_version'] == version and
            admitted['calls'] == n and admitted['indices'] == list(range(n)), 'Admission does not cover exactly this ledger')
    require(payload.stat().st_size == item['payload_bytes'], 'Payload metadata size changed')
    raw = budget.read(payload, item['payload_sha256_declared'], item['payload_bytes'], payload=True)
    records = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
    del raw
    require(type(records) is list and len(records) == n, 'Incomplete feedback payload')
    run_id = plan['run_identity']; protocol = run_id['config']['feedback']['score_protocol_hash']; numeric = run_id['numeric_execution_hash']
    ranks = []; rank_pins = []
    for rank in range(4):
        path = run / 'rollouts' / ('boundary_%06d' % int(version)) / ('rank_%d_COMPLETE.json' % rank)
        raw = budget.read(path, limit=65536); receipt = json.loads(raw); ranks.append(receipt['identity'])
        rank_pins.append(digest(raw))
        require(receipt['payload']['path'] == str(path.with_name('rank_%d.pt' % rank)), 'Foreign rank payload path')
    profiles = []; state_hashes = set(); scene_keys = []; cursor = 0
    for rank, saved in enumerate(ranks):
        old = saved['old_policy']; fb = saved['feedback']; slots = len(fb['rewards'])
        require(slots == n // 16 and len(fb['contexts']) == slots and len(fb['statuses']) == 4 * slots, 'Tail/rank batch inventory mismatch')
        require(old['version'] == old['optimizer_step'] == int(version) and old['base_binding_hash'] == run_id['base_binding'], 'Run/policy boundary mismatch')
        require(saved['numeric_execution_hash'] == numeric and fb['protocol'] == protocol and fb['reward_kind'] == 'C1-PDMS', 'Numeric/reward protocol mismatch')
        state_hashes.add(old['state_hash'])
        for slot in range(slots):
            first = cursor
            group = requests[first:first+4]
            require(len({q['scene_id'] for q in group}) == len({q['source_log'] for q in group}) == 1, 'Scene candidate grouping changed')
            scene_keys.append([group[0]['scene_id'], group[0]['source_log']])
            for cand in range(4):
                request, record = requests[cursor], records[cursor]; cursor += 1
                require(record['identity'] == request and record['status'] == 'VALID_SCORED' and record['valid'] is True, 'Invalid/foreign feedback record')
                require(request['candidate_id'] == '%s:%d:%d' % (saved['rollout_id'], slot, cand), 'Rank candidate identity mismatch')
                origin = request['generator_identity']
                require(request['policy_version'] == version and origin['policy_version'] == version and origin['optimizer_step'] == int(version) and origin['state_sha256'] == old['state_hash'], 'Generator policy identity mismatch')
                require(origin['numeric_execution_hash'] == request['inference_runtime_hash'] == numeric and request['score_protocol_hash'] == protocol, 'Request runtime/protocol mismatch')
                require(origin['kind'] == 'in_memory_parameter_snapshot' and request['generator_kind'] == 'policy_snapshot' and request['generator_source_hash'] == identity(origin), 'Generator origin hash mismatch')
                require(request['campaign_id'] == run_id['config']['feedback']['campaign_id'] and request['repetition'] == 0, 'Campaign/repetition changed')
                require(fb['statuses'][slot*4+cand] == 'valid_scored' and float(np.float32(record['metrics']['score'])) == fb['rewards'][slot][cand] and fb['contexts'][slot] == request['metric_cache_sha256'], 'Original training reward/context join failed')
                require(record['score_units'] == '0..1' and record['context_unchanged'] and set(record['metrics']) == set(METRICS), 'Metric schema/units changed')
                require(all(record['metrics_valid'][k] and np.isfinite(record['metrics'][k]) and 0 <= record['metrics'][k] <= 1 for k in METRICS), 'Invalid metric')
                require(array_hash(record['requested_trajectory']) == request['trajectory_sha256'], 'Requested trajectory checksum mismatch')
                ref = record['reference_contract']; inp, execution = ref['input'], ref['execution']
                require(ref['schema'] == 'st_world_reference_input_and_execution_v2' and identity(inp) == ref['input_sha256'] and identity(execution) == ref['execution_sha256'], 'Reference contract checksum mismatch')
                require(inp == records[first]['reference_contract']['input'] and inp['full_cache_sha256'] == request['metric_cache_sha256'] and inp['score_protocol_hash'] == protocol and inp['context_hash'] == record['context_hash'], 'Reference/context join failed')
                require(array_hash(record['reference_interpolated_states']) == inp['reference_interpolated_states_sha256'] and array_hash(record['executed_states']) == execution['candidate_executed_states_sha256'], 'Execution checksum mismatch')
                profiles.append(process_profile(record))
    require(cursor == n and len(state_hashes) == 1 and len({s for s, _ in scene_keys}) == n // 4, 'Duplicate scene or conflicting rank policy in boundary')
    metrics = {k: np.asarray([r['metrics'][k] for r in records]).reshape(-1, 4) for k in METRICS}
    A, terms, algebra = decompose(metrics)
    error = float(np.max(np.abs(terms.sum(axis=-1) - metrics['score'])))
    require(error <= 1e-12, 'Canonical score decomposition mismatch')
    changes = {k: np.ptp(metrics[k], axis=1) != 0 for k in METRICS if k != 'score'}
    active = np.any(A != 0, axis=1)
    safe_changes = changes['no_at_fault_collisions'] | changes['drivable_area_compliance'] | changes['time_to_collision_within_bound']
    ep_only = active & changes['ego_progress'] & ~safe_changes & ~changes['comfort'] & np.all(metrics['no_at_fault_collisions'] * metrics['drivable_area_compliance'] > 0, axis=1)
    nonzero = {'C1': int(active.sum())}
    support_slots = {'C1': np.flatnonzero(active).tolist(), 'EP_only': np.flatnonzero(ep_only).tolist()}
    for label, key in [('D', 'drivable_area_compliance'), ('C', 'no_at_fault_collisions'), ('T', 'time_to_collision_within_bound')]:
        slots = np.flatnonzero(np.any(advantages(metrics[key] - 1.) != 0, axis=1)).tolist()
        nonzero[label] = len(slots); support_slots[label] = slots
    ties = tie_counts(records, profiles)
    for row in ties.values():
        row['empty_event_pairs'] = row['eligible_pairs'] - row['nonempty_event_pairs']
    return {'policy_version': int(version), 'scene_presentations': n // 4, 'candidate_presentations': n,
            'scene_log_pairs': scene_keys, 'nonzero_advantage_scene_presentations': nonzero,
            'support_scene_slots': {**support_slots, **timing_support_slots(records, profiles)},
            'EP_only_scene_presentations': int(ep_only.sum()), 'C1_nonzero_candidate_presentations': int(np.count_nonzero(A)),
            'event_candidate_counts_D_C_T': [sum(bool(p['time'][k]) for p in profiles) for k in range(3)],
            'terminal_ties': ties, 'reward_value_decomposition': algebra,
            'centered_A_float32': A.tolist(),
            'within_group_A_range_float32': np.ptp(A.astype(np.float32), axis=1).tolist(),
            'canonical_score_reconstruction_max_error': error,
            'input_pins': {'COMPLETE': item['complete_sha256'], 'REQUESTS': digest(request_raw),
                           'ADMITTED': complete['admission_sha256'], 'RESULTS': item['payload_sha256_declared'],
                           'rank_COMPLETE': rank_pins}}


def timing_support_slots(records, profiles):
    """Keep scene identities for the same six immutable tie definitions."""
    conditions = ('C1_float32_tie', 'C1_float64_tie', 'safety_vector_tie', 'D_terminal_tie', 'C_terminal_tie', 'T_terminal_tie')
    out = {'nonempty_time_or_severity_diff_' + k: set() for k in conditions}
    for scene in range(len(records) // 4):
        for a, b in itertools.combinations(range(4), 2):
            i, j = 4 * scene + a, 4 * scene + b
            p1, p2 = profiles[i], profiles[j]
            s1, s2 = records[i]['metrics']['score'], records[j]['metrics']['score']
            masks = (np.float32(s1) == np.float32(s2), s1 == s2, p1['terminal'] == p2['terminal'],
                     *(p1['terminal'][k] == p2['terminal'][k] for k in range(3)))
            for index, condition in enumerate(conditions):
                relevant = (index - 3,) if index >= 3 else (0, 1, 2)
                if masks[index] and any(p1['time'][k] != p2['time'][k] for k in relevant):
                    out['nonempty_time_or_severity_diff_' + condition].add(scene)
    return {k: sorted(v) for k, v in out.items()}


def configure_cpu(cpu, *, numeric=True):
    require(type(cpu) is int and cpu in os.sched_getaffinity(0), 'One available CPU required')
    os.sched_setaffinity(0, {cpu})
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    if numeric: load_numeric()


@contextmanager
def wall_guard(deadline):
    old = {s: signal.getsignal(s) for s in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    def stop(*_):
        raise TimeoutError('Worker interrupted or shared wall deadline exhausted')
    try:
        for s in old: signal.signal(s, stop)
        signal.setitimer(signal.ITIMER_REAL, max(.001, deadline - time.time()))
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for s, handler in old.items(): signal.signal(s, handler)


def worker(plan_path, shard, cpu, deadline, *, resume=False, fixed_run=False):
    raw_plan = read_small(plan_path, MAX_PLAN); plan = validate_plan(raw_plan)
    require(not fixed_run or plan['run'] == str(RUN), 'CLI only authorizes the fixed C1 TRAIN run')
    require(type(shard) is int and 0 <= shard < SHARDS, 'Invalid fixed shard')
    require(time.time() < deadline <= time.time() + WALL, 'A common nonexpired deadline within 1200 seconds is required')
    root = Path(plan['output_root']); folder = root / ('shard_%02d' % shard)
    execution = {'plan_id': plan['plan_id'], 'deadline_unix': deadline}
    atomic(root / 'EXECUTION.json', encoded(execution), immutable=True)
    with lock(folder / 'WORKER.lock'):
        done = folder / 'SHARD_COMPLETE.json'
        budget = Budget(folder, plan['plan_id'], 'shard_%02d' % shard, deadline, SHARD_READ, SHARD_OUTPUT, resume or done.exists())
        ledger = None
        try:
            with wall_guard(deadline):
                bootstrap = len(raw_plan) + Path(__file__).stat().st_size + BOOTSTRAP_ALLOWANCE
                budget.charge(bootstrap); budget.pace(bootstrap)
                if done.exists():
                    summary = checked_document(budget.read(done), plan['plan_id'])
                    require(summary['status'] == 'SHARD_COMPLETE' and summary['shard'] == shard, 'Stale shard completion')
                    require(len(summary['receipts']) == len(plan['shards'][shard]['items']) and
                            {x['ledger'] for x in summary['receipts']} == {x['ledger'] for x in plan['shards'][shard]['items']}, 'Stale completed inventory')
                    return summary
                configure_cpu(cpu)
                for rel in ('RESULT.json', 'identity.json', 'official_feedback/IDENTITY.json'):
                    path = Path(plan['run']) / rel
                    budget.read(path, plan['control_pins'][str(path)]['sha256'])
                receipts = []
                for item in plan['shards'][shard]['items']:
                    budget.check(); ledger = item['ledger']; target = folder / (ledger + '.json')
                    if target.exists():
                        raw = budget.read(target, limit=65536); existing = checked_document(raw, plan['plan_id'])
                        require(existing['item'] == item, 'Stale ledger output inventory')
                    else:
                        analysis = verify_and_analyze(plan, item, budget)
                        raw = sealed({'schema': SCHEMA, 'plan_id': plan['plan_id'], 'item': item, 'analysis': analysis})
                        require(len(raw) <= 65536, 'Per-ledger output exceeds compact-summary limit')
                        budget.publish(target, raw)
                    receipts.append({'ledger': ledger, 'sha256': digest(raw)})
                summary = {'schema': SCHEMA, 'status': 'SHARD_COMPLETE', 'plan_id': plan['plan_id'],
                           'shard': shard, 'receipts': receipts, 'budget': dict(budget.state),
                           'cuda_initialized': torch.cuda.is_initialized(), 'threads': torch.get_num_threads()}
                budget.publish(done, sealed(summary))
                return summary
        except BaseException as exc:
            budget.state['failed_attempts'] += 1; budget.save()
            error = {'schema': SCHEMA, 'status': 'FAILED_RETAINED_NO_AUTOMATIC_RETRY', 'plan_id': plan['plan_id'],
                     'shard': shard, 'ledger': ledger, 'error': type(exc).__name__, 'message': str(exc)[:2048],
                     'budget': budget.state}
            name = folder / ('ERROR_%04d.json' % budget.state['failed_attempts'])
            budget.publish(name, sealed(error), error=True)
            raise


QUANTILES = (0., .25, .5, .75, .9, .99, 1.)
VERSION_WINDOWS = ((0, 1024), (1024, 2048), (2048, 3072), (3072, 4096))


def scalar_summary(values):
    """Exact pooled order statistics, linear interpolation at q * (n - 1)."""
    values = sorted(values)
    require(all(math.isfinite(x) for x in values), 'Nonfinite amplitude')
    n = len(values)
    quantiles = {}
    for q in QUANTILES:
        pos = q * (n - 1) if n else 0
        lo = math.floor(pos); hi = math.ceil(pos)
        quantiles[str(q)] = (values[lo] + (values[hi] - values[lo]) * (pos - lo)) if n else None
    return {'count': n, 'mean': math.fsum(values) / n if n else None,
            'RMS': math.sqrt(math.fsum(x * x for x in values) / n) if n else None,
            'quantiles': quantiles}


def amplitude_statistics(analyses):
    groups = []; ranges = []
    for a in analyses:
        rows = a['centered_A_float32']; rs = a['within_group_A_range_float32']
        require(len(rows) == len(rs) == a['scene_presentations'] and all(len(x) == 4 for x in rows), 'Amplitude population mismatch')
        groups.extend(rows); ranges.extend(rs)
    active = [i for i, g in enumerate(groups) if any(x != 0 for x in g)]
    return {'all_scene_presentations': {'groups': len(groups),
                'abs_A_per_candidate': scalar_summary(abs(x) for g in groups for x in g),
                'within_group_A_range': scalar_summary(ranges)},
            'nonzero_advantage_scene_presentations': {'groups': len(active),
                'abs_A_per_candidate': scalar_summary(abs(x) for i in active for x in groups[i]),
                'within_group_A_range': scalar_summary(ranges[i] for i in active)},
            'definition': 'A is original float32 reward minus torch float32 group mean; range is float32 max(A)-min(A). Nonzero selects groups with any A != 0 and retains all four candidates. Quantiles pool original values with linear interpolation at q*(n-1); never average shard quantiles.'}


def aggregate(analyses, *, windows=True):
    counts = {'ledgers': 0, 'scene_presentations': 0, 'candidate_presentations': 0,
              'nonzero_advantage_scene_presentations': {k: 0 for k in ('C1', 'D', 'C', 'T')},
              'EP_only_scene_presentations': 0, 'C1_nonzero_candidate_presentations': 0,
              'event_candidate_counts_D_C_T': [0, 0, 0]}
    scenes = {}; scene_exposures = collections.Counter(); support = {}; logs = collections.Counter(); versions = []; ties = {}; numerator = collections.Counter(); V = 0.; residual = error = 0.
    for a in analyses:
        counts['ledgers'] += 1; versions.append(a['policy_version'])
        for k in ('scene_presentations', 'candidate_presentations', 'EP_only_scene_presentations', 'C1_nonzero_candidate_presentations'):
            counts[k] += a[k]
        for k in counts['nonzero_advantage_scene_presentations']:
            counts['nonzero_advantage_scene_presentations'][k] += a['nonzero_advantage_scene_presentations'][k]
        for k in range(3): counts['event_candidate_counts_D_C_T'][k] += a['event_candidate_counts_D_C_T'][k]
        for scene, log in a['scene_log_pairs']:
            require(scene not in scenes or scenes[scene] == log, 'Scene maps to multiple source logs')
            scenes[scene] = log; logs[log] += 1; scene_exposures[scene] += 1
        for label, slots in a['support_scene_slots'].items():
            require(len(slots) == len(set(slots)) and all(type(i) is int and 0 <= i < a['scene_presentations'] for i in slots), 'Invalid support scene slots')
            row = support.setdefault(label, {'scene_presentations': 0, 'scenes': set(), 'logs': set()})
            row['scene_presentations'] += len(slots)
            for i in slots:
                scene, log = a['scene_log_pairs'][i]; row['scenes'].add(scene); row['logs'].add(log)
        for condition, row in a['terminal_ties'].items():
            out = ties.setdefault(condition, {})
            for k, v in row.items():
                if k.startswith('conditional_'): continue
                if isinstance(v, list): out[k] = [x + y for x, y in zip(out.get(k, [0] * len(v)), v)]
                else: out[k] = out.get(k, 0) + v
        d = a['reward_value_decomposition']; V += d['advantage_squared_sum']; numerator.update(d['signed_covariance_numerators'])
        residual = max(residual, d['max_absolute_rounding_residual']); error = max(error, a['canonical_score_reconstruction_max_error'])
    for row in ties.values():
        row['conditional_time_difference_fraction_given_nonempty'] = row['physical_time_or_severity_diff_pairs'] / row['nonempty_event_pairs'] if row['nonempty_event_pairs'] else None
    counts['candidate_pair_presentations'] = 6 * counts['scene_presentations']
    result = {'counts': counts,
            'support_unique_population': {k: {'scene_presentations': v['scene_presentations'], 'unique_scenes': len(v['scenes']), 'unique_logs': len(v['logs'])} for k, v in support.items()},
            'scene_exposure_distribution': {'presentations_per_unique_scene': scalar_summary(scene_exposures.values()),
                'histogram_scene_count_by_presentations': dict(sorted(collections.Counter(scene_exposures.values()).items())),
                'fraction_presentations_after_first_occurrence': (counts['scene_presentations'] - len(scenes)) / counts['scene_presentations'] if scenes else None,
                'fraction_unique_scenes_seen_multiple_times': sum(n > 1 for n in scene_exposures.values()) / len(scenes) if scenes else None}, 'unique_scenes': len(scenes), 'unique_logs': len(logs),
            'scene_presentations_by_log': dict(sorted(logs.items())), 'policy_versions': sorted(versions),
            'terminal_ties_over_scene_presentations': ties, 'centered_advantage_amplitude': amplitude_statistics(analyses),
            'reward_value_decomposition': {'advantage_squared_sum': V, 'signed_covariance_numerators': dict(numerator),
                'signed_covariance_fractions': {k: v / V for k, v in numerator.items()} if V else None,
                'max_absolute_rounding_residual': residual, 'canonical_score_reconstruction_max_error': error},
            'interpretation': 'Descriptive presentation-weighted algebra and support counts. Repeated scenes/logs across policies are not independent iid data; no confidence interval, parameter-gradient share or causal attribution is claimed.'}
    if windows:
        result['fixed_policy_version_windows'] = {
            '[%d,%d)' % (lo, hi): aggregate([a for a in analyses if lo <= a['policy_version'] < hi], windows=False)
            for lo, hi in VERSION_WINDOWS}
    return result


def merge(plan_path, cpu=None, *, fixed_run=False):
    raw_plan = read_small(plan_path, MAX_PLAN); plan = validate_plan(raw_plan); root = Path(plan['output_root'])
    require(not fixed_run or plan['run'] == str(RUN), 'CLI only authorizes the fixed C1 TRAIN run')
    if cpu is not None: configure_cpu(cpu, numeric=False)
    with lock(root / 'MERGE.lock'):
        paths = [root / ('shard_%02d' % s) / 'SHARD_COMPLETE.json' for s in range(SHARDS)]
        if not all(p.exists() for p in paths):
            return {'status': 'INCOMPLETE', 'missing_shards': [s for s, p in enumerate(paths) if not p.exists()], 'plan_id': plan['plan_id']}
        execution = json.loads(read_small(root / 'EXECUTION.json'))
        require(execution['plan_id'] == plan['plan_id'], 'Stale shared deadline')
        budget = Budget(root / 'merge', plan['plan_id'], 'merge', execution['deadline_unix'],
                        CONTROL_READ - plan['plan_metadata_read_bytes'], 16 << 20, (root / 'merge/BUDGET.json').exists())
        try:
            with wall_guard(execution['deadline_unix']):
                bootstrap = len(raw_plan) + Path(__file__).stat().st_size + BOOTSTRAP_ALLOWANCE
                budget.charge(bootstrap); budget.pace(bootstrap); analyses = []; all_keys = set(); shard_pins = []
                for shard, path in enumerate(paths):
                    raw = budget.read(path); summary = checked_document(raw, plan['plan_id']); shard_pins.append(digest(raw))
                    require(summary['status'] == 'SHARD_COMPLETE' and summary['shard'] == shard, 'Invalid shard completion')
                    expected = {x['ledger']: x for x in plan['shards'][shard]['items']}
                    refs = summary['receipts']
                    require(len(refs) == len(expected) and {x['ledger'] for x in refs} == set(expected), 'Incomplete/duplicate shard coverage')
                    state = budget.json(path.parent / 'BUDGET.json', limit=65536)
                    require(state['plan_id'] == plan['plan_id'] and state['label'] == 'shard_%02d' % shard and
                            state['deadline_unix'] == execution['deadline_unix'], 'Stale final shard budget')
                    require(summary['budget']['logical_read_bytes'] <= state['logical_read_bytes'] <= SHARD_READ and
                            summary['budget']['output_reserved_bytes'] + len(raw) <= state['output_reserved_bytes'] <= SHARD_OUTPUT,
                            'Shard resource bounds exceeded')
                    for ref in refs:
                        key = ref['ledger']; require(key not in all_keys, 'Duplicate ledger across shards'); all_keys.add(key)
                        row = checked_document(budget.read(path.parent / (key + '.json'), ref['sha256'], 65536), plan['plan_id'])
                        require(row['item'] == expected[key], 'Output input inventory changed')
                        analyses.append(row['analysis'])
                result = aggregate(analyses)
                budget.check(); want = plan['expected']
                require(result['policy_versions'] == list(range(want['ledgers'])), 'Missing/duplicate policy boundary; not COMPLETE')
                require(all(result['counts'][k] == v for k, v in want.items()), 'Full-run presentation/candidate reconciliation failed')
                result.update(schema=SCHEMA, status='COMPLETE', plan_id=plan['plan_id'], shard_receipt_sha256=shard_pins,
                              expected_counts_from_run_RESULT=want, new_score_calls=0, model_loads=0, optimizer_updates=0)
                target = root / 'RESULT.json'; raw = sealed(result)
                if target.exists():
                    require(budget.read(target, limit=16 << 20) == raw, 'Stale final output refused')
                else:
                    budget.publish(target, raw)
                return result
        except BaseException as exc:
            budget.state['failed_attempts'] += 1; budget.save()
            error = {'schema': SCHEMA, 'status': 'FAILED_MERGE_NOT_COMPLETE', 'plan_id': plan['plan_id'],
                     'error': type(exc).__name__, 'message': str(exc)[:2048], 'budget': budget.state}
            budget.publish(root / 'merge' / ('ERROR_%04d.json' % budget.state['failed_attempts']), sealed(error), error=True)
            raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    pp = sub.add_parser('plan'); pp.add_argument('--metadata', required=True); pp.add_argument('--output-root', required=True); pp.add_argument('--plan-out', required=True)
    wp = sub.add_parser('worker'); wp.add_argument('--plan', required=True); wp.add_argument('--shard', type=int, required=True); wp.add_argument('--cpu', type=int, required=True); wp.add_argument('--deadline-unix', type=float, required=True); wp.add_argument('--resume', action='store_true')
    mp = sub.add_parser('merge'); mp.add_argument('--plan', required=True); mp.add_argument('--cpu', type=int, required=True)
    args = p.parse_args(argv)
    if args.command == 'plan':
        result = create_plan(RUN, args.metadata, args.output_root); atomic(args.plan_out, encoded(result), immutable=True)
        print(json.dumps({'status': 'PLAN_READY_NO_PAYLOAD_READ', 'plan_id': result['plan_id'], 'expected': result['expected']})); return 0
    result = worker(args.plan, args.shard, args.cpu, args.deadline_unix, resume=args.resume, fixed_run=True) if args.command == 'worker' else merge(args.plan, args.cpu, fixed_run=True)
    print(json.dumps({k: result[k] for k in ('status', 'plan_id') if k in result}))
    return 0 if result['status'] in ('COMPLETE', 'SHARD_COMPLETE') else 2


if __name__ == '__main__':
    raise SystemExit(main())
