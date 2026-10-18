#!/usr/bin/env python3
"""Describe update-local rollout scale from completed, verified compact TRAIN receipts.

This does not create rewards, normalized advantages, gradients or coefficients.
The primary analysis and its original deadline/counters remain immutable.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack, contextmanager
import importlib.util
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import stat
import struct
import time

_spec = importlib.util.spec_from_file_location('full_feedback_pure_helpers', Path(__file__).with_name('analyze_full_feedback_support.py'))
base = importlib.util.module_from_spec(_spec)
_helper_raw = Path(base.__file__).read_bytes()
_HELPER_SHA = hashlib.sha256(_helper_raw).hexdigest()
if _HELPER_SHA != '30bb3237019a463e68bb5c0686d902521d814ee99bec228a94de7b537f92f4ca':
    raise RuntimeError('Frozen primary helper source hash mismatch before import')
exec(compile(_helper_raw, base.__file__, 'exec'), base.__dict__)  # Verified stdlib-only helper; no torch/numpy or RLIMIT.
require, identity, digest = base.require, base.identity, base.digest

SCHEMA = 'rollout_scale_train_evidence_v1'
EPSILON = 1e-10
READ_LIMIT = 64 << 20
OUTPUT_LIMIT = 2 << 20
WALL_SECONDS = CPU_SECONDS = 60
RSS_LIMIT = 2 << 30
BOOTSTRAP_BYTES = 128 << 10
PRIMARY_PLAN_SHA = '2d50faf1c690bd8ebc487504a32ecae9637432e4742c561a9ea5f6442a995b06'
PRIMARY_RESULT_SHA = '0a6a84d3652122aeb69760b5237a56c71fec269c16b4066e78a0ddb05876d8cc'
PRIMARY_OUTPUT = Path('/mnt/project/st-world-policy-20261016/audits/consequence/full_train_feedback_support_v1')
PRIMARY_SOURCE_SHA = '30bb3237019a463e68bb5c0686d902521d814ee99bec228a94de7b537f92f4ca'
EXPECTED = {'ledgers': 4096, 'scene_presentations': 131064, 'candidate_presentations': 524256}
PROPOSAL_SHA = 'a32d9e405316bdc257ce283343e23b6fb05d3ed54086d42a0e2243dcce82d69c'
NUMERIC_FLAGS = ('N_below_two', 'nonfinite_input_or_statistics', 'std_zero', 'nonzero_A_but_std_zero',
                 'positive_std_le_epsilon', 'constant_nonzero_group', 'rounding_bound_can_cover_all_energy')
STRATA = ('all', 'nonzero', 'allzero', 'safe_EP_only_update', 'DCT_event_present_update',
          'DCT_metric_discriminative_update', 'D_event_present_update', 'C_event_present_update',
          'T_event_present_update', 'D_metric_discriminative_update', 'C_metric_discriminative_update',
          'T_metric_discriminative_update')
VALUE_KEYS = ('std_correction1', 'gain', 'energy_A', 'max_abs_A', 'rounding_energy_bound', 'rounding_bound_fraction')
TABLE_COLUMNS = ('policy_version', 'N', 'zero_groups', 'nonzero_groups', 'sum_A', 'mean_A', 'M2',
                 'std_correction1', 'energy_A', 'max_abs_A', 'delta_rounding', 'rounding_energy_bound',
                 'rounding_bound_fraction', 'gain', 'constant_nonzero_groups', 'numeric_flags_mask', 'strata_mask')


def describe_values(values):
    """Two-pass centered FP64 M2, correction=1; never center the numerator again."""
    flags = []
    n = len(values)
    if n < 2: flags.append('N_below_two')
    if not all(type(x) in (int, float) and math.isfinite(x) for x in values):
        return {'N': n, 'flags': flags + ['nonfinite_input_or_statistics'], 'allzero': False,
                **{k: None for k in ('sum_A', 'mean_A', 'M2', 'std_correction1', 'energy_A', 'max_abs_A', 'gain')}}
    # JSON numbers exactly represent the stored float32 A; Python float is FP64.
    require(all(float(struct.unpack('!f', struct.pack('!f', x))[0]) == x for x in values), 'A is not an exact stored float32 value')
    total = math.fsum(values); mean = total / n if n else None
    energy = math.fsum(x * x for x in values)
    m2 = math.fsum((x - mean) ** 2 for x in values) if n else None
    std = math.sqrt(m2 / (n - 1)) if n >= 2 else None
    allzero = bool(n) and all(x == 0 for x in values)
    if std == 0:
        flags.append('std_zero')
        if not allzero: flags.append('nonzero_A_but_std_zero')
    if std is not None and 0 < std <= EPSILON: flags.append('positive_std_le_epsilon')
    if any(x is not None and not math.isfinite(x) for x in (total, mean, energy, m2, std)):
        flags.append('nonfinite_input_or_statistics')
    gain = 1. / (std + EPSILON) if std is not None and std > 0 and not allzero else None
    return {'N': n, 'sum_A': total, 'mean_A': mean, 'M2': m2, 'std_correction1': std,
            'energy_A': energy, 'max_abs_A': max(map(abs, values), default=None), 'gain': gain,
            'allzero': allzero, 'flags': flags}


def describe_update(a):
    groups = a['centered_A_float32']; scenes = a['scene_log_pairs']; n = a['candidate_presentations']
    require(len(groups) == len(scenes) == a['scene_presentations'] and all(len(g) == 4 for g in groups) and n == 4 * len(groups), 'Incomplete G4 compact population')
    require(len({s for s, _ in scenes}) == len(scenes), 'Duplicate scene inside update')
    values = [x for g in groups for x in g]; row = describe_values(values)
    require(not any(x == 'nonfinite_input_or_statistics' for x in row['flags']), 'Nonfinite compact advantages/statistics')
    active = [i for i, g in enumerate(groups) if any(x != 0 for x in g)]
    slots = a['support_scene_slots']
    for label in ('C1', 'D', 'C', 'T', 'EP_only'):
        require(len(slots[label]) == len(set(slots[label])) and all(type(i) is int and 0 <= i < len(groups) for i in slots[label]), 'Invalid compact support slots')
    require(slots['C1'] == active and set(slots['EP_only']) <= set(active), 'Active/EP-only support mismatch')
    require(all(len(slots[k]) == a['nonzero_advantage_scene_presentations'][k] for k in ('C1', 'D', 'C', 'T')) and
            len(slots['EP_only']) == a['EP_only_scene_presentations'], 'Compact support counts mismatch')
    events = a['event_candidate_counts_D_C_T']
    require(len(events) == 3 and all(type(x) is int and 0 <= x <= n for x in events), 'Invalid event counts')
    constant_nonzero = sum(bool(g) and g[0] != 0 and all(x == g[0] for x in g) for g in groups)
    if constant_nonzero: row['flags'].append('constant_nonzero_group')
    delta = a['reward_value_decomposition']['max_absolute_rounding_residual']
    reconstruction = a['canonical_score_reconstruction_max_error']
    require(math.isfinite(delta) and delta >= 0 and math.isfinite(reconstruction) and reconstruction >= 0, 'Invalid precision bounds')
    bound = n * delta * delta; fraction = bound / row['energy_A'] if row['energy_A'] else None
    if fraction is not None and fraction >= 1: row['flags'].append('rounding_bound_can_cover_all_energy')
    flag = {'all': True, 'nonzero': bool(active), 'allzero': row['allzero'],
            'safe_EP_only_update': bool(active) and set(slots['EP_only']) == set(active) and not any(events),
            'DCT_event_present_update': any(events), 'DCT_metric_discriminative_update': any(slots[k] for k in ('D', 'C', 'T'))}
    for i, k in enumerate(('D', 'C', 'T')):
        flag[k + '_event_present_update'] = bool(events[i])
        flag[k + '_metric_discriminative_update'] = bool(slots[k])
    row.update(policy_version=a['policy_version'], zero_groups=len(groups)-len(active), nonzero_groups=len(active),
               delta_rounding=delta, rounding_energy_bound=bound, rounding_bound_fraction=fraction,
               constant_nonzero_groups=constant_nonzero, canonical_score_reconstruction_max_error=reconstruction,
               numeric_flags_mask=sum(1 << i for i, k in enumerate(NUMERIC_FLAGS) if k in row['flags']),
               strata_mask=sum(1 << i for i, k in enumerate(STRATA) if flag[k]),
               _scenes=scenes, _active=active, _strata=flag)
    return row


def population_summary(rows):
    scenes, logs, active_scenes, active_logs = set(), set(), set(), set()
    mapping = {}; scene_count = candidate_count = active_count = 0
    for r in rows:
        scene_count += len(r['_scenes']); candidate_count += r['N']; active_count += len(r['_active'])
        for scene, log in r['_scenes']:
            require(scene not in mapping or mapping[scene] == log, 'Scene/log identity changed across updates')
            mapping[scene] = log; scenes.add(scene); logs.add(log)
        for slot in r['_active']:
            scene, log = r['_scenes'][slot]; active_scenes.add(scene); active_logs.add(log)
    energy_sum = math.fsum(r['energy_A'] for r in rows)
    bound_sum = math.fsum(r['rounding_energy_bound'] for r in rows)
    return {'sum_energy_A': energy_sum, 'sum_rounding_energy_bound': bound_sum,
            'ratio_of_rounding_bound_sum_to_energy_sum': bound_sum / energy_sum if energy_sum else None,
            'updates': len(rows), 'scene_presentations': scene_count, 'candidate_presentations': candidate_count,
            'unique_scenes': len(scenes), 'unique_logs': len(logs), 'nonzero_scene_presentations': active_count,
            'nonzero_unique_scenes': len(active_scenes), 'nonzero_unique_logs': len(active_logs),
            'allzero_updates': sum(r['allzero'] for r in rows),
            'numeric_flag_update_counts': {k: sum(k in r['flags'] for r in rows) for k in NUMERIC_FLAGS},
            'distributions': {k: base.scalar_summary(r[k] for r in rows if r[k] is not None) for k in VALUE_KEYS},
            'max_canonical_score_reconstruction_error': max((r['canonical_score_reconstruction_max_error'] for r in rows), default=None)}


def summarize(rows):
    return {'strata': {k: population_summary([r for r in rows if r['_strata'][k]]) for k in STRATA},
            'fixed_policy_version_windows': {'[%d,%d)' % (lo, hi): {
                k: population_summary([r for r in rows if lo <= r['policy_version'] < hi and r['_strata'][k]]) for k in STRATA}
                for lo, hi in base.VERSION_WINDOWS},
            'update_table': {'columns': TABLE_COLUMNS, 'numeric_flags_bits': NUMERIC_FLAGS, 'strata_bits': STRATA,
                             'rows': [[r[k] for k in TABLE_COLUMNS] for r in sorted(rows, key=lambda r: r['policy_version'])]},
            'precision_interpretation': 'N*delta^2 / sum(A^2) is an upper-bound comparison, not an attribution. Ratio >= 1 is UNRESOLVED: rounding could cover all energy, but dominance is not proved. Zero energy has null ratio (ZERO_ADV_ENERGY). Stored delta combines score casting/float32 centering residual; separate original score pairs are unavailable in compact receipts.',
            'scale_interpretation': 'Update-local denominator uses every original candidate, including zero groups. Correction=1; epsilon=1e-10 outside sqrt; numerator is not re-centered. Allzero gain is null, never reported as a real 1e10 policy amplification. Zero scene groups stay zero. No new A/reward/gradient is generated. Different update gains change relative weighting across updates and against BC; these statistics do not establish safety, gradient ratios or benefit. Repeated scene/log presentations are not iid.',
            'quantile_definition': 'Exact pooled per-update values, linear interpolation at q*(n-1); q=0,.25,.5,.75,.9,.99,1. Each field count is its actual denominator; empty strata remain undefined.'}


class ReadBudget:
    def __init__(self, output, call_id, start, bootstrap_plan_bytes=0):
        self.output, self.start = Path(output), start
        self.state = {'call_id': call_id, 'read_limit_bytes': READ_LIMIT, 'precharged_read_bytes': BOOTSTRAP_BYTES + bootstrap_plan_bytes + (1 if bootstrap_plan_bytes else 0),
                      'actual_json_read_bytes': bootstrap_plan_bytes, 'bootstrap_source_allowance_bytes': BOOTSTRAP_BYTES,
                      'reserved_once_only_allowance_bytes': READ_LIMIT, 'read_operations': 0,
                      'flush_every_reads': 64, 'abrupt_kill_counter_may_lag_reads': 64,
                      'started_unix': time.time(), 'independent_wall_seconds': WALL_SECONDS}
        self.save()
    def save(self):
        base.atomic(self.output / 'READ_BUDGET.json', base.encoded(self.state))
    def check(self):
        usage = resource.getrusage(resource.RUSAGE_SELF)
        require(time.monotonic()-self.start < WALL_SECONDS and usage.ru_utime+usage.ru_stime < CPU_SECONDS, 'Derived 60-second wall/CPU budget exhausted')
        require(usage.ru_maxrss * 1024 <= RSS_LIMIT, 'Derived sampled RSS exceeds 2 GiB')
    def read(self, path, limit, expected=None):
        self.check(); path = Path(path)
        require(not path.is_symlink(), 'Symlink input refused')
        before = path.stat(); require(stat.S_ISREG(before.st_mode) and before.st_size <= limit, 'Input type/size refused')
        require(self.state['precharged_read_bytes'] + before.st_size + 1 <= READ_LIMIT, 'Derived read limit exhausted before read')
        self.state['precharged_read_bytes'] += before.st_size + 1  # Hard in-memory precharge; this call cannot resume.
        with path.open('rb') as stream: raw = stream.read(before.st_size+1)
        after = path.stat(); self.state['actual_json_read_bytes'] += len(raw)
        self.state['read_operations'] += 1
        if self.state['read_operations'] % 64 == 0: self.save()
        require((before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns) and len(raw) == before.st_size, 'Compact input changed during read')
        require(expected is None or digest(raw) == expected, 'Input hash mismatch: ' + str(path))
        self.check(); return raw
    def json(self, path, limit, expected=None):
        return json.loads(self.read(path, limit, expected))


@contextmanager
def readonly_lock(path):
    require(Path(path).is_file() and not Path(path).is_symlink(), 'Missing primary lifecycle lock')
    with Path(path).open('rb') as handle:
        base.fcntl.flock(handle, base.fcntl.LOCK_EX | base.fcntl.LOCK_NB)
        yield


def read_completed_primary(plan_path, budget, *, production=True, plan_raw=None):
    """Hold all primary locks throughout verification; no original payload access."""
    if plan_raw is None: plan_raw = budget.read(plan_path, 8 << 20, PRIMARY_PLAN_SHA if production else None)
    require(not production or digest(plan_raw) == PRIMARY_PLAN_SHA, 'Primary plan pin mismatch')
    plan = json.loads(plan_raw); claimed = plan.pop('plan_id')
    require(identity(plan) == claimed and plan['schema'] == base.SCHEMA, 'Primary plan identity mismatch')
    plan['plan_id'] = claimed
    if production:
        require(plan['source_sha256'] == PRIMARY_SOURCE_SHA and plan['expected'] == EXPECTED and plan['run'] == str(base.RUN), 'Unregistered primary population/source')
    root = Path(plan['output_root']); output = budget.output.resolve()
    require(output != root.resolve() and root.resolve() not in output.parents, 'Derived output must be outside primary output tree')
    require(Path(plan['run']).resolve() not in output.parents, 'Derived output cannot touch original run')
    with ExitStack() as stack:
        stack.enter_context(readonly_lock(root / 'MERGE.lock'))
        for shard in range(8): stack.enter_context(readonly_lock(root / ('shard_%02d' % shard) / 'WORKER.lock'))
        full_raw = budget.read(root / 'RESULT.json', 16 << 20, PRIMARY_RESULT_SHA if production else None)
        full = base.checked_document(full_raw, claimed)
        require(full['status'] == 'COMPLETE' and full['expected_counts_from_run_RESULT'] == plan['expected'] and
                all(full['counts'][k] == v for k,v in plan['expected'].items()), 'Primary analysis is not fully COMPLETE')
        require(full['policy_versions'] == list(range(plan['expected']['ledgers'])), 'Primary policy coverage incomplete')
        execution = budget.json(root / 'EXECUTION.json', 65536)
        require(execution['plan_id'] == claimed, 'Primary execution identity mismatch')
        states = []
        for label in ['shard_%02d' % s for s in range(8)] + ['merge']:
            state = budget.json(root / label / 'BUDGET.json', 65536)
            require(state['plan_id'] == claimed and state['label'] == label and state['deadline_unix'] == execution['deadline_unix'], 'Primary budget identity mismatch')
            require(type(state['logical_read_bytes']) is int and state['logical_read_bytes'] >= 0, 'Invalid primary cumulative read accounting')
            states.append(state)
        primary_read = plan['plan_metadata_read_bytes'] + sum(s['logical_read_bytes'] for s in states)
        require(primary_read + READ_LIMIT < 32 << 30, 'Primary actual logical reads plus full derived allowance exceed original 32 GiB')
        require([s['shard'] for s in plan['shards']] == list(range(8)) and len(full['shard_receipt_sha256']) == 8, 'Primary shard topology mismatch')
        rows = []; seen = set(); shard_hashes = []
        for shard, inventory in enumerate(plan['shards']):
            folder = root / ('shard_%02d' % shard)
            raw = budget.read(folder / 'SHARD_COMPLETE.json', 512 << 10, full['shard_receipt_sha256'][shard])
            shard_hashes.append(digest(raw)); complete = base.checked_document(raw, claimed)
            expected = {i['ledger']: i for i in inventory['items']}; refs = complete['receipts']
            require(complete['status'] == 'SHARD_COMPLETE' and complete['shard'] == shard and len(expected) == len(inventory['items']), 'Invalid shard inventory')
            require(len(refs) == len(expected) and {r['ledger'] for r in refs} == set(expected), 'Missing or duplicate compact receipts')
            require(complete['budget']['logical_read_bytes'] <= states[shard]['logical_read_bytes'] and
                    complete['budget']['output_reserved_bytes'] + len(raw) <= states[shard]['output_reserved_bytes'], 'Primary budget predates completion')
            for ref in refs:
                key = ref['ledger']; require(key not in seen, 'Duplicate ledger across primary shards'); seen.add(key)
                doc = base.checked_document(budget.read(folder / (key + '.json'), 65536, ref['sha256']), claimed)
                require(doc['item'] == expected[key], 'Compact ledger inventory mismatch')
                pins = doc['analysis']['input_pins']
                require(pins['COMPLETE'] == expected[key]['complete_sha256'] and pins['RESULTS'] == expected[key]['payload_sha256_declared'], 'Compact original input pins mismatch')
                rows.append(describe_update(doc['analysis']))
        require(len(rows) == len(seen) == plan['expected']['ledgers'] and sorted(r['policy_version'] for r in rows) == list(range(len(rows))), 'Derived ledger/boundary coverage incomplete')
        require(sum(len(r['_scenes']) for r in rows) == plan['expected']['scene_presentations'] and sum(r['N'] for r in rows) == plan['expected']['candidate_presentations'], 'Derived presentation/candidate reconciliation failed')
        require(sum(r['nonzero_groups'] for r in rows) == full['counts']['nonzero_advantage_scene_presentations']['C1'], 'Derived nonzero support does not reconcile to primary')
        return rows, {'primary_plan_sha256': digest(plan_raw), 'primary_plan_id': claimed, 'primary_RESULT_sha256': digest(full_raw),
                      'primary_shard_COMPLETE_sha256': shard_hashes, 'primary_actual_logical_read_bytes': primary_read,
                      'primary_plus_full_derived_allowance_bytes': primary_read+READ_LIMIT,
                      'expected_counts': plan['expected'], 'primary_locks_held_through_all_compact_reads': True,
                      'primary_counters_and_deadline_unchanged': True}


def _execute(plan_path, output, cpu, *, production=True):
    start = time.monotonic(); output = Path(output)
    require(cpu in os.sched_getaffinity(0), 'Assigned CPU unavailable')
    os.sched_setaffinity(0, {cpu}); os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'): os.environ[name] = '1'
    # A bounded bootstrap plan read is charged below; isolation precedes every output write.
    resolved_output = output.resolve()
    forbidden = (PRIMARY_OUTPUT.resolve(), base.RUN.resolve()) if production else ()
    require(all(resolved_output != p and p not in resolved_output.parents for p in forbidden), 'Output overlaps fixed primary or original run')
    require(not output.exists(), 'Independent output already exists; no retry or overwrite')
    stat_plan = Path(plan_path).stat()
    require(stat_plan.st_size + BOOTSTRAP_BYTES + 1 <= READ_LIMIT, 'Bootstrap plan exceeds derived read allowance')
    plan_raw = base.read_small(plan_path, 8 << 20)
    require(not production or digest(plan_raw) == PRIMARY_PLAN_SHA, 'Primary plan pin mismatch')
    preview = json.loads(plan_raw)
    for path in (Path(preview['output_root']).resolve(), Path(preview['run']).resolve()):
        require(resolved_output != path and path not in resolved_output.parents, 'Output overlaps primary or original run')
    output.mkdir(parents=True, exist_ok=False)  # Independent once-only call; never reset primary or resume.
    call = {'schema': SCHEMA, 'started_unix_ns': time.time_ns(), 'source_sha256': digest(Path(__file__).read_bytes()),
            'helper_source_sha256': _HELPER_SHA, 'primary_plan_sha256': digest(plan_raw),
            'primary_RESULT_sha256_expected': PRIMARY_RESULT_SHA if production else None,
            'primary_plan_path': str(Path(plan_path).resolve()), 'cpu': cpu, 'epsilon': EPSILON, 'correction': 1,
            'proposal_sha256': PROPOSAL_SHA, 'limits': {'read_bytes': READ_LIMIT, 'output_bytes': OUTPUT_LIMIT,
            'wall_seconds': WALL_SECONDS, 'cpu_seconds': CPU_SECONDS, 'rss_bytes_sampled': RSS_LIMIT}}
    call_id = identity(call); base.atomic(output / 'EXECUTION.json', base.encoded({**call, 'call_id': call_id}), immutable=True)
    budget = ReadBudget(output, call_id, start, len(plan_raw))
    try:
        rows, evidence = read_completed_primary(plan_path, budget, production=production, plan_raw=plan_raw)
        result = summarize(rows); budget.check(); budget.save()
        result.update(schema=SCHEMA, status='COMPLETE_DESCRIPTIVE_SCALE_EVIDENCE_NOT_APPROVAL', call_id=call_id,
                      source_sha256=call['source_sha256'], helper_source_sha256=_HELPER_SHA, epsilon=EPSILON, correction=1, evidence=evidence,
                      new_rewards=0, new_scores=0, model_loads=0, optimizer_updates=0,
                      execution={'wall_seconds': time.monotonic()-start, 'cpu_seconds': resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,
                                 'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024, **budget.state})
        raw = base.sealed(result)
        require(len(raw) + (16 << 10) <= OUTPUT_LIMIT, 'Derived output exceeds 2 MiB; no quantile fallback is allowed')
        budget.check(); base.atomic(output / 'RESULT.json', raw, immutable=True)
        return result
    except BaseException as exc:
        budget.save()
        base.atomic(output / 'ERROR.json', base.encoded({'schema': SCHEMA, 'status': 'FAILED_RETAINED_NO_RETRY', 'call_id': call_id,
                    'error': type(exc).__name__, 'message': str(exc)[:2048], 'budget': budget.state}), immutable=True)
        raise


def execute(plan_path, output, cpu, *, production=True):
    with base.wall_guard(time.time() + WALL_SECONDS):
        return _execute(plan_path, output, cpu, production=production)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--primary-plan', type=Path, required=True, help='Exact approved primary PLAN_CANDIDATE.json; primary must be COMPLETE and all worker locks idle')
    parser.add_argument('--output-dir', type=Path, required=True, help='New independent output directory outside the primary output tree; no resume or overwrite')
    parser.add_argument('--cpu', type=int, required=True, help='One currently allocated CPU affinity ID')
    args = parser.parse_args(argv)
    try:
        result = execute(args.primary_plan, args.output_dir, args.cpu)
    except (Exception, KeyboardInterrupt) as exc:
        parser.exit(2, '%s: %s\n' % (type(exc).__name__, str(exc)[:2048]))
    print(json.dumps({'status': result['status'], 'call_id': result['call_id'], 'updates': result['strata']['all']['updates']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
