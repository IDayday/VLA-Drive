"""Bounded existing-TRAIN compact metadata selection; no experiment dispatch."""
import hashlib
import json
import os
from pathlib import Path
import resource
import time

OUT = Path(__file__).resolve().parent
AUDIT = Path('/mnt/project/st-world-policy-20261016/audits/consequence/full_train_feedback_support_v1')
PLAN = Path('/mnt/project/VLA-Drive-st-full-feedback-runtime-20261018/reports/st_world_policy/20261018/full_train_feedback_support_v1/PLAN_CANDIDATE.json')
PRIMARY_SHA = '0a6a84d3652122aeb69760b5237a56c71fec269c16b4066e78a0ddb05876d8cc'
PLAN_SHA = '2d50faf1c690bd8ebc487504a32ecae9637432e4742c561a9ea5f6442a995b06'
READ_CAP = 6 * 2**20
DOMAIN = 'st-road-failure-direction-v1'
SEED = 42
reads = []
started = time.monotonic()
resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def key(kind, value):
    return digest(('\0'.join((DOMAIN, str(SEED), kind, value))).encode())


def read(path, expected):
    path = Path(path)
    size = path.stat().st_size
    assert sum(r['bytes'] for r in reads) + size <= READ_CAP, 'METADATA_READ_CAP'
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 < 2 * 2**30, 'ACTUAL_RSS_CAP'
    assert path.suffix == '.json', 'METADATA_ONLY'
    raw = path.read_bytes()
    actual = digest(raw)
    reads.append({'path': str(path), 'bytes': len(raw), 'sha256': actual})
    assert actual == expected, 'METADATA_HASH_MISMATCH'
    value = json.loads(raw)
    if 'content_sha256' in value:
        content = {k: v for k, v in value.items() if k != 'content_sha256'}
        assert digest(encoded(content)) == value['content_sha256'], 'CONTENT_ID_MISMATCH'
    return value


def stat_only(path):
    path = Path(path)
    try:
        st = path.stat()
        return {'path': str(path), 'exists': True, 'bytes': st.st_size}
    except FileNotFoundError:
        return {'path': str(path), 'exists': False, 'bytes': None}


def publish(name, value):
    path = OUT / name
    assert not path.exists(), 'IMMUTABLE_OUTPUT_ALREADY_EXISTS'
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    primary = read(AUDIT / 'RESULT.json', PRIMARY_SHA)
    plan = read(PLAN, PLAN_SHA)
    assert primary['status'] == 'COMPLETE' and primary['plan_id'] == plan['plan_id']
    lookup = {}
    for shard, expected in enumerate(primary['shard_receipt_sha256']):
        receipt = read(AUDIT / ('shard_%02d' % shard) / 'SHARD_COMPLETE.json', expected)
        assert receipt['status'] == 'SHARD_COMPLETE' and receipt['plan_id'] == plan['plan_id']
        for row in receipt['receipts']:
            assert row['ledger'] not in lookup
            lookup[row['ledger']] = (shard, row['sha256'])
    assert len(lookup) == 4096
    expected_items = {r['ledger']: r for s in plan['shards'] for r in s['items']}
    selected, visited, seen = [], [], set()
    for ledger in sorted(lookup, key=lambda value: (key('ledger', value), value)):
        shard, expected = lookup[ledger]
        path = AUDIT / ('shard_%02d' % shard) / (ledger + '.json')
        compact = read(path, expected)
        assert compact['plan_id'] == plan['plan_id'] and compact['item'] == expected_items[ledger]
        a = compact['analysis']; eligible = a['support_scene_slots']['D']
        assert len(eligible) == a['nonzero_advantage_scene_presentations']['D']
        visited.append({'ledger': ledger, 'compact_sha256': expected, 'policy_version': a['policy_version'], 'D_slots': len(eligible)})
        n_local = a['scene_presentations'] // 4
        for slot in sorted(eligible, key=lambda s: (key('scene', a['scene_log_pairs'][s][0]), a['scene_log_pairs'][s][0], s)):
            scene, log = a['scene_log_pairs'][slot]
            if scene in seen:
                continue
            seen.add(scene)
            rank, local_slot = divmod(slot, n_local)
            run = Path(plan['run'])
            rank_receipt = run / 'rollouts' / ('boundary_%06d' % a['policy_version']) / ('rank_%d_COMPLETE.json' % rank)
            record_root = run / 'official_feedback' / ledger
            row = {'scene_id': scene, 'source_log': log, 'role': 'train', 'policy_version': a['policy_version'],
                   'ledger': ledger, 'shard': shard, 'global_scene_slot': slot, 'rank': rank, 'local_slot': local_slot,
                   'compact': {'path': str(path), 'sha256': expected}, 'original_input_hashes': a['input_pins'],
                   'original_feedback_payload': {**stat_only(record_root / compact['item']['attempt'] / 'RESULTS.pt'), 'sha256': compact['item']['payload_sha256_declared']},
                   'original_requests': {**stat_only(record_root / 'REQUESTS.json'), 'sha256': a['input_pins']['REQUESTS']},
                   'original_rank_receipt': {**stat_only(rank_receipt), 'sha256': a['input_pins']['rank_COMPLETE'][rank]},
                   'failed_candidate_id': None, 'safe_sibling_candidate_id': None, 'native_flag_0': None,
                   'candidate_binding_status': 'PENDING_SELECTED_ORIGINAL_RECORD_VERIFICATION_NO_PROXY_READ'}
            selected.append(row)
            if len(selected) == 32:
                break
        if len(selected) == 32:
            break
    assert len(selected) == len({r['scene_id'] for r in selected}) == 32
    # These actual rank metadata files are small; read once and retain source pins.
    rank_cache = {}
    for row in selected:
        ref = row['original_rank_receipt']
        if ref['path'] not in rank_cache:
            rank_cache[ref['path']] = read(ref['path'], ref['sha256'])
        receipt = rank_cache[ref['path']]
        row['original_rollout_payload'] = {**stat_only(receipt['payload']['path']), 'sha256': receipt['payload']['sha256']}
        row['original_policy_identity'] = receipt['identity']['old_policy']
        row['original_rollout_id'] = receipt['identity']['rollout_id']
        assert row['original_policy_identity']['optimizer_step'] == row['policy_version']
    stat_totals = {}
    for name in ('original_feedback_payload', 'original_requests', 'original_rank_receipt', 'original_rollout_payload'):
        unique = {r[name]['path']: r[name] for r in selected}
        stat_totals[name] = {'unique_files': len(unique), 'known_bytes': sum(v['bytes'] or 0 for v in unique.values()), 'missing': [v['path'] for v in unique.values() if not v['exists']]}
    frame = {'schema': 'road_failure_direction_frame_v1', 'status': 'GROUP_FRAME_FROZEN_CANDIDATE_BINDING_PENDING_NOT_APPROVED',
             'domain': DOMAIN, 'seed': SEED, 'population': 'C1 TRAIN D-discriminative groups; one failed candidate per unique scene',
             'source_primary_result_sha256': PRIMARY_SHA, 'source_plan_sha256': PLAN_SHA, 'run': plan['run'],
             'registered_ledger_population': len(lookup), 'visited_ledgers': len(visited), 'selected_scenes': len(selected),
             'selected_logs': len({r['source_log'] for r in selected}), 'selected': selected,
             'selection_frame_sha256': digest(encoded(selected)), 'final_candidate_selection_sha256': None,
             'future_asset_stat_totals': stat_totals, 'visited_compact_receipts': visited}
    publish('FRAME32.json', frame)
    publish('METADATA_READ_RECEIPT.json', {'status': 'COMPLETE_METADATA_ONLY', 'logical_read_bytes': sum(r['bytes'] for r in reads),
             'read_cap_bytes': READ_CAP, 'prior_exploratory_read_conservative_allowance_bytes': 8 * 2**20,
             'total_content_read_upper_bound_bytes': 8 * 2**20 + sum(r['bytes'] for r in reads), 'reads': reads,
             'cpu_seconds': resource.getrusage(resource.RUSAGE_SELF).ru_utime + resource.getrusage(resource.RUSAGE_SELF).ru_stime,
             'wall_seconds': time.monotonic() - started, 'peak_actual_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
             'cpu_affinity': sorted(os.sched_getaffinity(0)), 'model_payloads': 0, 'trajectory_payloads': 0, 'cache_payloads': 0,
             'new_scores': 0, 'optimizer_updates': 0, 'GPU_calls': 0})
    print(json.dumps({'status': frame['status'], 'visited_ledgers': len(visited), 'selected_scenes': len(selected),
          'selected_logs': frame['selected_logs'], 'metadata_read_bytes': sum(r['bytes'] for r in reads), 'future_asset_stat_totals': stat_totals}))


if __name__ == '__main__':
    main()
