"""Read bounded explicit package files, source pins and seven pilot stat inventories."""
import ast
import argparse
import hashlib
import json
from pathlib import Path
import os
import stat
import time

ROOT = Path('/mnt/project/st-world-policy-20261016')
parser = argparse.ArgumentParser()
parser.add_argument('--prepared', type=Path, default=ROOT / 'audits/metadata_migration_preparation_v1/PREPARED_V5.json')
parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'PREPARED_PACKAGE_REVIEW.json')
args = parser.parse_args()
PREP, OUT = args.prepared, args.output
MiB = 2**20
GiB = 2**30
checks = {}
pins = []


def read(path, expected=None):
    path = Path(path)
    if path.stat().st_size > MiB:
        raise ValueError('not a bounded control/source artifact: ' + str(path))
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected is not None:
        assert digest == expected, path
    pins.append({'path': str(path), 'sha256': digest, 'bytes': len(raw)})
    return raw


def doc(path, expected=None):
    return json.loads(read(path, expected))


p = doc(PREP)
old = doc(p['old_policy'], p['old_policy_sha256'])
new = doc(p['new_policy'], p['new_policy_sha256'])
ledger_path = Path(old['state_dir']) / 'CLAIMS.json'
ledger = doc(ledger_path, p['old_ledger_sha256'])
inv = doc(p['inventory'], p['inventory_sha256'])
for key in ('schema', 'campaign', 'state_dir', 'cache_cap_bytes', 'free_reserve_bytes', 'ledger_reserve_bytes'):
    checks['preserved_' + key] = old[key] == new[key]
checks['old_claims_unchanged'] = all(row in new['claims'] for row in old['claims'])
checks['old_reservations_unchanged'] = new['legacy_reservations'][:len(old['legacy_reservations'])] == old['legacy_reservations']
checks['same_single_ledger'] = new['state_dir'] == old['state_dir'] == str(ROOT / 'resources/cache_storage_ledger_v1')
checks['cap_floor'] = new['cache_cap_bytes'] == 768 * GiB and new['free_reserve_bytes'] == 1024 * GiB
package = new['metadata_package']
checks['one_package_maximum'] = package['maximum_bytes'] == 4 * GiB
checks['package_exact_baseline'] = (package['baseline_policy_sha256'] == p['old_policy_sha256'] and
    package['baseline_ledger_sha256'] == p['old_ledger_sha256'] and
    package['baseline_claims'] == {r['root']: r for r in old['claims']} and
    package['baseline_legacy_reservations'] == old['legacy_reservations'])
added_pilots = new['legacy_reservations'][len(old['legacy_reservations']):]
checks['seven_pilots'] = len(inv['roots']) == len(added_pilots) == 7
checks['pilot_roots_exact'] = {r['root'] for r in added_pilots} == {r['root'] for r in inv['roots']}
for root in inv['roots']:
    known = {e['relative']: e for e in root['entries']}
    actual = set()
    stack = [(Path(root['root']), '.')]
    while stack:
        path, rel = stack.pop()
        if len(actual) > 100:
            raise ValueError('pilot inventory unexpectedly expanded')
        st = path.lstat()
        actual.add(rel)
        assert rel in known
        e = known[rel]
        for key, value in {'device': st.st_dev, 'inode': st.st_ino, 'size': st.st_size,
                           'allocated_bytes': st.st_blocks * 512, 'mtime_ns': st.st_mtime_ns,
                           'ctime_ns': st.st_ctime_ns}.items():
            assert e[key] == value, (path, key)
        if stat.S_ISDIR(st.st_mode):
            stack.extend((c, c.relative_to(root['root']).as_posix()) for c in path.iterdir())
        else:
            assert stat.S_ISREG(st.st_mode) and st.st_nlink == 1
    assert actual == set(known)
checks['pilots_stat_unchanged'] = True
checks['pilot_rounded_charge'] = all(r['reserved_bytes'] == max(MiB, ((i['charged_bytes'] + MiB - 1)//MiB)*MiB)
    for r in added_pilots for i in inv['roots'] if r['root'] == i['root'])
added_claims = [r for r in new['claims'] if r['root'] not in {v['root'] for v in old['claims']}]
checks['two_new_claims'] = len(added_claims) == 2
runtime = Path(p['runtime'])
registrations = {}
for row in added_claims:
    reg = doc(row['registration_path'], row['registration_sha256'])
    registrations[reg['writer_kind']] = reg
    assert reg['output_root'] == row['root']
    assert reg['storage_requirements'] == {k: row[k] for k in ('output_bytes', 'staging_bytes', 'metadata_bytes')}
    assert row['cache_identity'] == row['registration_sha256']
    assert Path(row['root']).is_relative_to(ROOT / 'caches')
    assert reg['cwd'] == str(runtime)
    assert reg['callback_protocol'] == 'st_metadata_storage_check_v1'
    for pin in reg['source_pins']:
        read(pin['path'], pin['sha256'])
    module = reg['command'][2]
    entry = runtime.joinpath(*module.split('.')).with_suffix('.py')
    tree = ast.parse(read(entry))
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.startswith('METADATA_STORAGE_CALLBACK_'):
                    constants[target.id] = ast.literal_eval(node.value)
    assert constants['METADATA_STORAGE_CALLBACK_PROTOCOL'] == reg['callback_protocol']
    dependencies = constants['METADATA_STORAGE_CALLBACK_DEPENDENCIES']
    assert {Path(v['path']) for v in reg['source_pins']} == {entry} | {runtime.joinpath(*d.split('.')).with_suffix('.py') for d in dependencies}
    for pin in reg.get('preexisting_pins', []):
        read(Path(reg['output_root']) / pin['relative_path'], pin['sha256'])
checks['registration_source_pins_paths_protocol'] = True
index = registrations['current_index']
future = registrations['future_preparation']
checks['current_limits'] = index['limits'] == {'max_depth': 0, 'max_disk_bytes': GiB,
    'max_entries': 100, 'max_rss_bytes': 4*GiB, 'max_seconds': 1200}
checks['current_inner_disk_leaves_control'] = index['command'][index['command'].index('--max-disk-bytes')+1] == str(1008*MiB)
checks['future_limits'] = future['limits'] == {'max_depth': 1, 'max_disk_bytes': 32*MiB,
    'max_entries': 100, 'max_rss_bytes': 4*GiB, 'max_seconds': 450}
plan = doc(p['future_plan'], p['future_plan_sha256'])
prep_registration = doc(p['future_preparation_registration'], p['future_preparation_registration_sha256'])
checks['future_plan_registration'] = prep_registration['plan_identity'] == plan['identity'] == p['plan_identity'] and prep_registration['plan_sha256'] == p['future_plan_sha256']
checks['future_metadata_only'] = all(plan[k] is False and prep_registration[k] is False for k in ('gpu_authorized','teacher_extraction_authorized','training_authorized'))
checks['future_budget'] = (plan['selected_scene_count'] == 32 and plan['limits']['workers'] == 1 and
    plan['limits']['max_seconds'] == 450 and plan['limits']['max_rss_bytes'] == 4*GiB and
    plan['limits']['max_metadata_bytes'] == 32*MiB and plan['limits']['read_mib_per_second'] == 64)
old_plan = doc(ROOT/'caches/future_metadata_qualification32_v1/PLAN.json')
checks['future_original_scene_order_preserved'] = plan['selected_rows'] == old_plan['selected_rows'] and plan['ordered_scene_ids_hash'] == old_plan['ordered_scene_ids_hash']
added = sum(r['reserved_bytes'] for r in added_pilots) + sum(sum(r[k] for k in ('output_bytes','staging_bytes','metadata_bytes')) for r in added_claims)
baseline_reserved = old['ledger_reserve_bytes'] + sum(r['reserved_bytes'] for r in old['legacy_reservations']) + sum(sum(r[k] for k in ('output_bytes','staging_bytes','metadata_bytes')) for r in old['claims'])
checks['added_1072MiB'] = added == p['added_reserved_bytes'] == 1072 * MiB
checks['aggregate_below_package'] = added <= 4*GiB
checks['aggregate_below_cache_cap'] = baseline_reserved + added <= new['cache_cap_bytes']
fs = os.statvfs(ROOT)
result = {'schema':'st_prepared_metadata_package_independent_review_v1', 'checked_at_unix':time.time(),
    'status':'PASS_PREPARED_PACKAGE_PENDING_LIVE_ADMISSION', 'checks':checks, 'pins':pins,
    'old_ledger_read_only_no_mutation':True, 'added_reserved_bytes':added,
    'baseline_reserved_bytes':baseline_reserved, 'aggregate_reserved_bytes':baseline_reserved+added,
    'aggregate_remaining_bytes':new['cache_cap_bytes']-baseline_reserved-added,
    'filesystem_available_bytes':fs.f_bavail*fs.f_frsize,
    'not_covered':['fresh quiescence receipt','checkpoint CURRENT receipt','outer CPU/RSS/deadline/pipe monitor',
        'actual migration','actual build','full future metadata','teacher/GPU/training qualification']}
assert all(checks.values()), checks
OUT.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('pins',)},sort_keys=True))
