"""Final exact read-only receipt check; does not migrate or dispatch."""
from pathlib import Path
import hashlib
import json
import os
import re
import time

ROOT = Path('/mnt/project/st-world-policy-20261016')
P = ROOT/'audits/metadata_migration_preparation_v1'
HERE = Path(__file__).parent


def read(path, sha=None):
    path = Path(path)
    assert path.stat().st_size < 2**20
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if sha is not None:
        assert actual == sha, path
    return json.loads(raw), actual


prepared, prepared_sha = read(P/'PREPARED_V6.json', '1bd1b7cc4045ad5febf6ac831c5013263ce1af7c1cbda450117cdbea959e1f58')
q, qsha = read(P/'QUIESCENCE_V2.json', '46544ce8368d916a9d53b892fbdc380b174f771fbf67222f23a56bd1a15494e0')
cp, cpsha = read(P/'CHECKPOINT_COMMITMENTS_V1.json', '1bf12212b29202cd3f5cf80e73425ab5d663c778e7ba35df43a83bb2c27289ad')
read(cp['basis']['path'], cp['basis']['sha256'])
old, _ = read(prepared['old_policy'], prepared['old_policy_sha256'])
inv, _ = read(prepared['inventory'], prepared['inventory_sha256'])
assert set(q['checked_writer_roots']) == {r['root'] for r in old['claims']} | {r['root'] for r in old['legacy_reservations']} | {r['root'] for r in inv['roots']}
expected = {k: prepared[k] for k in ('old_policy_sha256','new_policy_sha256','old_ledger_sha256')}
expected.update(checkpoint_sha256=cpsha, pilot_inventory_sha256=prepared['inventory_sha256'])
assert q['bindings'] == expected
assert q['owner_role'] == 'Lead' and q['active_or_unknown_writers'] == [] and q['queued_writers_disabled']
assert q['future_admission_owner'] == 'Lead'
assert cp['remaining_bytes'] == 1983*2**30 and cp['free_floor_bytes'] == 1024*2**30
evidence = [read(e['path'], e['sha256'])[0] for e in q['process_evidence']]
census, interpretation = evidence
assert len(census['hosts']) == 8 and census['signals_sent'] is False
dispatch, dispatch_sha = read('/mnt/project/server_dispatch_policy.json', census['dispatch_policy_sha256'])
assert {r['requested_host'] for r in census['hosts']} == set(dispatch['allowed_hosts'])
observer = interpretation['observer_source']
assert hashlib.sha256(Path(observer['path']).read_bytes()).hexdigest() == observer['sha256']
matches = []
for host in census['hosts']:
    assert host['returncode'] == 0 and not host['error']
    result = host['result']
    assert not result['unreadable_processes'] and not result['oversize_cmdline_processes']
    matches.extend(result['matches'])
assert len(matches) == interpretation['matched_process_count'] == 4
assert all(str(observer['path']) in row['argv'] for row in matches)
assert interpretation['unknown_process_count'] == interpretation['cache_write_dispatchers_detected'] == 0
now = time.time()
assert q['checked_at_unix'] <= now < q['valid_until_unix']
assert cp['checked_at_unix'] <= now < cp['valid_until_unix']
assert now - census['recorded_unix'] < 20*60
for key in ('index_storage_registration', 'future_storage_registration'):
    reg, _ = read(prepared[key], prepared[key+'_sha256'])
    for name in ('CONTROL_CURRENT_INDEX_STARTED.json.az_01hij.tmp', 'COMPLETE.json.za_23hij.tmp'):
        assert any(re.fullmatch(pattern, name) for pattern in reg['file_patterns'])
controller = ROOT/'resources/run_bounded_metadata_v1.py'
controller_sha = hashlib.sha256(controller.read_bytes()).hexdigest()
assert controller_sha == 'a222cc453d236db5ffde0e21f42936b40f7f30108a3648cad23050ffe5df18d7'
fs = os.statvfs(ROOT)
available = fs.f_bavail*fs.f_frsize
# Stronger than needed: reserve full cache cap in addition to all original
# checkpoint commitments and the 1 TiB free floor. No existing files deducted.
assert available > (768+1983+1024)*2**30
result = {'schema':'st_final_metadata_admission_receipt_check_v1', 'checked_at_unix':now,
    'prepared_sha256':prepared_sha, 'controller_sha256':controller_sha,
    'quiescence_sha256':qsha, 'checkpoint_sha256':cpsha,
    'census_checked_at_unix':census['recorded_unix'], 'quiescence_valid_until_unix':q['valid_until_unix'],
    'checkpoint_valid_until_unix':cp['valid_until_unix'], 'filesystem_available_bytes':available,
    'checkpoint_remaining_bytes':cp['remaining_bytes'], 'scope':'bounded_control_files_only',
    'status':'PASS_RECEIPTS_NO_MIGRATION_OR_DISPATCH'}
(HERE/'FINAL_RECEIPTS_CHECK.json').write_text(json.dumps(result, sort_keys=True, indent=2)+'\n')
print(json.dumps(result,sort_keys=True))
