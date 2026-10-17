"""Finite metadata-only inspection; no imports of training/scoring runtimes."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import socket
import time

os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
signal.alarm(120)
START = time.monotonic()
ROOT = Path('/mnt/project/st-world-policy-20261016/formal')
OUT = Path(__file__).resolve().parent
pins = []

def read(path, *, mutable=False):
    path = Path(path)
    assert path.stat().st_size <= 512 * 1024, 'Metadata-only size bound'
    blob = path.read_bytes()
    pins.append({'path': str(path), 'sha256': hashlib.sha256(blob).hexdigest(),
                 'bytes': len(blob), 'mutable_snapshot': mutable})
    return json.loads(blob)

def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def receipt(folder, status, world):
    value = read(folder / 'COMPLETE.json')
    assert value['schema'] == 'st_native_checkpoint_v1'
    assert identity({k: v for k, v in value.items() if k != 'receipt_identity'}) == value['receipt_identity']
    assert value['state']['identity'] == status['identity'] and value['world_size'] == world
    assert value['state']['tag'] == folder.name
    expected = {'mp_rank_00_model_states.pt'} | {f'rng_rank{i}.pt' for i in range(world)} | {
        f'bf16_zero_pp_rank_{i}_mp_rank_00_optim_states.pt' for i in range(world)}
    assert set(value['files']) == expected == {p.name for p in folder.glob('*.pt')}
    assert all((folder / name).stat().st_size == info['bytes'] for name, info in value['files'].items())
    return {'path': str(folder / 'COMPLETE.json'), 'sha256': pins[-1]['sha256'],
        'receipt_identity': value['receipt_identity'], 'state': value['state'],
        'world_size': world, 'files': value['files'], 'inventory_and_sizes_match': True,
        'native_payload_hashes_recomputed': False}

runs, table = {}, []
for family, arms, suffix in [('consequence', [f'C{i}' for i in range(5)], '4096_v1'),
                            ('spatial', [f'SP{i}' for i in range(4)], '5k_v1')]:
    for arm in arms:
        name = f'{arm}_fulltrain_seed42_official_{suffix}'
        root = ROOT / family / 'runs' / name
        reg_root = ROOT / family / 'registrations' / name
        reg = read(reg_root / 'REGISTRATION.json')
        reg_sha = pins[-1]['sha256']
        config = read(reg_root / 'CONFIG.json')
        assert pins[-1]['sha256'] == reg['artifacts']['config']['sha256']
        assert read(root / 'resolved_config.json') == config
        approval = read(reg_root / 'RUN_APPROVAL.json')
        assert approval['formal_launch_allowed'] and approval['fingerprint']['registration_sha256'] == reg_sha
        assert reg['execution'] == config['execution'] and reg['execution']['scope'] == 'FORMAL'
        status = read(root / 'status.json', mutable=True)
        status_sha = pins[-1]['sha256']
        assert status['source_sha'] == reg['source_sha']
        latest_path = root / 'checkpoints/latest'
        latest_bytes = latest_path.read_bytes()
        latest = latest_bytes.decode().strip()
        pins.append({'path': str(latest_path), 'sha256': hashlib.sha256(latest_bytes).hexdigest(),
                     'bytes': len(latest_bytes), 'mutable_snapshot': True})
        complete_dirs = [p for p in (root / 'checkpoints').iterdir()
                         if p.is_dir() and (p / 'COMPLETE.json').is_file()]
        newest = max(complete_dirs, key=lambda p: (int(p.name.split('_')[1]), p.name))
        assert newest.name == latest, 'latest pointer is stale or incompatible'
        latest_receipt = receipt(newest, status, reg['execution']['world_size'])
        assert status['completed'] >= latest_receipt['state']['completed']
        endpoint_steps = (512, 1024, 4096) if family == 'consequence' else (5000,)
        endpoints = {}
        for step in endpoint_steps:
            found = sorted(p for p in complete_dirs if int(p.name.split('_')[1]) == step)
            assert len(found) <= 1, 'Ambiguous complete endpoint'
            endpoints[str(step)] = receipt(found[0], status, reg['execution']['world_size']) if found else None
        result = read(root / 'RESULT.json') if (root / 'RESULT.json').exists() else None
        runs[arm] = {'run': str(root), 'registration_sha256': reg_sha,
            'status_snapshot_sha256': status_sha,
            'status_snapshot': {k: status.get(k) for k in ('status','completed','updated_unix','started_unix',
                'source_sha','identity','host','pid','tag','pause_reason','termination_reason','exposure','epoch','offset')},
            'registered_budget': reg['execution'], 'latest_complete': latest_receipt,
            'fixed_endpoints': endpoints, 'result_receipt': result,
            'training_process_liveness': 'NOT_REMOTE_VERIFIED; status snapshot only; no SSH'}
        table.append({'arm': arm, 'reported_state': status['status'], 'reported_step': status['completed'],
            'status_updated_utc': datetime.fromtimestamp(status['updated_unix'], timezone.utc).isoformat(),
            'reported_host': status['host'], 'latest_complete_step': latest_receipt['state']['completed'],
            'latest_tag': latest, 'latest_receipt_sha256': latest_receipt['sha256'],
            'endpoint_target': config['execution']['stop_after']})

previous_pins = {}
for report in ('consequence_full_dev_snapshot_v1', 'spatial_SP0_full_dev_snapshot_v1'):
    prior = read(OUT.parent / report / 'EVIDENCE_PINS.json')
    previous_pins.update({p['path']: p['sha256'] for p in prior['pins']})
evaluations = {}
for folder in sorted((ROOT / 'evaluation/jobs').iterdir()):
    if not folder.is_dir():
        continue
    complete_path = folder / 'official_score/COMPLETE.json'
    export_path = folder / 'export/COMPLETE_rank0.json'
    entry = {'score_complete_present': complete_path.is_file(), 'export_complete_present': export_path.is_file()}
    if complete_path.is_file():
        complete = read(complete_path)
        assert complete['status'] == 'COMPLETE' and complete['valid_scored'] == 1696
        assert complete['invalid'] == complete['unresolved'] == 0
        assert previous_pins[str(complete_path)] == pins[-1]['sha256']
        assert complete['score_units'] == '0..1'
        entry.update({'complete_sha256': pins[-1]['sha256'], 'aggregate': complete['aggregate'],
            'prior_full_scene_hash_audit': 'UNCHANGED_PRIOR_VERIFIED_RECEIPT', 'logs': len(complete['by_log'])})
    entry['half_results_read'] = False
    evaluations[folder.name] = entry

known_exports = {}
for job_id, pid in [('SP1_step005000_dev1696_seed42_v1', 440343),
                    ('C3_step000512_dev1696_seed42_v1', 445432),
                    ('SP2_step005000_dev1696_seed42_v1', 449050),
                    ('C1_step001024_dev1696_seed42_v1', 449179)]:
    path = Path(f'/proc/{pid}/cmdline')
    argv = path.read_bytes() if path.exists() else b''
    expected = str(ROOT / 'evaluation/jobs' / job_id / 'export').encode()
    assert not argv or expected in argv
    reg_root = ROOT / 'evaluation/registrations' / job_id
    launch_paths = sorted(reg_root.glob('invocations/*/LAUNCH.json'))
    assert len(launch_paths) == 1
    launch = read(launch_paths[0])
    assert launch['pid'] == pid and launch['host'] == socket.gethostname()
    route = read(launch['route']['path'])
    assert pins[-1]['sha256'] == launch['route']['sha256']
    assert route['job_id'] == job_id and route['gpu'] == launch['gpu']
    native = read(route['complete_receipt']['path'])
    assert pins[-1]['sha256'] == route['complete_receipt']['sha256']
    assert native['receipt_identity'] == route['receipt_identity']
    assert native['state']['identity'] == route['runtime_identity']
    known_exports[job_id] = {'host': socket.gethostname(), 'pid': pid,
        'process_exists': bool(argv), 'matches_expected_output': bool(argv and expected in argv),
        'cmdline_sha256': hashlib.sha256(argv).hexdigest() if argv else None,
        'native_shell_exit_code': None, 'gpu': launch['gpu'],
        'campaign_export_slot': launch['campaign_export_slot'],
        'launch_receipt_path': str(launch_paths[0]), 'route': launch['route'],
        'launch_source_sha': launch['source_sha'], 'native_complete': route['complete_receipt']}

dependencies = {}
for arm, step in [('C1', 1024), ('C4', 512), ('SP2', 5000)]:
    job_id = f'{arm}_step{step:06d}_dev1696_seed42_v1'
    endpoint = runs[arm]['fixed_endpoints'][str(step)]
    reg = ROOT / 'evaluation/registrations' / job_id / 'REGISTRATION.json'
    dependencies[job_id] = {'checkpoint_ready': endpoint is not None,
        'complete_receipt': endpoint, 'evaluation_registration_present': reg.exists(),
        'evaluation_outputs': evaluations.get(job_id),
        'next_action': ('ALREADY_LAUNCHED_BY_LEAD; preserve active export and wait for complete prediction receipt'
            if job_id in known_exports and known_exports[job_id]['process_exists'] else
            'WAITING_EXPORT_SLOT; four registered exports active; no additional dispatch')
            if endpoint else 'WAITING_EXACT_COMPLETE_NATIVE_CHECKPOINT_THEN_EXPORT_SLOT; no fallback or dispatch'}
snapshot = {'schema': 'st_formal_metadata_dependency_snapshot_v1', 'checked_utc': datetime.now(timezone.utc).isoformat(),
    'source_commit': '4ae8d6c2b5d8ee198fe038689b63b3da7e14cb2a', 'runs': runs,
    'fixed_evaluations': evaluations, 'known_local_exports': known_exports,
    'new_dependencies': dependencies, 'evidence_pins': pins,
    'limits': 'Only small metadata read; no SSH, GPU, scoring, training, dispatch, full weight hashing or partial evaluation rows',
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'wall_seconds': time.monotonic() - START,
    'peak_RSS_MiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
(OUT / 'SNAPSHOT.json').write_text(json.dumps(snapshot, indent=2, sort_keys=True) + '\n')
with (OUT / 'STATUS.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(table[0]))
    writer.writeheader(); writer.writerows(table)
print(json.dumps({'checked_utc': snapshot['checked_utc'], 'status': table,
    'dependencies': {k: {x: v[x] for x in ('checkpoint_ready','evaluation_registration_present','next_action')} for k,v in dependencies.items()},
    'known_local_exports': known_exports, 'wall_seconds': snapshot['wall_seconds'],
    'peak_RSS_MiB': snapshot['peak_RSS_MiB']}), flush=True)
