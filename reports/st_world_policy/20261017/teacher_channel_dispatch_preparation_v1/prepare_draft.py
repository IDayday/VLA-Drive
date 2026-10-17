"""Fixed PCA256 dispatch handoff: small metadata/stat only; never dispatches."""
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import stat
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ROOT = Path('/mnt/project/st-world-policy-20261016')
OLD_ID = 'SP2_fulltrain_seed42_official_5k_v1'
NEW_ID = 'SP2_PCA256_fulltrain_seed42_official_5k_v1'
OLD_REG = ROOT / 'formal/spatial/registrations' / OLD_ID
OLD_RUN = ROOT / 'formal/spatial/runs' / OLD_ID
NEW_REG = ROOT / 'formal/spatial/registrations' / NEW_ID
CONFIG_DIR = REPO / 'configs/st_world_policy/teacher_channel_control_v1'
PINS = {}
READ_BYTES = 0


def read(path):
    global READ_BYTES
    path = Path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    try:
        s = os.fstat(fd)
        if not stat.S_ISREG(s.st_mode) or s.st_size > 1024 * 1024:
            raise ValueError(f'Not a small regular metadata file: {path}')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            raw = stream.read(1024 * 1024 + 1)
        if len(raw) != s.st_size:
            raise ValueError('Metadata changed during read')
    finally:
        os.close(fd)
    READ_BYTES += len(raw)
    if READ_BYTES > 16 * 1024 * 1024:
        raise ValueError('Fixed handoff metadata budget exceeded')
    PINS[str(path)] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    return json.loads(raw)


def write(name, value):
    # Own review material only, not the live registration or model/cache namespace.
    out = HERE / name
    raw = (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()
    if out.exists() and out.read_bytes() != raw:
        raise ValueError(f'Refusing to overwrite prior draft: {out}')
    out.write_bytes(raw)


def main():
    started = time.monotonic()
    old = read(OLD_REG / 'REGISTRATION.json')
    config = read(CONFIG_DIR / 'PCA256.json')
    read(CONFIG_DIR / 'TARGET_CONTRACT_256.json')
    old_cfg = read(OLD_REG / 'CONFIG.json')
    assert config['execution'] == old_cfg['execution'] == old['execution']
    assert config['optimizer'] == old_cfg['optimizer']
    assert config['runtime']['save_every'] == 1000
    assert config['runtime']['max_seconds'] == 28800
    assert config['execution']['stop_after'] == 5000
    assert config['execution']['max_updates'] == 20000
    assert config['optimizer']['warmup'] == 1000
    assert config['space_time']['teacher']['channels'] == 256
    for n in ('RUN_APPROVAL.json', 'RESOURCE_LEASE.json', 'FINGERPRINT.json',
              'DATA_AND_STORAGE_BOUNDARY.json', 'reviews/EFFICIENCY.json'):
        read(OLD_REG / n)
    result = read(OLD_RUN / 'RESULT.json')
    assert result['completed'] == 5000 and result['exposure'] == 159992
    complete_path = OLD_RUN / 'checkpoints' / result['tag'] / 'COMPLETE.json'
    complete = read(complete_path)
    payload_stats = {}
    for name, receipt in complete['files'].items():
        s = (complete_path.parent / name).stat()
        assert stat.S_ISREG(s.st_mode) and s.st_size == receipt['bytes']
        payload_stats[name] = {'actual_bytes_stat_only': s.st_size,
                               'matches_existing_receipt_size': True}
    train = Path(config['data']['teacher_root'])
    dev = ROOT / 'caches/full_current_native384_v1/dev'
    train_stage = read(train / 'STAGE_COMPLETE.json')
    train_verification = read(train / 'VERIFICATION.json')
    dev_complete = read(dev / 'COMPLETE.json')
    dev_verification = read(dev / 'VERIFICATION.json')
    assert train_stage['status'] == 'COMPLETE'
    assert train_verification['status'] == dev_verification['status'] == 'VERIFIED'
    assert train_verification['images_verified'] == 304776
    assert dev_verification['images_verified'] == 5088
    pca = Path(config['data']['pca_path'])
    for n in ('PCA.json', 'CHANNEL_SCOPE.json', 'COMPLETE.json', 'NAMESPACE_COMPLETE.json'):
        read(pca / n)
    topology = read(ROOT / 'audits/efficiency_official_runtime_fdf25f99/TOPOLOGY_COMPARISON_v1.json')
    timing = {}
    for name in ('8gpu', '4gpu_A', '4gpu_B'):
        p = ROOT / f'profiles/world_official_runtime_{name}_v1/TIMING_SUMMARY.json'
        d = read(p)
        timing[name] = {'identity': d['identity'], 'world_size': d['world_size'],
                       'windows': [{k: v for k, v in w.items() if k != 'steps'} for w in d['windows']]}
    draft = copy.deepcopy(old)
    draft['schema'] = 'st_world_run_dispatch_draft_v1'
    draft['run_id'] = NEW_ID
    draft['implementer_agents'] = sorted(set(old['implementer_agents'] + ['/root/st_indexprep_boundary_fix']))
    for key, filename, source in [('config', 'CONFIG.json', CONFIG_DIR / 'PCA256.json'),
                                  ('target_contract', 'TARGET_CONTRACT.json', CONFIG_DIR / 'TARGET_CONTRACT_256.json')]:
        draft['artifacts'][key] = {'path': str(NEW_REG / filename), 'sha256': PINS[str(source)]['sha256']}
    for key, filename in [('environment', 'ENVIRONMENT.json'), ('resource_lease', 'RESOURCE_LEASE.json')]:
        draft['artifacts'][key] = {'path': str(NEW_REG / filename), 'sha256': None,
                                  'status': 'LEAD_MUST_CREATE_AND_BIND_CURRENT_ARTIFACT'}
    draft['status'] = 'DRAFT_NOT_A_REGISTRATION_OR_APPROVAL'
    draft['formal_launch_allowed'] = False
    write('REGISTRATION_DRAFT.json', draft)
    fs = os.statvfs(ROOT)
    new_output = Path(config['runtime']['output_root']) / NEW_ID
    evidence = {
        'schema': 'st_teacher_channel_dispatch_evidence_draft_v1',
        'status': 'AUTHOR_PREPARATION_NOT_INDEPENDENT_APPROVAL',
        'formal_launch_allowed': False,
        'source_sha': old['source_sha'], 'source_fingerprint': old['source_fingerprint'],
        'config_author_commit': 'bd2d1dd9a11ed3b236a95b680884f1495eddbe6f',
        'new_registration_path': str(NEW_REG / 'REGISTRATION.json'),
        'new_output': str(new_output), 'new_output_exists_at_check': new_output.exists(),
        'old_SP2_result': result, 'old_SP2_checkpoint_receipt': str(complete_path),
        'old_SP2_checkpoint_file_stat_only': payload_stats,
        'old_SP2_checkpoint_total_bytes': sum(x['actual_bytes_stat_only'] for x in payload_stats.values()),
        'schedule': {'max_updates': 20000, 'warmup': 1000, 'stop_after': 5000,
                     'world_size': 4, 'micro_batch': 4, 'global_batch': 32, 'caller_accumulation': 2,
                     'expected_exposure_from_unchanged_sampler': 159992,
                     'new_256_actual_exposure': None},
        'teacher': {'train_stage': train_stage, 'train_verification': train_verification,
                    'dev_verification': dev_verification,
                    'dev_complete_summary': {k: v for k, v in dev_complete.items() if k != 'shards'},
                    'train_complete_stat_bytes': (train / 'COMPLETE.json').stat().st_size,
                    'new_teacher_bytes_required': 0, 'payload_revalidated_here': False},
        'inherited_profile': timing, 'topology_with_saves': topology['combined_windows'],
        'storage_proposal': {'checkpoint_count': 5, 'checkpoint_upper_bytes': 40 * 1024**3,
                             'native_reserve_bytes': 240 * 1024**3,
                             'five_checkpoints_at_old_SP2_size_bytes': 5 * sum(x['actual_bytes_stat_only'] for x in payload_stats.values()),
                             'contingency_bytes_at_upper_bound': 40 * 1024**3,
                             'automatic_cleanup': False, 'free_bytes_statvfs': fs.f_bavail * fs.f_frsize,
                             'quota_verified': False, 'new_reservation_created': False},
        'pending': ['Current environment/lease/registration by Lead',
                    'Independent affected-part evidence, correctness and Systems decisions',
                    'Fingerprint-bound approval and current GPU locks by existing launcher'],
        'actions': {'GPU': False, 'SSH': False, 'dispatch': False, 'payload_read': False,
                    'large_manifest_read': False, 'ledger_write': False, 'approval_signed': False},
        'pins': PINS,
    }
    write('SYSTEMS_EVIDENCE_DRAFT.json', evidence)
    metrics = {'source_metadata_read_bytes': READ_BYTES,
               'wall_seconds': time.monotonic() - started,
               'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
               'cpu_affinity': sorted(os.sched_getaffinity(0)),
               'payload_bytes_read': 0}
    write('PREPARATION_COST.json', metrics)
    print(json.dumps(metrics, sort_keys=True))


if __name__ == '__main__':
    main()
