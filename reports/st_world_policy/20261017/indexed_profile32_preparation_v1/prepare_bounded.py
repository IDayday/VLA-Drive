"""Lead-only, one bounded CPU invocation of the reviewed PROFILE factory.

Reuse the existing supervisor and factory, preserving their original evidence.
The only supervisor adaptation locks the mutable counter with its writer lock.
This emits metadata; it cannot launch the student or build teacher tensors.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import runpy
import subprocess
import sys

RUNTIME = Path('/mnt/project/VLA-Drive-st-indexed-profile-runtime-v1-20261017')
SOURCE = '2d5e9483a8c8bc80b71348211756f3cab6aa21b1'
HELPERS = Path('/mnt/project/VLA-Drive-st-spatial-probe-runtime-v2-20261017/reports/st_world_policy/20261017/spatial_probe_actual_preparation_v1')
HANDOFF = Path('/mnt/project/VLA-Drive-st-indexed-composition-20261017/reports/st_world_policy/20261017/indexed_profile32_handoff_v1')
CAMPAIGN = Path('/mnt/project/st-world-policy-20261016')
PARENT = CAMPAIGN / 'smoke/world_indexed_provider_native2_v1'
ROOT = PARENT / 'profile32_v1'
ATTEMPT = ROOT / 'preparation_control_v1'
READ_CAP = (4 << 30) - 1590864 - (16 << 20)
sys.path.insert(0, str(HELPERS))
from prepare_bindings import write_once, read, control_lock
import supervise_once as supervisor
from observed_cli import Counters, proc_io


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def commands():
    old = '/mnt/project/VLA-Drive-st-indexed-qualified-runtime-v1-20261017'
    cfg = str(HANDOFF / 'INPUT_PROFILE_CONFIG.json')
    first = ['tools.st_world_policy.qualify_indexed_provider', 'prepare-profile-migration',
             '--parent-descriptor', str(PARENT / 'PROVIDER.json'), '--parent-runtime', old,
             '--output', str(ROOT / 'SOURCE_MIGRATION.json')]
    second = ['tools.st_world_policy.teacher_provider',
              '--request-index', str(CAMPAIGN / 'caches/future32_request_metadata_index_v1'),
              '--source-index', str(CAMPAIGN / 'caches/full_current_metadata_index_v1'),
              '--teacher-config', str(CAMPAIGN / 'audits/history_data/native384_train64_cost_v1/resolved_config_build_rank0.json'),
              '--qualification-result', str(CAMPAIGN / 'audits/indexed_provider_composition_run_v1/RESULT.json'),
              '--output', str(ROOT / 'PROVIDER.json'),
              '--max-cached-bytes', '14159232', '--max-scene-mib', '64', '--max-batch-images', '8',
              '--max-metadata-cache-mib', '16', '--training-config', cfg,
              '--pca-fit-cache-root', str(CAMPAIGN / 'audits/history_data/native384_train64_cost_v1'),
              '--pca-fit-cache-index', str(CAMPAIGN / 'caches/original_pca_fit64_metadata_index_v1'),
              '--configured-output', str(ROOT / 'configured'),
              '--profile-parent-descriptor', str(PARENT / 'PROVIDER.json'),
              '--profile-source-migration', str(ROOT / 'SOURCE_MIGRATION.json'),
              '--profile-data-manifest', str(CAMPAIGN / 'audits/indexed_composition_preparation_v1/SELECTED_SCENES.json'),
              '--profile-reference-config', str(PARENT / 'configured/training_config.json')]
    third = [str(HANDOFF / 'relocate_config.py'), '--factory-root', str(ROOT),
             '--checkpoint-output-root', str(CAMPAIGN / 'profile/world_indexed_provider_profile32_updates64_v1')]
    return [first, second, third]


def child():
    baseline = proc_io()
    sys.path.insert(0, str(RUNTIME))
    import threading
    import torch
    counter = Counters(ATTEMPT / 'COUNTERS.json', torch, READ_CAP, baseline)
    counter.data['phase'] = 'PROFILE_METADATA_FACTORY_ONLY'
    counter.snapshot()
    thread = threading.Thread(target=counter.monitor, daemon=True)
    thread.start()
    try:
        for index, argv in enumerate(commands()):
            sys.argv = argv
            try:
                if index < 2:
                    runpy.run_module(argv[0], run_name='__main__')
                else:
                    runpy.run_path(argv[0], run_name='__main__')
            except SystemExit as exc:
                if exc.code not in (0, None):
                    raise
    finally:
        counter.stop.set()
        thread.join(timeout=2)
        snap = counter.snapshot()
        if snap['limit_exceeded']:
            raise RuntimeError(snap['limit_exceeded'])
        if torch.cuda.is_initialized():
            raise RuntimeError('Unexpected CUDA initialization')


def size():
    paths = [PARENT, *PARENT.rglob('*')]
    if len(paths) > 256 or any(p.is_symlink() for p in paths):
        raise ValueError('Bounded parent metadata namespace changed')
    return sum(max(p.stat().st_size, p.stat().st_blocks * 512) for p in paths)


def main():
    if sys.argv[1:] == ['--owned-child']:
        child()
        return
    if sys.argv[1:]:
        raise ValueError('This fixed preparation accepts no resource overrides')
    os.sched_setaffinity(0, {0, 1})
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=RUNTIME, text=True).strip() == SOURCE
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=RUNTIME).strip()
    for name, expected in {
        'supervise_once.py': 'a29527dfbe23a4563a66266532f44644f5f184f4a8f59842883524632a63f2d6',
        'observed_cli.py': 'c1af891b440663d75ca3b75c1227f5a7d4247d62319a566214848e0a5550f643',
        'prepare_bindings.py': '31d2d302d5dd441df9251762e34cd9d1762afc3463cdabfe8dbf5b1f12577c84',
    }.items():
        assert sha(HELPERS / name) == expected
    assert sha(HANDOFF / 'HANDOFF.json') == '0d085e6ded1010852ae047c69217d38307d93ba4c2be2c2dfa719bd04ef32f99'
    assert sha(HANDOFF / 'relocate_config.py') == '33eb91e4afbc0cc899e99643f37267ac4e5a93ee7755ecdbc112d20ea431f85e'
    handoff = json.loads((HANDOFF / 'HANDOFF.json').read_text())
    for key in ('input_config', 'data_manifest', 'base_binding', 'parent_descriptor', 'reference_config', 'target_contract'):
        assert sha(handoff[key]['path']) == handoff[key]['sha256'], key
    assert handoff['source_sha'] == SOURCE and handoff['metadata_output_root'] == str(ROOT)
    assert handoff['execution']['stop_after'] == 64 and handoff['execution']['sample_limit'] == 32
    assert not ROOT.exists(), 'Prior attempt preserved; no automatic retry'
    assert size() + (8 << 20) < 32 << 20
    ROOT.mkdir()
    ATTEMPT.mkdir()
    write_once(ATTEMPT / 'ARGV.json', commands())
    write_once(ATTEMPT / 'ADMITTED.json', {
        'schema': 'st_indexed_profile32_metadata_admission_v1',
        'status': 'ONE_SHOT_PROFILE_CONFIG_PREPARE', 'script_sha256': sha(__file__),
        'source_sha': SOURCE, 'scope': 'PROFILE_METADATA_ONLY', 'gpu': False,
        'existing_parent_preserved': True, 'parent_root_inclusive_cap_bytes': 32 << 20,
        'logical_read_budget_bytes': 4 << 30, 'rss_budget_bytes': 2 << 30, 'wall_seconds': 180,
        'known_existing_pca_read_bytes': 1590864, 'new_teacher_tensor_bytes': 0,
        'training_started': False, 'monitor_lock_wait_max_seconds': 2,
    })
    resource.setrlimit(resource.RLIMIT_FSIZE, (4 << 20, 4 << 20))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    # The mutable COUNTERS writer uses this same lock. All immutable reads and
    # strict inode/size/hash checks remain original; lock timeout still fails.
    original_read = supervisor.read
    def locked_counter_read(path, *args, **kwargs):
        if Path(path) == ATTEMPT / 'COUNTERS.json':
            with control_lock(ATTEMPT):
                return original_read(path, *args, **kwargs)
        return original_read(path, *args, **kwargs)
    supervisor.read = locked_counter_read
    sys.path.insert(0, str(RUNTIME))
    from tools.st_world_policy.metadata_storage import process_tree_rss
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2')
    with (ATTEMPT / 'RUN.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = supervisor.supervise([sys.executable, __file__, '--owned-child'],
            cwd=str(RUNTIME), env=env, cpus=[0, 1], descriptors=[lock.fileno()],
            attempt=ATTEMPT, wall=180, rss=2 << 30, rss_reader=process_tree_rss)
    expected = all((ROOT / p).is_file() for p in (
        'SOURCE_MIGRATION.json', 'PROVIDER.json', 'configured/training_config.json',
        'configured/PROVIDER_CHANNEL_SCOPE.json', 'configured/PROVIDER_CONFIG_LINEAGE.json',
        'TRAINING_CONFIG.json', 'EXECUTION_CONFIG_LINEAGE.json'))
    used = size()
    ok = result['exit_code'] == 0 and result['limit_reason'] is None and expected and used + (256 << 10) <= 32 << 20
    write_once(ATTEMPT / 'RESULT.json', {'status': 'COMPLETE' if ok else 'FAILED_NO_AUTO_RETRY',
        'resources': result, 'root_inclusive_before_receipt_bytes': used,
        'expected_files_present': expected, 'new_teacher_tensor_bytes': 0, 'training_started': False,
        'monitor_lock_wait_max_seconds': 2})
    assert size() <= 32 << 20
    print(json.dumps({'status': 'COMPLETE' if ok else 'FAILED_NO_AUTO_RETRY', 'result': str(ATTEMPT / 'RESULT.json')}))
    raise SystemExit(0 if ok else 2)


if __name__ == '__main__':
    main()
