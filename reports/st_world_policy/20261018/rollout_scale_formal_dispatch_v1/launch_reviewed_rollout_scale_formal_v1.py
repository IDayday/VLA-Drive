"""Lead-only launch/resume of the single reviewed 4096-update rollout-scale arm.

Adapted from launch_reviewed_flow_v1.py. The source, recipe, mature approval gate,
native recovery and bounded invocation policy are preserved. No automatic retry.
"""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import socket
import subprocess
import sys
import tempfile
import time

ROOT = Path('/mnt/project/st-world-policy-20261016')
REPO = Path('/mnt/project/VLA-Drive-st-rollout-scale-runtime-v2-20261018')
SOURCE = '751fcfa2b41549ae16f7b3204aef8dd18b912016'
RUN_ID = 'C1_rollout_std_fulltrain_seed42_official_4096_v1'
REG_SHA = '928b9788404c4dcb9316b8457dc1a627010f892b85308f52aacf07b2cbb0783f'
HOST = 'training-vlawm-tsinghua-zt5-worker-0'
PYTHON = '/root/miniconda3/envs/ddp/bin/python'
GIB = 2**30
sys.path.insert(0, str(REPO))
from tools.st_world_policy.check_formal_approval import check_formal_approval, verify_registration, file_hash
from tools.st_world_policy.flow_training_config import validate_config
from tools.st_world_policy.training_state import resolve_latest, terminal
from starVLA.dataloader.st_world_targets import atomic_json


def read(path):
    return json.loads(Path(path).read_text())


def artifact(path):
    return {'path': str(Path(path).resolve()), 'sha256': file_hash(path)}


def native_admission(saved, completed, resume):
    if not 0 <= completed <= 4096:
        raise ValueError('Invalid completed optimizer count')
    periodic = [step for step in range(512, 4097, 512) if step > completed]
    initial = int(not resume)
    extra = 2 if completed < 4096 else 0
    maximum = len(saved) + initial + len(periodic) + extra
    if maximum > 15:
        raise ValueError('Remaining periodic/pause/partial writes exceed fifteen native directories')
    return {'existing_directories': saved, 'verified_completed': completed,
            'initial_writes': initial, 'remaining_periodic_steps': periodic,
            'extra_pause_and_partial': extra, 'maximum_directories': maximum, 'reviewed_limit': 15}


def validate_storage(reserves, free):
    if (reserves['run_id'] != RUN_ID or reserves['registration_sha256'] != REG_SHA
            or reserves['own_run_reserve_bytes'] != 209*GIB
            or reserves['shared_free_floor_bytes'] != 1024*GIB
            or reserves['unresolved_old_remaining_bytes'] != 4*GIB):
        raise ValueError('Storage reservation differs from reviewed single-run plan')
    active = reserves['other_active_remaining_writes']
    if any(type(item['bytes']) is not int or item['bytes'] < 0 for item in active):
        raise ValueError('Invalid active remaining-write reservation')
    required = (209+1024+4)*GIB + sum(item['bytes'] for item in active)
    if free < required:
        raise ValueError('Insufficient free bytes for retained floor and remaining writes')
    return required


def fresh_resources(visible, expected_uuids, reserves):
    if not set(range(16)).issubset(os.sched_getaffinity(0)):
        raise RuntimeError('Reviewed CPU 0-15 affinity unavailable')
    rows = [line.split(',') for line in subprocess.check_output(
        ['nvidia-smi', '--id='+visible,
         '--query-gpu=index,uuid,memory.used,utilization.gpu',
         '--format=csv,noheader,nounits'], text=True, timeout=10).splitlines()]
    if [row[1].strip() for row in rows] != expected_uuids:
        raise RuntimeError('Leased GPU UUIDs differ')
    if not all(int(row[2]) < 64 and int(row[3]) < 5 for row in rows):
        raise RuntimeError('Real or unknown GPU jobs are protected')
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], text=True, timeout=10)
    if any(line.split(',')[0].strip() in set(expected_uuids) for line in apps.splitlines()):
        raise RuntimeError('Selected GPU has a compute process, including startup/zero-memory state')
    mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(mem['MemAvailable'].split()[0])*1024
    if available < 512*GIB or not Path('/mnt/navsim/maps').is_dir():
        raise RuntimeError('Reviewed RAM/maps resources unavailable')
    free = shutil.disk_usage(ROOT).free
    required = validate_storage(reserves, free)
    return {'gpu_rows': rows, 'free_bytes': free, 'required_free_bytes': required,
            'mem_available_bytes': available, 'cpu_affinity': '0-15', 'sampled_unix': time.time()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration', required=True)
    p.add_argument('--remaining-write-reserves', required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--acknowledge-stop', action='store_true')
    a = p.parse_args()
    if a.acknowledge_stop and not a.resume:
        p.error('--acknowledge-stop requires --resume')
    regpath = Path(a.registration).resolve()
    assert regpath == ROOT/'formal/consequence/registrations'/RUN_ID/'REGISTRATION.json'
    assert file_hash(regpath) == REG_SHA
    reg = read(regpath)
    assert reg['run_id'] == RUN_ID and reg['source_sha'] == SOURCE
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip() == SOURCE
    cfg = read(reg['artifacts']['config']['path'])
    validate_config(cfg)
    assert cfg['execution']['scope'] == 'FORMAL'
    assert cfg['execution']['max_updates'] == cfg['execution']['stop_after'] == 4096
    assert cfg['execution']['world_size'] == 4 and cfg['execution']['global_batch'] == 32
    assert cfg['execution']['micro_batch'] == 4 and cfg['execution']['sample_limit'] == 0
    assert cfg['runtime']['max_seconds'] == 28800 and cfg['runtime']['save_every'] == 512
    assert cfg['runtime'].get('profile_mode', 'none') == 'none'
    assert cfg['fpo']['advantage'] == 'scene_center_rollout_std_v1'
    fingerprint = verify_registration(regpath, repo=REPO)
    reviews = {}
    for role in ('evidence', 'correctness', 'efficiency'):
        path = regpath.parent/'reviews'/(role.upper()+'.json')
        review = read(path)
        assert review['fingerprint'] == fingerprint and review['status'] == 'PASS'
        reviews[role] = artifact(path)
    monitor_path = ROOT/'resources/monitor_rollout_scale_formal_v1.py'
    observer = ROOT/'resources/observe_formal_flow_cpu_v1.py'
    reserves_path = Path(a.remaining_write_reserves).resolve()
    required_controls = [artifact(path) for path in [__file__, monitor_path, observer, reserves_path]]
    efficiency = read(reviews['efficiency']['path'])
    assert all(value in efficiency['evidence_artifacts'] for value in required_controls), 'Final controls require exact systems review'
    reserves = read(reserves_path)
    approval = regpath.parent/'RUN_APPROVAL.json'
    proposal = {'schema': 'st_world_approval_v1', 'fingerprint': fingerprint,
                'reviews': reviews, 'formal_launch_allowed': True}
    with tempfile.TemporaryDirectory(prefix='st-flow-approval-') as temporary:
        proposed_path = Path(temporary)/'PROPOSED_APPROVAL.json'
        atomic_json(proposed_path, proposal, immutable=True)
        gate = check_formal_approval(regpath, proposed_path, repo=REPO)
    host = socket.gethostname()
    policy_path = Path('/mnt/project/server_dispatch_policy.json')
    auth_path = ROOT/'resources/AUTHORIZATION_20261016T203442.json'
    assert host == HOST and host in read(policy_path)['allowed_aliases']
    assert host.removesuffix('-worker-0') in read(auth_path)['allowed_hosts']
    lease = read(reg['artifacts']['resource_lease']['path'])
    seconds = cfg['runtime']['max_seconds']
    assert host in lease['hostnames'] and lease['valid_from_unix'] <= time.time()
    assert time.time()+seconds+780 < lease['valid_until_unix'], 'Insufficient lease for save/cleanup grace'
    visible = lease['cuda_visible_devices_by_host'][host]
    assert visible == '0,1,2,3'
    out = Path(cfg['runtime']['output_root'])/reg['run_id']
    assert out == ROOT/'formal/consequence/runs'/RUN_ID
    launch_lock = (regpath.parent/'DISPATCH.lock').open('a+')
    fcntl.flock(launch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    locks = [launch_lock]
    resolution = None
    saved = []
    attempts = [entry for entry in (regpath.parent/'invocations').glob('*') if entry.is_dir()]
    assert len(attempts) < 6, 'Six-invocation budget exhausted, including partial dispatch records'
    if a.resume:
        assert out.is_dir() and (regpath.parent/'LAUNCH.json').is_file()
        identity = read(out/'identity.json')
        assert identity['registration_sha256'] == REG_SHA
        assert (out/'STOP_REQUESTED').exists() == a.acknowledge_stop
        resolution = resolve_latest(out/'checkpoints', identity=identity['identity'], world_size=4)
        state = resolution['selected']['receipt']['state']
        completed = state['completed']
        if terminal(state, 4096):
            raise RuntimeError('Run is terminal; use its existing result, do not dispatch a GPU continuation')
        saved = [entry.name for entry in (out/'checkpoints').iterdir() if entry.is_dir()]
    else:
        assert not out.exists() and not (regpath.parent/'LAUNCH.json').exists(), 'Duplicate output or implicit resume refused'
        completed = 0
    storage = native_admission(saved, completed, a.resume)
    for index in visible.split(','):
        lock = open('/tmp/foresight_gpu_'+index+'.lock', 'a+')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
    fresh = fresh_resources(visible, lease['gpu_uuids_by_host'][host], reserves)
    atomic_json(approval, proposal, immutable=True)
    invocation = regpath.parent/'invocations'/str(time.time_ns())
    invocation.mkdir(parents=True, exist_ok=False)
    atomic_json(invocation/'NATIVE_STORAGE_ADMISSION.json', storage, immutable=True)
    if resolution is not None:
        atomic_json(invocation/'RESUME_RESOLUTION.json', resolution, immutable=True)
    cmd = ['timeout', '--signal=TERM', '--kill-after=180s', str(seconds+600)+'s',
           'taskset', '-c', '0-15', PYTHON, '-m', 'torch.distributed.run',
           '--standalone', '--nnodes=1', '--nproc_per_node=4',
           '-m', 'tools.st_world_policy.train_flow_policy', 'train',
           '--registration', str(regpath), '--approval', str(approval)]
    if a.resume:
        cmd.append('--resume')
    if a.acknowledge_stop:
        cmd.append('--acknowledge-stop')
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': visible, 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
           'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
           'NUMEXPR_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false',
           'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
    env.pop('FLASH_ATTENTION_DETERMINISTIC', None)
    spec = importlib.util.spec_from_file_location('formal_owned_monitor', monitor_path)
    monitor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(monitor)
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    with (invocation/'TRAIN.log').open('xb') as stream:
        child = subprocess.Popen(cmd, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
            pass_fds=tuple(lock.fileno() for lock in locks))
    record = {'schema': 'st_reviewed_formal_flow_launch_v1', 'pid': child.pid, 'host': host,
              'started_unix': time.time(), 'argv': cmd, 'cwd': str(REPO), 'scope': 'FORMAL',
              'source_sha': SOURCE, 'registration': str(regpath), 'registration_sha256': REG_SHA,
              'approval_sha256': file_hash(approval), 'dispatch_policy_sha256': file_hash(policy_path),
              'authorization_sha256': file_hash(auth_path), 'gate': gate, 'fresh_resources': fresh,
              'output': str(out), 'resume': a.resume, 'diagnostic_weights_used': False,
              'automatic_followup': False, 'invocation_seconds': seconds,
              'save_cleanup_grace_seconds': 780, 'controls': required_controls,
              'storage_reservations': reserves, 'total_update_ceiling': 4096}
    atomic_json(invocation/'LAUNCH.json', record, immutable=True)
    if not a.resume:
        atomic_json(regpath.parent/'LAUNCH.json', record, immutable=True)
    with (invocation/'OBSERVER.log').open('xb') as stream:
        obs = subprocess.Popen([PYTHON, str(observer), '--invocation', str(invocation)],
            cwd=REPO, env={**env, 'CUDA_VISIBLE_DEVICES': ''}, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    atomic_json(invocation/'OBSERVER_LAUNCH.json', {'pid': obs.pid, **artifact(observer),
        'interval_seconds': 10, 'max_bytes': 256*2**20, 'max_seconds': seconds+780,
        'signals_sent': False}, immutable=True)
    print(json.dumps({'status': 'FORMAL_RESUMED' if a.resume else 'FORMAL_LAUNCHED',
        'pid': child.pid, 'run_id': RUN_ID, 'invocation': str(invocation), 'output': str(out)}), flush=True)
    monitored = monitor.wait_owned(child, invocation, out, lease['gpu_uuids_by_host'][host])
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    atomic_json(invocation/'EXIT.json', {'exit_code': monitored['exit_code'], 'ended_unix': time.time(),
        'waited_child_cpu_seconds': after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
        'child_maxrss_high_water_bytes': after.ru_maxrss*1024,
        'resource_monitor': monitored, 'automatic_followup': False}, immutable=True)
    obs.wait(timeout=15)
    raise SystemExit(monitored['exit_code'] or (2 if monitored['stop_request'] else 0))


if __name__ == '__main__':
    main()
