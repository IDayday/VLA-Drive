"""Resume the two accepted spatial runs, retaining native registration and state.

The extra receipt admits one execution window. It never replaces a scientific
approval, changes the original lease, or creates another training identity.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

ROOT = Path('/mnt/project/st-world-policy-20261016')
REPO = Path('/mnt/project/VLA-Drive-st-official-runtime-v2-20261016')
SOURCE = 'fdf25f99b702754787904215bf1c2a2b7515ea75'
RUNS = ('SP2_fulltrain_seed42_official_20k_v1', 'SP3_fulltrain_seed42_official_20k_v1')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pinned(ref):
    assert set(ref) == {'path', 'sha256'}
    p = Path(ref['path'])
    assert p.is_absolute() and p.is_file() and p.stat().st_size <= 8 * 2**20
    assert sha(p) == ref['sha256'], 'Changed control artifact'
    return json.loads(p.read_text())


def validate(admission, regpath, reg, cfg, lease, host, now):
    assert admission['schema'] == 'st_spatial20k_resume_invocation_v1'
    assert admission['status'] == 'APPROVED'
    assert admission['run_id'] == reg['run_id'] in RUNS
    assert admission['registration'] == {'path': str(regpath), 'sha256': sha(regpath)}
    assert admission['source_sha'] == reg['source_sha'] == SOURCE
    assert admission['helper'] == {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)}
    assert admission['no_automatic_resume'] is True
    assert admission['max_invocation_seconds'] == cfg['runtime']['max_seconds'] == 28800
    assert admission['valid_from_unix'] <= now < admission['valid_until_unix'] - 29580
    assert lease['valid_from_unix'] <= now < lease['valid_until_unix'] - 29580
    assert host in admission['hostnames'] and host in lease['hostnames']
    for key in ('cuda_visible_devices_by_host', 'gpu_uuids_by_host'):
        assert admission[key][host] == lease[key][host], 'Do not migrate topology here'
    assert cfg['execution']['scope'] == 'FORMAL'
    assert cfg['execution']['max_updates'] == cfg['execution']['stop_after'] == 20000
    assert cfg['execution']['world_size'] == 4 and cfg['execution']['micro_batch'] == 4
    assert cfg['execution']['global_batch'] == 32
    assert 5000 < admission['expected_completed'] < 20000
    assert admission['remaining_cache_growth_bytes'] == 768 * 2**30
    assert admission['separate_conservative_metadata_buffer_bytes'] == 10 * 2**30


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration', required=True, type=Path)
    p.add_argument('--admission', required=True, type=Path)
    p.add_argument('--admission-sha256', required=True)
    a = p.parse_args()
    regpath = a.registration.resolve()
    assert regpath.is_relative_to(ROOT / 'formal/spatial/registrations')
    admission_ref = {'path': str(a.admission.resolve()), 'sha256': a.admission_sha256}
    admission = pinned(admission_ref)
    reg = json.loads(regpath.read_text())
    cfg = pinned(reg['artifacts']['config'])
    lease = pinned(reg['artifacts']['resource_lease'])
    host = socket.gethostname()
    validate(admission, regpath, reg, cfg, lease, host, time.time())
    sys.path.insert(0, str(REPO))
    from tools.st_world_policy.check_formal_approval import check_formal_approval
    from tools.st_world_policy.training_state import resolve_latest
    from starVLA.dataloader.st_world_targets import atomic_json
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip() == SOURCE
    assert not subprocess.check_output(['git', 'status', '--porcelain=v1'], cwd=REPO).strip()
    approval = regpath.parent / 'RUN_APPROVAL.json'
    gate = check_formal_approval(regpath, approval, repo=REPO)
    policy_path = Path('/mnt/project/server_dispatch_policy.json')
    policy = json.loads(policy_path.read_text())
    auth_path = ROOT / 'resources/AUTHORIZATION_20261016T203442.json'
    auth = json.loads(auth_path.read_text())
    canonical = host.removesuffix('-worker-0')
    assert host in policy['allowed_aliases'] and canonical in auth['allowed_hosts']
    for name, banned in policy['global_dispatch_prohibitions'].items():
        assert host not in [name, *banned.get('aliases', [])] and canonical != name
    out = Path(cfg['runtime']['output_root']) / reg['run_id']
    assert out == ROOT / 'formal/spatial/runs' / reg['run_id'] and out.is_dir()
    assert not (out / 'STOP_REQUESTED').exists(), 'An operator stop needs separate explicit handling'
    identity = json.loads((out / 'identity.json').read_text())
    assert identity['registration_sha256'] == sha(regpath)
    visible = lease['cuda_visible_devices_by_host'][host]
    ids = [int(x) for x in visible.split(',')]
    assert len(ids) == len(set(ids)) == 4
    locks = []
    for path in [regpath.parent / 'RESUME_DISPATCH.lock', *[Path(f'/tmp/foresight_gpu_{i}.lock') for i in ids]]:
        f = path.open('a+')
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(f)
    # Probe, then release the native run lock; train_world owns it itself.
    with (out / 'RUN.lock').open('a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rows = [x.split(',') for x in subprocess.check_output(['nvidia-smi', '--id=' + visible,
        '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).splitlines()]
    assert [r[1].strip() for r in rows] == lease['gpu_uuids_by_host'][host]
    assert all(int(r[2]) < 64 and int(r[3]) < 5 for r in rows), 'Protect other GPU processes'
    cp = pinned(admission['checkpoint_commitments'])
    assert cp['schema'] == 'st_checkpoint_storage_commitments_v1' and cp['status'] == 'CURRENT'
    assert cp['checked_at_unix'] <= time.time() < cp['valid_until_unix'] - 29580
    assert cp['remaining_bytes'] >= 0 and cp['credited_already_on_disk_bytes'] >= 0
    assert cp['remaining_bytes'] + cp['credited_already_on_disk_bytes'] == cp['total_retained_allocation_bytes'] == 4143 * 2**30
    assert cp['free_floor_bytes'] == 1024 * 2**30
    needed = cp['remaining_bytes'] + cp['free_floor_bytes'] + 778 * 2**30
    assert shutil.disk_usage(ROOT).free >= needed, 'Preserve cache, checkpoint and separate free-space reserve'
    expected = pinned(admission['expected_checkpoint'])
    assert expected['state']['identity'] == identity['identity']
    assert expected['state']['completed'] == admission['expected_completed']
    assert expected['state']['termination_reason'] == 'BUDGET_PAUSED'
    assert Path(admission['expected_checkpoint']['path']) == out / 'checkpoints' / expected['state']['tag'] / 'COMPLETE.json'
    # Native resolver verifies all payload checksums; never trusts stale RESULT.
    resolution = resolve_latest(out / 'checkpoints', identity=identity['identity'], world_size=4)
    assert resolution['selected']['receipt'] == expected, 'Latest verified progress changed; obtain a fresh binding'
    validate(admission, regpath, reg, cfg, lease, host, time.time())
    invocation = regpath.parent / 'resume_invocations' / admission_ref['sha256']
    invocation.mkdir(parents=True, exist_ok=False)  # one attempt per admission, including failed Popen
    atomic_json(invocation / 'ADMISSION.json', admission, immutable=True)
    atomic_json(invocation / 'RESUME_RESOLUTION.json', resolution, immutable=True)
    cmd = ['timeout', '--signal=TERM', '--kill-after=180s', '29400s',
        '/root/miniconda3/envs/ddp/bin/python', '-m', 'torch.distributed.run', '--standalone',
        '--nnodes=1', '--nproc_per_node=4', '-m', 'tools.st_world_policy.train_world',
        '--registration', str(regpath), '--approval', str(approval), '--resume']
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': visible, 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
        'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
        'TOKENIZERS_PARALLELISM': 'false', 'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
    env.pop('FLASH_ATTENTION_DETERMINISTIC', None)
    with (invocation / 'TRAIN.log').open('xb') as log:
        child = subprocess.Popen(cmd, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            pass_fds=tuple(f.fileno() for f in locks))
    record = dict(schema='st_spatial20k_resume_launch_v1', pid=child.pid, host=host,
        started_unix=time.time(), argv=cmd, cwd=str(REPO), admission=admission_ref,
        gate=gate, registration_sha256=sha(regpath), output=str(out),
        expected_completed=admission['expected_completed'], automatic_followup=False,
        checkpoint_and_optimizer_unchanged=True, policy_sha256=sha(policy_path))
    atomic_json(invocation / 'LAUNCH.json', record, immutable=True)
    print(json.dumps(record))


if __name__ == '__main__':
    main()
