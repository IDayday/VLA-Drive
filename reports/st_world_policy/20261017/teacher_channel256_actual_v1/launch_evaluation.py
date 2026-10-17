"""Lead-only bounded fixed PCA256 endpoint evaluation; no model code changes."""
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

HERE = Path(__file__).resolve().parent
ROOT = Path('/mnt/project/st-world-policy-20261016')
RUNTIME = Path('/mnt/project/VLA-Drive-st-flow-context-runtime-v3-20261017')
SOURCE = '6714b1e9fb2e0c8059b90b8b4d48cd40796fe345'
SPEC_SHA = '59cdf8a3ab692994f926ce460a4a353bb16ffba1a3bb21d1ff6c4beacb936452'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pinned(pin):
    p = Path(pin['path'])
    if not p.is_absolute() or p.stat().st_size > 4 << 20:
        raise ValueError('Small absolute control required')
    b = p.read_bytes()
    if hashlib.sha256(b).hexdigest() != pin['sha256']:
        raise ValueError('Control changed: ' + str(p))
    return json.loads(b)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('export', 'score'), required=True)
    p.add_argument('--review', required=True)
    p.add_argument('--review-sha256', required=True)
    a = p.parse_args()
    spec = pinned({'path': str(HERE/'EVALUATION_SPEC.json'), 'sha256': SPEC_SHA})
    review = pinned({'path': a.review, 'sha256': a.review_sha256})
    if (review['status'] != 'PASS' or review['spec_sha256'] != SPEC_SHA or
            review['launcher_sha256'] != digest(__file__) or review['reviewer'] == '/root'):
        raise ValueError('Actual independent fixed-evaluation inheritance review required')
    if socket.gethostname() != spec['hostname']:
        raise ValueError('Wrong host')
    policy = json.loads(Path('/mnt/project/server_dispatch_policy.json').read_text())
    if spec['hostname'] not in policy['allowed_hosts']['training-vla-zt']['aliases']:
        raise ValueError('Host no longer allowed')
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=RUNTIME, text=True).strip() != SOURCE:
        raise ValueError('Frozen evaluation runtime changed')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=RUNTIME).strip():
        raise ValueError('Dirty evaluation runtime')
    registration = pinned(spec['training_registration'])
    approval = pinned(spec['training_approval'])
    if not approval['formal_launch_allowed'] or approval['fingerprint']['registration_sha256'] != spec['training_registration']['sha256']:
        raise ValueError('Training identity lacks original approval')
    receipt = pinned(spec['checkpoint_complete'])
    if receipt['state']['completed'] != 5000 or receipt['state']['identity'] != spec['runtime_identity']:
        raise ValueError('Wrong completed training endpoint')
    folder = Path(spec['checkpoint_complete']['path']).parent
    for name, item in receipt['files'].items():
        if Path(name).name != name or (folder/name).stat().st_size != item['bytes']:
            raise ValueError('Incomplete native checkpoint inventory')
    # The unchanged native exporter revalidates payload checksums and strict load.
    out = Path(spec['output_root']); reg = Path(spec['registration_root'])
    reg.mkdir(parents=True, exist_ok=True)
    locks = []
    def lock(path):
        f = path.open('a+')
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            f.close(); raise
        locks.append(f)
    lock(reg/(a.stage.upper()+'_DISPATCH.lock'))
    for i in range(4 if a.stage == 'export' else 2):
        try:
            lock(ROOT/f'resources/dev_{a.stage}_slot_{i}.lock'); break
        except BlockingIOError:
            continue
    else:
        raise RuntimeError('All existing campaign evaluation slots occupied')
    if shutil.disk_usage(ROOT).free < (2703+1024+768+1)*2**30:
        raise RuntimeError('Protect checkpoint, cache and shared free-floor commitments')
    invocation_root = reg/(a.stage+'_invocations')
    previous = list(invocation_root.glob('*'))
    if len(previous) >= 2:
        raise RuntimeError('Two-invocation bound exhausted')
    if a.stage == 'export':
        if (out/'export/COMPLETE_rank0.json').exists():
            raise ValueError('Already exported; do not recompute')
        lock(Path(f'/tmp/foresight_gpu_{spec["gpu"]}.lock'))
        row = subprocess.check_output(['nvidia-smi', '--id='+str(spec['gpu']),
            '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(',')
        if row[0].strip() != spec['gpu_uuid'] or int(row[1]) >= 64 or int(row[2]) >= 5:
            raise RuntimeError('Protect occupied or changed GPU')
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
        if spec['gpu_uuid'] in processes:
            raise RuntimeError('Existing GPU process')
        mem = dict(x.split(':', 1) for x in Path('/proc/meminfo').read_text().splitlines())
        if int(mem['MemAvailable'].split()[0])*1024 < 64*2**30:
            raise RuntimeError('Insufficient memory for strict native reconstruction')
    else:
        if not (out/'export/COMPLETE_rank0.json').is_file():
            raise ValueError('Complete fixed-population export required')
        if (out/'official_score/COMPLETE.json').exists():
            raise ValueError('Already scored')
    cmdspec = spec['command'][a.stage]
    argv = cmdspec['argv_template' if a.stage == 'export' else 'argv']
    cmd = ['timeout', '--signal=TERM', '--kill-after=60s',
           str(cmdspec['outer_timeout_seconds'])+'s', *argv]
    env = {**os.environ, **cmdspec['env'], 'PYTHONUNBUFFERED': '1'}
    env.pop('FLASH_ATTENTION_DETERMINISTIC', None)
    invocation = invocation_root/str(time.time_ns()); invocation.mkdir(parents=True, exist_ok=False)
    admission = {'spec_sha256': SPEC_SHA, 'review': {'path': a.review, 'sha256': a.review_sha256},
        'stage': a.stage, 'argv': cmd, 'cwd': str(RUNTIME), 'started_unix': time.time(),
        'maximum_invocations': 2, 'ordinal': len(previous)+1, 'inference_runtime': SOURCE,
        'state': 'ADMITTED_BEFORE_POPEN', 'navtest': False, 'ACTIVE_update': False}
    (invocation/'ADMISSION.json').write_text(json.dumps(admission, indent=2)+'\n')
    with (invocation/'RUN.log').open('xb') as log:
        child = subprocess.Popen(cmd, cwd=RUNTIME, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            pass_fds=tuple(f.fileno() for f in locks))
    launch = {**admission, 'pid': child.pid, 'state': 'STARTED', 'host': socket.gethostname()}
    (invocation/'LAUNCH.json').write_text(json.dumps(launch, indent=2)+'\n')
    print(json.dumps({'stage': a.stage, 'pid': child.pid, 'invocation': str(invocation)}))


if __name__ == '__main__':
    main()
