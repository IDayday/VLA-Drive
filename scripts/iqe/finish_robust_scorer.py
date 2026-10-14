#!/usr/bin/env python3
"""Finish this registered four-head study after training releases its device.

No training is launched, no process is signalled, and no test split is read.
Waiting is bounded and only permanent authorized hosts are accepted.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from iqe.contracts import require
from iqe.io import atomic_json, file_hash, read_json
from iqe.robust_scorer.config import load, CELLS


def ready(config):
    for cell in CELLS:
        folder = config.root/'scorers'/cell
        if not (folder/'result.json').exists():
            return False
        result = read_json(folder/'result.json')
        if (result.get('status') != 'COMPLETE' or result.get('optimizer_steps') != config.updates
                or not result.get('termination', {}).get('terminal')):
            return False
        try:
            with open(folder/'RUN.lock', 'a') as guard:
                fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
    return True


def available_gpu(gpu):
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True)
    inventory = {}
    for line in rows.splitlines():
        index, uuid, free, utilization = map(str.strip, line.split(','))
        inventory[int(index)] = dict(uuid=uuid, free_MiB=int(free), utilization=int(utilization))
    require(gpu in inventory, 'requested GPU does not exist')
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True)
    row = inventory[gpu]
    row['unoccupied'] = row['uuid'] not in {line.strip() for line in apps.splitlines()}
    return row['unoccupied'] and row['free_MiB'] >= 32768 and row['utilization'] < 20, row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--expect-host', required=True)
    p.add_argument('--max-wait-seconds', type=int, default=7200)
    a = p.parse_args()
    require(0 < a.max_wait_seconds <= 10800, 'bounded three-hour maximum wait')
    config = load(a.config)
    host = socket.gethostname().removesuffix('-worker-0')
    require(host == a.expect_host and host != 'training-vla-zt2', 'wrong or protected host')
    policy_path = Path('/mnt/project/server_dispatch_policy.json')
    def allowed():
        require(host in read_json(policy_path)['allowed_hosts'], 'permanent host authorization withdrawn')
    allowed()
    folder = config.root/'runtime/finalizer'
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder/'RUN.lock', 'a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        launch = dict(host=host, pid=os.getpid(),
            start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
            launched_utc=datetime.now(timezone.utc).isoformat(), config_hash=config.identity,
            code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            gpu=a.gpu, max_wait_seconds=a.max_wait_seconds, no_other_process_signals=True,
            source_hashes={name: file_hash(ROOT/'iqe/robust_scorer'/name)
                           for name in ('evaluation.py', 'bundle.py', 'qualification.py', 'report.py')})
        atomic_json(folder/'LAUNCH.json', launch)
        expires = time.monotonic()+a.max_wait_seconds
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                   CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONUNBUFFERED='1')
        commands = [
            ('evaluation', ['-m', 'iqe.robust_scorer.evaluation', 'evaluate']),
            ('bundle', ['-m', 'iqe.robust_scorer.bundle']),
            ('online', ['-m', 'iqe.robust_scorer.qualification', 'online']),
            ('roundtrip', ['-m', 'iqe.robust_scorer.qualification', 'roundtrip']),
            ('resume', ['-m', 'iqe.robust_scorer.qualification', 'resume-audit']),
            ('report', ['-m', 'iqe.robust_scorer.report']),
        ]
        for stage, command in commands:
            while True:
                allowed()
                complete = ready(config)
                available, gpu = available_gpu(a.gpu)
                if complete and available:
                    break
                atomic_json(folder/'STATUS.json', dict(status='WAITING_INPUTS' if not complete else 'WAITING_RESOURCE',
                    stage=stage, gpu=gpu, checked_utc=datetime.now(timezone.utc).isoformat()))
                require(time.monotonic() < expires, 'bounded finalization wait expired; rerun after prerequisites finish')
                time.sleep(20)
            require(all(file_hash(ROOT/'iqe/robust_scorer'/name) == checksum for name, checksum in launch['source_hashes'].items()),
                    'finalization implementation changed after registration')
            cmd = [sys.executable, *command, '--config', str(Path(a.config).resolve())]
            if stage not in ('bundle', 'report'):
                cmd += ['--device', 'cuda:0']
            atomic_json(folder/'STATUS.json', dict(status='RUNNING', stage=stage, command=cmd,
                        gpu=gpu, started_utc=datetime.now(timezone.utc).isoformat()))
            with open(folder/(stage+'.log'), 'a') as log:
                result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
            receipt = dict(stage=stage, returncode=result.returncode, command=cmd, log=str(folder/(stage+'.log')),
                           finished_utc=datetime.now(timezone.utc).isoformat())
            atomic_json(folder/(stage+'.json'), receipt)
            require(result.returncode == 0, 'finalization failed; inspect '+str(folder/(stage+'.log')))
        atomic_json(folder/'STATUS.json', dict(status='COMPLETE', finished_utc=datetime.now(timezone.utc).isoformat(),
                    new_training_started=False, new_navtest=False, ACTIVE_updated=False))


if __name__ == '__main__':
    main()
