"""Wait for an already-running local replica job, then start a frozen slot.

This never restarts staging, allocates GPUs itself, or changes training source.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from tools.ddpolicy_vehicle.prepare_data import atomic_json


def main():
    p = argparse.ArgumentParser(__doc__)
    for key in ('replica', 'identity', 'candidate', 'launch-record', 'worktree', 'output'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--stager-pid', type=int, required=True)
    p.add_argument('--images', type=int, required=True)
    p.add_argument('--max-seconds', type=int, default=21600)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise FileExistsError('Use a unique staging handoff record')
    launch = json.loads(Path(a.launch_record).read_text())
    state = dict(status='WAITING_FOR_COMPLETE_LOCAL_REPLICA', arguments=vars(a),
                 started_unix=time.time(), command=launch['command'])
    atomic_json(out, state)
    try:
        while True:
            if Path(a.replica).exists():
                replica = json.loads(Path(a.replica).read_text())
                if replica['identity'] != a.identity:
                    raise ValueError('Foreign candidate replica')
                if replica['images'] == a.images and replica['verified_sha256']:
                    break
            proc = Path(f'/proc/{a.stager_pid}/cmdline')
            if not proc.exists():
                raise RuntimeError('Existing staging process exited before completion')
            args = proc.read_bytes().decode().split('\0')
            if 'tools.dino_tradeoff.stage_targets' not in args or '--candidate' not in args or args[args.index('--candidate') + 1] != a.candidate:
                raise RuntimeError('Existing staging process identity changed')
            if time.time() - state['started_unix'] > a.max_seconds:
                state.update(status='PAUSED', reason='Bounded local staging wait expired')
                atomic_json(out, state)
                return
            time.sleep(5)
        state.update(status='RUNNING_SLOT', staging_complete_unix=time.time())
        atomic_json(out, state)
        code = subprocess.call(launch['command'], cwd=a.worktree,
                               env=dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1'))
        state.update(status='SLOT_EXITED' if code == 0 else 'FAILED', exit_code=code, ended_unix=time.time())
        atomic_json(out, state)
        if code:
            raise SystemExit(code)
    except Exception as error:
        state.update(status='FAILED', error=repr(error), ended_unix=time.time())
        atomic_json(out, state)
        raise


if __name__ == '__main__':
    main()
