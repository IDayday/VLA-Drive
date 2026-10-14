"""Bounded ownership guard for the user's temporary, read-only audit workers."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

import psutil

from starVLA.rl.constrained_flow_noise.contracts import file_sha, require_dispatch
from starVLA.rl.flow_grpo.transactions import atomic_json


def authorize(path, *, admission=False):
    auth = json.loads(Path(path).read_text())
    policy = json.loads(Path('/mnt/project/server_dispatch_policy.json').read_text())
    entry = policy['temporary_task_authorizations'][auth['authorization_id']]
    host = socket.gethostname().removesuffix('-worker-0')
    if (entry['authorization_sha256'] != file_sha(path) or
            entry['task'] != 'flow_safe_improvement_20261013' or
            not entry['new_task_dispatch_allowed'] or
            host not in auth['hosts'] or host not in entry['hosts']):
        raise PermissionError('No matching user-authorized temporary lease')
    cutoff = auth['expires_unix'] - (auth['stop_admission_before_expiry_seconds'] if admission else 0)
    if not auth['starts_unix'] <= time.time() < cutoff:
        raise PermissionError('Temporary lease expired or closed to new jobs')
    return auth, host


def worker_authorization(lease):
    if not lease:
        return require_dispatch(), None
    auth, host = authorize(lease)
    receipt = json.loads(Path(os.environ['CFN_TEMP_GUARD_RECEIPT']).read_text())
    parent = psutil.Process(os.getppid())
    if (receipt['guard_pid'] != parent.pid or receipt['child_pid'] != os.getpid() or
            receipt['guard_created'] != parent.create_time() or
            receipt['authorization_sha256'] != file_sha(lease)):
        raise PermissionError('Temporary worker must be owned by its live deadline guard')
    return host, auth['expires_unix']-auth['graceful_stop_before_expiry_seconds']


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--authorization', required=True)
    parser.add_argument('--receipt', required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    auth, host = authorize(args.authorization, admission=True)
    if not args.command or args.command[0] != '--':
        raise ValueError('Explicit child argv is required after --')
    receipt = Path(args.receipt)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    lock = receipt.with_suffix('.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if receipt.exists():
        raise FileExistsError('Each invocation needs a distinct ownership receipt')
    env = os.environ.copy()
    env['CFN_TEMP_GUARD_RECEIPT'] = str(receipt)
    # Worker waits briefly for the atomic receipt before entering model loading.
    child = subprocess.Popen(args.command[1:], env=env, start_new_session=True)
    record = dict(schema='cfn_temporary_guard_v1', host=host, guard_pid=os.getpid(),
                  guard_created=psutil.Process().create_time(), child_pid=child.pid,
                  child_created=psutil.Process(child.pid).create_time(), argv=args.command[1:],
                  authorization_sha256=file_sha(args.authorization), expires_unix=auth['expires_unix'],
                  status='RUNNING', started_unix=time.time(), scientific_outputs_retained=True)
    atomic_json(receipt, record)
    signalled = False
    while child.poll() is None:
        revoked = False
        try:
            authorize(args.authorization)
        except PermissionError:
            revoked = True
        if (revoked or time.time() >= auth['expires_unix']-auth['graceful_stop_before_expiry_seconds']) and not signalled:
            child.send_signal(signal.SIGUSR1)
            signalled = True
            record.update(status='STOP_REQUESTED_AT_NEXT_ATOMIC_SCENE', stop_requested_unix=time.time())
            atomic_json(receipt, record)
        if revoked or time.time() >= auth['expires_unix']-auth['hard_exit_before_expiry_seconds']:
            # Only this newly-created session belongs to this guard. Completed
            # records are atomic; an uncommitted scene is regenerated from its
            # fixed key on resume, never skipped or converted to a zero reward.
            if psutil.Process(child.pid).create_time() != record['child_created']:
                raise RuntimeError('Owned child PID was recycled')
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            record['deadline_fallback_used'] = True
            break
        time.sleep(2)
    # A read-only worker closes its score pool before exiting. Clean only an
    # orphan still in this invocation's unique process session, if necessary.
    leftovers = []
    for process in psutil.process_iter(['pid']):
        try:
            if os.getsid(process.pid) == child.pid:
                leftovers.append(process.pid)
                process.terminate()
        except (ProcessLookupError, psutil.NoSuchProcess, PermissionError):
            pass
    _, alive = psutil.wait_procs([psutil.Process(p) for p in leftovers if psutil.pid_exists(p)], timeout=10)
    for process in alive:
        process.kill()
    record.update(status='EXITED_OWNED_PROCESSES_CLEANED', returncode=child.returncode,
                  orphan_cleanup_pids=leftovers, finished_unix=time.time())
    atomic_json(receipt, record)
    raise SystemExit(child.returncode)


if __name__ == '__main__':
    main()
