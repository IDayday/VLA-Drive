"""One reviewed, same-run TIME_RESIDUAL move to the restored vla-zt3 host.

Uses mature campaign dispatch and the trainer's native boundary-stop file.
No learner is killed and no saved scientific state is rewritten.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path('/mnt/project/qwen-world-postcot-20261025/fresh_loss_IL_20261029/TIME_RESIDUAL')
SCIENCE = Path('/mnt/project/VLA-Drive-qwen-world-loss-from-init-20261029')
OPERATOR = Path('/mnt/project/VLA-Drive-qwen-world-time-owner-adoption-20261029')
RECORD = Path('/mnt/project/qwen-world-il-research-20261030/maintenance/vla3_rehost_20261031')
DEST = 'training-vla-zt3'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path); tmp = path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    with tmp.open('w') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)


def journals(run, n=32):
    result = []
    for rank in range(8):
        path = run/f'metrics_rank{rank}.jsonl'
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size-512*1024))
            lines = stream.read().splitlines()
        result.append([json.loads(line) for line in lines[-n:]])
    return result


def main():
    if (RECORD/'TIME_OWNER_LAUNCH.json').exists():
        raise RuntimeError('Migration already dispatched; inspect the existing receipt')
    lock = (ROOT/'owner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ.update(WORLD_POSTCOT_ROOT=str(ROOT), WORLD_POSTCOT_TASK='qwen_world_fresh_loss_IL_20261029')
    sys.path.insert(0, str(OPERATOR))
    from tools.qwen_world_postcot import campaign as c
    from starVLA.rl.qwen_world_postcot.contracts import file_sha, fingerprint, allowed_hosts
    from starVLA.rl.qwen_world_postcot.resources import probe, gpu_activity_issue
    if DEST not in allowed_hosts(admit_long_job=True):
        raise PermissionError('Destination permission no longer valid')
    env = read(RECORD/'environment/COMPLETE.json')
    if env['status'] != 'PASS_VERSION_IMPORTS':
        raise RuntimeError('Restored host environment not qualified')
    samples = [probe(DEST), probe(DEST)]
    if ([g['uuid'] for g in samples[0]['gpus']] != [g['uuid'] for g in samples[1]['gpus']]
            or any(len(s['gpus']) != 8 or s['owners'] or not s['worker_python_exists']
                or min(g['free_MiB'] for g in s['gpus']) < 49152
                or any(gpu_activity_issue(g, s['owners']) for g in s['gpus']) for s in samples)):
        raise RuntimeError('Destination no longer a free qualified eight-GPU host; old learner untouched')
    write(RECORD/'PREDRAIN_CAPACITY.json', samples)
    state = read(ROOT/'STATE.json'); failure = read(ROOT/'FAILURE.json')
    if state['source'] != str(SCIENCE) or state['stage'] != 'FULL':
        raise ValueError('Changed source/stage')
    # CPU controller died before the SSH shell could start Python. The old
    # trainer remains active and is never re-created during this reconciliation.
    if 'returned non-zero exit status 127' not in failure['error'] or DEST not in failure['error']:
        raise RuntimeError('Unreviewed controller failure')
    for name in ('STATE.json', 'OWNER.json', 'FAILURE.json', 'LAUNCH.json', 'EXECUTION_ROUTING.json'):
        write(RECORD/('BEFORE_'+name), read(ROOT/name))
    identity = read(ROOT/'runs/IL_full_seed42/IDENTITY.json')
    for rel, digest in identity['source_files'].items():
        if file_sha(SCIENCE/rel) != digest:
            raise ValueError('Changed immutable scientific source: '+rel)
    c.poll_jobs(state)  # Reconcile durable intent from the failed remote dispatch.
    for name, job in state['jobs'].items():
        if job['status'] != 'FAILED':
            continue
        receipt = Path(job['receipt'])
        if (name != 'refresh_020000_rank006_attempt002' or job['host'] != DEST
                or any((receipt/p).exists() for p in ('RUNNING.json', 'RESULT.json', 'DISPATCH.json'))):
            raise RuntimeError('Other worker failure requires review: '+name)
        job.update(status='RESOLVED_PRECHILD_TRANSPORT_FAILURE', recovery_review=str(RECORD/'BEFORE_FAILURE.json'),
                   child_not_started=True, reason='SSH shell127: missing Python, before Popen; archive restored and CPU-qualified')
    old = [j for j in state['jobs'].values() if j['role'] == 'training' and j['status'] == 'RUNNING']
    if len(old) != 1:
        raise RuntimeError('Need exactly one existing learner')
    old = old[0]; old_result = Path(old['receipt'])/'RESULT.json'
    native = ROOT/'runs/IL_full_seed42'
    before = journals(native)
    write(RECORD/'BEFORE_TRAINING_WINDOW.json', dict(rank_tails=before,
        note='Shared-host observations; wall/update is the maximum across ranks, not rank sum'))
    routing = dict(schema='world_IL_execution_routing_v1', arm='TIME_RESIDUAL', world=8,
        training_hosts=[DEST], request=str(RECORD/'PREDRAIN_CAPACITY.json'),
        scientific_source=str(SCIENCE), previous_routing=str(RECORD/'BEFORE_EXECUTION_ROUTING.json'))
    write(ROOT/'EXECUTION_ROUTING.json', routing)
    write(ROOT/'STATE.json', state)
    write(RECORD/'STOP_REQUEST.json', dict(at=time.time(), receipt=old['receipt'], old_host=old['host'],
        action='cooperative native optimizer-boundary save and exit; no signal/kill',
        max_updates_unchanged=100000, seed=42, global_batch=32))
    (Path(old['receipt'])/'STOP_REQUESTED').touch()
    deadline = time.monotonic()+900
    while not old_result.exists():
        if time.monotonic() > deadline:
            raise TimeoutError('No native boundary result; do not force kill')
        time.sleep(5)
    result = read(old_result)
    if (result.get('status') != 'EXITED' or result.get('returncode') != 0
            or not c.owned_exit_verified(result)):
        raise RuntimeError('Old transport did not exit cleanly; preserve last native state')
    progress = read(native/'progress.json')
    checkpoint = Path(progress['checkpoint'])
    header = c.checkpoint_header(checkpoint, 'IL')
    if (progress['status'] != 'STOPPED' or header['identity'] != identity
            or header['updates'] != progress['updates']):
        raise ValueError('Wrong native stopped boundary')
    import torch
    clients = [torch.load(checkpoint/f'rank{rank:02d}_state.pt', map_location='cpu', weights_only=False) for rank in range(8)]
    after = journals(native, n=1)
    if (any(s['state'] != clients[0]['state'] or s['identity_sha256'] != fingerprint(identity) for s in clients)
            or any(r[-1]['updates'] != progress['updates'] or r[-1]['optimizer_global_steps'] != progress['updates'] for r in after)):
        raise ValueError('Rank state / effective update boundary differs')
    write(RECORD/'NATIVE_STOP_COMMITTED.json', dict(at=time.time(), checkpoint=str(checkpoint),
        updates=progress['updates'], cursor=progress['cursor'], files_sha256=header['files_sha256'],
        identity_sha256=fingerprint(identity), all_rank_states_match=True, old_exact_transport_exited=True,
        no_uncommitted_updates_discarded=True, original_result=result))
    c.poll_jobs(state)
    old_command = list(old['command']); old_command[old_command.index('--resume')+1] = str(checkpoint)
    route = c.choose_training_host(49152, eligible_hosts=[DEST])
    if route is None:
        # Preserve a useful fallback rather than stranding a sealed learner.
        routing['training_hosts'] = [DEST, old['host']]
        write(ROOT/'EXECUTION_ROUTING.json', routing)
    else:
        route['lease_slot'] = 'primary'
        invocation = c.dispatch(state, str(SCIENCE), 'full_IL', old_command, route, role='training')
        write(RECORD/'NEW_TRAINING_INVOCATION.json', dict(invocation=invocation, host=DEST,
            receipt=state['jobs'][invocation]['receipt'], source=str(SCIENCE), resume=str(checkpoint)))
    state['at'] = time.time(); write(ROOT/'STATE.json', state)
    # Single CPU owner takes over existing refreshes and the new native learner.
    lock.close()
    argv = [c.MODEL_PYTHON, '-u', '-m', 'tools.qwen_world_postcot.fresh_loss_owner',
            '--arm', 'TIME_RESIDUAL', '--runtime-source', str(SCIENCE), '--recover-owner']
    with (RECORD/'restored_owner.log').open('ab') as log:
        process = subprocess.Popen(argv, cwd=OPERATOR, env=os.environ.copy(), stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    launch = dict(pid=process.pid, start_ticks=int(Path(f'/proc/{process.pid}/stat').read_text().split()[21]),
        at=time.time(), source=str(OPERATOR), runtime_source=str(SCIENCE), command=argv,
        phase='AWAIT_REAL_NATIVE_UPDATES', original_checkpoint=str(checkpoint))
    write(RECORD/'TIME_OWNER_LAUNCH.json', launch)
    write(ROOT/'LAUNCH.json', launch)
    print(json.dumps(launch), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__); parser.parse_args()
    try:
        main()
    except BaseException:
        write(RECORD/'MIGRATION_FAILURE.json', dict(at=time.time(), error=traceback.format_exc(),
            automatic_retry=False, scientific_source_modified=False))
        raise
