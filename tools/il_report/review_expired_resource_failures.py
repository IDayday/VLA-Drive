"""Explicit review of seven authorization-boundary failures; no trainer changes."""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

BASE = Path('/mnt/project/qwen-world-safety-exploration-20261031')
DEADLINE = Path('/mnt/project/qwen-world-four-frameworks-20261029/resources/temporary_deadline_20261101T230000Z')
GROUPS = [
    ('standard_training_20261101', 'STANDARD_NATIVE', ['STANDARD_INVERSION_G8', 'STANDARD_COMBINED_G8', 'SAFETY_G8_CONTROL']),
    ('revival_training_20261101', 'REVIVAL_NATIVE', ['REVIVAL_G16', 'REVIVAL_SATURATED_G8']),
    ('improvement_training_20261101', 'IMPROVEMENT_NATIVE', ['IMPROVEMENT_EARLY_DECAY_G8', 'IMPROVEMENT_FP32_OUTPUT_G8']),
]
OUT = BASE / 'operations/EXPIRED_RESOURCE_REVIEW_20261101'


def atomic(path, value):
    temp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temp, path)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_failure(study, state, arm):
    root = BASE / study
    run = root / 'runs' / arm
    row = state['runs'][arm]
    stages = [(key, node) for key, node in row['stages'].items() if not node.get('complete')]
    if row['status'] != 'FAILED_REVIEW_REQUIRED' or len(stages) != 1:
        raise ValueError('Not one explicitly failed stage: ' + arm)
    target, node = stages[0]
    job = Path(node['job'])
    receipt = read(job / 'RESULT.json')
    plan = read(DEADLINE / 'PLAN.json')
    if node['host'] not in plan['temporary_hosts'] or receipt['returncode'] != 1:
        raise ValueError('Not the reviewed authorization failure: ' + arm)
    if not plan['guard_unix'] <= receipt['end_unix'] <= plan['guard_unix'] + 60:
        raise ValueError('Failure outside exact guard interval: ' + arm)
    if not (job / 'STOP_REQUESTED').exists():
        raise ValueError('Missing cooperative administrative stop intent')
    with (job / 'stdout.log').open('rb') as stream:
        stream.seek(max(0, stream.seek(0, 2) - 512 * 1024))
        text = stream.read().decode(errors='strict')
    errors = set(re.findall(r'^\[rank\d+\]: (\w+Error): (.+)$', text, re.M))
    expected = {('PermissionError', 'Host not authorized: ' + node['host'])}
    if errors != expected or 'group_runtime.py' not in text or 'in require_dispatch' not in text:
        raise ValueError('Unreviewed additional failure: ' + arm + ' ' + repr(errors))
    latest = read(run / 'latest.json')
    cp = Path(latest['checkpoint'])
    if not cp.is_relative_to(run / 'checkpoints'):
        raise ValueError('Foreign checkpoint')
    seal = read(cp / 'COMPLETE')
    if (seal['world'] != 8 or seal['updates'] != latest['updates'] or
            seal['phase'] not in ('READY', 'ROLLOUT_COMMITTED', 'INNER_COMMITTED') or
            seal['next_inner'] not in range(4) or latest['updates'] >= int(target) or
            sha(cp / 'files.json') != seal['files_sha256']):
        raise ValueError('Invalid saved transaction')
    files = read(cp / 'files.json')
    required = {'optimizer.pt', 'transaction.json', 'identity.json', *[f'rank{i}.pt' for i in range(8)]}
    if not required <= set(files):
        raise ValueError('Missing full optimizer/rank inventory')
    for name in files:
        p = cp / name
        if not p.resolve().is_relative_to(cp.resolve()) or not p.is_file() or p.stat().st_size == 0:
            raise ValueError('Incomplete native member')
    last = []
    for rank in range(8):
        p = run / f'metrics_rank{rank}.jsonl'
        with p.open('rb') as stream:
            stream.seek(max(0, stream.seek(0, 2) - 32768))
            lines = stream.read().splitlines()
        valid = []
        for line in lines[-4:]:
            try:
                valid.append(json.loads(line))
            except ValueError:
                pass
        last.append(valid[-1]['updates'])
    return dict(arm=arm, target=int(target), job=str(job), host=node['host'],
                actual_returncode=receipt['returncode'], failure='HOST_PERMISSION_REVOKED_BEFORE_COOPERATIVE_SAVE',
                checkpoint=str(cp), checkpoint_updates=latest['updates'], phase=seal['phase'],
                next_inner=seal['next_inner'], files_sha256=seal['files_sha256'],
                old_rank_updates=last, uncommitted_updates_to_replay=max(last)-latest['updates'],
                inventory_seal_verified=True, full_member_hashes_reverified_by_native_loader_on_resume=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if (OUT / 'APPLIED.json').exists():
        raise FileExistsError('Already applied; inspect current owners instead of duplicating')
    audit = read(DEADLINE / 'POST_DEADLINE_BROAD_AUDIT.json')
    if any(x.get('returncode') != 0 or x['owned_processes'] or x['visible_GPU_processes']
           for x in audit['results'].values()):
        raise ValueError('Temporary-host physical exit not verified')
    groups = []
    for study, owner_name, arms in GROUPS:
        directory = BASE / study / 'owners' / owner_name
        state = read(directory / 'STATE.json')
        groups.append((study, owner_name, [verified_failure(study, state, arm) for arm in arms]))
    atomic(OUT / 'QUALIFIED.json', dict(groups=groups, science_unchanged=True, temporary_nodes_not_reused=True))
    print('QUALIFIED', sum(len(rows) for _, _, rows in groups), flush=True)
    if not args.apply:
        return
    launched = []
    for study, owner_name, rows in groups:
        directory = BASE / study / 'owners' / owner_name
        old = read(directory / 'OWNER.json')
        proc = Path('/proc') / str(old['pid'])
        stat = (proc / 'stat').read_text().split()
        argv = [x for x in (proc / 'cmdline').read_bytes().decode().split(chr(0)) if x]
        module = {'STANDARD_NATIVE': 'standard_campaign', 'REVIVAL_NATIVE': 'revival_campaign',
                  'IMPROVEMENT_NATIVE': 'improvement_campaign'}[owner_name]
        if int(stat[21]) != old['start_ticks'] or 'tools.qwen_world_safety_exploration.' + module not in argv:
            raise ValueError('CPU owner changed; no signal')
        cwd = str((proc / 'cwd').resolve())
        atomic(OUT / (study + '__OWNER_BEFORE.json'), old)
        fd = os.pidfd_open(old['pid'])
        signal.pidfd_send_signal(fd, signal.SIGTERM)
        os.close(fd)
        with (directory / 'owner.lock').open('a') as lock:
            end = time.monotonic() + 60
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() > end:
                        raise RuntimeError('Old owner has not released native lock; no duplicate')
                    time.sleep(1)
            state = read(directory / 'STATE.json')
            atomic(OUT / (study + '__STATE_BEFORE.json'), state)
            for expected in rows:
                current = verified_failure(study, state, expected['arm'])
                if current != expected:
                    raise ValueError('Failure changed during review')
                row = state['runs'][expected['arm']]
                node = row['stages'][str(expected['target'])]
                node.setdefault('reviewed_authorization_failures', []).append(copy.deepcopy(expected))
                node.pop('job')
                node['resume'] = expected['checkpoint']
                for key in ('job', 'host', 'error'):
                    row.pop(key, None)
                row.update(status='WAITING_AUTHORIZED_RESOURCE',
                           authorization_failure_review=str(OUT / 'QUALIFIED.json'),
                           replay_from_updates=expected['checkpoint_updates'])
            atomic(directory / 'STATE.json', state)
        log = (OUT / (study + '__cpu_owner.log')).open('ab', buffering=0)
        child = subprocess.Popen(argv, cwd=cwd, env=dict(os.environ, PYTHONPATH=cwd, CUDA_VISIBLE_DEVICES=''),
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        receipt = dict(study=study, old_pid=old['pid'], new_pid=child.pid, source=cwd,
                       original_argv=argv, GPU_signals=0, reviewed_arms=[x['arm'] for x in rows])
        launched.append(receipt)
        atomic(OUT / (study + '__LAUNCH.json'), receipt)
        print(study, 'CPU', child.pid, 'reviewed', len(rows), flush=True)
    atomic(OUT / 'APPLIED.json', dict(utc=datetime.now(timezone.utc).isoformat(), launches=launched,
                                    old_failed_receipts_preserved=True, same_scientific_sources=True,
                                    same_Adam_RNG_chain_and_advantages_from_native_checkpoint=True,
                                    unsaved_suffix_replay_not_lossless=True, GPU_signals=0))


if __name__ == '__main__':
    main()
