"""Exact-receipt Qwen-World resource-return guardian, independent of workers.

--once is always inspection-only. Production requires --launch --execute.
No process group is killed: every signal uses a PIDFD after PID/start/cmdline
revalidation. Unknown processes and outputs outside explicit task roots remain
untouched. A newer unlimited/extended task grant supersedes this deadline.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time

SCHEMA = 'world_exact_QwenWorld_deadline_guard_v1'
POLICY = '/mnt/project/server_dispatch_policy.json'
TASK_ROOTS = {
    '/mnt/project/qwen-world-safety-exploration-20261031': 'qwen_world_safety_exploration_20261031',
    '/mnt/project/qwen-world-rl-optimization-20261030': 'qwen_world_safety_exploration_20261031',
    '/mnt/project/qwen-world-performance-rl-20261103': 'qwen_world_safety_exploration_20261031',
    '/mnt/project/qwen-world-four-frameworks-20261029': 'qwen_world_compare_20261029',
    '/mnt/project/qwen-world-il-research-20261030': 'qwen_world_compare_20261029',
}
TRAIN_MODULES = {
    'tools.qwen_world_abcd_research.train', 'tools.qwen_world_compare.train_il',
    'tools.qwen_world_compare.train_rl', 'tools.qwen_world_postcot.train_il',
    'tools.qwen_world_postcot.train_rl', 'tools.qwen_world_postcot.train_scope_rl',
    'tools.qwen_world_safety_exploration.train', 'tools.qwen_world_performance.train',
    'tools.qwen_world_multiscene.train', 'tools.qwen_world_verified_quality.train',
    'tools.qwen_world_multiscene.train',
    'tools.qwen_world_structured_progress.train',
}
EVAL_MODULES = {
    'tools.qwen_world_compare.evaluate', 'tools.qwen_world_research.evaluate',
    'tools.qwen_world_abcd_research.evaluate', 'tools.qwen_world_postcot.evaluate',
    'tools.qwen_world_safety_exploration.evaluate', 'tools.qwen_world_performance.evaluate',
    'tools.qwen_world_performance.qualify_nominal',
}
WORKER_MODULES = {'tools.qwen_world_safety_exploration.job_worker',
                  'tools.qwen_world_compare.job_worker', 'tools.qwen_world_performance.job_worker'}


def atomic_json(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.'+path.name+'.'+str(os.getpid())+'.tmp')
    with temp.open('w') as handle:
        json.dump(value, handle, indent=2, allow_nan=False);handle.write('\n');handle.flush();os.fsync(handle.fileno())
    os.replace(temp, path)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def read_process(pid, proc_root='/proc'):
    try:
        root = Path(proc_root)/str(pid)
        raw = (root/'stat').read_text();fields = raw[raw.rfind(')')+2:].split()
        args = [v for v in (root/'cmdline').read_bytes().decode(errors='replace').split('\0') if v]
        status = (root/'status').read_text()
        caught = next(int(line.split(':', 1)[1].strip(), 16) for line in status.splitlines() if line.startswith('SigCgt:'))
        return dict(pid=int(pid), start_ticks=int(fields[19]), ppid=int(fields[1]), state=fields[0],
                    args=args, cmdline_sha256=sha(json.dumps(args, separators=(',', ':')).encode()),
                    usr1_handler=bool(caught & (1 << (signal.SIGUSR1-1))))
    except (OSError, ValueError, IndexError, StopIteration):
        return None


def argument(args, name):
    try:return args[args.index(name)+1]
    except (ValueError, IndexError):return None


def module_names(args):
    names = [args[i+1] for i, value in enumerate(args[:-1]) if value == '-m']
    known = TRAIN_MODULES | EVAL_MODULES | WORKER_MODULES
    for value in args:
        if value.endswith('.py'):
            for name in known:
                if value.endswith('/'+name.replace('.', '/')+'.py'):
                    names.append(name)
    return names


def task_for_output(output, allowed_tasks):
    if not output:return None
    candidate = Path(output).resolve()
    for root, task in TASK_ROOTS.items():
        if task in allowed_tasks and (candidate == Path(root) or Path(root) in candidate.parents):return task
    return None


def classify_direct(row, allowed_tasks, host):
    """Module AND exact task output path are required, not a qwen substring."""
    args = row['args'];names = module_names(args)
    worker = next((n for n in names if n in WORKER_MODULES), None)
    receipt_root = None
    if worker:
        receipt_root = argument(args, '--receipt-root')
        nested = argument(args, '--command-json')
        try:
            args = json.loads(nested)
            if not isinstance(args, list) or not all(isinstance(v, str) for v in args):return None
            names = module_names(args)
            if not task_for_output(receipt_root, allowed_tasks):return None
            verified = False
            for name in ['RUNNING.json', 'DISPATCH.json']:
                p = Path(receipt_root)/name
                if not p.is_file():continue
                value = json.loads(p.read_text())
                if (value.get('host') == host and value.get('worker_pid') == row['pid']
                        and value.get('worker_start_ticks') == row['start_ticks']):verified = True
            if not verified:return None
        except (OSError, ValueError, TypeError):return None
    entry = next((n for n in reversed(names) if n in TRAIN_MODULES | EVAL_MODULES), None)
    if not entry:return None
    output = argument(args, '--output')
    if output is None:
        spec = argument(args, '--spec')
        if spec and task_for_output(spec, allowed_tasks):
            try:output = json.loads(Path(spec).read_text()).get('output')
            except (OSError, ValueError):return None
    task = task_for_output(output, allowed_tasks)
    if task is None:return None
    return dict(task=task, output=str(Path(output).resolve()), entry=entry,
                role='worker' if worker else 'native' if entry in TRAIN_MODULES else 'evaluation',
                receipt_root=receipt_root, ownership='exact_module_and_task_output')


def discover_owned(processes, allowed_tasks, host, previous):
    owned = {}
    known = {(r['pid'], r['start_ticks'], r['cmdline_sha256']): r for r in previous}
    for pid, row in processes.items():
        if row['state'] in ('Z', 'X'):continue
        direct = classify_direct(row, allowed_tasks, host)
        prior = known.get((pid, row['start_ticks'], row['cmdline_sha256']))
        if direct:
            owned[pid] = dict(row, **direct)
        elif prior and prior.get('task') in allowed_tasks:
            owned[pid] = dict(row, **{k: prior[k] for k in ['task', 'output', 'entry', 'role', 'receipt_root', 'ownership']})
            owned[pid]['ownership'] = 'previously_verified_exact_identity'
    for _ in range(32):
        added = False
        for pid, row in processes.items():
            if pid in owned or row['state'] in ('Z', 'X') or row['ppid'] not in owned:continue
            parent = owned[row['ppid']]
            owned[pid] = dict(row, task=parent['task'], output=parent['output'], entry=parent['entry'],
                              role='descendant', receipt_root=None, ownership='verified_live_parent_chain')
            added = True
        if not added:break
    return list(owned.values())


def phase_for(now, deadline, guard=900):
    if now >= deadline-30:return 'kill'
    if now >= deadline-90:return 'term'
    if now >= deadline-guard:return 'cooperative'
    return 'monitor'


def identity_matches(expected, actual):
    return bool(actual and actual['state'] not in ('Z', 'X') and all(expected[k] == actual[k] for k in ('pid', 'start_ticks', 'cmdline_sha256')))


def verified_pidfd_signal(row, sig, *, inspect_only):
    result = dict(pid=row['pid'], start_ticks=row['start_ticks'], signal=int(sig), status='INSPECT_ONLY')
    if inspect_only:return result
    current = read_process(row['pid'])
    if not identity_matches(row, current):
        return dict(result, status='REFUSED_IDENTITY_CHANGED')
    if sig == signal.SIGUSR1 and not current.get('usr1_handler'):
        return dict(result, status='REFUSED_NATIVE_HANDLER_CHANGED')
    fd = None
    try:
        if hasattr(os, 'pidfd_open'):
            fd = os.pidfd_open(row['pid'], 0)
        else:
            if os.uname().machine not in ('x86_64', 'aarch64'):raise RuntimeError('PIDFD syscall architecture not qualified')
            libc = ctypes.CDLL(None, use_errno=True);fd = libc.syscall(434, int(row['pid']), 0)
            if fd < 0:raise OSError(ctypes.get_errno(), 'pidfd_open')
        current = read_process(row['pid'])
        if not identity_matches(row, current):
            return dict(result, status='REFUSED_IDENTITY_CHANGED_AFTER_PIDFD')
        if sig == signal.SIGUSR1 and not current.get('usr1_handler'):
            return dict(result, status='REFUSED_NATIVE_HANDLER_CHANGED_AFTER_PIDFD')
        if hasattr(signal, 'pidfd_send_signal'):
            signal.pidfd_send_signal(fd, sig, None, 0)
        else:
            libc = ctypes.CDLL(None, use_errno=True)
            if libc.syscall(424, fd, int(sig), 0, 0) < 0:raise OSError(ctypes.get_errno(), 'pidfd_send_signal')
        return dict(result, status='SIGNALLED_EXACT_PIDFD')
    except ProcessLookupError:
        return dict(result, status='ALREADY_EXITED')
    except (OSError, RuntimeError) as error:
        return dict(result, status='REFUSED_OR_EXITED', error=str(error))
    finally:
        if fd is not None and fd >= 0:os.close(fd)


def actions_for(row, phase):
    actions = []
    if phase != 'monitor' and row['role'] == 'worker' and row.get('receipt_root'):
        actions.append(('stop_file', str(Path(row['receipt_root'])/'STOP_REQUESTED')))
    if phase != 'monitor' and row['role'] == 'native' and row['usr1_handler']:
        actions.append(('signal', int(signal.SIGUSR1)))
    if phase in ('term', 'kill'):actions.append(('signal', int(signal.SIGTERM)))
    if phase == 'kill':actions.append(('signal', int(signal.SIGKILL)))
    return actions


def checkpoint_receipt(output):
    """Small native metadata only; never load model/optimizer tensors."""
    candidates = []
    for name in ('progress.json', 'LATEST.json', 'latest.json'):
        try:
            value = json.loads((Path(output)/name).read_text())
            path = Path(value['checkpoint']).resolve()
            if not task_for_output(str(path), set(TASK_ROOTS.values())):continue
            marker = path/'COMPLETE'
            if not marker.is_file():continue
            raw = marker.read_bytes()
            try:seal = json.loads(raw)
            except ValueError:seal = {}
            inventory = path/'files.json'
            verified = bool(isinstance(seal, dict) and inventory.is_file() and seal.get('files_sha256') == sha(inventory.read_bytes()))
            candidates.append(dict(path=str(path), marker_sha256=sha(raw), updates=value.get('updates', seal.get('updates') if isinstance(seal, dict) else None),
                                   inventory_seal_verified=verified, tensor_hashes_not_reread=True))
        except (OSError, ValueError, KeyError, TypeError):continue
    return max(candidates, key=lambda r:r['updates'] if isinstance(r['updates'], (int, float)) else -1) if candidates else None


def remote_tick(payload):
    """Runs in each exact authorized host using standard-library Python only."""
    now = payload['now'] if payload['inspect_only'] and payload.get('now') is not None else time.time()
    phase = phase_for(now, payload['deadline_unix'], payload['guard_seconds'])
    processes = {}
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            row = read_process(int(path.name))
            if row:processes[row['pid']] = row
    owned = discover_owned(processes, payload['allowed_tasks'], payload['host'], payload.get('known', []))
    done = dict(payload.get('done', {}));actions = []
    # Native/evaluation ranks first; their admission workers are last.
    for row in sorted(owned, key=lambda r: (r['role'] == 'worker', -r['pid'])):
        for kind, value in actions_for(row, phase):
            key = f"{row['pid']}:{row['start_ticks']}:{kind}:{value}"
            if key in done:continue
            if not identity_matches(row, read_process(row['pid'])):continue
            if kind == 'stop_file':
                event = dict(pid=row['pid'], start_ticks=row['start_ticks'], stop_file=value, status='INSPECT_ONLY')
                if not payload['inspect_only']:
                    path = Path(value)
                    if path.is_symlink():
                        event['status'] = 'REFUSED_STOP_FILE_SYMLINK'
                    else:
                        fd = os.open(path, os.O_CREAT|os.O_WRONLY|os.O_NOFOLLOW, 0o600);os.close(fd)
                        event['status'] = 'OWN_JOB_STOP_REQUESTED'
            else:
                event = verified_pidfd_signal(row, value, inspect_only=payload['inspect_only'])
            event.update(kind=kind, at=now, identity_sha256=row['cmdline_sha256'], task=row['task'], output=row['output'])
            actions.append(event)
            if not payload['inspect_only'] and event['status'] in ('SIGNALLED_EXACT_PIDFD', 'OWN_JOB_STOP_REQUESTED', 'ALREADY_EXITED'):
                done[key] = event
    outputs = {r['output'] for r in owned+payload.get('known', [])}
    unmatched = []
    owned_ids = {row['pid'] for row in owned}
    for row in processes.values():
        names = [name for name in module_names(row['args']) if name.startswith('tools.qwen_world_')]
        if names and row['pid'] not in owned_ids and row['state'] not in ('Z', 'X'):
            unmatched.append(dict(pid=row['pid'], start_ticks=row['start_ticks'], modules=names,
                                  cmdline_sha256=row['cmdline_sha256'], args=row['args'], action='UNMATCHED_UNTOUCHED'))
    return dict(schema=SCHEMA, host=payload['host'], actual_hostname=socket.gethostname(), observed_unix=now,
                phase=phase, inspect_only=payload['inspect_only'], owned_processes=owned, actions=actions, done=done,
                latest_native_checkpoints={output:checkpoint_receipt(output) for output in sorted(outputs)},
                unmatched_QwenWorld_processes=unmatched,
                unowned_processes_untouched=True, unknown_process_count=len(processes)-len(owned),
                pidfd_open_available=hasattr(os, 'pidfd_open'), pidfd_send_signal_available=hasattr(signal, 'pidfd_send_signal'))


def load_receipt(path):
    raw = Path(path).read_bytes();value = json.loads(raw)
    if (value.get('schema') != 'world_performance_user_temporary_resource_deadline_v1'
            or value.get('checkpoint_guard_seconds') != 900 or len(value.get('hosts', [])) != 6
            or len(set(value['hosts'])) != 6 or not all(v.startswith('training-') for v in value['hosts'])
            or value['deadline_unix'] <= value['requested_unix']):
        raise ValueError('Explicit six-host/900s immutable deadline receipt required')
    return value, sha(raw)


def current_task_scope(host, deadline):
    policy = json.loads(Path(POLICY).read_text());result = []
    for task in sorted(set(TASK_ROOTS.values())):
        grant = policy.get('task_authorizations', {}).get(task, {})
        if host in grant.get('permanent_hosts', []) or host in grant.get('unbounded_temporary_hosts', []):continue
        if host not in grant.get('temporary_hosts', []) or grant.get('expires_unix') != deadline:continue
        record = json.loads(Path(grant['record']).read_text())
        keys = ('permanent_hosts', 'temporary_hosts', 'unbounded_temporary_hosts', 'expires_unix')
        if any(grant.get(key) != record.get(key) for key in keys):raise ValueError('Current grant differs from its sealed record')
        result.append(task)
    return result


def invoke_remote(host, payload, source, *, timeout=18):
    code = source.rsplit("\nif __name__ == '__main__':", 1)[0]
    code += '\nprint(json.dumps(remote_tick('+repr(payload)+'),allow_nan=False))\n'
    command = ['/usr/bin/python3', '-']
    if host != socket.gethostname().removesuffix('-worker-0'):
        command = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', host, shlex.join(command)]
    done = subprocess.run(command, input=code, text=True, capture_output=True, timeout=timeout)
    if done.returncode:raise RuntimeError(done.stderr[-1500:])
    return json.loads(done.stdout)


def one_tick(receipt, directory, source, *, inspect_only, now=None):
    directory = Path(directory);previous = {}
    for host in receipt['hosts']:
        path = directory/'hosts'/host/'LATEST.json'
        if path.is_file():previous[host] = json.loads(path.read_text())
    def run(host):
        try:
            tasks = current_task_scope(host, receipt['deadline_unix'])
            if not tasks:return host, dict(host=host, status='SUPERSEDED_OR_OUTSIDE_CURRENT_TASK_GRANTS', owned_processes=[], no_actions=True)
            old = previous.get(host, {})
            payload = dict(host=host, deadline_unix=receipt['deadline_unix'], guard_seconds=900,
                           allowed_tasks=tasks, inspect_only=inspect_only, now=now,
                           known=old.get('owned_processes', []), done=old.get('done', {}))
            return host, invoke_remote(host, payload, source)
        except Exception as error:
            return host, dict(host=host, status='HOST_REVIEW_REQUIRED', error=str(error)[-1800:], no_actions_claimed=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows = dict(pool.map(run, receipt['hosts']))
    stamp = time.time() if now is None else now
    for host, row in rows.items():
        folder = directory/'hosts'/host;atomic_json(folder/'LATEST.json', row)
        if row.get('actions'):atomic_json(folder/'events'/f'{int(stamp*1000)}.json', row)
    state = dict(schema=SCHEMA, status='INSPECTION_ONLY' if inspect_only else 'MONITORING', heartbeat_unix=time.time(),
                 deadline_unix=receipt['deadline_unix'], checkpoint_guard_unix=receipt['deadline_unix']-900,
                 stages=dict(cooperative=receipt['deadline_unix']-900, TERM=receipt['deadline_unix']-90, KILL=receipt['deadline_unix']-30),
                 hosts=rows, native_checkpoint_guarantee='Cooperative request first; forced expiry may lose uncommitted work, never called lossless.')
    atomic_json(directory/('INSPECTION.json' if inspect_only else 'STATE.json'), state)
    return state


def owner_alive(directory):
    path = Path(directory)/'OWNER.json'
    if not path.is_file():return False
    old = json.loads(path.read_text());row = read_process(old['pid'])
    return bool(row and row['state'] not in ('Z', 'X') and row['start_ticks'] == old['start_ticks']
                and '--run' in row['args'] and '--execute' in row['args'] and old.get('source') in row['args'])


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--receipt', required=True);p.add_argument('--directory')
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--once', action='store_true');modes.add_argument('--run', action='store_true');modes.add_argument('--launch', action='store_true')
    p.add_argument('--execute', action='store_true');p.add_argument('--now', type=float)
    args = p.parse_args()
    receipt, receipt_sha = load_receipt(args.receipt)
    directory = Path(args.directory or Path(args.receipt).parent/'DEADLINE_GUARD');directory.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().read_text()
    if args.once:
        result = one_tick(receipt, directory, source, inspect_only=True, now=args.now)
        print(json.dumps(dict(status=result['status'], directory=str(directory), hosts={k:len(v.get('owned_processes', [])) for k,v in result['hosts'].items()})))
        return 0
    if not args.execute or args.now is not None:raise ValueError('Production requires explicit --execute and real wall clock')
    if args.launch:
        with (directory/'launch.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            if owner_alive(directory):raise RuntimeError('A verified deadline guardian is already running')
            if (directory/'STOP_OWNER').exists():raise RuntimeError('Explicit owner stop persists; refusing automatic restart')
            frozen = directory/'runtime'/'deadline_guard_v1.py';frozen.parent.mkdir(parents=True, exist_ok=True)
            if frozen.exists() and sha(frozen.read_bytes()) != sha(source.encode()):raise ValueError('Immutable guardian source already differs')
            if not frozen.exists():frozen.write_text(source)
            command = [sys.executable, '-u', str(frozen), '--receipt', str(Path(args.receipt).resolve()), '--directory', str(directory), '--run', '--execute']
            log = (directory/'stdout.log').open('ab')
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            row = read_process(child.pid)
            launch = dict(pid=child.pid, start_ticks=row['start_ticks'], source=str(frozen), source_sha256=sha(source.encode()),
                          receipt=str(Path(args.receipt).resolve()), receipt_sha256=receipt_sha, command=command, launched_unix=time.time(),
                          scope=TASK_ROOTS, six_hosts=receipt['hosts'], no_scientific_source_changes=True)
            atomic_json(directory/'LAUNCH.json', launch);print(json.dumps(launch));return 0
    with (directory/'owner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        row = read_process(os.getpid())
        atomic_json(directory/'OWNER.json', dict(pid=os.getpid(), start_ticks=row['start_ticks'], source=str(Path(__file__).resolve()),
                    source_sha256=sha(source.encode()), host=socket.gethostname(), receipt_sha256=receipt_sha, started_unix=time.time()))
        last_scan = 0.;last = None
        while not (directory/'STOP_OWNER').exists():
            current, digest = load_receipt(args.receipt)
            if digest != receipt_sha:raise RuntimeError('Deadline request changed; explicit adoption required')
            now = time.time();near = now >= receipt['deadline_unix']-900
            if near or now-last_scan >= 300:
                last = one_tick(current, directory, source, inspect_only=False);last_scan = time.time()
                if now >= receipt['deadline_unix'] and all(not v.get('owned_processes') and v.get('status') != 'HOST_REVIEW_REQUIRED' for v in last['hosts'].values()):
                    superseded = any(v.get('status') == 'SUPERSEDED_OR_OUTSIDE_CURRENT_TASK_GRANTS' for v in last['hosts'].values())
                    last['status'] = 'COMPLETE_WITH_SUPERSEDED_HOSTS' if superseded else 'COMPLETE_OWNED_EXIT_VERIFIED'
                    atomic_json(directory/'STATE.json', last);return 0
                if now >= receipt['deadline_unix']+300:
                    last['status'] = 'DEADLINE_REVIEW_REQUIRED';atomic_json(directory/'STATE.json', last);return 2
            if last:
                last['heartbeat_unix'] = time.time();atomic_json(directory/'STATE.json', last)
            wait = 5 if near else min(30, max(1, receipt['deadline_unix']-900-time.time()))
            time.sleep(wait)
        if last:last['status'] = 'STOPPED_OWNER_ONLY';atomic_json(directory/'STATE.json', last)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
