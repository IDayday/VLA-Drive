#!/usr/bin/python3
"""Queue-aware idle pressure for the explicitly authorized vlawm-zt4 host.

This is a companion to the sole Navtest dispatcher, not another dispatcher. It
only releases/restores the already captured, hash-locked synthetic pressure
script via the existing PID/start-time lease helper. It never launches inference
or signals real work. Code/public status live under /mnt/project; the existing
private capture stays outside Git (this host's /mnt/project is itself a repo).
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import time
import uuid

import navtest_cadence as cadence
import navtest_pressure_lease as lease
import perf_pressure_lease as pressure


def queue_demand(output, policy):
    """Read small queue receipts; full score validation belongs to the dispatcher."""
    pending = []
    for path in sorted((output / 'requests').glob('*.json')):
        request = cadence.read(path)
        if path.stem != request['key']:
            raise ValueError('Request key/path differs')
        if not cadence.eligible(policy, request):
            continue
        done = output / 'completed' / path.name
        if done.exists():
            marker = cadence.read(done)
            if marker.get('valid') is not True or marker.get('request') != request:
                raise ValueError('Completion receipt does not bind the request')
            continue
        pending.append(request['key'])
    return pending


def live_workers(output, scripts):
    """Capture actual local worker identities, including pre-CUDA model loading."""
    result = []
    for path in Path('/proc').glob('[0-9]*'):
        try:
            argv = [os.fsdecode(x) for x in (path / 'cmdline').read_bytes().split(b'\0') if x]
            if not set(scripts).intersection(argv) or '--output' not in argv:
                continue
            destination = Path(argv[argv.index('--output') + 1]).resolve()
            if not destination.is_relative_to(output):
                continue
            if '--worker' not in argv and not any(x.endswith('/infer_checkpoint.py') for x in scripts if x in argv):
                continue
            record = lease.proc_identity(int(path.name))
            if record and record['state'] not in ('Z', 'X'):
                result.append(f"{record['pid']}:{record['start_ticks']}")
        except (FileNotFoundError, ProcessLookupError):
            continue
    return sorted(set(result))


def saved_workers(output, keys, host):
    """Recover short-lived workers missed between polls from existing reports."""
    result = set()
    for key in keys:
        if Path(key).name != key:
            raise ValueError('Invalid remembered queue key')
        job = output / 'jobs' / key
        paths = list((job / 'shards').glob('*.json'))
        paths += list((job / 'attempts').glob('*/shards/*.json'))
        paths += list((output / 'qualification' / key / host.removesuffix('-worker-0') / 'shards').glob('*.json'))
        for path in paths:
            value = cadence.read(path)
            if value.get('host') == host and value.get('pid') and value.get('proc_start_ticks'):
                result.add(f"{int(value['pid'])}:{int(value['proc_start_ticks'])}")
    return sorted(result)


def resource_activity(helper, inspection, selected):
    """Conservative readiness check, including work that has not allocated CUDA.

    A script process is only a reason to yield; this classifier never authorizes
    a signal. Actual signals remain scoped to the hash-locked pressure tree.
    """
    pressure_pids = {pid for root in inspection['pressure_roots'] for pid in root['descendant_pids']}
    pressure_gpus = {gpu for root in inspection['pressure_roots'] for gpu in root['gpu_uuids']}
    monitors = {'navtest_idle_pressure.py', 'navtest_pressure_lease.py',
                'perf_pressure_lease.py', 'gpu_pressure_guard.py'}
    workers = []
    for pid, process in helper.backend.processes().items():
        if pid == os.getpid() or pid in pressure_pids:
            continue
        argv = process['argv']
        if not argv:
            continue
        name = Path(argv[0]).name
        launcher = name in ('torchrun', 'deepspeed', 'accelerate')
        script = None
        if name.startswith('python'):
            index = 1
            while index < len(argv):
                token = argv[index]
                if token in ('-W', '-X'):
                    index += 2
                    continue
                if token == '-m':
                    module = argv[index + 1] if index + 1 < len(argv) else ''
                    launcher = not module.startswith(('multiprocessing.', 'ipykernel', 'jupyter'))
                    break
                if token in ('-c', '-'):
                    break
                if token == '--':
                    index += 1
                    if index < len(argv):
                        script = Path(argv[index]).name
                    break
                if not token.startswith('-'):
                    script = Path(token).name
                    break
                index += 1
        elif name.endswith('.py'):
            script = name
        if launcher or (script is not None and script not in monitors):
            workers.append(f"{pid}:{process['start_ticks']}")
    rows = helper.backend.query('uuid,memory.used,utilization.gpu', 'gpu')
    cards = [{'uuid': row[0].strip(), 'memory_MiB': int(row[1]), 'utilization_percent': int(row[2])}
             for row in rows if row[0].strip() in selected]
    if {card['uuid'] for card in cards} != set(selected):
        raise ValueError('Resource samples do not cover the selected GPUs')
    unaccounted = [card['uuid'] for card in cards if card['uuid'] not in pressure_gpus
                   and (card['memory_MiB'] > 64 or card['utilization_percent'] > 0)]
    return {'workers': sorted(set(workers)), 'unaccounted_busy_gpu_uuids': unaccounted,
            'GPUs': cards, 'cpu_load_1min': os.getloadavg()[0]}


def interrupt_owned_restoration(helper, lease_id):
    """Cancel a partial restore when useful work arrives during pressure startup.

    This is an explicit, recorded cancellation of OUR replacement, not a retry
    of an unexpectedly exited filler. Preserve the original restore obligation
    and all private capture evidence. Reuse the existing targeted pidfd stop.
    """
    with helper.locked():
        journal = helper.read_lease(lease_id)
        if journal['state'] == 'RESTORED':
            return None  # Normal acquire uses a new lease for completed restores.
        captured = helper.private(journal)
        for root in journal['roots']:
            if not root['replacement']:
                continue
            replacement = dict(root['replacement'])
            if lease.same_process(replacement):
                snapshot = helper.backend.snapshot()
                row = snapshot['roots'].get(replacement['pid']) or snapshot['unresolved_roots'].get(replacement['pid'])
                original = captured['roots'][str(root['original']['pid'])]['root']
                if not row or not helper.candidate_matches(row['identity'], original):
                    return helper.defer(journal, 'OWNED_REPLACEMENT_IDENTITY_NOT_VERIFIED')
                helper.stop_invalid_replacement(journal, root, snapshot,
                    'USEFUL_WORK_INTERRUPTED_PRESSURE_STARTUP')
            evidence_name = root.get('invalid_restoration_capture')
            if not evidence_name:
                return helper.defer(journal, 'REPLACEMENT_EXIT_WITHOUT_CONTROLLED_STOP_EVIDENCE')
            evidence = helper.load(evidence_name)
            if any(lease.same_process(p) for p in evidence['tree']):
                return helper.defer(journal, 'OWNED_REPLACEMENT_STILL_EXITING')
            root.setdefault('interrupted_restorations', []).append({
                'replacement': replacement, 'tree_capture': evidence_name,
                'tree_capture_sha256': lease.digest(helper.root / evidence_name),
                'all_captured_processes_exited': True, 'interrupted_unix': time.time()})
            root.update(replacement=None, state='RELEASED')
            root.pop('invalid_restoration_capture', None)
        journal['state'] = 'ACQUIRED'
        journal.pop('deferred_reason', None)
        helper.write_lease(journal)
        return {'ok': True, 'action': 'interrupt_owned_restore', 'journal': helper.public(journal)}


class IdlePressure:
    def __init__(self, policy, helper, clock=time.monotonic, activity=resource_activity):
        self.policy, self.helper, self.clock = policy, helper, clock
        self.activity = activity
        self.output = Path(policy['dispatch_output']).resolve()
        self.idle_since = None
        with helper.locked():
            name = 'idle-controller.json'
            if (helper.root / name).exists():
                self.state = helper.load(name)
                if self.state['policy_identity'] != policy['identity']:
                    raise ValueError('Existing idle controller belongs to another policy')
            else:
                self.state = {'policy_identity': policy['identity'],
                    'lease_id': policy['initial_lease_id'], 'seen_requests': []}
                helper.read_lease(self.state['lease_id'])
                helper.save(name, self.state)

    def save(self):
        with self.helper.locked():
            self.helper.save('idle-controller.json', self.state)

    def journal(self):
        with self.helper.locked():
            return self.helper.read_lease(self.state['lease_id'])

    def step(self):
        policy = cadence.load_policy(self.policy['cadence_policy'])
        pending = queue_demand(self.output, policy)
        self.state['seen_requests'] = sorted(set(self.state['seen_requests']) | set(pending))
        inspection = self.helper.inspect(self.policy['gpu_uuids'])
        activity = self.activity(self.helper, inspection, self.policy['gpu_uuids'])
        workers = live_workers(self.output, self.policy['worker_scripts'])
        workers += activity['workers']
        # Real CUDA occupants are protected too, even when unrelated to Navtest.
        for app in inspection['genuine_or_unreadable_apps']:
            record = lease.proc_identity(app['pid'])
            if record and record['state'] not in ('Z', 'X'):
                workers.append(f"{record['pid']}:{record['start_ticks']}")
        workers = sorted(set(workers) | set(saved_workers(self.output,
            self.state['seen_requests'], self.policy['host'])))
        journal_path = self.helper.root / self.helper.name(self.state['lease_id'])
        journal = self.journal() if journal_path.exists() else None
        live = live_workers(self.output, self.policy['worker_scripts'])
        recorded_live = any(lease.same_process(w) for w in
            [*(journal['workers'] if journal else []), *self.helper.worker_identities(workers)])
        busy = bool(pending or live or inspection['genuine_or_unreadable_apps']
                    or activity['workers'] or activity['unaccounted_busy_gpu_uuids'] or recorded_live)
        action = None
        if busy:
            self.idle_since = None
            if journal and journal['state'] != 'RESTORED' and any(r['replacement'] for r in journal['roots']):
                action = interrupt_owned_restoration(self.helper, self.state['lease_id'])
                journal = self.journal()
            if journal and journal['state'] == 'RESTORED':
                self.state['lease_id'] = 'navtest-idle-zt4-' + uuid.uuid4().hex
                self.save()  # Persist the new intent before any process signal.
                journal = None
            recorded = {f"{w['pid']}:{w['start_ticks']}" for w in journal['workers']} if journal else set()
            if not (action and action.get('deferred')) and (journal is None or set(workers) - recorded or journal['state'] not in ('ACQUIRED', 'ACQUIRED_NO_PRESSURE')):
                action = self.helper.acquire(self.state['lease_id'], self.policy['gpu_uuids'], workers)
            status = 'AVAILABLE_FOR_NAVTEST' if pending else 'YIELDING_TO_REAL_WORK'
            if action and action.get('deferred'):
                status = 'PRESSURE_RELEASE_DEFERRED'
        elif inspection['unresolved_pressure_roots']:
            self.idle_since = None
            status = 'DEFERRED_UNRESOLVED_PRESSURE'
        elif journal is None:
            # Recovery of a persisted acquire intent; never invent a filler.
            action = self.helper.acquire(self.state['lease_id'], self.policy['gpu_uuids'], workers)
            status = 'RECOVERED_ACQUIRE_INTENT'
        elif journal['state'] == 'RESTORED':
            public = self.helper.public(journal)
            status = ('IDLE_PRESSURE_RUNNING' if any(r['original_alive'] or r['replacement_alive'] for r in public['roots'])
                      else 'PRESSURE_EXITED_REVIEW_REQUIRED')
        else:
            if self.idle_since is None:
                self.idle_since = self.clock()
            status = 'WAITING_FOR_CONFIRMED_IDLE'
            if self.clock() - self.idle_since >= self.policy['idle_grace_seconds']:
                # An enqueue or loading worker between polls takes precedence.
                if queue_demand(self.output, policy) or live_workers(self.output, self.policy['worker_scripts']):
                    self.idle_since = None
                    status = 'AVAILABLE_FOR_NAVTEST'
                elif (fresh_activity := self.activity(self.helper,
                        self.helper.inspect(self.policy['gpu_uuids']), self.policy['gpu_uuids']))['workers'] or fresh_activity['unaccounted_busy_gpu_uuids']:
                    self.idle_since = None
                    status = 'YIELDING_TO_REAL_WORK'
                elif not journal['workers'] and not workers:
                    # Do not misreport unobserved workers as "never started".
                    status = 'DEFERRED_WORKER_EXIT_EVIDENCE_REQUIRED'
                else:
                    action = self.helper.restore(self.state['lease_id'], workers)
                    status = 'IDLE_PRESSURE_RUNNING' if action.get('ok') and not action.get('deferred') else 'RESTORE_DEFERRED'
        self.save()
        journal = self.journal()
        public = self.helper.public(journal)
        result = {'schema': 'navtest_idle_pressure_status_v1', 'host': self.policy['host'],
            'observed_utc': datetime.now(timezone.utc).isoformat(),
            'controller_pid': os.getpid(), 'policy_identity': self.policy['identity'],
            'status': status, 'pending_requests': pending, 'lease_id': journal['lease_id'],
            'lease_state': journal['state'],
            'idle_seconds': 0 if self.idle_since is None else round(self.clock() - self.idle_since, 1),
            'idle_required_seconds': self.policy['idle_grace_seconds'],
            'poll_seconds': self.policy['poll_seconds'], 'resource_activity': activity,
            'workers_alive': sum(w['alive'] for w in public['workers']),
            'pressure_alive': any(r['original_alive'] or r['replacement_alive'] for r in public['roots']),
            'deferred_reason': action.get('journal', {}).get('deferred_reason')
                if action and action.get('deferred') else None,
            'signals_limited_to_hash_locked_pressure': True,
            'training_and_dispatcher_unchanged': True}
        cadence.atomic(Path(self.policy['status_output']), result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    policy = cadence.read(args.policy)
    identity = hashlib.sha256(json.dumps({k: v for k, v in policy.items() if k != 'identity'},
        sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if policy.get('schema') != 'navtest_idle_pressure_policy_v1' or policy.get('identity') != identity:
        raise ValueError('Idle pressure policy identity differs')
    if policy['host'] != socket.gethostname() or policy['host'] != 'training-vlawm-zt4-worker-0':
        raise ValueError('Explicitly authorized vlawm-zt4 local host required')
    if lease.digest(pressure.SCRIPT) != pressure.SHA256:
        raise ValueError('Inspected pressure script changed')
    if policy['private_state_root'] != '/var/tmp/structured-world-fgtr-execution-private-20261011':
        raise ValueError('Use the existing private pressure journal, outside Git')
    if not 5 <= policy['poll_seconds'] <= 60 or policy['idle_grace_seconds'] != 600:
        raise ValueError('Invalid polling/idle grace')
    helper = lease.PressureLease(policy['private_state_root'],
        lease.LocalHost(pressure.SCRIPT), timeout=15)
    lock_path = Path(policy['status_output']).with_name('idle-controller.lock')
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stop = False
    def terminate(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        controller = IdlePressure(policy, helper)
        previous = None
        while not stop:
            try:
                if lease.digest(pressure.SCRIPT) != pressure.SHA256:
                    raise ValueError('Inspected pressure script changed')
                result = controller.step()
            except Exception as error:
                cadence.atomic(Path(policy['status_output']), {
                    'status': 'STOPPED_FOR_REVIEW', 'host': policy['host'],
                    'controller_pid': os.getpid(),
                    'observed_utc': datetime.now(timezone.utc).isoformat(),
                    'error_type': type(error).__name__,
                    'reason': str(error) if isinstance(error, lease.LeaseError) else 'Controller validation failed',
                    'private_restoration_obligation_preserved': True})
                raise
            current = (result['status'], result['lease_id'], result['deferred_reason'], tuple(result['pending_requests']))
            if current != previous:
                print(json.dumps(result, sort_keys=True), flush=True)
                previous = current
            if args.once:
                break
            time.sleep(policy['poll_seconds'])


if __name__ == '__main__':
    main()
