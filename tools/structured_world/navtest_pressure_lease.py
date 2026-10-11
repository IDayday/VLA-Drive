#!/usr/bin/python3
"""Local, UUID-scoped leases of existing gpu_stress.py processes (stdlib only).

The scheduler must run this CLI ON the target host, including when routed through SSH:
  /usr/bin/python3 -B navtest_pressure_lease.py inspect --gpu-uuids GPU-a,GPU-b
  /usr/bin/python3 -B navtest_pressure_lease.py acquire --lease-id JOB-HOST \
      --gpu-uuids GPU-a,GPU-b --state-root /controlled/artifacts/private-pressure
  # Record the actual remote worker, including its start ticks, before useful work:
  ... acquire --lease-id JOB-HOST --gpu-uuids GPU-a,GPU-b --worker-pid PID:TICKS
  # In finally, after waiting for every actual worker (not just its SSH client):
  ... restore --lease-id JOB-HOST --worker-pid PID:TICKS
  # If acquire/qualification failed BEFORE launching any workers instead:
  ... restore --lease-id JOB-HOST --no-workers-started
  ... status --lease-id JOB-HOST

All actions emit redacted JSON. Exit codes: 0 success, 3 deferred obligation,
2 refused/error. Keep calling restore for deferred leases; never discard their
journals. PID alone is accepted only while live, to capture its local start ticks;
use PID:TICKS thereafter. No SSH, broad signals, generic Python classification,
new idle fillers, fixed GPU count, or training-process intervention is performed.

Only acquire/restore mutate processes. The production CLI trusts ONLY
/mnt/project/gpu_stress.py. LocalHost's explicit alternate script argument exists
for owned CPU fixtures, not automatic alternative-script recognition. Reuses the
guard's exact-entrypoint classifier and descendant traversal, not its unsafe PID
signals, physical-index binding approximation, or new-filler creation routines.

Private captures include argv/env, executable, cwd, UID/GID/groups, descendants,
rlimits, affinity/nice/scheduling, script/executable hashes and GPU mapping.
They are separate 0600 files in a 0700 artifact directory, never stdout or Git.
Restore requires the original UID and cgroup placement (root may restore UID),
unchanged script/interpreter bytes and unchanged index->UUID mapping. Original
argv/env/limits are replayed; session/parent PID are necessarily new. A persisted
ambiguous signal/spawn intent is deferred rather than guessed or duplicated.
Full UUID masks and unmasked all-device selection resolve directly. Numeric
masks/default partial selection require complete actual compute-app UUID evidence;
nvidia-smi indices and CUDA_DEVICE_ORDER alone do not prove CUDA ordinal order.
That evidence and inventory are captured; replay requires unchanged host/boot/PID
namespace, inventory and environment, followed by actual UUID-app verification.
GPU ownership races with noncooperating jobs cannot be made atomic by this local
lease; recheck before worker dispatch. gpu_stress.py itself also yields to real
CUDA work. Unsupported/unreadable identities or bindings fail closed.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import shlex
import signal
import socket
import stat
import subprocess
import sys
import time
import uuid

sys.dont_write_bytecode = True
TRUSTED_SCRIPT = Path('/mnt/project/gpu_stress.py')
GUARD = Path('/mnt/project/skill/gpu-pressure-guard/scripts/gpu_pressure_guard.py')
DEFAULT_STATE_ROOT = Path('/mnt/project/structured-world-fgtr-round1-artifacts/20261008/navtest_pressure_leases_private')
SCHEMA = 'navtest_pressure_lease_v1'


class LeaseError(RuntimeError):
    """Only fixed, credential-free messages may cross the CLI boundary."""


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def proc_identity(pid):
    """Parse stat after its final ')' (comm can contain spaces/parentheses)."""
    try:
        fields = Path(f'/proc/{int(pid)}/stat').read_text().rpartition(')')[2].split()
        return {'pid': int(pid), 'start_ticks': int(fields[19]),
                'ppid': int(fields[1]), 'state': fields[0],
                'uid': Path(f'/proc/{int(pid)}').stat().st_uid}
    except (OSError, ValueError, IndexError):
        return None


def same_process(record):
    current = proc_identity(record['pid'])
    return bool(current and current['state'] not in ('Z', 'X') and
                current['start_ticks'] == record['start_ticks'] and
                current['uid'] == record['uid'])


def same_execution(record):
    if not same_process(record):
        return False
    try:
        pid = record['pid']
        argv = [os.fsdecode(x) for x in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if x]
        return (argv == record['argv'] and os.readlink(f'/proc/{pid}/exe') == record['executable'] and
                os.readlink(f'/proc/{pid}/cwd') == record['cwd'])
    except OSError:
        return False


def open_pidfd(record):
    """The second start-time check closes the PID-reuse-before-open race."""
    if not same_process(record):
        return None
    try:
        fd = os.pidfd_open(record['pid'], 0)
    except ProcessLookupError:
        return None
    if not same_process(record):
        os.close(fd)
        return None
    return fd


def read_environment(pid):
    raw = Path(f'/proc/{pid}/environ').read_bytes()
    result = {}
    for item in raw.split(b'\0'):
        if item:
            key, value = item.split(b'=', 1)
            result[os.fsdecode(key)] = os.fsdecode(value)
    return result


def binding(argv, env, inventory, observed_uuids=None):
    """gpu_stress --gpus uses LOGICAL indices within its own CVM, not ours."""
    by_index = {str(row['index']): row['uuid'] for row in inventory}
    by_uuid = {row['uuid'] for row in inventory}
    if len(by_index) != len(inventory) or len(by_uuid) != len(inventory):
        raise LeaseError('ambiguous_gpu_inventory')
    raw_visible = env.get('CUDA_VISIBLE_DEVICES')
    if raw_visible is None:
        visible = [None] * len(inventory)
    elif raw_visible in ('', '-1'):
        visible = []
    else:
        visible = []
        for token in raw_visible.split(','):
            token = token.strip()
            if token in by_uuid:
                resolved = token
            elif token.isdecimal() and int(token) < len(inventory):
                resolved = None  # CUDA ordinal: never infer from NVML index.
            else:
                raise LeaseError('unresolved_CUDA_VISIBLE_DEVICES_binding')
            visible.append(resolved)
        tokens = [token.strip() for token in raw_visible.split(',')]
        if len(set(tokens)) != len(tokens):
            raise LeaseError('duplicate_CUDA_VISIBLE_DEVICES_binding')
    choices = []
    for index, token in enumerate(argv):
        if token == '--gpus':
            if index + 1 == len(argv):
                raise LeaseError('missing_pressure_gpu_argument')
            choices.append(argv[index + 1])
        elif token.startswith('--gpus='):
            choices.append(token.split('=', 1)[1])
    if len(choices) > 1:
        raise LeaseError('ambiguous_pressure_gpu_argument')
    raw = choices[0] if choices else 'all'
    try:
        logical = list(range(len(visible))) if raw == 'all' else [int(x) for x in raw.split(',')]
    except ValueError:
        raise LeaseError('invalid_pressure_gpu_argument') from None
    if not logical or any(i < 0 or i >= len(visible) for i in logical):
        raise LeaseError('invalid_pressure_gpu_binding')
    logical = sorted(set(logical))
    if raw_visible is None and len(logical) == len(inventory):
        bound, proof = sorted(by_uuid), 'all_inventory'
    elif all(visible[i] is not None for i in logical) and all(v is not None for v in visible):
        bound, proof = sorted({visible[i] for i in logical}), 'full_uuid_mask'
    elif (raw_visible is not None and all(t.isdecimal() for t in tokens) and
          set(int(t) for t in tokens) == set(range(len(inventory))) and
          len(logical) == len(inventory)):
        bound, proof = sorted(by_uuid), 'all_inventory'
    else:
        observed = set(observed_uuids or ())
        if len(observed) != len(logical) or not observed.issubset(by_uuid):
            raise LeaseError('ambiguous_CUDA_order_requires_complete_actual_GPU_apps')
        bound, proof = sorted(observed), 'actual_compute_apps'
    return {'gpu_uuids': bound, 'proof': proof, 'logical_indices': logical,
            'inventory': sorted(inventory, key=lambda row: row['index'])}


class LocalHost:
    def __init__(self, trusted_script=TRUSTED_SCRIPT):
        self.script = Path(trusted_script).resolve()
        spec = importlib.util.spec_from_file_location('_navtest_pressure_guard', GUARD)
        self.guard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.guard)

    def context(self):
        return {'host': socket.gethostname(),
                'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'pid_namespace': os.readlink('/proc/self/ns/pid')}

    def query(self, fields, section):
        result = subprocess.run(['nvidia-smi', f'--query-{section}={fields}',
                                 '--format=csv,noheader,nounits'], capture_output=True,
                                text=True, timeout=15)
        if result.returncode:
            raise LeaseError('gpu_inventory_or_apps_query_failed')
        return [row.split(',') for row in result.stdout.splitlines() if row.strip()]

    def inventory(self):
        return [{'index': int(row[0]), 'uuid': row[1].strip()}
                for row in self.query('index,uuid', 'gpu')]

    def apps(self):
        return [{'pid': int(row[0]), 'gpu_uuid': row[1].strip()}
                for row in self.query('pid,gpu_uuid', 'compute-apps')]

    def processes(self):
        processes = {}
        for path in Path('/proc').iterdir():
            if not path.name.isdigit():
                continue
            pid = int(path.name)
            identity = proc_identity(pid)
            if not identity or identity['state'] in ('Z', 'X'):
                continue
            try:
                argv = [os.fsdecode(x) for x in (path / 'cmdline').read_bytes().split(b'\0') if x]
                cwd = os.readlink(path / 'cwd')
            except OSError:
                continue
            processes[pid] = dict(identity, argv=argv, cwd=cwd)
        return processes

    def is_pressure(self, pid, process):
        argv = process['argv']
        if not argv:
            return False
        try:
            exe = Path(os.readlink(f'/proc/{pid}/exe')).name
        except OSError:
            return False
        if not re.fullmatch(r'python(?:[0-9]+(?:\.[0-9]+)*)?', exe):
            return False
        # Parse before calling the mature classifier: its -W/-X handling can
        # mistake an option operand for a script. Pass only the resolved actual
        # entrypoint to its proven cwd/path comparison, never a search of argv.
        if argv[0].endswith('.py'):
            index = 0
        elif re.fullmatch(r'python(?:[0-9]+(?:\.[0-9]+)*)?', Path(argv[0]).name):
            index = 1
        else:
            return False
        switches = {'-B', '-E', '-I', '-O', '-OO', '-P', '-s', '-S', '-u', '-q', '-i', '-x', '-b', '-bb'}
        while index < len(argv):
            token = argv[index]
            if token == '--':
                index += 1
                break
            if token in ('-c', '-m'):
                return False
            if token in ('-W', '-X'):
                index += 2
            elif token in switches or token.startswith(('-W', '-X')):
                index += 1
            elif token.startswith('-'):
                return False
            else:
                break
        if index >= len(argv):
            return False
        path = Path(argv[index])
        candidate = path if path.is_absolute() else Path(process['cwd']) / path
        if candidate.resolve() != self.script:
            return False
        normalized = shlex.join([os.readlink(f'/proc/{pid}/exe'), argv[index]])
        return self.guard._command_uses_exact_script(pid, normalized, self.script)

    def snapshot(self):
        inventory, apps, processes = self.inventory(), self.apps(), self.processes()
        roots, unresolved = {}, {}
        for pid, process in processes.items():
            if not self.is_pressure(pid, process):
                continue
            env = read_environment(pid)
            tree = self.guard._descendants(processes, {pid})
            observed = {a['gpu_uuid'] for a in apps if a['pid'] in tree}
            try:
                resolved = binding(process['argv'], env, inventory, observed)
            except LeaseError as error:
                unresolved[pid] = {'identity': process, 'tree': sorted(tree),
                                   'observed_gpu_uuids': sorted(observed), 'reason': str(error)}
                continue
            if not observed.issubset(set(resolved['gpu_uuids'])):
                raise LeaseError('pressure_app_disagrees_with_exact_GPU_binding')
            roots[pid] = {'identity': process, 'gpu_uuids': resolved['gpu_uuids'],
                          'binding': resolved, 'tree': sorted(tree)}
        return {'inventory': inventory, 'apps': apps, 'processes': processes,
                'roots': roots, 'unresolved_roots': unresolved}

    def capture(self, process):
        pid = process['pid']
        if not same_process(process):
            raise LeaseError('process_changed_before_capture')
        status = {}
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            key, _, value = line.partition(':')
            status[key] = value.strip()
        uids, gids = [int(x) for x in status['Uid'].split()], [int(x) for x in status['Gid'].split()]
        if len(set(uids)) != 1 or len(set(gids)) != 1:
            raise LeaseError('unsupported_mixed_process_credentials')
        exe = os.readlink(f'/proc/{pid}/exe')
        limits = {}
        for name in dir(resource):
            if name.startswith('RLIMIT_') and name not in ('RLIMIT_OFILE',):
                limits[name] = list(resource.prlimit(pid, getattr(resource, name)))
        captured = dict(process, executable=exe, executable_sha256=digest(exe),
                        environment=read_environment(pid), gid=gids[0],
                        groups=[int(x) for x in status.get('Groups', '').split()],
                        rlimits=limits, affinity=sorted(os.sched_getaffinity(pid)),
                        nice=os.getpriority(os.PRIO_PROCESS, pid),
                        scheduler=os.sched_getscheduler(pid),
                        scheduler_priority=os.sched_getparam(pid).sched_priority,
                        cgroup=Path(f'/proc/{pid}/cgroup').read_text())
        if not same_process(process):
            raise LeaseError('process_changed_during_capture')
        return captured

    def spawn(self, captured, log_fd):
        def restore_limits():
            os.sched_setaffinity(0, captured['affinity'])
            os.setpriority(os.PRIO_PROCESS, 0, captured['nice'])
            os.sched_setscheduler(0, captured['scheduler'], os.sched_param(captured['scheduler_priority']))
            for name, pair in captured['rlimits'].items():
                resource.setrlimit(getattr(resource, name), tuple(pair))
            if os.geteuid() == 0:
                os.setgroups(captured['groups'])
                os.setgid(captured['gid'])
                os.setuid(captured['uid'])
        return subprocess.Popen(captured['argv'], executable=captured['executable'],
                                cwd=captured['cwd'], env=captured['environment'],
                                stdin=subprocess.DEVNULL, stdout=log_fd, stderr=log_fd,
                                start_new_session=True, preexec_fn=restore_limits)


class PressureLease:
    def __init__(self, state_root=DEFAULT_STATE_ROOT, backend=None, timeout=15):
        self.root = Path(state_root).absolute()
        self.backend = backend or LocalHost()
        if not 0 < timeout <= 60:
            raise LeaseError('timeout_must_be_in_0_to_60_seconds')
        self.timeout = timeout

    @contextmanager
    def locked(self):
        if any((parent / '.git').exists() for parent in [self.root, *self.root.parents]):
            raise LeaseError('private_journal_must_be_outside_git')
        if self.root.is_symlink() or self.root.resolve() != self.root:
            raise LeaseError('private_state_root_must_not_use_symlinks')
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = self.root.stat()
        if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise LeaseError('private_state_root_requires_owned_0700_directory')
        fd = self.open_private('host.lock', create=True)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    def open_private(self, name, create=False):
        if Path(name).name != name:
            raise LeaseError('invalid_private_file_reference')
        flags = os.O_RDWR | os.O_NOFOLLOW | (os.O_CREAT if create else 0)
        fd = os.open(self.root / name, flags, 0o600)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1:
            os.close(fd)
            raise LeaseError('private_file_requires_owned_single_link_0600')
        return fd

    def load(self, name):
        fd = self.open_private(name)
        with os.fdopen(fd) as stream:
            return json.load(stream)

    def save(self, name, value):
        # Directory is private and host.lock serializes every local lease.
        temporary = f'.{uuid.uuid4().hex}.tmp'
        fd = os.open(self.root / temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(fd, 'w') as stream:
                json.dump(value, stream, sort_keys=True, allow_nan=False)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(self.root / temporary, self.root / name)
            directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            (self.root / temporary).unlink(missing_ok=True)

    def name(self, lease_id):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', lease_id):
            raise LeaseError('invalid_lease_id')
        return f'lease-{lease_id}.json'

    def read_lease(self, lease_id):
        journal = self.load(self.name(lease_id))
        if journal['schema'] != SCHEMA or journal['context'] != self.backend.context():
            raise LeaseError('journal_host_boot_or_PID_namespace_mismatch')
        if journal['lease_id'] != lease_id:
            raise LeaseError('journal_lease_identity_mismatch')
        return journal

    def private(self, journal):
        name = journal['private_capture']
        payload = self.load(name)
        if digest(self.root / name) != journal['private_capture_sha256']:
            raise LeaseError('private_capture_hash_mismatch')
        return payload

    def write_lease(self, journal):
        journal['updated_unix'] = time.time()
        self.save(self.name(journal['lease_id']), journal)

    def worker_identities(self, workers):
        result = []
        for raw in workers:
            parts = str(raw).split(':')
            pid = int(parts[0])
            current = proc_identity(pid)
            if pid <= 1 or len(parts) > 2:
                raise LeaseError('invalid_worker_identity')
            if len(parts) == 1:
                if not current or current['state'] in ('Z', 'X'):
                    raise LeaseError('live_worker_required_for_PID_only_capture')
                result.append({k: current[k] for k in ('pid', 'start_ticks', 'uid')})
            else:
                ticks = int(parts[1])
                if ticks <= 0:
                    raise LeaseError('invalid_worker_start_ticks')
                result.append({'pid': pid, 'start_ticks': ticks,
                               'uid': current['uid'] if current and current['start_ticks'] == ticks else os.geteuid()})
        return result

    def add_workers(self, journal, workers):
        records = self.worker_identities(workers)
        existing = {(r['pid'], r['start_ticks']): r for r in journal['workers']}
        for record in records:
            existing[(record['pid'], record['start_ticks'])] = record
        journal['workers'] = list(existing.values())

    def public(self, journal):
        return dict(journal, workers=[dict(w, alive=same_process(w)) for w in journal['workers']],
                    roots=[dict(r, original_alive=same_process(r['original']),
                                replacement_alive=bool(r['replacement'] and same_process(r['replacement'])))
                           for r in journal['roots']])

    def inspect(self, gpu_uuids, workers=()):
        selected = set(gpu_uuids)
        snapshot = self.backend.snapshot()
        inventory = {r['uuid'] for r in snapshot['inventory']}
        if not selected or not selected.issubset(inventory):
            raise LeaseError('selected_GPU_UUID_not_in_local_inventory')
        roots = [dict(pid=pid, start_ticks=r['identity']['start_ticks'], uid=r['identity']['uid'],
                      gpu_uuids=r['gpu_uuids'], descendant_pids=r['tree'],
                      binding_proof=r['binding']['proof'],
                      additional_gpu_uuids_required=sorted(set(r['gpu_uuids']) - selected))
                 for pid, r in snapshot['roots'].items() if selected.intersection(r['gpu_uuids'])]
        pressure_pids = {pid for r in snapshot['roots'].values() for pid in r['tree']}
        return {'ok': True, 'action': 'inspect', 'context': self.backend.context(),
                'selected_gpu_uuids': sorted(selected), 'pressure_roots': roots,
                'unresolved_pressure_roots': [dict(pid=pid, start_ticks=r['identity']['start_ticks'],
                                                  observed_gpu_uuids=r['observed_gpu_uuids'], reason=r['reason'])
                                              for pid, r in snapshot['unresolved_roots'].items()],
                'genuine_or_unreadable_apps': [a for a in snapshot['apps'] if a['gpu_uuid'] in selected and a['pid'] not in pressure_pids],
                'worker_identities': self.worker_identities(workers)}

    def acquire(self, lease_id, gpu_uuids, workers=()):
        with self.locked():
            name = self.name(lease_id)
            if (self.root / name).exists():
                journal = self.read_lease(lease_id)
                if journal['selected_gpu_uuids'] != sorted(set(gpu_uuids)):
                    raise LeaseError('lease_GPU_selection_cannot_change')
                if journal['state'] == 'RESTORED':
                    raise LeaseError('completed_lease_id_cannot_be_reacquired')
                self.add_workers(journal, workers)
                self.write_lease(journal)
                if journal['state'] in ('ACQUIRED', 'ACQUIRED_NO_PRESSURE'):
                    return {'ok': True, 'action': 'acquire', 'changed': False, 'journal': self.public(journal),
                            'resource_recheck': self.inspect(journal['selected_gpu_uuids'])}
                if any(r['replacement'] or r['state'] in ('SPAWN_INTENT', 'VERIFYING', 'RESTORED') for r in journal['roots']):
                    raise LeaseError('restoration_started_lease_cannot_be_acquired')
                if any(r['state'] == 'SIGNAL_INTENT' for r in journal['roots']):
                    return self.defer(journal, 'AMBIGUOUS_SIGNAL_INTENT')
                captured = self.private(journal)
            else:
                inspection = self.inspect(gpu_uuids)
                selected = set(gpu_uuids)
                for other_path in self.root.glob('lease-*.json'):
                    other = self.load(other_path.name)
                    if other['state'] != 'RESTORED' and selected.intersection(other['selected_gpu_uuids']):
                        raise LeaseError('selected_GPU_has_an_outstanding_local_lease')
                snapshot = self.backend.snapshot()
                if snapshot['unresolved_roots']:
                    raise LeaseError('ambiguous_pressure_binding_requires_complete_actual_GPU_apps')
                roots = {pid: r for pid, r in snapshot['roots'].items() if selected.intersection(r['gpu_uuids'])}
                if any(set(r['gpu_uuids']) - selected for r in roots.values()):
                    raise LeaseError('selected_GPU_set_must_include_complete_pressure_parent_binding')
                captured = {'script': str(self.backend.script), 'roots': {}}
                if roots:
                    info = self.backend.script.stat()
                    if info.st_uid not in (0, os.geteuid()) or info.st_mode & 0o022:
                        raise LeaseError('trusted_script_has_untrusted_owner_or_write_permissions')
                    captured['script_sha256'] = digest(self.backend.script)
                for pid, root in roots.items():
                    tree = [self.backend.capture(snapshot['processes'][p]) for p in root['tree']]
                    captured['roots'][str(pid)] = {'root': next(p for p in tree if p['pid'] == pid),
                                                  'tree': tree, 'gpu_uuids': root['gpu_uuids'],
                                                  'binding': root['binding']}
                private_name = f'capture-{lease_id}-{uuid.uuid4().hex}.private.json'
                self.save(private_name, captured)
                journal = {'schema': SCHEMA, 'context': self.backend.context(), 'lease_id': lease_id,
                           'selected_gpu_uuids': sorted(selected), 'state': 'CAPTURED', 'workers': [],
                           'private_capture': private_name, 'private_capture_sha256': digest(self.root / private_name),
                           'roots': [{'original': {k: captured['roots'][str(pid)]['root'][k] for k in ('pid', 'start_ticks', 'uid')},
                                      'gpu_uuids': root['gpu_uuids'], 'state': 'CAPTURED', 'displaced': False,
                                      'replacement': None} for pid, root in roots.items()]}
                self.add_workers(journal, workers)
                self.write_lease(journal)  # Every private snapshot is durable before the first signal.
            if not journal['roots']:
                journal['state'] = 'ACQUIRED_NO_PRESSURE'
                self.write_lease(journal)
                return {'ok': True, 'action': 'acquire', 'changed': False, 'journal': self.public(journal)}
            fds = []
            try:
                for root in journal['roots']:
                    capture = captured['roots'][str(root['original']['pid'])]
                    for process in capture['tree']:
                        fd = open_pidfd(process)
                        if fd is not None:
                            fds.append((root, process, fd))
                # New descendants must also be captured before signalling their parent.
                fresh = self.backend.snapshot()
                if digest(self.backend.script) != captured['script_sha256']:
                    raise LeaseError('trusted_script_changed_before_stop')
                for root in journal['roots']:
                    pid = root['original']['pid']
                    if same_process(root['original']):
                        original = captured['roots'][str(pid)]['root']
                        if pid not in fresh['roots'] or not same_execution(original) or fresh['roots'][pid]['gpu_uuids'] != root['gpu_uuids']:
                            raise LeaseError('pressure_identity_or_execution_changed_before_stop')
                        known = {p['pid'] for p in captured['roots'][str(pid)]['tree']}
                        if set(fresh['roots'][pid]['tree']) - known:
                            raise LeaseError('pressure_tree_changed_before_stop_retry_capture_required')
                for root, process, fd in fds:
                    if not same_process(process):
                        continue
                    if not same_execution(process):
                        return self.defer(journal, 'CAPTURED_PROCESS_EXECUTION_CHANGED')
                    root['state'] = 'SIGNAL_INTENT'
                    self.write_lease(journal)
                    try:
                        signal.pidfd_send_signal(fd, signal.SIGTERM, None, 0)
                    except ProcessLookupError:
                        root['state'] = 'STOP_REQUESTED' if root['displaced'] else 'NOT_DISPLACED'
                    else:
                        root['displaced'] = True
                        root['state'] = 'STOP_REQUESTED'
                    self.write_lease(journal)
                deadline = time.monotonic() + self.timeout
                while any(same_process(p) for _, p, _ in fds) and time.monotonic() < deadline:
                    time.sleep(0.05)
                if any(same_process(p) for _, p, _ in fds):
                    return self.defer(journal, 'ORIGINAL_PROCESSES_STILL_RUNNING')
                after = self.backend.snapshot()
                if any(a['gpu_uuid'] in journal['selected_gpu_uuids'] and
                       a['pid'] in {p['pid'] for _, p, _ in fds} for a in after['apps']):
                    return self.defer(journal, 'ORIGINAL_GPU_ALLOCATION_NOT_RELEASED')
                for root in journal['roots']:
                    root['state'] = 'RELEASED' if root['displaced'] else 'NOT_DISPLACED'
                journal['state'] = 'ACQUIRED'
                self.write_lease(journal)
                return {'ok': True, 'action': 'acquire', 'changed': any(r['displaced'] for r in journal['roots']),
                        'journal': self.public(journal), 'resource_recheck': self.inspect(journal['selected_gpu_uuids'])}
            finally:
                for _, _, fd in fds:
                    os.close(fd)

    def defer(self, journal, reason):
        journal['state'] = 'DEFERRED'
        journal['deferred_reason'] = reason
        self.write_lease(journal)
        return {'ok': True, 'action': 'deferred', 'deferred': True, 'reason': reason,
                'journal': self.public(journal)}

    def candidate_matches(self, process, original):
        if process['argv'] != original['argv'] or process['cwd'] != original['cwd'] or process['uid'] != original['uid']:
            return False
        try:
            return (os.readlink(f"/proc/{process['pid']}/exe") == original['executable'] and
                    read_environment(process['pid']) == original['environment'])
        except OSError:
            return False

    def stop_invalid_replacement(self, journal, root, snapshot, reason):
        """Stop only OUR recorded replacement, after privately capturing its tree."""
        record = root['replacement']
        row = snapshot['roots'].get(record['pid']) or snapshot['unresolved_roots'].get(record['pid'])
        if not row or not same_process(record):
            return self.defer(journal, reason)
        tree = [self.backend.capture(snapshot['processes'][pid]) for pid in row['tree']]
        name = f'invalid-restoration-{journal["lease_id"]}-{record["pid"]}.private.json'
        self.save(name, {'tree': tree})
        root['invalid_restoration_capture'] = name
        root['state'] = 'STOPPING_INVALID_REPLACEMENT'
        self.write_lease(journal)
        fds = []
        try:
            for process in tree:
                fd = open_pidfd(process)
                if fd is not None:
                    fds.append((process, fd))
            for process, fd in fds:
                if same_execution(process):
                    try:
                        signal.pidfd_send_signal(fd, signal.SIGTERM, None, 0)
                    except ProcessLookupError:
                        pass
            deadline = time.monotonic() + self.timeout
            while any(same_process(p) for p, _ in fds) and time.monotonic() < deadline:
                time.sleep(.05)
            if any(same_process(p) for p, _ in fds):
                return self.defer(journal, 'INVALID_REPLACEMENT_STILL_RUNNING')
            root['state'] = 'INVALID_REPLACEMENT_STOPPED'
            return self.defer(journal, reason + '_REPLACEMENT_STOPPED')
        finally:
            for _, fd in fds:
                os.close(fd)

    def restore(self, lease_id, workers=(), no_workers_started=False, gpu_uuids=None):
        with self.locked():
            journal = self.read_lease(lease_id)
            if gpu_uuids is not None and sorted(set(gpu_uuids)) != journal['selected_gpu_uuids']:
                raise LeaseError('restore_GPU_selection_does_not_match_journal')
            self.add_workers(journal, workers)
            self.write_lease(journal)
            if journal['state'] == 'RESTORED':
                return {'ok': True, 'action': 'restore', 'changed': False, 'journal': self.public(journal)}
            if not any(r['displaced'] or r['state'] == 'SIGNAL_INTENT' for r in journal['roots']):
                journal['state'] = 'RESTORED'
                self.write_lease(journal)
                return {'ok': True, 'action': 'restore', 'changed': False, 'journal': self.public(journal)}
            if no_workers_started and journal['workers']:
                raise LeaseError('no_workers_started_conflicts_with_recorded_workers')
            if not journal['workers'] and not no_workers_started:
                return self.defer(journal, 'WORKER_EXIT_IDENTITIES_REQUIRED')
            if any(same_process(w) for w in journal['workers']):
                return self.defer(journal, 'EVALUATION_WORKERS_STILL_RUNNING')
            captured = self.private(journal)
            if any(r['state'] == 'SIGNAL_INTENT' for r in journal['roots']):
                return self.defer(journal, 'AMBIGUOUS_SIGNAL_INTENT')
            if any(same_process(p) for entry in captured['roots'].values() for p in entry['tree']):
                return self.defer(journal, 'ORIGINAL_PROCESSES_STILL_RUNNING')
            try:
                script_hash = digest(self.backend.script)
            except OSError:
                return self.defer(journal, 'TRUSTED_SCRIPT_UNAVAILABLE')
            if script_hash != captured['script_sha256']:
                return self.defer(journal, 'TRUSTED_SCRIPT_CHANGED')
            snapshot = self.backend.snapshot()
            if not set(journal['selected_gpu_uuids']).issubset({r['uuid'] for r in snapshot['inventory']}):
                return self.defer(journal, 'SELECTED_GPU_UUID_UNAVAILABLE')
            changed = False
            for root in journal['roots']:
                if not root['displaced']:
                    continue
                original = captured['roots'][str(root['original']['pid'])]['root']
                proof = captured['roots'][str(root['original']['pid'])]['binding']
                if proof['proof'] == 'actual_compute_apps':
                    if sorted(snapshot['inventory'], key=lambda row: row['index']) != proof['inventory']:
                        return self.defer(journal, 'GPU_INDEX_UUID_MAPPING_CHANGED')
                elif binding(original['argv'], original['environment'], snapshot['inventory'])['gpu_uuids'] != root['gpu_uuids']:
                    return self.defer(journal, 'GPU_INDEX_UUID_MAPPING_CHANGED')
                try:
                    executable_hash = digest(original['executable'])
                except OSError:
                    return self.defer(journal, 'ORIGINAL_EXECUTABLE_UNAVAILABLE')
                if executable_hash != original['executable_sha256']:
                    return self.defer(journal, 'ORIGINAL_EXECUTABLE_CHANGED')
                if Path('/proc/self/cgroup').read_text() != original['cgroup']:
                    return self.defer(journal, 'ORIGINAL_CGROUP_REQUIRED')
                if os.geteuid() != 0 and (original['uid'] != os.geteuid() or original['gid'] != os.getegid() or set(original['groups']) != set(os.getgroups())):
                    return self.defer(journal, 'ORIGINAL_CREDENTIALS_REQUIRED')
                if root['replacement'] is None and root['state'] == 'SPAWN_INTENT':
                    candidates = [r['identity'] for r in snapshot['roots'].values()
                                  if self.candidate_matches(r['identity'], original)]
                    if len(candidates) != 1:
                        return self.defer(journal, 'AMBIGUOUS_SPAWN_INTENT')
                    root['replacement'] = {k: candidates[0][k] for k in ('pid', 'start_ticks', 'uid')}
                    root['state'] = 'VERIFYING'
                    self.write_lease(journal)
                replacement = root['replacement']
                if replacement and not same_process(replacement):
                    return self.defer(journal, 'RESTORED_PROCESS_EXITED_NO_AUTOMATIC_RESPAWN')
                allowed = set()
                if replacement:
                    candidate = snapshot['roots'].get(replacement['pid']) or snapshot['unresolved_roots'].get(replacement['pid'])
                    if not candidate or not self.candidate_matches(candidate['identity'], original):
                        return self.defer(journal, 'REPLACEMENT_IDENTITY_NOT_VERIFIED')
                    allowed = set(candidate['tree'])
                occupants = [a for a in snapshot['apps'] if a['gpu_uuid'] in root['gpu_uuids'] and a['pid'] not in allowed]
                other_roots = [pid for pid, row in snapshot['roots'].items()
                               if set(row['gpu_uuids']).intersection(root['gpu_uuids']) and pid not in allowed]
                other_roots += [pid for pid in snapshot['unresolved_roots'] if pid not in allowed]
                if occupants or other_roots:
                    return self.defer(journal, 'GPU_ACQUIRED_BY_ANOTHER_JOB')
                if replacement is None:
                    # A second fresh check closes our own inspection/launch gap;
                    # noncooperating jobs can still race the NVIDIA driver.
                    snapshot = self.backend.snapshot()
                    if snapshot['unresolved_roots'] or any(a['gpu_uuid'] in root['gpu_uuids'] for a in snapshot['apps']) or any(set(r['gpu_uuids']).intersection(root['gpu_uuids']) for r in snapshot['roots'].values()):
                        return self.defer(journal, 'GPU_ACQUIRED_BY_ANOTHER_JOB')
                    root['state'] = 'SPAWN_INTENT'
                    self.write_lease(journal)
                    log_fd = self.open_private(f'restore-{lease_id}-{root["original"]["pid"]}.private.log', create=True)
                    try:
                        os.lseek(log_fd, 0, os.SEEK_END)
                        try:
                            child = self.backend.spawn(original, log_fd)
                        except (OSError, subprocess.SubprocessError):
                            root['state'] = 'RELEASED'
                            return self.defer(journal, 'ORIGINAL_LIMITS_OR_LAUNCH_FAILED')
                    finally:
                        os.close(log_fd)
                    identity = proc_identity(child.pid)
                    if not identity:
                        return self.defer(journal, 'SPAWN_IDENTITY_NOT_CAPTURED')
                    root['replacement'] = {k: identity[k] for k in ('pid', 'start_ticks', 'uid')}
                    root['state'] = 'VERIFYING'
                    self.write_lease(journal)
                    changed = True
                deadline = time.monotonic() + self.timeout
                while True:
                    snapshot = self.backend.snapshot()
                    replacement = root['replacement']
                    candidate = snapshot['roots'].get(replacement['pid'])
                    if not same_process(replacement):
                        return self.defer(journal, 'RESTORED_PROCESS_EXITED_NO_AUTOMATIC_RESPAWN')
                    if candidate and self.candidate_matches(candidate['identity'], original):
                        if candidate['gpu_uuids'] != root['gpu_uuids']:
                            return self.stop_invalid_replacement(journal, root, snapshot, 'RESTORED_GPU_BINDING_CHANGED')
                        tree = set(candidate['tree'])
                        apps = [a for a in snapshot['apps'] if a['gpu_uuid'] in root['gpu_uuids']]
                        if any(a['pid'] not in tree for a in apps):
                            return self.stop_invalid_replacement(journal, root, snapshot, 'GPU_ACQUIRED_BY_ANOTHER_JOB')
                        if {a['gpu_uuid'] for a in apps} == set(root['gpu_uuids']):
                            actual = self.backend.capture(candidate['identity'])
                            keys = ('rlimits', 'affinity', 'nice', 'scheduler', 'scheduler_priority', 'gid', 'groups', 'cgroup')
                            if any(actual[k] != original[k] for k in keys):
                                return self.defer(journal, 'RESTORED_LIMITS_NOT_VERIFIED')
                            root['state'] = 'RESTORED'
                            self.write_lease(journal)
                            break
                    if time.monotonic() >= deadline:
                        return self.defer(journal, 'RESTORED_GPU_BINDING_NOT_YET_VERIFIED')
                    time.sleep(0.05)
            journal['state'] = 'RESTORED'
            journal.pop('deferred_reason', None)
            self.write_lease(journal)
            return {'ok': True, 'action': 'restore', 'changed': changed, 'journal': self.public(journal)}

    def status(self, lease_id, gpu_uuids=None):
        with self.locked():
            journal = self.read_lease(lease_id)
            if gpu_uuids is not None and sorted(set(gpu_uuids)) != journal['selected_gpu_uuids']:
                raise LeaseError('status_GPU_selection_does_not_match_journal')
            return {'ok': True, 'action': 'status', 'journal': self.public(journal),
                    'resources': self.inspect(journal['selected_gpu_uuids'])}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=('inspect', 'acquire', 'restore', 'status'))
    parser.add_argument('--gpu-uuids', help='comma-separated exact physical UUIDs; required for inspect/acquire; restore/status default to the immutable journal selection')
    parser.add_argument('--lease-id')
    parser.add_argument('--state-root', type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument('--worker-pid', action='append', default=[], metavar='PID[:START_TICKS]')
    parser.add_argument('--no-workers-started', action='store_true', help='restore only: no evaluation worker was launched')
    parser.add_argument('--timeout', type=float, default=15)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
            raise LeaseError('Linux_pidfd_support_required_use_usr_bin_python3')
        if args.no_workers_started and args.action != 'restore':
            raise LeaseError('no_workers_started_is_restore_only')
        helper = PressureLease(args.state_root, timeout=args.timeout)
        selected = [x.strip() for x in (args.gpu_uuids or '').split(',') if x.strip()]
        if args.action in ('inspect', 'acquire') and not selected:
            raise LeaseError('gpu_uuids_required')
        if args.action != 'inspect' and not args.lease_id:
            raise LeaseError('lease_id_required')
        if args.action == 'inspect':
            payload = helper.inspect(selected, args.worker_pid)
        elif args.action == 'acquire':
            payload = helper.acquire(args.lease_id, selected, args.worker_pid)
        elif args.action == 'restore':
            payload = helper.restore(args.lease_id, args.worker_pid, args.no_workers_started,
                                     selected if args.gpu_uuids is not None else None)
        else:
            payload = helper.status(args.lease_id, selected if args.gpu_uuids is not None else None)
        print(json.dumps(payload, sort_keys=True, allow_nan=False))
        return 3 if payload.get('deferred') else 0
    except Exception as error:
        # Raw subprocess/OSError/argv/env text must never reach a scheduler log.
        message = str(error) if isinstance(error, LeaseError) else 'local_operation_failed_' + type(error).__name__
        print(json.dumps({'ok': False, 'action': args.action, 'error': message}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
