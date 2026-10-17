#!/usr/bin/env python3
"""Candidate one-shot supervisor for five fixed SP0 probe stages, Lead only.

No arbitrary command, retries, resource allocation or approval generation.
This preparation supplies no APPROVED admission. RSS/wall and PyTorch peak are
observed limits with a bounded reaction interval, not OS cgroup hard ceilings.
"""
import argparse
from contextlib import ExitStack, contextmanager
import ctypes
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import signal
import socket
import subprocess
import sys
import time

from prepare_bindings import BASE, JOB, ROOTS, TAG, read, write_once, append_log, check_ancestry, control_lock

STAGES = ('dev_plan', 'dev_extract', 'train_plan', 'train_extract', 'fit')
PREVIOUS = dict(zip(STAGES[1:], STAGES[:-1]))
LIMITS = {'plan': (600, 64 << 30), 'extract': (960, 64 << 30), 'fit': (1260, 2 << 30)}
LOG_LIMIT = 4 << 20
ROLE_READ_LIMIT = 256 << 30
FIT_READ_LIMIT = 2 << 30
LOCK = BASE / 'resources/SPATIAL_PROBE_ONE_LOADER.lock'
LEDGER = str(BASE / 'resources/cache_storage_ledger_v1')
SHARED_SOURCE = '100.97.192.3@tcp:100.97.192.4@tcp:/hbTcp001/78ebbad2386e922193d44ac249bbb12e_kl4gv5vivlcf06o2[/mnt/project]'
PYTHON = '/root/miniconda3/envs/ddp/bin/python'
HELPERS = ('supervise_once.py', 'observed_cli.py', 'prepare_bindings.py')


def acquire(path, stack):
    path = Path(path)
    check_ancestry(path)
    if path.is_symlink():
        raise ValueError('Symlink lock forbidden')
    file = stack.enter_context(path.open('a+b'))
    fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return file


def validate(admission, preparation, stage, now):
    action = stage.rsplit('_', 1)[-1]
    wall, rss = LIMITS[action]
    if (admission.get('schema') != 'st_SP0_probe_stage_admission_v1' or admission.get('status') != 'APPROVED'
            or admission.get('reviewer') in (None, '', '/root/st_evaluation')
            or admission.get('stage') != stage or admission.get('hostname') != socket.gethostname()
            or type(admission.get('valid_until_unix')) not in (int, float)
            or not math.isfinite(admission['valid_until_unix'])
            or admission['valid_until_unix'] < now + wall + 30):
        raise ValueError('Missing exact independent stage/resource admission or insufficient lease window')
    if (preparation['checkpoint_tag'] != TAG or preparation['job_root'] != str(JOB)
            or preparation['fit_output'] != str(JOB / 'fit')
            or any(preparation['caches'][s]['root'] != str(ROOTS[s]) for s in ROOTS)
            or preparation['runtime'] != admission.get('runtime')):
        raise ValueError('Fixed SP0 paths/runtime differ')
    cpus = admission.get('cpu_affinity')
    if (not isinstance(cpus, list) or len(cpus) != 2 or len(set(cpus)) != 2
            or any(type(n) is not int or n < 0 for n in cpus)
            or not set(cpus).issubset(os.sched_getaffinity(0))):
        raise ValueError('Exactly two available leased CPUs required')
    if set(admission.get('helpers', {})) != set(HELPERS):
        raise ValueError('Exact reviewed helper hashes required')
    for name, expected in admission['helpers'].items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != expected:
            raise ValueError('Helper changed after independent review')
    if action == 'extract':
        gpu = admission.get('gpu')
        if (not isinstance(gpu, dict) or set(gpu) != {'index', 'uuid'}
                or type(gpu['index']) is not int or not 0 <= gpu['index'] < 8
                or not isinstance(gpu['uuid'], str) or not gpu['uuid'].startswith('GPU-')):
            raise ValueError('Exact leased single GPU UUID/index required')
        policy, _ = read(admission['storage_policy']['path'], admission['storage_policy']['sha256'])
        commitments, _ = read(admission['checkpoint_commitments']['path'], admission['checkpoint_commitments']['sha256'])
        if (commitments.get('schema') != 'st_checkpoint_storage_commitments_v1' or commitments.get('status') != 'CURRENT'
                or not commitments['checked_at_unix'] <= now
                or not math.isfinite(commitments['valid_until_unix'])
                or commitments['valid_until_unix'] < now + wall + 30):
            raise ValueError('Checkpoint reserve must cover outer wall plus termination grace')
        # The original StorageBudget validator checks package/claim/member bytes.
        # Require a separate completed migration receipt; this helper never migrates.
        migration, _ = read(admission['storage_migration']['path'], admission['storage_migration']['sha256'])
        if (admission.get('typed_storage_approved') is not True
                or policy.get('state_dir') != LEDGER or policy.get('cache_cap_bytes') != 768 << 30
                or not isinstance(policy.get('frozen_w_package'), dict)
                or not 0 < policy['frozen_w_package'].get('maximum_bytes', 0) <= 1 << 30
                or sum(sum(row[k] for k in ('output_bytes', 'staging_bytes', 'metadata_bytes'))
                       for row in policy['frozen_w_package'].get('members', {}).values()) >= 1 << 30
                or migration.get('schema') != 'st_metadata_storage_migration_v1'
                or migration.get('status') != 'MIGRATED'
                or migration.get('new_policy_sha256') != admission['storage_policy']['sha256']):
            raise ValueError('Actual typed-W migration and independent review required')
        split = stage.split('_')[0]
        actual_signature, signature_pin = read(JOB / 'stages' / (split + '_plan') / 'SIGNATURE.json')
        claims = [c for c in policy.get('claims', []) if c.get('root') == preparation['caches'][split]['root']]
        if (len(claims) != 1 or claims[0].get('identity_kind') != 'st_frozen_spatial_w_v1'
                or claims[0].get('cache_identity') != preparation['caches'][split]['identity']
                or actual_signature.get('identity') != preparation['caches'][split]['identity']
                or claims[0].get('signature_sha256') != signature_pin['sha256']
                or not (Path(LEDGER) / 'CLAIMS.json').is_file()):
            raise ValueError('Exact actual-plan claim in the existing ledger required before native reads')
    elif admission.get('gpu') is not None or admission.get('storage_policy') is not None:
        raise ValueError('CPU plan/fit cannot consume a GPU or mutate storage claims')
    return wall, rss


def shared_mount():
    value = json.loads(subprocess.check_output(['findmnt', '--json', '--target', str(BASE),
                       '--output', 'TARGET,FSTYPE,OPTIONS,SOURCE'], text=True, timeout=10))['filesystems']
    if len(value) != 1:
        raise ValueError('One shared campaign mount required')
    entry = value[0]; options = set(entry['options'].split(','))
    if (entry['target'] != '/mnt/project' or entry['source'] != SHARED_SOURCE
            or entry['fstype'] != 'lustre' or 'flock' not in options or options & {'localflock', 'noflock'}):
        raise ValueError('One-loader lock requires the recorded shared global flock mount')


def fresh_gpu(gpu):
    row = subprocess.check_output(['nvidia-smi', '-i', str(gpu['index']),
        '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True, timeout=10).strip().split(',')
    if len(row) != 3 or row[0].strip() != gpu['uuid'] or int(row[1]) > 64 or int(row[2]) != 0:
        raise ValueError('Leased GPU is not freshly idle with the exact UUID')
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                   '--format=csv,noheader,nounits'], text=True, timeout=10)
    if any(line.split(',')[0].strip() == gpu['uuid'] for line in apps.splitlines()):
        raise ValueError('Leased GPU has a compute owner')


def proc_identity(pid):
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
    return dict(pid=pid, state=fields[0], ppid=int(fields[1]), pgid=int(fields[2]),
                sid=int(fields[3]), start_ticks=int(fields[19]))


def pidfd_open(pid):
    if os.uname().machine not in ('x86_64', 'aarch64'):
        raise RuntimeError('Reviewed Linux pidfd ABI required')
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.syscall(434, int(pid), 0)
    if fd < 0:
        error = ctypes.get_errno(); raise OSError(error, os.strerror(error))
    os.set_inheritable(fd, False)
    return fd


def pidfd_signal(fd, sig):
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.syscall(424, int(fd), int(sig), None, 0) < 0:
        error = ctypes.get_errno()
        if error != errno.ESRCH:
            raise OSError(error, os.strerror(error))


@contextmanager
def lifetime_scope():
    """This supervisor owns orphaned descendants only for its invocation."""
    libc = ctypes.CDLL(None, use_errno=True); previous = ctypes.c_int()
    fd = pidfd_open(os.getpid()); os.close(fd)  # fail before Popen if pidfds unavailable
    if libc.prctl(37, ctypes.byref(previous), 0, 0, 0) or libc.prctl(36, 1, 0, 0, 0):
        raise RuntimeError('Linux child subreaper is required before spawning')
    stopped = {'signal': None}
    old = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    def request_stop(sig, frame):
        stopped['signal'] = stopped['signal'] or sig  # do not interrupt cleanup/writes
    try:
        for sig in old: signal.signal(sig, request_stop)
        yield stopped
    finally:
        for sig, handler in old.items(): signal.signal(sig, handler)
        libc.prctl(36, previous.value, 0, 0, 0)


class OwnedSession:
    """Hold the spawn's unreaped leader as SID anchor; signal exact pidfds only.

    No Popen.poll()/wait() until closure: WNOWAIT leaves the dead leader's PID
    reserved, so a later foreign session cannot reuse its numeric identity.
    Members are checked before and after pidfd_open. Historical receipts are
    never an input. Subreaper adoption lets us reap our orphaned zombies too.
    """
    def __init__(self, child):
        self.child = child; self.anchor = proc_identity(child.pid); self.members = {}
        self.verified = 0
        if self.anchor['sid'] != child.pid or self.anchor['pgid'] != child.pid:
            raise RuntimeError('Refuse a foreign process group')
        self._remember(self.anchor)
        child._probe_owned_session = self

    @staticmethod
    def same(a, b):
        return all(a[k] == b[k] for k in ('pid', 'sid', 'start_ticks'))

    def _remember(self, record):
        old = self.members.get(record['pid'])
        if old is not None:
            if self.same(record, old[0]): return
            os.close(old[1]); del self.members[record['pid']]
        fd = pidfd_open(record['pid'])
        try:
            if not self.same(record, proc_identity(record['pid'])):
                raise RuntimeError('Process identity changed while opening pidfd')
        except BaseException:
            os.close(fd); raise
        self.members[record['pid']] = (record, fd); self.verified += 1

    def discover(self):
        if not self.same(self.anchor, proc_identity(self.child.pid)):
            raise RuntimeError('Spawn anchor changed; no numeric-session signalling allowed')
        entries = list(Path('/proc').iterdir())
        if len(entries) > 32768:
            raise RuntimeError('Bounded process metadata scan exceeded')
        for entry in entries:
            if not entry.name.isdigit(): continue
            try:
                record = proc_identity(int(entry.name))
                if record['sid'] == self.child.pid:
                    if record['start_ticks'] < self.anchor['start_ticks']:
                        raise RuntimeError('Session member predates this spawn')
                    self._remember(record)
            except (FileNotFoundError, ProcessLookupError):
                continue
        active = []
        for pid, (known, fd) in list(self.members.items()):
            try:
                current = proc_identity(pid)
                if self.same(known, current) and current['state'] not in ('Z', 'X'):
                    active.append((known, fd)); continue
                if pid != self.child.pid and self.same(known, current) and current['ppid'] == os.getpid():
                    os.waitpid(pid, os.WNOHANG)  # only our verified adopted child
            except (FileNotFoundError, ProcessLookupError, ChildProcessError):
                pass
            if pid != self.child.pid:
                os.close(fd); del self.members[pid]
        return active

    def leader_done(self):
        return os.waitid(os.P_PID, self.child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None

    def send(self, sig):
        try:
            active = self.discover()
        except BaseException:
            # Even if /proc discovery fails, already opened pidfds still name
            # only our verified tasks. Stop those; do not guess new numeric PIDs.
            for _, fd in self.members.values(): pidfd_signal(fd, sig)
            raise
        for _, fd in active:
            pidfd_signal(fd, sig)

    def finish(self, *, terminate, deadline=None):
        deadline = time.monotonic() + 40 if deadline is None else deadline
        if terminate: self.send(signal.SIGTERM)
        while self.discover():
            now = time.monotonic()
            if now >= deadline - 10:
                self.send(signal.SIGKILL)
            elif terminate:
                self.send(signal.SIGTERM)  # include forks discovered during cleanup
            if now >= deadline:
                raise RuntimeError('Owned session cleanup incomplete after TERM/KILL grace')
            time.sleep(.05)
        code = self.child.wait(timeout=1)  # first reap of the SID anchor
        return code

    def close(self):
        for _, fd in self.members.values(): os.close(fd)
        self.members.clear()


def abort_spawn_setup(child, leader_fd, deadline=None):
    """Close only this fresh, unreaped start_new_session Popen on setup failure.

    waitid proves it is still our unreaped child. Its kernel SID/PGID is checked
    before the fallback group signal; the unreaped leader prevents ID reuse.
    This authority comes from the current Popen call, never a saved receipt.
    """
    if child.returncode is not None:
        return
    deadline = time.monotonic() + 10 if deadline is None else min(deadline, time.monotonic() + 10)
    os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    if os.getsid(child.pid) != child.pid or os.getpgid(child.pid) != child.pid:
        # A pidfd still identifies our actual child, but never authorizes an
        # unrelated group. Do not reap until after the signal.
        if leader_fd is not None: pidfd_signal(leader_fd, signal.SIGKILL)
        else: child.kill()  # direct unreaped Popen child, not an old PID
        child.wait(timeout=max(.1, deadline - time.monotonic()))
        raise RuntimeError('Fresh spawn did not retain its own session')
    try:
        # Failures before ownership registration have not admitted useful work.
        # KILL here avoids granting a failed setup another 30s to fork/work.
        os.killpg(child.pid, signal.SIGKILL)
        if leader_fd is not None: pidfd_signal(leader_fd, signal.SIGKILL)
        while True:
            entries = list(Path('/proc').iterdir())
            if len(entries) > 32768:
                raise RuntimeError('Bounded setup cleanup scan exceeded')
            active = False
            for entry in entries:
                if not entry.name.isdigit(): continue
                try:
                    record = proc_identity(int(entry.name))
                    if record['sid'] != child.pid: continue
                    # SID cannot be reused while the actual leader is unreaped.
                    fd = pidfd_open(record['pid'])
                    try:
                        if not OwnedSession.same(record, proc_identity(record['pid'])):
                            continue
                        if record['state'] not in ('Z', 'X'):
                            active = True; pidfd_signal(fd, signal.SIGKILL)
                        elif record['pid'] != child.pid and record['ppid'] == os.getpid():
                            os.waitpid(record['pid'], os.WNOHANG)
                    finally:
                        os.close(fd)
                except (FileNotFoundError, ProcessLookupError, ChildProcessError):
                    continue
            if not active: break
            if time.monotonic() >= deadline:
                raise RuntimeError('Fresh spawn setup cleanup exceeded 10 seconds')
            time.sleep(.01)
    finally:
        # Even /proc or selector registration failure cannot abandon the fresh
        # child/group. Keep the spawn anchor reserved through the final signal.
        os.killpg(child.pid, signal.SIGKILL)
        if leader_fd is not None: pidfd_signal(leader_fd, signal.SIGKILL)
        child.wait(timeout=max(.1, deadline - time.monotonic()))


def own_group_signal(child, sig):
    owner = getattr(child, '_probe_owned_session', None)
    if owner is None or owner.child is not child:
        raise RuntimeError('Refuse to signal a foreign process group')
    owner.send(sig)


def read_counter_snapshot(path, on_error=None):
    """Serialize this mutable snapshot with its writer; immutable read stays strict.

    The existing control lock can add up to two seconds of waiting per sample,
    plus filesystem I/O/scheduling. Timeout fails closed, without a retry.
    """
    path = Path(path); operation = 'counter_lock'
    try:
        with control_lock(path.parent):
            operation = 'counter_exists'
            if not path.exists():
                return None, None
            operation = 'counter_read'
            value, _ = read(path, limit=65536)
            operation = 'counter_freshness'
            modified = path.stat().st_mtime
            return value, modified
    except Exception as exc:
        if on_error is not None:
            on_error(operation, exc)
        raise


def supervise(argv, *, cwd, env, cpus, descriptors, attempt, wall, rss, rss_reader, popen=subprocess.Popen):
    """Only small fake children are used by tests; real argv is fixed by main."""
    with lifetime_scope() as stopped:
        return _supervise(argv, cwd=cwd, env=env, cpus=cpus, descriptors=descriptors,
                          attempt=attempt, wall=wall, rss=rss, rss_reader=rss_reader,
                          popen=popen, stopped=stopped)


def _supervise(argv, *, cwd, env, cpus, descriptors, attempt, wall, rss, rss_reader, popen, stopped):
    started = time.monotonic(); maximum_rss = 0; emitted = 0; reason = None; term_at = None
    first_monitor_error = None
    def remember_monitor_error(operation, exc):
        nonlocal first_monitor_error
        if first_monitor_error is None:
            first_monitor_error = dict(operation=operation[:64], exception_class=type(exc).__name__[:128],
                message=str(exc)[:1024], time_unix=time.time(), elapsed_seconds=time.monotonic() - started)
    log = Path(attempt) / 'CHILD.log'
    child = popen(['taskset', '-c', ','.join(map(str, cpus)), *argv], cwd=cwd, env=env,
                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True,
                  pass_fds=tuple(descriptors))
    owner = None; selector = None; leader_fd = None
    next_sample = 0.; final_counters = None
    try:
        leader_fd = pidfd_open(child.pid)
        owner = OwnedSession(child)
        selector = selectors.DefaultSelector()
        selector.register(child.stdout, selectors.EVENT_READ)
        write_once(Path(attempt) / 'CHILD.json', dict(pid=child.pid, pgid=child.pid,
                   sid=owner.anchor['sid'], start_ticks=owner.anchor['start_ticks'],
                   started_unix=time.time(), argv=argv, owned_new_session=True,
                   ownership='unreaped spawn anchor + verified live pidfds; never reconstructed from receipt'))
        with log.open('xb', buffering=0) as stream:
            while True:
                active = owner.discover(); done = owner.leader_done()
                if stopped['signal'] is not None:
                    reason = reason or 'SUPERVISOR_SIGNAL_' + str(stopped['signal'])
                if done and active:
                    reason = reason or 'OWNED_DESCENDANTS_AFTER_LEADER_EXIT'
                if not active and not selector.get_map(): break
                for key, _ in selector.select(.2):
                    block = os.read(key.fileobj.fileno(), 65536)
                    if not block:
                        selector.unregister(key.fileobj); continue
                    allowed = max(0, LOG_LIMIT - emitted)
                    written = append_log(log, stream, block[:allowed]); emitted += len(block)
                    if written < min(len(block), allowed):
                        reason = reason or 'CONTROL_DIRECTORY_LIMIT'
                    if emitted > LOG_LIMIT:
                        reason = reason or 'LOG_LIMIT'
                now = time.monotonic()
                if now >= next_sample:
                    next_sample = now + 1
                    operation = 'rss'
                    try:
                        if not done:
                            maximum_rss = max(maximum_rss, rss_reader(child.pid))
                        counter = Path(attempt) / 'COUNTERS.json'
                        operation = 'counter_snapshot'
                        snapshot, modified = read_counter_snapshot(counter, remember_monitor_error)
                        if snapshot is not None:
                            final_counters = snapshot
                            operation = 'counter_status'
                            if final_counters.get('limit_exceeded'):
                                reason = reason or final_counters['limit_exceeded']
                            if not done and time.time() - modified > 15:
                                reason = reason or 'STALE_OBSERVER'
                    except Exception as exc:
                        remember_monitor_error(operation, exc)
                        reason = reason or 'MONITOR_UNREADABLE'
                    now = time.monotonic()  # include the bounded lock wait in wall enforcement
                    if maximum_rss > rss:
                        reason = reason or 'RSS_LIMIT'
                    if now - started >= wall:
                        reason = reason or 'WALL_LIMIT'
                if reason and term_at is None:
                    own_group_signal(child, signal.SIGTERM); term_at = now
                if term_at is not None and now - term_at >= 30:
                    own_group_signal(child, signal.SIGKILL)
                if term_at is not None and now - term_at >= 40:
                    raise RuntimeError('Owned session or pipe did not close within cleanup grace')
            stream.flush(); os.fsync(stream.fileno())
        code = owner.finish(terminate=reason is not None, deadline=term_at + 40 if term_at is not None else None)
    except BaseException:
        partial = getattr(child, '_probe_owned_session', None)
        if owner is None and partial is not None and partial.child is child:
            owner = partial
        cleanup_deadline = term_at + 40 if term_at is not None else None
        try:
            if owner is not None:
                owner.finish(terminate=True, deadline=term_at + 40 if term_at is not None else None)
            else:
                abort_spawn_setup(child, leader_fd, deadline=cleanup_deadline)
        except BaseException:
            # Discovery itself can fail; the newly spawned unreaped child still
            # has a bounded cleanup owner. This never consults old metadata.
            abort_spawn_setup(child, leader_fd, deadline=cleanup_deadline)
            raise
        raise
    finally:
        if owner is not None: owner.close()
        if selector is not None: selector.close()
        if leader_fd is not None: os.close(leader_fd)
        child.stdout.close()
    counter = Path(attempt) / 'COUNTERS.json'
    try:
        snapshot, _ = read_counter_snapshot(counter, remember_monitor_error)
        if snapshot is not None:
            final_counters = snapshot
            reason = reason or final_counters.get('limit_exceeded')
    except Exception as exc:
        remember_monitor_error('final_counter_snapshot', exc)
        reason = reason or 'MONITOR_UNREADABLE'
    elapsed = time.monotonic() - started
    if elapsed >= wall:
        reason = reason or 'WALL_LIMIT'
    return dict(exit_code=code, limit_reason=reason, wall_seconds=elapsed,
                sampled_tree_rss_peak_bytes=maximum_rss, log_bytes_stored=log.stat().st_size,
                log_bytes_received=emitted, counters=final_counters, first_monitor_error=first_monitor_error,
                process_cleanup=dict(verified_processes=owner.verified, live_owned_remaining=0,
                                     method='unreaped session anchor, start_ticks and pidfd; adopted owned zombies reaped'),
                monitoring='RSS/wall use a 1s sampling schedule; each counter snapshot may add <=2s control-lock waiting plus I/O/scheduling; thresholds unchanged; torch allocator peak sampled at 0.5s; cooperative kernel scheduling; 30s own-group TERM grace')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--admission', required=True); parser.add_argument('--admission-sha256', required=True)
    a = parser.parse_args()
    admission, admission_pin = read(a.admission, a.admission_sha256)
    preparation, _ = read(admission['preparation']['path'], admission['preparation']['sha256'])
    stage = admission.get('stage')
    if stage not in STAGES:
        raise ValueError('Only the five fixed SP0 stages exist')
    wall, rss = validate(admission, preparation, stage, time.time())
    os.sched_setaffinity(0, admission['cpu_affinity'])  # supervisor and child share the same two CPUs
    runtime = preparation['runtime']['path']
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=runtime, text=True).strip() != preparation['runtime']['source_sha']
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=runtime).strip()):
        raise ValueError('Frozen runtime changed')
    # Use existing resource observer, not a second process traversal implementation.
    sys.path.insert(0, runtime)
    from tools.st_world_policy.metadata_storage import process_tree_rss
    shared_mount()
    stage_root = JOB / 'stages'; check_ancestry(stage_root); stage_root.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        lock = acquire(LOCK, stack)  # one plan/extract/fit at a time across campaign hosts
        locks = [lock]
        if stage in PREVIOUS:
            previous, _ = read(stage_root / PREVIOUS[stage] / 'RESULT.json')
            if previous['status'] != 'COMPLETE' or previous['preparation'] != admission['preparation']:
                raise ValueError('Previous fixed stage incomplete/different identity')
        if stage.endswith('_extract'):
            locks.append(acquire(f"/tmp/foresight_gpu_{admission['gpu']['index']}.lock", stack))
            fresh_gpu(admission['gpu'])
        mem = {line.split(':')[0]: int(line.split()[1]) << 10 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')}
        if mem['MemAvailable'] < rss:
            raise ValueError('Insufficient current host memory for fixed admission')
        validate(admission, preparation, stage, time.time())  # recheck lease/space window at admission
        # Reserve exactly one durable attempt before Popen. Failed starts remain spent.
        attempt = stage_root / stage
        attempt.mkdir(exist_ok=False)
        try:
            write_once(attempt / 'ADMITTED.json', dict(admission=admission_pin, stage=stage,
                       preparation=admission['preparation'], consumed=True, automatic_retry=False))
            read_cap = FIT_READ_LIMIT if stage == 'fit' else ROLE_READ_LIMIT
            if stage.endswith('_extract'):
                planned, _ = read(stage_root / (stage.split('_')[0] + '_plan') / 'RESULT.json')
                counters = planned['resources']['counters']
                used = max(counters['hash_logical_bytes_admitted'] + counters['torch_load_nominal_bytes'],
                           counters['proc_self_io_delta']['rchar'])
                read_cap -= used
                if read_cap <= 0:
                    raise ValueError('Role read budget exhausted')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(admission['gpu']['index']) if admission.get('gpu') else '',
                       PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                       OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                       ST_PROBE_SUPERVISOR_LOCK_FD=str(lock.fileno()))
            env['PYTHONPATH'] = runtime
            argv = [PYTHON, str(Path(__file__).with_name('observed_cli.py')), '--admission', a.admission,
                    '--admission-sha256', a.admission_sha256, '--attempt', str(attempt), '--read-cap', str(read_cap)]
            resource = supervise(argv, cwd=runtime, env=env, cpus=admission['cpu_affinity'],
                                 descriptors=[f.fileno() for f in locks], attempt=attempt, wall=wall,
                                 rss=rss, rss_reader=process_tree_rss)
            cli = read(attempt / 'CLI_RESULT.json')[0] if (attempt / 'CLI_RESULT.json').exists() else None
            ok = (resource['exit_code'] == 0 and resource['limit_reason'] is None and resource['counters'] is not None
                  and cli is not None and cli.get('status') == ('PLANNED_ONLY' if stage.endswith('_plan') else 'COMPLETE'))
            result = dict(schema='st_SP0_probe_stage_result_v1', status='COMPLETE' if ok else 'STOPPED_NO_AUTO_RETRY',
                          stage=stage, preparation=admission['preparation'], admission=admission_pin,
                          resources=resource, cli_result=cli, scientific_result=False)
            write_once(attempt / 'RESULT.json', result)
            print(json.dumps({'status': result['status'], 'stage': stage, 'result': str(attempt / 'RESULT.json')}))
            return 0 if ok else 2
        except BaseException as exc:
            # Preserve every earlier file and the consumed marker. The 256KiB
            # final reserve is excluded from all other control/log admissions.
            if not (attempt / 'RESULT.json').exists():
                write_once(attempt / 'RESULT.json', dict(
                    schema='st_SP0_probe_stage_result_v1', status='FAILED_NO_AUTO_RETRY',
                    stage=stage, preparation=admission['preparation'], admission=admission_pin,
                    error_type=type(exc).__name__, error=str(exc)[:2048],
                    retained_diagnostics=True, scientific_result=False))
            return 2



if __name__ == '__main__':
    raise SystemExit(main())
