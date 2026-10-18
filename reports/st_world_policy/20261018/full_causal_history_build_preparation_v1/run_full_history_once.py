#!/usr/bin/env python3
"""Fixed full train/dev history binding of the original owned CPU controller."""
import time
ENTRY = time.monotonic()
with open('/proc/self/io') as _entry_io:
    ENTRY_IO = {k: int(v) for k, v in (line.split(':') for line in _entry_io)}
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import stat
import sys
import types

from storage_binding import RUNTIME, RUNTIME_COMMIT, HISTORY32, HISTORY32_SHA, inherited_binding, source_bytes

HERE = Path(__file__).resolve().parent
BASE = Path('/mnt/project/st-world-policy-20261016')
PLAN_PATH = HERE / 'FULL_ROLE_PLAN.json'
PLAN_SHA = 'f6c96c1cb48d20b1143fc4fd38451e1ba4d6ae02ba71fe1ac7d8fae8ff1a2079'
ORIGINAL = RUNTIME / 'reports/st_world_policy/20261017/full_causal_history_preparation_v1/run_pilot_once.py'
GLOBAL_CONTROL = BASE / 'audits/history_data/full_causal_population_v1/train_controls'
GLOBAL_LOCK = GLOBAL_CONTROL / 'FULL_POPULATION.lock'
PYTHON = '/root/miniconda3/envs/ddp/bin/python'


def limits(role, cpus):
    if role not in ('train', 'dev'):
        raise ValueError('Only fixed full train/dev populations')
    return dict(wall_seconds=14400 if role == 'train' else 900, cleanup_seconds=40,
                read_bytes=(512 if role == 'train' else 24) << 30, rss_bytes=4 << 30,
                cpu_affinity=cpus, max_disk_bytes=(2048 if role == 'train' else 128) << 20,
                max_invocations=1)


def selection_check(pin, expected_size):
    """Hash the fixed large list inside the admitted budget; do not parse twice."""
    p = Path(pin['path'])
    if not p.is_absolute() or '..' in p.parts or any(x.is_symlink() for x in (p, *p.parents)):
        raise ValueError('Canonical population required')
    fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size or expected_size > 32 << 20:
            raise ValueError('Exact bounded population size required')
        digest = hashlib.sha256()
        remaining = expected_size + 1
        read_bytes = 0
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            while remaining:
                chunk = stream.read(min(remaining, 1 << 20))
                if not chunk:
                    break
                digest.update(chunk)
                remaining -= len(chunk)
                read_bytes += len(chunk)
        after = os.fstat(fd)
        version = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if (read_bytes != expected_size or version(before) != version(after) or
                version(after) != version(p.stat()) or digest.hexdigest() != pin['sha256']):
            raise ValueError('Population identity changed')
    finally:
        os.close(fd)
    return {'verified_population_sha256': pin['sha256']}


def validate(a, cp, role, spec, inherited, *, now, child=False):
    cpus = a.get('limits', {}).get('cpu_affinity')
    if (not isinstance(cpus, list) or len(cpus) != 2 or len(set(cpus)) != 2 or
            any(type(x) is not int or x < 0 for x in cpus)):
        raise ValueError('Exactly two distinct CPU IDs required')
    if a.get('limits') != limits(role, cpus):
        raise ValueError('Fixed full-role resource limits required')
    if (a.get('schema') != 'st_full_causal_history_owned_admission_v1' or
            a.get('status') != 'APPROVED_FULL_CAUSAL_HISTORY_METADATA_ONLY' or a.get('role') != role):
        raise ValueError('Independent full-role metadata admission required')
    if a.get('reviewer') in (None, '', '/root', '/root/st_indexprep_boundary_fix'):
        raise ValueError('Implementation author cannot admit a job')
    if (a.get('runtime') != str(RUNTIME) or a.get('source_sha') != RUNTIME_COMMIT or
            a.get('output_root') != spec['output_root'] or a.get('control_root') != spec['control_root'] or
            a.get('control_reserved_bytes') != 1 << 20 or a.get('hostname') != socket.gethostname() or
            a.get('w_source_compatibility') != inherited.W_COMPAT or a.get('registration') != spec['registration'] or
            a.get('builder_budget') != spec['builder_budget']):
        raise ValueError('Fixed source/role/registration/host binding required')
    for key in ('issued_at_unix', 'latest_start_unix', 'valid_until_unix'):
        if type(a.get(key)) not in (int, float) or not math.isfinite(a[key]):
            raise ValueError('Finite lease required')
    if (not a['issued_at_unix'] <= now <= a['latest_start_unix'] or
            a['valid_until_unix'] < now + (0 if child else limits(role, cpus)['wall_seconds'] + 40)):
        raise ValueError('Expired or insufficient full-role lease')
    if (cp.get('schema') != 'st_checkpoint_storage_commitments_v1' or cp.get('status') != 'CURRENT' or
            type(cp.get('remaining_bytes')) is not int or cp['remaining_bytes'] < 0 or
            cp.get('free_floor_bytes') != 1 << 40 or
            not cp['checked_at_unix'] <= now <= a['valid_until_unix'] <= cp['valid_until_unix']):
        raise ValueError('Fresh full-duration checkpoint obligations required')
    return cpus


def bound_source(raw):
    """Only exact constants and the child-wrapper path change; monitor is original."""
    source = raw.decode()
    changes = [
        ("ROOT=BASE/'audits/history_data/causal_history_cost64_v2'", 'ROOT=BOUND_ROOT', 1),
        ("CONTROL_ROOT=BASE/'audits/history_data/causal_history_cost64_v2_controls'", 'CONTROL_ROOT=BOUND_CONTROL', 1),
        ("CP={'path':str(BASE/'smoke/official_history_T1_native4_v1/controls/CHECKPOINT_COMMITMENTS_2303G_V1.json'),\n    'sha256':'13f3f4b282db2aff24528bc5434db51a9351cf43cf02dcdea1fbef38d5434cb3'}", 'CP=BOUND_CP', 1),
        ("LIMITS=dict(wall_seconds=900,cleanup_seconds=40,read_bytes=16<<30,rss_bytes=4<<30,\n            cpu_affinity=[6,7],max_disk_bytes=32<<20,max_invocations=1)", 'LIMITS=BOUND_LIMITS', 1),
        ('st_history64_owned_admission_v1', 'st_full_causal_history_owned_admission_v1', 1),
        ('APPROVED_HISTORY64_PILOT_ONLY', 'APPROVED_FULL_CAUSAL_HISTORY_METADATA_ONLY', 1),
        ('(0 if child else 940)', "(0 if child else LIMITS['wall_seconds']+40)", 1),
        ("required|={str(helper/n) for n in ('supervise_once.py','prepare_bindings.py','observed_cli.py')}", "required|={str(helper/n) for n in ('supervise_once.py','prepare_bindings.py','observed_cli.py')}\n    required |= BOUND_EXTRA_SOURCES", 1),
        ("selection_pin={'path':str(BASE/'audits/history_data/smoke_train64_v1/SELECTED_SCENES.json'),'sha256':'dee94e75957fd8e9734f1669b47552b5cf6c39f719aa9d3298bc87f94aac1c01'}", 'selection_pin=BOUND_SELECTION', 1),
        ("expected={'max_scenes':64,'split':'train'", "expected={'max_scenes':BOUND_SCENES,'split':BOUND_ROLE", 1),
        ("str(BASE/'audits/history_data/smoke_train64_v1/SELECTED_SCENES.json')", "BOUND_SELECTION['path']", 1),
        ("'student_root':'/mnt/project/ddp-foresight-artifacts/20260928/student_train_v1'", "'student_root':BOUND_STUDENT", 1),
        ('2303*(1<<30)', 'BOUND_CP_REMAINING', 2),
        ("'SUPERVISOR.lock','INVOCATION.json'", "'FULL_POPULATION.lock','SUPERVISOR.lock','INVOCATION.json'", 1),
        ("command=[PYTHON,str(Path(__file__).resolve()),'--child'", "command=[PYTHON,str(WRAPPER_PATH),'--role',BOUND_ROLE,'--child'", 1),
        ('cpu_affinity=(6,7)', 'cpu_affinity=tuple(BOUND_CPUS)', 1),
        ('os.sched_setaffinity(0,{6,7})', 'os.sched_setaffinity(0,set(BOUND_CPUS))', 1),
    ]
    for old, new, count in changes:
        if source.count(old) != count:
            raise ValueError('Reviewed original controller changed at an adaptation boundary')
        source = source.replace(old, new)
    return source


def controller(raw, a, cp, role, spec, inherited, cpus):
    module = types.ModuleType('full_role_bound_history_controller')
    module.__dict__.update(__file__=str(ORIGINAL), BOUND_ROOT=Path(spec['output_root']),
        BOUND_CONTROL=Path(spec['control_root']), BOUND_CP=a['checkpoint_receipt'],
        BOUND_CP_REMAINING=cp['remaining_bytes'], BOUND_LIMITS=limits(role, cpus), BOUND_CPUS=cpus,
        BOUND_ROLE=role, BOUND_SCENES=spec['scenes'], BOUND_SELECTION=spec['selection'],
        BOUND_STUDENT=spec['student_root'], WRAPPER_PATH=Path(__file__).resolve(),
        BOUND_EXTRA_SOURCES={str(Path(__file__).resolve()), str(HERE/'storage_binding.py'), str(HISTORY32)})
    exec(compile(bound_source(raw), str(ORIGINAL), 'exec'), module.__dict__)
    module.ENTRY_MONOTONIC, module.ENTRY_IO = ENTRY, ENTRY_IO
    def read(pin):
        if pin == spec['selection']:
            return selection_check(pin, spec['selection_bytes'])
        return inherited.pinned(pin)
    module.read = read
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=('train', 'dev'), required=True)
    parser.add_argument('--admission', required=True)
    parser.add_argument('--admission-sha256', required=True)
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or signal.getitimer(signal.ITIMER_REAL) != (0., 0.):
        raise ValueError('CPU-only execution with no preexisting alarm required')
    def expire(*_): raise TimeoutError('Full-history bootstrap deadline')
    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, max(.001, ENTRY + limits(args.role, [])['wall_seconds'] - time.monotonic()))
    inherited = inherited_binding()
    a = inherited.pinned({'path': args.admission, 'sha256': args.admission_sha256})
    if a.get('plan') != {'path': str(PLAN_PATH), 'sha256': PLAN_SHA}: raise ValueError('Fixed full-role plan required')
    plan = inherited.pinned(a['plan'])
    if a.get('policy') != plan['append_only_policy_proposal']:
        raise ValueError('Exact append-only role policy required')
    spec = plan['roles'][args.role]
    cp = inherited.pinned(a['checkpoint_receipt'])
    cpus = validate(a, cp, args.role, spec, inherited, now=time.time(), child=args.child)
    os.sched_setaffinity(0, set(cpus))
    for rel, sha in {**inherited.PINS, **inherited.BUILDER_PINS}.items():
        inherited.pinned({'path': str(RUNTIME/rel), 'sha256': sha}, decode=False)
    required_extra = {str(Path(__file__).resolve()), str(HERE/'storage_binding.py'), str(HISTORY32)}
    for name in required_extra:
        found = [p for p in a['controller_sources'] if p['path'] == name]
        if len(found) != 1: raise ValueError('Actual wrapper/helper dependencies required')
        inherited.pinned(found[0], decode=False)
    reg = inherited.pinned(spec['registration'])
    expected = spec['expected_registration']
    if reg != expected or reg['cpu_affinity'] != cpus:
        raise ValueError('Exact full-role registered recipe required')
    inherited.pinned(spec['builder_budget'])
    inherited.pinned(spec['population_complete'])
    # The original controls() hashes the entire population under this same budget.
    m = controller(source_bytes(ORIGINAL, inherited.PINS[str(ORIGINAL.relative_to(RUNTIME))]),
                   a, cp, args.role, spec, inherited, cpus)
    m.W_SOURCE_COMPATIBILITY = inherited.install_current_w_compatibility(a['w_source_compatibility'])
    sys.argv = [str(Path(__file__).resolve()), '--admission', args.admission,
                '--admission-sha256', args.admission_sha256] + (['--child'] if args.child else [])
    if args.child:
        signal.setitimer(signal.ITIMER_REAL, 0)
        return m.main()
    # Global train/dev nonblocking lock: preserve single-builder shared-I/O policy.
    # Charged inside the train control's existing 1 MiB, including a dev-only run.
    m.control_census(GLOBAL_CONTROL)
    fd = os.open(GLOBAL_LOCK, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode): raise ValueError('Regular global lock required')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        signal.setitimer(signal.ITIMER_REAL, 0)
        return m.main()
    finally:
        os.close(fd)


if __name__ == '__main__':
    raise SystemExit(main())
