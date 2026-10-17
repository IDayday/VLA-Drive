#!/usr/bin/env python3
"""Lead-only, one-attempt future metadata PLAN. No build, GPU or retry path."""
# Start accounting before importing the monitored helpers or runtime dependencies.
with open('/proc/self/io') as _io:
    BOOT_IO = {k: int(v) for k, v in (line.split(':') for line in _io)}
import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import stat
import subprocess
import sys
import threading
import time

BOOT_TIME = time.monotonic()
BASE = Path('/mnt/project/st-world-policy-20261016')
PYTHON = '/root/miniconda3/envs/ddp/bin/python'
LIMITS = dict(attempts=1, automatic_retry=False, cleanup_seconds=30, core_dump_bytes=0,
    cpu_affinity=[0, 1], logical_read_bytes_including_imports=4 << 30,
    root_inclusive_bytes_including_control_temp_logs=32 << 20,
    rss_process_tree_bytes=2 << 30, single_file_bytes=4 << 20, wall_seconds=180)


def safe(path):
    p = Path(path)
    if not p.is_absolute() or '..' in p.parts or any(q.is_symlink() for q in (p, *p.parents)):
        raise ValueError('Absolute non-symlink path required')
    return p


def pinned(pin, *, json_value=True):
    p = safe(pin['path']); info = p.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > 4 << 20:
        raise ValueError('Bounded regular metadata/source input required')
    raw = p.read_bytes(); after = p.stat()
    if info != after or hashlib.sha256(raw).hexdigest() != pin['sha256']:
        raise ValueError('Pinned source/control changed')
    return json.loads(raw) if json_value else raw


def validate(admission, argv, now, *, child=False):
    scope = admission['scope']; role = scope.get('role')
    if (admission.get('schema') != 'st_full_future_plan_resource_admission_v1'
            or admission.get('status') != 'PASS_BOUNDED_' + str(role).upper() + '_PLAN_ONLY'
            or role not in ('dev', 'train') or admission.get('limits') != LIMITS
            or any(scope.get(k) is not False for k in ('build_authorized', 'gpu', 'image_read',
                'index_authorized', 'migration_authorized', 'model_load', 'raw_log_read',
                'teacher_tensor_read_or_write', 'training_authorized'))
            or type(scope.get('exact_scenes')) is not int or scope['exact_scenes'] < 1):
        raise ValueError('Exact CPU-only bounded plan admission required')
    for key in ('issued_at_unix', 'latest_start_unix', 'valid_until_unix'):
        if type(admission.get(key)) not in (int, float) or not math.isfinite(admission[key]):
            raise ValueError('Finite admission window required')
    if (now < admission['issued_at_unix'] or now > admission['latest_start_unix']
            or admission['valid_until_unix'] < now + (0 if child else 220)):
        raise ValueError('Expired/insufficient admission window; no automatic renewal')
    out = safe(admission['output_root'])
    if out != BASE / 'caches' / ('full_future_metadata_' + role + '_v1'):
        raise ValueError('Only fixed full-future metadata output roots allowed')
    if not isinstance(argv, list) or argv[:4] != [PYTHON, '-m', 'tools.st_world_policy.prepare_future_metadata', 'plan']:
        raise ValueError('Only the exact existing ddp-python plan entry is allowed')
    args = argv[4:]
    allowed = {'--student-root','--partition','--split','--current-population','--raw-log-root',
        '--sensor-root','--output','--scope','--max-scenes','--workers','--read-mib-per-second',
        '--max-seconds','--max-source-read-bytes','--max-metadata-bytes','--max-rss-bytes','--max-raw-log-bytes'}
    values = {}
    if len(args) % 2: raise ValueError('Exact flag/value plan argv required')
    for flag, value in zip(args[::2], args[1::2]):
        if flag not in allowed or not isinstance(value, str) or (flag in values and flag != '--sensor-root'):
            raise ValueError('Unsupported/duplicate plan option')
        values.setdefault(flag, []).append(value)
    if set(values) != allowed: raise ValueError('Missing explicit plan option')
    expected = {'--split':role, '--output':str(out), '--scope':'full_population',
        '--max-scenes':str(scope['exact_scenes']), '--workers':'1', '--read-mib-per-second':'64',
        '--max-seconds':'1800' if role == 'dev' else '14400',
        '--max-source-read-bytes':str((16 if role == 'dev' else 512) << 30),
        '--max-metadata-bytes':str(128 << 20 if role == 'dev' else 2 << 30), '--max-rss-bytes':str(4 << 30),
        '--max-raw-log-bytes':str((128 if role == 'dev' else 192) << 20)}
    if any(values[k] != [v] for k, v in expected.items()):
        raise ValueError('Plan population/proposed build limits changed')
    # Build limits remain in the proposal. They NEVER grant execution permission.
    if not set(LIMITS['cpu_affinity']).issubset(os.sched_getaffinity(0)):
        raise ValueError('Two admitted CPUs unavailable')
    return out


def root_bytes(root, *, reserve=0):
    root_info = Path(root).lstat()
    if not stat.S_ISDIR(root_info.st_mode): raise ValueError('Plain output root directory required')
    charged = lambda s: max(s.st_size, s.st_blocks * 512)
    total = charged(root_info); count = 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            p = Path(directory)/name; s = p.lstat(); count += 1
            if count > 64 or stat.S_ISLNK(s.st_mode): raise ValueError('Unsafe/expanded plan root')
            if stat.S_ISREG(s.st_mode):
                if s.st_nlink != 1 or charged(s) > LIMITS['single_file_bytes']:
                    raise ValueError('Plan file/alias limit')
            elif not stat.S_ISDIR(s.st_mode): raise ValueError('Nonregular plan output')
            total += charged(s)
    if total + reserve > LIMITS['root_inclusive_bytes_including_control_temp_logs']:
        raise ValueError('Root-inclusive plan output limit')
    return total


def load_controls(a, *, child=False):
    admission = pinned({'path':a.admission, 'sha256':a.admission_sha256})
    argv = pinned(admission['argv']); out = validate(admission, argv, time.time(), child=child)
    own = admission['wrapper']
    if safe(own['path']) != Path(__file__).absolute(): raise ValueError('Wrong wrapper identity')
    pinned(own, json_value=False)
    source = pinned(admission['source_pins'])
    if source['runtime'] != admission['runtime'] or source['source_commit'] != admission['source_commit']:
        raise ValueError('Runtime/source identity mismatch')
    for group in ('files', 'supervisor_helpers'):
        for pin in source[group].values(): pinned(pin, json_value=False)
    if set(source['supervisor_helpers']) != {'prepare_bindings.py','supervise_once.py','observed_cli.py'}:
        raise ValueError('Only original reviewed three monitor helpers')
    helper_dir = Path(source['supervisor_helpers']['supervise_once.py']['path']).parent
    if any(Path(p['path']).parent != helper_dir for p in source['supervisor_helpers'].values()):
        raise ValueError('Mixed monitor helper provenance')
    sys.path.insert(0, str(helper_dir))
    import prepare_bindings as pb
    import supervise_once as sup
    import observed_cli as obs
    # Only mutable COUNTERS is read under its original writer lock.
    def locked_read(path, expected=None, limit=1 << 20):
        if Path(path).name == 'COUNTERS.json':
            with pb.control_lock(Path(path).parent): return pb.read(path, expected, limit)
        return pb.read(path, expected, limit)
    sup.read = locked_read
    return admission, argv, out, pb, sup, obs


def child_run(a):
    admission, argv, out, pb, _, obs = load_controls(a, child=True)
    fd = int(os.environ['ST_PROBE_SUPERVISOR_LOCK_FD']); os.fstat(fd)
    if os.getpgrp() != os.getpid() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Owned CPU-only supervised session required')
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (4 << 20, 4 << 20))
    # Counters requires only the is_initialized query; no torch/model import needed.
    class NoCUDA:
        @staticmethod
        def is_initialized(): return False
    class CPUOnly: cuda = NoCUDA
    control = out/'control'
    counter = obs.Counters(control/'COUNTERS.json', CPUOnly, (4 << 30) - (8 << 20), BOOT_IO)
    counter.data['phase'] = 'IMPORTS'
    counter.snapshot(); thread = threading.Thread(target=counter.monitor, daemon=True); thread.start()
    try:
        sys.path.insert(0, admission['runtime'])
        from tools.st_world_policy import prepare_future_metadata as p
        from tools.st_world_policy import metadata_storage
        from starVLA.dataloader import st_world_metadata_index as idx
        original_check = metadata_storage.check_active_metadata_storage
        def guard(**kw):
            original_check(**kw); root_bytes(out, reserve=5 << 20)
            if obs.proc_io()['rchar'] - BOOT_IO['rchar'] > counter.read_cap:
                raise RuntimeError('Inclusive import/metadata read budget exhausted')
        metadata_storage.check_active_metadata_storage = guard
        idx.check_active_metadata_storage = guard
        originals = {p.file_sha256, idx.file_sha256}
        replacements = {f:counter.hash_wrapper(f) for f in originals}
        for module in tuple(sys.modules.values()):
            if module is not None and getattr(module, '__name__', '').startswith(('tools.','starVLA.')):
                for name, fn in list(vars(module).items()):
                    if any(fn is f for f in originals): setattr(module, name, replacements[fn])
        def audit(event, args):
            if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
                path = os.fsdecode(args[0])
                if Path(path).suffix.lower() in ('.pkl','.pt','.pth','.safetensors','.jpg','.jpeg','.png'):
                    raise RuntimeError('Plan-only forbids raw log/image/tensor reads')
        sys.addaudithook(audit)
        args = p.parser().parse_args(argv[3:])
        counter.data['phase'] = 'PLAN_ONLY'
        result = p.plan(args)  # exactly one real call, never p.main/build
        if result['status'] != 'PLAN_ONLY' or result['selected_scene_count'] != admission['scope']['exact_scenes']:
            raise ValueError('Unexpected plan result/population')
        root_bytes(out, reserve=1 << 20)
        pb.write_once(control/'CLI_RESULT.json', {'status':'PLAN_ONLY','identity':result['identity'],
            'selected_scene_count':result['selected_scene_count'],'plan_path':str(out/'PLAN.json'),
            'new_tensor_bytes':0,'build_executed':False,'training_authorized':False})
        return 0
    finally:
        counter.stop.set(); thread.join(2)
        if counter.snapshot()['limit_exceeded']: raise RuntimeError('Plan read counter limit')


def run(a):
    admission, argv, out, pb, sup, obs = load_controls(a)
    if out.exists(): raise ValueError('Single-attempt output already exists; no retry')
    if subprocess.check_output(['git','rev-parse','HEAD'], cwd=admission['runtime'], text=True).strip() != admission['source_commit']:
        raise ValueError('Immutable runtime commit mismatch')
    if subprocess.check_output(['git','status','--porcelain'], cwd=admission['runtime']).strip():
        raise ValueError('Immutable runtime dirty')
    pinned(admission['live_ledger'])
    commits = pinned(admission['checkpoint_commitments'])
    if (commits.get('schema') != 'st_checkpoint_storage_commitments_v1'
            or commits.get('status') != 'CURRENT' or commits.get('remaining_bytes') != 2183 << 30
            or commits.get('free_floor_bytes') != 1 << 40
            or type(commits.get('valid_until_unix')) not in (int,float)
            or not math.isfinite(commits['valid_until_unix'])
            or commits['valid_until_unix'] < time.time()+220):
        raise ValueError('Complete existing checkpoint obligations/floor required')
    required = (2183 << 30) + (1 << 40) + (32 << 20)
    v = os.statvfs(BASE)
    if v.f_bavail*v.f_frsize < required: raise ValueError('Insufficient fresh shared free space')
    mem = {line.split(':')[0]:int(line.split()[1]) << 10 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')}
    if mem['MemAvailable'] < 2 << 30: raise ValueError('Insufficient available host memory')
    sup.shared_mount()
    os.sched_setaffinity(0, [0,1])
    sys.path.insert(0, admission['runtime'])
    from tools.st_world_policy.metadata_storage import process_tree_rss
    with ExitStack() as stack:
        lock = sup.acquire(sup.LOCK, stack)
        validate(admission, argv, time.time()); pinned(admission['live_ledger'])
        out.mkdir(exist_ok=False); control = out/'control'; control.mkdir()
        pb.write_once(control/'ADMITTED.json', {'admission':{'path':a.admission,'sha256':a.admission_sha256},
            'argv':admission['argv'],'wrapper':admission['wrapper'],'attempts_consumed':1,'automatic_retry':False})
        try:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',
                ST_PROBE_SUPERVISOR_LOCK_FD=str(lock.fileno()), PYTHONPATH=admission['runtime'])
            def bounded_rss(pid):
                root_bytes(out, reserve=1 << 20)
                used = obs.proc_io()['rchar'] - BOOT_IO['rchar']
                c = control/'COUNTERS.json'
                if c.exists(): used += sup.read(c, limit=65536)[0]['proc_self_io_delta']['rchar']
                if used > 4 << 30: raise RuntimeError('Parent+child inclusive read limit')
                return process_tree_rss(os.getpid())
            resources = sup.supervise([PYTHON,str(Path(__file__).absolute()),'--child',
                '--admission',a.admission,'--admission-sha256',a.admission_sha256],
                cwd=admission['runtime'],env=env,cpus=[0,1],descriptors=[lock.fileno()],attempt=control,
                wall=max(0,180-(time.monotonic()-BOOT_TIME)),rss=2 << 30,rss_reader=bounded_rss)
            cli = pb.read(control/'CLI_RESULT.json')[0] if (control/'CLI_RESULT.json').exists() else None
            ok = resources['exit_code']==0 and resources['limit_reason'] is None and cli is not None
            final_counter = resources.get('counters')
            if not final_counter: raise ValueError('No final observed counters; cannot report success')
            total_read = obs.proc_io()['rchar']-BOOT_IO['rchar'] + final_counter['proc_self_io_delta']['rchar']
            ok = ok and total_read <= 4 << 30
            result = dict(status='PLAN_COMPLETE' if ok else 'STOPPED_NO_AUTO_RETRY',
                schema='st_full_future_plan_control_result_v1',resources=resources,cli_result=cli,
                parent_plus_child_rchar_bytes=total_read,admission=admission,automatic_retry=False,
                new_tensor_bytes=0,build_executed=False,scientific_result=False)
            root_bytes(out, reserve=256 << 10); pb.write_once(control/'RESULT.json',result)
            root_bytes(out); print(json.dumps({'status':result['status'],'result':str(control/'RESULT.json')}))
            return 0 if ok else 2
        except BaseException as exc:
            if not (control/'RESULT.json').exists():
                pb.write_once(control/'RESULT.json', {'status':'FAILED_NO_AUTO_RETRY',
                    'error_type':type(exc).__name__,'error':str(exc)[:2048],'automatic_retry':False})
            return 2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--admission',required=True);p.add_argument('--admission-sha256',required=True)
    p.add_argument('--child',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args();return child_run(args) if args.child else run(args)


if __name__=='__main__':raise SystemExit(main())
