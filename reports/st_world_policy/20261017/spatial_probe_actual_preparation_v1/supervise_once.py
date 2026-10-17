#!/usr/bin/env python3
"""Candidate one-shot supervisor for five fixed SP0 probe stages, Lead only.

No arbitrary command, retries, resource allocation or approval generation.
This preparation supplies no APPROVED admission. RSS/wall and PyTorch peak are
observed limits with a bounded reaction interval, not OS cgroup hard ceilings.
"""
import argparse
from contextlib import ExitStack
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

from prepare_bindings import BASE, JOB, ROOTS, TAG, read, write_once, append_log

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


def own_group_signal(child, sig):
    # Popen created this exact new session, and no existing PID/group is accepted.
    if child.poll() is None:
        if os.getsid(child.pid) != child.pid or os.getpgid(child.pid) != child.pid:
            raise RuntimeError('Refuse to signal a foreign process group')
        os.killpg(child.pid, sig)


def supervise(argv, *, cwd, env, cpus, descriptors, attempt, wall, rss, rss_reader, popen=subprocess.Popen):
    """Only small fake children are used by tests; real argv is fixed by main."""
    started = time.monotonic(); maximum_rss = 0; emitted = 0; reason = None; term_at = None
    log = Path(attempt) / 'CHILD.log'
    child = popen(['taskset', '-c', ','.join(map(str, cpus)), *argv], cwd=cwd, env=env,
                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True,
                  pass_fds=tuple(descriptors))
    selector = selectors.DefaultSelector(); selector.register(child.stdout, selectors.EVENT_READ)
    next_sample = 0.; final_counters = None
    try:
        write_once(Path(attempt) / 'CHILD.json', dict(pid=child.pid, pgid=child.pid,
                   started_unix=time.time(), argv=argv, owned_new_session=True))
        with log.open('xb', buffering=0) as stream:
            while child.poll() is None or selector.get_map():
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
                    try:
                        if child.poll() is None:
                            maximum_rss = max(maximum_rss, rss_reader(child.pid))
                        counter = Path(attempt) / 'COUNTERS.json'
                        if counter.exists():
                            final_counters, _ = read(counter, limit=65536)
                            if final_counters.get('limit_exceeded'):
                                reason = reason or final_counters['limit_exceeded']
                            if child.poll() is None and time.time() - counter.stat().st_mtime > 15:
                                reason = reason or 'STALE_OBSERVER'
                    except Exception:
                        reason = reason or 'MONITOR_UNREADABLE'
                    if maximum_rss > rss:
                        reason = reason or 'RSS_LIMIT'
                    if now - started >= wall:
                        reason = reason or 'WALL_LIMIT'
                if reason and term_at is None:
                    own_group_signal(child, signal.SIGTERM); term_at = now
                if term_at is not None and now - term_at >= 30 and child.poll() is None:
                    own_group_signal(child, signal.SIGKILL)
                    child.wait(timeout=10)
                # Pipes inherited by a stranded child may otherwise keep us alive.
                if child.poll() is not None and now - started > wall + 40:
                    reason = reason or 'DESCENDANT_PIPE_NOT_CLOSED'; break
            stream.flush(); os.fsync(stream.fileno())
        code = child.wait(timeout=10)
    except BaseException:
        own_group_signal(child, signal.SIGTERM)
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            own_group_signal(child, signal.SIGKILL); child.wait(timeout=10)
        raise
    finally:
        selector.close(); child.stdout.close()
    counter = Path(attempt) / 'COUNTERS.json'
    if counter.exists():
        final_counters, _ = read(counter, limit=65536)
        reason = reason or final_counters.get('limit_exceeded')
    return dict(exit_code=code, limit_reason=reason, wall_seconds=time.monotonic() - started,
                sampled_tree_rss_peak_bytes=maximum_rss, log_bytes_stored=log.stat().st_size,
                log_bytes_received=emitted, counters=final_counters,
                monitoring='RSS/wall sampled at <=1s between reads; torch allocator peak sampled at 0.5s; cooperative kernel scheduling; 30s own-group TERM grace')


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
    stage_root = JOB / 'stages'; stage_root.mkdir(parents=True, exist_ok=True)
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
