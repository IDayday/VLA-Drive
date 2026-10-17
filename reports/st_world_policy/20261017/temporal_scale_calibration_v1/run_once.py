#!/usr/bin/env python3
"""One no-update calibration invocation using the existing owned monitor.

No cache/ledger mutation, checkpoint, retry, formal approval or training launch.
"""
with open('/proc/self/io') as _stream:
    BOOT_IO = {k: int(v) for k, v in (line.split(':') for line in _stream)}

import argparse
from contextlib import ExitStack, redirect_stdout
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time

HERE = Path(__file__).resolve().parent
RUNTIME = HERE.parents[3]
BASE = Path('/mnt/project/st-world-policy-20261016')
ROOT = BASE / 'audits/loss_scale_calibration_real4_v1'
PYTHON = '/root/miniconda3/envs/ddp/bin/python'
LIMITS = dict(wall_seconds=3600, cleanup_seconds=40, read_bytes=256 << 30,
              rss_bytes=96 << 30, cpu_affinity=[2, 3], max_disk_bytes=16 << 20,
              max_invocations=1, optimizer_updates=0, new_tensor_bytes=0)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pin_read(pin, *, decode=True):
    path = Path(pin['path'])
    if not path.is_absolute() or '..' in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Canonical absolute source/control required')
    before = path.stat()
    if before.st_size > 4 << 20:
        raise ValueError('Small control/source only')
    raw = path.read_bytes()
    after = path.stat()
    version = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if version(before) != version(after) or hashlib.sha256(raw).hexdigest() != pin['sha256']:
        raise ValueError('Control/source checksum changed')
    return json.loads(raw) if decode else raw


def output_bytes(root):
    total = 0
    count = 0
    for parent, directories, files in os.walk(root, followlinks=False):
        for name in directories + files:
            path = Path(parent) / name
            count += 1
            if count > 128 or path.is_symlink():
                raise ValueError('Unexpected calibration output layout')
            if path.is_file():
                total += max(path.stat().st_size, path.stat().st_blocks * 512)
    return total


def publish(write, path, value):
    bound = len(json.dumps(value, sort_keys=True, allow_nan=False).encode()) * 2 + 8192
    if output_bytes(ROOT) + bound > LIMITS['max_disk_bytes']:
        raise ValueError('Insufficient output room for atomic result publication')
    write(path, value)


def storage_guard(control):
    cp = pin_read(control['checkpoint_receipt'])
    if (cp.get('schema') != 'st_checkpoint_storage_commitments_v1' or cp.get('status') != 'CURRENT'
            or type(cp.get('remaining_bytes')) is not int or cp['remaining_bytes'] < 2303 * (1 << 30)
            or cp.get('free_floor_bytes') != 1 << 40
            or cp.get('valid_until_unix', 0) < control['valid_until_unix']):
        raise ValueError('Fresh complete checkpoint commitments required')
    stat = os.statvfs(BASE)
    if stat.f_bavail * stat.f_frsize < cp['remaining_bytes'] + (1 << 40) + LIMITS['max_disk_bytes']:
        raise ValueError('Shared storage reserve is unavailable')
    if output_bytes(ROOT) > LIMITS['max_disk_bytes']:
        raise ValueError('Calibration output limit')


def controls(args):
    control = pin_read({'path': args.control, 'sha256': args.control_sha256})
    if (control.get('schema') != 'st_loss_scale_owned_control_v1' or control.get('limits') != LIMITS
            or control.get('host') != socket.gethostname() or control.get('runtime') != str(RUNTIME)
            or control.get('output_root') != str(ROOT)):
        raise ValueError('Exact one-shot resource controls required')
    now = time.time()
    if not control['issued_at_unix'] <= now <= control['latest_start_unix']:
        raise ValueError('Not in launch window')
    if control['valid_until_unix'] < now + (0 if args.child else 3640):
        raise ValueError('Insufficient absolute resource lease')
    for source in control['sources']:
        pin_read(source, decode=False)
    helper = HERE.parent / 'spatial_probe_actual_preparation_v1'
    required = {str(HERE / 'run_once.py'), str(HERE.parent / 'full_future_metadata_preparation_v1/run_build_once.py')}
    required.update(str(helper / name) for name in ('supervise_once.py', 'prepare_bindings.py', 'observed_cli.py'))
    if {p['path'] for p in control['sources']} != required:
        raise ValueError('Exact existing monitor dependency set required')
    review = pin_read(control['resource_review'])
    if (review.get('status') != 'PASS_CALIBRATION_CONTROLLER_ONLY'
            or review.get('reviewer') in (None, '', '/root', '/root/st_temporal_scale_calibration')
            or review.get('controller_sources') != control['sources']
            or review.get('limits') != LIMITS):
        raise ValueError('Independent exact controller review required')
    sys.path.insert(0, str(helper))
    sys.path.insert(0, str(RUNTIME))
    import supervise_once as supervisor
    import observed_cli as observer
    from tools.st_world_policy import calibrate_temporal_scale as cli
    from tools.st_world_policy import metadata_storage as metadata
    from tools.st_world_policy.base_binding import immutable_json
    core = load('calibration_owned_core', HERE.parent / 'full_future_metadata_preparation_v1/run_build_once.py')
    core.BOOT_IO = BOOT_IO  # Include this parent's imports/setup in accounting.
    admission = pin_read(control['admission'])
    cfg = pin_read(admission['config'])
    if (cfg['output_root'] != str(ROOT / 'result') or len(cfg['batch']['scene_indices']) != 4
            or cfg['purpose'] != 'DIAGNOSTIC' or admission['source_sha'] != control['source_sha']
            or admission['valid_until_unix'] > control['valid_until_unix']
            or admission['max_seconds'] != LIMITS['wall_seconds']):
        raise ValueError('Fixed real4 no-update diagnostic required')
    cli.validate(cfg)
    storage_guard(control)
    return control, admission, cli, metadata, immutable_json, supervisor, observer, core


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', required=True)
    parser.add_argument('--control-sha256', required=True)
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.sched_setaffinity(0, {2, 3})
    control, admission, cli, metadata, write, supervisor, observer, core = controls(args)
    if args.child:
        os.fstat(int(os.environ['ST_PROBE_SUPERVISOR_LOCK_FD']))
        if os.getpid() != os.getpgrp():
            raise ValueError('Actual owned session child required')
        before = observer.proc_io()
        code = 2
        error = None
        old_argv = sys.argv
        try:
            sys.argv = [str(RUNTIME / 'tools/st_world_policy/calibrate_temporal_scale.py'), 'measure',
                        '--config', admission['config']['path'], '--admission', control['admission']['path'],
                        '--admission-sha256', control['admission']['sha256']]
            with redirect_stdout(sys.stderr):
                code = cli.main()
        except BaseException as exc:
            error = {'type': type(exc).__name__, 'message': str(exc)}
        finally:
            sys.argv = old_argv
        import torch
        gpu = {'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
               'peak_reserved_bytes': torch.cuda.max_memory_reserved()} if torch.cuda.is_initialized() else None
        after = observer.proc_io()
        print(json.dumps({'schema': 'st_full_future_child_result_v2', 'cli_result': {'exit_code': code, 'cuda': gpu},
                          'error': error, 'counters': {'proc_self_io_baseline': before, 'proc_self_io_final': after,
                          'proc_self_io_delta': {k: after[k] - before[k] for k in after}}}))
        return code
    with ExitStack() as stack:
        # Both locks fail immediately on contention, before START or GPU launch.
        lock = supervisor.acquire(BASE / 'resources/LOSS_SCALE_REAL4_ONE_SHOT.lock', stack)
        gpu = {'index': int(admission['cuda_visible_devices']), 'uuid': admission['gpu_uuids'][0]}
        supervisor.acquire('/tmp/foresight_gpu_' + str(gpu['index']) + '.lock', stack)
        supervisor.fresh_gpu(gpu)
        if ROOT.exists():
            raise ValueError('One attempt only; preserve prior output without automatic retry')
        ROOT.mkdir()
        publish(write, ROOT / 'START.json', {'control': args.control, 'sha256': args.control_sha256, 'automatic_retry': False})
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=admission['cuda_visible_devices'], PYTHONDONTWRITEBYTECODE='1',
                   PYTHONPATH=str(RUNTIME), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                   ST_PROBE_SUPERVISOR_LOCK_FD=str(lock.fileno()), ST_CALIBRATION_CONTROLLER_TOKEN=admission['controller_token'])
        command = [PYTHON, str(Path(__file__).resolve()), '--child', '--control', args.control,
                   '--control-sha256', args.control_sha256]
        try:
            result = core.monitor(command, runtime=str(RUNTIME), env=env, fd=lock.fileno(), root=ROOT,
                                  s=supervisor, m=metadata, obs=observer, budget=LIMITS, cpu_affinity=(2, 3),
                                  storage_guard=lambda: storage_guard(control), lease_valid_until=control['valid_until_unix'])
        except BaseException as error:
            publish(write, ROOT / 'FAILED.json', {'status': 'FAILED_NO_AUTO_RETRY',
                    'error_type': type(error).__name__, 'error': str(error),
                    'parent_post_cleanup_io': observer.proc_io(), 'resource_result_complete': False})
            raise
        charge = core.charge_after_reap(result['read_accounting'], observer.proc_io())
        measurement = ROOT / 'result/MEASUREMENT.json'
        ok = result['returncode'] == 0 and result['failure'] is None and measurement.is_file() and charge <= LIMITS['read_bytes']
        storage_guard(control)
        publish(write, ROOT / 'RESULT.json', {'status': 'COMPLETE' if ok else 'FAILED_NO_AUTO_RETRY',
              'resources': result, 'charged_parent_child_rchar_bytes': charge,
              'optimizer_updates': 0, 'new_tensor_bytes': 0, 'scientific_calibration_selected': False})
        print(json.dumps({'status': 'COMPLETE' if ok else 'FAILED_NO_AUTO_RETRY', 'root': str(ROOT)}))
        return 0 if ok else 2


if __name__ == '__main__':
    raise SystemExit(main())
