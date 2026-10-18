#!/usr/bin/env python3
"""Continue ONLY fixed C2/C3 completed-dev scoring under a fresh score lease.

All scoring/registration/identity/lock/admission checks remain in pinned v5.main.
No lease delegates unchanged v5 behavior. No export or route rewrite is supported.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import stat
import sys
sys.dont_write_bytecode = True

ROOT = Path('/mnt/project/st-world-policy-20261016')
SOURCE = '6714b1e9fb2e0c8059b90b8b4d48cd40796fe345'
HELPER = {'path': str(ROOT / 'resources/launch_completed_dev_score_v5.py'),
          'sha256': '2e62c84a2eecde3e9af585d0a6fb77f9ff33d27a2ec76cd0fa2b594b3a8f15c3'}
JOBS = ('C2_step004096_dev1696_seed42_v1', 'C3_step004096_dev1696_seed42_v1')
LIMITS = {'score_workers': 2, 'max_calls_total_across_resume': 1728,
          'native_max_seconds': 3600, 'outer_wall_seconds': 4200,
          'cleanup_seconds': 60, 'maximum_score_invocations': 2,
          'maximum_campaign_score_slots': 2, 'gpu_count': 0,
          'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1',
          'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}


def pinned_small(pin, *, limit=128 << 10, decode=True):
    if not isinstance(pin, dict) or set(pin) != {'path','sha256'}:
        raise ValueError('Exact path and SHA256 required')
    path = Path(pin['path'])
    if (not path.is_absolute() or '..' in path.parts
            or any(p.is_symlink() for p in (path, *path.parents))):
        raise ValueError('Absolute nonsymlink immutable small source required')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise ValueError('Only bounded regular source/control files')
        raw = stream.read(limit + 1); after = os.fstat(stream.fileno())
    version = lambda s: (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    if len(raw) != before.st_size or version(before) != version(after) or version(after) != version(path.stat()):
        raise ValueError('Immutable source/control changed while reading')
    if hashlib.sha256(raw).hexdigest() != pin['sha256']:
        raise ValueError('Immutable source/control SHA256 mismatch')
    return json.loads(raw) if decode else raw


def validate_lease(lease, route, *, job, route_pin, hostname, now, remaining_seconds):
    fields = {'schema','job_id','original_export_route','hostname','source_sha',
              'environment','score_helper','valid_from_unix','valid_until_unix',
              'limits','scope','automatic_retry','export_authorized'}
    if set(lease) != fields or lease['schema'] != 'st_fixed_dev_score_execution_lease_v1':
        raise ValueError('Exact score-only execution lease schema required')
    if (job not in JOBS or lease['job_id'] != job or route['job_id'] != job
            or lease['original_export_route'] != route_pin
            or route_pin['path'] != str(ROOT/'formal/evaluation/registrations'/job/'ROUTE.json')
            or lease['hostname'] != hostname or route['hostname'] != hostname
            or lease['source_sha'] != SOURCE or route['source_sha'] != SOURCE
            or lease['environment'] != route['environment'] or lease['score_helper'] != HELPER):
        raise ValueError('Lease changed original job/route/host/source/environment')
    if (lease['limits'] != LIMITS or route['inference']['score_workers'] != 2
            or route['inference']['score_max_calls'] != 1728
            or lease['scope'] != 'score_existing_complete_export_only'
            or lease['automatic_retry'] is not False or lease['export_authorized'] is not False
            or remaining_seconds != 4260):
        raise ValueError('Original fixed score resource/dispatch limits required')
    start, end = lease['valid_from_unix'], lease['valid_until_unix']
    if any(type(x) not in (int,float) or not math.isfinite(x) for x in (start,end,now)):
        raise ValueError('Finite actual execution times required')
    if not (0 < end-start <= 21600 and start <= now and end-now >= remaining_seconds):
        raise ValueError('Fresh score execution lease expired/not yet valid/too short')


def install(module, *, lease_pin, job, route_pin, hostname):
    """Two narrow seams: real lease time checks and additive receipt provenance."""
    lease = pinned_small(lease_pin, limit=16 << 10)
    route = pinned_small(route_pin, limit=16 << 10)
    wrapper = {'path': str(Path(__file__).absolute()),
               'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    old_time, old_write = module.check_route_time_v3, module.atomic_json

    def check_time(actual_route, now, remaining_seconds):
        # Keep both original time-check sites. Re-read tiny immutable pins at
        # both sites; neither original route timestamps nor `now` are altered.
        if pinned_small(lease_pin, limit=16 << 10) != lease or pinned_small(route_pin, limit=16 << 10) != route:
            raise ValueError('Execution lease/original route changed during admission')
        if actual_route != route or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            raise ValueError('Wrong route object or GPU-visible scoring process')
        validate_lease(lease, actual_route, job=job, route_pin=route_pin,
                       hostname=hostname, now=now, remaining_seconds=remaining_seconds)

    def write_receipt(path, value, *args, **kwargs):
        path = Path(path)
        invocations = ROOT/'formal/evaluation/registrations'/job/'score_invocations'
        if path.parent.parent == invocations and path.name in ('ADMISSION.json','LAUNCH.json'):
            if not kwargs.get('immutable') or value.get('route') != route_pin:
                raise ValueError('Original immutable score receipt/route required')
            if any(key in value for key in ('score_execution_lease','execution_lease_wrapper')):
                raise ValueError('Existing receipt provenance must not be replaced')
            value = {**value, 'score_execution_lease': lease_pin, 'execution_lease_wrapper': wrapper}
        return old_write(path, value, *args, **kwargs)

    module.check_route_time_v3, module.atomic_json = check_time, write_receipt
    return old_time, old_write


def load_helper():
    pinned_small(HELPER, decode=False)
    spec = importlib.util.spec_from_file_location('fixed_dev_score_v5_with_lease', HELPER['path'])
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--route-sha256', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--score-lease')
    parser.add_argument('--score-lease-sha256')
    args = parser.parse_args()
    if bool(args.score_lease) != bool(args.score_lease_sha256):
        parser.error('--score-lease and --score-lease-sha256 must be used together')
    module = load_helper()  # Pinned v5 itself disables CUDA before imports.
    if args.score_lease:
        install(module, lease_pin={'path':args.score_lease,'sha256':args.score_lease_sha256},
                job=args.job_id, route_pin={'path':str(ROOT/'formal/evaluation/registrations'/args.job_id/'ROUTE.json'),
                                          'sha256':args.route_sha256}, hostname=socket.gethostname())
    sys.argv = [HELPER['path'], '--job-id', args.job_id, '--route-sha256', args.route_sha256]
    if args.resume: sys.argv.append('--resume')
    return module.main()


if __name__ == '__main__': raise SystemExit(main())
