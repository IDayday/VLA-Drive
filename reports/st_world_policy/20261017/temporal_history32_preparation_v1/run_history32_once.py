#!/usr/bin/env python3
"""Exact32 binding of the reviewed history64 controller; no auto admission/retry."""
import time
ENTRY = time.monotonic()
with open('/proc/self/io') as _entry_io:
    ENTRY_IO = {k: int(v) for k, v in (line.split(':') for line in _entry_io)}
import signal


def _expire(*_):
    raise TimeoutError('Exact32 absolute bootstrap deadline')


import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import stat
import types

RUNTIME = Path('/mnt/project/VLA-Drive-st-temporal-combination-20261017')
BASE = Path('/mnt/project/st-world-policy-20261016')
REPORT = RUNTIME / 'reports/st_world_policy/20261017'
ORIGINAL = REPORT / 'full_causal_history_preparation_v1/run_pilot_once.py'
ROOT = BASE / 'audits/history_data/temporal_composition32_history_v1'
CONTROL = ROOT.with_name(ROOT.name + '_controls')
SELECTED = BASE / 'audits/indexed_composition_preparation_v1/SELECTED_SCENES.json'
SELECTED_SHA = '7c78c58c40456649a93d152f9925ea5e1faa7e08bb65e98cd6c4af3b0c52b9b8'
PYTHON = '/root/miniconda3/envs/ddp/bin/python'
PINS = {
    'reports/st_world_policy/20261017/full_causal_history_preparation_v1/run_pilot_once.py': 'a8499c33fd778f5f68f285369086e3a19ad480d2e85ec5d7b9f51b1d06bfe016',
    'reports/st_world_policy/20261017/full_future_metadata_preparation_v1/run_build_once.py': '4b2f0fccfd4bf3d655198f61179c88f6b2a51840731465c720b8addb6169b1a2',
    'reports/st_world_policy/20261017/spatial_probe_actual_preparation_v1/supervise_once.py': 'd829cf48764b23b5fce08cdd6ea45c6d3e61e633763ad2cb876e5ab8d96d8902',
    'reports/st_world_policy/20261017/spatial_probe_actual_preparation_v1/prepare_bindings.py': '31d2d302d5dd441df9251762e34cd9d1762afc3463cdabfe8dbf5b1f12577c84',
    'reports/st_world_policy/20261017/spatial_probe_actual_preparation_v1/observed_cli.py': 'c1af891b440663d75ca3b75c1227f5a7d4247d62319a566214848e0a5550f643',
    'tools/st_world_policy/metadata_storage.py': 'aaa36f5383451eadcd92c160c6f3ffb8704110ab44eb12ee433b3207995e71ba',
    'tools/st_world_policy/cache_storage_budget.py': '2103a082271706513127404b35fb1ff9154309b5e5fbe56306b8e831a509b1d5',
}
BUILDER_PINS = {
    'tools/st_world_policy/prepare_causal_history.py': '87f9fb3f5e864962788198076618bdda3edc3d59e231b6ba9bd902080e5c3d63',
    'tools/st_world_policy/prepare_history.py': 'ce75c87cf8199f2b666fa7ec36cd07478fb83107ef87a83dcca70eb5b839a836',
    'tools/st_world_policy/history_shards.py': '53f85bf7e4f24cc3e0d6228b102df7350624f5dff433e04940a5922b6d32a043',
    'tools/st_world_policy/prepare_future_metadata.py': '30cf346e49ad88da30c61990dbe24863727e3979c711df74e2e58b97b4e7e281',
    'starVLA/dataloader/st_world_history.py': 'bdb33b97f820e251913d6c99cb782791d977e01599c75da620d244bd831fb97c',
    'starVLA/dataloader/st_world_metadata_index.py': 'ea8fe604c747d3a227f927c81c60851289b17c7f6fa1bfd36a95a8f9897d545f',
    'starVLA/dataloader/st_world_future_provenance.py': '3e8aceda843ec9724476c09f6c3c6738c1bf52a4a138c9e43084ba27ce4f0772',
    'tools/st_world_policy/prepare_current_teacher.py': 'e5b68a4467ff295b683c7f7f06cc7c0afbb268ff02c77e229c92433d42ad846d',
}


def pinned(pin, *, decode=True):
    p = Path(pin['path'])
    if not p.is_absolute() or '..' in p.parts or any(x.is_symlink() for x in (p, *p.parents)):
        raise ValueError('Canonical non-symlink pin required')
    fd = os.open(p, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > 4 << 20:
            raise ValueError('Small regular pinned input required')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            raw = stream.read((4 << 20) + 1)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    version = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if version(before) != version(after) or len(raw) != before.st_size or hashlib.sha256(raw).hexdigest() != pin['sha256']:
        raise ValueError('Pinned input changed')
    return json.loads(raw) if decode else raw


def limits(cpus):
    return dict(wall_seconds=900, cleanup_seconds=40, read_bytes=16 << 30,
                rss_bytes=4 << 30, cpu_affinity=cpus, max_disk_bytes=32 << 20, max_invocations=1)


def validate_binding(a, cp, *, now, child=False):
    cpus = a.get('limits', {}).get('cpu_affinity')
    if (not isinstance(cpus, list) or not 1 <= len(cpus) <= 2 or
            any(type(x) is not int or x < 0 for x in cpus) or len(set(cpus)) != len(cpus)):
        raise ValueError('One or two explicit CPU IDs required')
    if a.get('limits') != limits(cpus):
        raise ValueError('Exact bounded32 resource envelope required')
    if a.get('schema') != 'st_history32_owned_admission_v1' or a.get('status') != 'APPROVED_HISTORY32_PILOT_ONLY':
        raise ValueError('Independent exact32 admission required')
    if a.get('reviewer') in (None, '', '/root', '/root/st_indexprep_boundary_fix', '/root/st_temporal_readiness_inventory'):
        raise ValueError('Author cannot approve this binding')
    if (a.get('runtime') != str(RUNTIME) or a.get('output_root') != str(ROOT) or
            a.get('control_root') != str(CONTROL) or a.get('control_reserved_bytes') != 1 << 20 or
            a.get('hostname') != socket.gethostname()):
        raise ValueError('Wrong runtime/output/host')
    for key in ('issued_at_unix', 'latest_start_unix', 'valid_until_unix'):
        if type(a.get(key)) not in (int, float) or not math.isfinite(a[key]):
            raise ValueError('Finite lease timestamps required')
    if not a['issued_at_unix'] <= now <= a['latest_start_unix'] or a['valid_until_unix'] < now + (0 if child else 940):
        raise ValueError('Expired or insufficient lease')
    if cp.get('schema') != 'st_checkpoint_storage_commitments_v1' or cp.get('status') != 'CURRENT':
        raise ValueError('Current original checkpoint commitment required')
    if type(cp.get('remaining_bytes')) is not int or cp['remaining_bytes'] < 0 or cp.get('free_floor_bytes') != 1 << 40:
        raise ValueError('Checkpoint capacity/free floor invalid')
    if not cp['checked_at_unix'] <= now <= a['valid_until_unix'] <= cp['valid_until_unix']:
        raise ValueError('Checkpoint commitment does not cover invocation')
    return cpus


def bind_source(raw):
    """Exact-count substitutions only in fixed, SHA-verified historical binding."""
    source = raw.decode()
    replacements = [
        ("ROOT=BASE/'audits/history_data/causal_history_cost64_v2'", 'ROOT=BOUND_ROOT', 1),
        ("CONTROL_ROOT=BASE/'audits/history_data/causal_history_cost64_v2_controls'", 'CONTROL_ROOT=BOUND_CONTROL', 1),
        ("CP={'path':str(BASE/'smoke/official_history_T1_native4_v1/controls/CHECKPOINT_COMMITMENTS_2303G_V1.json'),\n    'sha256':'13f3f4b282db2aff24528bc5434db51a9351cf43cf02dcdea1fbef38d5434cb3'}", 'CP=BOUND_CP', 1),
        ('cpu_affinity=[6,7]', 'cpu_affinity=BOUND_CPUS', 1),
        ('st_history64_owned_admission_v1', 'st_history32_owned_admission_v1', 1),
        ('APPROVED_HISTORY64_PILOT_ONLY', 'APPROVED_HISTORY32_PILOT_ONLY', 1),
        ("required|={str(helper/n) for n in ('supervise_once.py','prepare_bindings.py','observed_cli.py')}", "required|={str(helper/n) for n in ('supervise_once.py','prepare_bindings.py','observed_cli.py')}\n    required.add(str(WRAPPER_PATH))", 1),
        ('audits/history_data/smoke_train64_v1/SELECTED_SCENES.json', 'audits/indexed_composition_preparation_v1/SELECTED_SCENES.json', 2),
        ('dee94e75957fd8e9734f1669b47552b5cf6c39f719aa9d3298bc87f94aac1c01', SELECTED_SHA, 1),
        ("expected={'max_scenes':64", "expected={'max_scenes':32", 1),
        ('2303*(1<<30)', 'BOUND_CP_REMAINING', 2),
        ('cpu_affinity=(6,7)', 'cpu_affinity=tuple(BOUND_CPUS)', 1),
        ('os.sched_setaffinity(0,{6,7})', 'os.sched_setaffinity(0,set(BOUND_CPUS))', 1),
        ('str(Path(__file__).resolve())', 'str(WRAPPER_PATH)', 1),
    ]
    for old, new, count in replacements:
        if source.count(old) != count:
            raise ValueError('Historical binding source no longer matches exact adaptation')
        source = source.replace(old, new)
    return source


def make_controller(raw, a, cp, cpus, baseline):
    module = types.ModuleType('history32_bound_controller')
    module.__dict__.update(__file__=str(ORIGINAL), BOUND_ROOT=ROOT, BOUND_CONTROL=CONTROL,
        BOUND_CP=a['checkpoint_receipt'], BOUND_CP_REMAINING=cp['remaining_bytes'],
        BOUND_CPUS=cpus, WRAPPER_PATH=Path(__file__).resolve())
    exec(compile(bind_source(raw), str(ORIGINAL), 'exec'), module.__dict__)
    module.ENTRY_MONOTONIC = ENTRY
    module.ENTRY_IO = baseline
    module.read = pinned
    return module


def validate_registration(reg, a, cpus):
    expected_limits = dict(max_depth=1, max_disk_bytes=31 << 20, max_entries=256,
                           max_rss_bytes=4 << 30, max_seconds=900)
    if reg.get('limits') != expected_limits:
        raise ValueError('Exact31MiB artifacts plus separate1MiB control required')
    budget_pin = a['builder_budget']
    expected = [PYTHON, '-m', 'tools.st_world_policy.prepare_causal_history', 'build',
        '--student-root', '/mnt/project/ddp-foresight-artifacts/20260928/student_train_v1',
        '--raw-log-root', '/mnt/navsim/trainval_navsim_logs/trainval',
        '--sensor-root', '/mnt/navsim/trainval_all/trainval_sensor_blobs/trainval',
        '--fallback-sensor-root', '/mnt/navsim/trainval_sensor_blobs/trainval',
        '--partition', '/mnt/project/VLA-Drive-st-history-population-agent-20261016/reports/ddpolicy_vehicle_from_scratch/NAVTRAIN_PARTITION.json',
        '--scene-list', str(SELECTED), '--split', 'train', '--mode', 'registered',
        '--max-scenes', '32', '--seed', '20261017', '--tolerance', '0.05',
        '--target-times', '0', '--record-layout', 'packed_log_v1',
        '--registration', budget_pin['path'], '--output', str(ROOT)]
    if reg.get('command') != expected or reg.get('cpu_affinity') != cpus:
        raise ValueError('Exact current/history observation recipe required')
    budget = pinned(budget_pin)
    if any(budget.get(k) != v for k, v in dict(phase='data_preparation', split='train',
           max_scenes=32, max_seconds=900, workers=1, target_times_s=[0.0]).items()):
        raise ValueError('Exact builder budget required')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--admission', required=True)
    parser.add_argument('--admission-sha256', required=True)
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CUDA must be hidden')
    if signal.getitimer(signal.ITIMER_REAL) != (0., 0.):
        raise ValueError('Do not override caller deadline')
    signal.signal(signal.SIGALRM, _expire)
    signal.setitimer(signal.ITIMER_REAL, max(.001, ENTRY + 900 - time.monotonic()))
    a = pinned({'path': args.admission, 'sha256': args.admission_sha256})
    cp = pinned(a['checkpoint_receipt'])
    cpus = validate_binding(a, cp, now=time.time(), child=args.child)
    os.sched_setaffinity(0, set(cpus))
    raw = None
    for relative, sha in {**PINS, **BUILDER_PINS}.items():
        data = pinned({'path': str(RUNTIME / relative), 'sha256': sha}, decode=False)
        if RUNTIME / relative == ORIGINAL:
            raw = data
    own_pin = next((p for p in a['controller_sources'] if p['path'] == str(Path(__file__).resolve())), None)
    if own_pin is None:
        raise ValueError('Actual wrapper source must be reviewed and pinned')
    pinned(own_pin, decode=False)
    selected = pinned({'path': str(SELECTED), 'sha256': SELECTED_SHA})
    if len(selected['scenes']) != 32:
        raise ValueError('Exact32 population required')
    reg = pinned(a['registration'])
    validate_registration(reg, a, cpus)
    expected_pins = [{'path': str(RUNTIME / p), 'sha256': sha} for p, sha in BUILDER_PINS.items()]
    if sorted(reg['source_pins'], key=lambda x: x['path']) != sorted(expected_pins, key=lambda x: x['path']):
        raise ValueError('Wrong current builder source pins')
    if (reg.get('record_layout') != 'packed_log_v1' or reg.get('max_scenes') != 32 or
            reg.get('inventory_strategy') != 'immutable_history_shards_v1' or
            reg.get('cpu_affinity') != cpus or reg.get('max_history_shards') != 32 or
            reg.get('max_source_read_bytes') != 16 << 30 or reg.get('read_mib_per_second') != 32):
        raise ValueError('Wrong exact32 packed resource contract')
    m = make_controller(raw, a, cp, cpus, ENTRY_IO)
    # Hand off to the existing absolute ENTRY+900 alarm before any I/O or import
    # in controls(). The original main parses the same already-validated argv.
    signal.setitimer(signal.ITIMER_REAL, 0)
    return m.main()


if __name__ == '__main__':
    raise SystemExit(main())
