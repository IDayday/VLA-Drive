#!/usr/bin/env python3
"""Two fixed existing CPU analysis commands. Lead reviews and invokes --execute.

No resource approval, training, scoring, cache admission or automatic retry.
"""
with open('/proc/self/io') as _stream:
    BOOT_IO = {k: int(v) for k, v in (line.split(':') for line in _stream)}
import argparse
from contextlib import contextmanager, redirect_stdout
import csv
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import time
import traceback

MiB = 1 << 20
TOTAL_READ = 1 << 30
OUTPUT_LIMIT = 32 * MiB
PAIR_LIMITS = dict(wall_seconds=1800, cleanup_seconds=40, rss_bytes=2 << 30,
                   cpu_affinity=[0, 1], stdout_bytes=MiB, stderr_bytes=MiB)
ROOT = Path('/mnt/project/st-world-policy-20261016')
CONTROL = ROOT / 'audits/evaluation/FULL_PAYLOAD_C2_C3_1024_EXECUTION_v2'
RUNTIME = Path('/mnt/project/VLA-Drive-st-flow-context-runtime-v3-20261017')
SOURCE = '6714b1e9fb2e0c8059b90b8b4d48cd40796fe345'
PYTHON = '/root/miniconda3/envs/ddp/bin/python'
PREP = Path('/mnt/project/VLA-Drive-st-c4-dev-preparation-20261017/reports/st_world_policy/20261016/evaluation/FULL_PAYLOAD_C2_C3_1024_BUDGET_v1')
COMMANDS = {'path': str(PREP / 'COMMANDS.json'), 'sha256': 'f93ff9abb821a632f7c08c1f961c5c6219ccd14b3c4218b1902bf398d07d8d90'}
BUDGET = {'path': str(PREP / 'BUDGET.json'), 'sha256': '21242f52da877cbe353de560e2c8298434aa3d13ec90ed866698d8eda0206854'}
CORE = {'path': '/mnt/project/VLA-Drive-st-world-policy-20261016/reports/st_world_policy/20261017/full_future_metadata_preparation_v1/run_build_once.py',
        'sha256': '8674aea1f2a3df894bb3c451287e6a14b5c20d59b01efe4883158f73a3690644'}
HELPERS = Path('/mnt/project/VLA-Drive-st-spatial-probe-runtime-v2-20261017/reports/st_world_policy/20261017/spatial_probe_actual_preparation_v1')
RSS_RUNTIME = Path('/mnt/project/VLA-Drive-st-metadata-build-dev-v1-20261017')
RSS_SOURCES = {
    'metadata_storage.py': '46233cf232a61e73e7e24f967b1cf316298473b017f49f4e96432d7e1b1a3591',
    'cache_storage_budget.py': '2103a082271706513127404b35fb1ff9154309b5e5fbe56306b8e831a509b1d5',
}


def safe(path):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Absolute non-symlink paths required')
    return path


def file_hash(path):
    h = hashlib.sha256()
    with safe(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(MiB), b''):
            h.update(chunk)
    return h.hexdigest()


def dependencies():
    if file_hash(CORE['path']) != CORE['sha256']:
        raise ValueError('Reviewed core changed')
    spec = importlib.util.spec_from_file_location('fixed_pair_reviewed_core', CORE['path'])
    core = importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
    for name, digest in core.MONITORS.items():
        core.pinned({'path': str(HELPERS / name), 'sha256': digest}, decode=False)
    sys.path.insert(0, str(HELPERS))
    import supervise_once as supervisor
    import observed_cli as observer
    if Path(supervisor.__file__).resolve() != HELPERS / 'supervise_once.py' or Path(observer.__file__).resolve() != HELPERS / 'observed_cli.py':
        raise ValueError('Mixed helper modules')
    return core, supervisor, observer


def fixed_commands(core):
    doc, budget = core.pinned(COMMANDS), core.pinned(BUDGET)
    if doc['source_sha'] != SOURCE or not doc['sequential_only'] or len(doc['commands']) != 2:
        raise ValueError('Only two fixed sequential pairs')
    result = []
    for arm, item in zip(('C2', 'C3'), doc['commands']):
        original = core.pinned(item['source_preparation_command'])
        command = item['original_command_unchanged']
        matches = [x for x in original['paired_analyses'] if x['comparison'] == arm + '_minus_C0' and x['stage'] == 'DIAGNOSTIC_1024']
        if matches != [command] or command['cwd'] != str(RUNTIME):
            raise ValueError('Original analysis command changed')
        argv = command['argv']
        if (argv[:3] != [PYTHON, '-m', 'tools.st_world_policy.evaluate_campaign']
                or command['env']['CUDA_VISIBLE_DEVICES'] != '' or command['max_seconds'] != 1800
                or command['comparison'] != arm + '_minus_C0' or command['stage'] != 'DIAGNOSTIC_1024'):
            raise ValueError('Wrong fixed command, stage, source or CPU environment')
        plan = core.pinned(command['analysis_plan'])
        if plan['statistics'] != budget['statistics_unchanged']:
            raise ValueError('Statistics changed')
        result.append(command)
    return result, budget


def rss_monitor(core):
    # Reuse only the existing /proc tree reader; do not invoke storage APIs.
    for name, digest in RSS_SOURCES.items():
        core.pinned({'path': str(RSS_RUNTIME / 'tools/st_world_policy' / name), 'sha256': digest}, decode=False)
    sys.path.insert(0, str(RSS_RUNTIME))
    from tools.st_world_policy import metadata_storage
    if Path(metadata_storage.__file__).resolve() != RSS_RUNTIME / 'tools/st_world_policy/metadata_storage.py':
        raise ValueError('Mixed RSS monitor source')
    return metadata_storage


def option(command, key):
    args = command['argv']; return args[args.index('--' + key) + 1]


def output_size(root):
    root = safe(root)
    if not root.exists(): return 0
    seen, size = set(), 0
    for p in root.iterdir():
        safe(p)
        if not p.is_file(): raise ValueError('Unexpected analysis output entry')
        stat = p.stat(); key = stat.st_dev, stat.st_ino
        if key not in seen: size += stat.st_size; seen.add(key)
    if size > OUTPUT_LIMIT: raise ValueError('Analysis output exceeds 32MiB')
    return size


def complete_output(command, core, budget):
    root = safe(option(command, 'output'))
    if not root.exists(): return None
    size = output_size(root)
    if {p.name for p in root.iterdir()} != {'ANALYSIS.json', 'PAIRED_SCENES.csv'}:
        raise ValueError('Partial/foreign output retained; no retry or overwrite')
    path = root / 'ANALYSIS.json'
    if path.stat().st_size > 4 * MiB: raise ValueError('Analysis receipt too large')
    record = json.loads(path.read_text())
    if (record.get('schema') != 'st_world_paired_analysis_v1' or record.get('status') != 'COMPLETE'
            or record.get('scope') != 'FORMAL' or record.get('comparison') != command['comparison']
            or record.get('stage') != 'DIAGNOSTIC_1024'
            or record.get('validity') != {'requested': 1696, 'paired_valid': 1696, 'invalid': 0, 'unresolved': 0}
            or record.get('plan_sha256') != command['analysis_plan']['sha256']
            or record.get('population_sha256') != budget['population']['sha256']
            or record.get('statistics_definition') != budget['statistics_unchanged']
            or record.get('analysis_source_sha256') != budget['analysis_source']['sha256']
            or record.get('paired_scenes_csv_sha256') != file_hash(root / 'PAIRED_SCENES.csv')):
        raise ValueError('Existing output binding differs')
    treatment = command['comparison'].split('_minus_')[0]
    for side, arm in (('control', 'C0'), ('treatment', treatment)):
        bound = budget['runs'][arm]
        score, export = core.pinned(bound['score_identity']), core.pinned(bound['export_identity'])
        complete = core.pinned(bound['score_complete'])
        if (record[side]['score_identity'] != score['identity'] or record[side]['export_identity'] != export['identity']
                or record[side]['checkpoint'] != export['checkpoint']
                or record[side]['scenes_csv_sha256'] != complete['scenes_csv_sha256']
                or record[side]['score_root'] != option(command, side + '-score')
                or record[side]['export_root'] != option(command, side + '-export')):
            raise ValueError('Existing result has foreign inputs')
    selected = core.pinned(budget['population'])
    expected = {x['token']: x['log'] for x in selected['scenes']}
    with (root / 'PAIRED_SCENES.csv').open(newline='') as stream:
        reader = csv.DictReader(stream); rows = list(reader)
    if len(rows) != 1696 or {r['scene_id']: r['source_log'] for r in rows} != expected:
        raise ValueError('Existing paired output population differs')
    if not {'control_raw_progress_m', 'treatment_mean_executed_speed_mps', 'new_stop_proxy'}.issubset(reader.fieldnames):
        raise ValueError('Stopping diagnostics absent')
    return {'path': str(path), 'sha256': file_hash(path), 'output_bytes': size}


class ReadMeter:
    """Parent /proc counters absorb each reaped child exactly once."""
    def __init__(self, cap=TOTAL_READ): self.cap, self.total, self.last = cap, 0, 0
    def parent(self, current):
        n = current['rchar']
        if n < self.last: raise ValueError('Nonmonotonic parent reads')
        self.total += n - self.last; self.last = n
        if self.total > self.cap: raise ValueError('Whole-process 1GiB read budget exhausted')
        return self.cap - self.total
    def reaped(self, core, accounting, current):
        charge = core.charge_after_reap(accounting, current)
        self.total += charge; self.last = current['rchar']
        if self.total > self.cap: raise ValueError('Whole-process 1GiB read budget exhausted')
        return charge


def immutable(path, value):
    raw = (json.dumps(value, sort_keys=True, allow_nan=False, ensure_ascii=True) + '\n').encode()
    # Two retained 1MiB pipe prefixes can expand through JSON escaping.
    if len(raw) > 16 * MiB: raise ValueError('Serialized control exceeds 16MiB')
    with safe(path).open('xb') as stream: stream.write(raw)
    return hashlib.sha256(raw).hexdigest()


class OutputGuard:
    def __init__(self, root, payloads, cap=OUTPUT_LIMIT):
        self.root, self.payloads, self.cap, self.written = safe(root), set(payloads), cap, 0
    def audit(self, event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes)): return
        path = Path(os.fsdecode(args[0])).absolute(); flags = args[2]
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT):
            self.allowed_write(path)
        elif path.suffix.lower() in ('.pt', '.pth', '.npz', '.npy', '.pkl', '.gz', '.safetensors', '.png', '.jpg', '.jpeg') or str(path).startswith('/dev/nvidia'):
            if path not in self.payloads: raise ValueError('Payload outside fixed ledger/prediction/dev-GT allowlist')
    def allowed_write(self, path):
        path = safe(path)
        if path.parent != self.root or not (path.name in ('ANALYSIS.json', 'PAIRED_SCENES.csv') or re.fullmatch(r'ANALYSIS\.json\.tmp\.[0-9a-f]{32}', path.name)):
            raise ValueError('Write outside fixed immutable analysis outputs')
    @contextmanager
    def writers(self):
        import builtins
        old_builtin, old_io, old_path = builtins.open, io.open, Path.open
        owner = self
        class Limited:
            def __init__(self, stream): self.stream = stream
            def __getattr__(self, key): return getattr(self.stream, key)
            def __enter__(self): return self
            def __exit__(self, *args): return self.stream.__exit__(*args)
            def write(self, value):
                data = value.encode(getattr(self.stream, 'encoding', None) or 'utf8') if isinstance(value, str) else value
                if owner.written + len(data) > owner.cap: raise ValueError('Serialized output exceeds 32MiB before write')
                owner.written += len(data)
                return self.stream.write(value)
            def writelines(self, lines):
                for line in lines: self.write(line)
        def wrap(original):
            def opened(file, mode='r', *args, **kwargs):
                if any(flag in mode for flag in 'wax+'):
                    owner.allowed_write(Path(file).absolute())
                    if 'x' not in mode or '+' in mode: raise ValueError('Only exclusive immutable output writes')
                    return Limited(original(file, mode, *args, **kwargs))
                return original(file, mode, *args, **kwargs)
            return opened
        builtins.open, io.open = wrap(old_builtin), wrap(old_io)
        # Python 3.10 pathlib caches its opener in an accessor; patch that public
        # entry too so the unchanged CLI's Path.open('xb'/'x') is counted.
        Path.open = wrap(old_io)
        try: yield
        finally: builtins.open, io.open, Path.open = old_builtin, old_io, old_path


class SmallText(io.StringIO):
    def __init__(self): super().__init__(); self.bytes = 0
    def write(self, value):
        self.bytes += len(value.encode('utf8'))
        if self.bytes > 64 * 1024: raise ValueError('CLI stdout exceeds 64KiB envelope allowance')
        return super().write(value)


def allowed_payloads(command, core):
    plan = core.pinned(command['analysis_plan'])
    selected = core.pinned(plan['populations']['dev']['manifest'])
    tokens = {r['token'] for r in selected['scenes']}
    if len(tokens) != 1696: raise ValueError('Full fixed population required')
    paths = {Path(plan['populations']['dev']['current_root']) / 'ego' / (t + '.pt') for t in tokens}
    for side in ('control', 'treatment'):
        paths.update(Path(option(command, side + '-export')) / 'predictions' / (t + '.npz') for t in tokens)
        ledger = Path(option(command, side + '-score')) / 'ledger'
        paths.update(ledger.glob('*/attempt_*/RESULTS.pt'))
    return paths


def child(pair):
    os.fstat(int(os.environ['ST_EVAL_CHILD_LOCK_FD']))
    if os.getpid() != os.getpgrp() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Only this invocation owned CPU session')
    core, _, obs = dependencies(); commands, _ = fixed_commands(core)
    command = commands[('C2', 'C3').index(pair)]
    guard = OutputGuard(option(command, 'output'), allowed_payloads(command, core))
    sys.addaudithook(guard.audit)
    sys.path.insert(0, str(RUNTIME)); sys.dont_write_bytecode = True
    stdout, code, error = SmallText(), 0, None
    try:
        sys.argv = ['tools.st_world_policy.evaluate_campaign', *command['argv'][3:]]
        with guard.writers(), redirect_stdout(stdout):
            try: runpy.run_module('tools.st_world_policy.evaluate_campaign', run_name='__main__', alter_sys=True)
            except SystemExit as exc: code = 0 if exc.code is None else int(exc.code)
        if code == 0: output_size(Path(option(command, 'output')))
    except BaseException as exc:
        code = 1; error = {'type': type(exc).__name__, 'message': str(exc)[:2048]}; traceback.print_exc()
    # Original final envelope and observer; no persistent observer/counter worker.
    class NoCUDA:
        @staticmethod
        def is_initialized(): return False
    class CPUOnly: cuda = NoCUDA
    counter = obs.Counters(Path('/unused-ram-only'), CPUOnly, TOTAL_READ, BOOT_IO)
    original_write = obs.write_control; obs.write_control = lambda *a, **k: None
    try: counters = counter.snapshot()
    finally: obs.write_control = original_write
    final = obs.proc_io()
    counters.update(proc_self_io_baseline=BOOT_IO, proc_self_io_final=final,
                    proc_self_io_delta={k: final[k] - BOOT_IO.get(k, 0) for k in final},
                    read_accounting_scope='pre-import delta shown separately; parent charges absolute child lifetime rchar including Python/Torch imports')
    print(json.dumps({'schema': 'st_full_future_child_result_v2',
        'cli_result': {'status': 'ANALYSIS_RETURNED', 'exit_code': code, 'stdout': stdout.getvalue(), 'error': error,
                       'serialized_output_bytes_admitted': guard.written}, 'counters': counters}, allow_nan=False))
    return code


def run_pair(command, budget, core, supervisor, observer, metadata, lock, meter):
    start = time.monotonic(); pair = command['comparison']; attempt = CONTROL / pair
    # A failed/partial consumed attempt is never automatically retried or promoted.
    if attempt.exists():
        result = attempt / 'RESULT.json'
        prior = json.loads(result.read_text()) if result.exists() and result.stat().st_size <= 16 * MiB else {}
        if prior.get('status') != 'COMPLETE':
            raise ValueError('Consumed attempt retained; no automatic retry')
        verified = complete_output(command, core, budget)
        if verified is None or verified != prior['analysis']:
            raise ValueError('Previously completed output missing or changed')
        meter.parent(observer.proc_io())
        return {'status': 'ALREADY_COMPLETE_VERIFIED', 'analysis': verified}
    verified = complete_output(command, core, budget)
    if verified:
        meter.parent(observer.proc_io())
        return {'status': 'ALREADY_COMPLETE_VERIFIED', 'analysis': verified}
    attempt.mkdir(exist_ok=False)
    resources = None
    try:
        immutable(attempt / 'ADMITTED.json', {'command': command, 'commands': COMMANDS, 'budget_preparation': BUDGET,
            'adapter_sha256': file_hash(Path(__file__).absolute()), 'core': CORE,
            'limits': PAIR_LIMITS, 'total_whole_process_read_cap_bytes': TOTAL_READ,
            'output_cap_bytes': OUTPUT_LIMIT, 'automatic_retry': False, 'started_unix': time.time()})
        env = {**os.environ, **command['env'], 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': str(RUNTIME),
               'ST_EVAL_CHILD_LOCK_FD': str(lock.fileno())}
        baseline = observer.proc_io(); remaining = meter.parent(baseline)
        if remaining <= 0: raise ValueError('No remaining whole-process read allowance')
        core.BOOT_IO = baseline; core.START = start
        resources = core.monitor([PYTHON, str(Path(__file__).absolute()), '--child', pair.split('_minus_')[0]],
            runtime=str(RUNTIME), env=env, fd=lock.fileno(), root=attempt, s=supervisor,
            m=metadata, obs=observer, budget={**PAIR_LIMITS, 'read_bytes': remaining})
        invocation_charge = meter.reaped(core, resources['read_accounting'], observer.proc_io())
        child_result = (resources.get('child_accounting') or {}).get('cli_result') or {}
        if resources['returncode'] or resources['failure'] is not None or child_result.get('exit_code') != 0:
            raise ValueError('Supervised analysis failed; retained output, no automatic retry')
        verified = complete_output(command, core, budget)
        if verified is None or time.monotonic() - start > PAIR_LIMITS['wall_seconds']:
            raise ValueError('Incomplete output or final wall limit')
        meter.parent(observer.proc_io())
        result = {'status': 'COMPLETE', 'analysis': verified, 'resources': resources,
                  'charged_invocation_rchar_bytes': invocation_charge, 'cumulative_whole_process_rchar_bytes': meter.total,
                  'charge_scope_end': 'after output validation and immediately before final control serialization/publication',
                  'automatic_retry': False, 'no_new_training_or_scoring_authorized': True}
        immutable(attempt / 'RESULT.json', result)
        return {'status': 'COMPLETE', 'analysis': verified, 'charged_cumulative_rchar_bytes': meter.total}
    except BaseException as exc:
        # Unknown/incomplete accounting consumes all remaining allowance; never retry.
        meter.total = max(meter.total, TOTAL_READ)
        if not (attempt / 'RESULT.json').exists():
            immutable(attempt / 'RESULT.json', {'status': 'FAILED_NO_AUTO_RETRY', 'error_type': type(exc).__name__,
                'error': str(exc)[:2048], 'resources': resources, 'charged_cumulative_rchar_bytes': meter.total,
                'failure_charge_is_at_least_full_budget': True, 'retained_outputs': True, 'automatic_retry': False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Lead only, after review of this exact fixed adapter')
    parser.add_argument('--child', choices=('C2', 'C3'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child: return child(args.child)
    if not args.execute: parser.print_help(); return 0
    core, supervisor, observer = dependencies(); commands, budget = fixed_commands(core)
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=RUNTIME, text=True).strip() != SOURCE or subprocess.check_output(['git', 'status', '--porcelain'], cwd=RUNTIME).strip():
        raise ValueError('Frozen analysis runtime changed')
    for pin in [budget['analysis_source'], *budget['source_files'].values()]: core.pinned(pin, decode=False)
    os.sched_setaffinity(0, [0, 1])
    metadata = rss_monitor(core)
    safe(CONTROL); CONTROL.mkdir(parents=True, exist_ok=True)
    with (CONTROL / 'RUN.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        meter = ReadMeter(); results = []
        for command in commands:
            results.append(run_pair(command, budget, core, supervisor, observer, metadata, lock, meter))
        meter.parent(observer.proc_io())
        print(json.dumps({'status': 'COMPLETE', 'pairs': results, 'charged_whole_process_rchar_bytes': meter.total,
                          'gpu': False, 'scoring_calls': 0, 'automatic_retry': False}))
    return 0


if __name__ == '__main__': raise SystemExit(main())
