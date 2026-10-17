#!/usr/bin/env python3
"""One fixed C4-minus-C0 diagnostic; only Lead runs --execute after review."""
with open('/proc/self/io') as _stream:
    BOOT_IO = {k: int(v) for k, v in (line.split(':') for line in _stream)}
import argparse
from contextlib import redirect_stdout
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import traceback

BASE = Path('/mnt/project/VLA-Drive-st-c4-dev-preparation-20261017/reports/st_world_policy/20261016/evaluation')
CONTROL = Path('/mnt/project/st-world-policy-20261016/audits/evaluation/FULL_PAYLOAD_C4_1024_v1')
WRAPPER = {'path': str(BASE / 'FULL_PAYLOAD_C2_C3_1024_EXECUTION_v2/run_fixed_pairs.py'),
           'sha256': '2a47c91254707d3390bb74f92516b9cb5f48bf64d2365d2e874c81a6064a0c5f'}
COMMANDS = {'path': str(BASE / 'full_dev_C4_execution_v1/COMMANDS.json'),
            'sha256': '334e8735f45da2f2fbf70d3b27e4e2cdabfdfa070c4549443533a3d57d125879'}
BINDINGS = {'path': str(BASE / 'FULL_PAYLOAD_C4_1024_PREPARATION_v1/BINDINGS.json'),
            'sha256': '9d0e9d049e5dd1fc7ce2173f7318b0c901cf97665e978c5ca209e67442eab287'}
PLAN_SHA = 'dddf0077c272e1310b873321006d469fcc261f82620ef6804135067bc40fca6c'
CPU_AFFINITY = [6, 7]  # Keep the live metadata writer's CPUs 0/1 available.


class RoutedSubprocess:
    """Route only the fixed monitor taskset prefix; leave its loop unchanged."""
    def __getattr__(self, name):
        return getattr(subprocess, name)

    def Popen(self, command, *args, **kwargs):
        if not isinstance(command, list) or command[:3] != ['taskset', '-c', '0,1']:
            raise ValueError('Expected the reviewed monitor taskset prefix')
        return subprocess.Popen(['taskset', '-c', ','.join(map(str, CPU_AFFINITY)),
                                 *command[3:]], *args, **kwargs)


def route_monitor(core):
    # This is a private dynamically imported module, never a shared source edit.
    core.subprocess = RoutedSubprocess()


def original():
    path = Path(WRAPPER['path'])
    if any(p.is_symlink() for p in (path, *path.parents)) or path.stat().st_size > 64 * 1024:
        raise ValueError('Exact small reviewed wrapper required')
    if hashlib.sha256(path.read_bytes()).hexdigest() != WRAPPER['sha256']:
        raise ValueError('Reviewed wrapper changed')
    spec = importlib.util.spec_from_file_location('c4_original_wrapper', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.PAIR_LIMITS = {**module.PAIR_LIMITS, 'cpu_affinity': CPU_AFFINITY.copy()}
    return module


def fixed_command(module, core):
    original_commands, budget = core.pinned(COMMANDS), core.pinned(BINDINGS)
    matches = [c for c in original_commands['paired_analyses']
               if c['comparison'] == 'C4_minus_C0' and c['stage'] == 'DIAGNOSTIC_1024']
    if len(matches) != 1: raise ValueError('Only the original C4 1024 command')
    command = matches[0]
    if (command['analysis_plan']['sha256'] != PLAN_SHA or command['cwd'] != str(module.RUNTIME)
            or command['argv'][:3] != [module.PYTHON, '-m', 'tools.st_world_policy.evaluate_campaign']
            or command['env']['CUDA_VISIBLE_DEVICES'] != '' or command['max_seconds'] != 1800
            or budget['schema'] != 'st_fixed_c4_payload_analysis_bindings_v1'
            or budget['source_sha'] != module.SOURCE or set(budget['runs']) != {'C0', 'C4'}):
        raise ValueError('Fixed source/command/pair changed')
    expected = {'comparison': 'C4_minus_C0', 'stage': 'DIAGNOSTIC_1024', 'scope': 'FORMAL',
        'plan': command['analysis_plan']['path'], 'population': budget['population']['path'],
        'output': '/mnt/project/st-world-policy-20261016/formal/evaluation/analyses/C4_minus_C0_DIAGNOSTIC_1024_dev1696_v1'}
    for side, arm in [('control', 'C0'), ('treatment', 'C4')]:
        root = '/mnt/project/st-world-policy-20261016/formal/evaluation/jobs/' + arm + '_step001024_dev1696_seed42_v1'
        expected[side + '-score'], expected[side + '-export'] = root + '/official_score', root + '/export'
    if len(command['argv']) != 3 + 2 * len(expected) or any(module.option(command,k) != v for k,v in expected.items()):
        raise ValueError('Fixed C0/C4 endpoint or population changed')
    plan = core.pinned(command['analysis_plan'])
    if (plan['statistics'] != budget['statistics_unchanged'] or plan['populations']['dev']['manifest'] != budget['population']
            or plan['populations']['dev']['scenes'] != 1696 or plan['populations']['dev']['logs'] != 16):
        raise ValueError('Original full dev/statistical definitions required')
    return command, budget


def verify_inputs(module, core, command, budget):
    if module.safe(module.option(command, 'output')).exists():
        raise ValueError('C4 analysis output must be absent; no overwrite/repeat')
    for arm, refs in budget['runs'].items():
        score, export, complete = (core.pinned(refs[k]) for k in ('score_identity','export_identity','score_complete'))
        if (complete.get('status') != 'COMPLETE' or complete.get('requested_scenes') != 1696
                or complete.get('valid_scored') != 1696 or complete.get('invalid') != 0 or complete.get('unresolved') != 0
                or complete['identity'] != score['identity'] or score['export_identity'] != export['identity']
                or score['export_file_sha256'] != refs['export_identity']['sha256']
                or score['checkpoint'] != export['checkpoint'] or export['checkpoint']['completed'] != 1024
                or '/' + arm + '_fulltrain_' not in export['checkpoint']['run'] or export['source_sha'] != module.SOURCE):
            raise ValueError('Actual full 1024 inputs changed')


def child():
    # The reviewed child body is retained; only its fixed C2/C3 command-selection
    # prologue is replaced by this single pinned C4 command. No CLI logic changes.
    os.fstat(int(os.environ['ST_EVAL_CHILD_LOCK_FD']))
    if os.getpid() != os.getpgrp() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Only this invocation owned CPU session')
    module = original(); core, _, obs = module.dependencies(); command, _ = fixed_command(module, core)
    guard = module.OutputGuard(module.option(command, 'output'), module.allowed_payloads(command, core))
    guard.enforce_writes = False
    sys.addaudithook(guard.audit)
    sys.path.insert(0, str(module.RUNTIME)); sys.dont_write_bytecode = True
    stdout, code, error = module.SmallText(), 0, None
    try:
        module.preload_analysis_imports()
        guard.enforce_writes = True
        sys.argv = ['tools.st_world_policy.evaluate_campaign', *command['argv'][3:]]
        with guard.writers(), redirect_stdout(stdout):
            try: runpy.run_module('tools.st_world_policy.evaluate_campaign', run_name='__main__', alter_sys=True)
            except SystemExit as exc: code = 0 if exc.code is None else int(exc.code)
        if code == 0: module.output_size(Path(module.option(command, 'output')))
    except BaseException as exc:
        code = 1; error = {'type': type(exc).__name__, 'message': str(exc)[:2048]}; traceback.print_exc()
    class NoCUDA:
        @staticmethod
        def is_initialized(): return False
    class CPUOnly: cuda = NoCUDA
    counter = obs.Counters(Path('/unused-ram-only'), CPUOnly, module.TOTAL_READ, BOOT_IO)
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


def run():
    module = original(); core, supervisor, observer = module.dependencies()
    route_monitor(core)
    command, budget = fixed_command(module, core)
    verify_inputs(module, core, command, budget)
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=module.RUNTIME, text=True).strip() != module.SOURCE
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=module.RUNTIME).strip()):
        raise ValueError('Frozen analyzer runtime changed')
    for pin in [budget['analysis_source'], *budget['source_files'].values()]: core.pinned(pin, decode=False)
    os.sched_setaffinity(0, CPU_AFFINITY); metadata = module.rss_monitor(core)
    module.safe(CONTROL); CONTROL.mkdir(parents=True, exist_ok=True)
    with (CONTROL / 'RUN.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # run_pair dispatches this C4 child front end; guards/monitor/verification
        # remain the pinned original functions, with only these explicit bindings.
        module.CONTROL, module.COMMANDS, module.BUDGET = CONTROL, COMMANDS, BINDINGS
        module.__file__ = str(Path(__file__).absolute())
        module.session_state([command])
        if Path(module.option(command, 'output')).exists(): raise ValueError('C4 output appeared before admission')
        module.immutable(CONTROL / 'SESSION_STARTED.json', {
            'schema': 'st_fixed_c4_analysis_once_v1', 'wrapper': WRAPPER, 'core': module.CORE,
            'adapter': {'path': str(Path(__file__).absolute()), 'sha256': module.file_hash(Path(__file__).absolute())},
            'commands': COMMANDS, 'bindings': BINDINGS, 'comparison': 'C4_minus_C0', 'stage': 'DIAGNOSTIC_1024',
            'limits': module.PAIR_LIMITS, 'read_cap_bytes': module.TOTAL_READ,
            'output_cap_including_controls_bytes': module.OUTPUT_LIMIT, 'max_invocations': 1,
            'C2_C3_reruns': False, 'automatic_retry': False}, cap=16 * 1024)
        meter = module.ReadMeter()
        try:
            pair = module.run_pair(command, budget, core, supervisor, observer, metadata, lock, meter)
            meter.parent(observer.proc_io())
            result = {'status': 'COMPLETE', 'pair': pair, 'charged_whole_process_rchar_bytes': meter.total,
                      'max_invocations': 1, 'C2_C3_reruns': False, 'automatic_retry': False}
            module.immutable(CONTROL / 'SESSION_TERMINAL.json', result, cap=16 * 1024)
            print(json.dumps(result)); return 0
        except BaseException as exc:
            if not (CONTROL / 'SESSION_TERMINAL.json').exists():
                module.immutable(CONTROL / 'SESSION_TERMINAL.json', {
                    'status': 'FAILED_NO_AUTO_RETRY', 'error': str(exc)[:2048],
                    'conservative_spent_bytes': max(module.TOTAL_READ, meter.total),
                    'max_invocations': 1, 'C2_C3_reruns': False, 'automatic_retry': False}, cap=16 * 1024)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Lead only after independent review of this fixed C4 delta')
    parser.add_argument('--child', choices=['C4'], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child: return child()
    if not args.execute: parser.print_help(); return 0
    return run()


if __name__ == '__main__': raise SystemExit(main())
