#!/usr/bin/env python3
"""Bind, then run ONE original COMMON_ENDPOINT pair. Lead dispatch only.

No scientific implementation: imports the reviewed fixed-pair monitor/child.
This frontend changes only fixed argv selection and 4096 completion bindings.
"""
with open('/proc/self/io') as _stream:
    BOOT_IO = {k: int(v) for k, v in (line.split(':') for line in _stream)}
import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode = True

HERE = Path(__file__).absolute().parent
FIXED = {'path': str(HERE / 'FIXED_COMMANDS.json'),
         'sha256': '4da48038c29d6473fe2b5a09f3e7629293a577e0aeafb19ae10449c1a5c34673'}
PREVIOUS = Path('/mnt/project/VLA-Drive-st-c4-dev-preparation-20261017/reports/st_world_policy/20261016/evaluation')
WRAPPER = {'path': str(PREVIOUS / 'FULL_PAYLOAD_C2_C3_1024_EXECUTION_v2/run_fixed_pairs.py'),
           'sha256': '2a47c91254707d3390bb74f92516b9cb5f48bf64d2365d2e874c81a6064a0c5f'}
CHILD_FRONT = {'path': '/mnt/project/VLA-Drive-st-world-policy-20261016/reports/st_world_policy/20261016/evaluation/FULL_PAYLOAD_C4_1024_PREPARATION_v1/run_c4_once.py',
               'sha256': 'c4806201c46caee4a770545fa6e00764a4b5da5ba65ea97b389268b08be14c5f'}
SOURCE_BINDINGS = {'path': str(PREVIOUS / 'FULL_PAYLOAD_C4_1024_PREPARATION_v1/BINDINGS.json'),
                   'sha256': '9d0e9d049e5dd1fc7ce2173f7318b0c901cf97665e978c5ca209e67442eab287'}
CONTROL_ROOT = Path('/mnt/project/st-world-policy-20261016/audits/evaluation/COMMON4096_FIXED_PAIRS_v1')
ARMS = ('C1', 'C2', 'C3', 'C4')


def load_source(pin, name):
    path = Path(pin['path'])
    if (not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents))
            or path.stat().st_size > 64 * 1024
            or hashlib.sha256(path.read_bytes()).hexdigest() != pin['sha256']):
        raise ValueError('Exact reviewed small source required')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def original():
    module = load_source(WRAPPER, 'common4096_reviewed_wrapper')
    # The old pin pointed at Lead's moving worktree. Use its byte-identical
    # retained source, keeping the original reviewed checksum and monitor code.
    module.CORE = {**module.CORE, 'path': '/mnt/project/VLA-Drive-st-c4-cpu6-execution-20261017/reports/st_world_policy/20261017/full_future_metadata_preparation_v1/run_build_once.py'}
    module.complete_output = lambda command, core, budget: complete_output(module, command, core, budget)
    return module


def fixed_command(module, core, arm):
    if arm not in ARMS: raise ValueError('Only the four preregistered arms')
    doc = core.pinned(FIXED); item = doc['arms'][arm]
    source = core.pinned(item['source_commands'])
    matches = [c for c in source['paired_analyses']
               if c['comparison'] == arm + '_minus_C0' and c['stage'] == 'COMMON_ENDPOINT']
    if matches != [item['original_command_unchanged']]:
        raise ValueError('Original preregistered command changed')
    command = dict(matches[0])
    # C1's original argv predates this metadata field. Adding its existing plan
    # reference satisfies run_pair's receipt API; argv/env/cwd remain unchanged.
    command['analysis_plan'] = item['plan']
    expected = {'comparison': arm + '_minus_C0', 'stage': 'COMMON_ENDPOINT', 'scope': 'FORMAL',
        'plan': item['plan']['path'],
        'population': '/mnt/project/st-world-policy-20261016/audits/history_data/full_current_population_v1/dev/SELECTED_SCENES.json',
        'output': '/mnt/project/st-world-policy-20261016/formal/evaluation/analyses/' + arm + '_minus_C0_COMMON_ENDPOINT_dev1696_v1'}
    for side, name in (('control', 'C0'), ('treatment', arm)):
        root = '/mnt/project/st-world-policy-20261016/formal/evaluation/jobs/' + name + '_step004096_dev1696_seed42_v1'
        expected[side + '-score'], expected[side + '-export'] = root + '/official_score', root + '/export'
    if (command['argv'][:3] != [module.PYTHON, '-m', 'tools.st_world_policy.evaluate_campaign']
            or len(command['argv']) != 3 + 2 * len(expected)
            or any(module.option(command, k) != v for k, v in expected.items())
            or command['cwd'] != str(module.RUNTIME) or command['max_seconds'] != 1800
            or command['env']['CUDA_VISIBLE_DEVICES'] != '' or doc['source_sha'] != module.SOURCE):
        raise ValueError('Fixed COMMON4096 command/source changed')
    return command, core.pinned(item['plan'])


def small_pin(module, path):
    path = module.safe(path)
    if path.stat().st_size > 4 << 20: raise ValueError('Only small JSON receipts may be bound')
    return {'path': str(path), 'sha256': module.file_hash(path)}


def verify_inputs(module, core, command, budget):
    if set(budget['runs']) != {'C0', command['comparison'].split('_minus_')[0]}:
        raise ValueError('Wrong fixed pair')
    for side, arm in (('control', 'C0'), ('treatment', command['comparison'].split('_minus_')[0])):
        refs = budget['runs'][arm]
        score, export, complete = (core.pinned(refs[k]) for k in ('score_identity','export_identity','score_complete'))
        if (refs['score_identity']['path'] != module.option(command, side + '-score') + '/IDENTITY.json'
                or refs['export_identity']['path'] != module.option(command, side + '-export') + '/IDENTITY.json'
                or refs['score_complete']['path'] != module.option(command, side + '-score') + '/COMPLETE.json'
                or complete.get('status') != 'COMPLETE' or complete.get('requested_scenes') != 1696
                or complete.get('valid_scored') != 1696 or complete.get('invalid') != 0 or complete.get('unresolved') != 0
                or complete['identity'] != score['identity'] or score['export_identity'] != export['identity']
                or score['export_file_sha256'] != refs['export_identity']['sha256']
                or score['checkpoint'] != export['checkpoint'] or export['checkpoint']['completed'] != 4096
                or '/' + arm + '_fulltrain_' not in export['checkpoint']['run'] or export['source_sha'] != module.SOURCE):
            raise ValueError('WAITING_INPUTS or changed complete COMMON4096 identity')


def read_binding(module, core, arm):
    command, plan = fixed_command(module, core, arm)
    pin = small_pin(module, CONTROL_ROOT / arm / 'BINDINGS.json')
    budget = core.pinned(pin)
    if (budget['schema'] != 'st_common4096_actual_bindings_v1'
            or budget['adapter_sha256'] != module.file_hash(Path(__file__).absolute())
            or budget['fixed_commands'] != FIXED or budget['source_sha'] != module.SOURCE
            or budget['stage'] != 'COMMON_ENDPOINT' or budget['comparison'] != command['comparison']
            or plan['statistics'] != budget['statistics_unchanged']
            or plan['populations']['dev']['manifest'] != budget['population']
            or plan['populations']['dev']['scenes'] != 1696 or plan['populations']['dev']['logs'] != 16):
        raise ValueError('Fixed source/statistics/population binding changed')
    verify_inputs(module, core, command, budget)
    return command, budget, pin


def bind(arm):
    module = original(); core, _, observer = module.dependencies()
    command, plan = fixed_command(module, core, arm)
    control = module.safe(CONTROL_ROOT / arm)
    if (control / 'BINDINGS.json').exists():
        _, _, pin = read_binding(module, core, arm)
        print(json.dumps({'status': 'BOUND_REUSED', 'bindings': pin})); return 0
    if Path(module.option(command, 'output')).exists(): raise ValueError('Analysis output already exists; no reanalysis')
    old = core.pinned(SOURCE_BINDINGS)
    budget = {k: old[k] for k in ('analysis_source','source_files','source_sha','population','statistics_unchanged')}
    budget.update(schema='st_common4096_actual_bindings_v1', stage='COMMON_ENDPOINT', comparison=command['comparison'],
                  fixed_commands=FIXED, adapter_sha256=module.file_hash(Path(__file__).absolute()),
                  statistics_unchanged=plan['statistics'], runs={}, launch_authorized=False)
    for side, name in (('control', 'C0'), ('treatment', arm)):
        budget['runs'][name] = {
            'score_identity': small_pin(module, module.option(command, side + '-score') + '/IDENTITY.json'),
            'export_identity': small_pin(module, module.option(command, side + '-export') + '/IDENTITY.json'),
            'score_complete': small_pin(module, module.option(command, side + '-score') + '/COMPLETE.json')}
    verify_inputs(module, core, command, budget)
    if plan['populations']['dev']['manifest'] != budget['population']: raise ValueError('Original population changed')
    budget['binding_process_rchar_bytes'] = observer.proc_io()['rchar']
    if budget['binding_process_rchar_bytes'] >= module.TOTAL_READ: raise ValueError('Binding consumed read allowance')
    control.mkdir(parents=True, exist_ok=True)
    with (control / 'RUN.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        module.CONTROL = control; module.session_state([command])
        sha = module.immutable(control / 'BINDINGS.json', budget, cap=16 * 1024)
    print(json.dumps({'status': 'BOUND_NOT_DISPATCHED', 'path': str(control / 'BINDINGS.json'), 'sha256': sha})); return 0


def complete_output(module, command, core, budget):
    # Verbatim original checks with only the fixed stage literal changed.
    root = module.safe(module.option(command, 'output'))
    if not root.exists(): return None
    size = module.output_size(root)
    if {p.name for p in root.iterdir()} != {'ANALYSIS.json', 'PAIRED_SCENES.csv'}:
        raise ValueError('Partial/foreign output retained; no retry or overwrite')
    path = root / 'ANALYSIS.json'
    if path.stat().st_size > 4 * module.MiB: raise ValueError('Analysis receipt too large')
    record = json.loads(path.read_text())
    if (record.get('schema') != 'st_world_paired_analysis_v1' or record.get('status') != 'COMPLETE'
            or record.get('scope') != 'FORMAL' or record.get('comparison') != command['comparison']
            or record.get('stage') != 'COMMON_ENDPOINT'
            or record.get('validity') != {'requested': 1696, 'paired_valid': 1696, 'invalid': 0, 'unresolved': 0}
            or record.get('plan_sha256') != command['analysis_plan']['sha256']
            or record.get('population_sha256') != budget['population']['sha256']
            or record.get('statistics_definition') != budget['statistics_unchanged']
            or record.get('analysis_source_sha256') != budget['analysis_source']['sha256']
            or record.get('paired_scenes_csv_sha256') != module.file_hash(root / 'PAIRED_SCENES.csv')):
        raise ValueError('Existing output binding differs')
    treatment = command['comparison'].split('_minus_')[0]
    for side, arm in (('control', 'C0'), ('treatment', treatment)):
        bound = budget['runs'][arm]
        score, export = core.pinned(bound['score_identity']), core.pinned(bound['export_identity'])
        complete = core.pinned(bound['score_complete'])
        if (record[side]['score_identity'] != score['identity'] or record[side]['export_identity'] != export['identity']
                or record[side]['checkpoint'] != export['checkpoint']
                or record[side]['scenes_csv_sha256'] != complete['scenes_csv_sha256']
                or record[side]['score_root'] != module.option(command, side + '-score')
                or record[side]['export_root'] != module.option(command, side + '-export')):
            raise ValueError('Existing result has foreign inputs')
    selected = core.pinned(budget['population'])
    expected = {x['token']: x['log'] for x in selected['scenes']}
    with (root / 'PAIRED_SCENES.csv').open(newline='') as stream:
        reader = csv.DictReader(stream); rows = list(reader)
    if len(rows) != 1696 or {r['scene_id']: r['source_log'] for r in rows} != expected:
        raise ValueError('Existing paired output population differs')
    if not {'control_raw_progress_m', 'treatment_mean_executed_speed_mps', 'new_stop_proxy'}.issubset(reader.fieldnames):
        raise ValueError('Stopping diagnostics absent')
    return {'path': str(path), 'sha256': module.file_hash(path), 'output_bytes': size}


def child(arm):
    # Reuse the entire reviewed CPU child and output guard; only command lookup
    # is injected. No simulator, model forward or new statistics are introduced.
    front = load_source(CHILD_FRONT, 'common4096_reviewed_child')
    front.BOOT_IO = BOOT_IO
    front.original = original
    front.fixed_command = lambda module, core: read_binding(module, core, arm)[:2]
    return front.child()


def run(arm):
    module = original(); core, supervisor, observer = module.dependencies()
    command, budget, binding_pin = read_binding(module, core, arm)
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=module.RUNTIME, text=True).strip() != module.SOURCE
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=module.RUNTIME).strip()):
        raise ValueError('Frozen analyzer runtime changed')
    for pin in [budget['analysis_source'], *budget['source_files'].values()]: core.pinned(pin, decode=False)
    os.sched_setaffinity(0, [0, 1]); metadata = module.rss_monitor(core)
    control = module.safe(CONTROL_ROOT / arm)
    with (CONTROL_ROOT / 'SERIAL.lock').open('a+') as serial, (control / 'RUN.lock').open('a+') as lock:
        fcntl.flock(serial, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        module.CONTROL, module.COMMANDS, module.BUDGET = control, FIXED, binding_pin
        module.__file__ = str(Path(__file__).absolute())
        module.session_state([command])
        if Path(module.option(command, 'output')).exists(): raise ValueError('Output appeared before admission')
        module.immutable(control / 'SESSION_STARTED.json', {
            'schema': 'st_common4096_analysis_once_v1', 'wrapper': WRAPPER, 'child_front': CHILD_FRONT,
            'commands': FIXED, 'bindings': binding_pin, 'comparison': command['comparison'], 'stage': 'COMMON_ENDPOINT',
            'limits': module.PAIR_LIMITS, 'read_cap_including_binding_bytes': module.TOTAL_READ,
            'output_cap_including_controls_bytes': module.OUTPUT_LIMIT, 'max_invocations': 1,
            'automatic_retry': False}, cap=8 * 1024)
        meter = module.ReadMeter(spent=budget['binding_process_rchar_bytes'])
        try:
            pair = module.run_pair(command, budget, core, supervisor, observer, metadata, lock, meter)
            meter.parent(observer.proc_io())
            result = {'status': 'COMPLETE', 'pair': pair, 'charged_whole_process_rchar_bytes': meter.total,
                      'max_invocations': 1, 'automatic_retry': False}
            module.immutable(control / 'SESSION_TERMINAL.json', result, cap=8 * 1024)
            print(json.dumps(result)); return 0
        except BaseException as exc:
            if not (control / 'SESSION_TERMINAL.json').exists():
                module.immutable(control / 'SESSION_TERMINAL.json', {
                    'status': 'FAILED_NO_AUTO_RETRY', 'error': str(exc)[:2048],
                    'conservative_spent_bytes': max(module.TOTAL_READ, meter.total),
                    'max_invocations': 1, 'automatic_retry': False}, cap=8 * 1024)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=ARMS)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--bind', action='store_true', help='Pin completed small receipts only; never runs analysis')
    modes.add_argument('--execute', action='store_true', help='Lead only, one fixed pair after actual binding/review')
    modes.add_argument('--child', choices=ARMS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child: return child(args.child)
    if not (args.bind or args.execute): parser.print_help(); return 0
    if not args.arm: parser.error('--arm is required')
    try: return bind(args.arm) if args.bind else run(args.arm)
    except (ValueError, FileNotFoundError) as exc:
        print(json.dumps({'status': 'BLOCKED_OR_WAITING_INPUTS', 'error': str(exc)})); return 2


if __name__ == '__main__': raise SystemExit(main())
