#!/usr/bin/env python3
"""One explicitly authorized C3-only retry; preparation does not dispatch it."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

MiB = 1 << 20
OLD = Path('/mnt/project/st-world-policy-20261016/audits/evaluation/FULL_PAYLOAD_C2_C3_1024_EXECUTION_v2')
NEW = OLD.with_name('FULL_PAYLOAD_C3_1024_RETRY_v1')
WRAPPER = {
    'path': '/mnt/project/VLA-Drive-st-c4-dev-preparation-20261017/reports/st_world_policy/20261016/evaluation/FULL_PAYLOAD_C2_C3_1024_EXECUTION_v2/run_fixed_pairs.py',
    'sha256': '2a47c91254707d3390bb74f92516b9cb5f48bf64d2365d2e874c81a6064a0c5f',
}
PRIOR_CHARGE = 1103776571
NEW_ALLOWANCE = 1 << 30
PRIOR = {
    'session': {'path': str(OLD / 'SESSION_TERMINAL.json'), 'sha256': 'c0c07203570bec9a87c4955379e3140d44d639649a2ad67bb1b1f084d7c94848'},
    'C3': {'path': str(OLD / 'C3_minus_C0/RESULT.json'), 'sha256': '9695527dc0378624f8f42399c43a11ca4784aedaa32a5d490eaa51247e628df3'},
    'C2': {'path': str(OLD / 'C2_minus_C0/RESULT.json'), 'sha256': '752c5c916c53e5928dba92deb0e96b47b0ea05d335c8476cb6d3f98a97fa51f7'},
}


def original():
    path = Path(WRAPPER['path'])
    if any(p.is_symlink() for p in (path, *path.parents)) or path.stat().st_size > 64 * 1024:
        raise ValueError('Exact small reviewed wrapper required')
    if hashlib.sha256(path.read_bytes()).hexdigest() != WRAPPER['sha256']:
        raise ValueError('Reviewed 2a47 wrapper changed')
    spec = importlib.util.spec_from_file_location('c3_retry_original_wrapper', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    if module.TOTAL_READ != NEW_ALLOWANCE or module.CONTROL != OLD:
        raise ValueError('Original wrapper scope changed')
    return module


def prepare(module, core):
    commands, budget = module.fixed_commands(core)
    if [c['comparison'] for c in commands] != ['C2_minus_C0', 'C3_minus_C0']:
        raise ValueError('Only the original second command may be selected')
    command = commands[1]
    if command['stage'] != 'DIAGNOSTIC_1024' or module.option(command, 'scope') != 'FORMAL':
        raise ValueError('Fixed C3 diagnostic required')
    prior = {k: core.pinned(pin) for k, pin in PRIOR.items()}
    if (prior['session'].get('status') != 'FAILED_NO_AUTO_RETRY'
            or prior['session'].get('conservative_spent_bytes') != PRIOR_CHARGE
            or prior['C3'].get('status') != 'FAILED_NO_AUTO_RETRY'
            or prior['C3'].get('charged_cumulative_rchar_bytes') != PRIOR_CHARGE
            or prior['C3']['resources'].get('failure') != 'PARENT_CHILD_READ_LIMIT'
            or prior['C3']['resources'].get('returncode') != -15
            or prior['C3']['resources'].get('child_accounting') is not None
            or prior['C2'].get('status') != 'COMPLETE'):
        raise ValueError('Exact retained failed-session lineage required')
    for arm in ('C2', 'C3'):
        if prior[arm]['resources']['cleanup'].get('live_owned_remaining') != 0:
            raise ValueError('Prior owned cleanup incomplete')
    output = module.safe(module.option(command, 'output'))
    if output.exists():
        raise ValueError('Original C3 output must be absent; no overwrite or second evaluation')
    return command, budget


@contextmanager
def old_session_inactive(module):
    # Original parent and child inherited this flock. Fresh acquisition plus
    # pinned zero-remnant receipts is the boundary; never signal historical PIDs.
    with module.safe(OLD / 'RUN.lock').open('rb') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try: yield
        finally: fcntl.flock(lock, fcntl.LOCK_UN)


def run():
    module = original(); core, supervisor, observer = module.dependencies()
    command, budget = prepare(module, core)
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=module.RUNTIME, text=True).strip() != module.SOURCE
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=module.RUNTIME).strip()):
        raise ValueError('Frozen analysis runtime changed')
    for pin in [budget['analysis_source'], *budget['source_files'].values()]: core.pinned(pin, decode=False)
    os.sched_setaffinity(0, [0, 1]); metadata = module.rss_monitor(core)
    module.safe(NEW)
    with old_session_inactive(module):
        NEW.mkdir(parents=True, exist_ok=True)
        with (NEW / 'RUN.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            module.CONTROL = NEW  # only the imported wrapper's small control root changes
            module.session_state([command])
            # Check absence again under both locks before consuming this new session.
            prepare(module, core)
            module.immutable(NEW / 'SESSION_STARTED.json', {
                'schema': 'st_fixed_c3_retry_once_v1', 'wrapper': WRAPPER, 'core': module.CORE,
                'adapter': {'path': str(Path(__file__).absolute()), 'sha256': module.file_hash(Path(__file__).absolute())},
                'prior_evidence': PRIOR, 'prior_campaign_read_charge_bytes': PRIOR_CHARGE,
                'new_single_invocation_rchar_allowance_bytes': NEW_ALLOWANCE,
                'declared_campaign_read_ceiling_bytes': PRIOR_CHARGE + NEW_ALLOWANCE,
                'comparison': command['comparison'], 'stage': command['stage'],
                'original_command_unchanged': command, 'max_invocations': 1,
                'C2_rerun': False, 'automatic_retry': False,
                'output_limit_including_controls_bytes': module.OUTPUT_LIMIT,
                'limits': module.PAIR_LIMITS,
                'fresh_old_flock_acquired': True, 'prior_cleanup_live_owned_remaining': 0,
                'scope': 'new explicit allowance, not continuation/reset of failed session'}, cap=16 * 1024)
            meter = module.ReadMeter()  # new explicit allowance; old charge remains separately declared
            try:
                pair = module.run_pair(command, budget, core, supervisor, observer, metadata, lock, meter)
                meter.parent(observer.proc_io())
                result = {'status': 'COMPLETE', 'pair': pair, 'prior_evidence': PRIOR,
                    'prior_campaign_read_charge_bytes': PRIOR_CHARGE,
                    'new_invocation_rchar_bytes': meter.total,
                    'declared_campaign_read_charge_bytes': PRIOR_CHARGE + meter.total,
                    'charge_scope': 'sum of original recorded boundary and new recorded boundary, not a reset',
                    'max_invocations': 1, 'automatic_retry': False, 'C2_rerun': False}
                module.immutable(NEW / 'SESSION_TERMINAL.json', result, cap=16 * 1024)
                print(json.dumps(result))
                return 0
            except BaseException as exc:
                if not (NEW / 'SESSION_TERMINAL.json').exists():
                    charge = max(NEW_ALLOWANCE, meter.total)
                    module.immutable(NEW / 'SESSION_TERMINAL.json', {
                        'status': 'FAILED_NO_AUTO_RETRY', 'error': str(exc)[:2048],
                        'prior_evidence': PRIOR, 'prior_campaign_read_charge_bytes': PRIOR_CHARGE,
                        'new_invocation_conservative_charge_bytes': charge,
                        'declared_campaign_read_charge_bytes': PRIOR_CHARGE + charge,
                        'automatic_retry': False, 'C2_rerun': False}, cap=16 * 1024)
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Lead only after review of this exact one-retry delta')
    args = parser.parse_args()
    if not args.execute: parser.print_help(); return 0
    return run()


if __name__ == '__main__': raise SystemExit(main())
