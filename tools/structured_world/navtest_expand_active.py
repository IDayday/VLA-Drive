"""Add qualified workers to an active leased Navtest bank without restarting it.

The frozen dispatcher waits a fixed list of launchers. Park only that CPU
dispatcher while supplemental workers drain; its original GPU workers and CPU
scorer keep running. A separate bounded watchdog always resumes that exact
dispatcher. No model, original worker, lease or prediction is modified.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


def process_matches(pid, start, tool, output):
    root = Path('/proc') / str(pid)
    try:
        fields = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        argv = (root / 'cmdline').read_bytes().decode().split('\0')
    except (OSError, UnicodeError):
        return False
    return (fields[19] == str(start) and fields[0] != 'Z' and str(tool) in argv
            and '--serve' in argv and '--output' in argv
            and argv[argv.index('--output') + 1] == str(output))


def resume_exact(gate):
    if process_matches(gate['pid'], gate['start'], gate['tool'], gate['output']):
        os.kill(gate['pid'], signal.SIGCONT)


def watchdog(path):
    gate = json.loads(path.read_text())
    while time.time() < gate['deadline']:
        if (path.parent / 'RELEASED.json').exists():
            return
        parent = Path('/proc') / str(gate['expander_pid']) / 'stat'
        try:
            fields = parent.read_text().rsplit(')', 1)[1].split()
            if fields[19] != gate['expander_start'] or fields[0] == 'Z':
                break
        except OSError:
            break
        time.sleep(1)
    # On an unexpected operator failure, stop publication at scene boundaries
    # before resuming the normal coordinator. Existing valid predictions remain.
    failure = Path(gate['destination']) / 'INFERENCE_FAILED.json'
    try:
        with failure.open('x') as stream:
            json.dump({'classification': 'supplemental_dispatch_watchdog',
                       'gate': str(path), 'error': 'Expansion owner exited or deadline exceeded'}, stream)
    except FileExistsError:
        pass
    resume_exact(gate)


def expand(args):
    sys.path.insert(0, str(args.runtime))
    spec = importlib.util.spec_from_file_location('frozen_navtest', args.runtime / 'run_navsim_navtest.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    m.authorized()
    req = json.loads((args.output / 'requests' / (args.key + '.json')).read_text())
    destination = args.output / 'jobs' / args.key
    controller = json.loads((args.output / 'CONTROLLER.json').read_text())
    assert controller['host'] == socket.gethostname(), 'Run alongside the owning CPU dispatcher'
    pid = controller['pid']
    start = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()[19]
    assert process_matches(pid, start, controller['tool'], args.output)
    assert controller['tool'] == str(m.SCRIPT)
    assert not (destination / 'INFERENCE_FAILED.json').exists()
    if (destination / 'COMPLETE.json').exists():
        raise RuntimeError('Inference already complete; do not allocate redundant GPUs')
    state = json.loads((args.output / 'STATE.json').read_text())
    assert state['status'] == 'INFERENCE_RUNNING' and state['request']['key'] == args.key
    qualified = json.loads(args.qualified_routes.read_text())['routes']
    assert qualified and len({r['uuid'] for r in qualified}) == len(qualified)
    active = json.loads((destination / 'REGISTRATION.json').read_text())['routes']
    assert not ({r['uuid'] for r in active} & {r['uuid'] for r in qualified})
    # Reapply the original capacity/ownership/runtime gate immediately before
    # launch. This never turns unknown occupants into owned training.
    route_args = argparse.Namespace(output=args.output, routes=args.qualified_routes)
    admitted = {r['uuid']: r for r in m.refresh_routes(route_args, req)}
    routes = []
    for old in qualified:
        new = admitted.get(old['uuid'])
        if new and new['host'] == old['host'] and new['python'] == old['python']:
            new['fp32_export'] = old['fp32_export']; routes.append(new)
    if not routes:
        raise RuntimeError('No qualified supplemental GPU remains available')
    for route in routes:
        profile = json.loads((args.output / 'qualification' / args.key / route['host'] /
                              'QUALIFICATION_PASSED.json').read_text())
        assert profile['passed'] and profile['request'] == req
    if (destination / 'COMPLETE.json').exists():
        raise RuntimeError('Primary workers finished during resource checks')
    attempt = m.new_attempt(destination, routes)
    root = destination / 'expansions' / attempt
    gate = dict(pid=pid, start=start, tool=controller['tool'], output=str(args.output),
                destination=str(destination), expander_pid=os.getpid(),
                expander_start=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
                deadline=time.time() + args.timeout, primary_attempt=state['attempt'],
                supplemental_attempt=attempt, routes=routes, registered_utc=m.now())
    m.atomic(root / 'GATE.json', gate)
    processes = []
    paused = False
    try:
        assert process_matches(pid, start, controller['tool'], args.output)
        os.kill(pid, signal.SIGSTOP); paused = True
        with (root / 'watchdog.log').open('a') as stream:
            watcher = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()),
                                       '--watchdog', str(root / 'GATE.json')],
                                      stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        m.atomic(root / 'WATCHDOG.json', dict(pid=watcher.pid, deadline=gate['deadline']))
        # Recheck after parking, closing the small completion race.
        assert not (destination / 'COMPLETE.json').exists()
        launch_args = argparse.Namespace(output=args.output)
        for i, route in enumerate(routes):
            processes.append(m.launch_worker(launch_args, req, destination, route, i,
                                             len(routes), args.inputs, attempt))
        m.atomic(root / 'RUNNING.json', dict(gate=gate, supplemental_workers=len(routes),
                 total_registered_workers=len(active) + len(routes),
                 original_GPU_workers_and_CPU_scorer_continue=True, observed_utc=m.now()))
        codes = [p.wait(timeout=max(1, gate['deadline'] - time.time())) for p in processes]
        if any(codes):
            raise RuntimeError('Supplemental launcher failed: ' + repr(codes))
        reports = [json.loads(p.read_text()) for p in sorted(
            (destination / 'attempts' / attempt / 'shards').glob('*.json'))]
        identity = json.loads((destination / 'identity.json').read_text())['identity']
        assert len(reports) == len(routes)
        assert all(r['status'] == 'COMPLETE' and r['prediction_identity'] == identity for r in reports)
        m.atomic(root / 'COMPLETE.json', dict(gate=gate, exit_codes=codes,
                 all_supplemental_launchers_exited=True, reports=reports,
                 actual_total_registered_workers=len(active) + len(routes), completed_utc=m.now(),
                 semantics='Primary COMPLETE.shards describes its original attempt; attempt_reports/cost include this supplemental attempt'))
    except BaseException as error:
        m.first_failure(destination / 'INFERENCE_FAILED.json', dict(
            classification='supplemental_dispatch_failure', error=repr(error), expansion=str(root), observed_utc=m.now()))
        raise
    finally:
        if paused:
            resume_exact(gate)
            m.atomic(root / 'RELEASED.json', dict(resumed_exact_controller=pid, observed_utc=m.now()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--watchdog', type=Path)
    p.add_argument('--runtime', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--inputs', type=Path)
    p.add_argument('--key')
    p.add_argument('--qualified-routes', type=Path)
    p.add_argument('--timeout', type=int, default=1800)
    args = p.parse_args()
    if args.watchdog:
        watchdog(args.watchdog)
    else:
        assert 60 <= args.timeout <= 3600
        expand(args)


if __name__ == '__main__':
    main()
