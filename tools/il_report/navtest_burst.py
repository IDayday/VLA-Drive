"""High-concurrency scene sharding around qualified, unchanged IL evaluators.

Worker count changes only token partitioning. Model batch=1, token-bound root
noise, single K10 shift5 trajectory and the canonical scorer remain unchanged.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def write(path, value):
    path = Path(path); tmp = path.with_name(path.name+'.tmp.'+str(os.getpid()))
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)


class Pool:
    """Cached administrative probes; each worker still repeats its admission."""
    def __init__(self, delegate):
        self.delegate = delegate; self.sample = None; self.at = 0

    def __getattr__(self, key):
        return getattr(self.delegate, key)

    def available(self, min_free_MiB=32768):
        if self.sample is not None and time.monotonic()-self.at < 15:
            return self.sample
        from starVLA.rl.qwen_world_postcot.resources import probe, gpu_activity_issue
        hosts = self.delegate.allowed_hosts()
        def get(host):
            try:
                return host, probe(host), None
            except Exception as error:
                return host, None, str(error)
        samples, errors = [], {}
        for _ in range(2):
            with ThreadPoolExecutor(max_workers=max(1, len(hosts))) as executor:
                rows = list(executor.map(get, hosts))
            samples.append({h: d for h, d, e in rows if e is None})
            errors.update({h: e for h, d, e in rows if e is not None})
        routes = []; seen = set()
        for host, after in samples[1].items():
            before = samples[0].get(host)
            if not before or [g['uuid'] for g in before['gpus']] != [g['uuid'] for g in after['gpus']]:
                continue
            after['available_memory_bytes'] = after['MemAvailable_bytes']
            for a, b in zip(before['gpus'], after['gpus']):
                if (b['uuid'] in seen or min(a['free_MiB'], b['free_MiB']) < min_free_MiB
                        or not after['worker_python_exists']
                        or any(not o['visible'] for s in (before, after) for o in s['owners'] if o['uuid'] == b['uuid'])
                        or gpu_activity_issue(a, before['owners']) or gpu_activity_issue(b, after['owners'])):
                    continue
                seen.add(b['uuid'])
                routes.append(dict(host=host, gpu=b['index'], uuid=b['uuid'],
                                   free_MiB=min(a['free_MiB'], b['free_MiB'])))
        self.sample = routes, dict(first=samples[0], second=samples[1], errors=errors)
        self.at = time.monotonic()
        return self.sample


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--family', required=True, choices=['four', 'world'])
    parser.add_argument('--workers', type=int, choices=[16, 32], default=32)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); root = args.root.resolve()
    with (root/'owner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root/'owner.json').exists() and not args.resume:
            raise FileExistsError('Existing burst invocation must be reviewed; never duplicate')
        if args.resume:
            previous = json.loads((root/'owner.json').read_text())
            attempt = root/'operator_attempts'/str(previous['pid']); attempt.mkdir(parents=True, exist_ok=False)
            (attempt/'owner.json').write_bytes((root/'owner.json').read_bytes())
            if (root/'failure.json').exists():
                (root/'failure.json').rename(attempt/'failure.json')
        sys.path.insert(0, str(args.source.resolve()))
        if args.family == 'four':
            from tools.qwen_world_compare.interim_owner import InterimRequest as Qualified
        else:
            from tools.qwen_world_postcot.latest_snapshot_owner import SnapshotRequest as Qualified
        from tools.qwen_world_postcot.user_interim_owner import Request as Base
        dispatch_lock = Path('/mnt/project/runtime_periodic_navtest_dispatch.lock')

        class Burst(Qualified):
            def done(self, name):
                if args.family == 'four': return super().done(name)
                if name not in self.jobs: return None
                path = Path(self.jobs[name]['receipt_dir'])/'result.json'
                if not path.exists(): return None
                result = json.loads(path.read_text())
                return result.get('status') == 'PROCESS_COMPLETE' and result.get('exit_code') == 0

            def route(self, routes, inventory, *, parity=False):
                guard = dispatch_lock.open('a'); fcntl.flock(guard, fcntl.LOCK_EX)
                try:
                    result = Base.route(self, routes, inventory, parity=parity)
                    if result:
                        self.dispatch_guard = guard
                        return result
                except BaseException:
                    guard.close(); raise
                guard.close()
                return None

            def submit(self, name, module, argv, route=None):
                try:
                    if args.family == 'four':
                        super().submit(name, module, argv, route)
                    else:
                        # Platform-only adaptation: unique retry receipt dirs.
                        # Commands, worker, scopes, memory guards and register()
                        # are the qualified World implementation unchanged.
                        number = self.burst_attempts.get(name, 0)+1
                        self.burst_attempts[name] = number
                        receipt = self.root/'jobs'/f'{name}_attempt{number:03d}'
                        receipt.mkdir(parents=True, exist_ok=False)
                        argv = ['--user-snapshot-registration' if a == '--user-interim-registration' else a for a in argv]
                        spec = dict(source=str(self.source), source_sha=self.source_sha,
                            host=route['host'] if route else __import__('socket').gethostname().removesuffix('-worker-0'),
                            gpus=[route['gpu']] if route else [], gpu_uuids=[route['uuid']] if route else [],
                            min_free_MiB=(49152 if name == 'rank0' else 32768) if route else 0,
                            world_shared_lease=bool(route), module=module,
                            command=[self.resources.MODEL_PYTHON, '-u', '-m', module, *argv], receipt_dir=str(receipt))
                        write(receipt/'spec.json', spec); self.jobs[name] = spec
                        write(receipt/'operator.json', self.resources.launch(receipt/'spec.json', str(self.source)))
                    if route:
                        receipt = Path(self.jobs[name]['receipt_dir'])
                        deadline = time.monotonic()+180
                        while not any((receipt/n).exists() for n in ('RUNNING.json', 'process.json', 'RESULT.json', 'result.json')):
                            if time.monotonic() > deadline:
                                write(receipt/'STARTUP_PENDING.json', dict(at=time.time(),
                                    state='UNCONFIRMED_STARTUP_WAIT', no_redispatch=True,
                                    reason='Existing durable intent retained; process metadata may be delayed'))
                                break
                            time.sleep(.1)
                finally:
                    if getattr(self, 'dispatch_guard', None):
                        self.dispatch_guard.close(); self.dispatch_guard = None

        pins = json.loads((root/'PINNED.json').read_text())
        label, pin = next(iter(pins['models'].items()))
        if len(pins['models']) != 1:
            raise ValueError('One fixed checkpoint per burst')
        directory = root/label; directory.mkdir(exist_ok=args.resume)
        request = Burst(directory, pin); request.world = args.workers
        request.burst_attempts = {}; retry_after = {}
        if pin.get('reuse_export') and not request.export.exists():
            export = Path(pin['reuse_export'])
            if not (export/'COMPLETE').exists(): raise ValueError('Missing qualified reused export')
            request.export.symlink_to(export, target_is_directory=True)
        if args.resume:
            if args.family != 'world':
                raise ValueError('Only the explicitly reviewed World burst adoption is enabled')
            for path in sorted((directory/'jobs').glob('*_attempt*/spec.json')):
                spec = json.loads(path.read_text())
                if spec['source'] != str(request.source) or spec['source_sha'] != request.source_sha:
                    raise ValueError('Cross-source adoption requires a separate receipt')
                logical, number = path.parent.name.rsplit('_attempt', 1)
                if int(number) > request.burst_attempts.get(logical, 0):
                    request.burst_attempts[logical] = int(number)
                    request.jobs[logical] = spec
            write(root/'OPERATOR_ADOPTION.json', dict(at=time.time(), previous_owner=previous,
                adopted_jobs=list(request.jobs), completed_shards_not_reexecuted=True,
                existing_registration_and_scene_identity_preserved=True))
        # The façade changes administrative probing only, no worker imports.
        if args.family == 'world':
            request.resources = Pool(request.resources)
        else:
            from starVLA.rl.qwen_sourceflow import resources as common_resources
            pool = Pool(common_resources)
        owner = dict(pid=os.getpid(), start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),
            status='RUNNING', at=time.time(), source=str(args.source),
            workers=args.workers, model_batch=1, family=args.family, full_population=12146,
            noise_seed='seed_for(native_qwen_sourceflow_ODE,2101,scene_token)',
            model_and_scoring_code_unchanged=True)
        write(root/'owner.json', owner)
        try:
            if args.family == 'world':
                if not (request.export/'COMPLETE').exists():
                    request.submit('export', 'tools.qwen_world_postcot.export',
                                   ['--checkpoint', pin['checkpoint'], '--output', str(request.export)])
                while not all(request.done('rank'+str(i)) is True for i in range(args.workers)):
                    for name in list(request.jobs):
                        if request.done(name) is not False: continue
                        receipt = Path(request.jobs[name]['receipt_dir'])
                        result = json.loads((receipt/'result.json').read_text())
                        prechild = (not (receipt/'process.json').exists() and not (receipt/'worker.log').exists()
                            and ('BlockingIOError' in result.get('error', '') or 'Insufficient measured headroom' in result.get('error', '')))
                        if prechild and request.burst_attempts[name] < 4:
                            write(receipt/'PRECHILD_RESOURCE_CLASSIFICATION.json', dict(
                                child_not_started=True, scorer_calls=0, physical_inferences=0,
                                retained_failed_attempt=True, retry_after_seconds=30*request.burst_attempts[name]))
                            retry_after[name] = time.monotonic()+30*request.burst_attempts[name]
                            request.jobs.pop(name); continue
                        while request.active(): time.sleep(15)
                        raise RuntimeError('World worker failed; retain all receipts and completed scene records: '+str(receipt))
                    exported = (request.export/'COMPLETE').exists() and ('export' not in request.jobs or request.done('export') is True)
                    if exported:
                        if not request.reg.exists(): request.register()
                        parity = request.proof.exists() and json.loads(request.proof.read_text()).get('status') == 'PASS'
                        pending = [rank for rank in (range(args.workers) if parity else [0])
                                   if 'rank'+str(rank) not in request.jobs
                                   and time.monotonic() >= retry_after.get('rank'+str(rank), 0)]
                        routes, inventory = request.resources.available() if pending else ([], {})
                        for rank in pending:
                            name = 'rank'+str(rank)
                            if name in request.jobs or time.monotonic() < retry_after.get(name, 0): continue
                            route = request.route(routes, inventory, parity=rank == 0)
                            if route is None: break
                            argv = ['--export', str(request.export), '--output', str(request.output), '--cohort', 'NAVTEST',
                                '--data', '/mnt/project/qwen-sourceflow-native-20261018/data/NAVTEST', '--seed', '2101',
                                '--rank', str(rank), '--world', str(args.workers), '--user-interim-registration', str(request.reg)]
                            if rank == 0: argv += ['--native-parity-output', str(request.proof)]
                            request.submit(name, 'tools.qwen_world_postcot.evaluate', argv, route)
                        write(root/'burst_progress.json', dict(at=time.time(), submitted=list(request.jobs), workers=args.workers))
                    time.sleep(15)
                with (directory/'summary.log').open('x') as out:
                    subprocess.run([request.resources.MODEL_PYTHON, '-m', 'tools.qwen_world_postcot.summarize',
                        '--evaluation', str(request.output), '--output', str(directory/'SUMMARY.json')],
                        cwd=request.source, stdout=out, stderr=subprocess.STDOUT, check=True)
                from tools.qwen_world_postcot.user_interim_owner import validate_complete_summary
                summary = json.loads((directory/'SUMMARY.json').read_text())
                counts = validate_complete_summary(summary, json.loads((request.output/'identity.json').read_text()),
                    json.loads(Path('/mnt/project/flow-safe-improvement-20261013/population/manifest.json').read_text()))
                write(directory/'COMPLETE.json', dict(status='COMPLETE', PDMS=summary['PDMS'], scenes=12146,
                    logs=counts['manifest_logs'], physical_logs=counts['physical_logs'], all_owned_workers_exited=True))
                request.publish(summary)
            else:
                request.submit('export', '', ['--checkpoint', pin['checkpoint'], '--output', str(request.export)])
                while not all(request.done('rank'+str(i)) is True for i in range(args.workers)):
                    for name in list(request.jobs):
                        if request.done(name) is False:
                            result = json.loads((Path(request.jobs[name]['receipt_dir'])/'RESULT.json').read_text())
                            if result.get('status') == 'REFUSED_RESOURCE' and result.get('child_not_started') and request.attempts[name] < 4:
                                request.jobs.pop(name); continue
                            raise RuntimeError('Qualified worker failed; evidence retained: '+str(request.jobs[name]['receipt_dir']))
                    if request.done('export') is True:
                        if not request.reg.exists(): request.register()
                        parity = (request.proof.exists() and json.loads(request.proof.read_text()).get('status') == 'PASS'
                                  and request.done('parity') is True)
                        pending = [rank for rank in (range(args.workers) if parity else [0])
                                   if ('rank'+str(rank) if parity else 'parity') not in request.jobs]
                        routes, inventory = pool.available() if pending else ([], {})
                        for rank in pending:
                            name = 'rank'+str(rank) if parity else 'parity'
                            if name in request.jobs: continue
                            route = request.route(routes, inventory, parity=not parity)
                            if route is None: break
                            argv = ['--export', str(request.export), '--output', str(request.output), '--cohort', 'NAVTEST',
                                '--data', '/mnt/project/qwen-sourceflow-native-20261018/data/NAVTEST', '--rank', str(rank),
                                '--world', str(args.workers), '--seed', '2101', '--user-interim-registration', str(request.reg)]
                            if rank == 0: argv += ['--native-parity-output', str(request.proof)]
                            if not parity: argv += ['--parity-only']
                            request.submit(name, '', argv, route)
                        write(root/'burst_progress.json', dict(at=time.time(), submitted=list(request.jobs), workers=args.workers))
                    time.sleep(15)
                with (directory/'summary.log').open('x') as out:
                    subprocess.run([request.resources.MODEL_PYTHON, '-m', 'tools.qwen_world_compare.evaluate', '--summarize',
                        '--evaluation', str(request.output), '--output', str(directory/'SUMMARY.json')], cwd=request.source,
                        stdout=out, stderr=subprocess.STDOUT, check=True)
                summary = json.loads((directory/'SUMMARY.json').read_text())
                if summary['scenes'] != 12146 or summary['invalid_scenes'] or summary['qualification']:
                    raise ValueError('Incomplete full Navtest')
                write(directory/'COMPLETE.json', dict(status='COMPLETE', PDMS=summary['PDMS'], scenes=12146,
                    all_owned_workers_exited=True))
            owner.update(status='COMPLETE', ended_unix=time.time())
            write(root/'COMPLETE.json', dict(status='COMPLETE', models=[label], all_owned_workers_exited=True))
        except BaseException:
            owner.update(status='FAILED', error=traceback.format_exc(), ended_unix=time.time())
            write(root/'failure.json', owner)
            raise
        finally:
            write(root/'owner.json', owner)


if __name__ == '__main__':
    main()
