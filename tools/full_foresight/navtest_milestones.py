"""Background exact-step Navtest queue, independent of immutable training code.

The watch process preserves checkpoints quickly; separate task processes perform
hashing, FP32 inference and canonical CPU scoring. One task per host is allowed.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import time

from .navtest_schedule import (SCHEMA, TERMINAL, atomic, busy, find_exact, host_ready,
    job_key, lease, live_owner, load_registration, preserve, read, sha, signature,
    source_identity, validate, validate_scores, write_csv)

MODULE = 'tools.full_foresight.navtest_milestones'


def environment(gpu=''):
    return dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUBLAS_WORKSPACE_CONFIG=':4096:8',
                OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                TOKENIZERS_PARALLELISM='false')


def execute(config, module, args, log):
    with Path(log).open('a') as stream:
        subprocess.run([config['inference_python'], '-m', module, *map(str, args)],
                       cwd=config['source_worktree'], env=environment(),
                       stdout=stream, stderr=subprocess.STDOUT, check=True,
                       timeout=config['task_timeout_seconds'])


def model_for(config, key):
    arm, step = key.split('_')
    model = next(m for m in config['models'] if m['arm'] == arm)
    update = int(step)
    if update not in config['updates'] or key != job_key(arm, update):
        raise ValueError('Unregistered task')
    return model, update


def job_dir(config, key):
    return Path(config['artifact_root']) / 'jobs' / key


def task_id(registration, key, attempt):
    return f'navtest_auto_{registration["identity"][:12]}_{key}_seed42_a{attempt}'


def release_owned_pressure(config, model, job):
    """Only exact ledger/cmdline/env/log/UID/process-group verified parents."""
    result_path = Path(job) / 'pressure_releases.json'
    records = read(result_path) if result_path.exists() else []
    seen = {(r['ledger'], r['pid']) for r in records}
    ledgers = Path(config['campaign_root']) / 'allocations'
    for path in ledgers.glob(Path(model['training_run']).name + '*.json'):
        try:
            record = read(path)
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        if record.get('host') != socket.gethostname():
            continue
        for item in record.get('restored_pressure', []):
            pid, gpu = int(item['parent_pid']), int(item['gpu'])
            if gpu not in config['gpus'] or (str(path), pid) in seen:
                continue
            proc = Path(f'/proc/{pid}')
            try:
                args = [x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
                env = dict(x.decode().split('=', 1) for x in (proc/'environ').read_bytes().split(b'\0') if b'=' in x)
                expected = path.with_name(path.stem + f'_pressure_gpu{gpu}.log')
                actual = (proc/'fd/1').resolve(strict=True)
                if (args[:3] != [config['pressure_python'], '-u', config['pressure_script']]
                        or env.get('CUDA_VISIBLE_DEVICES') != str(gpu)
                        or actual != expected or proc.stat().st_uid != os.getuid()
                        or os.getpgid(pid) != pid):
                    continue
                os.kill(pid, signal.SIGINT)
                records.append({'pid': pid, 'gpu': gpu, 'host': socket.gethostname(),
                                'ledger': str(path), 'verified_cmdline': args,
                                'verified_log': str(actual), 'unix': time.time(),
                                'reason': 'User-authorized exact-milestone Navtest GPU sharing'})
                atomic(result_path, records)
            except (FileNotFoundError, ProcessLookupError):
                continue
    return records


def restore_owned_pressure(config, model, job):
    """Restore a released reserve only after final training and idle GPU proof."""
    path = Path(job)/'pressure_releases.json'
    if not path.exists() or read(Path(model['training_run'])/'status.json')['status'] != 'COMPLETE':
        return
    values = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                     '--format=csv,noheader,nounits'], text=True)
    used = {int(line.split(',')[0]): int(line.split(',')[1]) for line in values.splitlines()}
    records = [];seen = set()
    ledger = Path(config['campaign_root'])/'allocations'/(Path(model['training_run']).name+
              '_navtest_' + Path(job).name + '_pressure_restore.json')
    if ledger.exists():
        return
    for item in read(path):
        gpu = item['gpu']
        if gpu in seen or used.get(gpu, 81920) > 512:
            continue
        seen.add(gpu)
        log = ledger.with_name(ledger.stem+f'_pressure_gpu{gpu}.log')
        with log.open('x') as stream:
            proc = subprocess.Popen(item['verified_cmdline'], env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
                                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        records.append({'gpu': gpu, 'parent_pid': proc.pid})
        atomic(ledger, {'host': socket.gethostname(), 'restored_pressure': records,
                        'source': config['evaluation_source_sha'], 'reason': 'Restore released user reserve after final idle GPU evaluation'})


def gpu_free(config):
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                     '--format=csv,noheader,nounits'], text=True)
    values = {int(line.split(',')[0]): int(line.split(',')[1]) for line in output.splitlines()}
    return all(values.get(g, 0) >= config['minimum_free_mib'] for g in config['gpus'])


def export_group(registration_path, key, attempt, preflight=False):
    registration = load_registration(registration_path);config = registration['config']
    model, update = model_for(config, key);job = job_dir(config, key)
    if socket.gethostname() != model['hostname']:
        raise ValueError('Wrong authorized GPU host')
    if preflight:
        output = {'host': socket.gethostname(), 'source': source_identity(config['source_worktree']),
                  'gpu_headroom': gpu_free(config), 'GPU_count': len(config['gpus']),
                  'optimizer_updates': 0, 'processes_launched': 0}
        print(json.dumps(output), flush=True);return
    with lease(job/'gpu_group.lock'):
        release_owned_pressure(config, model, job)
        for _ in range(12):
            if gpu_free(config):
                break
            time.sleep(5)
        else:
            atomic(job/'gpu_group_status.json', {'status': 'WAITING_RESOURCE', 'pid': os.getpid()})
            raise SystemExit(75)
        snapshot = read(job/'snapshot.json');bank = job/'predictions'
        launch_path = job/f'gpu_launch_a{attempt}.json'
        children = [];records = []
        try:
            for rank, gpu in enumerate(config['gpus']):
                completed = bank/f'shard_{rank}.json'
                if completed.exists() and read(completed)['status'] == 'complete' and not read(completed)['failed']:
                    continue
                rid = task_id(registration, key, attempt) + f'_rank{rank}'
                if (Path(config['campaign_root'])/'runs'/rid).exists():
                    raise FileExistsError('Attempt ID already used')
                args = [config['inference_python'], '-u', '-m', 'tools.foresight.export_predictions',
                        '--training-run', snapshot['training_run'], '--checkpoint-tag', snapshot['tag'],
                        '--current-root', config['current_root'], '--output', str(bank),
                        '--campaign-root', config['campaign_root'], '--run-id', rid,
                        '--sampling-seed', str(config['sampling_seed']), '--rank', str(rank),
                        '--world-size', '8', '--max-seconds', str(config['task_timeout_seconds']),
                        '--campaign-gpu-hours', str(config['campaign_gpu_hours']),
                        '--final-lock', str(job/'lock.json'),
                        '--gpu-memory-fraction', str(config['gpu_memory_fraction'])]
                log = job/(rid+'.log')
                with log.open('x') as stream:
                    child = subprocess.Popen(args, cwd=config['source_worktree'],
                        env=dict(environment(gpu), OMP_NUM_THREADS='2'), stdout=stream,
                        stderr=subprocess.STDOUT, start_new_session=True)
                children.append(child)
                records.append({'pid': child.pid, 'gpu': gpu, 'host': socket.gethostname(),
                                'run_id': rid, 'command': args, 'log': str(log)})
                atomic(launch_path, records)
            atomic(job/'gpu_group_status.json', {'status': 'RUNNING', 'pid': os.getpid(),
                                                'attempt': attempt, 'host': socket.gethostname()})
            while any(p.poll() is None for p in children):
                release_owned_pressure(config, model, job)
                time.sleep(10)
            failed = [p.returncode for p in children if p.returncode]
            if failed:
                raise RuntimeError('Exporter failure; all rows retained: ' + str(failed))
            if not all((bank/f'shard_{r}.json').exists() and read(bank/f'shard_{r}.json')['status'] == 'complete'
                       and read(bank/f'shard_{r}.json')['failed'] == 0 for r in range(8)):
                atomic(job/'gpu_group_status.json', {'status': 'PAUSED', 'attempt': attempt})
                raise SystemExit(76)
            atomic(job/'gpu_group_status.json', {'status': 'COMPLETE', 'attempt': attempt, 'host': socket.gethostname()})
        except BaseException as error:
            # Already launched exporters continue to their own registered boundary.
            # Never signal a trainer or an unrelated process on an error path.
            while any(p.poll() is None for p in children):
                release_owned_pressure(config, model, job)
                time.sleep(10)
            if not (isinstance(error, SystemExit) and error.code == 76):
                atomic(job/'gpu_group_status.json', {'status': 'FAILED', 'error': repr(error),
                                                    'attempt': attempt, 'children': records})
            raise
        finally:
            try:
                restore_owned_pressure(config, model, job)
            except Exception as error:
                atomic(job/'pressure_restore_error.json', {'error': repr(error), 'unix': time.time()})


def remote_command(config, model, args):
    command = [config['worker_python'], '-u', '-m', MODULE, *map(str, args)]
    if model['host'] == 'local':
        return command
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', model['host'],
            'cd ' + shlex.quote(config['source_worktree']) + ' && exec ' + shlex.join(command)]


def score_command(config, registration, key, job, attempt, merge=0):
    args = [config['scoring_python'], '-u', '-m', 'tools.foresight.score_pdms',
            '--devkit', config['devkit'], '--metric-index', config['metric_index'],
            '--current-index', str(Path(config['current_root'])/'index.json'),
            '--predictions', str(job/'predictions'), '--output', str(job/'scores'),
            '--campaign-root', config['campaign_root'], '--run-id',
            task_id(registration, key, attempt)+f'_score_m{merge}',
            '--workers', str(config['cpu_workers']), '--timeout-seconds', str(config['task_timeout_seconds'])]
    if (job/'scores/identity.json').exists():
        args.append('--resume')
    return args


def finalize(config, job, key, update):
    summary, rows = validate_scores(job/'scores', config['cache_snapshot'], update)
    if not (job/'planning/summary.json').exists():
        execute(config, 'tools.full_foresight.summarize_checkpoint_navtest',
                ['--lock', job/'lock.json', '--score-dirs', job/'scores', '--output', job/'planning'], job/'finalize.log')
    if not (job/'ego_fit/summary.json').exists():
        execute(config, 'tools.foresight.evaluate_ego',
                ['--predictions', job/'predictions', '--current-root', config['current_root'],
                 '--processed-root', config['ego_labels'], '--output', job/'ego_fit'], job/'finalize.log')
    fit = read(job/'ego_fit/summary.json')
    if not fit['valid'] or fit['failed'] or fit['scenes'] != 12146:
        raise ValueError('Invalid ego fit; rows retained')
    ego = {r['token']: r for r in csv.DictReader((job/'ego_fit/scenes.csv').open())}
    arm = key.split('_')[0]
    combined = []
    for row in rows:
        item = ego[row['token']]
        if item['failure'] or item['log'] != row['log']:
            raise ValueError('Ego-fit population differs')
        combined.append(dict(row, candidate=arm, checkpoint_update=update, precision='FP32',
                             training_seed=42, sampling_seed=42,
                             **{k: item[k] for k in ('ADE', 'FDE', 'yaw_MAE_rad', 'yaw_endpoint_rad', 'motion_group')}))
    write_csv(job/'complete.csv', combined)
    prior_steps = [s for s in config['updates'] if s < update and
                   (job_dir(config, job_key(arm, s))/'result.json').exists()]
    baseline = job_dir(config, job_key(arm, max(prior_steps)))/'scores' if prior_steps else Path(config['baseline_scores'][arm])
    if not (job/'progression/summary.json').exists():
        execute(config, 'tools.full_foresight.compare_checkpoint_updates',
                ['--first', job/'scores', '--baseline', baseline, '--output', job/'progression'], job/'finalize.log')
    result = {'status': 'COMPLETE', 'arm': arm, 'update': update, 'scenes': 12146, 'logs': 136,
              'failed': 0, 'zero_scenes': sum(float(r['score']) == 0 for r in rows),
              'PDMS': summary['PDMS']*100, 'metrics_points': {k: v*100 for k, v in summary['metrics'].items()},
              'ego_fit': fit['groups']['all'], 'complete_csv_sha256': sha(job/'complete.csv'),
              'checkpoint': summary['export_identity']['checkpoint'],
              'evaluation_source': config['evaluation_source_sha'], 'training_updates_in_evaluator': 0,
              'progression': read(job/'progression/summary.json'), 'completed_unix': time.time()}
    atomic(job/'result.json', result)
    return result


def task(registration_path, key):
    registration = load_registration(registration_path);config = registration['config']
    model, update = model_for(config, key);job = job_dir(config, key)
    with lease(job/'worker.lock'):
        gpu = cpu = None
        state = read(job/'status.json') if (job/'status.json').exists() else {}
        if state.get('status') == 'COMPLETE':
            return
        attempt = int(state.get('attempt', 0)) + 1
        state.update(status='RUNNING', phase='CHECKPOINT_LOCK', pid=os.getpid(), attempt=attempt,
                     registration=registration['identity'], key=key, started_unix=time.time())
        def status(**values):
            state.update(values, updated_unix=time.time());atomic(job/'status.json', state)
        status()
        try:
            if not (job/'lock.json').exists():
                execute(config, 'tools.full_foresight.lock_checkpoint_navtest',
                    ['--models', job/'models.json', '--development-report', model['development_report'],
                     '--current-root', config['current_root'], '--metric-index', config['metric_index'],
                     '--output', job/'lock.json', '--update', update, '--sampling-seed', config['sampling_seed'],
                     '--allow-prior-development-evidence'], job/'lock.log')
            bank = job/'predictions';gpu = None;cpu = None;cpu_attempts = [];started = time.monotonic()
            complete = lambda: all((bank/f'shard_{r}.json').exists() and read(bank/f'shard_{r}.json')['status'] == 'complete'
                                    and not read(bank/f'shard_{r}.json')['failed'] for r in range(8))
            if not complete():
                cmd = remote_command(config, model, ['export-group', '--registration', registration_path,
                                                     '--task', key, '--attempt', attempt])
                with (job/f'gpu_group_a{attempt}.log').open('x') as stream:
                    gpu = subprocess.Popen(cmd, cwd=config['source_worktree'], env=environment(),
                                           stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                atomic(job/f'gpu_group_launch_a{attempt}.json', {'pid': gpu.pid, 'command': cmd})
            status(phase='EXPORT_AND_SCORE')
            while (not (bank/'identity.json').exists() or (gpu is not None and
                    (not (job/'gpu_group_status.json').exists() or
                     read(job/'gpu_group_status.json').get('attempt') != attempt))):
                if gpu is not None and gpu.poll() is not None:
                    if gpu.returncode == 75:
                        status(status='QUEUED', reason='Insufficient GPU headroom', retry_after=time.time()+60)
                        return
                    raise RuntimeError('GPU group exited before export identity; inspect group log')
                if time.monotonic()-started > 900:
                    raise TimeoutError('No export identity')
                time.sleep(5)
            if not (job/'scores/summary.json').exists():
                command = score_command(config, registration, key, job, attempt)
                with (job/f'score_a{attempt}_m0.log').open('x') as stream:
                    cpu = subprocess.Popen(command, cwd=config['source_worktree'], env=environment(),
                                           stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                cpu_attempts.append({'pid': cpu.pid, 'command': command})
                atomic(job/f'cpu_launch_a{attempt}.json', cpu_attempts)
            while not ((gpu is None or gpu.poll() is not None) and (cpu is None or cpu.poll() is not None)):
                if time.monotonic()-started > config['task_timeout_seconds']+900:
                    raise TimeoutError('Task runtime cap reached; partial artifacts retained')
                status();time.sleep(15)
            if gpu is not None and gpu.returncode == 76:
                status(status='PAUSED', reason='Export budget/runtime pause; exact checkpoint retained')
                return
            if gpu is not None and gpu.returncode:
                raise RuntimeError('GPU group failed; logs and all rows retained')
            if not complete():
                status(status='PAUSED', reason='Incomplete export; exact checkpoint retained');return
            # The CPU scorer may finish all rows just before the last GPU shard
            # publishes COMPLETE. Resume only its completion merge, once.
            if not (job/'scores/summary.json').exists():
                record = read(Path(config['campaign_root'])/'runs'/cpu_attempts[-1]['command'][cpu_attempts[-1]['command'].index('--run-id')+1]/'status.json')
                if record['status'] != 'PAUSED':
                    raise RuntimeError('Canonical scorer failed; do not delete failed rows')
                command = score_command(config, registration, key, job, attempt, 1)
                cpu_attempts.append({'command': command, 'merge_resume': True})
                atomic(job/f'cpu_launch_a{attempt}.json', cpu_attempts)
                with (job/f'score_a{attempt}_m1.log').open('x') as stream:
                    subprocess.run(command, cwd=config['source_worktree'], env=environment(), stdout=stream,
                                   stderr=subprocess.STDOUT, timeout=config['task_timeout_seconds'], check=True)
            status(phase='VALIDATION_AND_SUMMARY')
            finalize(config, job, key, update)
            status(status='COMPLETE', phase='COMPLETE', completed_unix=time.time())
        except BaseException as error:
            status(status='FAILED', error=repr(error), evidence_preserved=True)
            raise


def summary(config):
    root = Path(config['artifact_root']);rows = [];states = {}
    for model in config['models']:
        for update in config['updates']:
            key = job_key(model['arm'], update);job = job_dir(config, key)
            if (job/'result.json').exists():
                r = read(job/'result.json');m = r['metrics_points']
                rows.append({'arm': r['arm'], 'updates': update, 'PDMS': r['PDMS'],
                    'NC': m['no_at_fault_collisions'], 'DAC': m['drivable_area_compliance'],
                    'TTC': m['time_to_collision_within_bound'], 'EP': m['ego_progress'], 'Comfort': m['comfort'],
                    'zero_scenes': r['zero_scenes'], 'failed': r['failed'],
                    'ADE': r['ego_fit']['ADE'], 'FDE': r['ego_fit']['FDE'], 'complete_csv': str(job/'complete.csv')})
            states[key] = read(job/'status.json').get('status', 'WAITING') if (job/'status.json').exists() else 'WAITING'
    write_csv(root/'SUMMARY.csv', rows)
    atomic(root/'TASKS.json', states)
    return states


def cpu_active(job):
    for p in Path(job).glob('cpu_launch_a*.json'):
        for record in read(p):
            if 'pid' not in record:
                continue
            try:
                args = [x.decode() for x in Path(f'/proc/{record["pid"]}/cmdline').read_bytes().split(b'\0') if x]
                if args == record['command']:
                    return True
            except FileNotFoundError:
                pass
    return False


def usage(config, identity):
    total = 0.0
    for p in (Path(config['campaign_root'])/'runs').glob(f'navtest_auto_{identity[:12]}_*/status.json'):
        r = read(p)
        total += r.get('gpu_hours', (time.time()-r['start_unix'])*r['gpu_count']/3600)
    return total


def watch(registration_path, once=False):
    registration = load_registration(registration_path);config = registration['config'];root = Path(config['artifact_root'])
    with lease(root/'observer.lock'):
        if (root/'status.json').exists() and read(root/'status.json')['registration'] != registration['identity']:
            raise ValueError('Artifact root belongs to a different schedule')
        start = registration['created_unix'];children = []
        while True:
            for child in children:
                child.poll()  # Reap only our direct task children.
            active = [];errors = []
            # Preserve every newly available exact snapshot before slow hashing,
            # CPU scoring, or an unavailable host can delay the observer.
            for model in config['models']:
                state = read(Path(model['training_run'])/'status.json')
                for update in config['updates']:
                    key = job_key(model['arm'], update);job = job_dir(config, key)
                    if (job/'status.json').exists() and read(job/'status.json').get('status') in TERMINAL:
                        continue
                    try:
                        snapshot = preserve(model, update, job)
                        if snapshot is None and state['completed'] >= update+200:
                            atomic(job/'status.json', {'status': 'MISSED_CHECKPOINT', 'key': key,
                                   'error': 'Exact complete checkpoint unavailable; no newer substitution', 'updated_unix': time.time()})
                        elif snapshot is not None and not (job/'status.json').exists():
                            atomic(job/'status.json', {'status': 'QUEUED', 'key': key, 'updated_unix': time.time()})
                    except FileNotFoundError:
                        continue  # Trainer may be atomically publishing or rolling GC.
                    except Exception as error:
                        atomic(job/'status.json', {'status': 'FAILED', 'key': key, 'error': repr(error)})
            charged = usage(config, registration['identity'])
            paused = (root/'STOP_SCHEDULING').exists() or charged >= config['evaluation_gpu_hours'] or time.time()-start >= config['watch_timeout_seconds']
            for model in config['models']:
                for update in config['updates']:
                    key = job_key(model['arm'], update);job = job_dir(config, key)
                    launch = job/'launch.json'
                    if (busy(job/'worker.lock') or busy(job/'gpu_group.lock') or cpu_active(job)
                            or (launch.exists() and live_owner(read(launch), registration_path, 'task', key))):
                        active.append(model['arm']);break
            if not paused:
                for model in config['models']:
                    if model['arm'] in active or not host_ready(config, model):
                        continue
                    for update in config['updates']:
                        key = job_key(model['arm'], update);job = job_dir(config, key)
                        launch = job/'launch.json'
                        if not (job/'snapshot.json').exists():
                            continue
                        state = read(job/'status.json')
                        if state['status'] in TERMINAL or state['status'] == 'PAUSED' or state.get('retry_after', 0) > time.time():
                            continue
                        command = [config['worker_python'], '-u', '-m', MODULE, 'task',
                                   '--registration', str(Path(registration_path).resolve()), '--task', key]
                        with (job/'worker.log').open('a') as stream:
                            child = subprocess.Popen(command, cwd=config['source_worktree'], env=environment(),
                                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                        children.append(child)
                        atomic(launch, {'pid': child.pid, 'command': command, 'source': config['evaluation_source_sha']})
                        active.append(model['arm']);break
            states = summary(config)
            finished = all(s in TERMINAL for s in states.values())
            outcome = 'COMPLETE' if finished and all(s == 'COMPLETE' for s in states.values()) else 'FINISHED_WITH_FAILURES' if finished else 'PAUSED' if paused else 'RUNNING'
            atomic(root/'status.json', {'status': outcome, 'pid': os.getpid(), 'registration': registration['identity'],
                'source': config['evaluation_source_sha'], 'started_unix': start, 'updated_unix': time.time(),
                'active_models': active, 'completed_tasks': sum(s == 'COMPLETE' for s in states.values()),
                'total_tasks': 12, 'evaluation_gpu_process_hours': charged,
                'training_modified': False, 'tasks': states})
            if finished or paused or once:
                return
            time.sleep(config['poll_seconds'])


def register(config_path, output):
    config = validate(read(config_path))
    if source_identity(config['source_worktree']) != config['evaluation_source_sha']:
        raise ValueError('Wrong committed evaluation source')
    out = Path(output)
    if out.exists():
        raise FileExistsError('Registration is immutable')
    files = [config['metric_index'], config['cache_snapshot'], config['cache_audit'],
             str(Path(config['current_root'])/'identity.json'), str(Path(config['current_root'])/'index.json')]
    for model in config['models']:
        files.extend([model['development_report'], str(Path(model['training_run'])/'identity.json')])
    atomic(out, {'identity': signature(config), 'config': config, 'created_unix': time.time(),
                 'asset_hashes': {p: sha(p) for p in sorted(set(files))}})
    print(json.dumps({'registration': str(out), 'identity': signature(config), 'tasks': 12}))


def main():
    parser = argparse.ArgumentParser(__doc__);sub = parser.add_subparsers(dest='mode', required=True)
    p = sub.add_parser('register');p.add_argument('--config', required=True);p.add_argument('--output', required=True)
    for name in ('watch', 'task', 'export-group'):
        p = sub.add_parser(name);p.add_argument('--registration', required=True)
        if name == 'watch':p.add_argument('--once', action='store_true')
        else:p.add_argument('--task', required=True)
        if name == 'export-group':p.add_argument('--attempt', type=int, default=1);p.add_argument('--preflight', action='store_true')
    a = parser.parse_args()
    if a.mode == 'register':register(a.config, a.output)
    elif a.mode == 'watch':watch(a.registration, a.once)
    elif a.mode == 'task':task(str(Path(a.registration).resolve()), a.task)
    else:export_group(a.registration, a.task, a.attempt, a.preflight)


if __name__ == '__main__':
    main()
