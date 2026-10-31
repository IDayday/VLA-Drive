"""Persistent 2k IL Navtest queue with burst evaluation and no training changes."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid

from .health import atomic, alive
from .evidence import OLD, FOUR, classify

REPORT = Path('/mnt/project/qwen-world-il-research-20261030')
ROOT = REPORT/'navtest_cadence_20261031'
PYTHON = '/mnt/project/flow-safe-improvement-20261013/venv/bin/python'
SOURCES = dict(four='/mnt/project/VLA-Drive-qwen-world-matched-il-navtest-20261030',
               world='/mnt/project/VLA-Drive-qwen-world-il-loss-later-navtest-20261030')


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runs():
    result = dict(V12=OLD/'optimized_labels_IL_20261028/runs/IL_full_seed42')
    selected = read(OLD/'fresh_loss_IL_20261029/SELECTION.json', {}).get('selected', [])
    for arm in ('TIME_RESIDUAL', 'HUBER'):
        if arm in selected:
            result[arm] = OLD/f'fresh_loss_IL_20261029/{arm}/runs/IL_full_seed42'
    result.update({'ARCH_'+a: FOUR/f'runs/ARCH_{a}/IL_seed42' for a in 'ABCD'})
    return result


def registered_points():
    """Bounded control roots only; include legacy nested owner registrations."""
    roots = [p for base in (OLD, FOUR) for p in base.iterdir()
             if p.is_dir() and 'navtest' in p.name.lower()]
    roots += [OLD/'optimized_labels_IL_20261028/NAVTEST']
    roots += [OLD/f'fresh_loss_IL_20261029/{a}/NAVTEST' for a in ('TIME_RESIDUAL', 'HUBER')]
    found = set()
    for root in roots:
        indices = [root]
        if root.name == 'NAVTEST':
            indices += list(root.glob('update*'))
        for index in indices:
            for label, pin in read(index/'PINNED.json', {}).get('models', {}).items():
                method = classify(label, index)
                if method and type(pin.get('updates')) is int:
                    found.add((method, pin['updates']))
    return found


def choose_step(available, requested, interval=2000):
    last = max(requested, default=0)
    eligible = [u for u in available if u % interval == 0 and u >= last+interval]
    return max(eligible, default=None)


def pin_checkpoint(original, destination):
    """Protect a sealed rolling checkpoint against concurrent trainer retention."""
    seal = read(original/'COMPLETE'); files = read(original/'files.json')
    if seal['files_sha256'] != sha(original/'files.json'):
        raise ValueError('Native checkpoint inventory changed')
    staging = destination.with_name(destination.name+'.pinning.'+uuid.uuid4().hex)
    staging.mkdir(parents=True, exist_ok=False)
    try:
        for name in set(files) | {'identity.json', 'files.json', 'COMPLETE'}:
            source = original/name
            if not source.resolve().is_relative_to(original.resolve()):
                raise ValueError('Unsafe checkpoint member')
            target = staging/name; target.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, target)
        os.rename(staging, destination)
    except BaseException:
        shutil.rmtree(staging)  # Only our uncommitted hardlinks, never original files.
        raise
    return seal


def enqueue(method, original, step):
    family = 'four' if method.startswith('ARCH_') else 'world'
    base = FOUR if family == 'four' else OLD
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    root = base/f'periodic_NAVTEST_{method}_{step:06d}_{stamp}'
    root.mkdir(exist_ok=False)
    label = (method[-1] if family == 'four' else method)+'_IL_'+f'{step:06d}'
    destination = root/'checkpoints'/label
    seal = pin_checkpoint(original, destination)
    pin = dict(updates=step, checkpoint=str(destination), original_checkpoint=str(original),
        files_sha256=seal['files_sha256'], user_instruction='当前各IL任务每至少2k有效更新完整Navtest，集中高并发尽快完成',
        evaluation_hosts=None)
    if family == 'four':
        pin['architecture_id'] = method[-1]
    else:
        pin.update(kind='IL', parent_checkpoint=str(destination), source=SOURCES[family])
    atomic(root/'PINNED.json', dict(schema='periodic_IL_NAVTEST_v1', models={label: pin},
        method=method, interval_updates=2000, evaluation_workers=32, seed=2101,
        full_population=12146, model_batch=1, no_parent_selection=True,
        no_training_or_reward_changes=True, selected_before_new_scores=True))
    return dict(method=method, updates=step, family=family, root=str(root), label=label,
        status='PENDING', source=SOURCES[family], checkpoint=str(destination),
        original_checkpoint=str(original), registered_unix=time.time())


def tick(state):
    # Reconcile an interrupted registration before considering a new point.
    known_roots = {j['root'] for j in state['jobs']}
    for base in (OLD, FOUR):
        for root in base.glob('periodic_NAVTEST_*'):
            if str(root) in known_roots: continue
            registration = read(root/'PINNED.json', {})
            if registration.get('schema') != 'periodic_IL_NAVTEST_v1': continue
            label, pin = next(iter(registration['models'].items()))
            method = registration['method']; family = 'four' if method.startswith('ARCH_') else 'world'
            previous = read(root/'LAUNCH.json', {})
            dispatching = (root/'LAUNCH_INTENT.json').exists()
            state['jobs'].append(dict(method=method, updates=pin['updates'], family=family,
                root=str(root), label=label, source=SOURCES[family], checkpoint=pin['checkpoint'],
                original_checkpoint=pin['original_checkpoint'], status='RUNNING' if previous.get('process') else
                'DISPATCHING' if dispatching else 'PENDING', process=previous.get('process'),
                registered_unix=root.stat().st_mtime, reconciled_uncommitted_intent=True))
    # Mark actual child completion; PID disappearance alone is never success.
    for job in state['jobs']:
        if job['status'] not in ('RUNNING', 'DISPATCHING'): continue
        root = Path(job['root'])
        if (root/'COMPLETE.json').exists():
            job.update(status='COMPLETE', end_unix=time.time())
            # Native copies were staging for export/parity only. Preserve the
            # exact deployment export, parent manifest and every score/trajectory.
            checkpoint = Path(job['checkpoint'])
            result = read(root/'COMPLETE.json')
            if result.get('all_owned_workers_exited') is True and checkpoint.is_dir():
                atomic(root/'NATIVE_STAGING_RETIREMENT.json', dict(at=time.time(),
                    retired=str(checkpoint), original_checkpoint=job['original_checkpoint'],
                    deployment_export=str(root/job['label']/'export'),
                    reason='Completed evaluator; own redundant full optimizer/model hardlinks only',
                    original_native_files_modified=False, scores_and_deployment_weights_retained=True))
                shutil.rmtree(checkpoint)
        elif (root/'failure.json').exists():
            job.update(status='FAILED_REVIEW_REQUIRED', failure=str(root/'failure.json'))
        else:
            receipt = read(root/'LAUNCH.json', {})
            observed_owner = read(root/'owner.json', {})
            process = job.get('process') or receipt.get('process') or observed_owner
            if alive(process):
                job.update(status='RUNNING', process={k: process[k] for k in ('pid', 'start_ticks')})
            elif time.time()-job.get('dispatched_unix', job['registered_unix']) > 180:
                job.update(status='BLOCKED_OWNER_EXIT_WITHOUT_COMPLETE', no_automatic_duplicate=True)
    existing = registered_points()
    existing.update((j['method'], j['updates']) for j in state['jobs'])
    if shutil.disk_usage(ROOT).free > 500*1024**3:
        for method, run in runs().items():
            if any(j['method'] == method and j['status'] in ('FAILED_REVIEW_REQUIRED', 'BLOCKED_OWNER_EXIT_WITHOUT_COMPLETE') for j in state['jobs']): continue
            # At most one queued checkpoint per method. Pin the latest eligible
            # available boundary; do not backfill an unlimited history.
            if any(j['method'] == method and j['status'] == 'PENDING' for j in state['jobs']): continue
            available = {int(p.name[6:]): p for p in (run/'checkpoints').glob('update*')
                         if p.name[6:].isdigit() and (p/'COMPLETE').exists()}
            step = choose_step(available, [u for m, u in existing if m == method])
            if step is None: continue
            try:
                job = enqueue(method, available[step], step)
            except FileNotFoundError:
                continue  # Trainer retired the rolling checkpoint while pinning.
            state['jobs'].append(job); existing.add((method, step))
            atomic(ROOT/'STATE.json', state)
    for job in state['jobs']:
        if job['status'] != 'PENDING': continue
        if sum(j['status'] in ('RUNNING', 'DISPATCHING') for j in state['jobs']) >= 2: break
        root = Path(job['root'])
        operator = root/'BURST_OPERATOR.py'
        if not operator.exists():
            operator.write_bytes(Path(__file__).with_name('navtest_burst.py').read_bytes())
        job['operator_sha256'] = sha(operator)
        command = [PYTHON, '-u', str(operator),
            '--root', str(root), '--source', job['source'], '--family', job['family'], '--workers', '32']
        job.update(status='DISPATCHING', command=command, dispatched_unix=time.time())
        atomic(root/'LAUNCH_INTENT.json', dict(command=command, registered_job=job))
        atomic(ROOT/'STATE.json', state)
        with (root/'owner.log').open('ab') as stream:
            process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[2],
                env=dict(os.environ, PYTHONUTF8='1', LANG='C.UTF-8', LC_ALL='C.UTF-8'),
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        job.update(status='RUNNING', process=dict(pid=process.pid,
            start_ticks=int(Path(f'/proc/{process.pid}/stat').read_text().split()[21])),
            command=command, dispatched_unix=time.time())
        atomic(root/'LAUNCH.json', job)
        atomic(ROOT/'STATE.json', state)
    state['at'] = time.time(); atomic(ROOT/'STATE.json', state)
    signature = hashlib.sha256(json.dumps([(j['method'], j['updates'], j['status'])
        for j in state['jobs']], sort_keys=True).encode()).hexdigest()
    if signature != state.get('published_queue_signature'):
        state['published_queue_signature'] = signature; atomic(ROOT/'STATE.json', state)
        atomic(REPORT/'reporting/refresh_request.json', dict(id=signature,
            reason='periodic_IL_Navtest_queue_updated', administrative=False))


def main():
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--once', action='store_true')
    parser.add_argument('--interval-seconds', type=int, default=600); args = parser.parse_args()
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'owner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        owner = dict(pid=os.getpid(), start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),
            status='RUNNING', at=time.time(), interval_seconds=args.interval_seconds)
        atomic(ROOT/'OWNER.json', owner)
        atomic(ROOT/'REGISTRATION.json', dict(interval_updates=2000, initial='latest eligible available sealed checkpoint',
            seed=2101, full_population=12146, metric='canonical_NAVSIM_v1.1', inference='single_cleanx_K10_shift5',
            workers_per_model=32, max_concurrent_models=2, burst_mode=True, actual_placement='verified idle then qualified sharing',
            model_batch=1, original_actor_and_scorer_unchanged=True, metadata_only_wakeup_seconds=args.interval_seconds,
            no_training_log_polling=True, no_duplicate_existing_points=True, raw_completed_and_screened_arms_excluded=True,
            native_staging_retired_after_export_and_completed_evaluation=True, deployment_weights_and_scores_retained=True,
            minimum_free_disk_bytes=500*1024**3, all_private_assets_stay_out_of_git=True))
        state = read(ROOT/'STATE.json', dict(jobs=[]))
        while True:
            try:
                tick(state)
            except Exception:
                import traceback
                atomic(ROOT/'FAILURE.json', dict(at=time.time(), error=traceback.format_exc(),
                    no_scientific_worker_killed=True, no_blind_retry=True))
                raise
            if args.once: break
            time.sleep(args.interval_seconds)


if __name__ == '__main__':
    main()
