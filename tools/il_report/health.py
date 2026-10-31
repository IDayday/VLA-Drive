"""Bounded operational health checks; scientific trainers remain immutable.

Reports may adopt a reviewed recovery entry. They never invent a new experiment,
extend a pilot, change a recipe, or kill a learner based on journal age alone.
"""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

BASE = Path('/mnt/project/qwen-world-postcot-20261025')
HUBER = BASE/'fresh_loss_IL_20261029/HUBER'
SCIENCE = Path('/mnt/project/VLA-Drive-qwen-world-loss-from-init-20261029')
OPERATOR = Path('/mnt/project/VLA-Drive-qwen-world-time-owner-adoption-20261029')
PYTHON = '/mnt/project/flow-safe-improvement-20261013/venv/bin/python'


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp.'+str(os.getpid()))
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def alive(owner):
    if not owner:
        return False
    try:
        fields = Path(f"/proc/{owner['pid']}/stat").read_text().rsplit(')', 1)[1].split()
        return fields[0] != 'Z' and int(fields[19]) == owner['start_ticks']
    except FileNotFoundError:
        return False


def rewind_journal(path, boundary, archive):
    """Retain every aborted byte, publish only an unambiguous committed prefix.

    Refuse mixed/duplicate sequences and malformed JSON rather than deduplicating.
    Caller must hold the arm owner and all rank writer locks.
    """
    path, archive = Path(path), Path(archive)
    prior = read(archive/(path.name+'.receipt.json'))
    if prior and digest(path) == prior['committed_sha256']:
        if digest(prior['aborted_file']) != prior['aborted_sha256']:
            raise ValueError('Archived aborted journal changed')
        return prior
    prefix, suffix, previous = [], [], None
    for line in path.read_bytes().splitlines(keepends=True):
        row = json.loads(line)
        update = row['updates']
        if type(update) is not int or (previous is not None and update != previous+1):
            raise ValueError('Noncontinuous update journal: '+str(path))
        previous = update
        (prefix if update <= boundary else suffix).append(line)
    if not prefix or json.loads(prefix[-1])['updates'] != boundary:
        raise ValueError('Checkpoint boundary missing from journal: '+str(path))
    archive.mkdir(parents=True, exist_ok=True)
    saved = archive/path.name
    payload = b''.join(suffix)
    if saved.exists() and saved.read_bytes() != payload:
        raise ValueError('Refusing to replace recovery evidence')
    if not saved.exists():
        with saved.open('xb') as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    committed = b''.join(prefix)
    receipt = dict(file=str(path), aborted_file=str(saved), aborted_sha256=digest(saved),
        aborted_updates=len(suffix), last_observed_update=previous, resume_boundary=boundary,
        committed_sha256=hashlib.sha256(committed).hexdigest(), original_sha256=digest(path),
        physical_seconds=sum(json.loads(line).get('seconds', 0) for line in suffix))
    atomic(archive/(path.name+'.receipt.json'), receipt)
    tmp = path.with_name(path.name+'.recover.'+str(os.getpid()))
    with tmp.open('xb') as stream:
        stream.write(committed); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)
    return receipt


def recover_huber(root):
    """One explicitly reviewed infrastructure recovery; native restore verifies all tensors."""
    root = Path(root)
    directory = root/'maintenance/huber_recovery_20261031'
    directory.mkdir(parents=True, exist_ok=True)
    with ExitStack() as resources:
        return _recover_huber_locked(root, directory, resources)


def _recover_huber_locked(root, directory, resources):
    lock = resources.enter_context((directory/'recovery.lock').open('a'))
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (directory/'LAUNCH.json').exists():
        return read(directory/'LAUNCH.json')  # single attempt, including terminal failure
    review = read(directory/'REVIEW.json')
    if not review or review['approved_action'] != 'resume_same_native_HUBER_from_update014000':
        raise PermissionError('A concrete recovery review is required')
    verified = read(root/'maintenance/HUBER14K_VERIFIED_20261031.json')
    checkpoint = HUBER/'runs/IL_full_seed42/checkpoints/update014000'
    seal = read(checkpoint/'COMPLETE')
    if (not verified or verified['status'] != 'PASS' or
            verified['files_sha256'] != digest(checkpoint/'files.json') or
            verified['identity_sha256'] != seal['identity_sha256'] or
            verified['updates'] != 14000):
        raise ValueError('Missing verified full native recovery state')
    if alive(read(HUBER/'OWNER.json')):
        raise RuntimeError('Existing arm owner is alive; refusing duplicate')
    run = HUBER/'runs/IL_full_seed42'
    locks = []
    for p in [HUBER/'owner.lock', *[run/f'rank{i}.lock' for i in range(8)]]:
        stream = resources.enter_context(p.open('a'))
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(stream)
    state = read(HUBER/'STATE.json')
    if state['source'] != str(SCIENCE) or state['stage'] != 'FULL':
        raise ValueError('Recovery cannot change the registered science/stage')
    if any(j['status'] in ('RUNNING', 'WAITING_OWNED_EXIT') for j in state['jobs'].values()):
        raise RuntimeError('Existing worker intent requires reconciliation first')
    failed = [j for j in state['jobs'].values() if j['status'] == 'FAILED']
    if len(failed) != 1 or failed[0]['receipt'] != review['failed_receipt']:
        raise ValueError('Unreviewed worker failure')
    for name in ('STATE.json', 'OWNER.json', 'FAILURE.json', 'LAUNCH.json'):
        p = HUBER/name
        if p.exists() and not (directory/('previous_'+name)).exists():
            (directory/('previous_'+name)).write_bytes(p.read_bytes())
    rows = [rewind_journal(run/f'metrics_rank{r}.jsonl', 14000, directory/'aborted_journals') for r in range(8)]
    atomic(directory/'JOURNAL_ROLLBACK.json', dict(status='ARCHIVED', ranks=rows,
        logical_updates_recomputed=661, algorithmic_fresh_data_counted_twice=False,
        historical_physical_cost_retained=True, original_failure_cause='rank6_SIGKILL_cause_not_proven'))
    # Preserve superseded plot points as failed-attempt evidence as well.
    history_path = root/'reporting/training_history.json'
    with (root/'reporting/render.lock').open('a') as report_lock:
        fcntl.flock(report_lock, fcntl.LOCK_EX)
        history = read(history_path, [])
        aborted = [r for r in history if r['id'] == 'HUBER' and r['updates'] > 14000]
        atomic(directory/'aborted_report_history.json', aborted)
        atomic(history_path, [r for r in history if not (r['id'] == 'HUBER' and r['updates'] > 14000)])
    failed[0]['status'] = 'RESOLVED_INFRASTRUCTURE_FAILURE'
    failed[0]['recovery_review'] = str(directory/'REVIEW.json')
    resolution = dict(at=time.time(), review=str(directory/'REVIEW.json'),
        original_receipt=failed[0]['receipt'], native_resume=str(checkpoint))
    resolved = state.setdefault('resolved_failures', {})
    if isinstance(resolved, dict):
        resolved['HUBER_native14k_20261031'] = resolution
    elif isinstance(resolved, list):
        resolved.append(resolution)
    else:
        raise ValueError('Unknown resolved failure receipt schema')
    atomic(HUBER/'STATE.json', state)
    atomic(HUBER/'EXECUTION_ROUTING.json', dict(schema='world_IL_execution_routing_v1', arm='HUBER', world=8,
        training_hosts=['training-vlawm-tsinghua-zt2'], reason='New unlimited authorization; idle complete eight-GPU host'))
    for stream in locks:
        stream.close()
    env = dict(os.environ, WORLD_POSTCOT_ROOT=str(HUBER), WORLD_POSTCOT_TASK='qwen_world_fresh_loss_IL_20261029')
    command = [PYTHON, '-u', '-m', 'tools.qwen_world_postcot.fresh_loss_owner', '--arm', 'HUBER',
               '--runtime-source', str(SCIENCE), '--recover-owner']
    with (directory/'owner.log').open('ab') as log:
        process = subprocess.Popen(command, cwd=OPERATOR, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = dict(pid=process.pid, start_ticks=int(Path(f'/proc/{process.pid}/stat').read_text().split()[21]),
        status='OWNER_LAUNCHED_AWAIT_NATIVE_UPDATES', at=time.time(), source=str(OPERATOR),
        scientific_source=str(SCIENCE), command=command, checkpoint=str(checkpoint),
        target_updates=100000, retry_limit=1, review=str(directory/'REVIEW.json'))
    atomic(directory/'LAUNCH.json', receipt)
    return receipt


def check(root, runs):
    root = Path(root)
    statuses = []
    for run in runs:
        status = run.get('status')
        protected = run['id'] in ('PHYSICAL', 'DELTA') or status == '已完成'
        controller_lock = None
        if run['id'] in ('TIME_RESIDUAL', 'HUBER'):
            path = BASE/f"fresh_loss_IL_20261029/{run['id']}/owner.lock"
            if path.exists():
                with path.open('r+') as stream:
                    try:
                        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        controller_lock = False
                    except BlockingIOError:
                        controller_lock = True
        issue = not protected and (run.get('worker_status') in ('FAILED', 'FORCED_EXIT_REVIEW_REQUIRED')
            or (run.get('journal_age_seconds') is not None and run['journal_age_seconds'] > 1800)
            or controller_lock is False)
        statuses.append(dict(id=run['id'], status=status, updates=run.get('updates'),
            action='REVIEW_REQUIRED' if issue else 'OBSERVE',
            intentional_stop_preserved=protected, controller_owner_lock_held=controller_lock))
    directory = root/'maintenance/huber_recovery_20261031'
    recovery = read(directory/'LAUNCH.json')
    review = read(directory/'REVIEW.json')
    if (review and recovery is None and not (directory/'REPAIR_FAILURE.json').exists()
            and (root/'maintenance/HUBER14K_VERIFIED_20261031.json').exists()):
        # Detached helper takes exclusive locks. A concurrent report never launches a learner directly.
        guard = (directory/'dispatch.lock').open('a')
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
            helper = read(directory/'HELPER.json')
            if not alive(helper):
                with (directory/'helper.log').open('ab') as log:
                    p = subprocess.Popen(['/usr/bin/python3', '-m', 'tools.il_report.health', '--recover-huber',
                        '--root', str(root)], cwd=Path(__file__).resolve().parents[2], stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                atomic(directory/'HELPER.json', dict(pid=p.pid,
                    start_ticks=int(Path(f'/proc/{p.pid}/stat').read_text().split()[21]), at=time.time()))
        except BlockingIOError:
            pass
        finally:
            guard.close()
    result = dict(schema='world_report_health_v1', checked_unix=time.time(), runs=statuses,
        recovery=recovery, recovery_process_alive=alive(recovery),
        policy='Only reviewed native infrastructure recovery; no auto restart of stopped/completed/pilot-review arms',
        scientific_source_modified=False)
    observed = next((r for r in runs if r['id'] == 'HUBER'), None)
    if recovery and observed:
        counts = observed.get('observed_rank_updates', [])
        confirmed = (len(counts) == 8 and all(type(n) is int and n > 14000 for n in counts)
            and observed.get('optimizer_count_contract_valid') is True
            and observed.get('recent_gradient', 0) > 0
            and all(n == 0 for n in observed.get('optimizer_skipped_steps', [])))
        result['huber_new_native_updates_confirmed'] = confirmed
        if confirmed and not (directory/'RECOVERY_CONFIRMED.json').exists():
            atomic(directory/'RECOVERY_CONFIRMED.json', dict(at=time.time(),
                observed_rank_updates=counts, resumed_from=14000, optimizer_count_matches=True,
                finite_nonzero_gradient=True, host=observed.get('host')))
        if not confirmed and alive(recovery):
            observed['status'] = '恢复初始化 / 等待新的真实optimizer更新'
    rl_registry = read('/mnt/project/qwen-world-rl-optimization-20261030/RUNS.json', {})
    result['rl_runs'] = []
    from .evidence import tail
    for run in rl_registry.get('runs', []):
        status = run['status']
        protected = status in ('STOPPED_FUTILITY', 'PILOT_COMPLETE_REVIEW_REQUIRED', 'REGISTERED', 'COMPLETE')
        rank_updates, ages = [], []
        run_root = Path(run['root']) if run.get('root') else None
        if run_root:
            for rank in range(8):
                journal = run_root/f'metrics_rank{rank}.jsonl'
                rows = tail(journal, limit=1)
                rank_updates.append(rows[-1].get('updates') if rows else None)
                if journal.exists():
                    ages.append(time.time()-journal.stat().st_mtime)
        job = Path(run['job']) if run.get('job') else None
        terminal = read(job/'RESULT.json', {}) if job else {}
        if job and not terminal:
            terminal = read(job/'result.json', {})
        stale = (status == 'TRAINING_DISPATCHED' and len(ages) == 8 and max(ages) > 1800)
        failed = (terminal.get('status') in ('FAILED', 'FORCED_EXIT_REVIEW_REQUIRED')
            or terminal.get('returncode', 0) not in (0, None))
        action = ('PRESERVE_NO_RESTART' if protected else 'REVIEW_REQUIRED' if stale or failed
                  else 'OBSERVE_EXISTING_OWNER')
        result['rl_runs'].append(dict(id=run['id'], status=status, host=run.get('host'),
            committed_registry_updates=run.get('completed_updates'),
            observed_rank_updates=rank_updates,
            journal_updates=min(rank_updates) if len(rank_updates) == 8 and all(type(n) is int for n in rank_updates) else None,
            stale_active_warning=stale, terminal_worker_status=terminal.get('status'), action=action,
            owner_state=run.get('owner_state')))
    result['automatic_action_failed'] = read(directory/'REPAIR_FAILURE.json')
    cadence = Path('/mnt/project/qwen-world-il-research-20261030/navtest_cadence_20261031')
    result['navtest_cadence'] = dict(registration=read(cadence/'REGISTRATION.json'),
        owner=read(cadence/'OWNER.json'), state=read(cadence/'STATE.json'))
    atomic(root/'maintenance/HEALTH_LATEST.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--recover-huber', action='store_true', required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(recover_huber(args.root)))
    except BaseException:
        import traceback
        atomic(args.root/'maintenance/huber_recovery_20261031/REPAIR_FAILURE.json',
               dict(at=time.time(), error=traceback.format_exc(), automatic_retry=False))
        raise
