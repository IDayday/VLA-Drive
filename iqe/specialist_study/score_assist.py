"""Bounded CPU collaboration on explicitly record-locked official score jobs.

This never creates a job, trajectory, context or label. Legacy jobs without the
protocol receipt are excluded. The primary coordinator remains responsible for
validating every response and declaring the stage complete.
"""
import argparse
import os
from pathlib import Path
import socket
import sys
import time

from iqe.contracts import require
from iqe.io import atomic_json, file_hash, lock, read_json


def compatible_job(path, worker_code):
    marker = path.parent / 'COOPERATIVE_PROTOCOL.json'
    if not marker.exists():
        return None
    record = read_json(marker)
    require(record['schema'] == 'record_locked_official_score_v1', 'unsupported score collaboration protocol')
    require(record['job_hash'] == file_hash(path), 'cooperative job contents changed')
    require(record['worker_code_hash'] == file_hash(worker_code), 'cooperative worker code changed')
    job = read_json(path)
    require(1 <= job['attempts_per_invocation'] <= 3, 'bounded original scoring retry budget required')
    return job


def pending_items(job, worker, workers):
    # Reverse traversal reduces contention with the original forward workers.
    # Striding is execution scheduling only; the primary manifest order is kept.
    return [item for item in reversed(job['items'][worker::workers])
            if not Path(item['response']).with_suffix('.COMPLETE.json').exists()]


def run(config_path, worker, workers, deadline_hours):
    from .config import load_config
    from .resource_guard import check
    from iqe.internal_baseline import score_worker
    require(0 <= worker < workers <= 16 and 0 < deadline_hours <= 36, 'bounded CPU helper allocation required')
    config = load_config(config_path)
    check(config, 'cooperative-official-scoring')
    folder = config.root / 'runtime/cooperative_scoring' / socket.gethostname()
    folder.mkdir(parents=True, exist_ok=True)
    name = f'helper_{worker}_of_{workers}'
    deadline = time.monotonic() + deadline_hours * 3600
    processed = set()
    totals = {'records_scored_or_reused': 0, 'failed': 0, 'jobs': 0}
    # This is a task mutex, not a global server reservation.
    with lock(folder / (name + '.RUN.lock')):
        atomic_json(folder / (name + '.LAUNCH.json'), {
            'pid': os.getpid(), 'host': socket.gethostname(), 'worker': worker, 'workers': workers,
            'started_unix': time.time(), 'deadline_hours': deadline_hours,
            'worker_code_hash': file_hash(score_worker.__file__), 'config_hash': config.identity,
            'official_numeric_function_changed': False, 'legacy_jobs_excluded': True})
        while time.monotonic() < deadline:
            paths = sorted((config.root / 'scoring_jobs').glob('*/COOPERATIVE_PROTOCOL.json'),
                           key=lambda p: p.stat().st_mtime_ns)
            activity = False
            for marker in paths:
                path = marker.parent / 'JOB.json'
                key = str(path)
                if key in processed:
                    continue
                job = compatible_job(path, score_worker.__file__)
                todo = pending_items(job, worker, workers)
                processed.add(key)
                if not todo:
                    continue
                # A fresh scorer/simulator is still instantiated per record by
                # the original backend; only heavyweight Python imports persist.
                sys.path.insert(0, job['nuplan_root'])
                sys.path.insert(0, job['navsim_root'])
                import nuplan
                require(Path(nuplan.__file__).resolve().is_relative_to(Path(job['nuplan_root']).resolve()),
                        'wrong NuPlan runtime')
                from iqe.scoring.worker import main as official_single
                start = time.monotonic()
                done = failed = skipped_failed = 0
                for item in todo:
                    response = Path(item['response'])
                    # Helpers do not revisit prior failed attempts. Primary
                    # retry and error policy remain authoritative and visible.
                    attempt_dir = response.parent / 'attempts' / response.stem
                    if list(attempt_dir.glob('attempt_*.json')):
                        skipped_failed += 1
                        continue
                    ok = score_worker.score_item(job, item, official_single)
                    done += int(ok)
                    failed += int(not ok)
                    if (done + failed) % 20 == 0:
                        atomic_json(folder / (name + '.STATUS.json'), {
                            'status': 'RUNNING', 'job': key, 'done': done, 'failed': failed,
                            'skipped_prior_attempts': skipped_failed, 'seconds': time.monotonic() - start,
                            'updated_unix': time.time(), **totals})
                    require(time.monotonic() < deadline, 'bounded helper deadline reached; primary jobs continue')
                result = {'job': key, 'job_hash': file_hash(path), 'done': done, 'failed': failed,
                          'skipped_prior_attempts': skipped_failed, 'seconds': time.monotonic() - start,
                          'finished_unix': time.time()}
                atomic_json(folder / (name + '_' + marker.parent.name + '.json'), result, immutable=True)
                totals['jobs'] += 1
                totals['records_scored_or_reused'] += done
                totals['failed'] += failed
                activity = True
            status_path = config.root / 'runtime/completion_driver/STATUS.json'
            if status_path.exists() and read_json(status_path)['stage'].startswith('COMPLETE'):
                atomic_json(folder / (name + '.STATUS.json'), {'status': 'COMPLETE', **totals})
                return
            if not activity:
                atomic_json(folder / (name + '.STATUS.json'), {'status': 'WAITING_NEW_COOPERATIVE_JOB',
                            'updated_unix': time.time(), **totals})
                time.sleep(20)
        atomic_json(folder / (name + '.STATUS.json'), {'status': 'BOUNDED_WINDOW_COMPLETE', **totals})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--worker', required=True, type=int)
    parser.add_argument('--workers', required=True, type=int)
    parser.add_argument('--deadline-hours', type=float, default=12.)
    args = parser.parse_args()
    run(args.config, args.worker, args.workers, args.deadline_hours)


if __name__ == '__main__':
    main()
