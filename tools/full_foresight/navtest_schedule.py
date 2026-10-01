"""Checkpoint preservation and validation for an independent Navtest observer.

Only external evaluation artifacts are written. Training directories are read-only.
"""
import contextlib
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid

SCHEMA = 'ddp_navtest_milestone_schedule_v1'
TERMINAL = {'COMPLETE', 'FAILED', 'MISSED_CHECKPOINT'}


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def source_identity(source):
    source = Path(source)
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=source).strip():
        raise ValueError('Evaluation source must be a committed, clean checkout')
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()


@contextlib.contextmanager
def lease(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield stream


def busy(path):
    try:
        with lease(path):
            return False
    except BlockingIOError:
        return True


def live_owner(record, registration, role, task=None):
    """No PID alone is accepted as proof of our own process."""
    try:
        pid = int(record['pid'])
        args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        args = [x.decode() for x in args if x]
        state = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0]
        return (state != 'Z' and 'tools.full_foresight.navtest_milestones' in args
                and role in args and str(Path(registration).resolve()) in args
                and (task is None or task in args))
    except (KeyError, FileNotFoundError, ProcessLookupError, ValueError):
        return False


def validate(config):
    keys = {'schema', 'evaluation_source_sha', 'source_worktree', 'artifact_root',
            'campaign_root', 'campaign_gpu_hours', 'evaluation_gpu_hours',
            'updates', 'models', 'current_root', 'metric_index', 'devkit',
            'cache_snapshot', 'cache_audit', 'ego_labels', 'inference_python',
            'scoring_python', 'worker_python', 'sampling_seed', 'gpus',
            'cpu_workers', 'gpu_memory_fraction', 'minimum_free_mib',
            'poll_seconds', 'task_timeout_seconds', 'watch_timeout_seconds',
            'pressure_script', 'pressure_python', 'baseline_scores'}
    if set(config) != keys:
        raise ValueError('Unknown/missing schedule fields: ' + str(set(config) ^ keys))
    if (config['schema'] != SCHEMA or config['updates'] != [70000, 80000, 90000, 100000]
            or config['sampling_seed'] != 42 or config['gpus'] != list(range(8))
            or not 0 < config['gpu_memory_fraction'] <= .45
            or not 1 <= config['cpu_workers'] <= 16
            or min(config[k] for k in ('campaign_gpu_hours', 'evaluation_gpu_hours',
                   'minimum_free_mib', 'poll_seconds', 'task_timeout_seconds', 'watch_timeout_seconds')) <= 0):
        raise ValueError('Invalid registered milestone/precision/resource protocol')
    if {m['arm'] for m in config['models']} != {'C0', 'C1', 'C4'} or len(config['models']) != 3:
        raise ValueError('One existing formal run per C0/C1/C4 required')
    if len({m['host'] for m in config['models']}) != 3:
        raise ValueError('One evaluation slot per authorized host required')
    for model in config['models']:
        if set(model) != {'arm', 'training_run', 'run_identity', 'training_source_sha',
                          'host', 'hostname', 'development_report'}:
            raise ValueError('Unknown model fields')
        identity = read(Path(model['training_run']) / 'identity.json')
        if (identity['identity'] != model['run_identity']
                or identity['source_sha'] != model['training_source_sha']
                or identity['schema'] != 'ddp_full_foresight_student_v1'
                or identity['scope'] != 'formal' or identity['updates'] != 100000
                or identity['config']['foresight']['arm'] != model['arm']
                or identity['world_size'] != 8 or identity['global_batch'] != 32):
            raise ValueError('Formal training identity changed')
    current = read(Path(config['current_root']) / 'identity.json')
    index = read(Path(config['current_root']) / 'index.json')
    metric = read(config['metric_index'])
    if (current['split'] != 'navtest' or len(index) != 12146
            or len({s['token'] for s in index}) != 12146
            or len({s['log'] for s in index}) != 136
            or {(s['token'], s['log']) for s in index} != {(s['token'], s['log']) for s in metric}
            or len(metric) != 12146 or not read(config['cache_audit'])['passed']):
        raise ValueError('Complete canonical Navtest assets required')
    return config


def load_registration(path):
    r = read(path)
    if set(r) != {'identity', 'config', 'created_unix', 'asset_hashes'} or r['identity'] != signature(r['config']):
        raise ValueError('Changed immutable registration')
    c = r['config']
    if source_identity(c['source_worktree']) != c['evaluation_source_sha']:
        raise ValueError('Registered evaluation source changed')
    for path, expected in r['asset_hashes'].items():
        if sha(path) != expected:
            raise ValueError('Registered asset changed: ' + path)
    return r


def job_key(arm, update):
    return f'{arm}_{update:06d}'


def find_exact(training_run, update):
    root = Path(training_run) / 'checkpoints'
    for tag in (f'milestone_{update:06d}', f'periodic_{update:06d}', f'final_{update:06d}'):
        folder = root / tag
        p = folder / 'COMPLETE.json'
        if p.exists():
            complete = read(p)
            if complete['completed'] != update or complete['tag'] != tag:
                raise ValueError('Wrong exact checkpoint contents')
            return folder
    return None


def preserve(model, update, job):
    """Hard-link a COMPLETE checkpoint before rolling GC; atomically publish it."""
    job = Path(job)
    dest = job / 'frozen_student'
    if (job / 'snapshot.json').exists():
        snapshot = read(job / 'snapshot.json')
        if snapshot['completed'] != update or snapshot['run_identity'] != model['run_identity']:
            raise ValueError('Foreign preserved snapshot')
        if read(dest / 'checkpoints' / snapshot['tag'] / 'COMPLETE.json')['identity'] != model['run_identity']:
            raise ValueError('Preserved checkpoint changed')
        if not (job / 'models.json').exists():
            atomic(job / 'models.json', [{'arm': model['arm'], 'training_run': str(dest),
                                          'checkpoint_tag': snapshot['tag']}])
        return snapshot
    if dest.exists():
        # Recover a crash between publishing the linked directory and metadata.
        folders = list((dest / 'checkpoints').iterdir())
        if len(folders) != 1:
            raise ValueError('Ambiguous preserved snapshot')
        folder = folders[0]
        complete = read(folder / 'COMPLETE.json')
        identity = read(dest / 'identity.json')
        if (identity['identity'] != model['run_identity'] or complete['identity'] != model['run_identity']
                or complete['completed'] != update or complete['tag'] != folder.name):
            raise ValueError('Foreign unpublished snapshot')
        sizes = {p.name: p.stat().st_size for p in folder.iterdir() if p.is_file()}
        if (len([n for n in sizes if 'optim_states.pt' in n]) != 8
                or not any('model_states.pt' in n for n in sizes)
                or not {f'rng_rank{r}.pt' for r in range(8)} <= sizes.keys()):
            raise ValueError('Incomplete unpublished snapshot')
        snapshot = {'completed': update, 'tag': folder.name, 'exposure': complete['exposure'],
                    'run_identity': model['run_identity'], 'source_training_run': model['training_run'],
                    'training_run': str(dest), 'file_sizes': sizes, 'bytes': sum(sizes.values()),
                    'preserved_unix': time.time(), 'metadata_recovered': True}
        atomic(job / 'snapshot.json', snapshot)
        atomic(job / 'models.json', [{'arm': model['arm'], 'training_run': str(dest),
                                      'checkpoint_tag': folder.name}])
        return snapshot
    source = find_exact(model['training_run'], update)
    if source is None:
        return None
    complete = read(source / 'COMPLETE.json')
    if complete['identity'] != model['run_identity']:
        raise ValueError('Foreign training checkpoint')
    files = list(source.iterdir())
    names = {p.name for p in files}
    if (len([n for n in names if 'optim_states.pt' in n]) != 8
            or not any('model_states.pt' in n for n in names)
            or not {f'rng_rank{r}.pt' for r in range(8)} <= names):
        raise ValueError('Incomplete optimizer/model/RNG checkpoint')
    stage = job / ('.snapshot_' + uuid.uuid4().hex)
    stage.mkdir(parents=True)
    try:
        shutil.copy2(Path(model['training_run']) / 'identity.json', stage / 'identity.json')
        folder = stage / 'checkpoints' / source.name
        folder.mkdir(parents=True)
        sizes = {}
        for path in files:
            if not path.is_file() or path.is_symlink():
                raise ValueError('Unexpected checkpoint entry')
            os.link(path, folder / path.name)  # Never copy or mutate trainer files.
            sizes[path.name] = (folder / path.name).stat().st_size
        os.replace(stage, dest)
        snapshot = {'completed': update, 'tag': source.name, 'exposure': complete['exposure'],
                    'run_identity': model['run_identity'], 'source_training_run': model['training_run'],
                    'training_run': str(dest), 'file_sizes': sizes,
                    'bytes': sum(sizes.values()), 'preserved_unix': time.time()}
        atomic(job / 'snapshot.json', snapshot)
        atomic(job / 'models.json', [{'arm': model['arm'], 'training_run': str(dest),
                                      'checkpoint_tag': source.name}])
        return snapshot
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)  # Only our unpublished, uniquely named staging directory.
        raise


def host_ready(config, model):
    """Let the existing fixed development export finish before borrowing its GPUs."""
    state = read(Path(model['training_run']) / 'status.json')
    if state['status'] == 'RUNNING':
        return True
    if state['status'] == 'FAILED':
        return False
    queue = Path(config['campaign_root']) / 'priority_queues' / (Path(model['training_run']).name + '.json')
    if not queue.exists():
        return False
    return read(queue)['status'] == 'MAIN_TRAINING_COMPLETE_EVALUATION_TRACKED_SEPARATELY'


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as stream:
        if rows:
            w = csv.DictWriter(stream, sorted(set().union(*(r.keys() for r in rows))), lineterminator='\n')
            w.writeheader(); w.writerows(rows)
    os.replace(tmp, path)


def validate_scores(score_folder, snapshot_csv, expected_update):
    summary = read(Path(score_folder) / 'summary.json')
    rows = list(csv.DictReader((Path(score_folder) / 'scenes.csv').open()))
    original = {r['token']: r for r in csv.DictReader(Path(snapshot_csv).open())}
    if (not summary['valid'] or summary['failed'] or summary['scenes'] != 12146
            or summary['logs'] != 136 or len(rows) != 12146
            or len({r['token'] for r in rows}) != 12146
            or {r['token'] for r in rows} != set(original)
            or summary['export_identity']['checkpoint']['completed'] != expected_update
            or summary['export_identity']['protocol']['precision'] != 'FP32'):
        raise ValueError('Invalid/incomplete/wrong-checkpoint scores; retain all rows')
    for r in rows:
        if (r['status'] != 'ok' or r['metric_cache_sha256'] != original[r['token']]['sha256']
                or r['log'] != original[r['token']]['log']):
            raise ValueError('Failed scene or changed original cache')
        v = [float(r[k]) for k in ('score', 'no_at_fault_collisions', 'drivable_area_compliance',
                                 'ego_progress', 'time_to_collision_within_bound', 'comfort')]
        if not all(math.isfinite(x) for x in v) or abs(v[0] - v[1]*v[2]*(5*v[3]+5*v[4]+2*v[5])/12) > 1e-10:
            raise ValueError('Invalid score formula or nonfinite metrics')
    return summary, rows
