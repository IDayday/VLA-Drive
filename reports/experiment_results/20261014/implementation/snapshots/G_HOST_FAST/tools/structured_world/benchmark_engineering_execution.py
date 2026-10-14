"""Bounded native-origin benchmark: six alternating IO/fast 220-update jobs.

Default --manifest INPUT.json --output NEW_DIR only writes a reviewable plan.
An operator must explicitly add --execute ON the qualified host. No SSH,
Navtest, job retries, process termination or checkpoint deletion. Explicit
--continue-from validates and references a completed prefix in a fresh output.

Manifest: source_root, source_sha (full frozen commit), python, host,
gpu_uuids (eight, rank order), qualification {path, sha256}, artifact_root,
minimum_free_MiB, budget {GPU_hours_limit<=120, already_spent, reserved_elsewhere,
per_job_reserve_GPU_hours}, candidate_mode (optional: engineering_fast default,
engineering_timevec, precision_fast or engineering_host_fast; one candidate only), and cases (1..4). Each case contains name,
pilot_origin_run, pilot_origin_tag, origin_identity, origin_complete_sha256,
train_args: an argv list of --config/--cache/--geometry/--geometry-identity/
--dino-root/--dino-identity/--image-root/--micro-batch pairs. All six repetitions
restore the SAME native model/optimizer/LR/data/RNG prefix. Outputs are private.

Qualification must declare passed:true, host and physical_gpu_UUIDs. Optional
environment.packages is checked against the actual private Python runtime.
The frozen trainer must support --benchmark-no-end-save. An owned worker shim
only runs that trainer and reads cumulative allocated/reserved peaks at exit.
IO wall includes rank0 stdout arrivals; fast wall includes true log flushes.
"""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import runpy
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time

TASK = 'structured_world_fgtr_round1'
POLICY = Path('/mnt/project/server_dispatch_policy.json')
MODES = ('io_preserving_v1', 'engineering_fast')
CANDIDATES = ('engineering_fast', 'engineering_timevec', 'precision_fast', 'engineering_host_fast')
BASE_FLAGS = {'--config', '--cache', '--geometry', '--geometry-identity',
              '--dino-root', '--dino-identity', '--image-root', '--micro-batch'}
PACKAGES = ('torch', 'transformers', 'deepspeed', 'mmcv', 'mmengine', 'timm', 'omegaconf', 'numpy')
SCRIPT = Path(__file__).resolve()


def require(condition, message):
    if not condition: raise ValueError(message)


def read(path): return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x') as stream: json.dump(value, stream, indent=2)


def checked(command, cwd=None, env=None, timeout=None):
    return subprocess.check_output(command, cwd=cwd, env=env, text=True, timeout=timeout).strip()


def immutable_source(root, sha):
    require(re.fullmatch('[0-9a-f]{40}', sha), 'Supply the full frozen commit')
    require(checked(['git', 'rev-parse', 'HEAD'], root) == sha, 'Frozen source commit changed')
    require(not checked(['git', 'status', '--porcelain'], root), 'Frozen source must be clean')


def authorize(host):
    policy = read(POLICY)
    require(host in policy['task_authorizations'][TASK]['allowed_hosts'], 'Current task policy rejects host')
    require(host not in policy.get('global_dispatch_prohibitions', {}), 'Host withdrawn for all new work')
    reservation = policy.get('host_reservations', {}).get(host, {})
    require(not (reservation.get('exclude_new_other_workloads') and reservation.get('task') != TASK),
            'Host reserved for another active task; a benchmark needs a coordinated idle window')
    return {'path': str(POLICY), 'sha256': digest(POLICY),
            'task_authorization': policy['task_authorizations'][TASK], 'host_reservation': reservation}


def origin(case):
    run = Path(case['pilot_origin_run']).resolve()
    tag = case['pilot_origin_tag']
    require(isinstance(tag, str) and Path(tag).name == tag and not tag.startswith('writing_'), 'Unsafe origin tag')
    contract, folder = read(run/'identity.json'), run/'checkpoints'/tag
    complete = read(folder/'COMPLETE.json')
    require(digest(folder/'COMPLETE.json') == case['origin_complete_sha256'], 'Origin COMPLETE changed')
    require(contract['identity'] == complete['identity'] == case['origin_identity'], 'Foreign native origin identity')
    require(complete['ranks'] == contract['world_size'] == 8 and complete['tag'] == tag, 'Full8 native origin required')
    require(contract['effective_global_batch'] == 32 and contract['FM_repeat'] == 8 and contract['training_seed'] == 42,
            'Original global32/FM8/seed42 contract required')
    expected = {'mp_rank_00_model_states.pt'} | {f'rng_rank{r}.pt' for r in range(8)} | {
        f'zero_pp_rank_{r}_mp_rank_00_optim_states.pt' for r in range(8)}
    require(set(complete['file_sha256']) == expected, 'Native 17-file origin required')
    require(all(re.fullmatch('[0-9a-f]{64}', value) for value in complete['file_sha256'].values()), 'Malformed native payload digest')
    require(all((folder/p).is_file() and (folder/p).stat().st_size > 0 for p in expected), 'Origin payload missing')
    require(contract['scope'] in ('formal', 'profile') and contract['dataset'] == 'navsim'
            and contract['schedule_horizon'] == 100000 and contract['schedule_warmup'] == 5000,
            'Origin must preserve native NAV 100k/5000 schedule')
    require(contract['scenes_per_epoch'] == contract['source_population_scenes'] == 101592
            and contract['engineering_training_selection'] is None, 'Full original training population required')
    recipe = contract['config']['structured_world']['formal_recipe_assets']
    require(recipe == {'label_cache_identity': contract['cache_identity'], 'current_DINO_identity': contract['DINO_identity'],
                       'shared_geometry_identity': contract['shared_geometry_identity']}, 'Origin frozen asset/config mismatch')
    require(contract['config']['framework']['action_model']['num_inference_timesteps'] == 10,
            'Original 10-step action contract required')
    require(0 <= complete['completed'] and complete['completed']+220 <= contract['schedule_horizon'], 'Origin is outside native schedule')
    return {'run': str(run), 'tag': tag, 'complete_sha256': case['origin_complete_sha256'],
            'complete': complete, 'contract': contract, 'origin_scope': contract['scope'],
            'origin_source_sha': contract['training_source_sha'], 'optimizer_groups_sha256': digest(run/'optimizer_groups.json'),
            'payload_verification': 'Inventory/metadata checked here; original trainer hashes every owned native payload and verifies all-rank boundary'}


def build_plan(manifest, output, driver=SCRIPT):
    root = Path(manifest['source_root']).resolve()
    cases = manifest['cases']
    candidate = manifest.get('candidate_mode', MODES[1])
    require(candidate in CANDIDATES, 'Choose exactly one supported candidate mode')
    modes = (MODES[0], candidate)
    require(1 <= len(cases) <= 4, 'Bounded manifest allows 6..24 jobs, six per case')
    require(len(set(manifest['gpu_uuids'])) == len(manifest['gpu_uuids']) == 8, 'Eight unique UUIDs required')
    require(all(uuid.startswith('GPU-') for uuid in manifest['gpu_uuids']), 'Physical GPU UUIDs required')
    require(Path(manifest['python']).is_absolute() and Path(driver).is_absolute(), 'Absolute runtime/driver paths required')
    require(Path(manifest['artifact_root']).is_absolute() and manifest['minimum_free_MiB'] > 0, 'Explicit resource reserve/lock root required')
    budget = manifest['budget']
    require(0 < budget['GPU_hours_limit'] <= 120 and min(budget['already_spent'], budget['reserved_elsewhere']) >= 0
            and budget['per_job_reserve_GPU_hours'] > 0 and all(math.isfinite(v) for v in budget.values()), 'Finite shared budget <=120 GPUh required')
    require(budget['already_spent']+budget['reserved_elsewhere']+len(cases)*6*budget['per_job_reserve_GPU_hours'] <= budget['GPU_hours_limit'],
            'Whole matrix reservation exceeds shared GPUh budget')
    jobs, names = [], set()
    for case in cases:
        name = case['name']
        require(re.fullmatch('[A-Za-z0-9_.-]+', name) and name not in names, 'Unique safe case names required')
        names.add(name)
        args = case['train_args']
        require(len(args) % 2 == 0 and all(isinstance(a, str) for a in args), 'train_args must be option/value argv pairs')
        flags = args[::2]
        require(set(flags) <= BASE_FLAGS and len(flags) == len(set(flags)), 'Only explicit scientific asset args allowed, without duplicates')
        values = dict(zip(flags, args[1::2]))
        require(set(flags) == BASE_FLAGS and values['--micro-batch'] in ('1', '2', '4'), 'Explicit complete frozen asset paths and microbatch required')
        require(Path(case['pilot_origin_run']).is_absolute(), 'Absolute native origin run required')
        for repeat in range(6):
            mode = modes[repeat % 2]
            key = name+'_'+str(repeat+1).zfill(2)+'_'+mode
            run = output/'runs'/key
            original_args = args+['--scope', 'profile', '--debug-updates', '220',
                '--pilot-origin-run', case['pilot_origin_run'], '--pilot-origin-tag', case['pilot_origin_tag'],
                '--benchmark-no-end-save', '--execution-mode', mode, '--logging-every', '20',
                '--evidence-every', '100', '--save-every', '2147483647', '--output', str(run)]
            launcher = [manifest['python'], '-m', 'torch.distributed.run', '--nnodes=1', '--nproc_per_node=8',
                        '--max_restarts=0', '--master_port='+str(manifest.get('master_port', 29580))]
            require(1024 <= manifest.get('master_port', 29580) <= 65535, 'Invalid torchrun port')
            # -- must precede each script: torchrun otherwise consumes --run as
            # an ambiguous abbreviation of its own --run-path option.
            jobs.append({'key': key, 'case': name, 'mode': mode, 'run': str(run), 'train_args': original_args,
                'original_command': launcher+['--', str(root/'tools/structured_world/train_vla.py')]+original_args,
                'command': launcher+['--', str(driver), '--worker', '--trainer', str(root/'tools/structured_world/train_vla.py'),
                                    '--run', str(run), '--']+original_args})
    return {'schema': 'engineering_execution_benchmark_plan_v1', 'manifest': manifest, 'source_root': str(root),
            'source_sha': manifest['source_sha'], 'driver': str(driver), 'driver_sha256': digest(driver),
            'candidate_mode': candidate, 'mode_pair': list(modes),
            'warmup_updates': 20, 'stable_updates': 200, 'total_jobs': len(jobs), 'maximum_job_updates': 220,
            'maximum_total_updates': len(jobs)*220, 'jobs': jobs}


def runtime_snapshot(python):
    code = 'import sys,json,importlib.metadata as m;print(json.dumps({"python":sys.version,"executable":sys.executable,"prefix":sys.prefix,"packages":{n:m.version(n) for n in '+repr(PACKAGES)+'}}))'
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''
    return json.loads(checked([python, '-B', '-c', code], env=env))


def readiness(manifest, record_sample=None):
    policy = authorize(manifest['host'])
    require(socket.gethostname().removesuffix('-worker-0') == manifest['host'], 'Execute on the qualified host, no remote dispatch')
    started, samples = time.monotonic(), []
    def record(sample):
        sample.update(attempt=len(samples)+1, observed_unix=time.time(), elapsed_seconds=time.monotonic()-started)
        samples.append(sample)
        if record_sample is not None: record_sample(sample)
    while True:
        sample = {'gpus': [], 'compute_apps': []}
        try:
            timeout = lambda: max(.001, min(2., 10.-(time.monotonic()-started)))
            rows = checked(['nvidia-smi', '--query-gpu=uuid,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], timeout=timeout()).splitlines()
            for row in rows:
                uuid, used, free, util = [value.strip() for value in row.split(',')]
                sample['gpus'].append(dict(uuid=uuid, used_MiB=int(used), free_MiB=int(free), utilization=int(util)))
            sample['compute_apps'] = checked(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv,noheader,nounits'], timeout=timeout()).splitlines()
        except Exception as exc:
            record({**sample, 'decision': 'query_failed', 'error': repr(exc)})
            raise
        gpus, apps = sample['gpus'], sample['compute_apps']
        inventory = len(gpus) == 8 and {g['uuid'] for g in gpus} == set(manifest['gpu_uuids'])
        memory = all(g['used_MiB'] <= 128 and g['free_MiB'] >= manifest['minimum_free_MiB'] for g in gpus)
        idle = all(g['utilization'] <= 5 for g in gpus)
        sample['decision'] = 'occupied' if apps or not memory else 'inventory_changed' if not inventory else 'ready' if idle else 'settling_utilization'
        record(sample)
        require(not apps, 'GPU compute app present; no settling, process stopped or job retried')
        require(inventory, 'Physical UUID inventory changed')
        require(memory, 'GPU memory occupied/insufficiently free; no settling or process stopped')
        if idle:
            return {'policy': policy, **sample, 'samples': samples, 'settle_limit_seconds': 10., 'sample_interval_seconds': .5}
        remaining = 10.-(time.monotonic()-started)
        require(remaining > 0, 'Empty-app GPU utilization did not settle within 10 seconds; samples retained, no job started')
        time.sleep(min(.5, remaining))


def acquire(stack, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = stack.enter_context(path.open('a'))
    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return stream


def json_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def summary(values):
    require(values and all(math.isfinite(v) and v > 0 for v in values), 'Invalid/missing duration samples')
    ordered = sorted(values)
    at = (len(ordered)-1)*.9
    p90 = ordered[math.floor(at)]*(math.ceil(at)-at)+ordered[math.ceil(at)]*(at-math.floor(at)) if at % 1 else ordered[int(at)]
    return {'samples': len(values), 'median_seconds': statistics.median(values), 'P90_seconds': p90, 'mean_seconds': statistics.mean(values)}


class BoundaryTimer:
    """Observe optimizer boundaries only; no RNG or per-update synchronization."""
    def __init__(self, synchronize, maximum, clock=time.monotonic):
        self.synchronize, self.maximum, self.clock = synchronize, maximum, clock
        self.count, self.start, self.local_seconds, self.max_rank_seconds = 0, None, None, None

    def after_step(self, boundary):
        if not boundary: return
        self.count += 1
        if self.count in (20, 220):
            self.synchronize()
            observed = self.clock()
            if self.count == 20: self.start = observed
            else:
                self.local_seconds = observed-self.start
                self.max_rank_seconds = self.maximum(self.local_seconds)


def analyze_job(job, native, process):
    run, start = Path(job['run']), native['complete']['completed']
    identity, completed, boundary = read(run/'identity.json'), read(run/'TRAINING_COMPLETE.json'), read(run/'RESUME_BOUNDARY_VERIFIED.json')
    expected = native['contract']
    for key in ('config', 'cache_identity', 'current_and_GT_identity', 'DINO_identity', 'shared_geometry_identity',
                'training_seed', 'world_size', 'effective_global_batch', 'FM_repeat', 'microbatch', 'dataset',
                'scenes_per_epoch', 'updates_per_epoch', 'schedule_horizon', 'schedule_warmup', 'determinism', 'precision', 'initialization_assets'):
        require(identity[key] == expected[key], 'Pilot changed native scientific contract: '+key)
    ancestry = identity['engineering_continue_origin']
    require(ancestry['identity'] == expected['identity'] and ancestry['tag'] == native['tag'] and ancestry['complete_sha256'] == native['complete_sha256'], 'Pilot ancestry changed')
    require(identity['benchmark_no_end_checkpoint'] and identity['execution_mode'] == job['mode'], 'Incorrect benchmark mode/save policy')
    require(completed['identity'] == boundary['identity'] == identity['identity'] and completed['updates'] == start+220,
            'Pilot did not finish the exact 220 updates')
    require(boundary['passed'] and boundary['completed'] == start and boundary['all_task_RNG'] == 'exact all ranks', 'Native boundary/RNG proof missing')
    require(digest(run/'optimizer_groups.json') == native['optimizer_groups_sha256'], 'Actual optimizer/LR groups changed')
    require(not (run/'checkpoints').exists() or not list((run/'checkpoints').iterdir()), 'Disposable benchmark unexpectedly saved checkpoints')
    metrics = json_lines(run/'metrics.jsonl')
    require([row['updates'] for row in metrics] == list(range(start+1, start+221)), 'Metric population incomplete/duplicated')
    arrivals = json_lines(run/'stdout_metric_arrivals.jsonl')
    require([row['updates'] for row in arrivals] == (list(range(start+1, start+221)) if job['mode'] == MODES[0] else list(range(start+20, start+221, 20))), 'Metric stdout arrival population changed')
    clocks = {row['updates']: row['monotonic_ns']/1e9 for row in arrivals}
    stable = metrics[20:]
    scenes = metrics[-1]['scene_exposure']-metrics[19]['scene_exposure']
    peaks = [read(run/f'worker_rank{rank}.json') for rank in range(8)]
    require(all(p['passed'] and p['rank'] == rank and p['GPU_UUID'] == job['gpu_uuids'][rank] for rank, p in enumerate(peaks)), 'Missing/mismatched full8 worker completion')
    require(all(p['optimizer_boundaries'] == 220 for p in peaks), 'Incorrect actual optimizer boundary population')
    require(all(math.isfinite(p['stable_200_local_wall_seconds']) and p['stable_200_local_wall_seconds'] > 0
                and p['peak_allocated_bytes'] > 0 and p['peak_reserved_bytes'] >= p['peak_allocated_bytes'] for p in peaks), 'Invalid rank wall/peak receipt')
    end_wall = max(p['stable_200_local_wall_seconds'] for p in peaks)
    require(end_wall > 0 and all(p['stable_200_max_rank_wall_seconds'] == end_wall for p in peaks), 'All-rank boundary wall proof missing')
    if job['mode'] != MODES[0]:
        windows = json_lines(run/'performance_windows.jsonl')
        require(len(windows) == 11 and all(w['start_update'] == start+20*i+1 and w['end_update'] == start+20*(i+1) and w['updates'] == 20 for i, w in enumerate(windows)), 'Exact warm20/stable200 window boundaries required')
        selected = windows[1:]
        trainer_window_wall = sum(w['wall_seconds_max_rank_including_log_flush'] for w in selected)
        window_rates = [w['wall_seconds_max_rank_including_log_flush']/20 for w in selected]
        durations = [d for w in selected for d in w['update_current_stream_seconds_max_rank']]
        require(sum(w['scene_exposure'] for w in selected) == scenes, 'Stable scene exposure mismatch')
        trainer_max_rank_seconds = sum(durations)
    else:
        trainer_window_wall = None
        window_rates = [(clocks[start+20*(i+1)]-clocks[start+20*i])/20 for i in range(1, 11)]
        durations = [clocks[u]-clocks[u-1] for u in range(start+21, start+221)]
        trainer_max_rank_seconds = sum(row['seconds'] for row in stable)
    wall = end_wall
    timing = 'identical all-rank MAX optimizer boundary20->220 wall for both modes; includes log20 through219, excludes final update220 log and end-edge MAX collective'
    require(len(durations) == 200 and wall > 0 and scenes > 0, 'Invalid stable measurement')
    return {'key': job['key'], 'case': job['case'], 'mode': job['mode'], 'passed': True,
        'stable_updates': 200, 'stable_scene_exposure': scenes, 'stable_wall_seconds': wall,
        'stable_seconds_per_update': wall/200, 'trainer_window_wall_seconds': trainer_window_wall,
        'stable_200_window_max_rank_wall_seconds': end_wall,
        'all_rank_stable_200_wall_seconds': [p['stable_200_local_wall_seconds'] for p in peaks],
        'boundary_timing_semantics': 'CUDA sync only after optimizer step20 and220; MAX collective at220 outside measured span; final step log excluded',
        'stable_200_window_rank0_stdout_wall_seconds': clocks[start+220]-clocks[start+20],
        'timing_semantics': timing, 'stable_global_scenes_per_second': scenes/wall,
        'wall_per_update_including_logs': summary(window_rates), 'update_intervals': summary(durations),
        'update_interval_semantics': 'max-rank CUDA stream events, excludes final log flush' if job['mode'] != MODES[0] else 'rank0 stdout arrival deltas, includes logging',
        'sum_trainer_max_rank_update_seconds': trainer_max_rank_seconds,
        'all_rank_peak_allocated_reserved_bytes': [[p['peak_allocated_bytes'], p['peak_reserved_bytes']] for p in peaks],
        'process_wall_seconds': process['wall_seconds'], 'process_GPU_hours': process['wall_seconds']*8/3600,
        'startup_through_native_boundary_seconds': process.get('native_boundary_observed_seconds'),
        'startup_observation_poll_seconds': .05,
        'startup_through_first_metric_arrival_seconds': arrivals[0]['since_process_start_seconds'],
        'startup_includes': 'Python/DS/model/native hash+restore+boundary plus first update (IO) or first20 updates (fast)',
        'checkpoint_save_seconds': 0., 'checkpoint_save_policy': 'read-only origin; no new checkpoint; native hash/load costs included in startup',
        'origin_checkpoint_save_seconds': native['complete'].get('save_seconds'),
        'checkpoint_cost_limit': 'Historical origin save_seconds is disclosed separately; no new save timing or amortization is inferred from these disposable windows',
        'post_last_stdout_until_process_exit_seconds': process['wall_seconds']-arrivals[-1]['since_process_start_seconds'],
        'native_boundary': boundary, 'run_identity': identity['identity'],
        'source': {key: identity[key] for key in ('training_source_sha', 'training_source_tree_sha256', 'config', 'cache_identity', 'current_and_GT_identity', 'DINO_identity', 'shared_geometry_identity', 'initialization_assets')},
        'artifacts': {'run': str(run), 'stdout': str(run/'stdout.log'), 'arrivals': str(run/'stdout_metric_arrivals.jsonl')}}


def run_job(job, locks):
    run = Path(job['run'])
    run.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1',
               PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(Path(job['trainer']).parents[2]), CUDA_VISIBLE_DEVICES=','.join(job['gpu_uuids']))
    started, unix = time.monotonic_ns(), time.time()
    finished, boundary_observed = threading.Event(), []
    def observe_boundary():
        while not finished.is_set():
            if (run/'RESUME_BOUNDARY_VERIFIED.json').exists():
                boundary_observed.append((time.monotonic_ns()-started)/1e9); return
            finished.wait(.05)
    with (run/'stdout.log').open('x') as log, (run/'stdout_metric_arrivals.jsonl').open('x') as arrivals:
        proc = subprocess.Popen(job['command'], env=env, cwd=Path(job['trainer']).parents[2], stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, errors='replace', bufsize=1, start_new_session=True, pass_fds=tuple(lock.fileno() for lock in locks))
        observer = threading.Thread(target=observe_boundary, daemon=True); observer.start()
        monitor_error = None
        try: write(run/'PROCESS.json', {'pid': proc.pid, 'started_unix': unix, 'command': job['command']})
        except OSError as exc: monitor_error = repr(exc)
        try:
            for line_number, line in enumerate(proc.stdout, 1):
                now = time.monotonic_ns()
                if monitor_error is not None: continue  # Drain while actual children finish; never drop their lease.
                try: log.write(line); log.flush()
                except OSError as exc: monitor_error = repr(exc); continue
                try: record = json.loads(line)
                except (ValueError, TypeError): continue
                if isinstance(record, dict) and isinstance(record.get('updates'), int) and ('seconds' in record or 'seconds_per_update' in record):
                    try:
                        arrivals.write(json.dumps({'updates': record['updates'], 'monotonic_ns': now, 'observed_unix': time.time(),
                            'since_process_start_seconds': (now-started)/1e9, 'stdout_line': line_number})+'\n'); arrivals.flush()
                    except OSError as exc: monitor_error = repr(exc)
        finally:
            # A monitor failure must not drop the lease while its actual workers
            # continue. torchrun also inherits both FDs if this parent is lost.
            returncode = proc.wait()
            finished.set(); observer.join()
        outcome = {'returncode': returncode, 'wall_seconds': (time.monotonic_ns()-started)/1e9, 'pid': proc.pid,
                   'native_boundary_observed_seconds': boundary_observed[0] if boundary_observed else None, 'monitor_error': monitor_error}
        write(run/'PROCESS_EXIT.json', outcome)
        require(monitor_error is None, 'Metric monitor failed; actual children exited, logs retained')
        require(returncode == 0, 'Real worker stage failed; logs retained, no retry')
        return outcome


def worker():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--trainer', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    rank = int(os.environ['RANK']); local = int(os.environ['LOCAL_RANK'])
    sys.path.insert(0, str(args.trainer.resolve().parents[2]))
    import torch
    import torch.distributed as dist
    from deepspeed.runtime.engine import DeepSpeedEngine
    def maximum(seconds):
        value = torch.tensor(seconds, dtype=torch.float64, device=torch.device('cuda', local))
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
        return value.item()
    timer = BoundaryTimer(lambda: torch.cuda.synchronize(local), maximum)
    original_step = DeepSpeedEngine.step
    def timed_step(engine, *step_args, **kwargs):
        boundary = engine.is_gradient_accumulation_boundary()
        result = original_step(engine, *step_args, **kwargs)
        timer.after_step(boundary)
        return result
    DeepSpeedEngine.step = timed_step
    started = time.monotonic()
    passed = False
    try:
        sys.argv = [str(args.trainer)]+(args.args[1:] if args.args[:1] == ['--'] else args.args)
        runpy.run_path(str(args.trainer), run_name='__main__')
        passed = True
    finally:
        DeepSpeedEngine.step = original_step
        initialized = torch.cuda.is_initialized()
        write(args.run/f'worker_rank{rank}.json', {'rank': rank, 'passed': passed,
            'GPU_UUID': os.environ['CUDA_VISIBLE_DEVICES'].split(',')[local], 'trainer': str(args.trainer),
            'worker_wall_seconds': time.monotonic()-started,
            'optimizer_boundaries': timer.count, 'stable_200_local_wall_seconds': timer.local_seconds,
            'stable_200_max_rank_wall_seconds': timer.max_rank_seconds,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(local) if initialized else None,
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(local) if initialized else None})


def aggregate(results, candidate_mode=MODES[1]):
    modes_pair = (MODES[0], candidate_mode)
    cases = {}
    for result in results: cases.setdefault(result['case'], {}).setdefault(result['mode'], []).append(result)
    summary_cases = {}
    for case, modes in cases.items():
        row = {}
        for mode, jobs in modes.items():
            wall, scenes = sum(j['stable_wall_seconds'] for j in jobs), sum(j['stable_scene_exposure'] for j in jobs)
            row[mode] = {'finished_repetitions': len(jobs), 'stable_wall_seconds': wall, 'scene_exposure': scenes,
                         'global_scenes_per_second': scenes/wall, 'repetition_seconds_per_update': summary([j['stable_wall_seconds']/200 for j in jobs])}
        if set(row) == set(modes_pair) and all(row[m]['finished_repetitions'] == 3 for m in modes_pair):
            row['candidate_to_IO_throughput_ratio'] = row[candidate_mode]['global_scenes_per_second']/row[MODES[0]]['global_scenes_per_second']
            row['mode_pair'] = list(modes_pair)
        summary_cases[case] = row
    return summary_cases


def completed_prefix(plan, previous, natives):
    """Read-only verification; never admit an earlier failed/unfinished worker."""
    previous = Path(previous).resolve()
    old, registration, outcome = [read(previous/name) for name in ('MANIFEST.json', 'REGISTRATION.json', 'RESULTS.json')]
    require({k: v for k, v in old['manifest'].items() if k != 'candidate_mode'} ==
            {k: v for k, v in plan['manifest'].items() if k != 'candidate_mode'},
            'Continuation changed source/assets/origins/cases/resources/budget')
    for key in ('schema', 'source_root', 'source_sha', 'warmup_updates', 'stable_updates', 'total_jobs', 'maximum_job_updates', 'maximum_total_updates'):
        require(old[key] == plan[key], 'Continuation job/update/timing grid changed: '+key)
    candidate_changed = old['candidate_mode'] != plan['candidate_mode']
    require(not candidate_changed or all(r['mode'] == MODES[0] for r in outcome['jobs']),
            'Candidate switch only allowed for an entirely reference-only completed prefix')
    expected = build_plan(old['manifest'], previous, Path(old['driver']))
    require(all(old.get(k) == v for k, v in expected.items()), 'Previous plan job grid/commands/timing/source altered')
    require(registration['training_source_sha'] == plan['source_sha'] and registration['origins'] == natives
            and registration['budget'] == plan['manifest']['budget'], 'Previous registered native/scientific/budget identity changed')
    require(registration['driver_code_sha256'] == old['driver_sha256'], 'Previous driver identity changed')
    immutable_source(Path(old['driver']).parents[2], registration['driver_sha'])
    count = outcome['completed_jobs']
    require(type(count) is int and 0 < count <= plan['total_jobs'] and len(outcome['jobs']) == count
            and outcome['planned_jobs'] == plan['total_jobs'], 'Require a nonempty exact completed prefix')
    require(outcome['status'] in ('FAILED', 'STOPPED_BEFORE_NEXT_JOB', 'COMPLETE')
            and outcome['candidate_mode'] == old['candidate_mode'] and outcome['mode_pair'] == old['mode_pair'], 'Previous controller/result mode or completion state changed')
    require([r['key'] for r in outcome['jobs']] == [j['key'] for j in old['jobs'][:count]], 'Completed jobs are not the ordered prefix')
    inherited = read(previous/'CONTINUE_FROM.json') if (previous/'CONTINUE_FROM.json').exists() else {'jobs': [], 'completed_prefix': 0}
    require(outcome.get('reused_jobs', 0) == inherited['completed_prefix'] == len(inherited['jobs']) <= count, 'Mixed inherited prefix evidence')
    own_keys = {j['key'] for j in old['jobs'][len(inherited['jobs']):count]}
    runs = previous/'runs'
    require({p.name for p in runs.iterdir()} == own_keys if runs.exists() else not own_keys,
            'Earlier started/failed/unfinished or mixed worker exists; no actual job retry')
    require({p.name for p in previous.glob('*_RESULT.json')} == {j['key']+'_RESULT.json' for j in old['jobs'][:count]}, 'Missing/extra success receipt outside prefix')
    references, results = [], []
    for i, job in enumerate(old['jobs'][:count]):
        new_job = plan['jobs'][i]
        def scientific_args(j):
            args = list(j['train_args'])
            args[args.index('--output')+1] = '<private-output>'
            return args
        require(all(job[k] == new_job[k] for k in ('key', 'case', 'mode')) and scientific_args(job) == scientific_args(new_job),
                'Reused job case/mode/scientific/native/timing args changed')
        path = previous/(job['key']+'_RESULT.json')
        result = read(path)
        require(result == outcome['jobs'][i] and result['passed'], 'Success receipt differs from previous RESULTS')
        run = Path(result['artifacts']['run']).resolve()
        driver, driver_sha = Path(old['driver']), registration['driver_sha']
        if i < len(inherited['jobs']):
            ref = inherited['jobs'][i]
            require(ref['key'] == job['key'] and digest(ref['result_path']) == ref['result_sha256']
                    and read(ref['result_path']) == result and ref['run'] == str(run), 'Inherited success reference changed')
            driver, driver_sha = Path(ref['worker_driver']), ref['worker_driver_sha']
            require(digest(driver) == ref['worker_driver_code_sha256'], 'Inherited worker driver changed')
            immutable_source(driver.parents[2], driver_sha)
        else:
            require(run == Path(job['run']).resolve(), 'Success run is not owned by previous controller')
        process = read(run/'PROCESS_EXIT.json')
        require(process['returncode'] == 0 and process.get('monitor_error') is None, 'Earlier actual job failed; no retry')
        command = [str(run) if a == job['run'] else a for a in job['command']]
        command[command.index('--')+1] = str(driver)
        require(read(run/'PROCESS.json')['command'] == command, 'Success worker command differs from registered native job')
        actual_job = {**job, 'run': str(run), 'gpu_uuids': plan['manifest']['gpu_uuids']}
        require(result == analyze_job(actual_job, natives[job['case']], process), 'Success receipt cannot be reproduced from actual rank/boundary/window artifacts')
        require(result['source']['training_source_sha'] == plan['source_sha'], 'Success training source differs')
        require(all(read(run/f'worker_rank{rank}.json')['trainer'] == str(Path(plan['source_root'])/'tools/structured_world/train_vla.py') for rank in range(8)), 'Rank trainer source changed')
        proof_names = ['PROCESS.json', 'PROCESS_EXIT.json', 'RESUME_BOUNDARY_VERIFIED.json', 'identity.json',
                       'TRAINING_COMPLETE.json', 'optimizer_groups.json', 'metrics.jsonl', 'stdout_metric_arrivals.jsonl']
        proof_names += [f'worker_rank{rank}.json' for rank in range(8)]
        if job['mode'] != MODES[0]: proof_names.append('performance_windows.jsonl')
        references.append({'key': job['key'], 'result_path': str(path), 'result_sha256': digest(path), 'run': str(run),
            'worker_driver': str(driver), 'worker_driver_sha': driver_sha, 'worker_driver_code_sha256': digest(driver),
            'verified_artifact_sha256': {str(run/name): digest(run/name) for name in proof_names}})
        results.append(result)
    hours = sum(r['process_GPU_hours'] for r in results)
    require(math.isclose(outcome['process_GPU_hours'], hours, rel_tol=1e-12), 'Previous process budget includes unverified/failed work')
    budget = plan['manifest']['budget']
    require(budget['already_spent']+budget['reserved_elsewhere']+hours+(plan['total_jobs']-count)*budget['per_job_reserve_GPU_hours'] <= budget['GPU_hours_limit'], 'Continuation remaining matrix exceeds shared GPUh budget')
    files = ['MANIFEST.json', 'REGISTRATION.json', 'RESULTS.json']
    files += [name for name in ('FAILURE.json', 'CONTINUE_FROM.json') if (previous/name).exists()]
    return results, {'previous_output': str(previous), 'completed_prefix': count, 'remaining_jobs': plan['total_jobs']-count,
        'planned_candidate_changed': candidate_changed, 'previous_candidate_mode': old['candidate_mode'], 'candidate_mode': plan['candidate_mode'],
        'reused_process_GPU_hours': hours, 'previous_environment': registration['environment'], 'jobs': references,
        'previous_record_sha256': {str(previous/name): digest(previous/name) for name in files}, 'old_files_modified': False}


def main():
    if '--worker' in sys.argv[1:]:
        worker(); return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--continue-from', type=Path, help='Verify/reuse successful ordered prefix; fresh output only, no retry of any previously started incomplete job')
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.exists(), 'Output must be fresh; no retry/overwrite')
    manifest = read(args.manifest)
    plan = build_plan(manifest, output)
    immutable_source(plan['source_root'], plan['source_sha'])
    trainer = Path(plan['source_root'])/'tools/structured_world/train_vla.py'
    require("'--benchmark-no-end-save'" in trainer.read_text(), 'Need new immutable trainer commit supporting --benchmark-no-end-save')
    require(repr(plan['candidate_mode']) in trainer.read_text(), 'Frozen trainer does not support selected candidate mode')
    natives = {case['name']: origin(case) for case in manifest['cases']}
    results, continuation = [], None
    if args.continue_from:
        require(not output.is_relative_to(args.continue_from.resolve()), 'Continuation output must be outside previous artifacts')
        results, continuation = completed_prefix(plan, args.continue_from, natives)
    output.mkdir(parents=True, exist_ok=False)
    write(output/'MANIFEST.json', plan)
    if continuation:
        write(output/'CONTINUE_FROM.json', continuation)
        for result in results: write(output/(result['key']+'_RESULT.json'), result)
    if not args.execute:
        print(json.dumps({'status': 'PLANNED_ONLY', 'jobs': len(plan['jobs']), 'reused_jobs': len(results),
                          'remaining_jobs': len(plan['jobs'])-len(results), 'manifest': str(output/'MANIFEST.json')})); return 0
    reused = len(results)
    status, error = 'COMPLETE', None
    stop_requested = [False]
    def stop_after_current(*unused): stop_requested[0] = True
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, stop_after_current)
    try:
        driver_root = SCRIPT.parents[2]
        driver_sha = checked(['git', 'rev-parse', 'HEAD'], driver_root)
        immutable_source(driver_root, driver_sha)
        require(checked(['git', 'ls-files', '--', str(SCRIPT.relative_to(driver_root))], driver_root), 'Driver must be committed before execution')
        q = manifest['qualification']
        require(digest(q['path']) == q['sha256'], 'Qualification record changed')
        qualified = read(q['path'])
        require(qualified['passed'] and qualified['host'] == manifest['host'] and set(qualified['physical_gpu_UUIDs']) == set(manifest['gpu_uuids']), 'Host/full8 qualification mismatch')
        environment = runtime_snapshot(manifest['python'])
        if continuation: require(environment == continuation['previous_environment'], 'Private runtime differs from completed prefix')
        for name, value in qualified.get('environment', {}).get('packages', {}).items():
            require(environment['packages'].get(name) == value, 'Private runtime version differs: '+name)
        write(output/'REGISTRATION.json', {'manifest_sha256': digest(args.manifest), 'driver_sha': driver_sha,
            'driver_code_sha256': digest(SCRIPT), 'training_source_sha': plan['source_sha'], 'qualification': qualified,
            'qualification_sha256': q['sha256'], 'environment': environment, 'origins': natives,
            'budget': manifest['budget'], 'continue_from': continuation, 'no_formal_training_or_Navtest_dispatch': True})
        with ExitStack() as stack:
            root = Path(manifest['artifact_root'])
            queue_key = hashlib.sha256(json.dumps([plan['source_sha'], [[c['pilot_origin_run'], c['pilot_origin_tag']] for c in manifest['cases']]], sort_keys=True).encode()).hexdigest()
            physical_key = hashlib.sha256(json.dumps(sorted(manifest['gpu_uuids'])).encode()).hexdigest()
            locks = [acquire(stack, root/'engineering_benchmark_locks'/(queue_key+'.lock')),
                     acquire(stack, root/'physical_gpu_locks'/(physical_key+'.lock'))]
            for job in plan['jobs'][reused:]:
                if stop_requested[0] or (output/'STOP_BEFORE_NEXT_JOB').exists(): status = 'STOPPED_BEFORE_NEXT_JOB'; break
                budget = manifest['budget']
                spent = sum(result['process_GPU_hours'] for result in results)
                require(budget['already_spent']+budget['reserved_elsewhere']+spent+budget['per_job_reserve_GPU_hours'] <= budget['GPU_hours_limit'], 'GPUh admission budget exhausted; no next job')
                immutable_source(plan['source_root'], plan['source_sha'])
                require(digest(SCRIPT) == plan['driver_sha256'] and digest(q['path']) == q['sha256'], 'Immutable driver/qualification changed')
                native = origin(next(case for case in manifest['cases'] if case['name'] == job['case']))
                def record_sample(sample):
                    with (output/'READINESS_SAMPLES.jsonl').open('a') as stream:
                        stream.write(json.dumps({'job': job['key'], **sample})+'\n')
                gate = readiness(manifest, record_sample)
                write(output/(job['key']+'_RESOURCE_GATE.json'), gate)
                job.update(gpu_uuids=manifest['gpu_uuids'], trainer=str(trainer))
                process = run_job(job, locks)
                result = analyze_job(job, native, process)
                require(origin(next(case for case in manifest['cases'] if case['name'] == job['case'])) == native, 'Read-only origin changed during pilot')
                results.append(result)
                write(output/(job['key']+'_RESULT.json'), result)
                print(json.dumps({'job': job['key'], 'status': 'COMPLETE', 'scenes_per_second': result['stable_global_scenes_per_second']}), flush=True)
    except BaseException as exc:
        status, error = 'FAILED', repr(exc)
        write(output/'FAILURE.json', {'error': error, 'completed_jobs': len(results), 'logs_preserved': True})
    new_process_hours = sum(read(p)['wall_seconds']*8/3600 for p in (output/'runs').glob('*/PROCESS_EXIT.json'))
    process_hours = new_process_hours+(continuation['reused_process_GPU_hours'] if continuation else 0.)
    write(output/'RESULTS.json', {'status': status, 'error': error, 'completed_jobs': len(results), 'planned_jobs': len(plan['jobs']),
        'process_GPU_hours': process_hours, 'new_process_GPU_hours': new_process_hours, 'reused_jobs': reused,
        'remaining_jobs': len(plan['jobs'])-len(results), 'candidate_mode': plan['candidate_mode'], 'mode_pair': plan['mode_pair'],
        'cases': aggregate(results, plan['candidate_mode']), 'jobs': results,
        'limits': ['Both modes measure all-rank optimizer20->220 wall (final small log excluded); IO also records stdout arrival wall and fast records full log-window wall.',
                   'Rank peaks are cumulative per process, including warmup/startup.',
                   'GPUh budget governs next-job admission; running workers finish without termination. Other tasks budgeted by supplied already_spent/reserved_elsewhere.']})
    return 0 if status == 'COMPLETE' else 2


if __name__ == '__main__': sys.exit(main())
