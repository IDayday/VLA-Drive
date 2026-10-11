"""Single G0 routing wrapper around the unchanged frozen NAVSIM train/dev tools.

The frozen pair interface only accepts two groups and two original routes.
This wrapper adds a qualified rl-zt2 route, not a training implementation.
Qualification checkpoints are separate and never initialize the formal run.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import statistics
import subprocess
import sys
import threading
import time

sys.dont_write_bytecode = True
ROOT = Path('/mnt/project/VLA-Drive-structured-world-fgtr-io-formal-frozen-20261009')
SHA = '0c54f1c49e7ed2a76b22826b654d6c72625f5b6d'
TREE = '11915ed4b770a1a9060784f31661d15125a256e7f41cf8474fc3fa9ca8d72173'
ART = Path('/mnt/project/structured-world-fgtr-round1-artifacts/20261008')
CONTROL = ART/'navsim_g0_rlzt2_deploy_20261010T2355'
LOCAL = Path('/var/tmp/structured-world-fgtr-round1-20261008/navsim_g0_deploy_20261010T2355')
PYTHON = '/tmp/structured-world-round1/envs/qwen/bin/python'
CANONICAL = '/root/miniconda3/envs/navsim/bin/python'
GROUP = 'G0_NO_FUTURE_LABEL'
LABEL = '9e8996413379c8421e657bf89006415963263df55d4ae87af3e954b588d0c268'
DINO = 'bda28c73a00fa2d47df2720dd03007f09d006f34482e406b0418a2112ae6924e'
GEOMETRY = 'e07c1a00d8961e10f88d71799957bcd647bc4a58009cd6cfbae22f42dfa9fecd'
TARGETS = (25000, 50000, 75000, 100000)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
    temp.replace(path)


def require(value, message):
    if not value: raise ValueError(message)


def authorize():
    host = socket.gethostname().removesuffix('-worker-0')
    task = read('/mnt/project/server_dispatch_policy.json')['task_authorizations']['structured_world_fgtr_round1']
    require(host == 'training-rl-zt2' and host in task['allowed_hosts'] and
            host in task['latest_explicit_resource_hosts'], 'G0 destination no longer authorized')
    require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == SHA,
            'Frozen source SHA differs')
    require(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip(),
            'Frozen source is dirty')


def devices():
    command = ['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu',
               '--format=csv,noheader,nounits']
    rows = subprocess.check_output(command, text=True).strip().splitlines()
    return [dict(zip(('index', 'uuid', 'name', 'used_MiB', 'total_MiB', 'utilization'),
                [int(v[0]), v[1], v[2], *map(int, v[3:])]))
            for v in ([x.strip() for x in row.split(',')] for row in rows)]


def native_command(run, scope='formal', target=25000, resume=False):
    command = [PYTHON, '-B', '-u', '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=8',
               '--', str(ROOT/'tools/structured_world/train_vla.py'),
               '--config', str(ROOT/'configs/structured_world/navsim_G0_NO_FUTURE_LABEL_seed42.yaml'),
               '--cache', str(LOCAL/'navsim_train_full_label_v5_centered'), '--output', str(run),
               '--scope', scope, '--micro-batch', '4',
               '--geometry', str(ART/'navsim_shared_geometry_5epoch_seed42_v1/shared_geometry.pt'),
               '--geometry-identity', GEOMETRY, '--dino-root', str(LOCAL/'navsim_current_C1_only_v1'),
               '--dino-identity', DINO, '--save-every', '5000', '--execution-mode', 'io_preserving_v1',
               '--execution-acceptance', str(ART/'IO_EXECUTION_ACCEPTANCE_v1.json')]
    if scope == 'profile': command += ['--debug-updates', '16', '--profile-stages']
    if target < (100000 if scope == 'formal' else 16): command += ['--stop-after', str(target)]
    if resume: command.append('--resume')
    require('--resume-origin-run' not in command and '--debug-scenes' not in command,
            'No parent planner or selected population may initialize G0')
    return command


def latest(run):
    pointer = run/'checkpoints/latest'
    if not pointer.exists(): return None
    tag = pointer.read_text().strip()
    require(Path(tag).name == tag and tag.startswith(('latest_', 'milestone_')), 'Unsafe checkpoint pointer')
    complete = read(run/'checkpoints'/tag/'COMPLETE.json')
    identity = read(run/'identity.json')
    names = {'mp_rank_00_model_states.pt'} | {f'rng_rank{i}.pt' for i in range(8)} | {
            f'zero_pp_rank_{i}_mp_rank_00_optim_states.pt' for i in range(8)}
    require(complete['identity'] == identity['identity'] and complete['tag'] == tag and complete['ranks'] == 8
            and set(complete['file_sha256']) == names, 'Foreign or incomplete native checkpoint')
    return complete


def run(stage, command, uuids, gpu=False, formal=False):
    authorize()
    if gpu:
        snapshot = devices()
        require(sorted(r['uuid'] for r in snapshot) == uuids and all(r['used_MiB'] <= 128 for r in snapshot),
                'Eight GPUs are not idle; unrelated tasks preserved')
    require(shutil.disk_usage(CONTROL).free >= 300*(1 << 30), 'Shared checkpoint reserve below 300 GiB')
    record = {'stage': stage, 'command': command, 'host': 'training-rl-zt2',
              'formal': formal, 'started_unix': time.time()}
    write(CONTROL/'STATE.json', {'status': 'RUNNING', **record})
    with (CONTROL/'executed_commands.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2',
               MAX_JOBS='4', TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1',
               CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7')
    stop = threading.Event()
    def telemetry():
        while not stop.is_set():
            try:
                with (CONTROL/'GPU_SNAPSHOTS.jsonl').open('a') as f:
                    f.write(json.dumps({'stage': stage, 'unix': time.time(), 'devices': devices()})+'\n')
            except Exception as error:
                with (CONTROL/'telemetry_errors.log').open('a') as f: f.write(repr(error)+'\n')
            stop.wait(5)
    monitor = threading.Thread(target=telemetry, daemon=True) if gpu else None
    if monitor: monitor.start()
    try:
        with (CONTROL/(stage+'.log')).open('a') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
            record['pid'] = child.pid
            write(CONTROL/'ACTIVE_PROCESS.json', record)
            if formal and stage.endswith('_train_to_25000'):
                write(CONTROL/'FORMAL_LAUNCH.json', {**record, 'controller_pid': os.getpid(),
                      'initialization': 'generic public Qwen + random action head + shared geometry',
                      'qualification_checkpoint_used': False, 'training_source_sha': SHA,
                      'training_source_tree_sha256': TREE, 'physical_gpu_UUIDs': uuids})
            result = child.wait()
    finally:
        stop.set()
        if monitor: monitor.join(timeout=10)
    require(result == 0, 'Native stage failed: '+stage+' exit '+str(result))
    with (CONTROL/'completed_commands.jsonl').open('a') as f:
        f.write(json.dumps({**record, 'completed_unix': time.time(), 'exit_code': result})+'\n')


def qualify(uuids):
    preparation = read(CONTROL/'PREPARATION_COMPLETE.json')
    require(read(CONTROL/'CANONICAL_ENVIRONMENT.json')['passed'], 'Canonical development scoring environment differs')
    require(read(CONTROL/'METADATA_ENV_REPLICA_VERIFIED.json')['passed'], 'Current metadata or environment replica differs')
    require(not read(CONTROL/'DUPLICATE_CHECK.json')['G0_training_or_controller_found'], 'A duplicate G0 exists')
    require(preparation['passed'] and preparation['host'] == 'training-rl-zt2-worker-0'
            and preparation['source_tree_sha256'] == TREE and preparation['all_source_RGB_SHA256_passed']
            and preparation['CPU_prefetch_passed'], 'Preparation evidence is incomplete')
    profile = CONTROL/'qualification'/GROUP
    require(not (profile/'identity.json').exists(), 'Qualification output must be a new independent run')
    run('G0_profile_pause_after_8', native_command(profile, 'profile', 8), uuids, gpu=True)
    checkpoint = latest(profile)
    require(checkpoint['completed'] == 8 and checkpoint['status'] == 'PAUSED', 'Qualification pause boundary differs')
    paused = {**checkpoint, 'complete_sha256': sha(profile/'checkpoints'/checkpoint['tag']/'COMPLETE.json')}
    write(CONTROL/'QUALIFICATION_PAUSE.json', paused)
    run('G0_profile_resume_to_16', native_command(profile, 'profile', 16, resume=True), uuids, gpu=True)
    boundary = read(profile/'RESUME_BOUNDARY_VERIFIED.json')
    require(boundary['passed'] and boundary['completed'] == 8 and boundary['offset'] == 8
            and boundary['exposure'] == 256, 'Exact native recovery did not pass before update 9')
    final = latest(profile)
    require(final['completed'] == 16 and final['status'] == 'COMPLETE', 'Full eight-GPU profile incomplete')
    identity = read(profile/'identity.json')
    require(identity['training_source_sha'] == SHA and identity['training_source_tree_sha256'] == TREE
            and identity['config']['structured_world']['group'] == GROUP and identity['world_size'] == 8
            and identity['source_population_scenes'] == identity['scenes_per_epoch'] == 101592
            and identity['effective_global_batch'] == 32 and identity['FM_repeat'] == 8
            and identity['microbatch'] == 4 and identity['engineering_training_selection'] is None
            and identity['initialization_assets'] == preparation['initialization_assets'], 'Profile scientific inputs differ')
    metrics = [json.loads(row) for row in (profile/'metrics.jsonl').read_text().splitlines()]
    require([m['updates'] for m in metrics] == list(range(1, 17)), 'Missing or duplicate profile updates')
    for m in metrics:
        require(m['fp32_master_update_verified'] and m['sampled_changed_elements'] > 0,
                'Missing real FP32 optimizer update')
        require(all(math.isfinite(v) for v in m['weighted_losses'].values()) and
                math.isfinite(m['global_gradient_norm']) and m['weighted_losses']['future_semantics'] == 0,
                'Nonfinite or foreign G0 objective')
        require(len(m['per_rank_stage_timing']) == 8 and m['scene_exposure'] == 32*m['updates'],
                'Profile is not full-eight-GPU/global32')
    # Exclude initial two updates after each process start from steady throughput.
    steady = [m for m in metrics if m['updates'] not in (1, 2, 9, 10)]
    snapshots = [json.loads(row) for row in (CONTROL/'GPU_SNAPSHOTS.jsonl').read_text().splitlines()
                 if json.loads(row)['stage'].startswith('G0_profile_')]
    peak = {u: max(d['used_MiB'] for s in snapshots for d in s['devices'] if d['uuid'] == u) for u in uuids}
    median = statistics.median(m['seconds'] for m in steady)
    require(all(peak[u] > 20000 and peak[u] < 76000 for u in uuids), 'Full-eight-GPU memory evidence/headroom differs')
    require(median < 6, 'Prepared I/O/model throughput is too slow; investigate before formal launch')
    report = {'passed': True, 'host': 'training-rl-zt2', 'group': GROUP, 'updates': 16,
              'full_training_population': 101592, 'global_batch': 32, 'FM_repeat': 8, 'microbatch': 4,
              'profile_horizon': 16, 'formal_horizon_unchanged': 100000,
              'steady_updates': len(steady), 'steady_median_seconds': median,
              'steady_mean_seconds': statistics.mean(m['seconds'] for m in steady),
              'rank0_peak_allocated_bytes': max(m['peak_GPU_allocated_bytes_rank0'] for m in metrics),
              'per_GPU_peak_used_MiB': peak, 'physical_gpu_UUIDs': uuids,
              'recovery_boundary': boundary, 'paused_checkpoint': paused,
              'final_checkpoint': final, 'training_source_sha': SHA, 'training_source_tree_sha256': TREE,
              'profile_identity': identity['identity'], 'profile_as_formal_initialization': False,
              'future_CUDA_updates_bitwise_guaranteed': False,
              'controller_code_sha256': sha(__file__), 'created_unix': time.time()}
    write(CONTROL/'QUALIFICATION_COMPLETE.json', report)
    return report


def formal(uuids):
    require(read(CONTROL/'QUALIFICATION_COMPLETE.json')['passed'], 'G0 qualification has not passed')
    run_root = CONTROL/'formal'/GROUP
    require(not (run_root/'identity.json').exists() and not (run_root/'checkpoints/latest').exists(),
            'Formal G0 must start from fresh common initialization')
    registration = {'kind': 'single_NAVSIM_G0_routing_only', 'group': GROUP, 'host': 'training-rl-zt2',
                    'training_source_sha': SHA, 'training_source_tree_sha256': TREE,
                    'controller_file': str(Path(__file__).resolve()), 'controller_code_sha256': sha(__file__),
                    'original_pair_interface': str(ROOT/'tools/structured_world/run_navsim_pair.py'),
                    'route_reason': 'frozen interface accepts only two groups and original local/vla-zt2 hosts',
                    'physical_gpu_UUIDs': uuids, 'run_root': str(run_root), 'horizon': 100000,
                    'warmup': 5000, 'global_batch': 32, 'FM_repeat': 8, 'microbatch': 4,
                    'proposal_steps': 10, 'development_updates': list(TARGETS), 'evaluation_seed': 42,
                    'evaluation_population': 1696, 'evaluation_logs': 16,
                    'execution_mode': 'io_preserving_v1',
                    'execution_acceptance_sha256': sha(ART/'IO_EXECUTION_ACCEPTANCE_v1.json'),
                    'initialization': 'generic public Qwen + random action head + shared geometry',
                    'S0_G1_profile_weights_used': False, 'geometry_identity': GEOMETRY,
                    'DINO_identity': DINO, 'label_cache_identity': LABEL,
                    'final_Navtest': 'not part of this dev queue', 'created_unix': time.time()}
    write(CONTROL/'REGISTRATION.json', registration)
    for target in TARGETS:
        complete = latest(run_root)
        if complete is None or complete['completed'] < target:
            run(GROUP+'_train_to_'+str(target), native_command(run_root, target=target, resume=complete is not None),
                uuids, gpu=True, formal=True)
        tag = f'milestone_{target:07d}'
        boundary = read(run_root/'checkpoints'/tag/'COMPLETE.json')
        require(boundary['completed'] == target and boundary['identity'] == read(run_root/'identity.json')['identity'],
                'Formal development boundary differs')
        predictions = run_root/f'dev_{target}_seed42'
        if not (predictions/'COMPLETE.json').exists():
            run(GROUP+'_dev_infer_'+str(target), [PYTHON, '-B', '-u', '-m', 'torch.distributed.run',
                '--standalone', '--nproc-per-node=8', '--', str(ROOT/'tools/structured_world/infer_checkpoint.py'),
                '--run', str(run_root), '--tag', tag, '--inputs', str(LOCAL/'navsim_dev_current_v1'),
                '--output', str(predictions), '--sampling-seed', '42', '--scene-fields'], uuids, gpu=True, formal=True)
        planning = run_root/f'dev_{target}_canonical'
        if not (planning/'COMPLETE.json').exists():
            run(GROUP+'_dev_score_'+str(target), [CANONICAL, '-B', '-u', str(ROOT/'tools/structured_world/score_navsim.py'),
                '--predictions', str(predictions), '--index', str(Path('/mnt/project/local-interaction-mask-v2-artifacts/20260927/official_dev_metric_v2/cache_index.json')),
                '--devkit', '/mnt/project/local-interaction-mask-v2-artifacts/20260927/reference_sources/navsim-v1.1',
                '--output', str(planning), '--population', 'dev', '--workers', '12'], uuids, formal=True)
        fields = run_root/f'dev_{target}_scene_metrics'
        if not (fields/'COMPLETE.json').exists():
            run(GROUP+'_dev_fields_'+str(target), [PYTHON, '-B', '-u', str(ROOT/'tools/structured_world/score_scene_fields.py'),
                '--predictions', str(predictions), '--labels', str(LOCAL/'navsim_dev_label_v1_centered'),
                '--output', str(fields)], uuids, formal=True)
        write(run_root/f'OBSERVATION_{target}.json', {'updates': target, 'planning': str(planning),
              'scene_metrics': str(fields), 'source_sha': SHA})
    write(CONTROL/'STATE.json', {'status': 'G0_TRAIN_AND_DEV_COMPLETE', 'final_Navtest_pending': True,
                                'completed_unix': time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualify-and-start', action='store_true', required=True)
    parser.parse_args()
    require(Path(sys.executable).absolute() == Path(PYTHON), 'Use original Qwen Python')
    authorize()
    snapshot = devices()
    uuids = sorted(r['uuid'] for r in snapshot)
    require(len(uuids) == len(set(uuids)) == 8 and all(r['used_MiB'] <= 128 for r in snapshot),
            'Complete idle eight-GPU host required')
    locks = ART/'physical_gpu_locks'
    locks.mkdir(exist_ok=True)
    with ExitStack() as held:
        for path in (locks/(hashlib.sha256(json.dumps(uuids).encode()).hexdigest()+'.lock'),
                     locks/'NAVSIM_G0_NO_FUTURE_LABEL_seed42.lock'):
            lock = held.enter_context(path.open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(CONTROL/'RESOURCE_REGISTRATION.json', {'host': socket.gethostname(), 'devices': snapshot,
              'controller_pid': os.getpid(), 'CPU_threads_per_rank': 2, 'data_prefetch_workers_per_rank': 2,
              'unrelated_processes_terminated': False, 'registered_unix': time.time()})
        qualify(uuids)
        formal(uuids)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        write(CONTROL/'QUEUE_FAILURE.json', {'error': repr(error), 'owned_checkpoints_preserved': True,
              'unrelated_processes_stopped': False, 'unix': time.time()})
        write(CONTROL/'STATE.json', {'status': 'FAILED', 'error': repr(error), 'unix': time.time()})
        raise
