"""Run qualification or the fixed 24-epoch nuScenes queue using existing tools.

The two approved extra hosts each use all eight GPUs. Sharing with an existing
scientific workload is explicit, measured and recorded; no process is killed.
Qualification checkpoints are never used to initialize a formal student.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.structured_world.download_nuscenes import atomic_json


def observation_updates(population):
    per_epoch = math.ceil(population/32)
    return [epoch*per_epoch for epoch in (6, 12, 18, 24)]


def profile_summary(run):
    rows = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
    if len(rows) != 16 or rows[-1]['updates'] != 16:
        raise ValueError('Incomplete full-model profile')
    if not all(math.isfinite(r['seconds']) and math.isfinite(r['global_gradient_norm'])
               and all(math.isfinite(x) for x in r['weighted_losses'].values()) for r in rows):
        raise ValueError('Nonfinite profile')
    costs = sorted(row['seconds'] for row in rows[4:])
    return {'updates': len(rows), 'steady_step_seconds_median': (costs[5]+costs[6])/2,
        'steady_step_seconds_max': max(costs),
        'peak_rank0_GPU_allocated_GiB': max(r['peak_GPU_allocated_bytes_rank0'] for r in rows)/(1<<30),
        'training_GPU_hours_estimate_24epochs': sum(costs)/len(costs)*observation_updates(23230)[-1]*8/3600,
        'estimate_excludes': 'initialization, checkpoint writes, validation and future sharing-load changes'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', choices=('training-vla-zt3', 'training-vlawm-zt'), required=True)
    parser.add_argument('--group', choices=('G1_FULL_UNIFORM', 'G3_EVENT_LOCAL'), required=True)
    parser.add_argument('--phase', choices=('qualify', 'formal'), required=True)
    parser.add_argument('--qwen-python', required=True)
    for name in ('output', 'fast-assets', 'geometry', 'val-inputs', 'val-labels', 'dataset', 'execution-acceptance', 'gpu-lock-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--geometry-identity', required=True)
    parser.add_argument('--dino-identity', required=True)
    parser.add_argument('--qualification', type=Path)
    parser.add_argument('--allow-sharing', action='store_true')
    args = parser.parse_args()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise ValueError('A clean committed frozen source is required')
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    config = ROOT/('configs/structured_world/nuscenes_'+args.group+'_seed42.yaml')
    def authorize():
        task = json.loads(Path('/mnt/project/server_dispatch_policy.json').read_text())['task_authorizations']['structured_world_fgtr_round1']
        if args.host not in task['allowed_hosts'] or (args.allow_sharing and not task.get('shared_GPU_deployment_authorized')):
            raise ValueError('Deployment authorization changed')
    def route(command):
        return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', args.host,
            'cd '+shlex.quote(str(ROOT))+' && env OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false '+shlex.join(command)]
    def inspect():
        lines = subprocess.check_output(route(['nvidia-smi', '--query-gpu=uuid,memory.used,memory.free,utilization.gpu',
                                              '--format=csv,noheader,nounits']), text=True).strip().splitlines()
        devices = []
        for line in lines:
            uuid, used, free, utilization = [value.strip() for value in line.split(',')]
            devices.append({'uuid': uuid, 'used_MiB': int(used), 'free_MiB': int(free), 'utilization': int(utilization)})
        if len(devices) != 8 or len({r['uuid'] for r in devices}) != 8:
            raise ValueError('Eight distinct physical GPUs required')
        return devices
    authorize()
    staging = args.fast_assets/'STAGING_COMPLETE.json'
    # Staging belongs to the separately authorized CPU task, not a model run.
    copied = json.loads(subprocess.check_output(route(['cat', str(staging)]), text=True))
    if copied['scenes'] != 23230 or copied['DINO_identity'] != args.dino_identity:
        raise ValueError('Complete official training assets are not ready')
    rows = json.loads((args.val_labels/'index.json').read_text())
    if len(rows) != 4969:
        raise ValueError('Fixed official validation population required')
    devices = inspect(); uuids = sorted(row['uuid'] for row in devices)
    required_MiB = 48*1024
    if args.phase == 'formal':
        if not args.qualification:
            raise ValueError('Formal queue requires actual qualification evidence')
        qualification = json.loads(args.qualification.read_text())
        if (not qualification['passed'] or qualification['group'] != args.group or qualification['host'] != args.host
                or qualification['label_identity'] != copied['label_identity']
                or qualification['DINO_identity'] != args.dino_identity
                or qualification['geometry_identity'] != args.geometry_identity
                or qualification['physical_gpu_UUIDs'] != uuids):
            raise ValueError('Qualification does not match formal assets, method or GPUs')
        required_MiB = math.ceil((qualification['profile']['peak_rank0_GPU_allocated_GiB']+12)*1024)
    def available(snapshot):
        if sorted(r['uuid'] for r in snapshot) != uuids:
            raise ValueError('Physical GPUs changed')
        if any(r['free_MiB'] < required_MiB for r in snapshot):
            raise ValueError('Insufficient memory reserve; checkpoint retained, unrelated jobs not stopped')
        if not args.allow_sharing and any(r['used_MiB'] > 128 for r in snapshot):
            raise ValueError('Exclusive host is occupied')
    available(devices)
    locks = args.gpu_lock_root; locks.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256(json.dumps(uuids).encode()).hexdigest()
    with (locks/(name+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.output.mkdir(parents=True, exist_ok=True)
        registration = {'source_sha': source, 'host': args.host, 'group': args.group, 'phase': args.phase,
            'physical_gpu_UUIDs': uuids, 'initial_other_GPU_workloads': devices,
            'allow_sharing': args.allow_sharing, 'unrelated_processes_stopped': False,
            'label_identity': copied['label_identity'], 'DINO_identity': args.dino_identity,
            'geometry_identity': args.geometry_identity, 'global_batch': 32, 'FM_repeat': 8,
            'microbatch': 4, 'epochs': 24, 'updates': observation_updates(23230),
            'training_initialization': 'generic Qwen, random driving/action, own nuScenes shared five-epoch perception',
            'evaluation': 'locked UniAD v2.0 open-loop; FP32 restored masters, TF32 disabled, proposal seed42'}
        old = args.output/'REGISTRATION.json'
        if old.exists():
            previous = json.loads(old.read_text())
            for key in ('source_sha', 'host', 'group', 'phase', 'physical_gpu_UUIDs', 'label_identity', 'geometry_identity', 'DINO_identity'):
                if previous[key] != registration[key]:
                    raise ValueError('Foreign queue identity')
        else:
            atomic_json(old, registration)
        def run(stage, command, gpu=True):
            authorize()
            snapshot = inspect() if gpu else None
            if gpu: available(snapshot)
            record = {'stage': stage, 'command': command, 'host': args.host if gpu else 'local CPU',
                      'GPU_snapshot': snapshot, 'started_unix': time.time()}
            atomic_json(args.output/'STATE.json', {'status': 'RUNNING', **record})
            with (args.output/'executed_commands.jsonl').open('a') as stream: stream.write(json.dumps(record)+'\n')
            env = os.environ.copy(); env.update(OMP_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
            with (args.output/(stage+'.log')).open('a') as log:
                result = subprocess.run(route(command) if gpu else command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode: raise RuntimeError('Stage '+stage+' failed with exit '+str(result.returncode))
        def training(output, scope):
            command = [args.qwen_python, '-u', '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=8', '--',
                str(ROOT/'tools/structured_world/train_vla.py'), '--config', str(config),
                '--cache', str(args.fast_assets/'train_labels'), '--output', str(output), '--scope', scope,
                '--micro-batch', '4', '--geometry', str(args.geometry), '--geometry-identity', args.geometry_identity,
                '--dino-root', str(args.fast_assets/'current_C1'), '--dino-identity', args.dino_identity,
                '--image-root', str(args.fast_assets/'images'), '--execution-mode', 'io_preserving_v1',
                '--execution-acceptance', str(args.execution_acceptance), '--save-every', '2000']
            return command
        try:
            if args.phase == 'qualify':
                profile = args.output/'profile'
                if not (profile/'TRAINING_COMPLETE.json').exists():
                    run('profile', training(profile, 'profile')+['--debug-updates', '16', '--profile-stages'])
                small = args.output/'small_fit'
                if not (small/'TRAINING_COMPLETE.json').exists():
                    command = training(small, 'small_fit')+['--debug-updates', '100', '--debug-scenes', '64']
                    if not (small/'checkpoints/latest').exists():
                        run('small_fit_first20', command+['--stop-after', '20'])
                    run('small_fit_resume100', command+['--resume'])
                recovered = json.loads((small/'RESUME_BOUNDARY_VERIFIED.json').read_text())
                if recovered['completed'] != 20 or not recovered['passed']:
                    raise ValueError('Real eight-rank recovery check failed')
                measurement = args.output/'gradient_calibration.json'
                if not measurement.exists():
                    run('gradient_calibration', [args.qwen_python, '-u', '-m', 'torch.distributed.run', '--standalone',
                        '--nproc-per-node=8', '--', str(ROOT/'tools/structured_world/calibrate_objectives.py'),
                        '--config', str(config), '--cache', str(args.fast_assets/'train_labels'), '--output', str(measurement),
                        '--geometry', str(args.geometry), '--geometry-identity', args.geometry_identity,
                        '--dino-root', str(args.fast_assets/'current_C1'), '--dino-identity', args.dino_identity,
                        '--image-root', str(args.fast_assets/'images'), '--samples-per-rank', '2'])
                report = json.loads(measurement.read_text())
                for item in report['rows']:
                    for objective in item['objectives'].values():
                        if not math.isfinite(objective['loss']) or not math.isfinite(objective['total_parameter_gradient_norm']):
                            raise ValueError('Nonfinite calibration gradient')
                atomic_json(args.output/'QUALIFICATION_COMPLETE.json', {**registration, 'passed': True,
                    'profile': profile_summary(profile), 'real_eight_rank_recovery': recovered,
                    'small_learning': str(small), 'gradient_calibration': str(measurement),
                    'caveat': 'Engineering train-only checks; no formal planning results or initialization from these checkpoints'})
                atomic_json(args.output/'STATE.json', {'status': 'QUALIFICATION_COMPLETE', 'group': args.group})
                return
            run_root = args.output/args.group
            for target in observation_updates(23230):
                latest = run_root/'checkpoints/latest'
                completed = 0
                if latest.exists():
                    completed = json.loads((run_root/'checkpoints'/latest.read_text().strip()/'COMPLETE.json').read_text())['completed']
                if completed < target:
                    command = training(run_root, 'formal')
                    if target < observation_updates(23230)[-1]: command += ['--stop-after', str(target)]
                    if latest.exists(): command.append('--resume')
                    run('train_to_'+str(target), command)
                tag = 'milestone_'+str(target).zfill(7)
                checkpoint = json.loads((run_root/'checkpoints'/tag/'COMPLETE.json').read_text())
                if checkpoint['completed'] != target: raise ValueError('Wrong epoch observation boundary')
                predictions = run_root/('val_'+str(target)+'_seed42')
                if not (predictions/'COMPLETE.json').exists():
                    run('val_infer_'+str(target), [args.qwen_python, '-u', '-m', 'torch.distributed.run', '--standalone',
                        '--nproc-per-node=8', '--', str(ROOT/'tools/structured_world/infer_checkpoint.py'),
                        '--run', str(run_root), '--tag', tag, '--inputs', str(args.val_inputs), '--output', str(predictions),
                        '--sampling-seed', '42', '--scene-fields'])
                planning = run_root/('val_'+str(target)+'_planning')
                if not (planning/'COMPLETE.json').exists():
                    run('val_score_'+str(target), [sys.executable, '-u', str(ROOT/'tools/structured_world/score_nuscenes.py'),
                        '--root', str(args.dataset), '--population', str(args.val_labels/'index.json'),
                        '--predictions', str(predictions), '--output', str(planning)], gpu=False)
                fields = run_root/('val_'+str(target)+'_scene_metrics')
                if not (fields/'COMPLETE.json').exists():
                    run('val_fields_'+str(target), [sys.executable, '-u', str(ROOT/'tools/structured_world/score_scene_fields.py'),
                        '--predictions', str(predictions), '--labels', str(args.val_labels), '--output', str(fields)], gpu=False)
                atomic_json(run_root/('OBSERVATION_'+str(target)+'.json'), {'updates': target, 'epoch': target/726,
                    'planning': str(planning), 'scene_metrics': str(fields), 'source_sha': source})
            atomic_json(args.output/'STATE.json', {'status': 'NUSCENES_24EPOCH_TRAIN_AND_VAL_COMPLETE',
                'group': args.group, 'training_seeds_completed': [42], 'ended_unix': time.time()})
        except Exception as error:
            atomic_json(args.output/'STATE.json', {'status': 'FAILED', 'error': repr(error)})
            atomic_json(args.output/'QUEUE_FAILURE.json', {'error': repr(error), 'source_sha': source,
                'checkpoints_preserved': True, 'unrelated_processes_stopped': False})
            raise


if __name__ == '__main__':
    main()
