"""Execute one registered two-group NAVSIM queue on a complete eight-GPU host.

The existing trainer, strict inference and canonical scorers do the work.
Fixed development observations interrupt at a verified checkpoint boundary;
the original 100k scheduler is preserved. No milestone Navtest selection.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.structured_world.download_nuscenes import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', choices=('local', 'training-vla-zt2'), required=True)
    parser.add_argument('--qwen-python', required=True)
    parser.add_argument('--canonical-python', required=True)
    parser.add_argument('--groups', nargs=2, required=True)
    for name in ('output', 'cache', 'dino-root', 'geometry', 'dev-inputs', 'dev-labels', 'dev-index', 'devkit'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--geometry-identity', required=True)
    parser.add_argument('--dino-identity', required=True)
    args = parser.parse_args()
    if tuple(args.groups) not in (('G1_FULL_UNIFORM', 'G0_NO_FUTURE_LABEL'), ('G3_EVENT_LOCAL', 'G2_EVENT_UNIFORM')):
        raise ValueError('Only preregistered G1/G0 or G3/G2 queues are allowed')
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise ValueError('Queue requires clean committed source')
    canonical_host = socket.gethostname().removesuffix('-worker-0') if args.host == 'local' else args.host
    def authorize():
        policy = json.loads(Path('/mnt/project/server_dispatch_policy.json').read_text())
        if canonical_host not in policy['task_authorizations']['structured_world_fgtr_round1']['allowed_hosts']:
            raise ValueError('Host authorization changed; no new work dispatched')
    def route(command, gpu=False):
        if gpu and args.host != 'local':
            return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', args.host,
                'cd '+shlex.quote(str(ROOT))+' && env OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false '+shlex.join(command)]
        return command
    authorize()
    inspect = ['nvidia-smi', '--query-gpu=uuid,memory.used', '--format=csv,noheader,nounits']
    devices = subprocess.check_output(route(inspect, True), text=True).strip().splitlines()
    if len(devices) != 8 or any(int(row.split(',')[1]) > 128 for row in devices):
        raise ValueError('A complete idle eight-GPU server is required; unrelated work is not stopped')
    device_ids = sorted(row.split(',')[0].strip() for row in devices)
    locks = args.output.parent/'physical_gpu_locks'; locks.mkdir(parents=True, exist_ok=True)
    lock_name = hashlib.sha256(json.dumps(device_ids).encode()).hexdigest()
    with (locks/(lock_name+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.output.mkdir(parents=True, exist_ok=True)
        contract = {'training_source_sha': source, 'host': canonical_host, 'physical_gpu_UUIDs': device_ids,
            'groups': args.groups, 'horizon': 100000, 'global_batch': 32, 'FM_repeat': 8,
            'microbatch': 4, 'development_updates': [25000, 50000, 75000, 100000],
            'geometry_identity': args.geometry_identity, 'DINO_identity': args.dino_identity,
            'pause_restore': 'exact boundary verification; ordinary native CUDA continuation variation disclosed',
            'final_Navtest': 'required after fixed endpoint; separate from this development-only queue'}
        registration = args.output/'REGISTRATION.json'
        if registration.exists() and json.loads(registration.read_text()) != contract:
            raise ValueError('Different queue identity')
        atomic_json(registration, contract)
        def run(stage, command, gpu=False):
            authorize()
            if gpu:
                rows = subprocess.check_output(route(inspect, True), text=True).strip().splitlines()
                if sorted(row.split(',')[0].strip() for row in rows) != device_ids or any(int(row.split(',')[1]) > 128 for row in rows):
                    raise ValueError('Physical GPU identity changed or another task acquired GPUs')
            record = {'stage': stage, 'command': command, 'GPU_host': canonical_host if gpu else 'local CPU',
                      'started_unix': time.time()}
            atomic_json(args.output/'STATE.json', {'status': 'RUNNING', **record})
            with (args.output/'executed_commands.jsonl').open('a') as stream: stream.write(json.dumps(record)+'\n')
            env = os.environ.copy(); env.update(OMP_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
            with (args.output/(stage+'.log')).open('a') as log:
                result = subprocess.run(route(command, gpu), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                atomic_json(args.output/'STATE.json', {'status': 'FAILED', 'stage': stage, 'exit_code': result.returncode})
                raise RuntimeError('Registered stage failed: '+stage)
        try:
            for group in args.groups:
                run_root = args.output/group
                config = ROOT/('configs/structured_world/navsim_'+group+'_seed42.yaml')
                for target in (25000, 50000, 75000, 100000):
                    latest = run_root/'checkpoints/latest'
                    completed = 0
                    if latest.exists():
                        tag = latest.read_text().strip()
                        completed = json.loads((run_root/'checkpoints'/tag/'COMPLETE.json').read_text())['completed']
                    if completed < target:
                        command = [args.qwen_python, '-u', '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=8',
                            str(ROOT/'tools/structured_world/train_vla.py'), '--config', str(config),
                            '--cache', str(args.cache), '--output', str(run_root), '--scope', 'formal', '--micro-batch', '4',
                            '--geometry', str(args.geometry), '--geometry-identity', args.geometry_identity,
                            '--dino-root', str(args.dino_root), '--dino-identity', args.dino_identity, '--save-every', '5000']
                        if target < 100000: command += ['--stop-after', str(target)]
                        if latest.exists(): command.append('--resume')
                        run(group+'_train_to_'+str(target), command, True)
                    tag = 'milestone_'+str(target).zfill(7)
                    complete = json.loads((run_root/'checkpoints'/tag/'COMPLETE.json').read_text())
                    if complete['completed'] != target: raise ValueError('Milestone boundary differs')
                    predictions = run_root/('dev_'+str(target)+'_seed42')
                    if not (predictions/'COMPLETE.json').exists():
                        run(group+'_dev_infer_'+str(target), [args.qwen_python, '-u', '-m', 'torch.distributed.run',
                            '--standalone', '--nproc-per-node=8', str(ROOT/'tools/structured_world/infer_checkpoint.py'),
                            '--run', str(run_root), '--tag', tag, '--inputs', str(args.dev_inputs), '--output', str(predictions),
                            '--sampling-seed', '42', '--scene-fields'], True)
                    planning = run_root/('dev_'+str(target)+'_canonical')
                    if not (planning/'COMPLETE.json').exists():
                        run(group+'_dev_score_'+str(target), [args.canonical_python, '-u',
                            str(ROOT/'tools/structured_world/score_navsim.py'), '--predictions', str(predictions),
                            '--index', str(args.dev_index), '--devkit', str(args.devkit), '--output', str(planning),
                            '--population', 'dev', '--workers', '12'])
                    fields = run_root/('dev_'+str(target)+'_scene_metrics')
                    if not (fields/'COMPLETE.json').exists():
                        local_python = sys.executable
                        run(group+'_dev_fields_'+str(target), [local_python, '-u', str(ROOT/'tools/structured_world/score_scene_fields.py'),
                            '--predictions', str(predictions), '--labels', str(args.dev_labels), '--output', str(fields)])
                    atomic_json(run_root/('OBSERVATION_'+str(target)+'.json'), {'updates': target,
                        'planning': str(planning), 'scene_metrics': str(fields), 'source_sha': source})
            atomic_json(args.output/'STATE.json', {'status': 'NAVSIM_PAIR_TRAIN_AND_DEV_COMPLETE',
                'groups': args.groups, 'final_Navtest_pending': True, 'ended_unix': time.time()})
        except Exception as error:
            atomic_json(args.output/'QUEUE_FAILURE.json', {'error': repr(error), 'source_sha': source,
                'owned_checkpoints_preserved': True, 'unrelated_processes_stopped': False})
            raise


if __name__ == '__main__':
    main()
