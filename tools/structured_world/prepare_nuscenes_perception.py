"""Prepare full official nuScenes current C1 and shared geometry on an idle host.

Waits for the existing official-label queue and uses the same physical GPU locks
as formal NAVSIM. It never evicts a training job or initializes from NAVSIM
driving/perception weights. Existing teacher and geometry trainers do the work.
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
    for name in ('labels', 'output', 'teacher-model', 'imagenet'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--host', help='Pin one currently authorized complete eight-GPU host')
    parser.add_argument('--qwen-python', help='Isolated Python on the pinned host')
    parser.add_argument('--runtime-labels', type=Path,
                        help='Byte-preserving local copy on the selected host; source identity stays unchanged')
    args = parser.parse_args()
    if bool(args.host) != bool(args.qwen_python):
        parser.error('--host and --qwen-python must be supplied together')
    if args.host and any(c.isspace() for c in args.host):
        parser.error('A single SSH host alias is required')
    args.output.mkdir(parents=True, exist_ok=True)
    state = args.output/'STATE.json'
    def route(command, host):
        if host == 'local': return command
        return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host,
                'cd '+shlex.quote(str(ROOT))+' && env OMP_NUM_THREADS=2 '+shlex.join(command)]
    hosts = [('local', '/tmp/structured-world-round1/envs/qwen/bin/python'),
             ('training-vla-zt2', '/var/tmp/structured-world-fgtr-round1-20261008/qwen/bin/python')]
    if args.host:
        hosts = [(args.host, args.qwen_python)]
    locks = args.output.parent/'physical_gpu_locks'; locks.mkdir(exist_ok=True)
    lock = None
    while lock is None:
        complete = args.labels/'COMPLETE.json'
        if not complete.exists():
            atomic_json(state, {'status': 'WAITING_FOR_FULL_OFFICIAL_LABELS', 'updated_unix': time.time()})
            time.sleep(30); continue
        labels = json.loads((args.labels/'identity.json').read_text())
        done = json.loads(complete.read_text())
        if labels['dataset'] != 'nuscenes' or labels['population_kind'] != 'full_train_population' or done['scenes'] != 23230 or done['scenes'] != done['expected_scenes'] or done['errors']:
            raise ValueError('Perception preparation requires the registered full official training population')
        policy = json.loads(Path('/mnt/project/server_dispatch_policy.json').read_text())
        for host, python in hosts:
            canonical = socket.gethostname().removesuffix('-worker-0') if host == 'local' else host
            if canonical not in policy['task_authorizations']['structured_world_fgtr_round1']['allowed_hosts']: continue
            try:
                rows = subprocess.check_output(route(['nvidia-smi', '--query-gpu=uuid,memory.used',
                    '--format=csv,noheader,nounits'], host), text=True, timeout=20).strip().splitlines()
                if len(rows) != 8 or any(int(row.split(',')[1]) > 128 for row in rows): continue
                ids = sorted(row.split(',')[0].strip() for row in rows)
                name = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
                candidate = (locks/(name+'.lock')).open('a')
                try: fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError: candidate.close(); continue
                lock = candidate; selected = (host, python, ids); break
            except (OSError, subprocess.SubprocessError): continue
        if lock is None:
            atomic_json(state, {'status': 'WAITING_FOR_IDLE_FULL8_GPU_HOST',
                'protected_NAVSIM_queues': True, 'label_identity': labels['identity'], 'updated_unix': time.time()})
            time.sleep(30)
    with lock:
        host, python, ids = selected
        runtime_labels = args.runtime_labels or args.labels
        probe = [python, '-c', 'import json,pathlib; p=pathlib.Path('+repr(str(runtime_labels))+'); '
                 'print(json.dumps({"identity":json.loads((p/"identity.json").read_text())["identity"],'
                 '"complete":json.loads((p/"COMPLETE.json").read_text())}))']
        runtime = json.loads(subprocess.check_output(route(probe, host), text=True, timeout=30))
        if runtime['identity'] != labels['identity'] or runtime['complete'] != done:
            raise ValueError('Runtime label copy differs from the complete official source population')
        contract = {'scope': 'official nuScenes own current-only shared perception preparation',
            'training_source_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'host': host, 'physical_gpu_UUIDs': ids, 'label_identity': labels['identity'],
            'runtime_labels': str(runtime_labels), 'source_labels': str(args.labels),
            'full_training_samples': done['scenes'], 'epochs': 5,
            'initialization': 'public ImageNet R50, random remaining geometric modules; no driving head or Qwen',
            'current_DINO': 'same locked C1 teacher recipe for both nuScenes groups'}
        atomic_json(args.output/'REGISTRATION.json', contract)
        commands = [
            ('current_C1', [python, '-u', '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=8',
                '--', str(ROOT/'tools/structured_world/build_current_dino.py'), '--cache', str(runtime_labels),
                '--model-root', str(args.teacher_model), '--output', str(args.output/'current_C1')]),
            ('shared_geometry', [python, '-u', '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=8',
                '--', str(ROOT/'tools/structured_world/train_geometry.py'), '--cache', str(runtime_labels),
                '--imagenet', str(args.imagenet), '--output', str(args.output/'shared_geometry'), '--epochs', '5'])]
        for stage, command in commands:
            output = args.output/stage
            if stage == 'current_C1' and (output/'COMPLETE.json').exists(): continue
            if stage == 'shared_geometry' and (output/'shared_geometry.pt').exists(): continue
            if stage == 'shared_geometry' and (output/'latest').exists(): command.append('--resume')
            policy = json.loads(Path('/mnt/project/server_dispatch_policy.json').read_text())
            canonical = socket.gethostname().removesuffix('-worker-0') if host == 'local' else host
            if canonical not in policy['task_authorizations']['structured_world_fgtr_round1']['allowed_hosts']:
                raise ValueError('Host is no longer authorized')
            atomic_json(state, {'status': 'RUNNING', 'stage': stage, 'command': command, 'host': host})
            env = os.environ.copy(); env['OMP_NUM_THREADS'] = '2'
            with (args.output/(stage+'.log')).open('a') as log:
                result = subprocess.run(route(command, host), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                atomic_json(state, {'status': 'FAILED', 'stage': stage, 'exit_code': result.returncode})
                raise SystemExit(result.returncode)
        atomic_json(state, {'status': 'OFFICIAL_CURRENT_TEACHER_AND_SHARED_GEOMETRY_COMPLETE',
            'label_identity': labels['identity'], 'host': host,
            'next_required': 'two full-model profiles, small learning, registered24epoch G1/G3 training',
            'formal_nuscenes_training_started': False, 'updated_unix': time.time()})


if __name__ == '__main__':
    main()
