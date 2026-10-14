"""Wait for the pinned official download, then run the existing label builders.

This is a finite asset dependency queue. It does not allocate GPUs, start
training or change scientific configurations. Provisional samples are rejected.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.structured_world.download_nuscenes import NAMES, atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--population', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--poll-seconds', type=float, default=30.)
    args = parser.parse_args()
    policy = json.loads(Path('/mnt/project/server_dispatch_policy.json').read_text())
    host = socket.gethostname().removesuffix('-worker-0')
    if host not in policy['task_authorizations']['structured_world_fgtr_round1']['allowed_hosts']:
        raise ValueError('Host not authorized for this task')
    if (args.root/'PROVISIONAL.json').exists() or 'provisional' in str(args.root).lower():
        raise ValueError('Provisional sensors cannot enter official asset preparation')
    if not 1 <= args.workers <= 32 or not 1 <= args.poll_seconds <= 60:
        raise ValueError('Invalid queue resource or polling interval')
    args.output.mkdir(parents=True, exist_ok=True)
    state_file = args.output/'PREPARATION_STATE.json'
    registration = {'schema': 'official_nuscenes_label_dependency_v1',
        'source_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'dataset_root': str(args.root.resolve()), 'population': str(args.population.resolve()),
        'workers': args.workers, 'required_archives': NAMES,
        'scope': 'full official train and validation labels; no GPU or training dispatch'}
    registration_file = args.output/'REGISTRATION.json'
    if registration_file.exists() and json.loads(registration_file.read_text()) != registration:
        raise ValueError('Refusing a different asset queue identity')
    atomic_json(registration_file, registration)
    while True:
        manifest_path = args.root/'download_manifest.json'
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        archives = manifest.get('archives', {})
        ready = [name for name in NAMES if archives.get(name, {}).get('downloaded')
                 and archives[name].get('extracted') and archives[name].get('sha256')]
        if len(ready) == len(NAMES) and not manifest.get('errors'):
            break
        state = {'status': 'WAITING_FOR_FULL_OFFICIAL_DATA', 'ready_archives': ready,
                 'pending_archives': [name for name in NAMES if name not in ready],
                 'download_errors': manifest.get('errors', []), 'updated_unix': time.time()}
        atomic_json(state_file, state)
        print(json.dumps(state), flush=True)
        time.sleep(args.poll_seconds)
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    for split in ('train', 'val'):
        cache = args.output/(split+'_labels')
        command = [sys.executable, '-u', str(ROOT/'tools/structured_world/build_nuscenes_cache.py'),
            '--root', str(args.root), '--population', str(args.population), '--split', split,
            '--output', str(cache), '--workers', str(args.workers)]
        atomic_json(state_file, {'status': 'BUILDING_'+split.upper(), 'command': command,
            'download_manifest_sha256': manifest_hash, 'updated_unix': time.time()})
        with (args.output/(split+'_build.log')).open('a') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            atomic_json(state_file, {'status': 'BUILD_FAILED', 'split': split,
                'exit_code': result.returncode, 'log': str(args.output/(split+'_build.log'))})
            raise SystemExit(result.returncode)
        done = json.loads((cache/'COMPLETE.json').read_text())
        if done['scenes'] != done['expected_scenes'] or done['errors']:
            raise ValueError('Label build did not cover the official population')
    atomic_json(state_file, {'status': 'LABELS_COMPLETE',
        'download_manifest_sha256': manifest_hash,
        'train': json.loads((args.output/'train_labels/COMPLETE.json').read_text()),
        'val': json.loads((args.output/'val_labels/COMPLETE.json').read_text()),
        'next_required_assets': ['current_C1_teacher', 'five_epoch_shared_geometry'],
        'formal_training_started': False, 'updated_unix': time.time()})


if __name__ == '__main__':
    main()
