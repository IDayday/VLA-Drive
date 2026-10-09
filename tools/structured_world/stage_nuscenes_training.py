"""Copy verified official training assets to local storage without changing inputs.

Only the six current training JPEGs are selected. Labels and current C1 targets
retain their original identities and per-file hashes. No future camera files are
needed by the student. Re-running verifies existing files before using them.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.structured_world.download_nuscenes import atomic_json


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def checked_copy(source, target, expected):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if digest(target) != expected:
            raise ValueError('Existing local asset has a different hash: '+str(target))
    else:
        temporary = target.with_suffix(target.suffix+'.partial')
        shutil.copyfile(source, temporary)
        if digest(temporary) != expected:
            raise ValueError('Copied asset differs from the immutable source: '+str(source))
        temporary.replace(target)
    return target.stat().st_size


def main():
    parser = argparse.ArgumentParser()
    for name in ('labels', 'dino', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    policy = json.loads(Path('/mnt/project/server_dispatch_policy.json').read_text())
    host = socket.gethostname().removesuffix('-worker-0')
    if host not in policy['task_authorizations']['structured_world_fgtr_round1']['allowed_hosts']:
        raise ValueError('Unauthorized staging host')
    identity = json.loads((args.labels/'identity.json').read_text())
    complete = json.loads((args.labels/'COMPLETE.json').read_text())
    teacher = json.loads((args.dino/'identity.json').read_text())
    teacher_complete = json.loads((args.dino/'COMPLETE.json').read_text())
    rows = json.loads((args.labels/'index.json').read_text())
    if (identity['dataset'] != 'nuscenes' or identity['population_kind'] != 'full_train_population'
            or complete['identity'] != identity['identity'] or complete['errors']
            or len(rows) != complete['scenes'] or len(rows) != complete['expected_scenes']
            or teacher['structured_cache_identity'] != identity['identity']
            or teacher_complete['identity'] != teacher['identity'] or teacher_complete['scenes'] != len(rows)):
        raise ValueError('Only complete matching official training assets can be staged')
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    local_labels, local_dino = args.output/'train_labels', args.output/'current_C1'
    def stage(row):
        token = row['token']
        label_record = args.labels/'records'/(token+'.json')
        dino_record = args.dino/'records'/(token+'.json')
        label = json.loads(label_record.read_text())
        target = json.loads(dino_record.read_text())
        if label['cache_identity'] != identity['identity'] or target['identity'] != teacher['identity']:
            raise ValueError('Foreign training record')
        total = checked_copy(args.labels/'labels'/(token+'.npz'), local_labels/'labels'/(token+'.npz'), label['label_sha256'])
        # The teacher writer stores its dense target with the same token basename.
        total += checked_copy(args.dino/'targets'/(token+'.npz'), local_dino/'targets'/(token+'.npz'), target['sha256'])
        for source, destination in ((label_record, local_labels/'records'/(token+'.json')),
                                    (dino_record, local_dino/'records'/(token+'.json'))):
            total += checked_copy(source, destination, digest(source))
        images = label['current_record']['image_paths']
        hashes = target['source_image_sha256']
        if len(images) != 6 or len(hashes) != 6:
            raise ValueError('Six current cameras are required')
        for image, expected in zip(images, hashes):
            source = Path(image)
            relative = source.relative_to(identity['input_root'])
            if relative.parts[0] != 'samples':
                raise ValueError('Unexpected camera image source')
            total += checked_copy(source, args.output/'images'/relative, expected)
        return total
    byte_count = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for completed, size in enumerate(pool.map(stage, rows), 1):
            byte_count += size
            if completed % 500 == 0:
                atomic_json(args.output/'STAGING_STATE.json', {'host': host, 'scenes': completed,
                    'expected_scenes': len(rows), 'bytes': byte_count, 'seconds': time.monotonic()-start})
    for source_root, target_root in ((args.labels, local_labels), (args.dino, local_dino)):
        for name in ('identity.json', 'index.json', 'COMPLETE.json'):
            source = source_root/name
            if source.exists():
                checked_copy(source, target_root/name, digest(source))
    atomic_json(args.output/'STAGING_COMPLETE.json', {'host': host, 'scenes': len(rows),
        'cameras': 6, 'image_files': 6*len(rows), 'label_identity': identity['identity'],
        'DINO_identity': teacher['identity'], 'bytes': byte_count, 'seconds': time.monotonic()-start,
        'verification': 'Every label, target, record and current RGB SHA256 checked; no changed tensors',
        'image_root': str(args.output/'images'), 'labels': str(local_labels), 'dino': str(local_dino)})


if __name__ == '__main__':
    main()
