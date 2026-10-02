"""Atomically stage only one identified native clip cache on this host's local disk.

Completed source chunks can be copied while extraction continues. A local COMPLETE
marker is published only after the entire immutable source population is verified.
No decoding, pooling, precision conversion, or identity rewriting occurs here.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import time

from starVLA.model.modules.vehicle_joint.initialization import file_sha256
from tools.ddpolicy_vehicle.prepare_data import atomic_json


def copy_chunk(source, output, number, identity):
    name = f'chunk_{number:06d}'
    metadata = json.loads((source / (name + '.json')).read_text())
    if metadata['identity'] != identity:
        raise ValueError('Source chunk identity changed')
    origin = source / (name + '.safetensors')
    target = output / origin.name
    if target.exists():
        if file_sha256(target) != metadata['sha256']:
            raise ValueError('Existing local native target is corrupt')
    else:
        temporary = target.with_suffix(f'.{os.getpid()}.tmp')
        shutil.copy2(origin, temporary)
        if file_sha256(temporary) != metadata['sha256']:
            temporary.unlink()
            raise ValueError('Native target copy corruption')
        temporary.replace(target)
    atomic_json(output / (name + '.json'), metadata)
    return metadata['bytes']


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--expected-identity', required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError('Bounded I/O workers required')
    source, output = Path(args.source), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if source.resolve() == output.resolve():
        raise ValueError('Local staging must not overwrite source cache')
    lock = (output / 'STAGE.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = json.loads((source / 'identity.json').read_text())
    if identity['schema'] != 'action_video_clip_targets_v1' or identity['identity'] != args.expected_identity:
        raise ValueError('Wrong native target source')
    if (output / 'identity.json').exists() and json.loads((output / 'identity.json').read_text()) != identity:
        raise ValueError('Foreign local target replica')
    atomic_json(output / 'identity.json', identity)
    chunks = (identity['scene_count'] + identity['chunk_size'] - 1) // identity['chunk_size']
    verified = set()
    copied_bytes = 0
    begin = time.time()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            while True:
                pending = [i for i in range(chunks) if i not in verified and
                           (source / f'chunk_{i:06d}.json').exists()]
                ready_bytes = sum(json.loads((source / f'chunk_{i:06d}.json').read_text())['bytes']
                                  for i in pending if not (output / f'chunk_{i:06d}.safetensors').exists())
                if shutil.disk_usage(output).free < ready_bytes + 100 * 2**30:
                    raise RuntimeError('Insufficient local disk including 100GiB reserve')
                for number, size in zip(pending, pool.map(
                        lambda i: copy_chunk(source, output, i, args.expected_identity), pending)):
                    verified.add(number)
                    copied_bytes += size
                atomic_json(output / 'local_replica.json', {
                    'status': 'STAGING', 'host': socket.gethostname(),
                    'identity': args.expected_identity, 'verified_chunks': len(verified),
                    'required_chunks': chunks, 'bytes': copied_bytes,
                    'seconds': time.time() - begin, 'verified_sha256': True,
                    'teacher_encoding_repeated': False})
                if len(verified) == chunks and (source / 'COMPLETE.json').exists():
                    complete = json.loads((source / 'COMPLETE.json').read_text())
                    if (complete['identity'], complete['scenes'], complete['chunks']) != (
                            args.expected_identity, identity['scene_count'], chunks):
                        raise ValueError('Source complete population changed')
                    if complete['cache_bytes'] != copied_bytes:
                        raise ValueError('Native target size mismatch')
                    atomic_json(output / 'COMPLETE.json', complete)
                    atomic_json(output / 'local_replica.json', {
                        'status': 'COMPLETE', 'host': socket.gethostname(),
                        'identity': args.expected_identity, 'scenes': identity['scene_count'],
                        'chunks': chunks, 'bytes': copied_bytes,
                        'seconds': time.time() - begin, 'verified_sha256': True,
                        'teacher_encoding_repeated': False})
                    return
                time.sleep(10)
    except BaseException as error:
        atomic_json(output / 'stage_failure.json', {'error': repr(error), 'verified_chunks': len(verified)})
        raise


if __name__ == '__main__':
    main()
