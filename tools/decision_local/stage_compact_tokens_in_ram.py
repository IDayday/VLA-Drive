"""Byte-identical temporary RAM mirrors of completed compact token/label banks.

Canonical data and every durable receipt remain on /mnt/project. No image,
full-Qwen feature bank, new dtype or model result is generated. The original
trainer can resume from either location with the same strict logical identity.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.decision_local.common import atomic_json, check_host, digest, read, source_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-cache', type=Path, nargs='+', required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, required=True)
    args = parser.parse_args()
    resources = check_host()
    started = time.monotonic()
    target = args.destination.resolve()
    if not str(target).startswith('/dev/shm/decision-local-') or target.exists():
        raise ValueError('Use a new task-specific temporary RAM directory')
    if subprocess.check_output(['stat', '-f', '-c', '%T', str(target.parent)], text=True).strip() != 'tmpfs':
        raise ValueError('Do not mirror onto container disk')
    if not str(args.receipt.resolve()).startswith('/mnt/project/') or args.receipt.exists():
        raise ValueError('A new durable shared receipt is required')
    sources, files, total = [], [], 0
    for source in args.source_cache:
        source = source.resolve()
        if not str(source).startswith('/mnt/project/decision-local-consequence-artifacts/'):
            raise ValueError('Only this round compact immutable banks are eligible')
        meta, complete = read(source/'identity.json'), read(source/'COMPLETE.json')
        if complete['identity'] != meta['identity']:
            raise ValueError('Completed bank identity mismatch')
        chunks = sorted((source/'chunks').glob('*.safetensors'))
        if not chunks or any(source.name == old.name for old in sources):
            raise ValueError('Empty or duplicate compact bank')
        for path in chunks:
            receipt = source/'records'/(path.stem+'.json')
            row = read(receipt)
            if row['identity'] != meta['identity']:
                raise ValueError('Foreign chunk')
            files.append((path, target/source.name/'chunks'/path.name, row['sha256']))
            total += path.stat().st_size
        sources.append(source)
    meminfo = dict((line.split(':')[0], int(line.split()[1])*1024) for line in Path('/proc/meminfo').read_text().splitlines())
    free = shutil.disk_usage(target.parent).free
    if total > 80*1024**3 or min(meminfo['MemAvailable'], free) < 2*total+64*1024**3:
        raise ValueError('Insufficient RAM headroom for the bounded compact working set')
    target.mkdir(mode=0o750)
    for source in sources:
        destination = target/source.name
        (destination/'chunks').mkdir(parents=True)
        shutil.copytree(source/'records', destination/'records')
        shutil.copyfile(source/'identity.json', destination/'identity.json')
    def copy_verified(item):
        src, dst, expected = item
        if time.monotonic()-started >= args.max_seconds:
            raise TimeoutError('Finite RAM staging budget reached')
        temporary = dst.with_suffix('.tmp')
        checksum = hashlib.sha256()
        with src.open('rb') as inp, temporary.open('xb') as out:
            for block in iter(lambda: inp.read(8 << 20), b''):
                checksum.update(block)
                out.write(block)
            out.flush()
            os.fsync(out.fileno())
        if checksum.hexdigest() != expected or digest(temporary) != expected:
            raise ValueError('Source or RAM copy differs from committed chunk')
        temporary.replace(dst)
        return dst.stat().st_size
    with ThreadPoolExecutor(max_workers=2) as pool:
        copied = list(pool.map(copy_verified, files))
    if sum(copied) != total:
        raise ValueError('Incomplete working set')
    for source in sources:
        destination = target/source.name
        for path in (source/'records').glob('*.json'):
            if digest(path) != digest(destination/'records'/path.name):
                raise ValueError('Record copy differs')
        if digest(source/'identity.json') != digest(destination/'identity.json'):
            raise ValueError('Input contract copy differs')
        shutil.copyfile(source/'COMPLETE.json', destination/'COMPLETE.json')
    atomic_json(args.receipt, {'complete': True, 'source_sha': source_sha(ROOT), 'tool_sha256': digest(__file__),
        'resources': resources, 'canonical_sources': [str(x) for x in sources],
        'temporary_RAM_caches': [str(target/x.name) for x in sources],
        'identities': {x.name: read(x/'identity.json')['identity'] for x in sources},
        'bytes': total, 'chunks': len(files), 'seconds': time.monotonic()-started,
        'all_chunk_bytes_and_metadata_exact': True, 'new_container_disk_bytes': 0,
        'persistent_model_dataset_or_training_source_changed': False,
        'rollback': 'Resume the same native checkpoint and original frozen trainer using the canonical /mnt/project cache paths',
        'lifetime': 'Temporary RAM only; no cache dependency after native training stops. Shared canonical banks retained unchanged.'})


if __name__ == '__main__':
    main()
