"""Download the fixed nuScenes trainval assets into a new, resumable directory.

Only public Motional bucket URLs are used. Existing completed archives are never
replaced. Download/extraction identities stay next to the data, not in Git.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import threading
import time
import urllib.request
import zipfile

BASE = 'https://motional-nuscenes.s3.amazonaws.com/public/v1.0/'
NAMES = ['v1.0-trainval_meta.tgz', 'nuScenes-map-expansion-v1.3.zip', 'can_bus.zip']
NAMES += [f'v1.0-trainval{i:02d}_blobs.tgz' for i in range(1, 11)]
LOCK = threading.Lock()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_destination(root, name):
    result = (root / name).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError(f'Archive path escapes the dataset directory: {name}')
    return result


def extract(archive, root):
    count = 0
    if archive.name.endswith('.tgz'):
        with tarfile.open(archive, 'r:gz') as source:
            for member in source:
                target = safe_destination(root, member.name)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists() and target.stat().st_size == member.size:
                        continue  # Resume only the same fixed archive population.
                    temporary = target.with_name(target.name + '.extracting')
                    with source.extractfile(member) as incoming, temporary.open('wb') as outgoing:
                        shutil.copyfileobj(incoming, outgoing, length=8 << 20)
                    os.replace(temporary, target)
                    count += 1
                else:
                    raise ValueError(f'Unsupported link/device in official archive: {member.name}')
    else:
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                target = safe_destination(root, member.filename)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('Symlinks are not accepted in dataset archives')
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and target.stat().st_size == member.file_size:
                    continue
                temporary = target.with_name(target.name + '.extracting')
                with source.open(member) as incoming, temporary.open('wb') as outgoing:
                    shutil.copyfileobj(incoming, outgoing, length=8 << 20)
                os.replace(temporary, target)
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--reserve-gib', type=int, default=200)
    parser.add_argument('--metadata-only', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    archives = root / '_downloads'
    archives.mkdir(exist_ok=True)
    manifest_path = root / 'download_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {'source': BASE, 'version': 'v1.0-trainval', 'archives': {},
                    'download_started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    'dataset_homepage': 'https://www.nuscenes.org/nuscenes',
                    'scope': 'official trainval metadata, all 10 sensor archives, maps v1.3, CAN bus'}
    names = NAMES[:3] if args.metadata_only else NAMES
    for name in names:
        with urllib.request.urlopen(urllib.request.Request(BASE+name, method='HEAD'), timeout=90) as response:
            identity = {'url': BASE+name, 'bytes': int(response.headers['Content-Length']),
                        'etag': response.headers.get('ETag'),
                        'last_modified': response.headers.get('Last-Modified')}
        previous = manifest['archives'].get(name)
        if previous and (previous['bytes'] != identity['bytes'] or previous['etag'] != identity['etag']):
            raise RuntimeError(f'Official object identity changed for {name}; refusing mixed data')
        manifest['archives'].setdefault(name, identity)
    atomic_json(manifest_path, manifest)
    print(json.dumps({'event': 'registered', 'compressed_bytes': sum(manifest['archives'][n]['bytes'] for n in names),
                      'disk_free_bytes': shutil.disk_usage(root).free}), flush=True)

    def complete(name):
        identity = manifest['archives'][name]
        archive = archives / name
        start = time.monotonic()
        if not archive.exists():
            if shutil.disk_usage(root).free < identity['bytes']*3 + args.reserve_gib*(1 << 30):
                raise RuntimeError(f'Insufficient storage reserve for {name}')
            partial = archive.with_name(archive.name + '.part')
            subprocess.run(['curl', '--fail', '--location', '--continue-at', '-', '--retry', '12',
                            '--retry-delay', '5', '--retry-all-errors', '--connect-timeout', '45',
                            '--speed-limit', '1024', '--speed-time', '180', '--silent', '--show-error',
                            '--output', str(partial), identity['url']], check=True)
            if partial.stat().st_size != identity['bytes']:
                raise RuntimeError(f'Download size mismatch: {name}')
            os.replace(partial, archive)
        if archive.stat().st_size != identity['bytes']:
            raise RuntimeError(f'Existing completed archive has a different size: {name}')
        digest = sha256(archive)
        if identity.get('sha256') and identity['sha256'] != digest:
            raise RuntimeError(f'Archive SHA256 mismatch: {name}')
        with LOCK:
            identity.update(sha256=digest, downloaded=True)
            atomic_json(manifest_path, manifest)
        if not identity.get('extracted'):
            destination = root/'maps' if name == 'nuScenes-map-expansion-v1.3.zip' else root
            files = extract(archive, destination)
            with LOCK:
                identity.update(extracted=True, extracted_files_this_attempt=files,
                                seconds=time.monotonic()-start)
                atomic_json(manifest_path, manifest)
        print(json.dumps({'event': 'complete', 'archive': name, 'sha256': digest,
                          'seconds': time.monotonic()-start}), flush=True)
        return name

    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(complete, name): name for name in names}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                errors.append({'archive': futures[future], 'error': repr(error)})
                print(json.dumps(errors[-1]), flush=True)
    with LOCK:
        manifest['errors'] = errors
        manifest['complete'] = not errors
        atomic_json(manifest_path, manifest)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
