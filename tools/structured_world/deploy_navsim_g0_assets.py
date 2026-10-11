"""G0-only CPU asset staging; immutable scientific caches retain their identities."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path('/mnt/project/VLA-Drive-structured-world-fgtr-io-formal-frozen-20261009')
ART = Path('/mnt/project/structured-world-fgtr-round1-artifacts/20261008')
CONTROL = ART/'navsim_g0_rlzt2_deploy_20261010T2355'
LOCAL = Path('/var/tmp/structured-world-fgtr-round1-20261008/navsim_g0_deploy_20261010T2355')
CURRENT = Path('/var/tmp/ddp-full-foresight-20260929/student_train_v1')
IMAGES = Path('/var/tmp/ddp-full-foresight-20260929/images')
LABEL = '9e8996413379c8421e657bf89006415963263df55d4ae87af3e954b588d0c268'
DINO = 'bda28c73a00fa2d47df2720dd03007f09d006f34482e406b0418a2112ae6924e'
VERSIONS = ('torch', 'deepspeed', 'flash-attn', 'transformers', 'accelerate',
            'omegaconf', 'Pillow', 'numpy', 'diffusers', 'qwen-vl-utils', 'peft', 'safetensors')


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
    require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() ==
            '0c54f1c49e7ed2a76b22826b654d6c72625f5b6d', 'Frozen source SHA differs')
    require(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip(),
            'Frozen source is dirty')


def exact_copy(source, target, expected=None):
    """Read source once, hash both ends, and never overwrite a foreign payload."""
    payload = Path(source).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    require(expected is None or digest == expected, 'Source hash differs: '+str(source))
    target = Path(target)
    if target.exists():
        require(sha(target) == digest, 'Existing destination differs: '+str(target))
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix+'.g0-partial')
        temporary.write_bytes(payload)
        require(sha(temporary) == digest, 'Copied payload differs: '+str(target))
        temporary.replace(target)
    return len(payload), digest


def stage():
    authorize()
    require(shutil.disk_usage('/var/tmp').free > 300*(1 << 30), 'Insufficient local reserve')
    reports = {}
    for name, payload_dir, field in (
            ('navsim_train_full_label_v5_centered', 'labels', 'label_sha256'),
            ('navsim_current_C1_only_v1', 'targets', 'sha256'),
            ('navsim_dev_label_v1_centered', 'labels', 'label_sha256')):
        start = time.monotonic()
        source, target = ART/name, LOCAL/name
        done, ident = read(source/'COMPLETE.json'), read(source/'identity.json')
        require(done['identity'] == ident['identity'], 'Foreign source cache')
        rows = read(ART/'navsim_train_full_label_v5_centered/index.json') if payload_dir == 'targets' else read(source/'index.json')
        require(done['scenes'] == len(rows), 'Source cache population differs')
        for part in (payload_dir, 'records'): (target/part).mkdir(parents=True, exist_ok=True)
        def payload(row):
            token = row['token']
            record = source/'records'/(token+'.json')
            meta = read(record)
            key = 'identity' if payload_dir == 'targets' else 'cache_identity'
            require(meta[key] == ident['identity'], 'Foreign payload identity')
            size, _ = exact_copy(source/payload_dir/(token+'.npz'), target/payload_dir/(token+'.npz'), meta[field])
            exact_copy(record, target/'records'/record.name)
            return size
        processed, total = 0, 0
        with ThreadPoolExecutor(max_workers=8) as pool:
            for size in pool.map(payload, rows):
                processed += 1; total += size
                if processed % 5000 == 0:
                    write(CONTROL/'ASSET_PROGRESS.json', {'component': name, 'processed': processed,
                          'scenes': len(rows), 'bytes': total, 'seconds': time.monotonic()-start})
                    print(json.dumps({'component': name, 'processed': processed, 'bytes': total}), flush=True)
        hashes = {}
        for path in source.glob('*.json'):
            if path.name == 'COMPLETE.json': continue
            _, hashes[path.name] = exact_copy(path, target/path.name)
        _, hashes['COMPLETE.json'] = exact_copy(source/'COMPLETE.json', target/'COMPLETE.json')
        reports[name] = {'identity': ident['identity'], 'scenes': processed, 'bytes': total,
                         'all_payload_SHA256_passed': True, 'metadata_SHA256': hashes,
                         'seconds': time.monotonic()-start, 'source': str(source), 'local': str(target)}
        write(CONTROL/'CACHE_STAGING.json', reports)
    # Deployment-only geometry/calibration pixels remain byte-for-byte copies.
    name = 'navsim_dev_current_v1'
    source, target = ART/name, LOCAL/name
    rows = read(source/'index.json')
    def current(row):
        token = row['token']; meta = read(source/'current'/(token+'.json'))
        exact_copy(source/'current'/(token+'.json'), target/'current'/(token+'.json'))
        return exact_copy(source/'pixels'/(token+'.npz'), target/'pixels'/(token+'.npz'), meta['pixels_sha256'])[0]
    for part in ('current', 'pixels'): (target/part).mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=8) as pool: total = sum(pool.map(current, rows))
    for path in source.glob('*.json'): exact_copy(path, target/path.name)
    reports[name] = {'scenes': len(rows), 'bytes': total, 'all_payload_SHA256_passed': True,
                     'identity': read(target/'identity.json')['identity'], 'local': str(target)}
    write(CONTROL/'CACHE_STAGING.json', reports)
    print('CACHE_STAGING_COMPLETE', flush=True)


def verify():
    authorize()
    sys.path.insert(0, str(ROOT))
    from tools.structured_world.train_vla import source_hashes
    source_identity = read(ART/'formal_navsim_zt2_G1_G0_v3_io/G1_FULL_UNIFORM/identity.json')
    require(source_hashes() == source_identity['training_source_tree_sha256'], 'Training source digest differs')
    versions = {k: importlib.metadata.version(k) for k in VERSIONS}
    expected = read(CONTROL/'EXPECTED_ENVIRONMENT.json')
    require(versions == expected['versions'] and sys.version == expected['python'], 'Qwen environment differs')
    hashes = read(CURRENT/'local_image_hashes.json')
    require(sha(CURRENT/'local_image_hashes.json') == read(CURRENT/'local_stage.json')['image_hash_manifest'],
            'Image manifest differs from the original residency evidence')
    require(len(hashes) == 304776 and read(CURRENT/'COMPLETE.json')['scenes'] == 101592,
            'Incomplete current camera/ego population')
    start = time.monotonic()
    def image(item):
        path, digest = item
        local = IMAGES/(hashlib.sha256(path.encode()).hexdigest()+'.jpg')
        require(sha(local) == digest, 'Local source RGB differs: '+str(local))
        return local.stat().st_size
    with ThreadPoolExecutor(max_workers=8) as pool: total = sum(pool.map(image, hashes.items()))
    # The full source-camera image bytes and the immutable manifest are checked;
    # this routing record only changes which container has the local replica.
    stage_record = read(CURRENT/'local_stage.json')
    stage_record['host'] = socket.gethostname()
    write(CURRENT/'local_stage.json', stage_record)
    initialization = source_identity['initialization_assets']
    qwen = Path(source_identity['config']['framework']['qwenvl']['base_vlm'])
    qwen_hashes = {p: sha(qwen/p) for p in initialization['Qwen_public_pretraining_files']}
    require(qwen_hashes == initialization['Qwen_public_pretraining_files'], 'Generic Qwen differs')
    geometry = ART/'navsim_shared_geometry_5epoch_seed42_v1/shared_geometry.pt'
    require(sha(geometry) == initialization['shared_geometry']['sha256'], 'Shared geometry differs')
    require(sha(source_identity['config']['structured_world']['imagenet_checkpoint']) ==
            initialization['ImageNet_R50_sha256'], 'ImageNet R50 differs')
    prefetch = CONTROL/'CPU_PREFETCH_EQUIVALENCE.json'
    command = [sys.executable, '-B', str(ROOT/'tools/structured_world/validate_prefetch.py'),
               '--cache', str(LOCAL/'navsim_train_full_label_v5_centered'), '--current', str(CURRENT),
               '--dino', str(LOCAL/'navsim_current_C1_only_v1'), '--images', str(IMAGES),
               '--DINO-identity', DINO, '--output', str(prefetch)]
    subprocess.run(command, cwd=ROOT, check=True)
    require(read(prefetch)['passed'], 'Native ordered prefetch verification failed')
    report = {'passed': True, 'host': socket.gethostname(), 'versions': versions,
              'python': sys.version, 'source_sha': '0c54f1c49e7ed2a76b22826b654d6c72625f5b6d',
              'source_tree_sha256': source_hashes(), 'cache_staging': read(CONTROL/'CACHE_STAGING.json'),
              'source_RGB_images': len(hashes), 'source_RGB_bytes': total,
              'all_source_RGB_SHA256_passed': True, 'verification_seconds': time.monotonic()-start,
              'Qwen_public_pretraining_files': qwen_hashes,
              'initialization_assets': initialization, 'CPU_prefetch_command': command,
              'CPU_prefetch_passed': True, 'profile_initialization_used_for_formal': False,
              'local_free_bytes': shutil.disk_usage('/var/tmp').free}
    write(CONTROL/'PREPARATION_COMPLETE.json', report)
    print(json.dumps({'passed': True, 'images': len(hashes), 'source_RGB_bytes': total}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('stage', 'verify'))
    args = parser.parse_args()
    try:
        {'stage': stage, 'verify': verify}[args.phase]()
    except Exception as error:
        write(CONTROL/('ASSET_FAILURE_'+args.phase+'.json'), {'error': repr(error), 'phase': args.phase})
        raise
