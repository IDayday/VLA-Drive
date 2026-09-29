"""Download a pinned public timm DINOv3 ViT-L/16 artifact, independently of the study."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import time

from .download_flux_vae import HTTP, download
from .score_pdms import atomic_json
from starVLA.model.modules.vehicle_joint.initialization import file_sha256

REPOSITORY = 'timm/vit_large_patch16_dinov3.lvd1689m'
REVISION = '30c1109559f65dea34316b0d4842d35c5771fe11'
WEIGHT_SHA256 = '45172f209c9583c40538afc26b60a07033e6fcc2e8c30228338e6b2e932e7941'
WEIGHT_BYTES = 1212347640
SMALL_FILES = {
    'config.json': '8ed25750a7bf2b6e25acdced526638b9a715fd3a',
    'README.md': 'd63349686db061067c11ea9425211bc23a1f14c9',
    'LICENSE.md': 'f531b1e6b5ab2318957bbf8ad1bda9f800a23e17',
}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'DOWNLOAD.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.time()
    url = f'https://huggingface.co/api/models/{REPOSITORY}/revision/{REVISION}?blobs=true'
    response = HTTP.get(url, timeout=40)
    response.raise_for_status()
    metadata = response.json()
    if (metadata['id'] != REPOSITORY or metadata['sha'] != REVISION
            or metadata.get('private') or metadata.get('gated')):
        raise ValueError('Expected pinned public timm distribution')
    entries = {row['rfilename']: row for row in metadata['siblings']}
    for name, expected_blob in SMALL_FILES.items():
        if entries[name]['blobId'] != expected_blob:
            raise ValueError(f'Pinned metadata mismatch: {name}')
    weight = entries['model.safetensors']
    if weight['size'] != WEIGHT_BYTES or weight['lfs']['sha256'] != WEIGHT_SHA256:
        raise ValueError('Pinned weight metadata mismatch')
    files = {}
    for name in [*SMALL_FILES, 'model.safetensors']:
        entry = entries[name]
        print(f'Downloading/verifying {name}: {entry["size"]} bytes', flush=True)
        download(f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}',
                 root / name, entry['size'], WEIGHT_SHA256 if name == 'model.safetensors' else None)
        if name in SMALL_FILES:
            content = (root / name).read_bytes()
            blob = hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest()
            if blob != SMALL_FILES[name]:
                raise ValueError(f'Downloaded Git blob mismatch: {name}')
        files[name] = {'bytes': entry['size'], 'sha256': file_sha256(root / name)}
    identity = {
        'repository': REPOSITORY, 'revision': REVISION, 'format': 'timm',
        'architecture': 'vit_large_patch16_dinov3', 'pretraining_dataset': 'LVD-1689M',
        'files': files, 'metadata_url': url,
        'original_reference': 'https://github.com/facebookresearch/dinov3',
        'conversion_notes': [
            'timm omits all-zero QKV biases; this is not the Meta Transformers checkpoint.',
            'timm generates FP32 RoPE periods, unlike original persistent BF16 periods.',
            'No bytewise or numerical equivalence to the original format is claimed.',
        ],
        'download_purpose': 'Separate user-requested generic artifact; not integrated into student training.',
        'generic_pretraining_data_fully_auditable': False,
    }
    identity_path = root / 'IDENTITY.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('Existing identity differs; refusing overwrite')
    atomic_json(identity_path, identity)
    report = {'status': 'COMPLETE', 'identity': identity,
              'elapsed_seconds': time.time() - started, 'GPU_used': False}
    atomic_json(root / 'DOWNLOAD_COMPLETE.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
