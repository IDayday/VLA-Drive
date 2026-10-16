"""Immutable target cache IO. Target tensors are never inference inputs.

Channel-last tensors preserve the actual teacher grid. Per-image content keys
permit deduplication; the separate scene manifest retains split/log provenance.
The reader verifies each immutable shard once per file version and reads only
one record range, rather than materializing the complete shard.
"""
from __future__ import annotations

from contextlib import contextmanager
from collections import OrderedDict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .st_world_history import file_sha256
from .st_world_schema import TeacherFeatureContract, identity_hash, require_hash


def atomic_json(path, value, *, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'
    if immutable and path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Immutable artifact conflict: ' + str(path))
        return
    fd, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextmanager
def file_lock(path, *, blocking=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as f:
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        fcntl.flock(f, flags)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def tensor_digest(value):
    cpu = value.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(cpu.dtype).encode())
    h.update(json.dumps(list(cpu.shape)).encode())
    h.update(cpu.numpy().tobytes())
    return h.hexdigest()


def payload_key(ref, contract):
    require_hash(ref['sha256'], 'image sha256')
    return identity_hash({'image_sha256': ref['sha256'], 'teacher_contract': contract.identity})


def validate_target_manifest(manifest):
    if manifest.get('schema') != 'st_teacher_requests_v1':
        raise ValueError('Unrecognized teacher request manifest')
    if identity_hash({k:v for k,v in manifest.items() if k != 'identity'}) != manifest.get('identity'):
        raise ValueError('Teacher manifest checksum mismatch')
    if manifest['split'] not in ('train', 'dev'):
        raise ValueError('Only locked train/dev teacher requests are supported')
    if manifest['view_names'] != ['CAM_F0', 'CAM_L0', 'CAM_R0']:
        raise ValueError('Changed view order')
    times = manifest['times_s']
    if times[0] != 0 or sorted(set(times)) != times or any(t < 0 for t in times):
        raise ValueError('Targets need current0 and distinct positive future times')
    tokens = set()
    for scene in manifest['scenes']:
        if scene['token'] in tokens or scene['split'] != manifest['split']:
            raise ValueError('Duplicate scene or role leakage')
        tokens.add(scene['token'])
        if len(scene['frames']) != len(times):
            raise ValueError('Teacher time layout mismatch')
        for t, frame in zip(times, scene['frames']):
            if frame['requested_s'] != t or len(frame['views']) != len(manifest['view_names']):
                raise ValueError('Teacher view/time layout mismatch')
            ts = frame['timestamp_us']
            if ts is None:
                if any(v is not None for v in frame['views']):
                    raise ValueError('Missing target timestamp with real pixels')
            else:
                delta = (ts - scene['timestamp_us']) / 1e6
                if ((t == 0 and delta != 0) or (t > 0 and delta <= 0) or
                        abs(delta - t) > manifest['timestamp_tolerance_s'] + 1e-9):
                    raise ValueError('Wrong physical target timestamp')
            for ref in frame['views']:
                if ref is not None:
                    require_hash(ref['sha256'], 'target image sha256')
    return manifest


def unique_image_records(manifest, contract):
    validate_target_manifest(manifest)
    if tuple(manifest['view_names']) != contract.view_names:
        raise ValueError('Teacher/request camera contract differs')
    images = {}
    for scene in manifest['scenes']:
        for frame in scene['frames']:
            for ref in frame['views']:
                if ref is None:
                    continue
                key = payload_key(ref, contract)
                if key in images and images[key]['image']['sha256'] != ref['sha256']:
                    raise ValueError('Conflicting content addressed image')
                images.setdefault(key, {'key': key, 'image': ref})
    return [images[key] for key in sorted(images)]


class TeacherCacheWriter:
    """No model/optimizer. Each caller locks one shard through its whole commit.

    Incomplete tensor files without receipts remain as named orphan evidence
    before deterministic reconstruction. Complete shards are never overwritten.
    """
    def __init__(self, root, contract, manifest, *, images_per_shard=32, execution=None):
        self.root = Path(root)
        self.contract = contract.validate()
        self.manifest = validate_target_manifest(manifest)
        if type(images_per_shard) is not int or not 1 <= images_per_shard <= 1024:
            raise ValueError('images_per_shard must be 1..1024')
        self.records = unique_image_records(manifest, contract)
        self.images_per_shard = images_per_shard
        self.shard_count = (len(self.records) + images_per_shard - 1) // images_per_shard
        self.identity = {'schema': 'st_teacher_cache_v1', 'contract': contract.to_dict(),
                         'request_manifest_identity': manifest['identity'], 'images_per_shard': images_per_shard,
                         'records': self.records, 'execution': execution or {}, 'shards': self.shard_count}
        self.identity['identity'] = identity_hash(self.identity)
        self.root.mkdir(parents=True, exist_ok=True)
        with file_lock(self.root / 'IDENTITY.lock'):
            atomic_json(self.root / 'identity.json', self.identity, immutable=True)
            atomic_json(self.root / 'requests.json', manifest, immutable=True)

    def rows(self, shard):
        if type(shard) is not int or not 0 <= shard < self.shard_count:
            raise ValueError('Shard outside registered scope')
        return self.records[shard * self.images_per_shard:(shard + 1) * self.images_per_shard]

    def paths(self, shard):
        return self.root / f'shard_{shard:06d}.safetensors', self.root / f'shard_{shard:06d}.json'

    def shard_lock(self, shard, *, blocking=True):
        self.rows(shard)
        return file_lock(self.root / f'shard_{shard:06d}.lock', blocking=blocking)

    def committed(self, shard):
        tensor, meta = self.paths(shard)
        if not meta.exists():
            return False
        receipt = json.loads(meta.read_text())
        expected_keys = [r['key'] for r in self.rows(shard)]
        if (receipt['identity'] != self.identity['identity'] or receipt['keys'] != expected_keys or
                not tensor.is_file() or file_sha256(tensor) != receipt['sha256']):
            raise ValueError('Corrupt or incompatible complete teacher shard')
        return True

    def commit_shard(self, shard, features, valid, *, metrics=None):
        """Caller must hold shard_lock. Tensor/receipt crash window is resumable."""
        rows = self.rows(shard)
        expected = (len(rows), *self.contract.output_grid_hw, self.contract.channels)
        if tuple(features.shape) != expected or valid.shape != features.shape[:-1] or valid.dtype != torch.bool:
            raise ValueError('Teacher shard shape/mask differs from registered contract')
        if features.dtype != getattr(torch, self.contract.storage_dtype) or features.requires_grad or not torch.isfinite(features).all():
            raise ValueError('Teacher shard finite/dtype/detach mismatch')
        values = {'features': features.cpu().contiguous(), 'valid': valid.cpu().contiguous()}
        record_hashes = [{'features': tensor_digest(values['features'][i]), 'valid': tensor_digest(values['valid'][i])}
                         for i in range(len(rows))]
        tensor, meta = self.paths(shard)
        if self.committed(shard):
            old = json.loads(meta.read_text())
            if old['record_hashes'] != record_hashes:
                raise ValueError('Conflicting tensor content for immutable teacher shard')
            return old
        if tensor.exists():
            tensor.rename(tensor.with_suffix(f'.orphan_{time.time_ns()}.safetensors'))
        temp = tensor.with_suffix(f'.{os.getpid()}.tmp')
        save_file(values, str(temp), metadata={'identity': self.identity['identity'], 'contract': self.contract.identity})
        with temp.open('rb') as f:
            os.fsync(f.fileno())
        os.replace(temp, tensor)
        receipt = {'schema': 'st_teacher_shard_receipt_v1', 'identity': self.identity['identity'],
                   'keys': [r['key'] for r in rows], 'record_hashes': record_hashes,
                   'sha256': file_sha256(tensor), 'bytes': tensor.stat().st_size, 'metrics': metrics or {}}
        atomic_json(meta, receipt, immutable=True)
        return receipt

    def finalize(self):
        # No COMPLETE until every requested image has a verified committed shard.
        with file_lock(self.root / 'FINALIZE.lock'):
            missing = [i for i in range(self.shard_count) if not self.paths(i)[1].exists()]
            if missing:
                return {'status': 'PARTIAL', 'missing_shards': missing, 'identity': self.identity['identity']}
            receipts = []
            for i in range(self.shard_count):
                self.committed(i)
                receipts.append({'shard': i, 'receipt_sha256': file_sha256(self.paths(i)[1])})
            done = {'status': 'COMPLETE', 'identity': self.identity['identity'], 'unique_images': len(self.records),
                    'scenes': len(self.manifest['scenes']), 'shards': receipts,
                    'identity_sha256': file_sha256(self.root / 'identity.json'),
                    'requests_sha256': file_sha256(self.root / 'requests.json'),
                    'bytes': sum(self.paths(i)[0].stat().st_size for i in range(self.shard_count))}
            atomic_json(self.root / 'COMPLETE.json', done, immutable=True)
            return done


class TeacherCacheReader:
    def __init__(self, root, *, expected_identity, expected_contract=None, max_verified_shards=16):
        if type(max_verified_shards) is not int or max_verified_shards < 1:
            raise ValueError('Reader verification LRU must be bounded and positive')
        self.root = Path(root)
        self.identity = json.loads((self.root / 'identity.json').read_text())
        done = json.loads((self.root / 'COMPLETE.json').read_text())
        if (self.identity['identity'] != expected_identity or
                identity_hash({k:v for k,v in self.identity.items() if k != 'identity'}) != expected_identity or
                done['identity'] != expected_identity or done['status'] != 'COMPLETE' or
                done['identity_sha256'] != file_sha256(self.root / 'identity.json') or
                done['requests_sha256'] != file_sha256(self.root / 'requests.json')):
            raise ValueError('Teacher cache incomplete/identity mismatch')
        self.contract = TeacherFeatureContract.from_dict(self.identity['contract'])
        if expected_contract is not None and self.contract.identity != expected_contract.identity:
            raise ValueError('Wrong teacher/transform/precision/grid')
        self.manifest = validate_target_manifest(json.loads((self.root / 'requests.json').read_text()))
        if self.manifest['identity'] != self.identity['request_manifest_identity']:
            raise ValueError('Teacher request population changed')
        expected_records = unique_image_records(self.manifest, self.contract)
        if self.identity['records'] != expected_records or done['scenes'] != len(self.manifest['scenes']):
            raise ValueError('Teacher records do not exactly cover the declared request population')
        self.locations = {r['key']: divmod(i, self.identity['images_per_shard']) for i, r in enumerate(self.identity['records'])}
        if len(self.locations) != len(self.identity['records']) or len(self.locations) != done['unique_images']:
            raise ValueError('Teacher record coverage mismatch')
        self.receipt_hashes = {r['shard']: r['receipt_sha256'] for r in done['shards']}
        if set(self.receipt_hashes) != set(range(self.identity['shards'])):
            raise ValueError('Incomplete teacher shard receipt list')
        self.scenes = {s['token']: s for s in self.manifest['scenes']}
        self.verified = OrderedDict()
        # Only tiny immutable version stamps scale with declared shard count.
        # Receipt objects remain LRU-bounded; evicting a receipt must not force
        # rereading gigabytes of unchanged tensor data on the next access.
        self.verified_versions = {}
        self.max_verified_shards = max_verified_shards

    def verify_population(self, requested_manifest):
        requested_manifest = validate_target_manifest(requested_manifest)
        if (requested_manifest['identity'] != self.manifest['identity'] or
                requested_manifest != self.manifest or
                unique_image_records(requested_manifest, self.contract) != self.identity['records']):
            raise ValueError('Requested manifest/scene/record coverage differs from this cache')

    def _verify(self, shard):
        path = self.root / f'shard_{shard:06d}.safetensors'
        receipt_path = path.with_suffix('.json')
        stat = path.stat()
        meta_stat = receipt_path.stat()
        version = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, meta_stat.st_size, meta_stat.st_mtime_ns)
        if shard in self.verified and self.verified[shard][0] == version:
            self.verified.move_to_end(shard)
            return path, self.verified[shard][1]
        if file_sha256(receipt_path) != self.receipt_hashes[shard]:
            raise ValueError('Changed teacher receipt')
        receipt = json.loads(receipt_path.read_text())
        begin = shard * self.identity['images_per_shard']
        expected = [r['key'] for r in self.identity['records'][begin:begin + self.identity['images_per_shard']]]
        if receipt['identity'] != self.identity['identity'] or receipt['keys'] != expected:
            raise ValueError('Changed teacher shard')
        if self.verified_versions.get(shard) != version:
            if file_sha256(path) != receipt['sha256']:
                raise ValueError('Changed teacher shard')
            self.verified_versions[shard] = version
        self.verified[shard] = (version, receipt)
        self.verified.move_to_end(shard)
        while len(self.verified) > self.max_verified_shards:
            self.verified.popitem(last=False)
        return path, receipt

    def read_image(self, key):
        shard, row = self.locations[key]
        path, receipt = self._verify(shard)
        with safe_open(str(path), framework='pt', device='cpu') as f:
            if f.metadata() != {'identity': self.identity['identity'], 'contract': self.contract.identity}:
                raise ValueError('Teacher tensor header identity mismatch')
            z, valid = f.get_slice('features')[row], f.get_slice('valid')[row]
        expected = receipt['record_hashes'][row]
        if tensor_digest(z) != expected['features'] or tensor_digest(valid) != expected['valid']:
            raise ValueError('Teacher record corruption')
        if (tuple(z.shape) != (*self.contract.output_grid_hw, self.contract.channels) or valid.shape != z.shape[:-1] or
                z.dtype != getattr(torch, self.contract.storage_dtype) or valid.dtype != torch.bool or not torch.isfinite(z).all()):
            raise ValueError('Teacher record dtype/grid mismatch')
        return z, valid

    def read_scene(self, token, *, expected_split=None, need_future=True):
        scene = self.scenes[token]
        if expected_split is not None and scene['split'] != expected_split:
            raise ValueError('Teacher scene role mismatch')
        if type(need_future) is not bool:
            raise ValueError('need_future must be an explicit bool')
        times = self.manifest['times_s'] if need_future else self.manifest['times_s'][:1]
        shape = (len(times), len(self.contract.view_names), *self.contract.output_grid_hw, self.contract.channels)
        features = torch.zeros(shape, dtype=getattr(torch, self.contract.storage_dtype))
        valid = torch.zeros(shape[:-1], dtype=torch.bool)
        for t, frame in enumerate(scene['frames'][:len(times)]):
            for v, ref in enumerate(frame['views']):
                if ref is not None:
                    features[t,v], valid[t,v] = self.read_image(payload_key(ref, self.contract))
        return {'current_dense_features': features[0], 'current_valid': valid[0],
                'future_dense_features': features[1:], 'future_valid': valid[1:],
                'future_times_s': times[1:], 'teacher_contract': self.contract,
                'cache_identity': self.identity['identity']}


class LegacyC1Reader:
    """Exact old C1 SP0 adapter. Never labels pooled9x12 data as native18x24.

    Only the audited immutable C1 identity is accepted. Sources are joined by
    actual image SHA256, not path/scene ordinal. Old FP16 values are retained.
    """
    IDENTITY = '7663c45b77dd711e8304e8d202644dcaa8467f7e265ff8b4c1816d59d2c131cb'

    def __init__(self, root):
        self.root = Path(root)
        self.identity = json.loads((self.root / 'identity.json').read_text())
        d = self.identity
        if (d.get('identity') != self.IDENTITY or identity_hash({k:v for k,v in d.items() if k != 'identity'}) != self.IDENTITY or
                d['schema'] != 'ddp_full_dino_cache_v1' or d['candidate']['pool'] != 2 or d['grid_hw'] != [6,8] or
                d['dtype'] != 'float16' or d['recipe']['actual_tensor_chw'] != [3,192,256]):
            raise ValueError('Legacy SP0 requires the exact registered C1 cache')
        done = json.loads((self.root / 'COMPLETE.json').read_text())
        if done['identity'] != self.IDENTITY or done['images'] != d['image_count']:
            raise ValueError('Legacy C1 source incomplete')
        self.lookup = {}
        self.checked = {}
        count = 0
        for path in sorted(self.root.glob('chunk_*.json')):
            meta = json.loads(path.read_text())
            if meta['identity'] != self.IDENTITY or len(meta['keys']) != len(meta['image_sha256']):
                raise ValueError('Legacy image receipt mismatch')
            for row, sha in enumerate(meta['image_sha256']):
                self.lookup.setdefault(sha, (path, row))
                count += 1
        if count != d['image_count']:
            raise ValueError('Legacy C1 record coverage incomplete')

    def read_image(self, image_sha256):
        meta_path, row = self.lookup[image_sha256]
        path = meta_path.with_suffix('.safetensors')
        meta = json.loads(meta_path.read_text())
        version = (path.stat().st_size, path.stat().st_mtime_ns, path.stat().st_ctime_ns)
        if self.checked.get(str(path)) != version:
            if file_sha256(path) != meta['sha256']:
                raise ValueError('Corrupt legacy C1 shard')
            self.checked[str(path)] = version
        with safe_open(str(path), framework='pt', device='cpu') as f:
            if f.metadata()['identity'] != self.IDENTITY:
                raise ValueError('Wrong legacy tensor identity')
            z, valid = f.get_slice('features')[row], f.get_slice('valid')[row]
        if z.shape != (1024,6,8) or z.dtype != torch.float16 or valid.shape != (6,8) or not torch.isfinite(z).all():
            raise ValueError('Legacy C1 stored grid/dtype mismatch')
        return z.permute(1,2,0).contiguous(), valid, {'source_identity':self.IDENTITY, 'shard_sha256':meta['sha256'], 'row':row}
