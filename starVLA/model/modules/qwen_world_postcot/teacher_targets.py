"""Versioned target contracts. Teacher clips never become actor inputs."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any
from collections import OrderedDict
import fcntl
import os
import time
import uuid

import torch
from torch import Tensor

CAMERAS = ('cam_l0', 'cam_f0', 'cam_r0')
IMAGE_MEAN = (.485, .456, .406)
IMAGE_STD = (.229, .224, .225)


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(16*1024*1024), b''):
            h.update(data)
    return h.hexdigest()


def identity_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def clip_timing_valid(timestamps_us: Tensor, *, tolerance_us=50000):
    if timestamps_us.dtype != torch.int64 or timestamps_us.ndim != 2 or timestamps_us.shape[1] != 12:
        raise ValueError('WA teacher requires actual twelve frame timestamps in integer microseconds')
    if (timestamps_us < -1).any():
        raise ValueError('Unknown timestamps must use explicit-1, never fabricated times')
    delta = timestamps_us[:, 1:] - timestamps_us[:, :-1]
    return (timestamps_us >= 0).all(1) & (delta > 0).all(1) & (delta-500000).abs().le(tolerance_us).all(1)


def validate_clip_timestamps(timestamps_us: Tensor, *, tolerance_us=50000, allow_incomplete=False):
    timing_valid=clip_timing_valid(timestamps_us,tolerance_us=tolerance_us)
    if not allow_incomplete and not timing_valid.all():
        raise ValueError('Teacher clip is not the registered chronological2Hz sequence')
    # Four historical frames end at frame3/current; each future tubelet spans
    # two real future frames. No one-to-one action/token timestamp assumption.
    return timestamps_us[:, 4:].reshape(len(timestamps_us), 4, 2)


@dataclass(frozen=True)
class DINOTargetBatch:
    features: Tensor
    valid: Tensor
    timestamps_us: Tensor
    teacher_identity: dict[str, Any]
    camera_names: tuple[str, ...] = CAMERAS

    def validate(self):
        if self.features.ndim != 5 or tuple(self.features.shape[1:]) != (3, 9, 16, 1024):
            raise ValueError('DINO target must be[B,3,9,16,1024]')
        if self.valid.shape != self.features.shape[:-1] or self.valid.dtype != torch.bool:
            raise ValueError('DINO validity must track every pooled patch')
        if self.features.requires_grad or not torch.isfinite(self.features[self.valid]).all():
            raise ValueError('Teacher targets must be finite, detached tensors')
        if self.timestamps_us.shape != (len(self.features),) or self.timestamps_us.dtype != torch.int64:
            raise ValueError('DINO current timestamps must be int64[B] microseconds')
        if self.camera_names != CAMERAS:
            raise ValueError('Teacher camera order differs from current actor order')
        return self


@dataclass(frozen=True)
class WAFutureTargetBatch:
    features: Tensor
    valid: Tensor
    frame_timestamps_us: Tensor
    tubelet_timestamps_us: Tensor
    teacher_identity: dict[str, Any]
    camera_names: tuple[str, ...] = CAMERAS

    def validate(self):
        if self.features.ndim != 6 or tuple(self.features.shape[1:]) != (3, 4, 8, 16, 512):
            raise ValueError('WA targets must be projected/pooled[B,3,4,8,16,512]')
        if self.valid.shape != self.features.shape[:-1] or self.valid.dtype != torch.bool:
            raise ValueError('WA validity must track every projected pooled patch')
        if self.features.requires_grad or not torch.isfinite(self.features[self.valid]).all():
            raise ValueError('WA targets must be finite and detached')
        expected = validate_clip_timestamps(self.frame_timestamps_us,allow_incomplete=True)
        if (self.valid.flatten(2).any(2) & ~clip_timing_valid(self.frame_timestamps_us)[:,None]).any():
            raise ValueError('Missing/non2Hz clip cannot have valid future teacher targets')
        if self.tubelet_timestamps_us.shape != expected.shape or not torch.equal(expected, self.tubelet_timestamps_us):
            raise ValueError('Tubelets do not bind their actual two source timestamps')
        if self.camera_names != CAMERAS:
            raise ValueError('WA camera order mismatch')
        return self


def current_depth_from_ego(points_xyz: Tensor, camera_from_ego: Tensor):
    """Camera optical z-depth in metres, from registered current extrinsics.

    camera_from_ego[...,4,4] maps ego coordinates into camera coordinates.
    This is not Euclidean radial distance and never uses future annotations.
    """
    if points_xyz.shape[-1] != 3 or camera_from_ego.shape[-2:] != (4, 4):
        raise ValueError('Physical xyz and homogeneous ego→camera transform required')
    if not torch.isfinite(points_xyz).all() or not torch.isfinite(camera_from_ego).all():
        raise ValueError('Nonfinite current calibration/geometry')
    homogeneous = torch.cat([points_xyz, torch.ones_like(points_xyz[..., :1])], dim=-1)
    if points_xyz.ndim == 3 and camera_from_ego.ndim == 4:
        camera = torch.einsum('bvij,bnj->bvni', camera_from_ego, homogeneous)
    elif points_xyz.ndim == 2 and camera_from_ego.ndim == 3:
        camera = torch.einsum('vij,nj->vni', camera_from_ego, homogeneous)
    else:
        camera = torch.matmul(camera_from_ego, homogeneous.unsqueeze(-1)).squeeze(-1)
    return camera[..., 2]


def visible_current_centers(boxes: Tensor, intrinsics: Tensor, camera_from_ego: Tensor,
                            image_hw: tuple[int, int], *, min_depth=.1):
    """Conservative centre visibility under actual calibrated current cameras.

    Full 3D box/FOV clipping may be supplied by the dataset; this helper does
    not label an entirely invisible area as an observed background negative.
    """
    points = boxes[..., :3]
    homogeneous = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
    if points.ndim == 3 and camera_from_ego.ndim == 4:
        camera = torch.einsum('bvij,bnj->bvni', camera_from_ego, homogeneous)[..., :3]
        projection = torch.einsum('bvij,bvnj->bvni', intrinsics, camera)
    elif points.ndim == 2 and camera_from_ego.ndim == 3:
        camera = torch.einsum('vij,nj->vni', camera_from_ego, homogeneous)[..., :3]
        projection = torch.einsum('vij,vnj->vni', intrinsics, camera)
    else:
        camera = torch.matmul(camera_from_ego, homogeneous.unsqueeze(-1)).squeeze(-1)[..., :3]
        projection = torch.matmul(intrinsics, camera.unsqueeze(-1)).squeeze(-1)
    z = camera[..., 2]
    xy = projection[..., :2] / projection[..., 2:].clamp_min(min_depth)
    h, w = image_hw
    return (z > min_depth) & (xy[..., 0] >= 0) & (xy[..., 0] < w) & (xy[..., 1] >= 0) & (xy[..., 1] < h)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.partial.'+uuid.uuid4().hex)
    with temporary.open('x') as stream:
        json.dump(value, stream, sort_keys=True, separators=(',', ':'))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _sealed(value):
    value = dict(value)
    value['sha256'] = identity_hash(value)
    return value


def _read_sealed(path):
    value = json.loads(Path(path).read_text())
    body = dict(value)
    if body.pop('sha256', None) != identity_hash(body):
        raise ValueError('Target cache receipt checksum mismatch: '+str(path))
    return value


_DTYPES = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
           'float16': torch.float16, 'bool': torch.bool, 'int64': torch.int64}


def tensor_layout(tensors):
    result, offset = {}, 0
    for name, value in tensors.items():
        dtype = str(value.dtype).removeprefix('torch.')
        if dtype not in _DTYPES:
            raise TypeError('Unsupported exact teacher storage dtype: '+dtype)
        size = value.numel()*value.element_size()
        result[name] = dict(dtype=dtype, shape=list(value.shape), offset=offset, bytes=size)
        offset += size
    return dict(fields=result, record_bytes=offset)


def _cache_index(root, identity):
    records, shards = {}, {}
    for path in sorted((root/'shards').glob('*.json')):
        shard = _read_sealed(path)
        if shard['status'] != 'COMPLETE' or shard['identity'] != identity['sha256']:
            raise ValueError('Foreign/incomplete teacher shard receipt')
        data = root/shard['file']
        stat = data.stat()
        if stat.st_size != shard['bytes'] or stat.st_mtime_ns != shard['mtime_ns']:
            raise ValueError('Teacher shard version changed: '+str(data))
        shards[shard['file']] = dict(sha256=shard['data_sha256'], bytes=shard['bytes'],
                                     mtime_ns=shard['mtime_ns'], receipt_sha256=shard['sha256'])
        for row in shard['records']:
            if (row['metadata'].get('cohort') != identity['cohort'] or row['offset'] < 0
                    or row['bytes'] != identity['layout']['record_bytes']
                    or row['offset']+row['bytes'] > shard['bytes']):
                raise ValueError('Invalid teacher record role/range')
            entry = dict(row, file=shard['file'])
            previous = records.get(row['token'])
            if previous is not None:
                if previous['record_sha256'] != row['record_sha256'] or previous['metadata'] != row['metadata']:
                    raise ValueError('Conflicting duplicated teacher target: '+row['token'])
                continue
            records[row['token']] = entry
    return records, shards


class BinaryTeacherCacheWriter:
    """Bounded immutable shards with per-record and whole-shard checksums."""
    def __init__(self, root, identity, *, rank=0, shard_records=256):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root/'shards').mkdir(exist_ok=True)
        body = dict(identity)
        body.setdefault('schema', 'world_postcot_teacher_cache_v1')
        self.identity = _sealed(body)
        lock = (self.root/'identity.lock').open('a')
        with lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = self.root/'IDENTITY.json'
            if path.exists():
                if _read_sealed(path) != self.identity:
                    raise ValueError('Target cache identity differs; use a new namespace')
            else:
                atomic_json(path, self.identity)
        self.rank, self.shard_records = rank, shard_records
        if shard_records < 1:
            raise ValueError('Positive bounded shard size required')
        self.existing, _ = _cache_index(self.root, self.identity)
        self.rows = []
        self.pending = {}
        self.stream = None
        self.partial = None
        self.shard_hash = None
        self.sequence = len(list((self.root/'shards').glob(f'rank{rank:04d}_*.json')))

    def add(self, token, tensors, metadata):
        layout = tensor_layout(tensors)
        if layout != self.identity['layout']:
            raise ValueError('Teacher field/dtype layout changed within cache')
        if metadata.get('cohort') != self.identity['cohort']:
            raise ValueError('Target role differs from immutable cache role')
        payload = b''.join(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                           for value in tensors.values())
        record_hash = hashlib.sha256(payload).hexdigest()
        previous = self.existing.get(token, self.pending.get(token))
        if previous is not None:
            if previous['metadata'] != metadata or previous['record_sha256'] != record_hash:
                raise ValueError('Previously cached token now names a different observation/target')
            return False
        if self.stream is None:
            name = f'rank{self.rank:04d}_{self.sequence:06d}_{uuid.uuid4().hex}'
            self.partial = self.root/'shards'/(name+'.partial')
            self.stream = self.partial.open('xb')
            self.shard_hash = hashlib.sha256()
        offset = self.stream.tell()
        self.stream.write(payload)
        self.shard_hash.update(payload)
        self.rows.append(dict(token=token, offset=offset, bytes=len(payload),
                              record_sha256=record_hash, metadata=metadata))
        self.pending[token] = self.rows[-1]
        if len(self.rows) >= self.shard_records:
            self.commit()
        return True

    def commit(self):
        if self.stream is None:
            return None
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        destination = self.partial.with_suffix('.bin')
        os.replace(self.partial, destination)
        stat = destination.stat()
        receipt = _sealed(dict(status='COMPLETE', identity=self.identity['sha256'],
            file=str(destination.relative_to(self.root)), bytes=stat.st_size,
            mtime_ns=stat.st_mtime_ns, data_sha256=self.shard_hash.hexdigest(), records=self.rows))
        # Publication of the receipt is the only admission point. A completed
        # data file without a receipt remains an orphan diagnostic, not a hit.
        atomic_json(destination.with_suffix('.json'), receipt)
        for row in self.rows:
            self.existing[row['token']] = dict(row, file=receipt['file'])
        self.rows, self.stream, self.partial, self.shard_hash = [], None, None, None
        self.pending = {}
        self.sequence += 1
        return receipt

    def close(self):
        self.commit()


def finalize_teacher_cache(root, expected_tokens, *, verify_full_shards=False):
    root = Path(root)
    identity = _read_sealed(root/'IDENTITY.json')
    records, shards = _cache_index(root, identity)
    expected = set(expected_tokens)
    if len(expected) != len(expected_tokens) or set(records) != expected:
        raise ValueError(f'Target coverage incomplete/foreign: {len(records)} versus {len(expected)}')
    if verify_full_shards:
        for name, info in shards.items():
            if file_sha256(root/name) != info['sha256']:
                raise ValueError('Corrupt whole teacher shard: '+name)
    manifest = _sealed(dict(status='COMPLETE', schema='world_postcot_teacher_manifest_v1',
        identity=identity['sha256'], cohort=identity['cohort'], count=len(records),
        token_sha256=identity_hash(sorted(expected)), records=records, shards=shards,
        whole_shards_rehashed=bool(verify_full_shards)))
    path = root/'MANIFEST.json'
    if path.exists():
        previous = _read_sealed(path)
        # Re-verification may be stronger; never replace immutable provenance.
        if any(previous.get(k) != manifest.get(k) for k in ('identity','cohort','count','token_sha256','records','shards')):
            raise ValueError('Previously complete teacher cache changed')
        return previous
    atomic_json(path, manifest)
    return manifest


class TeacherCacheReader:
    """pread only the requested record; bounded FDs and no giant-shard reload.

    Whole-shard checksums are computed by the committing builder and optionally
    audited once by cache-verify. Every access validates the immutable file
    version and full record checksum; the index/manifest itself is sealed.
    """
    def __init__(self, root, *, expected_cohort, expected_identity=None,
                 max_open_shards=8, allow_building=False):
        self.root = Path(root).resolve()
        self.identity = _read_sealed(self.root/'IDENTITY.json')
        if self.identity['cohort'] != expected_cohort:
            raise ValueError('Teacher cache data-role mismatch')
        if expected_identity is not None and self.identity['sha256'] != expected_identity:
            raise ValueError('Teacher cache identity mismatch')
        if max_open_shards < 1:
            raise ValueError('Bounded positive shard-handle budget required')
        self.max_open_shards, self.allow_building = max_open_shards, allow_building
        self._fds = OrderedDict()
        self._pid = os.getpid()
        self.records, self.shards = {}, {}
        self.refresh()

    def refresh(self):
        path = self.root/'MANIFEST.json'
        if path.exists():
            manifest = _read_sealed(path)
            if manifest['status'] != 'COMPLETE' or manifest['identity'] != self.identity['sha256']:
                raise ValueError('Wrong/incomplete teacher manifest')
            self.records, self.shards = manifest['records'], manifest['shards']
        elif self.allow_building:
            self.records, self.shards = _cache_index(self.root, self.identity)
        else:
            raise FileNotFoundError('Teacher cache is not COMPLETE; partial admission must be explicit')

    def get(self, token, *, source_record_hash=None):
        if token not in self.records and self.allow_building:
            self.refresh()
        if token not in self.records:
            raise KeyError('Teacher target not committed: '+str(token))
        row = self.records[token]
        if source_record_hash is not None and row['metadata'].get('source_record_hash') != source_record_hash:
            raise ValueError('Cached teacher target belongs to a different observation')
        if os.getpid() != self._pid:
            self.close()
            self._pid = os.getpid()
        path = (self.root/row['file']).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Teacher shard escaped cache namespace')
        if row['file'] in self._fds:
            fd = self._fds.pop(row['file'])
        else:
            fd = os.open(path, os.O_RDONLY)
        self._fds[row['file']] = fd
        while len(self._fds) > self.max_open_shards:
            _, expired = self._fds.popitem(last=False)
            os.close(expired)
        stat = os.fstat(fd)
        info = self.shards.get(row['file'])
        if info is None:
            # BUILDING mode uses only complete shard receipts.
            self.refresh()
            info = self.shards[row['file']]
        if stat.st_size != info['bytes'] or stat.st_mtime_ns != info['mtime_ns']:
            raise ValueError('Teacher shard was replaced or modified')
        payload = os.pread(fd, row['bytes'], row['offset'])
        if len(payload) != row['bytes'] or hashlib.sha256(payload).hexdigest() != row['record_sha256']:
            raise ValueError('Corrupt/truncated teacher target record')
        buffer = bytearray(payload)
        output = {}
        for name, field in self.identity['layout']['fields'].items():
            view = memoryview(buffer)[field['offset']:field['offset']+field['bytes']]
            output[name] = torch.frombuffer(view, dtype=_DTYPES[field['dtype']]).reshape(field['shape'])
        output['metadata'] = row['metadata']
        return output

    __getitem__ = get

    def close(self):
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()

    def __del__(self):
        if hasattr(self, '_fds'):
            self.close()
