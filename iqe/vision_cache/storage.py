"""Rank-independent immutable shards with authenticated, bounded record reads.

One record is a small JSON tensor directory followed by contiguous raw tensors.
pread reads only the selected record. Full-shard checksums are checked by the
offline verifier; every online read verifies its own record against the signed
(SHA256-addressed) index. No pickle, whole-shard deserialization or tensor LRU.
"""
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
import fcntl
import hashlib
import json
import os
import struct
import threading
import time
import uuid
import numpy as np
import torch
from ..contracts import require
from ..io import atomic_json, digest, file_hash, json_bytes, read_json
from ..performance.vision_cache import FrozenVisionPayload

SCHEMA = 'iqe_frozen_visual_fp32_v2'
MAGIC = b'IQEVIS02'
DTYPES = {'float32': np.dtype('<f4'), 'int64': np.dtype('<i8')}


def validate_payload(p, slot_names=None):
    p.validate(slot_names or p.slots)
    for t in (p.image_embeddings, *p.deepstack):
        require(t.dtype == torch.float32 and t.device.type == 'cpu', 'FP32 CPU visual payload required')
        require(bool(torch.isfinite(t).all()), 'nonfinite frozen visual payload')
    require(p.position_ids.dtype == p.image_grid_thw.dtype == torch.int64, 'integer visual layout required')
    require(bool((p.image_grid_thw > 0).all()), 'positive image grid required')
    require(p.input_ids.numel() > 0, 'empty token payload')
    return p


def encode_record(payload):
    validate_payload(payload)
    tensors = {'input_ids': payload.input_ids, 'position_ids': payload.position_ids,
               'image_grid_thw': payload.image_grid_thw, 'image_embeddings': payload.image_embeddings,
               **{f'deepstack/{i}': t for i, t in enumerate(payload.deepstack)},
               **{f'slots/{n}': t for n, t in payload.slots.items()}}
    directory, pieces, offset = {}, [], 0
    for n, tensor in sorted(tensors.items()):
        value = tensor.detach().cpu().contiguous().numpy()
        dtype = str(value.dtype)
        require(dtype in DTYPES, 'unsupported production payload dtype')
        part = value.tobytes(order='C')
        directory[n] = {'dtype': dtype, 'shape': list(value.shape), 'offset': offset, 'bytes': len(part)}
        pieces.append(part); offset += len(part)
    header = json_bytes({'schema': SCHEMA, 'tensors': directory})
    return MAGIC + struct.pack('<Q', len(header)) + header + b''.join(pieces)


def decode_record(raw):
    require(raw[:8] == MAGIC and len(raw) >= 16, 'invalid visual record magic')
    size = struct.unpack('<Q', raw[8:16])[0]
    require(0 < size < 1024 * 1024 and 16 + size <= len(raw), 'invalid record header length')
    header = json.loads(raw[16:16+size]); require(header['schema'] == SCHEMA, 'visual schema mismatch')
    arrays = {}; cursor = 0
    for name, info in sorted(header['tensors'].items()):
        require(info['dtype'] in DTYPES and info['offset'] == cursor, 'invalid tensor directory')
        shape = tuple(info['shape']); require(all(type(v) is int and v >= 0 for v in shape), 'invalid shape')
        dtype = DTYPES[info['dtype']]; count = int(np.prod(shape, dtype=np.int64))
        require(count * dtype.itemsize == info['bytes'] and cursor + info['bytes'] <= len(raw)-16-size,
                'tensor byte range mismatch')
        # copy owns exactly the record tensors, not a complete shard or mmap.
        arrays[name] = torch.from_numpy(np.frombuffer(raw, dtype=dtype, count=count,
                                      offset=16+size+cursor).reshape(shape).copy())
        cursor += info['bytes']
    require(cursor == len(raw)-16-size, 'trailing/partial record bytes')
    allowed = {'input_ids', 'position_ids', 'image_grid_thw', 'image_embeddings',
               'deepstack/0', 'deepstack/1', 'deepstack/2'}
    require(allowed <= arrays.keys() and all(n in allowed or n.startswith('slots/') for n in arrays),
            'illegal payload tensor (trainable state or labels forbidden)')
    return validate_payload(FrozenVisionPayload(arrays['input_ids'], arrays['position_ids'],
        {n[6:]: t for n, t in arrays.items() if n.startswith('slots/')}, arrays['image_embeddings'],
        tuple(arrays[f'deepstack/{i}'] for i in range(3)), arrays['image_grid_thw']))


@contextmanager
def shard_lease(root, shard_id):
    """flock is released by kernel on interruption; diagnostics are append-only."""
    folder = Path(root)/'leases'; folder.mkdir(parents=True, exist_ok=True)
    with (folder/f'{shard_id}.lock').open('a+b') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


class ShardWriter:
    def __init__(self, root, shard_id, identity_hash, max_bytes=2*1024**3):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.shard_id, self.identity_hash, self.max_bytes = shard_id, identity_hash, max_bytes
        self.path = self.root/f'{shard_id}.bin'
        require(not self.path.exists(), 'immutable shard already exists; verify/reuse its receipt')
        self.temporary = self.root/f'.{shard_id}.{uuid.uuid4().hex}.partial'
        self.file = self.temporary.open('xb'); self.sha = hashlib.sha256(); self.records = {}; self.bytes = 0

    def add(self, key, payload):
        require(key not in self.records, 'duplicate key within shard')
        raw = encode_record(payload)
        require(self.bytes + len(raw) <= self.max_bytes, 'shard capacity exceeded before commit')
        self.file.write(raw); self.sha.update(raw)
        self.records[key] = {'shard': self.path.name, 'offset': self.bytes, 'length': len(raw),
                             'sha256': hashlib.sha256(raw).hexdigest()}
        self.bytes += len(raw)

    def commit(self):
        self.file.flush(); os.fsync(self.file.fileno()); self.file.close()
        receipt = {'schema': SCHEMA, 'identity_hash': self.identity_hash, 'bytes': self.bytes,
                   'sha256': self.sha.hexdigest(), 'records': self.records}
        # Intent is durable before rename. Recovery can finish this precise commit
        # window, while an uncommitted temporary file never looks complete.
        atomic_json(str(self.path)+'.INTENT.json', receipt, immutable=True)
        os.replace(self.temporary, self.path)
        _fsync_dir(self.root)
        atomic_json(str(self.path)+'.COMPLETE.json', receipt, immutable=True)
        return receipt

    def abort(self):
        if not self.file.closed: self.file.close()
        # Keep partial bytes for evidence; a later build starts a new temporary.


def _fsync_dir(path):
    fd = os.open(path, os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def recover_shard(root, shard_id, identity_hash):
    path = Path(root)/f'{shard_id}.bin'; receipt = Path(str(path)+'.COMPLETE.json')
    if not path.exists():
        require(not receipt.exists(), 'receipt without shard bytes')
        return None
    source = receipt if receipt.exists() else Path(str(path)+'.INTENT.json')
    require(source.exists(), 'orphan final shard without commit intent; preserve for audit')
    value = read_json(source)
    require(value['identity_hash'] == identity_hash and value['bytes'] == path.stat().st_size and
            value['sha256'] == file_hash(path), 'existing shard identity/checksum conflict')
    if not receipt.exists(): atomic_json(receipt, value, immutable=True)
    return value


def commit_cache(root, identity, access, shard_ids, *, plan_hash):
    root = Path(root); records, shards = {}, {}
    for sid in shard_ids:
        value = recover_shard(root, sid, digest(identity))
        require(value is not None, f'incomplete shard: {sid}')
        for key, row in value['records'].items():
            require(key not in records, 'ambiguous cross-shard duplicate key')
            records[key] = row
        shards[f'{sid}.bin'] = {k: value[k] for k in ('sha256', 'bytes')}
    required = {row['key'] for row in access.values()}
    require(required == set(records), 'full requested observation coverage required')
    index = {'schema': SCHEMA, 'identity': identity, 'plan_hash': plan_hash,
             'access': access, 'records': records, 'shards': shards}
    atomic_json(root/'INDEX.json', index, immutable=True)
    receipt = {'status': 'COMPLETE', 'schema': SCHEMA, 'index_sha256': file_hash(root/'INDEX.json'),
               'identity_hash': digest(identity), 'plan_hash': plan_hash, 'scenes': len(access),
               'unique_payloads': len(records), 'bytes': sum(s['bytes'] for s in shards.values())}
    atomic_json(root/'COMPLETE.json', receipt, immutable=True)
    return receipt


class CacheReader:
    def __init__(self, root, *, identity=None, allowed_scene_ids=None, max_open_shards=8):
        require(1 <= max_open_shards <= 128, 'bounded open-file cache required')
        self.root = Path(root); self.complete = read_json(self.root/'COMPLETE.json')
        require(self.complete['status'] == 'COMPLETE' and self.complete['schema'] == SCHEMA,
                'cache incomplete/foreign schema')
        require(file_hash(self.root/'INDEX.json') == self.complete['index_sha256'], 'cache index checksum mismatch')
        self.index = read_json(self.root/'INDEX.json')
        require(digest(self.index['identity']) == self.complete['identity_hash'], 'cache identity mismatch')
        if identity is not None: require(self.index['identity'] == identity, 'foreign visual numeric identity')
        self.allowed = set(allowed_scene_ids or self.index['access'])
        require(self.allowed <= self.index['access'].keys(), 'requested role/list lacks cache coverage')
        self.handles = OrderedDict(); self.max_open = max_open_shards; self.mutex = threading.RLock()
        self.stats = {'reads': 0, 'bytes': 0, 'read_seconds': 0., 'checksum_seconds': 0.,
                      'misses': 0, 'max_open_handles': 0}

    def access(self, scene_id):
        require(scene_id in self.allowed, 'scene not authorized by this access manifest')
        return self.index['access'][scene_id]

    def read(self, scene_id):
        row = self.access(scene_id); key = row['key']
        require(key in self.index['records'], 'CACHE_MISS')
        item = self.index['records'][key]; name = item['shard']; path = self.root/name
        require(Path(name).name == name, 'illegal shard path')
        with self.mutex:
            stat = path.stat(); stamp = (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            require(stat.st_size == self.index['shards'][name]['bytes'], 'shard byte-size changed')
            if name in self.handles:
                fd, previous = self.handles.pop(name); require(stamp == previous, 'immutable shard changed')
            else: fd = os.open(path, os.O_RDONLY)
            self.handles[name] = (fd, stamp)
            while len(self.handles) > self.max_open:
                _, (old, _) = self.handles.popitem(last=False); os.close(old)
            self.stats['max_open_handles'] = max(self.stats['max_open_handles'], len(self.handles))
            # dup keeps a read safe while a concurrent thread evicts its LRU entry.
            fd = os.dup(fd)
        began = time.perf_counter()
        try:
            raw = os.pread(fd, item['length'], item['offset'])
        finally: os.close(fd)
        elapsed = time.perf_counter()-began; began = time.perf_counter()
        require(len(raw) == item['length'] and hashlib.sha256(raw).hexdigest() == item['sha256'],
                'corrupt/truncated visual record')
        check_seconds = time.perf_counter()-began
        result = decode_record(raw)
        with self.mutex:
            self.stats['reads'] += 1; self.stats['bytes'] += len(raw)
            self.stats['read_seconds'] += elapsed; self.stats['checksum_seconds'] += check_seconds
        return result

    def close(self):
        with self.mutex:
            for fd, _ in self.handles.values(): os.close(fd)
            self.handles.clear()

    def verify(self, *, full_shards=False, scene_ids=None):
        if full_shards:
            for name, row in self.index['shards'].items():
                require(file_hash(self.root/name) == row['sha256'], 'full shard checksum mismatch')
        ids = list(scene_ids or sorted(self.allowed))
        for token in ids: self.read(token)
        return {'status': 'PASS', 'scenes': len(ids), 'full_shards': full_shards,
                'index_sha256': self.complete['index_sha256'], 'stats': dict(self.stats)}
