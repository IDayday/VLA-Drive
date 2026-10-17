"""Bounded, label-only teacher provider; no model construction or tensor writes.

The synchronous batch executor is injected by the training *owner*, never by a
DataLoader worker. This first adapter preserves read_scene's all-future order;
request-aware pair fetching is deliberately not implemented here.
"""
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import os
import random
import threading

import numpy as np
import torch

from .st_world_schema import identity_hash, require_hash
from .st_world_history import file_sha256
from .st_world_targets import TeacherCacheReader, payload_key, validate_target_manifest


@dataclass(frozen=True)
class TeacherBatch:
    keys: tuple
    contract_identity: str
    execution_identity: str
    features: torch.Tensor
    valid: torch.Tensor


@contextmanager
def preserve_cpu_rng():
    """Only legal on the owning main thread, with no concurrent training call."""
    py, np_state, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu)


class BoundedTeacherProvider:
    """Reuse COMPLETE same-role stores, compute exact misses, retain bounded RAM.

    executor.compute_batch(records, contract=..., execution_identity=...) must
    return TeacherBatch with the exact requested key order and CPU stored-dtype
    tensors. It may block until those records finish or raise; no background
    retry, substitution, rank collective, disk publication or sampler exists.
    Caller must use the trainer's existing collective_call around read_scene.
    A process-group timeout remains mandatory for an executor/process hang.
    """
    def __init__(self, manifest, contract, sources=(), *, executor=None,
                 execution_identity=None, max_cached_bytes, max_scene_bytes,
                 max_batch_images, max_manifest_bytes):
        import json
        for name, value in locals().copy().items():
            if name.startswith('max_') and (type(value) is not int or value <= 0):
                raise ValueError(name + ' must be a positive explicit integer')
        if max_batch_images > 64:
            raise ValueError('Batch executor is bounded to at most 64 images')
        if len(json.dumps(manifest, sort_keys=True).encode()) > max_manifest_bytes:
            raise ValueError('Request manifest exceeds the declared metadata budget')
        self.manifest = validate_target_manifest(deepcopy(manifest))
        self.contract = contract.validate()
        if tuple(self.manifest['view_names']) != self.contract.view_names:
            raise ValueError('Request/teacher view contract differs')
        self.scenes = {s['token']: s for s in self.manifest['scenes']}
        self.sources = tuple(sources)
        for source in self.sources:
            if not isinstance(source, TeacherCacheReader) or source.contract.identity != contract.identity:
                raise ValueError('Reuse needs a COMPLETE reader with exactly the same teacher contract')
            for key in ('split', 'source_student_identity', 'source_index_sha256', 'partition_sha256'):
                if not self.manifest.get(key) or self.manifest[key] != source.manifest.get(key):
                    raise ValueError('Source role/population lineage differs: ' + key)
        self.executor = executor
        if executor is not None:
            require_hash(execution_identity, 'executor execution identity')
            if getattr(executor, 'execution_identity', None) != execution_identity:
                raise ValueError('Executor identity mismatch')
            if getattr(executor, 'contract_identity', None) != contract.identity:
                raise ValueError('Executor teacher identity mismatch')
        elif execution_identity is not None:
            raise ValueError('No executor exists for the declared execution identity')
        self.execution_identity = execution_identity
        h, w = contract.output_grid_hw
        self.record_bytes = h*w*(contract.channels*(4 if contract.storage_dtype == 'float32' else 2)+1)
        self.max_cached_bytes, self.max_scene_bytes = max_cached_bytes, max_scene_bytes
        self.max_batch_images = max_batch_images
        self.identity = {'schema': 'st_bounded_teacher_provider_v1',
            'provider_source_sha256': file_sha256(__file__),
            'request_manifest_identity': self.manifest['identity'], 'contract': contract.to_dict(),
            'source_cache_identities': [x.identity['identity'] for x in self.sources],
            'execution_identity': execution_identity, 'max_cached_bytes': max_cached_bytes,
            'max_scene_bytes': max_scene_bytes, 'max_batch_images': max_batch_images,
            'max_manifest_bytes': max_manifest_bytes, 'tensor_disk_writes': 0,
            'miss_policy': 'same_request_block_or_raise', 'fetch_policy': 'all_declared_future_times'}
        self.identity['identity'] = identity_hash(self.identity)
        self._pid, self._thread = os.getpid(), threading.get_ident()
        self._resident = OrderedDict()
        self._bytes = 0
        self.stats = dict(ram_hits=0, immutable_hits=0, computed_images=0, batches=0,
                          evictions=0, failures=0, tensor_disk_bytes=0, peak_resident_bytes=0)

    def _owner(self):
        if (os.getpid() != self._pid or threading.get_ident() != self._thread or
                threading.current_thread() is not threading.main_thread() or
                torch.utils.data.get_worker_info() is not None):
            raise RuntimeError('On-demand labels require the owning main thread; loader workers cannot compute teachers')

    def verify_population(self, manifest):
        if validate_target_manifest(manifest) != self.manifest:
            raise ValueError('Requested teacher population differs')

    def _validate(self, z, valid, *, batch):
        shape = (*self.contract.output_grid_hw, self.contract.channels)
        expected = (batch, *shape) if batch is not None else shape
        if (not isinstance(z, torch.Tensor) or not isinstance(valid, torch.Tensor) or
                tuple(z.shape) != expected or valid.shape != z.shape[:-1] or
                z.dtype != getattr(torch, self.contract.storage_dtype) or valid.dtype != torch.bool or
                z.device.type != 'cpu' or valid.device.type != 'cpu' or
                z.requires_grad or valid.requires_grad or not torch.isfinite(z).all()):
            raise ValueError('Teacher result violates finite/dtype/grid/CPU/detach contract')

    def _remember(self, key, z, valid):
        if self.record_bytes > self.max_cached_bytes:
            return
        # A record owns its storage: never keep a slice pinning a whole batch.
        if key in self._resident:
            self._resident.move_to_end(key)
            return
        while self._bytes + self.record_bytes > self.max_cached_bytes:
            self._resident.popitem(last=False)
            self._bytes -= self.record_bytes
            self.stats['evictions'] += 1
        with torch.inference_mode(False):
            self._resident[key] = (z.clone(), valid.clone())
        self._bytes += self.record_bytes
        self.stats['peak_resident_bytes'] = max(self.stats['peak_resident_bytes'], self._bytes)

    def _existing(self, key):
        if key in self._resident:
            self.stats['ram_hits'] += 1
            self._resident.move_to_end(key)
            return self._resident[key]
        for source in self.sources:
            if key in source.locations:
                # Corrupt existing records are not silently replaced by compute.
                result = source.read_image(key)
                self._validate(*result, batch=None)
                self.stats['immutable_hits'] += 1
                return result
        return None

    def read_scene(self, token, *, expected_split=None, need_future=True):
        self._owner()
        if type(need_future) is not bool:
            raise ValueError('need_future must be an explicit bool')
        scene = self.scenes[token]
        if expected_split is not None and expected_split != scene['split']:
            raise ValueError('Teacher scene role mismatch')
        times = self.manifest['times_s'] if need_future else self.manifest['times_s'][:1]
        record_count = len(times)*len(self.contract.view_names)
        if record_count*self.record_bytes > self.max_scene_bytes:
            raise ValueError('Requested scene exceeds output tensor RAM budget; no partial scene allowed')
        shape = (len(times), len(self.contract.view_names), *self.contract.output_grid_hw, self.contract.channels)
        features = torch.zeros(shape, dtype=getattr(torch, self.contract.storage_dtype))
        valid = torch.zeros(shape[:-1], dtype=torch.bool)
        locations, records = {}, {}
        for t, frame in enumerate(scene['frames'][:len(times)]):
            for v, ref in enumerate(frame['views']):
                if ref is not None:
                    key = payload_key(ref, self.contract)
                    locations.setdefault(key, []).append((t, v))
                    records.setdefault(key, {'key': key, 'image': deepcopy(ref)})
        def deliver(key, z, mask):
            for t, v in locations[key]:
                features[t, v].copy_(z)
                valid[t, v].copy_(mask)
            self._remember(key, z, mask)
        missing = []
        try:
            for key in locations:
                value = self._existing(key)
                if value is None:
                    missing.append(records[key])
                else:
                    deliver(key, *value)
            for begin in range(0, len(missing), self.max_batch_images):
                if self.executor is None:
                    raise FileNotFoundError('Valid future/current source has no cached label and no admitted executor')
                batch = missing[begin:begin+self.max_batch_images]
                # Restore even on error. Dedicated horizon/patch streams are never passed.
                with preserve_cpu_rng():
                    result = self.executor.compute_batch(tuple(batch), contract=self.contract,
                                                         execution_identity=self.execution_identity)
                keys = tuple(row['key'] for row in batch)
                if (not isinstance(result, TeacherBatch) or result.keys != keys or
                        result.contract_identity != self.contract.identity or
                        result.execution_identity != self.execution_identity):
                    raise ValueError('Wrong/missing/reordered computed teacher records')
                self._validate(result.features, result.valid, batch=len(batch))
                # The native/direct-pool teacher uses the full patch-divisible
                # image with no padding. An executor failure cannot masquerade
                # as an absent source or an unavailable patch.
                if not result.valid.all():
                    raise ValueError('Computed native teacher record may not hide missing labels with false masks')
                for i, key in enumerate(keys):
                    deliver(key, result.features[i], result.valid[i])
                self.stats['batches'] += 1
                self.stats['computed_images'] += len(batch)
                # Do not hold a previous executor batch while allocating the next.
                del result
        except Exception:
            self.stats['failures'] += 1
            raise
        return {'current_dense_features': features[0], 'current_valid': valid[0],
                'future_dense_features': features[1:], 'future_valid': valid[1:],
                'future_times_s': times[1:], 'teacher_contract': self.contract,
                'cache_identity': self.identity['identity']}

    def state_dict(self):
        self._owner()
        return {'schema': 'st_bounded_teacher_state_v1', 'identity': self.identity['identity']}

    def load_state_dict(self, state):
        self._owner()
        if state != self.state_dict():
            raise ValueError('Resume teacher provider identity changed')
        self._resident.clear()
        self._bytes = 0  # Hot records are recomputable, never a consumed data cursor.

    def memory_report(self):
        return {**self.stats, 'resident_bytes': self._bytes,
                'resident_limit_bytes': self.max_cached_bytes, 'scene_limit_bytes': self.max_scene_bytes,
                'executor_output_limit_bytes': self.max_batch_images*self.record_bytes,
                'limit_scope': 'tensor payloads only; metadata, teacher weights/activations, pixels and caller batches separate'}
