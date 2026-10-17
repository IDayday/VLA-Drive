"""Explicit owner-process DINO executor for BoundedTeacherProvider.

This is not a service or a training launcher. Construction must be separately
resource-admitted. No CUDA discovery/teacher loading occurs on module import.
"""
import os
import random
import threading
from contextlib import contextmanager

import numpy as np
import torch

from starVLA.dataloader.st_world_ondemand import TeacherBatch
from starVLA.dataloader.st_world_schema import identity_hash
from starVLA.dataloader.st_world_history import file_sha256
from starVLA.dataloader.st_world_targets import payload_key


@contextmanager
def isolated_rng(device):
    py, np_state = random.getstate(), np.random.get_state()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)


class FrozenDINOExecutor:
    """Same preprocess/model/pool/storage operators as cache_teacher.run.

    Native contracts and directly computed pool3 are supported. Legacy C1 and
    source-cache-derived pooling identities cannot be regenerated under an old
    identity and are explicitly rejected. Changing batch shape still needs real
    numerical qualification against the corresponding immutable teacher cache.
    """
    def __init__(self, teacher_root, *, contract, variant, device, max_batch_images,
                 expected_execution_identity=None):
        if threading.current_thread() is not threading.main_thread() or torch.utils.data.get_worker_info() is not None:
            raise RuntimeError('Create frozen teacher only in the owning main process')
        if type(max_batch_images) is not int or not 1 <= max_batch_images <= 64:
            raise ValueError('Explicit teacher batch limit must be 1..64')
        if variant not in ('SP1', 'SP2', 'SP3') or contract.source_cache_identity is not None:
            raise ValueError('Cannot regenerate a legacy/derived-cache contract under its old identity')
        from tools.st_world_policy.cache_teacher import make_contract, preprocess_image
        from starVLA.model.modules.foresight.dinov3_target_encoder import DINOv3TargetEncoder
        actual, teacher, transform = make_contract(teacher_root, variant, contract.storage_dtype)
        if actual.identity != contract.identity:
            raise ValueError('Actual teacher/transform differs from requested target contract')
        self.device = torch.device(device)
        if self.device.type not in ('cpu', 'cuda'):
            raise ValueError('Only explicitly admitted CPU/CUDA teacher devices are supported')
        self.execution = {'schema': 'st_ondemand_dino_execution_v1', 'teacher': teacher,
            'executor_source_sha256': file_sha256(__file__),
            'transform': transform, 'contract_identity': contract.identity, 'device_type': self.device.type,
            'max_batch_images': max_batch_images, 'compute_dtype': 'float32', 'tf32': False,
            'autocast': False, 'tensor_disk_writes': 0,
            'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version() if self.device.type == 'cuda' else None,
            'flash_sdp_enabled': torch.backends.cuda.flash_sdp_enabled(),
            'mem_efficient_sdp_enabled': torch.backends.cuda.mem_efficient_sdp_enabled(),
            'math_sdp_enabled': torch.backends.cuda.math_sdp_enabled(),
            'cpu_threads': torch.get_num_threads(),
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cudnn_benchmark': torch.backends.cudnn.benchmark, 'cudnn_deterministic': torch.backends.cudnn.deterministic}
        self.execution_identity = identity_hash(self.execution)
        if expected_execution_identity is not None and expected_execution_identity != self.execution_identity:
            raise ValueError('On-demand execution runtime changed')
        self.contract_identity = contract.identity
        self.max_batch_images = max_batch_images
        self._pid, self._thread = os.getpid(), threading.get_ident()
        self._preprocess = preprocess_image
        self._check_backend()
        with isolated_rng(self.device):
            self.model = DINOv3TargetEncoder(teacher_root, device=self.device)
        if (self.model.patch_size != contract.patch_hw or self.model.feature_dim != contract.channels or self.model.prefix != 5):
            raise ValueError('Teacher architecture differs from target contract')
        self._versions = self._frozen_versions()

    def _check_backend(self):
        if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
            raise ValueError('Teacher requires already configured TF32-disabled runtime; do not mutate active trainer flags')
        for name, actual in (
                ('deterministic_algorithms', torch.are_deterministic_algorithms_enabled()),
                ('flash_sdp_enabled', torch.backends.cuda.flash_sdp_enabled()),
                ('mem_efficient_sdp_enabled', torch.backends.cuda.mem_efficient_sdp_enabled()),
                ('math_sdp_enabled', torch.backends.cuda.math_sdp_enabled()),
                ('cpu_threads', torch.get_num_threads()),
                ('cudnn_benchmark', torch.backends.cudnn.benchmark),
                ('cudnn_deterministic', torch.backends.cudnn.deterministic)):
            if actual != self.execution[name]:
                raise ValueError('Teacher execution backend changed after binding: ' + name)

    def _frozen_versions(self):
        if any(m.training for m in self.model.modules()) or any(p.requires_grad for p in self.model.parameters()):
            raise ValueError('Teacher must be frozen and eval throughout')
        tensors = list(self.model.named_parameters()) + list(self.model.named_buffers())
        if any(t.is_floating_point() and t.dtype != torch.float32 for _, t in tensors):
            raise ValueError('Teacher parameters/buffers must retain FP32')
        return tuple((name, id(t), t._version) for name, t in tensors)

    def compute_batch(self, records, *, contract, execution_identity):
        if (os.getpid() != self._pid or threading.get_ident() != self._thread or
                threading.current_thread() is not threading.main_thread() or torch.utils.data.get_worker_info() is not None):
            raise RuntimeError('Teacher executor cannot run in loader/process/thread workers')
        if (contract.identity != self.contract_identity or execution_identity != self.execution_identity or
                not records or len(records) > self.max_batch_images):
            raise ValueError('Teacher batch outside admitted identity/budget')
        keys = tuple(row['key'] for row in records)
        if len(set(keys)) != len(keys) or any(payload_key(row['image'], contract) != row['key'] for row in records):
            raise ValueError('Teacher batch keys differ from image content contract')
        self._check_backend()
        if self._frozen_versions() != self._versions:
            raise ValueError('Frozen teacher was mutated')
        with isolated_rng(self.device), torch.inference_mode():
            pixels = torch.stack([self._preprocess(row['image'], contract) for row in records]).to(self.device)
            native = self.model(pixels)
            if contract.pooling['kind'] == 'avg':
                factor = contract.pooling['kernel_hw']
                native = torch.nn.functional.avg_pool2d(native, factor, factor)
            z = native.permute(0, 2, 3, 1).contiguous().cpu().to(getattr(torch, contract.storage_dtype))
            mask = torch.ones(z.shape[:-1], dtype=torch.bool)
        if self._frozen_versions() != self._versions:
            raise ValueError('Teacher inference mutated frozen state')
        return TeacherBatch(keys, contract.identity, self.execution_identity, z, mask)
