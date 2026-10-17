"""Opt-in, versioned fixed-8 frozen teacher execution; no launch or cache writes.

The legacy executor remains unchanged. Padding exists only inside the teacher
compute batch; returned keys, features and masks contain actual requests only.
Mathematical image independence does not establish bitwise hardware equivalence:
the separate qualification runner must exercise regrouping and partial LRU hits.
"""
from copy import deepcopy
import inspect
import os
from pathlib import Path
import threading

import torch

from starVLA.dataloader.st_world_history import file_sha256
from starVLA.dataloader.st_world_ondemand import TeacherBatch
from starVLA.dataloader.st_world_schema import identity_hash
from starVLA.dataloader.st_world_targets import payload_key
from tools.st_world_policy import ondemand_teacher as legacy


FIXED_BATCH_POLICY = {
    'schema': 'st_teacher_fixed_compute_batch_v1',
    'compute_batch_images': 8,
    'request_batch_images': [1, 8],
    'padding': 'repeat_last_real_preprocessed_input',
    'return': 'actual_requests_only_original_order',
    'budget': 'count_all_eight_compute_slots_including_padding',
}


def pad_teacher_pixels(pixels):
    """No synthetic record is created; padding never reaches the provider."""
    if (pixels.ndim != 4 or pixels.shape[1] != 3 or
            not 1 <= len(pixels) <= 8 or pixels.dtype != torch.float32 or
            pixels.requires_grad or not torch.isfinite(pixels).all()):
        raise ValueError('Fixed teacher pixels require 1..8 finite detached FP32 RGB inputs')
    if len(pixels) == 8:
        return pixels
    return torch.cat((pixels, pixels[-1:].expand(8-len(pixels), -1, -1, -1)), dim=0)


def _operator_sources(model):
    """Bind actual installed code, not only a potentially reused package tag."""
    if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()):
        raise ValueError('Fixed teacher requires per-image operators; BatchNorm is not admitted')
    sources = {}
    for module in model.modules():
        cls = type(module)
        key = cls.__module__ + '.' + cls.__qualname__
        source = inspect.getsourcefile(cls)
        if source and key not in sources:
            sources[key] = {'file_name': Path(source).name, 'sha256': file_sha256(source)}
    return sources


def _operator_settings(model):
    return {name: {key:getattr(module,key) for key in ('fused_attn','grad_checkpointing','_fast_norm')
                   if hasattr(module,key)} for name,module in model.named_modules()
            if any(hasattr(module,key) for key in ('fused_attn','grad_checkpointing','_fast_norm'))}


def require_matching_qualification(qualification, execution_identity):
    """Narrow numeric check, not formal launch approval or legacy cache approval."""
    if (qualification.get('schema') != 'st_fixed_teacher_qualification_v1' or
            qualification.get('identity') != identity_hash({k:v for k,v in qualification.items() if k!='identity'}) or
            qualification.get('execution_identity') != execution_identity or
            qualification.get('batch_policy') != FIXED_BATCH_POLICY or
            qualification.get('fixed_policy_repeatability') is not True or
            qualification.get('completed_all_cases') is not True or
            qualification.get('backend', {}).get('deterministic_algorithms') is not True):
        raise ValueError('Missing or mismatched fixed-batch numerical qualification')


class FixedBatchDINOExecutor(legacy.FrozenDINOExecutor):
    """FP32 DINO with exactly eight compute slots and a distinct execution ID.

    Constructor does not alter global backend flags/threads. The owner must
    configure its registered deterministic backend before construction. No CUDA
    discovery or weight loading happens at import. Instantiation needs separate
    resource admission, just like the legacy executor.
    """
    def __init__(self, teacher_root, *, contract, variant, device, max_batch_images=8,
                 expected_execution_identity=None):
        if max_batch_images != 8 or type(max_batch_images) is not int:
            raise ValueError('Fixed teacher compute batch must be exactly 8')
        if contract.storage_dtype != 'float32' or contract.compute_dtype != 'float32':
            raise ValueError('Fixed teacher contract requires FP32 compute and FP32 storage')
        if (not torch.are_deterministic_algorithms_enabled() or
                torch.is_deterministic_algorithms_warn_only_enabled()):
            raise ValueError('Fixed teacher requires deterministic algorithms=True, warn_only=False')
        super().__init__(teacher_root, contract=contract, variant=variant, device=device,
                         max_batch_images=8)
        self.execution = {
            **self.execution,
            'schema': 'st_fixed8_dino_execution_v1',
            'legacy_executor_source_sha256': file_sha256(legacy.__file__),
            'executor_source_sha256': file_sha256(__file__),
            'batch_policy': deepcopy(FIXED_BATCH_POLICY),
            'deterministic_warn_only': False,
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'operator_sources': _operator_sources(self.model),
            'operator_settings': _operator_settings(self.model),
            'device_hardware': ({'name':torch.cuda.get_device_name(self.device),
                                 'capability':list(torch.cuda.get_device_capability(self.device))}
                                if self.device.type=='cuda' else {'type':'cpu'}),
        }
        self.execution_identity = identity_hash(self.execution)
        if expected_execution_identity is not None and expected_execution_identity != self.execution_identity:
            raise ValueError('Fixed teacher execution identity differs from requested qualification')
        self.compute_stats = {'calls': 0, 'requested_images': 0, 'compute_image_slots': 0,
                              'padding_image_slots': 0}

    def _check_backend(self):
        super()._check_backend()
        # During the superclass constructor the fixed identity is not set yet.
        if self.execution.get('schema') == 'st_fixed8_dino_execution_v1':
            if (torch.is_deterministic_algorithms_warn_only_enabled() or
                    os.environ.get('CUBLAS_WORKSPACE_CONFIG') != self.execution['cublas_workspace_config'] or
                    _operator_settings(self.model) != self.execution['operator_settings']):
                raise ValueError('Fixed teacher execution backend changed after binding')

    def compute_batch(self, records, *, contract, execution_identity):
        if (os.getpid() != self._pid or threading.get_ident() != self._thread or
                threading.current_thread() is not threading.main_thread() or
                torch.utils.data.get_worker_info() is not None):
            raise RuntimeError('Fixed teacher executor requires its owning main process/thread')
        if (contract.identity != self.contract_identity or execution_identity != self.execution_identity or
                not records or not 1 <= len(records) <= 8):
            raise ValueError('Fixed teacher request outside admitted identity/budget')
        keys = tuple(row['key'] for row in records)
        if len(set(keys)) != len(keys) or any(payload_key(row['image'], contract) != row['key'] for row in records):
            raise ValueError('Fixed teacher record keys differ from actual image contract')
        self._check_backend()
        if self._frozen_versions() != self._versions:
            raise ValueError('Frozen teacher was mutated')
        try:
            with legacy.isolated_rng(self.device), torch.inference_mode(), torch.autocast(self.device.type, enabled=False):
                pixels = torch.stack([self._preprocess(row['image'], contract) for row in records]).to(self.device)
                pixels = pad_teacher_pixels(pixels)
                self.compute_stats['calls'] += 1
                self.compute_stats['requested_images'] += len(records)
                self.compute_stats['compute_image_slots'] += 8
                self.compute_stats['padding_image_slots'] += 8-len(records)
                native = self.model(pixels)
                if native.dtype != torch.float32 or len(native) != 8 or not torch.isfinite(native).all():
                    raise ValueError('Fixed teacher output must preserve all eight finite FP32 compute slots')
                if contract.pooling['kind'] == 'avg':
                    factor = contract.pooling['kernel_hw']
                    native = torch.nn.functional.avg_pool2d(native, factor, factor)
                z = native[:len(records)].permute(0, 2, 3, 1).contiguous().cpu()
                if tuple(z.shape) != (len(records), *contract.output_grid_hw, contract.channels):
                    raise ValueError('Fixed teacher output grid differs from target contract')
                mask = torch.ones(z.shape[:-1], dtype=torch.bool)
        finally:
            if self._frozen_versions() != self._versions:
                raise ValueError('Teacher inference mutated frozen state')
        return TeacherBatch(keys, contract.identity, self.execution_identity, z, mask)
