"""Strict data boundaries for ST experiments; importing this never loads a teacher.

Evidence: DINO_NATIVE_PATCH / DINO_PCA_NONWHITEN in the campaign evidence
cards. Grid order is always (height, width); targets are channel-last. History
is model input, while all current/future teacher tensors stay label-side.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import math
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

import torch


def identity_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def require_hash(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(name + ' must be a lowercase SHA256')
    return value


def pair(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 2 or any(type(x) is not int or x <= 0 for x in value):
        raise ValueError(name + ' must contain two positive integers (height,width)')
    return tuple(value)


@dataclass(frozen=True)
class TeacherFeatureContract:
    image_hw: Tuple[int, int]
    patch_hw: Tuple[int, int]
    native_grid_hw: Tuple[int, int]
    output_grid_hw: Tuple[int, int]
    channels: int
    storage_dtype: str
    compute_dtype: str
    view_names: Tuple[str, ...]
    teacher_identity: str
    transform_identity: str
    pooling: Mapping
    schema_version: str = 'st_teacher_feature_contract_v1'
    output_layout: str = 'HWC'
    feature_layer: str = 'last_block_post_norm_dense_patch'
    source_cache_identity: Optional[str] = None

    def __post_init__(self):
        for name in ('image_hw', 'patch_hw', 'native_grid_hw', 'output_grid_hw'):
            object.__setattr__(self, name, pair(getattr(self, name), name))
        if type(self.channels) is not int or self.channels <= 0:
            raise ValueError('channels must be a positive integer from the real teacher')
        if self.storage_dtype not in ('float16', 'float32') or self.compute_dtype != 'float32':
            raise ValueError('Teacher compute is FP32; storage must explicitly be float16 or float32')
        if (not isinstance(self.view_names, (list, tuple)) or not self.view_names or
                any(not isinstance(v, str) or not v for v in self.view_names) or len(set(self.view_names)) != len(self.view_names)):
            raise ValueError('view_names must be a nonempty ordered unique sequence')
        object.__setattr__(self, 'view_names', tuple(self.view_names))
        for name in ('teacher_identity', 'transform_identity'):
            require_hash(getattr(self, name), name)
        if self.source_cache_identity is not None:
            require_hash(self.source_cache_identity, 'source_cache_identity')
        if self.schema_version != 'st_teacher_feature_contract_v1' or self.output_layout != 'HWC':
            raise ValueError('Unsupported teacher schema/layout')
        if self.feature_layer != 'last_block_post_norm_dense_patch':
            raise ValueError('Unregistered feature layer; do not silently include prefix tokens')
        if any(i % p for i, p in zip(self.image_hw, self.patch_hw)):
            raise ValueError('Teacher image must be patch divisible; no silent border truncation')
        if tuple(i // p for i, p in zip(self.image_hw, self.patch_hw)) != self.native_grid_hw:
            raise ValueError('Native teacher grid does not match image and patch geometry')
        if not isinstance(self.pooling, Mapping) or set(self.pooling) != {'kind', 'kernel_hw', 'stride_hw'}:
            raise ValueError('pooling requires exactly kind/kernel_hw/stride_hw')
        k, s = pair(self.pooling['kernel_hw'], 'kernel_hw'), pair(self.pooling['stride_hw'], 'stride_hw')
        kind = self.pooling['kind']
        if kind not in ('none', 'avg') or k != s:
            raise ValueError('Only nonoverlapping registered average pooling is supported')
        if kind == 'none' and k != (1, 1):
            raise ValueError('Unpooled teacher must have kernel/stride 1')
        if any(n % x for n, x in zip(self.native_grid_hw, k)):
            raise ValueError('Pooling must retain the complete grid without border loss')
        if tuple(n // x for n, x in zip(self.native_grid_hw, k)) != self.output_grid_hw:
            raise ValueError('Stored grid disagrees with actual pooling; native is not pooled')
        object.__setattr__(self, 'pooling', MappingProxyType({'kind': kind, 'kernel_hw': k, 'stride_hw': s}))

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError('Teacher contract must be a resolved mapping')
        allowed = {f.name for f in fields(cls)}
        if set(value) - allowed:
            raise ValueError('Unknown teacher contract fields: ' + str(sorted(set(value) - allowed)))
        try:
            return cls(**value)
        except TypeError as exc:
            raise ValueError('Missing/unresolved teacher contract fields') from exc

    def to_dict(self):
        result = {f.name: getattr(self, f.name) for f in fields(self)}
        for name in ('image_hw', 'patch_hw', 'native_grid_hw', 'output_grid_hw', 'view_names'):
            result[name] = list(result[name])
        result['pooling'] = {'kind': self.pooling['kind'], 'kernel_hw': list(self.pooling['kernel_hw']),
                             'stride_hw': list(self.pooling['stride_hw'])}
        return result

    @property
    def identity(self):
        return identity_hash(self.to_dict())

    def __reduce__(self):
        # MappingProxy protects nested identity, and this supports spawn workers.
        return (type(self).from_dict, (self.to_dict(),))

    def validate(self):
        # Construction does full validation. Return self for explicit call sites.
        return self

    def validate_targets(self, values, valid, *, batched=False, future=False):
        expected_ndim = 4 + int(batched) + int(future)
        if values.ndim != expected_ndim or valid.shape != values.shape[:-1] or valid.dtype != torch.bool:
            raise ValueError('Teacher targets must be [...,view,height,width,channel] with matching bool mask')
        if tuple(values.shape[-4:]) != (len(self.view_names), *self.output_grid_hw, self.channels):
            raise ValueError('Teacher target view/grid/channel dimensions mismatch')
        if values.dtype != getattr(torch, self.storage_dtype) or values.requires_grad or not torch.isfinite(values).all():
            raise ValueError('Teacher target dtype/finite/detach contract violated')


@dataclass(frozen=True)
class ModelInputs:
    """Controlled adapter; never add history/targets to legacy current whitelist.

    ``current`` is the original current_observation return. ``history`` is a
    validated history batch supplied separately to the new framework. The
    framework must explicitly opt into it, rather than loosening encode_current.
    """
    current: Mapping
    history: Optional[Mapping] = None

    def __post_init__(self):
        if set(self.current) != {'image', 'state', 'lang', 'token'}:
            raise ValueError('Current input permits only image/state/lang/token')
        if len(self.current['image']) != 3 or not isinstance(self.current['lang'], str):
            raise ValueError('Current three-view/prompt contract changed')
        state = torch.as_tensor(self.current['state'])
        if state.shape != (1, 4) or not torch.isfinite(state).all():
            raise ValueError('Current ego state must retain original 1x4 representation')
        if self.history is not None:
            from .st_world_history import validate_history_record
            validate_history_record(self.history)
            if self.history['token'] != self.current['token']:
                raise ValueError('History/current scene mismatch')

    def legacy_current(self):
        return dict(self.current)


@dataclass(frozen=True)
class TrainingTargets:
    ego: torch.Tensor
    current_dense_features: torch.Tensor
    current_valid: torch.Tensor
    future_dense_features: torch.Tensor
    future_valid: torch.Tensor
    future_times_s: Tuple[float, ...]
    teacher_contract: TeacherFeatureContract
    cache_identity: str

    def __post_init__(self):
        require_hash(self.cache_identity, 'cache_identity')
        self.teacher_contract.validate_targets(self.current_dense_features, self.current_valid)
        self.teacher_contract.validate_targets(self.future_dense_features, self.future_valid, future=True)
        if self.future_dense_features.shape[0] != len(self.future_times_s):
            raise ValueError('Future time/feature mismatch')
        if tuple(sorted(set(self.future_times_s))) != tuple(self.future_times_s) or any(not math.isfinite(t) or t <= 0 for t in self.future_times_s):
            raise ValueError('Future labels require distinct increasing positive physical times')
        if self.ego.shape[-1] != 4 or not torch.isfinite(self.ego).all() or self.ego.requires_grad:
            raise ValueError('Original normalized xy/sincos ego labels required')
