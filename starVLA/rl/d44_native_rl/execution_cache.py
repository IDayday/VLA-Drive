"""Opt-in per-_run invariants, without changing Qwen model/scientific source.

Only native rotary constants and additive masks are reused. Hidden states,
current C/P, prefix K/V, LoRA outputs and all autograd graphs are never cached.
"""
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps

import torch


PROTOCOL = 'd46_per_run_rotary_additive_execution_cache_v1'


def _tensor_key(value):
    if not isinstance(value, torch.Tensor):
        return None
    try:
        version = value._version
    except RuntimeError:  # Inference tensors have no mutation/version guard.
        return None
    return (id(value), version, tuple(value.shape), tuple(value.stride()),
            value.storage_offset(), value.dtype, value.device, value.layout)


def _constant_key(value):
    values = (value,) if isinstance(value, torch.Tensor) else value
    if not isinstance(values, (tuple, list)) or not values:
        return None
    keys = []
    for tensor in values:
        if not isinstance(tensor, torch.Tensor) or tensor.requires_grad or tensor.grad_fn is not None:
            return None
        key = _tensor_key(tensor)
        if key is None:
            return None
        keys.append(key)
    return tuple(keys)


@dataclass
class _Scope:
    enabled: bool = True
    rotary: dict = field(default_factory=dict)
    additive: dict = field(default_factory=dict)

    def clear(self):
        self.rotary.clear()
        self.additive.clear()


class InvariantCache:
    """Instance-local hooks, active only inside one original decision._run.

    Defaults install nothing. ``close`` exactly restores instance/class method
    lookup. Nested _run calls disable both caches and invalidate the outer scope;
    checkpoint backward replay outside _run calls the original functions.
    """
    def __init__(self, decision, *, rotary=False, additive=False):
        if type(rotary) is not bool or type(additive) is not bool:
            raise TypeError('Execution cache flags must be bool')
        self.decision, self.rotary, self.additive = decision, rotary, additive
        self.stats = Counter()
        self._scope = ContextVar('d46_invariant_scope_' + str(id(self)), default=None)
        self._patches = []
        self._closed = False
        if not (rotary or additive):
            return
        if getattr(decision, '_d46_invariant_cache', None) is not None:
            raise ValueError('An invariant cache is already installed on this decision')
        rotation = decision.language_model.rotary_emb
        if any(p.requires_grad for p in rotation.parameters()):
            raise ValueError('Rotary cache requires a frozen no-grad native module')
        original_run, original_rotation, original_additive = decision._run, rotation.forward, decision._additive

        @wraps(original_run)
        def run(*args, **kwargs):
            parent = self._scope.get()
            if parent is not None:
                parent.clear()
                self.stats['nested_bypass'] += 1
            scope = _Scope(enabled=parent is None)
            token = self._scope.set(scope)
            self.stats['runs'] += 1
            try:
                return original_run(*args, **kwargs)
            finally:
                scope.clear()
                self._scope.reset(token)

        @wraps(original_rotation)
        def rotate(hidden, positions):
            scope = self._scope.get()
            if not rotary or scope is None or not scope.enabled:
                self.stats['rotary_bypass'] += 1
                return original_rotation(hidden, positions)
            key = _tensor_key(positions)
            if key is not None:
                # Native rotary reads hidden dtype/device, never hidden values.
                key = (key, hidden.dtype, hidden.device, tuple(hidden.shape), hidden.layout)
            return self._cached(scope.rotary, key, (positions,),
                lambda: original_rotation(hidden, positions), 'rotary')

        @wraps(original_additive)
        def mask(allowed, dtype):
            scope = self._scope.get()
            if not additive or scope is None or not scope.enabled:
                self.stats['additive_bypass'] += 1
                return original_additive(allowed, dtype)
            key = _tensor_key(allowed)
            if key is not None:
                key = (key, dtype)
            return self._cached(scope.additive, key, (allowed,),
                lambda: original_additive(allowed, dtype), 'additive')

        self._patch(decision, '_run', run)
        self._patch(rotation, 'forward', rotate)
        self._patch(decision, '_additive', mask)
        object.__setattr__(decision, '_d46_invariant_cache', self)

    def _patch(self, target, name, wrapper):
        self._patches.append((target, name, name in target.__dict__, target.__dict__.get(name), wrapper))
        object.__setattr__(target, name, wrapper)

    def _cached(self, entries, key, inputs, compute, name):
        if key is not None and key in entries:
            parents, value, output_key = entries[key]
            # Strong input references prevent Python identity reuse; tensor
            # versions protect against in-place edits of inputs OR cached output.
            if all(a is b for a, b in zip(parents, inputs)) and _constant_key(value) == output_key:
                self.stats[name + '_hits'] += 1
                return value
        self.stats[name + '_misses'] += 1
        result = compute()
        output_key = _constant_key(result)
        if key is not None and output_key is not None:
            entries[key] = (inputs, result, output_key)
        return result

    @property
    def active(self):
        return self._scope.get() is not None

    def close(self):
        if self._closed:
            return
        if self.active:
            raise RuntimeError('Cannot uninstall a cache inside decision._run')
        for target, name, existed, previous, wrapper in reversed(self._patches):
            if target.__dict__.get(name) is not wrapper:
                raise RuntimeError('An execution hook was replaced outside this cache')
            if existed:
                object.__setattr__(target, name, previous)
            else:
                object.__delattr__(target, name)
        if self._patches:
            object.__delattr__(self.decision, '_d46_invariant_cache')
        self._patches.clear()
        self._closed = True


def install_invariant_cache(decision, *, rotary=False, additive=False):
    return InvariantCache(decision, rotary=rotary, additive=additive)


@contextmanager
def invariant_cache(decision, *, rotary=False, additive=False):
    """Install for an explicit execution context; restore even after failure."""
    cache = install_invariant_cache(decision, rotary=rotary, additive=additive)
    try:
        yield cache
    finally:
        cache.close()
