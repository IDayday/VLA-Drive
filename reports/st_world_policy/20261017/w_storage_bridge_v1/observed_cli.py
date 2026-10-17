#!/usr/bin/env python3
"""Fixed SP0 observer around existing spatial_probe calls; no recipe changes.

Only supervise_once.py invokes this file after independent stage admission.
Successful full-file hashes and torch.load's nominal input sizes are counted
separately from /proc/self/io. Neither is a physical bandwidth measurement.
"""
import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

from storage_bridge import upstream_module, install
_control = upstream_module('prepare_bindings')
read, write_once, write_control = _control.read, _control.write_once, _control.write_control


def proc_io():
    try:
        return {k: int(v) for k, v in (line.split(':') for line in Path('/proc/self/io').read_text().splitlines())}
    except OSError:
        return None


class Counters:
    def __init__(self, target, torch, read_cap, baseline):
        self.target, self.torch, self.read_cap = Path(target), torch, read_cap
        self.started = time.monotonic(); self.baseline = baseline; self.stop = threading.Event()
        self.lock = threading.Lock()
        self.data = dict(hash_calls_started=0, hash_calls_completed=0,
                         hash_logical_bytes_admitted=0, hash_logical_bytes_completed=0,
                         hash_wall_seconds=0., torch_load_calls=0, torch_load_nominal_bytes=0,
                         torch_load_wall_seconds=0., native_load_calls=0, native_load_wall_seconds=0., gpu_peak_allocated_bytes=0,
                         gpu_peak_reserved_bytes=0, phase='IMPORTS_COMPLETE', limit_exceeded=None)

    def hash_wrapper(self, original):
        @functools.wraps(original)
        def wrapped(path, *a, **kw):
            size = Path(path).stat().st_size
            with self.lock:
                self.data['hash_calls_started'] += 1
                self.data['hash_logical_bytes_admitted'] += size
                if self.data['hash_logical_bytes_admitted'] + self.data['torch_load_nominal_bytes'] > self.read_cap:
                    self.data['limit_exceeded'] = 'known_logical_read_admission'
                    raise RuntimeError('Role logical read admission exhausted before hash')
            started = time.monotonic()
            try:
                result = original(path, *a, **kw)
            finally:
                with self.lock:
                    self.data['hash_wall_seconds'] += time.monotonic() - started
            with self.lock:
                self.data['hash_calls_completed'] += 1
                self.data['hash_logical_bytes_completed'] += size
            return result
        return wrapped

    def load_wrapper(self, original):
        @functools.wraps(original)
        def wrapped(path, *a, **kw):
            size = Path(path).stat().st_size if isinstance(path, (str, os.PathLike)) else 0
            with self.lock:
                self.data['torch_load_calls'] += 1
                self.data['torch_load_nominal_bytes'] += size
                if self.data['hash_logical_bytes_admitted'] + self.data['torch_load_nominal_bytes'] > self.read_cap:
                    self.data['limit_exceeded'] = 'known_logical_read_admission'
                    raise RuntimeError('Role logical read admission exhausted before torch.load')
            started = time.monotonic()
            try:
                return original(path, *a, **kw)
            finally:
                with self.lock:
                    self.data['torch_load_wall_seconds'] += time.monotonic() - started
        return wrapped

    def native_wrapper(self, original):
        @functools.wraps(original)
        def wrapped(*a, **kw):
            started = time.monotonic()
            self.data['native_load_calls'] += 1
            try:
                return original(*a, **kw)
            finally:
                with self.lock:
                    self.data['native_load_wall_seconds'] += time.monotonic() - started
        return wrapped

    def snapshot(self):
        with self.lock:
            if self.torch.cuda.is_initialized():
                self.data['gpu_peak_allocated_bytes'] = self.torch.cuda.max_memory_allocated(0)
                self.data['gpu_peak_reserved_bytes'] = self.torch.cuda.max_memory_reserved(0)
            current = proc_io()
            delta = ({k: current[k] - self.baseline.get(k, 0) for k in current}
                     if current is not None and self.baseline is not None else None)
            if delta is None:
                self.data['limit_exceeded'] = 'proc_io_unreadable'
            elif delta['rchar'] > self.read_cap:
                self.data['limit_exceeded'] = 'proc_rchar_limit'
            if self.data['gpu_peak_allocated_bytes'] > 24 << 30:
                self.data['limit_exceeded'] = 'gpu_peak_allocated_limit'
            value = {**self.data, 'wall_seconds': time.monotonic() - self.started,
                     'proc_self_io_delta': delta, 'pid': os.getpid(),
                     'read_accounting_scope': 'hash full-file bytes + nominal torch.load bytes; proc rchar/read_bytes separate; safetensors mmap/library/image-decode reads not individually instrumented'}
        write_control(self.target, value, replace=True)
        return value

    def monitor(self):
        while not self.stop.wait(.5):
            try:
                self.snapshot()
            except Exception as exc:
                # Parent independently enforces wall/RSS and fails a stale observer.
                self.data['limit_exceeded'] = 'observer_error:' + type(exc).__name__
                break


def run(preparation, stage, attempt, policy, commitments, read_cap, *, compatibility=None, admission=None):
    baseline = proc_io()  # before torch/runtime imports, including their actual syscalls
    sys.path.insert(0, preparation['runtime']['path'])
    provenance = install(compatibility, preparation, stage, policy, admission=admission)
    if provenance is not None:
        write_once(Path(attempt) / 'STORAGE_COMPATIBILITY.json', provenance)
    import torch
    from tools.st_world_policy import spatial_probe as p
    from tools.st_world_policy import checkpoints, training_state, base_binding
    from starVLA.dataloader import st_world_observation, st_world_history
    from starVLA.model.modules.vehicle_joint import initialization
    counter = Counters(Path(attempt) / 'COUNTERS.json', torch, read_cap, baseline)
    originals = (initialization.file_sha256, st_world_history.file_sha256)
    replacements = {f: counter.hash_wrapper(f) for f in originals}
    # Cover existing aliases plus future imports from the two defining modules.
    for module in tuple(sys.modules.values()):
        if module is not None and getattr(module, '__name__', '').startswith(('tools.', 'starVLA.')):
            for name, fn in list(vars(module).items()):
                if any(fn is f for f in originals):
                    setattr(module, name, replacements[fn])
    torch.load = counter.load_wrapper(torch.load)
    checkpoints.load_world_checkpoint = counter.native_wrapper(checkpoints.load_world_checkpoint)
    counter.snapshot()
    thread = threading.Thread(target=counter.monitor, daemon=True); thread.start()
    result = None
    try:
        if stage == 'fit':
            registration = p.pinned(preparation['fit_registration'])
            counter.data['phase'] = 'CPU_FIT_ORIGINAL_300_CYCLES'
            result = p.fit(registration, preparation['fit_output'])
        else:
            split, action = stage.split('_')
            binding = p.pinned(preparation['controls'][split]['binding'])
            root = preparation['caches'][split]['root']
            counter.data['phase'] = action.upper()
            if action == 'plan':
                signature, _, required = p.cache_plan(binding)  # exactly one real call
                if signature['identity'] != preparation['caches'][split]['identity']:
                    raise ValueError('Actual plan differs from metadata expectation')
                write_once(Path(attempt) / 'SIGNATURE.json', signature)
                result = dict(status='PLANNED_ONLY', cache_identity=signature['identity'], **required,
                              no_cache_written=True, runtime_checks_deferred=['cuDNN', 'device'])
            else:
                result = p.extract(binding, root, policy['path'], checkpoint_commitments=commitments)
        write_once(Path(attempt) / 'CLI_RESULT.json', {k:v for k,v in result.items() if k not in ('metrics', 'exposure')})
        return 0 if result['status'] in ('COMPLETE', 'PLANNED_ONLY') else 2
    finally:
        counter.stop.set(); thread.join(timeout=2)
        snapshot = counter.snapshot()
        if snapshot['limit_exceeded']:
            raise RuntimeError(snapshot['limit_exceeded'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--admission', required=True); p.add_argument('--admission-sha256', required=True)
    p.add_argument('--attempt', required=True); p.add_argument('--read-cap', required=True, type=int)
    a = p.parse_args()
    admission, _ = read(a.admission, a.admission_sha256)
    preparation, _ = read(admission['preparation']['path'], admission['preparation']['sha256'])
    # An inherited lock descriptor proves this is an owned supervised child,
    # rather than an accidental direct invocation copied from a command example.
    fd = int(os.environ['ST_PROBE_SUPERVISOR_LOCK_FD']); os.fstat(fd)
    if os.getpgrp() != os.getpid():
        raise ValueError('Dedicated child session required')
    return run(preparation, admission['stage'], a.attempt, admission.get('storage_policy'),
               admission.get('checkpoint_commitments'), a.read_cap,
               compatibility=admission.get('storage_compatibility'), admission=admission)


if __name__ == '__main__':
    raise SystemExit(main())
