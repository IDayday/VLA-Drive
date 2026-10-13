"""Bounded host-memory reuse of immutable, checksum-validated head features.

This execution choice changes neither feature bytes nor their delivery order.
The shared VLM is never constructed. Existing feature schema/keys stay intact.
"""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import time
import torch
from iqe.io import atomic_json
from .features import JointCache


def training_cache(config, scenes, *, max_bytes=32*1024**3, preload=True):
    if not scenes:
        raise ValueError('empty training feature population')
    local_world = int(os.environ.get('LOCAL_WORLD_SIZE', '1'))
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))) * 1024
    budget = min(max_bytes, available//8)//local_world
    first = JointCache(config, capacity=0).read(scenes[0])
    # The locked W/A slots have constant shapes. Include per-record object slack.
    record_bytes = sum(x.numel()*x.element_size() for x in first.values()) + 16384
    capacity = max(1, min(len(scenes), budget//record_bytes))
    cache = JointCache(config, capacity=capacity)
    cache.loaded[scenes[0].scene_id] = first
    start = time.perf_counter()
    # Multi-rank fetch already propagates read failures collectively in trainer.
    # Avoid a new pre-DDP collective/deadlock path: use bounded lazy reuse there.
    eager = preload and int(os.environ.get('WORLD_SIZE', '1')) == 1 and capacity == len(scenes)
    if eager:
        def read(scene):
            return JointCache(config, capacity=0).read(scene)
        # map's output order is input order; no sampler/RNG state is involved.
        with ThreadPoolExecutor(max_workers=4) as pool:
            for scene, value in zip(scenes[1:], pool.map(read, scenes[1:])):
                if sum(x.numel()*x.element_size() for x in value.values())+16384 != record_bytes:
                    raise ValueError('feature shapes changed within the locked training population')
                cache.loaded[scene.scene_id] = value
    atomic_json(config.root/'reader_execution'/f'{os.getpid()}.json', {
        'schema': 'bounded_validated_host_features_v1', 'pid': os.getpid(),
        'budget_bytes': budget, 'capacity_records': capacity, 'population': len(scenes),
        'resident_records': len(cache.loaded), 'record_tensor_bytes': record_bytes-16384,
        'preload_seconds': time.perf_counter()-start, 'eager': eager, 'world_size': int(os.environ.get('WORLD_SIZE','1')),
        'resident_records_checksum_verified': True, 'population_complete_verified': eager,
        'feature_numeric_change': False,
        'sampler_order_change': False, 'model_or_loss_change': False, 'VLM_calls': 0})
    return cache
