"""Typed cache-only current inputs; original target loader and online path kept.

Only the frozen visual boundary is replaced. Token lookup, learned history and
foresight substitution, language, Query and auxiliary graphs run every update.
"""
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import threading
import time
import types
import numpy as np
import torch
from ..contracts import require
from ..io import digest
from ..performance.vision_cache import combine_payloads
from ..training.execution import _record_stream

_CURRENT_READER = ContextVar('iqe_current_cache_reader', default=None)
_INSTALL_LOCK = threading.Lock()


@dataclass(frozen=True)
class CacheOnlyCurrentInput:
    scene_id: str
    observation_hash: str
    key: str
    lang: str
    state: np.ndarray
    access_identity: str

    def __getitem__(self, key):
        require(key in {'token', 'lang', 'state'}, 'cache-only inputs expose only legal current fields')
        return self.scene_id if key == 'token' else getattr(self, key)


def current_numeric_fields(record):
    """Same arithmetic/order as locked foresight_dataset.current_observation."""
    from starVLA.dataloader.foresight_dataset import encode_ego
    poses = np.asarray(record['global_pose_history'], dtype=np.float64)
    require(poses.shape == (4, 3) and np.isfinite(poses).all(), 'invalid current history')
    prev, now = poses[2:4]; c, s = np.cos(now[2]), np.sin(now[2])
    xy = (now[:2]-prev[:2]) @ np.array([[c, -s], [s, c]])
    yaw = (now[2]-prev[2]+np.pi) % (2*np.pi)-np.pi
    state = encode_ego(np.array([[*xy, yaw]]))
    command = record['navigation']; require(command in (0, 1, 2, 3), 'invalid navigation')
    text = ('turn left', 'keep straight', 'turn right', 'unknown')[command]
    lang = f'You are an autonomous driving agent. The navigation command for the current timestep is {text}. Your task is to plan future actions based on the understanding of the driving scene.'
    return state, lang


def current_identity(record):
    # Strictly current fields: excludes ego targets/GT/metric contexts.
    names = ('token', 'identity', 'image_paths', 'global_pose_history', 'navigation')
    return digest({name: record[name] for name in names})


def cache_observation(record, reader):
    row = reader.access(record['token'])
    require(current_identity(record) == row['observation_hash'], 'current observation/context changed')
    require(len(record['image_paths']) == 3, 'exact current three-view order required')
    state, lang = current_numeric_fields(record)
    require(digest(lang) == row['lang_hash'], 'current instruction changed')
    return CacheOnlyCurrentInput(record['token'], row['observation_hash'], row['key'], lang,
                                 state, reader.complete['index_sha256'])


def install_current_dispatch():
    """Thread-local replacement at the existing current-only loader boundary.

    Source files and other dataset instances are untouched. Target loading stays
    in its original inheritance chain with all original checks. Online datasets
    call the original image function whenever no cache context is active.
    """
    from starVLA.dataloader import foresight_dataset as source
    with _INSTALL_LOCK:
        if getattr(source.current_observation, '_iqe_cache_dispatch', False): return
        original = source.current_observation
        def dispatch(record, image_root=None):
            reader = _CURRENT_READER.get()
            if reader is None: return original(record, image_root)
            require(image_root is None, 'unrepresented local image transform/staging')
            return cache_observation(record, reader)
        dispatch._iqe_cache_dispatch = True
        dispatch._online_original = original
        source.current_observation = dispatch


class CacheOnlyDataset:
    def __init__(self, original, reader):
        install_current_dispatch()
        self.original, self.reader, self.index = original, reader, original.index

    def __len__(self): return len(self.original)

    def __getitem__(self, index):
        context = _CURRENT_READER.set(self.reader)
        try:
            observation, target = self.original[index]
            require(isinstance(observation, CacheOnlyCurrentInput), 'source did not consume cache-only current boundary')
            return observation, target
        finally: _CURRENT_READER.reset(context)


def input_signature(inputs):
    return tuple((v.scene_id, v.observation_hash, v.key, v.access_identity, digest(v.lang)) for v in inputs)


class CachedVisionAhead:
    def __init__(self, model, reader, *, read_workers=4):
        self.model, self.reader = model, reader
        self.device = next(model.parameters()).device
        self.original_encode = model.encode_current
        require(1 <= read_workers <= 4, 'bounded per-rank read concurrency')
        self.read_pool = ThreadPoolExecutor(max_workers=read_workers)
        self.active = None
        self.stream = torch.cuda.Stream(device=self.device) if self.device.type == 'cuda' else None
        if self.stream: self.stream.wait_stream(torch.cuda.current_stream(self.device))
        self.calls = {'cached_encode': 0, 'online_encode': 0, 'visual_calls_on_hit': 0}
        self.visual = model.qwen_vl_interface.model.model.visual
        require(not self.visual.training and all(not p.requires_grad for p in self.visual.parameters()), 'visual must stay frozen eval')
        model.encode_current = self.encode_current

    def prepare(self, observations, micro_batch):
        require(all(isinstance(x, CacheOnlyCurrentInput) for x in observations), 'typed cache-only current inputs required')
        began = time.perf_counter(); pieces = []
        for at in range(0, len(observations), micro_batch):
            part = observations[at:at+micro_batch]
            for x in part:
                row = self.reader.access(x.scene_id)
                require(x.access_identity == self.reader.complete['index_sha256'] and x.key == row['key'] and
                        x.observation_hash == row['observation_hash'], 'untrusted cached input')
                require(digest(x.state.tolist()) == row['state_hash'] and digest(x.lang) == row['lang_hash'],
                        'cached current numeric input/instruction changed')
            payloads = list(self.read_pool.map(self.reader.read, [x.scene_id for x in part]))
            values = combine_payloads(payloads, self.model.qwen_vl_interface.processor.tokenizer.pad_token_id,
                                      'cpu', self.model._special_token_ids)
            pieces.append((input_signature(part), values))
        io_seconds = time.perf_counter()-began
        def transfer(value):
            if isinstance(value, torch.Tensor):
                return value.pin_memory().to(self.device, non_blocking=True) if self.stream else value
            if isinstance(value, dict): return {n: transfer(v) for n, v in value.items()}
            if isinstance(value, list): return [transfer(v) for v in value]
            if isinstance(value, tuple): return tuple(transfer(v) for v in value)
            return value
        if self.stream:
            with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
                pieces = [(signature, transfer(value)) for signature, value in pieces]
                ready = torch.cuda.Event(); ready.record(self.stream)
        else: ready = None
        return {'pieces': deque(pieces), 'ready': ready, 'prepare_wall_seconds': time.perf_counter()-began,
                'cache_io_seconds': io_seconds}

    @contextmanager
    def activate(self, prepared):
        require(self.active is None, 'overlapping cached consumers')
        if prepared['ready']:
            stream = torch.cuda.current_stream(self.device); stream.wait_event(prepared['ready'])
            for _, value in prepared['pieces']: _record_stream(value, stream)
        self.active = prepared['pieces']
        try:
            yield
            require(not self.active, 'cache batch not consumed completely')
        finally: self.active = None

    def encode_current(self, observations):
        if not any(isinstance(x, CacheOnlyCurrentInput) for x in observations):
            self.calls['online_encode'] += 1
            return self.original_encode(observations)
        require(all(type(x) is CacheOnlyCurrentInput for x in observations), 'mixed cached/online inputs forbidden')
        require(self.active is not None and self.active, 'missing prepared cache payload')
        signature, value = self.active.popleft()
        require(signature == input_signature(observations), 'cache input/order/prompt mismatch')
        ids, attention, positions, slots, visual, deepstack = value
        self.model._diagnostic_world_positions = slots.get('foresight')
        self.model.last_sequence_lengths = attention.sum(-1).detach()
        rows = torch.arange(len(observations), device=ids.device)
        # This is the exact post-_build_qwen_batch portion of the locked source.
        # It retains trainable lookup/replacements/LLM and Query ego state.
        from starVLA.model.modules.foresight.tokens import replace_query_embeddings
        with self.model.amp():
            embeddings = self.model.qwen_vl_interface.model.get_input_embeddings()(ids)
            states = torch.as_tensor(np.asarray([e.state for e in observations]), device=ids.device, dtype=torch.float32)
            require(states.shape == (len(observations), 1, 4) and bool(torch.isfinite(states).all()), 'invalid allowed ego state')
            embeddings = replace_query_embeddings(embeddings, slots['history'], self.model.action_input_model(states))
            if self.model.foresight_tokens:
                require(bool((slots['history'][:, -1] < slots['foresight'][:, 0]).all()) and
                        bool((slots['foresight'][:, -1] < slots['action'][:, 0]).all()), 'causal token order changed')
                embeddings = replace_query_embeddings(embeddings, slots['foresight'], self.model.query_embeddings())
            hidden = self.model._qwen_language_forward(ids, embeddings, attention, positions, visual, deepstack)
        self.model.action_model.ego_state = states
        self.calls['cached_encode'] += 1
        return {'W': hidden[rows[:, None], slots['foresight']] if self.model.foresight_tokens else hidden[:, :0],
                'action_queries': hidden[rows[:, None], slots['action']]}

    def close(self):
        if self.stream: self.stream.synchronize()
        self.read_pool.shutdown(wait=True, cancel_futures=True)
        self.model.encode_current = self.original_encode
        self.reader.close()
