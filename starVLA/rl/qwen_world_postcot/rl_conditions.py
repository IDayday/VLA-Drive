"""Frozen current student memory, shared across G8 without persisting policy KV."""
import time

import torch

from starVLA.model.modules.qwen_sourceflow.qwen_encoder import QwenObservationProcessor
from starVLA.rl.qwen_sourceflow.contracts import QWEN_PATH
from .contracts import fingerprint
from .rl_actor import OBSERVATION_SCHEMA, RAW_CONTEXT_SCHEMA


class FrozenWorldConditions:
    def __init__(self, model, data, parent_identity, *, processor=None, group_size=8):
        if group_size != 8:
            raise ValueError('The registered canonical RL group size is8')
        # Only current-observation modules are ever executed below. train-only
        # readouts/teachers are absent from the strict deployment model.
        self.model = model.eval()
        self.qwen = model.qwen.eval().requires_grad_(False)
        self.scene_provider = model.scene_provider.eval().requires_grad_(False)
        self.predictive_provider = model.predictive_provider.eval().requires_grad_(False)
        self.data = data
        self._tokens = frozenset(data.tokens)
        self.processor = processor if processor is not None else QwenObservationProcessor(QWEN_PATH)
        self.group_size = group_size
        self.identity = dict(schema='world_postcot_frozen_current_memory_v1',
            parent=parent_identity, data_manifest_sha256=data.sha,
            precision='FP32 frozen weights and actual FP32 residual memory; BF16 linear/attention compute and projected KV; no memory recast',
            cache='one compact current memory per rank; policy KV rebuilt each sampler/logprob call',
            group_size=group_size, prediction_type='x0', teachers_or_future_or_post_inputs=False)
        self.sha = fingerprint(self.identity)
        self.cached = self.token = None
        self.prefill_calls = 0
        self.prefill_seconds = 0.0
        self.observed_memory_dtype = None

    def get(self, token, device):
        if token not in self._tokens:
            raise ValueError('Observation is outside the locked RL TRAIN role')
        return dict(schema=OBSERVATION_SCHEMA, token=token, identity=self.sha, repeats=self.group_size)

    @torch.no_grad()
    def materialize(self, descriptor, device):
        if descriptor != self.get(descriptor['token'], device):
            raise ValueError('Frozen observation/parent/data identity changed')
        if self.token != descriptor['token']:
            self.cached = None
            started = time.monotonic()
            observation = self.data.observation(descriptor['token'])
            batch = self.processor([observation])
            ego = torch.tensor([observation['proprio']], device=device, dtype=torch.float32)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                memory = self.model.encode_condition(batch, ego)
            current = memory.actor_memory.detach()
            valid = memory.actor_valid.detach()
            if current.shape[0] != 1 or current.shape[1] != 192 or valid.shape != current.shape[:2]:
                raise ValueError('Frozen full-framework memory must contain144+16+32 current-derived tokens')
            if current.dtype not in (torch.bfloat16,torch.float32):
                raise ValueError('Frozen current memory has an unqualified actual provider dtype')
            if self.observed_memory_dtype is not None and self.observed_memory_dtype!=str(current.dtype):
                raise ValueError('Current provider residual-memory dtype changed within this runtime')
            self.observed_memory_dtype=str(current.dtype)
            self.cached = dict(base_memory=current, base_valid=valid, base_ego=ego.detach())
            self.token = descriptor['token']
            self.prefill_calls += 1
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            self.prefill_seconds += time.monotonic() - started
        value = self.cached
        # Expanded views preserve a compactB1 cache. The actor projects KV forB1
        # and expands those values only inside the current call's graph.
        return dict(schema=RAW_CONTEXT_SCHEMA, **value,
            context=value['base_memory'].expand(self.group_size, -1, -1),
            valid=value['base_valid'].expand(self.group_size, -1),
            ego=value['base_ego'].expand(self.group_size, -1))


FrozenConditions = FrozenWorldConditions
