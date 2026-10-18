"""Frozen prefill once per fresh scene; dense 30-layer KV is not written per chain."""
import time
import torch
from starVLA.model.modules.qwen_sourceflow.qwen_encoder import QwenObservationProcessor
from starVLA.model.modules.qwen_sourceflow.cached_action_runner import expand_condition
from .contracts import QWEN_PATH,fingerprint


class FrozenConditions:
    def __init__(self, model, data, parent_identity):
        self.qwen=model.qwen.eval().requires_grad_(False)
        self.provider=model.provider.eval().requires_grad_(False)
        self.data=data;self.processor=QwenObservationProcessor(QWEN_PATH)
        self.identity=dict(schema='native_frozen_full_Qwen_condition_v1',parent=parent_identity,
            data_manifest_sha256=data.sha,precision='FP32 weights; BF16 autocast prefill/provider',
            layer_mapping=self.provider.layer_mapping,cache='process-local one scene; no persistent dense KV')
        self.sha=fingerprint(self.identity)
        self.cached=None;self.token=None;self.prefill_calls=0;self.prefill_seconds=0.

    def get(self, token, device):
        if token not in self.data.records:raise ValueError('Observation outside registered TRAIN')
        return dict(schema='native_frozen_observation_v1',token=token,identity=self.sha,repeats=8)

    @torch.no_grad()
    def materialize(self, descriptor, device):
        if descriptor!=self.get(descriptor['token'],device):raise ValueError('Frozen condition identity changed')
        if descriptor['token']!=self.token:
            self.cached=None
            begin=time.monotonic();obs=self.data.observation(descriptor['token'])
            batch=self.processor([obs]);state=torch.tensor([obs['proprio']],device=device,dtype=torch.float32)
            with torch.autocast(device.type,dtype=torch.bfloat16):
                value=self.provider(self.qwen(batch),state)
            self.cached=expand_condition(value,8);self.token=descriptor['token'];self.prefill_calls+=1
            torch.cuda.synchronize();self.prefill_seconds+=time.monotonic()-begin
        return self.cached
