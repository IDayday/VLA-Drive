"""Frozen Qwen features, permanently frozen IL memory, trainable current memory.

Only compact descriptors are checkpointed. No trainable representation or graph
is cached across updates. Re-encoding after resume reads the same legal images.
"""
import time
import torch

from .contracts import fingerprint
from .rl_actor import RAW_CONTEXT_SCHEMA
from .rl_conditions import FrozenWorldConditions
from .scope_contracts import MEMORY_OBSERVATION


def detached_tree(value):
    if torch.is_tensor(value):return value.detach()
    if isinstance(value,dict):return {k:detached_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [detached_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(detached_tree(v) for v in value)
    return value


class MemoryLearningConditions(FrozenWorldConditions):
    def __init__(self,model,data,parent_identity,actor,*,processor=None):
        # Do not invoke FrozenWorldConditions.__init__: that would freeze the
        # very same current providers registered in the actor optimizer.
        from starVLA.model.modules.qwen_sourceflow.qwen_encoder import QwenObservationProcessor
        from starVLA.rl.qwen_sourceflow.contracts import QWEN_PATH
        self.qwen=model.qwen.eval().requires_grad_(False)
        self.reference_scene=actor.ref_scene_provider
        self.reference_predictive=actor.ref_predictive_provider
        self.data=data;self._tokens=frozenset(data.tokens)
        self.processor=processor if processor is not None else QwenObservationProcessor(QWEN_PATH)
        self.group_size=8
        self.identity=dict(schema='world_postcot_frozen_Qwen_features_trainable_memory_v1',
            parent=parent_identity,data_manifest_sha256=data.sha,group_size=8,
            cache='frozen Qwen selected8/16/28, merger and text; frozen IL reference memory only',
            current_memory='registered actor parameters; rebuild in each differentiable probability invocation',
            prediction_type='x0',teachers_or_future_or_post_inputs=False)
        self.sha=fingerprint(self.identity)
        self.cached=self.token=None;self.prefill_calls=0;self.prefill_seconds=0.
        self.observed_memory_dtype=None

    def get(self,token,device):
        if token not in self._tokens:raise ValueError('Observation is outside locked RL TRAIN')
        return dict(schema=MEMORY_OBSERVATION,token=token,identity=self.sha,repeats=8)

    @torch.no_grad()
    def materialize(self,descriptor,device):
        if descriptor!=self.get(descriptor['token'],device):
            raise ValueError('Frozen Qwen/parent/data descriptor changed')
        if self.token!=descriptor['token']:
            self.cached=None;started=time.monotonic()
            observation=self.data.observation(descriptor['token'])
            batch=self.processor([observation])
            ego=torch.tensor([observation['proprio']],device=device,dtype=torch.float32)
            with torch.autocast(device.type,dtype=torch.bfloat16):
                features=detached_tree(self.qwen(batch))
                reference=self.reference_predictive(self.reference_scene(features,ego))
            current=reference.actor_memory.detach();valid=reference.actor_valid.detach()
            if current.shape[0]!=1 or current.shape[1]!=192 or valid.shape!=current.shape[:2]:
                raise ValueError('Frozen IL reference memory must preserve144+16+32 tokens')
            self.observed_memory_dtype=str(current.dtype)
            self.cached=dict(frozen_features=features,base_memory=current,
                base_valid=valid,base_ego=ego.detach())
            self.token=descriptor['token'];self.prefill_calls+=1
            if device.type=='cuda':torch.cuda.synchronize(device)
            self.prefill_seconds+=time.monotonic()-started
        value=self.cached
        # context is the genuine frozen IL memory, also a source-sampler batch
        # carrier. The current actor always rebuilds its OWN memory from features.
        return dict(schema=RAW_CONTEXT_SCHEMA,**value,
            context=value['base_memory'].expand(8,-1,-1),
            valid=value['base_valid'].expand(8,-1),ego=value['base_ego'].expand(8,-1))
