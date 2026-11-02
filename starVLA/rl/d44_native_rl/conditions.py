"""Bounded LRU of frozen vision/embeddings; never cache current text C/P."""
from collections import OrderedDict
import torch
from starVLA.rl.qwen_world_compare.runtime import FrozenCurrentConditions
from starVLA.rl.qwen_world_compare.policy_adapter import DESCRIPTOR_SCHEMA
from starVLA.rl.qwen_world_compare.contracts import fingerprint


class DConditions(FrozenCurrentConditions):
    def __init__(self,model,data,parent,group=16,capacity=8):
        super().__init__(model,data,parent)
        self.group=group;self.capacity=capacity;self.lru=OrderedDict()
        self.identity.update(schema='d44_frozen_visual_input_v1',group_size=group,capacity=capacity)
        self.sha=self.identity_sha256=fingerprint(self.identity)

    def get(self,token,device=None):
        if token not in self.tokens:raise ValueError('Outside legal NAVTRAIN')
        return dict(schema=DESCRIPTOR_SCHEMA,token=token,identity=self.sha,repeats=self.group)

    @torch.no_grad()
    def materialize(self,descriptor,device):
        token=descriptor['token']
        if token in self.lru:self.token=token;self.cached=self.lru.pop(token)
        result=super().materialize(descriptor,device)
        self.lru[token]=self.cached
        while len(self.lru)>self.capacity:self.lru.popitem(last=False)
        result['context']=torch.zeros((self.group,1,1),device=device)
        return result
