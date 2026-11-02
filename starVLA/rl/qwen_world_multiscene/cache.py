"""Immutable frozen-Qwen/reference cache: no device-to-host copy on a hit."""
from collections import OrderedDict
import torch
from starVLA.rl.simwam_recipe import checkpoint


def storage_bytes(tree):
    seen={}
    def visit(x):
        if torch.is_tensor(x):
            s=x.untyped_storage();seen[(str(x.device),s.data_ptr())]=s.nbytes()
        elif isinstance(x,dict):
            for v in x.values():visit(v)
        elif isinstance(x,(tuple,list)):
            for v in x:visit(v)
    visit(tree)
    return sum(seen.values())


class ImmutableFeatureCache:
    def __init__(self,base,capacity=32):
        self.base,self.capacity=base,capacity
        self.device_cache=OrderedDict();self.host_cache=OrderedDict()
        self.hits=self.misses=self.host_restores=self.host_backups=0

    def __getattr__(self,key):return getattr(self.base,key)

    def materialize(self,descriptor,device):
        token=descriptor['token']
        if token in self.device_cache:
            value=self.device_cache.pop(token);self.device_cache[token]=value
            self.base.token,self.base.cached=token,value;self.hits+=1
        elif token in self.host_cache:
            self.base.token=token;self.base.cached=checkpoint.device_tree(self.host_cache[token],device)
            self.device_cache[token]=self.base.cached;self.host_restores+=1
        else:
            self.base.materialize(descriptor,device);self.misses+=1
            self.device_cache[token]=self.base.cached
            #Exactly one backup per frozen feature miss, never per inner/hit.
            self.host_cache[token]=checkpoint.cpu_tree(self.base.cached);self.host_backups+=1
        while len(self.device_cache)>self.capacity:self.device_cache.popitem(last=False)
        while len(self.host_cache)>self.capacity:self.host_cache.popitem(last=False)
        return self.base.materialize(descriptor,device)

    def statistics(self):
        return dict(hits=self.hits,misses=self.misses,host_restores=self.host_restores,host_backups=self.host_backups,
            device_unique_storage_bytes=storage_bytes(self.device_cache),host_unique_storage_bytes=storage_bytes(self.host_cache),
            entries=len(self.device_cache),contract='frozen Qwen and frozen IL memory only; current trainable Memory/KV graphs are never cached')
