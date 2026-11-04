"""Reuse the native BF16 casts of immutable FP32 Linear weights.

    FP32 source parameters and every trainable LoRA parameter remain intact.
    Only the read-only value normally passed to BF16 F.linear is memoized.
"""
from functools import wraps
import torch
from torch import nn
from torch.nn import functional as F


class FrozenLinearCache:
    def __init__(self,module):
        self.entries={};self.patches=[];self.hits=0;self.misses=0;self.bytes=0
        for child in module.modules():
            if type(child)is not nn.Linear:continue
            if child.weight.requires_grad or(child.bias is not None and child.bias.requires_grad):continue
            original=child.forward
            @wraps(original)
            def forward(x,*,layer=child,fallback=original):
                device=x.device.type
                if (not torch.is_autocast_enabled(device)or torch.get_autocast_dtype(device)!=torch.bfloat16
                        or layer.weight.requires_grad or(layer.bias is not None and layer.bias.requires_grad)
                        or layer.weight.dtype!=torch.float32):return fallback(x)
                return F.linear(x,self.cast(layer.weight),self.cast(layer.bias)if layer.bias is not None else None)
            existed='forward'in child.__dict__;previous=child.__dict__.get('forward')
            object.__setattr__(child,'forward',forward);self.patches.append((child,forward,existed,previous))
    def cast(self,parameter):
        if parameter.requires_grad:raise ValueError('Never cache trainable parameter casts')
        key=(id(parameter),parameter._version,parameter.data_ptr(),parameter.device,parameter.dtype,
            tuple(parameter.shape),tuple(parameter.stride()),parameter.storage_offset())
        value=self.entries.get(id(parameter))
        if value is not None and value[0]==key:
            self.hits+=1;return value[2]
        if value is not None:self.bytes-=value[2].numel()*value[2].element_size()
        with torch.no_grad():converted=parameter.detach().to(dtype=torch.bfloat16)
        self.entries[id(parameter)]=(key,parameter,converted);self.misses+=1;self.bytes+=converted.numel()*converted.element_size()
        return converted
    def close(self):
        for module,wrapper,existed,previous in reversed(self.patches):
            if module.__dict__.get('forward')is not wrapper:raise RuntimeError('Frozen Linear hook replaced externally')
            if existed:object.__setattr__(module,'forward',previous)
            else:object.__delattr__(module,'forward')
        self.patches.clear();self.entries.clear();self.bytes=0
    def __enter__(self):return self
    def __exit__(self,*_):self.close()
