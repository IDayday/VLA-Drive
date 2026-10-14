"""Cache only exact FP32 casts of frozen Linear parameters, never activations.

The registered BF16 parameter storage and dtype introspection remain unchanged.
Refresh buffers after every load_state_dict; current autograd actors are rejected.
"""
import types

import torch
from torch import nn
import torch.nn.functional as F


class FrozenLinearCasts:
    def __init__(self,actor):
        if actor.training or any(p.requires_grad for p in actor.parameters()):
            raise ValueError('Read-only casts are restricted to a frozen eval model')
        self.actor=actor
        self.entries=[]
        for name,module in actor.policy.action_model.named_modules():
            if not isinstance(module,nn.Linear):continue
            weight=torch.empty_like(module.weight,dtype=torch.float32)
            bias=torch.empty_like(module.bias,dtype=torch.float32) if module.bias is not None else None
            entry=dict(name=name,module=module,weight=weight,bias=bias,
                       original_forward=module.forward,
                       weight_storage=(id(module.weight),module.weight.data_ptr(),module.weight.dtype),
                       bias_storage=(id(module.bias),module.bias.data_ptr(),module.bias.dtype) if module.bias is not None else None)
            self.entries.append(entry)
            def forward(layer,x,slot=entry):
                if torch.is_grad_enabled() or layer.training or layer.weight.requires_grad:
                    raise ValueError('Frozen cast cache cannot replace an actor autograd path')
                return F.linear(x,slot['weight'],slot['bias'])
            module.forward=types.MethodType(forward,module)
        self.refreshes=0
        self.refresh()
        self.hook=actor.register_load_state_dict_post_hook(self._after_load)

    @torch.no_grad()
    def refresh(self):
        for entry in self.entries:
            module=entry['module']
            if (id(module.weight),module.weight.data_ptr(),module.weight.dtype)!=entry['weight_storage']:
                raise ValueError('Frozen source storage replaced; invalidate graphs and rebuild cast cache')
            if module.bias is not None and (id(module.bias),module.bias.data_ptr(),module.bias.dtype)!=entry['bias_storage']:
                raise ValueError('Frozen bias storage replaced')
            # Exact BF16->FP32 conversion into stable buffers also used by the
            # existing CUDA graph. No BF16 master or actor parameter is changed.
            entry['weight'].copy_(module.weight)
            if module.bias is not None:entry['bias'].copy_(module.bias)
        self.refreshes+=1

    def _after_load(self,module,incompatible):
        if module is not self.actor or incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError('Frozen behavior loading changed parameter closure')
        self.refresh()

    def close(self):
        self.hook.remove()
        for entry in self.entries:entry['module'].forward=entry['original_forward']


def install(actor):
    if hasattr(actor,'_execution_frozen_linear_casts'):
        raise ValueError('Frozen cast adapter already installed')
    cache=FrozenLinearCasts(actor)
    actor._execution_frozen_linear_casts=cache
    return cache
