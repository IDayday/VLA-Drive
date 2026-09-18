"""Scoped FP32 parameter snapshots for no-grad DiT calls only.

The inherited action kernel casts BF16 parameter values to FP32. These copies
reuse those same values across a rollout/transition evaluation, without changing
candidate layout. Snapshots cannot escape their lexical scope or enter backward.
This does NOT cache policy conditions, gradients or optimizer parameters.
"""
from contextlib import contextmanager
import torch
from torch import nn


class _Velocity(nn.Module):
    def __init__(self, head):
        super().__init__();self.head=head

    def forward(self, x, bucket, condition):
        return self.head.predict_velocity(x,bucket,condition)


@contextmanager
def inference_velocity_snapshot(head):
    if torch.is_grad_enabled():
        raise RuntimeError('inference weight snapshots are forbidden in autograd')
    if head.training:
        raise ValueError('inference snapshots require eval mode')
    wrapper=_Velocity(head)
    tensors=dict(wrapper.named_parameters())|dict(wrapper.named_buffers())
    # Torch2.5 functional_call can leave a substituted tensor behind when two
    # module paths alias the same child. Retain each actual attribute slot and
    # restore the original objects through nn.Module's public setter, including
    # on forward errors. This never copies values into optimizer parameters.
    slots={}
    named=list(wrapper.named_parameters(remove_duplicate=False))+list(wrapper.named_buffers(remove_duplicate=False))
    for name,value in named:
        parent,_,field=name.rpartition('.')
        module=wrapper.get_submodule(parent)
        slots[id(module),field]=(module,field,value)
    if any(x.is_floating_point() and x.dtype not in (torch.bfloat16,torch.float32) for x in tensors.values()):
        raise ValueError('snapshot supports only audited BF16/FP32 action weights')
    identity={n:(id(p),p.data_ptr(),p.dtype) for n,p in tensors.items()}
    state={n:p.detach().to(dtype=torch.float32 if p.is_floating_point() else p.dtype,copy=True)
           for n,p in tensors.items()}
    active=True
    def call(x,bucket,condition):
        if not active or torch.is_grad_enabled():
            raise RuntimeError('snapshot is expired or autograd is enabled')
        try:
            with torch.autocast(x.device.type,dtype=torch.float32,enabled=x.device.type=='cuda'):
                return torch.func.functional_call(wrapper,state,(x,bucket,condition),strict=True).float()
        finally:
            for module,field,original in slots.values():
                if getattr(module,field) is not original:
                    setattr(module,field,original)
    try:
        yield call
    finally:
        active=False;state.clear()
        restored=dict(wrapper.named_parameters())|dict(wrapper.named_buffers())
        if identity != {n:(id(p),p.data_ptr(),p.dtype) for n,p in restored.items()}:
            raise RuntimeError('functional inference did not restore original parameter objects')
