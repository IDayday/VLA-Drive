"""Identity-bound delta checkpoint; never silently skip a teacher/base parameter."""
import torch
from torch import nn


class TrainableDelta(nn.Module):
    def __init__(self, model, names):
        super().__init__()
        self.model=model;self.names=tuple(names)
        if not self.names or len(set(self.names))!=len(self.names):raise ValueError("Empty/duplicate parameter inventory")
        if not set(self.names)<=dict(model.named_parameters()).keys():raise ValueError("Unknown trainable parameter")

    def state_dict(self,*args,**kwargs):
        parameters=dict(self.model.named_parameters())
        return {name:parameters[name].detach().clone() for name in self.names}

    def load_state_dict(self,state,strict=True):
        if not strict or set(state)!=set(self.names):raise ValueError("Delta parameter inventory mismatch")
        parameters=dict(self.model.named_parameters())
        with torch.no_grad():
            for name,value in state.items():
                if value.shape!=parameters[name].shape:raise ValueError("Delta shape mismatch: "+name)
                parameters[name].copy_(value)
        return torch.nn.modules.module._IncompatibleKeys([],[])
