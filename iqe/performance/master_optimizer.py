"""Unsharded FP32-master AdamW for a BF16-executed Query model.

This isolates precision/master-state effects; it is NOT called ZeRO2. Stable
names and separate master tensors are serialized, with strict name-order checks.
"""
import torch
from ..contracts import require

class MasterAdamW(torch.optim.AdamW):
    def __init__(self,named_parameters,**kwargs):
        self.names=[n for n,p in named_parameters]
        self.execution=[p for n,p in named_parameters]
        self.masters=[torch.nn.Parameter(p.detach().float().clone(),requires_grad=True) for p in self.execution]
        super().__init__(self.masters,**kwargs)

    def initialize_masters(self,model_state):
        with torch.no_grad():
            for n,p,m in zip(self.names,self.execution,self.masters):
                require(model_state[n].shape==m.shape,'master shape mismatch: '+n)
                m.copy_(model_state[n]);p.copy_(m)

    def zero_grad(self,set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)
        for p in self.execution:
            if set_to_none:p.grad=None
            elif p.grad is not None:p.grad.zero_()

    @torch.no_grad()
    def step(self,closure=None):
        require(closure is None,'closure unsupported by master protocol')
        for p,m in zip(self.execution,self.masters):m.grad=None if p.grad is None else p.grad.float()
        result=super().step()
        for p,m in zip(self.execution,self.masters):p.copy_(m)
        return result

    def state_dict(self):
        state=super().state_dict()
        state['master_names']=self.names
        state['master_weights']=[m.detach() for m in self.masters]
        return state

    def load_state_dict(self,state_dict):
        state=dict(state_dict)
        names=state.pop('master_names',None);masters=state.pop('master_weights',None)
        if names is not None:
            require(names==self.names and masters is not None and len(masters)==len(self.masters),'master name/order mismatch')
            with torch.no_grad():
                for m,v,p in zip(self.masters,masters,self.execution):
                    require(v.dtype==torch.float32 and v.shape==m.shape,'master precision/shape mismatch')
                    m.copy_(v);p.copy_(m)
        else:require(masters is None,'master weights lack names')
        super().load_state_dict(state)
        for values in self.state.values():
            for key in ('exp_avg','exp_avg_sq'):
                if key in values:require(values[key].dtype==torch.float32,'Adam moments must be FP32')
