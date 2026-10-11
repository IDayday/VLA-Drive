"""First-update probes, outside the reported steady window. No RNG consumption."""
from pathlib import Path
import hashlib
import torch
from ..io import atomic_torch,atomic_json

class NumericalProbe:
    def __init__(self, model, enabled, root, rank):
        self.enabled=enabled;self.root=Path(root);self.rank=rank;self.active=enabled
        self.values={};self.handles=[];self.before={};self.grad={}
        self.parameters={n:p for n,p in model.named_parameters() if p.requires_grad}
        if not enabled:return
        def keep(name,v):
            if torch.is_tensor(v):self.values[name]=v.detach().cpu().clone()
            elif isinstance(v,(list,tuple)):
                for i,t in enumerate(v):keep(f'{name}/{i}',t)
            elif isinstance(v,dict):
                for k,t in v.items():keep(f'{name}/{k}',t)
            elif hasattr(v,'physical'):
                keep(name+'/raw',v.raw);keep(name+'/physical',v.physical)
        self.keep=keep
        for name,module in [('trajectory',model.action_model.expert)]:
            self.handles.append(module.register_forward_hook(lambda m,a,out,n=name:keep(n,out) if self.active else None))
        original=model.encode_current
        def encoded(*args,**kwargs):
            out=original(*args,**kwargs)
            if self.active:keep('encoded',out)
            return out
        model.encode_current=encoded

    def capture_prepared(self,prepared):
        if self.active:
            if prepared['ready']:torch.cuda.current_stream().wait_event(prepared['ready'])
            value=prepared['pieces'][0][1]
            self.keep('visual',value[4]);self.keep('deepstack',value[5])
            self.values['sequence']={'valid_lengths':value[1].sum(-1).cpu(),'padded_length':torch.tensor(value[0].shape[1]),
                'slot_counts':{k:torch.tensor(v.shape[1]) for k,v in value[3].items()}}

    @staticmethod
    def sample(p):
        v=p.detach().reshape(-1);count=min(4096,len(v))
        # Fixed index arithmetic, never random; evenly spans each named tensor.
        idx=torch.arange(count,device=v.device,dtype=torch.int64)*(max(0,len(v)-1))//max(1,count-1)
        return v[idx].float().cpu().clone()

    def before_step(self,optimizer,result):
        if not self.active:return
        source=dict(zip(optimizer.names,optimizer.masters)) if hasattr(optimizer,'masters') else self.parameters
        self.before={n:self.sample(p) for n,p in source.items()}
        self.grad={n:self.sample(p.grad) for n,p in self.parameters.items() if p.grad is not None}
        self.values['losses']={k:v.detach().float().cpu() for k,v in result['losses'].items()}
        self.values['global_metrics']={k:v.detach().cpu() if torch.is_tensor(v) else v for k,v in result.get('metrics',{}).items()}

    def after_step(self,optimizer):
        if not self.active:return
        source=dict(zip(optimizer.names,optimizer.masters)) if hasattr(optimizer,'masters') else self.parameters
        updates={n:self.sample(p)-self.before[n] for n,p in source.items()}
        atomic_torch(self.root/f'probe_rank{self.rank}.pt',{'values':self.values,'gradient_samples':self.grad,'update_samples':updates,
            'scope':'up to4096 deterministic stratified elements per named trainable parameter; not a full-gradient equivalence proof'})
        self.active=False;self.values.clear();self.before.clear();self.grad.clear()
        for h in self.handles:h.remove()
