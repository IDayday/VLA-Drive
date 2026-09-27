"""Small common public-language adaptation; original pretrained tensors stay frozen."""
import math
import torch
from torch import nn


class LowRankDelta(nn.Module):
    def __init__(self,in_features,out_features,rank,alpha):
        super().__init__();self.scale=alpha/rank
        self.down=nn.Linear(in_features,rank,bias=False);self.up=nn.Linear(rank,out_features,bias=False)
        nn.init.kaiming_uniform_(self.down.weight,a=math.sqrt(5));nn.init.zeros_(self.up.weight)
    def forward(self,x):return self.up(self.down(x))*self.scale


class AdaptedLinear(nn.Module):
    def __init__(self,original,adapter):super().__init__();self.original=original;self.adapter=adapter
    def forward(self,x):
        y=self.original(x)
        return y+self.adapter(x.to(self.adapter.down.weight.dtype)).to(y.dtype)


def install_language_adapters(model,rank=8,alpha=16,seed=3042):
    if not 1<=rank<=32:raise ValueError('Only bounded low-rank adaptation is supported')
    adapters=nn.ModuleDict();paths=[]
    for name,module in list(model.model.language_model.named_modules()):
        if isinstance(module,nn.Linear) and name.endswith(('.self_attn.q_proj','.self_attn.v_proj')):
            parent_name,leaf=name.rsplit('.',1);parent=model.model.language_model.get_submodule(parent_name)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed+len(paths));delta=LowRankDelta(module.in_features,module.out_features,rank,alpha)
            delta=delta.to(device=module.weight.device,dtype=torch.float32)
            key=name.replace('.','__');adapters[key]=delta;setattr(parent,leaf,AdaptedLinear(module,delta));paths.append(name)
    if not paths:raise ValueError('Qwen language attention projections not found')
    return adapters,{'rank':rank,'alpha':alpha,'seed':seed,'paths':paths,'trainable_parameters':sum(p.numel() for p in adapters.parameters()),
                     'original_pretrained_parameters_frozen':True,'vision_frozen':True,'scope':'Common foundation adaptation shared by all graph/control variants; not full paper reproduction'}
