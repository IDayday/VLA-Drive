"""Attach only missing current-branch modules; never reload/randomize a policy."""
import hashlib
import torch
from torch import nn
from starVLA.model.modules.qwen_world_compare.heads import EvaluationReadout
from .protocol import PROTOCOL_SHA256


class MinimalRewardReadouts(nn.Module):
    def __init__(self,reasoning_dim):
        super().__init__()
        self.bridge=nn.Sequential(nn.LayerNorm(reasoning_dim),nn.Linear(reasoning_dim,512))
        self.evaluation=EvaluationReadout(512)


def tensor_sha(value):
    x=value.detach().cpu().contiguous()
    return hashlib.sha256(str((str(x.dtype),tuple(x.shape))).encode()+x.numpy().tobytes()).hexdigest()


def attach_current_only(model,*,seed=42):
    if (model.architecture_id!='D'or hasattr(model,'readouts')or model.decision.r_embeddings is not None
            or model.train_heads is not False or seed!=42):
        raise ValueError('Fresh auxiliaries require the exact already-pruned D46 current policy')
    if torch.get_default_dtype()!=torch.float32 or any(p.dtype!=torch.float32 for p in model.parameters()):
        raise ValueError('Fresh auxiliary initialization requires the registered FP32 parameter protocol')
    device=model.decision.action_input.weight.device;width=model.decision.hidden_dim
    before={k:id(v)for k,v in model.named_parameters()}
    # torch.manual_seed would also touch CUDA generators. This explicitly uses
    # only the CPU default generator that fork_rng(devices=[]) saves/restores.
    with torch.random.fork_rng(devices=[]),torch.device('cpu'):
        torch.random.default_generator.manual_seed(seed)
        queries=torch.randn((16,width),device='cpu',dtype=torch.float32)*.02
        heads=MinimalRewardReadouts(width).float()
    model.decision.r_embeddings=nn.Parameter(queries.to(device),requires_grad=True)
    model.readouts=heads.to(device).eval().requires_grad_(True);model.train_heads=True
    current=dict(model.named_parameters())
    if any(name not in current or id(current[name])!=old for name,old in before.items()):
        raise ValueError('Adding auxiliary modules replaced an existing policy parameter')
    new={k:v for k,v in current.items()if k not in before}
    expected={'decision.r_embeddings'}|{'readouts.'+name for name,_ in heads.named_parameters()}
    if set(new)!=expected:raise ValueError('Fresh modules have an unexpected parameter closure')
    return dict(schema='d46_fresh_auxiliary_initial_tensor_manifest_v1',protocol_sha256=PROTOCOL_SHA256,
        seed=seed,new_parameter_names=sorted(new),new_parameter_numel=sum(p.numel()for p in new.values()),
        initial_tensor_sha256={name:tensor_sha(p)for name,p in sorted(new.items())},
        original_policy_parameter_objects_preserved=True,original_IL_auxiliary_restored=False)
