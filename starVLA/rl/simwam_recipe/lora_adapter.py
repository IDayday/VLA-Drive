"""Exact-path mapping onto the existing, unchanged DDP attention topology."""
import torch
from .vendor.lora import LoRALinear, merged_state_dict


def attention_paths(head, targets):
    if list(targets)!=['q','k','v','o']:raise ValueError('Unregistered attention target scope')
    paths=[]
    for i,block in enumerate(head.model.transformer_blocks):
        attn=block.attn1
        if getattr(attn,'fused_projections',False):raise ValueError('Fused projections bypass source LoRA wrappers')
        for suffix in ('to_q','to_k','to_v','to_out.0'):
            path=f'model.transformer_blocks.{i}.attn1.{suffix}'
            if not isinstance(head.get_submodule(path),torch.nn.Linear):raise ValueError('Expected unfused Linear: '+path)
            paths.append(path)
    if len(paths)!=96:raise ValueError('Expected four used projections in each original 24 blocks')
    return paths


def parameter_contract(actor):
    adapters={n:p for n,p in actor.named_parameters() if p.requires_grad}
    if not adapters or any(not n.startswith('action_expert.model.transformer_blocks.') or
                           not n.endswith(('.lora_A','.lora_B')) for n in adapters):
        raise ValueError('Only original action attention LoRA A/B may train')
    if len(adapters)!=192:raise ValueError('Unexpected LoRA parameter tensor closure')
    if any(p.requires_grad for p in actor.ref_action_expert.parameters()):raise ValueError('Reference is not frozen')
    info=[]
    for name,m in actor.action_expert.named_modules():
        if isinstance(m,LoRALinear):
            index=int(name.split('.')[2]); block=actor.action_expert.model.transformer_blocks[index]
            info.append(dict(path='action_model.'+name,attention='self' if block.cross_attention_dim is None else 'cross',
                             in_features=m.base.in_features,out_features=m.base.out_features,r=m.r,alpha=m.scaling*m.r,
                             scaling=m.scaling,dtype=str(m.lora_A.dtype),parameters=m.lora_A.numel()+m.lora_B.numel(),
                             processor=type(block.attn1.processor).__name__))
    return dict(adapters=info,trainable_names=sorted(adapters),trainable_numel=sum(p.numel() for p in adapters.values()),
                original_action_tensors=359,original_action_numel=819503620,
                qwen_proj_frozen=not any(p.requires_grad for p in actor.action_expert.qwen_proj.parameters()))


def adapter_state(actor):
    return {n:p.detach().cpu().clone() for n,p in actor.named_parameters() if p.requires_grad}


def restore_adapters(actor, state):
    params={n:p for n,p in actor.named_parameters() if p.requires_grad}
    if set(params)!=set(state):raise ValueError('Changed adapter closure on resume')
    with torch.no_grad():
        for n,p in params.items():p.copy_(state[n].to(p))
