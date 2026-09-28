"""Bound ZeRO flat tensors and verify actual FP32-master updates.

The installed DeepSpeed0.16.9 FusedAdam metadata stores tensor sizes in signed
32-bit integers, including in its nominal64-bit functor path. A single ZeRO
partition above INT_MAX can silently keep parameters and moments unchanged.
We do not patch the shared installation: use equal-hyperparameter bounded groups.
"""
import torch

MAX_GROUP_ELEMENTS = 500_000_000


def bounded_parameter_groups(named_parameters, maximum=MAX_GROUP_ELEMENTS):
    if not 0 < maximum < 2**31: raise ValueError('Optimizer group bound must fit signed32-bit indexing')
    groups=[];records=[];current=[];names=[];size=0
    for name, parameter in named_parameters:
        if not parameter.requires_grad:continue
        n=parameter.numel()
        if n>maximum:raise ValueError(f'Individual parameter exceeds the safe optimizer bound: {name}')
        if current and size+n>maximum:
            groups.append({'params':current});records.append({'elements':size,'names':names})
            current=[];names=[];size=0
        current.append(parameter);names.append(name);size+=n
    if current:groups.append({'params':current});records.append({'elements':size,'names':names})
    if not groups:raise ValueError('No trainable parameters')
    return groups,records


def capture_master_samples(optimizer, samples=512):
    captured=[]
    for parameter in optimizer.single_partition_of_fp32_groups:
        if parameter.dtype!=torch.float32 or parameter.numel()>=2**31:
            raise ValueError('Unsafe optimizer partition dtype/extent')
        indices=torch.linspace(0,parameter.numel()-1,min(samples,parameter.numel()),device=parameter.device).long().unique()
        captured.append((indices,parameter.detach()[indices].clone()))
    return captured


def master_update_evidence(optimizer, captured, device):
    changed=torch.zeros((),device=device,dtype=torch.int64)
    maximum=torch.zeros((),device=device,dtype=torch.float32)
    for parameter,(indices,before) in zip(optimizer.single_partition_of_fp32_groups,captured):
        after=parameter.detach()[indices]
        delta=(after-before).abs().to(device)
        if not torch.isfinite(delta).all():raise FloatingPointError('Nonfinite optimizer master update')
        changed+=(delta!=0).sum();maximum=torch.maximum(maximum,delta.max())
    import torch.distributed as dist
    if dist.is_initialized():
        dist.all_reduce(changed,op=dist.ReduceOp.SUM);dist.all_reduce(maximum,op=dist.ReduceOp.MAX)
    if not changed.item():raise RuntimeError('Optimizer step changed no sampled FP32 master parameter; stop before counting progress')
    return {'sampled_changed_elements':int(changed),'maximum_sampled_change':float(maximum),
            'fp32_master_update_verified':True}
