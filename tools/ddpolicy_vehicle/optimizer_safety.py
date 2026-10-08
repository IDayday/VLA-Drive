"""Bound ZeRO flat tensors and verify actual FP32-master updates.

The installed DeepSpeed0.16.9 FusedAdam metadata stores tensor sizes in signed
32-bit integers, including in its nominal64-bit functor path. A single ZeRO
partition above INT_MAX can silently keep parameters and moments unchanged.
We do not patch the shared installation: use equal-hyperparameter bounded groups.
"""
import math
import torch

MAX_GROUP_ELEMENTS = 500_000_000


def bounded_parameter_groups(named_parameters, maximum=MAX_GROUP_ELEMENTS, learning_rates=None):
    if not 0 < maximum < 2**31: raise ValueError('Optimizer group bound must fit signed32-bit indexing')
    groups=[];records=[];current=[];names=[];size=0;current_lr=None
    def append():
        group={'params':current};record={'elements':size,'names':names}
        if learning_rates is not None:
            group['lr']=current_lr;record['base_lr']=current_lr
        groups.append(group);records.append(record)
    for name, parameter in named_parameters:
        if not parameter.requires_grad:continue
        n=parameter.numel()
        rate=None
        if learning_rates is not None:
            rate=float(learning_rates.get(name.split('.')[0], learning_rates['base']))
            if not math.isfinite(rate) or rate<=0:raise ValueError('Invalid module learning rate: '+name)
        if n>maximum:raise ValueError(f'Individual parameter exceeds the safe optimizer bound: {name}')
        if current and (size+n>maximum or current_lr!=rate):
            append()
            current=[];names=[];size=0
        current_lr=rate
        current.append(parameter);names.append(name);size+=n
    if current:append()
    if not groups:raise ValueError('No trainable parameters')
    return groups,records


def configured_gradient_clipping(trainer):
    """The trainer uses gradient_clipping; max_grad_norm is its legacy alias."""
    value=float(trainer.get('gradient_clipping', trainer.get('max_grad_norm', 1.)))
    if not math.isfinite(value) or value<0:raise ValueError('Invalid gradient clipping threshold')
    return value


def sample_indices(size, samples, device):
    if size<1 or samples<1:raise ValueError('Positive master tensor/sample sizes required')
    count=min(samples,size)
    # Float32 linspace can round a large final index from size-1 to size.
    return torch.arange(count,device=device,dtype=torch.int64)*(size-1)//max(1,count-1)


def capture_master_samples(optimizer, samples=512):
    captured=[]
    for parameter in optimizer.single_partition_of_fp32_groups:
        if parameter.dtype!=torch.float32 or parameter.numel()>=2**31:
            raise ValueError('Unsafe optimizer partition dtype/extent')
        indices=sample_indices(parameter.numel(),samples,parameter.device)
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
