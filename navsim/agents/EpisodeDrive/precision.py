"""FP32 storage contract, independent of autocast activation precision."""
from collections import Counter

import torch


def promote_trainable_parameters(module):
    """Call after freeze rules and before shared initialization / optimizer."""
    for parameter in module.parameters():
        if parameter.requires_grad and parameter.dtype != torch.float32:
            parameter.data = parameter.detach().float()
            if parameter.grad is not None:
                parameter.grad.data = parameter.grad.detach().float()


def audit_precision(module, optimizer=None, require_initialized_moments=False):
    trainable, frozen, moments = Counter(), Counter(), Counter()
    invalid = []
    for name, parameter in module.named_parameters():
        (trainable if parameter.requires_grad else frozen)[str(parameter.dtype)] += parameter.numel()
        if parameter.requires_grad and parameter.dtype != torch.float32:
            invalid.append(name)
    if optimizer is not None:
        for state in optimizer.state.values():
            for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if name in state:
                    value = state[name]
                    moments[str(value.dtype)] += value.numel()
                    if value.dtype != torch.float32:
                        invalid.append("optimizer." + name)
        if require_initialized_moments and not moments:
            raise RuntimeError("Adam moments have not been materialized by an optimizer update")
    if invalid:
        raise RuntimeError("FP32 trainable/master contract violated: " + repr(invalid[:32]))
    return {"schema_version": 1, "trainable_storage": dict(trainable),
            "frozen_storage": dict(frozen), "adam_moment_storage": dict(moments),
            "moments_observed": bool(moments)}


@torch.no_grad()
def parameter_update_statistics(before, module):
    parameters = dict(module.named_parameters())
    result = {}
    for name, previous in before.items():
        current = parameters[name].detach().float()
        delta = current - previous.to(current.device)
        result[name] = {"update_weight_ratio": float(delta.norm() / previous.norm().clamp_min(1e-30)),
                        "changed_fraction": float((delta != 0).float().mean())}
    return result
