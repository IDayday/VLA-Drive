"""Global scene weighting and synchronized phase failures."""
import torch
import torch.distributed as dist


def global_count(local, device):
    count = torch.tensor(float(local), device=device)
    if dist.is_initialized():
        dist.all_reduce(count)
    return count


def globally_weighted_sum(local_sum, global_denominator):
    world = dist.get_world_size() if dist.is_initialized() else 1
    if global_denominator <= 0:
        raise ValueError("no globally valid scenes")
    return local_sum * world / global_denominator


def synchronized_call(function, device):
    value = None
    error = None
    try:
        value = function()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    failed = torch.tensor(int(error is not None), device=device)
    if dist.is_initialized():
        dist.all_reduce(failed)
    if failed.item():
        errors = [None] * (dist.get_world_size() if dist.is_initialized() else 1)
        if dist.is_initialized():
            dist.all_gather_object(errors, error)
        else:
            errors = [error]
        raise RuntimeError(f"collective phase failed: {errors}")
    return value
