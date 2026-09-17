"""Global scene weighting and synchronized phase failures."""
import torch
import torch.distributed as dist
from datetime import timedelta

_CONTROL_GROUP = None


def initialize_control_group(timeout_seconds=120):
    """CPU control plane, used only around operations with NO collectives."""
    global _CONTROL_GROUP
    if dist.is_initialized():
        _CONTROL_GROUP = dist.new_group(
            backend="gloo", timeout=timedelta(seconds=timeout_seconds)
        )


def rank0_call(function, device=None):
    return synchronized_call(
        lambda: function()
        if not dist.is_initialized() or dist.get_rank() == 0
        else None,
        device,
    )


def rank0_result(function, device=None):
    value = rank0_call(function, device)
    items = [value]
    if dist.is_initialized():
        dist.broadcast_object_list(items, src=0, group=_CONTROL_GROUP)
    return items[0]


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
    """Coordinate recoverable local I/O/CPU errors, never distributed forwards.

    CUDA/communication/abrupt rank failures must escape to torchrun. Calling a
    new barrier on a failed communicator cannot recover those failures.
    """
    value = None
    error = None
    try:
        value = function()
    except Exception as exc:
        if isinstance(exc, torch.cuda.OutOfMemoryError) or any(
            word in str(exc).lower()
            for word in ("nccl", "cuda error", "connection closed by peer")
        ):
            raise
        error = f"{type(exc).__name__}: {exc}"
    control_device = "cpu" if _CONTROL_GROUP is not None else device
    failed = torch.tensor(int(error is not None), device=control_device)
    if dist.is_initialized():
        dist.all_reduce(failed, group=_CONTROL_GROUP)
    if failed.item():
        errors = [None] * (dist.get_world_size() if dist.is_initialized() else 1)
        if dist.is_initialized():
            dist.all_gather_object(errors, error, group=_CONTROL_GROUP)
        else:
            errors = [error]
        raise RuntimeError(f"collective phase failed: {errors}")
    return value
