"""Version-locked correction for DeepSpeed 0.16.9's partition-list dtype bug.

The installed non-offload ZeRO-2 epilogue requests FP32 partitions but
get_flat_partition(return_tensor_list=True) ignores dtype for nonempty grads.
Its next microbatch then adds into BF16 tensors. Convert each *new contribution*
before it enters averaged_gradients. This does not recover precision lost in a
BF16 backward or the communication result's cast back to BF16. It specifically
prevents further rounding of the running microbatch sum.

Only this optimizer instance is changed; no installed files, global class or
autograd hooks are patched. Unsupported implementations/configurations fail
closed. Checkpoint tensors and DeepSpeed's step/partition/collective logic stay
under the installed implementation's ownership.
"""

import hashlib
import inspect
import types
import torch

PROFILE = "bf16_zero2_fp32_partition_v2"
EXPECTED_SOURCE = {
    "get_flat_partition": "e6fb8ea86bbbff48a7adaefdbb97b0cba1b89c486677ba6db6f7355f88fdef28",
    "independent_gradient_partition_epilogue": "772d891ad261c387fa6cb37779cae8da9a5dd56ea45873406bca9255f4aa4921",
}


def partition_in_requested_dtype(original, *args, **kwargs):
    """Honor the original API's dtype, before the epilogue's in-place addition."""
    # Bind the real signature to support positional and keyword callers alike.
    bound = inspect.signature(original).bind(*args, **kwargs)
    bound.apply_defaults()
    result = original(*args, **kwargs)
    dtype = bound.arguments["dtype"]
    if bound.arguments["return_tensor_list"]:
        return [tensor.to(dtype=dtype) for tensor in result]
    return result.to(dtype=dtype)


def install_fp32_partitions(engine):
    import deepspeed
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

    optimizer = engine.optimizer
    if getattr(optimizer, "_flow_fp32_partitions", None):
        return optimizer._flow_fp32_partitions
    if deepspeed.__version__ != "0.16.9" or type(optimizer) is not DeepSpeedZeroOptimizer:
        raise RuntimeError("FP32 partition correction requires audited DeepSpeed 0.16.9 ZeRO-2")
    actual = {
        name: hashlib.sha256(inspect.getsource(getattr(type(optimizer), name)).encode()).hexdigest()
        for name in EXPECTED_SOURCE
    }
    if actual != EXPECTED_SOURCE:
        raise RuntimeError("DeepSpeed partition implementation changed; re-audit before training")
    if (
        not optimizer.partition_gradients
        or optimizer.cpu_offload
        or optimizer.dtype != torch.bfloat16
        or optimizer.gradient_accumulation_dtype != torch.float32
        or optimizer.communication_data_type != torch.float32
        or optimizer.overlap_comm
    ):
        raise ValueError("FP32 partition profile requires BF16 ZeRO-2, FP32 communication, no offload/overlap")
    if optimizer.averaged_gradients:
        raise RuntimeError("install precision correction before the first backward")
    original = optimizer.get_flat_partition

    def corrected(self, *args, **kwargs):
        return partition_in_requested_dtype(original, *args, **kwargs)

    optimizer.get_flat_partition = types.MethodType(corrected, optimizer)
    optimizer._flow_fp32_partitions = {
        "profile": PROFILE,
        "deepspeed_version": deepspeed.__version__,
        "original_method_sha256": actual,
        "scope": "new reduced partition contributions cast before microbatch accumulation",
    }
    return optimizer._flow_fp32_partitions
