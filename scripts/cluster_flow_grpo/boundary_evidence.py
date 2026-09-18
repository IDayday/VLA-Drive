"""Exact boundary comparison with a fast path for identical finite tensors.

The native comparator still performs its own dtype, torch.equal, nonidentical,
metadata, shard inventory and Python/RNG checks. Only its diagnostic statistics
callback is specialized: identical tensors need no float64 error histograms.
Every unequal tensor takes the original exhaustive numerical diagnostic path.
This read-only subprocess never loads or patches the training actor.
"""
import argparse
import json
import torch
from scripts.flow_grpo import compare_boundaries as native


def exact_statistics(reference, actual, atol=0.0, rtol=0.0):
    if atol != 0.0 or rtol != 0.0:
        raise ValueError("boundary observer requires exact zero tolerance")
    if (reference.shape == actual.shape and reference.dtype == actual.dtype
            and torch.equal(reference, actual) and torch.isfinite(reference).all()):
        return {"reference_present": True, "actual_present": True,
                "numel": reference.numel(), "max_abs": 0.0,
                "difference_norm": 0.0, "finite": True, "allclose": True,
                "outside_tolerance": 0, "atol": 0.0, "rtol": 0.0,
                "statistics": "finite torch.equal fast path; norms not computed"}
    return original_statistics(reference, actual, atol=atol, rtol=rtol)


original_statistics = native.tensor_comparison


def compare(left, right, output, streaming_load=False):
    # The replacement is scoped to the original report's diagnostic callback;
    # the independent exact equality checks in native.compare_values remain.
    previous = native.tensor_comparison
    original_load = torch.load
    def load(*args, **kwargs):
        kwargs['mmap'] = False
        return original_load(*args, **kwargs)
    try:
        native.tensor_comparison = exact_statistics
        if streaming_load:
            torch.load = load
        return native.compare_boundaries(left, right, output)
    finally:
        native.tensor_comparison = previous
        torch.load = original_load


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--continuous", required=True)
    p.add_argument("--resumed", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--cpu-threads", type=int, default=8)
    p.add_argument("--streaming-load", action="store_true", help="read one shard pair sequentially instead of network mmap page faults")
    a = p.parse_args()
    if not 1 <= a.cpu_threads <= 16:
        p.error("cpu threads must be within the bounded1..16 allocation")
    torch.set_num_threads(a.cpu_threads)
    result = compare(a.continuous, a.resumed, a.output, a.streaming_load)
    print(json.dumps({"status": result["status"], "files": len(result["files"])}))
