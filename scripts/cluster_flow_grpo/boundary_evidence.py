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


def compare(left, right, output):
    # The replacement is scoped to the original report's diagnostic callback;
    # the independent exact equality checks in native.compare_values remain.
    previous = native.tensor_comparison
    try:
        native.tensor_comparison = exact_statistics
        return native.compare_boundaries(left, right, output)
    finally:
        native.tensor_comparison = previous


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--continuous", required=True)
    p.add_argument("--resumed", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--cpu-threads", type=int, default=8)
    a = p.parse_args()
    if not 1 <= a.cpu_threads <= 16:
        p.error("cpu threads must be within the bounded1..16 allocation")
    torch.set_num_threads(a.cpu_threads)
    result = compare(a.continuous, a.resumed, a.output)
    print(json.dumps({"status": result["status"], "files": len(result["files"])}))
