"""Read an existing bounded Kineto trace without adding its inclusive times.

GPU stream interval unions distinguish work, overlap, and NCCL-only activity.
NCCL-only activity is not proof of critical-path waiting: synchronization and
dependencies must still be inspected. This analyzer never launches training.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


def merged_intervals(intervals):
    out = []
    for start, end in sorted(intervals):
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            raise ValueError("Invalid trace interval")
        if start == end:
            continue
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def interval_measure(intervals):
    return sum(end - start for start, end in merged_intervals(intervals))


def interval_overlap(first, second):
    first, second = merged_intervals(first), merged_intervals(second)
    i = j = 0
    total = 0.
    while i < len(first) and j < len(second):
        total += max(0., min(first[i][1], second[j][1]) - max(first[i][0], second[j][0]))
        if first[i][1] < second[j][1]:
            i += 1
        else:
            j += 1
    return total


def summarize_trace(document):
    if not isinstance(document, dict) or not isinstance(document.get("traceEvents"), list):
        raise ValueError("Expected Kineto traceEvents")
    device_intervals = defaultdict(lambda: defaultdict(list))
    kernel_work, kernel_count, cpu_regions, launch_count = Counter(), Counter(), Counter(), Counter()
    gpu_regions, collective_input_bytes, unknown_collective_inputs = Counter(), Counter(), Counter()
    attention, communication, graphs = set(), [], Counter()
    for event in document["traceEvents"]:
        if event.get("ph") != "X" or "dur" not in event or "ts" not in event:
            continue
        start, duration = float(event["ts"]), float(event["dur"])
        if not math.isfinite(start) or not math.isfinite(duration) or duration < 0:
            raise ValueError("Invalid event timing")
        name, category, args = str(event.get("name", "")), str(event.get("cat", "")), event.get("args", {})
        end = start + duration
        lower = name.lower()
        if category in ("kernel", "cuda_kernel"):
            device = str(args.get("device", args.get("Device Id", event.get("pid", "unknown"))))
            kind = "nccl" if "nccl" in lower else "compute"
            device_intervals[device][kind].append((start, end))
            kernel_work[name] += duration
            kernel_count[name] += 1
            if any(token in lower for token in ("flash", "attention", "fmha", "scaled_dot")):
                attention.add(name)
        elif category in ("gpu_memcpy", "gpu_memset", "cuda_memcpy", "cuda_memset"):
            device = str(args.get("device", args.get("Device Id", event.get("pid", "unknown"))))
            device_intervals[device]["transfer"].append((start, end))
        elif category in ("cuda_runtime", "cuda_driver") and "launch" in lower:
            launch_count[name] += 1
        if name.startswith(("st.", "st_profile.")) and category == "user_annotation":
            cpu_regions[name] += duration
        elif name.startswith(("st.", "st_profile.")) and category == "gpu_user_annotation":
            # Kineto emits a second, GPU-side annotation with the same name.
            # Adding it to CPU wall double-counts overlapping execution.
            gpu_regions[name] += duration
        if category == "user_annotation" and any(x in lower for x in ("c10d", "nccl", "collective")):
            # Keep only small structured metadata; no tensor contents.
            communication.append({"name": name, "duration_us": duration,
                "metadata": {k: v for k, v in args.items() if k in (
                    "Collective name", "In msg nelems", "Out msg nelems", "dtype",
                    "Process Group Size", "Rank", "Input Dims", "Input type")}})
            dims, types = args.get("Input Dims"), args.get("Input type")
            item_bytes = {"c10::BFloat16": 2, "c10::Half": 2, "float": 4,
                          "double": 8, "long int": 8, "int": 4, "unsigned char": 1}
            if (isinstance(dims, list) and isinstance(types, list) and len(dims) == len(types)
                    and all(isinstance(shape, list) and all(type(n) is int and n >= 0 for n in shape)
                            and dtype in item_bytes for shape, dtype in zip(dims, types))):
                collective_input_bytes[name] += sum(math.prod(shape) * item_bytes[dtype]
                                                    for shape, dtype in zip(dims, types))
            else:
                unknown_collective_inputs[name] += 1
        if "torch-compiled" in lower or "graph break" in lower:
            graphs[name] += 1
    devices = {}
    for device, kinds in sorted(device_intervals.items()):
        compute, nccl, transfer = (kinds[x] for x in ("compute", "nccl", "transfer"))
        overlap = interval_overlap(compute, nccl)
        last_compute = max((end for _, end in compute), default=None)
        tail = [(max(start, last_compute), end) for start, end in nccl
                if last_compute is not None and end > last_compute]
        all_intervals = compute + nccl + transfer
        devices[device] = {
            "gpu_activity_span_ms": (max(e for _, e in all_intervals) - min(s for s, _ in all_intervals)) / 1000,
            "compute_union_ms": interval_measure(compute) / 1000,
            "nccl_union_ms": interval_measure(nccl) / 1000,
            "compute_nccl_overlap_ms": overlap / 1000,
            "nccl_without_compute_ms": (interval_measure(nccl) - overlap) / 1000,
            "nccl_after_last_compute_ms": interval_measure(tail) / 1000,
            "transfer_union_ms": interval_measure(transfer) / 1000,
            "all_activity_union_ms": interval_measure(all_intervals) / 1000,
            "kernel_count": len(compute) + len(nccl),
        }
    return {"schema": "st_kineto_trace_summary_v1", "eligible_for_throughput": False,
        "semantics": "per-device interval unions; NCCL-only activity is not a proven critical-path dependency",
        "devices": devices,
        "top_kernel_work": [{"name": name, "inclusive_work_ms": total / 1000,
                             "calls": kernel_count[name]} for name, total in kernel_work.most_common(30)],
        "cpu_region_inclusive_ms_not_additive": {name: value / 1000 for name, value in cpu_regions.items()},
        "gpu_annotation_inclusive_ms_not_additive": {name: value / 1000 for name, value in gpu_regions.items()},
        "cuda_launch_calls": dict(launch_count), "observed_attention_kernel_names": sorted(attention),
        "communication_metadata": communication,
        "collective_logical_input_bytes": dict(collective_input_bytes),
        "collective_input_records_without_complete_size": dict(unknown_collective_inputs),
        "communication_bytes": "Logical input tensor bytes where metadata permits; network wire bytes NOT_MEASURED",
        "observed_compile_or_graph_break_events": dict(graphs),
        "missing_graph_break_events_do_not_prove_zero_graph_breaks": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = args.trace.read_bytes()
    result = summarize_trace(json.loads(data))
    result.update(trace_path=str(args.trace.resolve()), trace_sha256=hashlib.sha256(data).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"status": "ANALYZED_NOT_AN_APPROVAL", "output": str(args.output)}))


if __name__ == "__main__":
    main()
