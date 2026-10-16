"""Bounded instrumentation and honest complete-window topology comparisons.

This module does not launch training or change an objective. The trainer calls
begin_update immediately before fetching an optimizer batch and end_update only
AFTER reductions, logging/status writes and any checkpoint. Region/trace capture
is a separate diagnostic run: a trace run is ineligible for throughput approval.
CUDA event times are inclusive single-stream intervals, never additive critical
path components. Full-window wall time is the throughput denominator.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import resource
import socket
import time

import numpy as np
import torch


@dataclass(frozen=True)
class TimingPlan:
    warmup: int = 20
    steady: int = 100
    windows: int = 2

    def __post_init__(self):
        if any(type(x) is not int for x in (self.warmup, self.steady, self.windows)):
            raise ValueError("Resolved integer window sizes required")
        if self.warmup < 20 or self.steady < 100 or self.windows != 2 or self.updates > 256:
            raise ValueError("Two >=20 warmup + >=100 steady windows, at most256 updates")

    @property
    def updates(self):
        return (self.warmup + self.steady) * self.windows

    def steady_steps(self, window):
        if not 0 <= window < self.windows:
            raise ValueError("Unknown timing window")
        start = window * (self.warmup + self.steady) + self.warmup + 1
        return list(range(start, start + self.steady))


def _cpu_usage():
    own, child = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return {"self_seconds": own.ru_utime + own.ru_stime,
            "waited_children_seconds": child.ru_utime + child.ru_stime,
            "peak_rss_bytes": own.ru_maxrss * 1024,
            "children_scope": "terminated-and-waited children only; live workers require process sampler"}


def _tensor_summary(value, limit=12):
    result = []
    def visit(x):
        if len(result) >= limit:
            return
        if isinstance(x, torch.Tensor):
            result.append({"shape": list(x.shape), "dtype": str(x.dtype),
                           "device": str(x.device), "requires_grad": x.requires_grad})
        elif isinstance(x, dict):
            for v in x.values():
                visit(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                visit(v)
    visit(value)
    return result


def dtype_inventory(model, optimizer=None):
    """Actual parameter/gradient/master/moment dtypes; uncreated states stay absent."""
    modules = {}
    for name, module in model.named_modules():
        counts = Counter()
        for parameter in module.parameters(recurse=False):
            counts[f"parameter:{parameter.dtype}:trainable={parameter.requires_grad}"] += parameter.numel()
            if parameter.grad is not None:
                counts[f"gradient:{parameter.grad.dtype}"] += parameter.grad.numel()
        for buffer in module.buffers(recurse=False):
            counts[f"buffer:{buffer.dtype}"] += buffer.numel()
        if counts:
            modules[name] = dict(counts)
    opt = Counter()
    if optimizer is not None:
        for master in getattr(optimizer, "single_partition_of_fp32_groups", ()):
            opt[f"zero_master:{master.dtype}"] += master.numel()
        inner = getattr(optimizer, "optimizer", optimizer)
        for state in getattr(inner, "state", {}).values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    opt[f"{key}:{value.dtype}"] += value.numel()
    return {"modules": modules, "optimizer": dict(opt),
            "optimizer_states_created": bool(opt), "count_unit": "tensor_elements"}


class PipelineProfiler:
    """Per-rank recorder. No collectives; caller preserves original RNG and math.

    identity must pin actual SHA/config/data/teacher/environment/lease. Timestamp
    clocks are comparable only within a host. GPU work is drained at update
    boundaries, not between regions. The profiling API itself consumes no RNG.
    """
    def __init__(self, output, *, identity, rank, world_size, gpu_uuids,
                 mode="timing", trace_steps=(2, 3), cuda=True):
        if mode not in ("timing", "trace"):
            raise ValueError("mode must be timing or trace")
        if type(rank) is not int or not 0 <= rank < world_size:
            raise ValueError("Invalid distributed rank")
        if not isinstance(identity, dict) or not identity:
            raise ValueError("Pinned runtime identity is required")
        if len(gpu_uuids) != world_size or len(set(gpu_uuids)) != world_size:
            raise ValueError("One unique GPU per rank must be recorded")
        if mode == "trace" and (not trace_steps or len(trace_steps) > 4 or min(trace_steps) < 1):
            raise ValueError("A trace run may record only1..4 explicitly selected updates")
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.identity, self.rank, self.world_size = identity, rank, world_size
        self.gpu_uuids, self.mode, self.cuda = list(gpu_uuids), mode, cuda
        self.trace_steps = tuple(trace_steps) if mode == "trace" else ()
        self.rows, self.regions, self.calls, self.handles = [], [], [], []
        self.current = None
        self.trace = None
        self._region_stack = []

    def _sync(self):
        if self.cuda:
            torch.cuda.synchronize()

    def begin_update(self, step):
        if self.current is not None or type(step) is not int or step < 1:
            raise ValueError("Nested/invalid optimizer update")
        if self.rows and step != self.rows[-1]["step"] + 1:
            raise ValueError("Profile cannot silently omit or duplicate optimizer updates")
        if step > 256:
            raise ValueError("PROFILE exceeds256 optimizer updates")
        self._sync()
        if self.mode == "trace" and step in self.trace_steps:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if self.cuda:
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            self.trace = torch.profiler.profile(activities=activities, record_shapes=True,
                                                profile_memory=True, with_stack=False)
            self.trace.start()
        if self.cuda:
            torch.cuda.reset_peak_memory_stats()
        self.current = {"step": step, "start_ns": time.perf_counter_ns(),
                        "start_unix": time.time(), "cpu_start": _cpu_usage()}

    @contextmanager
    def region(self, name):
        """Optional trace-only range; zero synchronization between regions."""
        if self.mode != "trace" or self.trace is None:
            yield
            return
        start = time.perf_counter_ns()
        events = None
        if self.cuda:
            events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            events[0].record()
        with torch.profiler.record_function("st_profile." + name):
            yield
        if events:
            events[1].record()
        self.regions.append({"step": self.current["step"], "name": name,
                             "cpu_dispatch_seconds": (time.perf_counter_ns()-start)/1e9,
                             "events": events})

    def attach_modules(self, modules):
        """Explicit name->module mapping. Hooks never replace inputs or outputs."""
        if self.mode != "trace":
            return
        for name, module in modules.items():
            active = []
            def before(mod, args, kwargs, scope=name, stack=active):
                if self.trace is None:
                    stack.append(None)
                    return
                context = self.region(scope)
                context.__enter__()
                call = {"step": self.current["step"], "module": scope,
                    "grad_enabled": torch.is_grad_enabled(), "training": mod.training,
                    "inputs": _tensor_summary((args, kwargs))}
                stack.append((context, call))
                self.calls.append(call)
            def after(mod, args, kwargs, output, stack=active):
                active_call = stack.pop()
                if active_call is not None:
                    context, call = active_call
                    context.__exit__(None, None, None)
                    call["outputs"] = _tensor_summary(output)
            self.handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
            self.handles.append(module.register_forward_hook(after, with_kwargs=True, always_call=True))

    def end_update(self, *, local_scenes, checkpoint_seconds=0., metadata=None):
        if self.current is None or type(local_scenes) is not int or local_scenes < 1:
            raise ValueError("A completed real optimizer batch is required")
        self._sync()
        end = time.perf_counter_ns()
        row = {**self.current, "end_ns": end, "end_unix": time.time(),
               "seconds": (end-self.current["start_ns"])/1e9,
               "local_scenes": local_scenes, "checkpoint_seconds": checkpoint_seconds,
               "cpu_end": _cpu_usage(), "metadata": metadata or {},
               "peak_allocated_bytes": torch.cuda.max_memory_allocated() if self.cuda else None,
               "peak_reserved_bytes": torch.cuda.max_memory_reserved() if self.cuda else None}
        if not math.isfinite(checkpoint_seconds) or not 0 <= checkpoint_seconds <= row["seconds"]:
            raise ValueError("Checkpoint timing must be a measured nonoverlapping subset of this update")
        self.rows.append(row)
        if self.trace is not None:
            self.trace.stop()
            self.trace.export_chrome_trace(str(self.output / f"trace_rank{self.rank}_step{row['step']:04d}.json"))
            self.trace = None
        self.current = None

    def finish(self):
        if self.current is not None:
            raise ValueError("An incomplete update cannot be marked complete")
        for handle in self.handles:
            handle.remove()
        for row in self.regions:
            events = row.pop("events")
            row["cuda_inclusive_milliseconds"] = events[0].elapsed_time(events[1]) if events else None
        result = {"schema": "st_pipeline_profile_rank_v1", "identity": self.identity,
                  "hostname": socket.gethostname(), "pid": os.getpid(), "rank": self.rank,
                  "world_size": self.world_size, "gpu_uuids": self.gpu_uuids,
                  "mode": self.mode, "eligible_for_throughput": self.mode == "timing",
                  "updates": self.rows, "module_calls": self.calls, "regions": self.regions,
                  "cuda_region_semantics": "inclusive current-stream; may overlap and must not be summed"}
        path = self.output / f"rank{self.rank}.json"
        with path.open("x") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        return result


def summarize_ranks(records, plan=TimingPlan()):
    if not records or any(not r["eligible_for_throughput"] or r["mode"] != "timing" for r in records):
        raise ValueError("Only uninstrumented timing runs can establish throughput")
    first = records[0]
    world = first["world_size"]
    if len(records) != world or sorted(r["rank"] for r in records) != list(range(world)):
        raise ValueError("Every rank must be present exactly once")
    for r in records:
        if any(r[k] != first[k] for k in ("identity", "hostname", "gpu_uuids", "world_size")):
            raise ValueError("Mixed run or cross-host monotonic clock identity")
        if [x["step"] for x in r["updates"]] != list(range(1, plan.updates+1)):
            raise ValueError("Incomplete/noncontinuous two-window measurement")
    windows = []
    for number in range(plan.windows):
        ids = plan.steady_steps(number)
        selected = [[r["updates"][step-1] for step in ids] for r in records]
        begin = min(rows[0]["start_ns"] for rows in selected)
        end = max(rows[-1]["end_ns"] for rows in selected)
        wall = (end-begin)/1e9
        if wall <= 0:
            raise ValueError("Nonpositive measured interval")
        sample_count = sum(x["local_scenes"] for rows in selected for x in rows)
        expected_batch = first["identity"].get("global_batch")
        if expected_batch is None or any(sum(rows[i]["local_scenes"] for rows in selected) != expected_batch
                                         for i in range(len(ids))):
            raise ValueError("Measured global batch differs from the registered comparison")
        slow = [max(rows[i]["seconds"] for rows in selected) for i in range(len(ids))]
        normal = [max(rows[i]["seconds"] for rows in selected) for i in range(len(ids))
                  if all(rows[i]["checkpoint_seconds"] == 0 for rows in selected)]
        cpu = sum(rows[-1]["cpu_end"]["self_seconds"]-rows[0]["cpu_start"]["self_seconds"]
                  + rows[-1]["cpu_end"]["waited_children_seconds"]-rows[0]["cpu_start"]["waited_children_seconds"]
                  for rows in selected)
        windows.append({"window": number+1, "steps": ids, "start_ns": begin, "end_ns": end,
            "complete_wall_seconds": wall, "scenes": sample_count, "scenes_per_second": sample_count/wall,
            "slowest_rank_update_mean": float(np.mean(slow)), "slowest_rank_update_p50": float(np.quantile(slow,.5)),
            "slowest_rank_update_p95": float(np.quantile(slow,.95)),
            "normal_updates": len(normal), "normal_update_mean": float(np.mean(normal)) if normal else None,
            "normal_update_p50": float(np.quantile(normal,.5)) if normal else None,
            "normal_update_p95": float(np.quantile(normal,.95)) if normal else None,
            "allocation_gpu_seconds": world*wall, "cpu_self_and_waited_child_seconds": cpu,
            "peak_allocated_bytes": max((x["peak_allocated_bytes"] or 0) for rows in selected for x in rows),
            "peak_reserved_bytes": max((x["peak_reserved_bytes"] or 0) for rows in selected for x in rows),
            "checkpoint_events": sum(any(rows[i]["checkpoint_seconds"]>0 for rows in selected) for i in range(len(ids))),
            "checkpoint_max_rank_seconds": sum(max(rows[i]["checkpoint_seconds"] for rows in selected) for i in range(len(ids)))})
    return {"schema": "st_pipeline_timing_summary_v1", "identity": first["identity"],
            "hostname": first["hostname"], "gpu_uuids": first["gpu_uuids"], "world_size": world,
            "windows": windows, "status": "MEASURED_NOT_AN_APPROVAL"}


def aggregate_concurrent(first, second):
    """True 2x4 makespan throughput, never twice a standalone job measurement."""
    if (first["hostname"] != second["hostname"] or first["world_size"] != 4 or second["world_size"] != 4
            or set(first["gpu_uuids"]).intersection(second["gpu_uuids"])):
        raise ValueError("2x4 requires the same node and disjoint four-GPU jobs")
    keys = ("source_sha", "scientific_config_sha256", "data_manifest_sha256", "teacher_identity",
            "precision_policy", "global_batch", "sample_order_sha256")
    if any(k not in first["identity"] or first["identity"][k] != second["identity"].get(k) for k in keys):
        raise ValueError("Concurrent jobs do not have matching scientific/data/sample contracts")
    if len(first["windows"]) != 2 or len(second["windows"]) != 2:
        raise ValueError("Both complete steady windows are required for both jobs")
    output = []
    for a, b in zip(first["windows"], second["windows"]):
        if a["window"] != b["window"]:
            raise ValueError("Concurrent windows are misaligned")
        overlap = min(a["end_ns"], b["end_ns"]) - max(a["start_ns"], b["start_ns"])
        if overlap <= 0:
            raise ValueError("These jobs were not measured concurrently")
        span = (max(a["end_ns"], b["end_ns"]) - min(a["start_ns"], b["start_ns"]))/1e9
        samples = a["scenes"] + b["scenes"]
        output.append({"window": a["window"], "joint_wall_seconds": span,
                       "measured_overlap_seconds": overlap/1e9, "scenes": samples,
                       "aggregate_scenes_per_second": samples/span, "allocation_gpu_seconds": 8*span,
                       "both_jobs_finished": True})
    return {"schema": "st_true_two_by_four_profile_v1", "windows": output,
            "status": "MEASURED_NOT_AN_APPROVAL"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    summary = commands.add_parser("summarize")
    summary.add_argument("--ranks", nargs="+", required=True)
    aggregate = commands.add_parser("aggregate-two-by-four")
    aggregate.add_argument("--first", required=True)
    aggregate.add_argument("--second", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    load = lambda path: json.loads(Path(path).read_text())
    result = summarize_ranks([load(p) for p in args.ranks]) if args.command == "summarize" else aggregate_concurrent(load(args.first),load(args.second))
    with Path(args.output).open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "output": args.output}))


if __name__ == "__main__":
    main()
