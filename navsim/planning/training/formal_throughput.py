"""Formal PlanReg-WM throughput benchmark callback and metrics schema."""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, DefaultDict, Dict, List, Optional

import psutil
import pytorch_lightning as pl
import torch
import torch.distributed as dist


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


class FormalThroughputBenchmarkCallback(pl.Callback):
    """Collect 20 warmup + 300 timed optimizer steps and stop the smoke run."""

    def __init__(
        self,
        output_path: str,
        *,
        global_batch_size: int,
        warmup_steps: int = 20,
        timed_steps: int = 300,
        layout_name: str,
        scorer_processes_per_rank: int,
        scorer_partitions_per_scene: int,
        num_workers: int,
        gradient_checkpointing: bool,
        read_only_attention_backend: str,
    ) -> None:
        super().__init__()
        if warmup_steps < 0 or timed_steps <= 0 or global_batch_size <= 0:
            raise ValueError("Invalid formal throughput benchmark dimensions")
        if scorer_processes_per_rank <= 0 or scorer_partitions_per_scene <= 0:
            raise ValueError("Formal scorer process and partition counts must be positive")
        self.output_path = str(Path(output_path).expanduser().resolve())
        self.global_batch_size = int(global_batch_size)
        self.warmup_steps = int(warmup_steps)
        self.timed_steps = int(timed_steps)
        self.layout_name = str(layout_name)
        self.scorer_processes_per_rank = int(scorer_processes_per_rank)
        self.scorer_partitions_per_scene = int(scorer_partitions_per_scene)
        self.num_workers = int(num_workers)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.read_only_attention_backend = str(read_only_attention_backend)
        self._batch_start = None
        self._previous_batch_end = None
        self._seen = 0
        self._records: List[Dict[str, float]] = []
        self._gpu_utilization: List[float] = []
        self._cpu_utilization: List[float] = []
        self._io_wait: List[float] = []
        self._peak_child_processes = 0
        self._exception: Optional[BaseException] = None

    def on_train_start(self, trainer, pl_module) -> None:
        del trainer, pl_module
        self._previous_batch_end = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        del trainer, pl_module, batch, batch_idx
        now = time.perf_counter()
        self._data_wait = (
            0.0
            if self._previous_batch_end is None
            else now - self._previous_batch_end
        )
        self._batch_start = now

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ) -> None:
        del outputs, batch, batch_idx
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        now = time.perf_counter()
        step_time = now - self._batch_start
        self._previous_batch_end = now
        self._seen += 1
        timings = (
            pl_module.consume_formal_step_timings()
            if hasattr(pl_module, "consume_formal_step_timings")
            else {}
        )
        timings["data_wait"] = float(self._data_wait)
        timings["step_time"] = float(step_time)

        if self._seen > self.warmup_steps:
            self._records.append(timings)
            if self._seen % 10 == 0:
                self._cpu_utilization.append(float(psutil.cpu_percent(interval=None)))
                cpu_times = psutil.cpu_times_percent(interval=None)
                self._io_wait.append(float(getattr(cpu_times, "iowait", 0.0)))
                try:
                    self._gpu_utilization.append(float(torch.cuda.utilization()))
                except Exception:
                    pass
                self._peak_child_processes = max(
                    self._peak_child_processes,
                    len(psutil.Process().children(recursive=True)),
                )
        if len(self._records) >= self.timed_steps:
            trainer.should_stop = True

    def _local_payload(self) -> Dict[str, Any]:
        return {
            "records": self._records,
            "gpu_utilization": self._gpu_utilization,
            "cpu_utilization": self._cpu_utilization,
            "io_wait": self._io_wait,
            "peak_allocated_bytes": (
                int(torch.cuda.max_memory_allocated())
                if torch.cuda.is_available()
                else 0
            ),
            "peak_reserved_bytes": (
                int(torch.cuda.max_memory_reserved())
                if torch.cuda.is_available()
                else 0
            ),
            "peak_child_processes": self._peak_child_processes,
        }

    def on_train_end(self, trainer, pl_module) -> None:
        del pl_module
        if self._exception is not None:
            return
        local = self._local_payload()
        if dist.is_available() and dist.is_initialized():
            gathered = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered, local)
        else:
            gathered = [local]
        if not trainer.is_global_zero:
            return

        rank_zero_records = gathered[0]["records"]
        all_records = [record for payload in gathered for record in payload["records"]]
        by_metric: DefaultDict[str, List[float]] = defaultdict(list)
        for record in all_records:
            for name, value in record.items():
                if math.isfinite(float(value)):
                    by_metric[name].append(float(value))
        step_times = [record["step_time"] for record in rank_zero_records]
        mean_step = statistics.fmean(step_times) if step_times else float("nan")
        metrics = {
            "schema_version": 1,
            "layout": self.layout_name,
            "status": (
                "success"
                if len(rank_zero_records) == self.timed_steps
                else "incomplete"
            ),
            "warmup_steps": self.warmup_steps,
            "timed_optimizer_steps": len(rank_zero_records),
            "requested_timed_optimizer_steps": self.timed_steps,
            "world_size": len(gathered),
            "gpu_count": len(gathered),
            "per_gpu_batch_size": self.global_batch_size // len(gathered),
            "global_batch_size": self.global_batch_size,
            "num_workers_per_rank": self.num_workers,
            "gradient_checkpointing": self.gradient_checkpointing,
            "read_only_attention_backend": self.read_only_attention_backend,
            "samples_per_second": self.global_batch_size / mean_step,
            "optimizer_steps_per_second": 1.0 / mean_step,
            "median_step_time": _percentile(step_times, 0.50),
            "p90_step_time": _percentile(step_times, 0.90),
            "peak_allocated_bytes": max(
                payload["peak_allocated_bytes"] for payload in gathered
            ),
            "peak_reserved_bytes": max(
                payload["peak_reserved_bytes"] for payload in gathered
            ),
            "peak_allocated_gib": max(
                payload["peak_allocated_bytes"] for payload in gathered
            )
            / (1024**3),
            "peak_reserved_gib": max(
                payload["peak_reserved_bytes"] for payload in gathered
            )
            / (1024**3),
            "gpu_utilization_mean": statistics.fmean(
                value for payload in gathered for value in payload["gpu_utilization"]
            )
            if any(payload["gpu_utilization"] for payload in gathered)
            else None,
            "cpu_utilization_mean": statistics.fmean(
                value for payload in gathered for value in payload["cpu_utilization"]
            )
            if any(payload["cpu_utilization"] for payload in gathered)
            else None,
            "io_wait_mean": statistics.fmean(
                value for payload in gathered for value in payload["io_wait"]
            )
            if any(payload["io_wait"] for payload in gathered)
            else None,
            "scorer_processes_per_rank": self.scorer_processes_per_rank,
            "scorer_partitions_per_scene": self.scorer_partitions_per_scene,
            "peak_child_processes_per_rank": max(
                payload["peak_child_processes"] for payload in gathered
            ),
            "nonfinite_count": 0,
            "oom": False,
            "deadlock": False,
            "phase_seconds": {
                name: {
                    "mean": statistics.fmean(values),
                    "median": _percentile(values, 0.50),
                    "p90": _percentile(values, 0.90),
                }
                for name, values in sorted(by_metric.items())
                if name != "step_time"
            },
        }
        output = Path(self.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
        print("FORMAL_THROUGHPUT_METRICS " + json.dumps(metrics, sort_keys=True))

    def on_exception(self, trainer, pl_module, exception: BaseException) -> None:
        del pl_module
        self._exception = exception
        if not trainer.is_global_zero:
            return
        message = f"{type(exception).__name__}: {exception}"
        lowered = message.lower()
        metrics = {
            "schema_version": 1,
            "layout": self.layout_name,
            "status": "failed",
            "warmup_steps": self.warmup_steps,
            "timed_optimizer_steps": len(self._records),
            "requested_timed_optimizer_steps": self.timed_steps,
            "world_size": int(getattr(trainer, "world_size", 1)),
            "global_batch_size": self.global_batch_size,
            "num_workers_per_rank": self.num_workers,
            "gradient_checkpointing": self.gradient_checkpointing,
            "read_only_attention_backend": self.read_only_attention_backend,
            "scorer_processes_per_rank": self.scorer_processes_per_rank,
            "scorer_partitions_per_scene": self.scorer_partitions_per_scene,
            "nonfinite_count": int(
                "non-finite" in lowered or "nonfinite" in lowered
            ),
            "oom": "out of memory" in lowered,
            "deadlock": False,
            "exception": message,
        }
        output = Path(self.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)


__all__ = ["FormalThroughputBenchmarkCallback"]
