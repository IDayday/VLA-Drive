"""Low-overhead CUDA/CPU phase timing used only by formal throughput smoke."""

from __future__ import annotations

from collections import defaultdict
import time
from typing import DefaultDict, Dict, List, Optional, Tuple

import torch


class PhaseTimer:
    def __init__(self, enabled: bool = False):
        self.enabled = bool(enabled)
        self._cuda_events: DefaultDict[str, List[Tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)
        self._cpu_seconds: DefaultDict[str, List[float]] = defaultdict(list)

    def start(self, name: str, *, cuda: bool = True):
        if not self.enabled:
            return None
        if cuda and torch.cuda.is_available():
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return ("cuda", name, event)
        return ("cpu", name, time.perf_counter())

    def stop(self, token) -> None:
        if token is None:
            return
        kind, name, start = token
        if kind == "cuda":
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self._cuda_events[name].append((start, end))
        else:
            self._cpu_seconds[name].append(time.perf_counter() - start)

    def add_seconds(self, name: str, seconds: float) -> None:
        if self.enabled:
            self._cpu_seconds[name].append(float(seconds))

    def consume(self) -> Dict[str, float]:
        if not self.enabled:
            return {}
        if self._cuda_events and torch.cuda.is_available():
            torch.cuda.synchronize()
        result: Dict[str, float] = {}
        for name, pairs in self._cuda_events.items():
            result[name] = result.get(name, 0.0) + sum(
                start.elapsed_time(end) / 1000.0 for start, end in pairs
            )
        for name, values in self._cpu_seconds.items():
            result[name] = result.get(name, 0.0) + sum(values)
        self._cuda_events.clear()
        self._cpu_seconds.clear()
        return result


__all__ = ["PhaseTimer"]
