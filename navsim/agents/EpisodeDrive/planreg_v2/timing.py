"""CUDA event timing without per-stage synchronizations; resolved at optimizer boundary."""
from contextlib import contextmanager
import time
import torch


class StepTiming:
    def __init__(self):self.events=[];self.cpu={}

    @contextmanager
    def stage(self,name,cuda=True):
        if cuda and torch.cuda.is_available():
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            try:yield
            finally:end.record();self.events.append((name,start,end))
        else:
            start=time.perf_counter()
            try:yield
            finally:self.cpu[name]=self.cpu.get(name,0.)+time.perf_counter()-start

    def collect(self):
        # Caller has synchronized the optimizer step once, not once per stage.
        result=dict(self.cpu)
        for name,start,end in self.events:result[name]=result.get(name,0.)+start.elapsed_time(end)/1000.
        self.events.clear();self.cpu.clear()
        return result
