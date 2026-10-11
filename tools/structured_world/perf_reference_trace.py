"""Bounded profiler adapter around the frozen, accepted IO trainer.

This does not change its optimizer, data, random streams or model operators.
Run with torchrun and ordinary train_vla arguments following --trainer.
"""
import argparse
from functools import wraps
import os
from pathlib import Path
import runpy
import sys
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainer', type=Path, required=True)
    parser.add_argument('--trace-dir', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    root = args.trainer.resolve().parents[2]
    sys.path.insert(0, str(root))
    from tools.structured_world.stage_timing import TrainingStageTiming
    import deepspeed
    rank = int(os.environ['RANK'])
    args.trace_dir.mkdir(parents=True, exist_ok=True)
    previous_wrap = TrainingStageTiming._wrap
    def profiled_wrap(self, name, original):
        measured = previous_wrap(self, name, original)
        @wraps(original)
        def call(*a, **kw):
            with torch.profiler.record_function(name):
                return measured(*a, **kw)
        return call
    TrainingStageTiming._wrap = profiled_wrap
    def export(profile):
        profile.export_chrome_trace(str(args.trace_dir/f'rank{rank}.json'))
        (args.trace_dir/f'rank{rank}_operators.txt').write_text(
            profile.key_averages().table(sort_by='self_cuda_time_total', row_limit=80))
    original_step = deepspeed.DeepSpeedEngine.step
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=5, warmup=1, active=3, repeat=1),
        on_trace_ready=export, record_shapes=True, with_stack=False,
    ) as profile:
        @wraps(original_step)
        def step(engine, *a, **kw):
            result = original_step(engine, *a, **kw)
            profile.step()
            return result
        deepspeed.DeepSpeedEngine.step = step
        sys.argv = [str(args.trainer), *remaining]
        try:
            runpy.run_path(str(args.trainer), run_name='__main__')
        finally:
            deepspeed.DeepSpeedEngine.step = original_step
            TrainingStageTiming._wrap = previous_wrap


if __name__ == '__main__':
    main()
