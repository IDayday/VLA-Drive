"""Opt-in measurements of the real profile run, without changing its RNG/loss.

CUDA events measure current-stream intervals, including host launch gaps. They
are not a kernel-only or mutually exclusive breakdown across NCCL streams.
"""
from functools import wraps
import time
import torch


class TrainingStageTiming:
    def __init__(self, model, engine=None):
        self.pending = []
        self.precision = []
        for name, owner, attribute in (
            ('Qwen_current_inclusive', model, 'encode_current'),
            ('Qwen_vision', model.qwen_vl_interface.model.model.visual, 'forward'),
            ('FM_supervision', model.action_model, 'forward'),
            ('proposal_10steps', model.action_model, 'predict_action'),
            ('current_geometry', model.geometry, 'forward'),
            ('future_space', model.future_space, 'forward'),
            ('FGTR', model.refiner, 'forward'),
            ('scene_heads', getattr(model, 'scene_semantics', None), 'forward'),
            ('backward_inclusive', engine, 'backward'),
            ('optimizer_step', engine, 'step'),
        ):
            if owner is None or (engine is None and name in ('FM_supervision', 'scene_heads')):
                continue
            original = getattr(owner, attribute)
            setattr(owner, attribute, self._wrap(name, original))

    def _wrap(self, name, original):
        @wraps(original)
        def measured(*args, **kwargs):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            amp_enabled = torch.is_autocast_enabled('cuda')
            started = time.perf_counter()
            start.record()
            output = original(*args, **kwargs)
            end.record()
            self.pending.append((name, start, end, time.perf_counter()-started))
            if name == 'Qwen_vision':
                self.precision.append({'autocast_enabled': amp_enabled,
                    'first_output_dtype': str(output[0].dtype)})
            return output
        return measured

    def consume_after_synchronize(self):
        result = {}
        for name, start, end, host in self.pending:
            row = result.setdefault(name, {'calls': 0, 'CUDA_stream_ms': 0., 'host_dispatch_seconds': 0.})
            row['calls'] += 1
            row['CUDA_stream_ms'] += start.elapsed_time(end)
            row['host_dispatch_seconds'] += host
        result['observed_Qwen_vision_precision'] = self.precision
        self.pending, self.precision = [], []
        return result
