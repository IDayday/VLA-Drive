"""Finite phase profiling; measurements never modify the training objective.

Phase CPU/device times are inclusive profiler intervals, not exclusive GPU busy
time and not quantities that may be added across nested phases. End-to-end wall
and shared worker/device occupancy are reported separately.
"""
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import statistics
import time

import torch


def _atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temporary.open('w') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class PhaseProfiler:
    """Use around an existing train loop, calling step after each valid update.

    wait1/warmup1/active2 records updates3/4 with one finite trace. Rows retain
    all observed effective updates so initialization, warmup and tracing are not
    silently confused with a long-run steady-state throughput measurement.
    """

    def __init__(self, output_dir, *, rank=None, wait=1, warmup=1, active=2,
                 enabled=True, use_cuda=None, metadata=None):
        if min(wait, warmup) < 0 or active < 1:
            raise ValueError('Invalid finite profiler schedule')
        self.output = Path(output_dir)
        self.rank = int(os.environ.get('RANK', 0) if rank is None else rank)
        self.wait, self.warmup, self.active = int(wait), int(warmup), int(active)
        self.enabled = bool(enabled)
        self.use_cuda = torch.cuda.is_available() if use_cuda is None else bool(use_cuda)
        if self.enabled and self.use_cuda and not torch.cuda.is_available():
            raise RuntimeError('CUDA profiling requested without an available CUDA device')
        self.metadata = dict(metadata or {})
        self.rows, self.windows = [], []
        self.profiler = None
        self.started = None

    def __enter__(self):
        self.output.mkdir(parents=True, exist_ok=True)
        if (self.output / f'PROFILE_rank{self.rank:02d}.json').exists():
            raise FileExistsError('Profiles must use a fresh invocation output directory')
        self.started = time.monotonic()
        if self.enabled:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if self.use_cuda:
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            self.profiler = torch.profiler.profile(activities=activities,
                schedule=torch.profiler.schedule(wait=self.wait, warmup=self.warmup,
                    active=self.active, repeat=1), on_trace_ready=self._on_trace,
                record_shapes=False, profile_memory=True, with_stack=False)
            self.profiler.__enter__()
        return self

    def phase(self, name):
        if not isinstance(name, str) or not name:
            raise ValueError('A nonempty named profiling phase is required')
        return torch.profiler.record_function('compare/' + name) if self.enabled else nullcontext()

    def step(self, *, update, scenes=32, wall_seconds, data_wait_seconds=None,
             optimizer_seconds=None, **extra):
        if self.started is None:
            raise RuntimeError('Enter the profiler before recording updates')
        if (type(update) is not int or update < 1 or
                self.rows and update != self.rows[-1]['update'] + 1):
            raise ValueError('Profiler updates must be a continuous effective optimizer sequence')
        if type(scenes) is not int or scenes <= 0:
            raise ValueError('scenes means the logical GLOBAL scene exposure of this update')
        for value in (wall_seconds, data_wait_seconds, optimizer_seconds):
            if value is not None and (not math.isfinite(float(value)) or float(value) < 0):
                raise ValueError('Profiler timing must be finite nonnegative seconds')
        row = dict(update=update, global_scenes=scenes, wall_seconds=float(wall_seconds),
                   data_wait_seconds=data_wait_seconds, optimizer_seconds=optimizer_seconds, **extra)
        if self.use_cuda:
            row.update(peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                       peak_reserved_bytes=int(torch.cuda.max_memory_reserved()))
        else:
            row.update(peak_allocated_bytes=None, peak_reserved_bytes=None)
        self.rows.append(row)
        if self.profiler is not None:
            self.profiler.step()

    def _on_trace(self, profiler):
        trace = self.output / f'rank{self.rank:02d}_trace{len(self.windows):02d}.json'
        profiler.export_chrome_trace(str(trace))
        phases, communication = {}, {}
        for event in profiler.key_averages():
            if not event.key.startswith('compare/') and not any(
                    term in event.key.lower() for term in ('nccl', 'allreduce', 'all_reduce',
                                                           'reduce_scatter', 'allgather', 'all_gather')):
                continue
            record = dict(calls=int(event.count), cpu_inclusive_ms=float(event.cpu_time_total) / 1000.,
                cpu_self_ms=float(event.self_cpu_time_total) / 1000.,
                device_inclusive_ms=float(getattr(event, 'device_time_total', 0.)) / 1000.
                    if self.use_cuda else None,
                device_self_ms=float(getattr(event, 'self_device_time_total', 0.)) / 1000.
                    if self.use_cuda else None)
            (phases if event.key.startswith('compare/') else communication)[event.key] = record
        self.windows.append(dict(trace=str(trace), phases=phases,
                                 communication_operations=communication,
                                 active_updates=self.active))

    def __exit__(self, kind, error, traceback):
        profile_error = None
        try:
            if self.profiler is not None:
                self.profiler.__exit__(kind, error, traceback)
        except Exception as failure:
            profile_error = failure
        wall = time.monotonic() - self.started
        step_wall = sum(row['wall_seconds'] for row in self.rows)
        result = dict(schema='qwen_world_compare_phase_profile_v1',
            status='FAILED' if kind is not None or profile_error is not None else 'COMPLETE',
            rank=self.rank, metadata=self.metadata, CUDA_profiled=self.use_cuda and self.enabled,
            enabled=self.enabled, schedule=dict(wait=self.wait,warmup=self.warmup,active=self.active,repeat=1),
            observed_effective_updates=len(self.rows), updates=self.rows, windows=self.windows,
            interval_wall_seconds=wall, measured_update_wall_seconds=step_wall,
            global_scene_exposures=sum(row['global_scenes'] for row in self.rows),
            global_scenes_per_second=sum(row['global_scenes'] for row in self.rows)/step_wall if step_wall else None,
            shared_worker_GPU_wall_hours=wall/3600 if self.use_cuda else None,
            exclusive_GPU_busy_hours=None,
            timing_semantics='Profiler phase CPU/device durations are inclusive and may overlap; do not sum nested phases. '
                'Device intervals/worker wall under shared GPUs are not exclusive GPU busy time. '
                'This finite qualification includes profiler overhead and is not a steady-state training ETA.',
            error=None if error is None and profile_error is None else repr(error or profile_error))
        _atomic(self.output/f'PROFILE_rank{self.rank:02d}.json',result)
        if profile_error is not None and kind is None:
            raise profile_error
        return False


def aggregate_profiles(root, world_size=8, *, output=None):
    """Call only after owned workers exit; refuses missing/inconsistent ranks."""
    if type(world_size)is not int or world_size<1:
        raise ValueError('A positive actual world size is required')
    root=Path(root)
    paths=[root/f'PROFILE_rank{rank:02d}.json'for rank in range(world_size)]
    if any(not path.is_file()for path in paths):
        raise ValueError('All rank profiles are required; an incomplete matrix is not PASS')
    ranks=[json.loads(path.read_text())for path in paths]
    first=ranks[0]
    if any(row['status']!='COMPLETE' or row['rank']!=rank for rank,row in enumerate(ranks)):
        raise ValueError('Failed/misassigned rank in finite profile')
    sequence=[row['update']for row in first['updates']]
    if not sequence or any([row['update']for row in rank['updates']]!=sequence for rank in ranks):
        raise ValueError('Rank effective-update sequences differ')
    if any(rank['metadata']!=first['metadata'] for rank in ranks):
        raise ValueError('Rank profile identities differ')
    if any([row['global_scenes']for row in rank['updates']]!=[row['global_scenes']for row in first['updates']]for rank in ranks):
        raise ValueError('Global scene exposure differs by rank')
    slowest=[max(rank['updates'][index]['wall_seconds']for rank in ranks)for index in range(len(sequence))]
    logical_scenes=sum(row['global_scenes']for row in first['updates'])
    result=dict(schema='qwen_world_compare_distributed_profile_v1',status='COMPLETE',
        world_size=world_size,metadata=first['metadata'],observed_effective_updates=len(sequence),
        updates=sequence,logical_global_scene_exposures=logical_scenes,
        slowest_rank_update_wall_seconds=slowest,
        global_scenes_per_second=logical_scenes/sum(slowest)if sum(slowest)else None,
        median_update_wall_seconds=statistics.median(slowest),
        max_rank_interval_wall_seconds=max(row['interval_wall_seconds']for row in ranks),
        cumulative_worker_wall_seconds=sum(row['interval_wall_seconds']for row in ranks),
        exclusive_GPU_busy_hours=None,
        peak_allocated_bytes=max((step['peak_allocated_bytes']for rank in ranks for step in rank['updates']
                                  if step['peak_allocated_bytes']is not None),default=None),
        peak_reserved_bytes=max((step['peak_reserved_bytes']for rank in ranks for step in rank['updates']
                                 if step['peak_reserved_bytes']is not None),default=None),
        rank_profiles=[str(path)for path in paths],
        timing_semantics=first['timing_semantics'])
    if output is not None:_atomic(output,result)
    return result


@torch.no_grad()
def profile_inference(model,current_inputs,ego,initial_noise,*,repeats=3,warmup=1):
    """Current-only single clean-x K10; no reward/scorer/teacher is invoked."""
    from starVLA.model.modules.qwen_world_postcot.flow import clean_x_step
    from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
    device=initial_noise.device
    if repeats<1 or warmup<0:raise ValueError('Finite positive inference repeats required')
    rows=[]
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
    for iteration in range(warmup+repeats):
        sync();tick=time.monotonic()
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type in('cuda','cpu')):
            context=model.encode_current(current_inputs,ego);sync();prefill=time.monotonic()-tick
            start=time.monotonic();prefix=model.prepare_prefix(context,auxiliary=False);sync();prefix_time=time.monotonic()-start
            grid=torch.linspace(1,0,11,device=device,dtype=torch.float32);grid=5*grid/(1+4*grid)
            state=initial_noise.float();denoisers=[];events=[];start=time.monotonic()
            for sigma,nxt in zip(grid[:-1],grid[1:]):
                if device.type=='cuda':
                    begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record()
                else:begin=time.monotonic()
                predicted=model.denoise_x0(state,sigma.expand(len(state)),prefix)
                if device.type=='cuda':end.record();events.append((begin,end))
                else:denoisers.append(time.monotonic()-begin)
                state=clean_x_step(state,predicted,sigma,nxt)
            sync();K10=time.monotonic()-start
            if device.type=='cuda':denoisers=[begin.elapsed_time(end)/1000 for begin,end in events]
        requested=ActionCodec.denorm_odo(state)
        if not torch.isfinite(state).all()or not torch.isfinite(requested).all():
            raise FloatingPointError('Nonfinite deployment benchmark trajectory')
        if iteration>=warmup:rows.append(dict(prefill_seconds=prefill,prefix_seconds=prefix_time,
            denoiser_seconds=denoisers,K10_from_prefix_seconds=K10,
            segmented_end_to_end_seconds=time.monotonic()-tick))
    return dict(schema='qwen_world_compare_inference_profile_v1',status='COMPLETE',rows=rows,
        repeats=repeats,warmup=warmup,logical_single_candidate_inferences=warmup+repeats,
        denoiser_forward_calls=10*(warmup+repeats),scorer_calls=0,
        prefill_median_seconds=statistics.median(row['prefill_seconds']for row in rows),
        prefix_median_seconds=statistics.median(row['prefix_seconds']for row in rows),
        denoiser_median_seconds=statistics.median(value for row in rows for value in row['denoiser_seconds']),
        K10_from_prefix_median_seconds=statistics.median(row['K10_from_prefix_seconds']for row in rows),
        end_to_end_median_seconds=statistics.median(row['segmented_end_to_end_seconds']for row in rows),
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device))if device.type=='cuda'else None,
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device))if device.type=='cuda'else None,
        timing_semantics='Finite same-scene current-only benchmark with synchronization at prefill/prefix/end; '
            'per-denoiser CUDA stream intervals are not exclusive GPU busy time under sharing.')

