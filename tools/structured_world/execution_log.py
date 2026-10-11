"""Bounded detached GPU metric windows, flushed at recovery boundaries."""
import json
import time
import torch
from torch import distributed as dist


class ExecutionLogWindow:
    def __init__(self, output, world, interval):
        self.output, self.world, self.interval = output, world, interval
        self.rows = []
        self.started = None

    def begin(self):
        if not self.rows:
            torch.cuda.synchronize()
            self.started = time.monotonic()

    def append(self, record, scalars, weighted_keys, events):
        record = self._pack_record(record, scalars)
        # Per-rank batch has at most four scenes, including an unequal tail.
        # Empty-label rows may omit neighbor diagnostics; keep collective shape.
        prototype = next(iter(scalars.values()))
        for row in range(4):
            for name in ('fallback', 'neighbor_time_queries'):
                scalars.setdefault(f'diagnostic/record/rank0_query_strata/{row}/{name}', prototype.new_zeros(()))
        names = tuple(sorted(scalars))
        if self.rows and self.rows[0][1] != names:
            raise ValueError('Execution metric inventory changed within a window')
        packed = torch.stack([scalars[name].detach().float().reshape(()) for name in names])
        self.rows.append((record, names, packed, tuple(weighted_keys), events))

    def _pack_record(self, value, scalars, path='record'):
        if path.rsplit('/', 1)[-1] in ('fallback', 'neighbor_time_queries') and isinstance(value, int):
            value = next(iter(scalars.values())).new_tensor(value, dtype=torch.int64)
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError('Only scalar diagnostics belong in execution logs')
            key = 'diagnostic/'+path
            scalars[key] = value.detach()
            return {'_execution_scalar': key, 'dtype': str(value.dtype)}
        if isinstance(value, dict):
            return {k: self._pack_record(v, scalars, path+'/'+str(k)) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._pack_record(v, scalars, path+'/'+str(i)) for i, v in enumerate(value)]
        return value

    def _unpack_record(self, value, scalars):
        if isinstance(value, dict) and '_execution_scalar' in value:
            number = scalars[value['_execution_scalar']]
            return int(number) if 'int' in value['dtype'] else bool(number) if 'bool' in value['dtype'] else number
        if isinstance(value, dict):
            return {k: self._unpack_record(v, scalars) for k, v in value.items()}
        if isinstance(value, list):
            return [self._unpack_record(v, scalars) for v in value]
        return value

    def due(self):
        return len(self.rows) >= self.interval

    def flush(self):
        if not self.rows:
            return 0.
        window_started = self.started
        start_update, end_update = self.rows[0][0]['updates'], self.rows[-1][0]['updates']
        scene_count, update_count = sum(r[0]['batch_scenes'] for r in self.rows), len(self.rows)
        torch.cuda.synchronize()
        wall = torch.tensor(time.monotonic()-self.started, device='cuda', dtype=torch.float64)
        dist.all_reduce(wall, op=dist.ReduceOp.MAX)
        names = self.rows[0][1]
        values = torch.stack([row[2] for row in self.rows])
        safe = torch.isfinite(values[:, names.index('gradient_norm')]).all().to(torch.int32)
        dist.all_reduce(safe, op=dist.ReduceOp.MIN)
        if not safe:
            raise FloatingPointError('Nonfinite gradient norm; all ranks stop at the same safety boundary')
        weighted = [names.index('weighted/'+name) for name in self.rows[0][3]]
        reduced = values[:, weighted].contiguous()
        dist.all_reduce(reduced)
        values[:, weighted] = reduced/self.world
        local_events = torch.tensor([row[4][0].elapsed_time(row[4][1])/1000.
                                     for row in self.rows], device='cuda', dtype=torch.float64)
        dist.all_reduce(local_events, op=dist.ReduceOp.MAX)
        # One packed transfer per window, instead of one per individual metric.
        host = values.cpu().tolist()
        durations = local_events.cpu().tolist()
        cost = float(wall)
        if dist.get_rank() == 0:
            with (self.output/'metrics.jsonl').open('a') as stream:
                for (record, _, _, _, _), numbers, seconds in zip(self.rows, host, durations):
                    metrics = dict(zip(names, numbers))
                    record = self._unpack_record(record, metrics)
                    record.update(weighted_losses={k.removeprefix('weighted/'): v for k, v in metrics.items() if k.startswith('weighted/')},
                        rank0_raw_losses={k.removeprefix('raw/'): v for k, v in metrics.items() if k.startswith('raw/')},
                        rank0_effective_auxiliary_counts={k.removeprefix('count/'): v for k, v in metrics.items() if k.startswith('count/')},
                        rank0_activation_gradient_norms=({k.removeprefix('activation/'): v for k, v in metrics.items() if k.startswith('activation/')}
                            if record['activation_metrics_sampled'] else None),
                        rank0_proposal_out_of_range_steps=metrics['out_of_range'],
                        global_gradient_norm=metrics['gradient_norm'],
                        CUDA_current_stream_update_seconds_max_rank=seconds,
                        window_start_update=self.rows[0][0]['updates'],
                        window_end_update=self.rows[-1][0]['updates'],
                        window_seconds_max_rank=cost,
                        window_updates=len(self.rows),
                        timing_semantics='window wall includes IO/training; update CUDA intervals include host launch gaps, not kernel-only')
                    stream.write(json.dumps(record)+'\n')
            print(json.dumps({'updates': self.rows[-1][0]['updates'],
                'window_updates': len(self.rows), 'seconds_per_update': cost/len(self.rows),
                'global_scenes_per_second': sum(r[0]['batch_scenes'] for r in self.rows)/cost,
                'weighted_losses_last_update': record['weighted_losses']}), flush=True)
        self.rows = []
        self.started = None
        # Include collective/scalar transfer and normal metric-file overhead.
        # The final small window record itself is the only excluded operation.
        torch.cuda.synchronize()
        full = torch.tensor(time.monotonic()-window_started, device='cuda', dtype=torch.float64)
        dist.all_reduce(full, op=dist.ReduceOp.MAX)
        peaks = torch.tensor([torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()], device='cuda', dtype=torch.int64)
        ranks = [torch.empty_like(peaks) for _ in range(self.world)]
        dist.all_gather(ranks, peaks)
        actual = float(full)
        if dist.get_rank() == 0:
            record = {'start_update': start_update, 'end_update': end_update,
                'updates': update_count, 'scene_exposure': scene_count,
                'wall_seconds_max_rank_including_log_flush': actual,
                'global_scenes_per_second': scene_count/actual,
                'GPU_hours': actual*self.world/3600.,
                'update_current_stream_seconds_max_rank': durations,
                'all_rank_peak_allocated_reserved_bytes': [p.cpu().tolist() for p in ranks]}
            with (self.output/'performance_windows.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')
        return actual
