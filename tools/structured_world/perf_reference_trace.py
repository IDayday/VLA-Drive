"""Bounded profiler adapter around the frozen, accepted IO trainer.

This does not change its optimizer, data, random streams or model operators.
Run with torchrun and ordinary train_vla arguments following --trainer.
Optional --extended-stages requires trainer --profile-stages and adds host
scopes/CPU rows. All GPU/NCCL/sync operators remain the original Kineto events.
"""
import argparse
from functools import wraps
import importlib
import json
import os
from pathlib import Path
import runpy
import sys
import threading
import time
import torch


EXTENDED_STAGE_SCOPES = {
    'Qwen_language_inclusive': 'Language forward, including its nested operators; not the whole encode_current.',
    'Qwen_processor_apply_chat_template_host': 'Processor call host wall; CPU preprocessing can run in a worker.',
    'Qwen_get_rope_index_host': 'RoPE metadata call host wall; original input device is unchanged.',
    'Qwen_build_batch_including_vision': '_build_qwen_batch includes vision extraction and input packing, not just processor.',
    'sample_queries_inclusive': 'Framework-imported query sampler, including nested sampling/validation.',
    'structured_aux_losses_inclusive': 'Framework-imported compute_structured_losses, including reductions.',
    'final_trajectory_loss_inclusive': 'Framework-imported final trajectory loss, including reductions.',
    'geometry_depth_loss_inclusive': 'geometry.view.get_depth_loss, including target processing/reductions.',
    'prepared_current_transfer_optimizer_batch': 'Original batched input/target transfers; host wall does not imply copy completion.',
    'execution_log_flush_inclusive': 'Original ExecutionLogWindow.flush, including existing sync/collectives/host copies/file IO.',
}


class _ExtendedStages:
    """Process-local record_function scopes plus payload-free, thread-safe CPU rows.

    No CUDA API/event or tensor inspection belongs here, including in workers.
    Rows cover all calls, not only the three active Kineto steps. Step hints may
    span updates for prefetched work; timestamps use this process's perf_counter.
    """
    def __init__(self, trace_dir, rank):
        self.trace_dir, self.rank = Path(trace_dir), rank
        self.rows, self._patches, self._keys = [], [], set()
        self.installed_labels = set()
        self._lock, self._local = threading.Lock(), threading.local()
        self._sequence = 0
        self.profile = None

    def _step(self):
        return getattr(self.profile, 'step_num', None)

    def _wrap(self, label, original):
        @wraps(original)
        def call(*args, **kwargs):
            stack = getattr(self._local, 'stack', None)
            if stack is None:
                stack = self._local.stack = []
            with self._lock:
                self._sequence += 1
                sequence = self._sequence
            parent = stack[-1] if stack else None
            stack.append(sequence)
            step_start, started = self._step(), time.perf_counter()
            raised = True
            try:
                with torch.profiler.record_function(label):
                    result = original(*args, **kwargs)
                    raised = False
                    return result
            finally:
                ended = time.perf_counter()
                stack.pop()
                row = {'label': label, 'call_index': sequence, 'parent_call_index': parent,
                    'thread_id': threading.get_ident(), 'thread_name': threading.current_thread().name,
                    'start_perf_counter_seconds': started, 'end_perf_counter_seconds': ended,
                    'host_wall_seconds': ended-started, 'raised': raised,
                    'profiler_step_at_start': step_start, 'profiler_step_at_end': self._step()}
                with self._lock:
                    self.rows.append(row)
        return call

    def _patch(self, owner, attribute, replacement):
        key = (id(owner), attribute)
        if key in self._keys:
            return
        owned = attribute in vars(owner)
        previous = vars(owner).get(attribute)
        setattr(owner, attribute, replacement)
        self._patches.append((owner, attribute, owned, previous))
        self._keys.add(key)

    def attach(self, model, framework, prepared, log_class):
        qwen = model.qwen_vl_interface.model.model
        for label, owner, attribute in (
            ('Qwen_language_inclusive', qwen.language_model, 'forward'),
            ('Qwen_processor_apply_chat_template_host', model.qwen_vl_interface.processor, 'apply_chat_template'),
            ('Qwen_get_rope_index_host', qwen, 'get_rope_index'),
            ('Qwen_build_batch_including_vision', model, '_build_qwen_batch'),
            ('sample_queries_inclusive', framework, 'sample_queries'),
            ('structured_aux_losses_inclusive', framework, 'compute_structured_losses'),
            ('final_trajectory_loss_inclusive', framework, 'final_trajectory_loss'),
            ('geometry_depth_loss_inclusive', model.geometry.view, 'get_depth_loss'),
            ('prepared_current_transfer_optimizer_batch', prepared, 'transfer_optimizer_batch'),
            ('execution_log_flush_inclusive', log_class, 'flush'),
        ):
            self._patch(owner, attribute, self._wrap(label, getattr(owner, attribute)))
            self.installed_labels.add(label)

    def install(self, timing_class):
        previous_init = timing_class.__init__
        @wraps(previous_init)
        def initialize(timing, model, *args, **kwargs):
            result = previous_init(timing, model, *args, **kwargs)
            # The model/framework already exist. Deferred imports contain only
            # helper definitions; trainer's later from-imports see these wrappers.
            framework = importlib.import_module('starVLA.model.framework.vla_structured_fgtr')
            prepared = importlib.import_module('starVLA.dataloader.structured_world.prepared_current')
            log = importlib.import_module('tools.structured_world.execution_log')
            self.attach(model, framework, prepared, log.ExecutionLogWindow)
            return result
        self._patch(timing_class, '__init__', initialize)

    def restore(self):
        for owner, attribute, owned, previous in reversed(self._patches):
            if owned:
                setattr(owner, attribute, previous)
            else:
                delattr(owner, attribute)
        self._patches.clear()
        self._keys.clear()

    def flush(self):
        # No inputs, outputs, tensor values or exception messages are serialized.
        header = {'schema': 'extended_profile_cpu_rows_v1', 'rank': self.rank,
            'rows_scope': 'all wrapped calls, including workers and calls outside active3',
            'clock': 'process perf_counter seconds; step hints are not input/update identities',
            'timing': 'inclusive host wall; nested/parallel ranges must not be summed; not kernel-only',
            'CUDA_events_added': 0, 'stage_scopes': EXTENDED_STAGE_SCOPES,
            'installed_stage_labels': sorted(self.installed_labels)}
        path = self.trace_dir/f'rank{self.rank}_extended_cpu.jsonl'
        with self._lock:
            rows = sorted(self.rows, key=lambda row: row['call_index'])
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as stream:
            for row in (header, *rows):
                stream.write(json.dumps(row)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainer', type=Path, required=True)
    parser.add_argument('--trace-dir', type=Path, required=True)
    parser.add_argument('--extended-stages', action='store_true',
        help='Add process-local processor/language/query/loss/transfer/log scopes and CPU rows.')
    args, remaining = parser.parse_known_args()
    if args.extended_stages and '--profile-stages' not in remaining:
        parser.error('--extended-stages requires the trainer --profile-stages flag')
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
    def export(profile):
        profile.export_chrome_trace(str(args.trace_dir/f'rank{rank}.json'))
        (args.trace_dir/f'rank{rank}_operators.txt').write_text(
            profile.key_averages().table(sort_by='self_cuda_time_total', row_limit=80))
    original_step = deepspeed.DeepSpeedEngine.step
    extended = _ExtendedStages(args.trace_dir, rank) if args.extended_stages else None
    previous_argv = sys.argv
    try:
        TrainingStageTiming._wrap = profiled_wrap
        if extended is not None:
            extended.install(TrainingStageTiming)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=5, warmup=1, active=3, repeat=1),
            on_trace_ready=export, record_shapes=True, with_stack=False,
        ) as profile:
            if extended is not None:
                extended.profile = profile
            @wraps(original_step)
            def step(engine, *a, **kw):
                result = original_step(engine, *a, **kw)
                profile.step()
                return result
            deepspeed.DeepSpeedEngine.step = step
            sys.argv = [str(args.trainer), *remaining]
            runpy.run_path(str(args.trainer), run_name='__main__')
    finally:
        deepspeed.DeepSpeedEngine.step = original_step
        TrainingStageTiming._wrap = previous_wrap
        sys.argv = previous_argv
        if extended is not None:
            extended.restore()
            if sys.exc_info()[0] is None:
                extended.flush()
            else:
                try:
                    extended.flush()
                except OSError:
                    # Preserve the actual trainer exception if sidecar IO fails.
                    print('Extended CPU rows could not be written; original exception preserved.', file=sys.stderr)


if __name__ == '__main__':
    main()
