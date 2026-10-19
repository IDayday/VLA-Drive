"""Finite, isolated full-IL and deployment timing; never resumes a formal run.

Use a sealed IL model, fresh throwaway Adam state and real TRAIN observations.
Single GPU timings explain components; they do not estimate DDP by subtraction.
CUDA events measure elapsed time on the stream, including CPU launch gaps.
The optional Kineto trace separately attributes kernels and remains private.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import gc
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import yaml

from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.qwen_sourceflow.contracts import (
    ROOT, QWEN_PATH, configure_native_numerics, file_sha, require_dispatch,
)
from starVLA.rl.qwen_sourceflow.data import RawSceneData
from starVLA.rl.qwen_sourceflow.il_optimization import parameter_groups, set_stage
from starVLA.rl.qwen_sourceflow.loading import load_il
from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
from starVLA.model.modules.qwen_sourceflow.qwen_encoder import QwenObservationProcessor


class Timings:
    def __init__(self):
        self.events = []
        self.cpu = Counter()

    @contextmanager
    def region(self, name):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        before = time.perf_counter()
        start.record()
        with torch.profiler.record_function('qsf/' + name):
            yield
        end.record()
        self.cpu[name] += time.perf_counter() - before
        self.events.append((name, start, end))

    def result(self):
        elapsed = Counter()
        for name, start, end in self.events:
            elapsed[name] += start.elapsed_time(end) / 1000
        return dict(cuda_stream_seconds=dict(elapsed), host_region_seconds=dict(self.cpu))


def tensor_info(x):
    return dict(shape=list(x.shape), dtype=str(x.dtype), bytes=x.numel()*x.element_size())


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--updates', type=int, default=6)
    parser.add_argument('--accumulation', type=int, default=4)
    parser.add_argument('--trace', action='store_true')
    args = parser.parse_args()
    if min(args.warmup, args.updates, args.accumulation) < 1:
        raise ValueError('Positive finite qualification budget required')
    host = require_dispatch()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    configure_native_numerics()
    torch.set_num_threads(8)
    torch.cuda.set_device(0)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    started = time.perf_counter()
    cfg = yaml.safe_load(Path('configs/qwen_sourceflow/il.yaml').read_text())
    model, parent = load_il(args.checkpoint, qualification=True, device='cuda')
    model.qwen.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    set_stage(model, cfg['interface_updates'], cfg)
    optimizer = torch.optim.AdamW(parameter_groups(model, cfg), betas=tuple(cfg['betas']),
        eps=cfg['eps'], weight_decay=cfg['weight_decay'], foreach=False, fused=False)
    data = RawSceneData(ROOT/'data/TRAIN_v3', cohort='TRAIN')
    processor = QwenObservationProcessor(QWEN_PATH)
    groups = Counter()
    dtypes = Counter()
    for name, param in model.named_parameters():
        key = 'qwen_visual' if name.startswith('qwen.model.visual.') else (
            'qwen_text' if name.startswith('qwen.') else 'provider' if name.startswith('provider.') else 'action')
        groups[key] += param.numel()
        dtypes[str(param.dtype)] += param.numel()
    identity = dict(schema='native_efficiency_profile_v1', host=host,
        scope='single shared GPU; fresh throwaway optimizer; real TRAIN; no model publication',
        parent_model_sha256=parent['files']['model.pt'], parent_updates=parent['seal']['updates'],
        data_sha256=data.sha, script_sha256=file_sha(__file__), config=cfg,
        GPU=torch.cuda.get_device_name(), parameter_counts=dict(groups),
        parameter_dtypes=dict(dtypes), parameter_tensors=sum(1 for _ in model.parameters()),
        accumulation=args.accumulation, warmup_updates=args.warmup, measured_updates=args.updates,
        load_and_setup_seconds=time.perf_counter()-started)
    atomic_json(out/'identity.json', identity)
    counters = Counter(); handles = []
    modules = {'qwen_vision_first_block': model.qwen.model.visual.blocks[0],
        'qwen_text_first_block': model.qwen.model.language_model.layers[0],
        'provider_first_visual_projection': model.provider.visual_projections[0],
        'action_first_FFN': model.policy.action_expert.blocks[0].ffn}
    for key, module in modules.items():
        def count(_module, _inputs, _output, label=key):
            counters[label] += 1
        handles.append(module.register_forward_hook(count))
    records = []; cursor = 0; condition_info = None

    def update(accumulation, index, traced=False):
        nonlocal cursor, condition_info
        timing = Timings(); before = time.perf_counter(); counters.clear()
        with timing.region('zero_grad'):
            optimizer.zero_grad(set_to_none=True)
        losses = []; token_counts = []
        for micro in range(accumulation):
            with timing.region('data_processor_target'):
                token = data.tokens[data.index(cursor)]; cursor += 1
                observation = data.observation(token)
                batch = processor([observation])
                proprio = torch.tensor([observation['proprio']], device='cuda', dtype=torch.float32)
                target = ActionCodec.norm_odo(data.target(token)[None].cuda())
            with torch.autocast('cuda', dtype=torch.bfloat16):
                with timing.region('qwen_forward'):
                    encoded = model.qwen(batch)
                with timing.region('provider_forward'):
                    condition = model.provider(encoded, proprio)
                if condition_info is None:
                    condition_info = dict(image_grid_thw=batch['image_grid_thw'].tolist(),
                        sequence_tokens=int(batch['attention_mask'].sum()),
                        visual_tokens=int(encoded['visual_mask'].sum()),
                        text_tokens=int(encoded['text_mask'].sum()),
                        visual_K=tensor_info(condition['visual_kv'][0]['k']),
                        visual_V=tensor_info(condition['visual_kv'][0]['v']),
                        context=tensor_info(condition['context']),
                        all_layer_KV_bytes=sum(v.numel()*v.element_size()
                            for layer in condition['visual_kv'] for v in layer.values()),
                        selected_hidden_indices=model.provider.layer_mapping)
                token_counts.append(int(batch['attention_mask'].sum()))
                del encoded
                with timing.region('action_forward_FM'):
                    loss, _ = model.policy(condition, target)
                del condition
            if not torch.isfinite(loss):
                raise FloatingPointError('Invalid profile loss')
            with timing.region('backward_all'):
                (loss/accumulation).backward()
            losses.append(float(loss.detach()))
            del loss
        with timing.region('gradient_validation_and_clip'):
            grads = [p.grad for p in model.parameters() if p.requires_grad]
            if any(g is None for g in grads):
                raise ValueError('Missing profile gradient')
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                cfg['gradient_clip'], error_if_nonfinite=True, foreach=False)
            del grads
        with timing.region('AdamW_step'):
            optimizer.step()
        torch.cuda.synchronize()
        return dict(index=index, warmup=index < args.warmup, traced=traced,
            wall_seconds=time.perf_counter()-before, accumulation=accumulation,
            losses=losses, grad_norm=float(norm), tokens=token_counts,
            module_forward_calls=dict(counters), peak_GPU_bytes=torch.cuda.max_memory_allocated(),
            **timing.result())

    for index in range(args.warmup + args.updates):
        row = update(args.accumulation, index)
        records.append(row)
        atomic_json(out/'timings.json', dict(identity=identity, condition=condition_info, updates=records))
        print(json.dumps(row), flush=True)
    if args.trace:
        # Separate, one-scene update: profiler overhead is excluded from timing.
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA], record_shapes=True,
                profile_memory=False, with_stack=False, with_flops=True) as prof:
            traced = update(1, args.warmup+args.updates, traced=True)
        prof.export_chrome_trace(str(out/'trace.json'))
        events = []
        for event in prof.key_averages(group_by_input_shape=True):
            events.append(dict(name=event.key, count=event.count, input_shapes=event.input_shapes,
                self_CPU_us=event.self_cpu_time_total, CPU_us=event.cpu_time_total,
                self_device_us=event.self_device_time_total, device_us=event.device_time_total,
                FLOPs=event.flops))
        events.sort(key=lambda e:e['self_device_us'], reverse=True)
        atomic_json(out/'operators.json', dict(events=events, traced_update=traced))
    for handle in handles: handle.remove()
    optimizer.zero_grad(set_to_none=True)
    optimizer.state.clear(); del optimizer
    gc.collect(); torch.cuda.empty_cache()
    model.eval()
    observation = data.observation(data.tokens[data.index(0)])
    batch = processor([observation]); proprio = torch.tensor([observation['proprio']], device='cuda')
    deployment = []
    for index in range(4):
        timing=Timings(); before=time.perf_counter()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            with timing.region('qwen_forward'):
                encoded=model.qwen(batch)
            with timing.region('provider_forward'):
                condition=model.provider(encoded,proprio)
        del encoded
        root=torch.randn((1,8,3),device='cuda')
        with timing.region('ODE_10_steps_FP32'):
            model.policy.sample_ode(condition,root)
        del condition
        torch.cuda.synchronize()
        deployment.append(dict(warmup=index==0, wall_seconds=time.perf_counter()-before, **timing.result()))
    atomic_json(out/'COMPLETE.json', dict(status='PASS', identity=identity, condition=condition_info,
        updates=records, deployment=deployment, qualification_only=True,
        formal_parameters_and_RNG_untouched=True, wall_seconds=time.perf_counter()-started))


if __name__ == '__main__':
    main()
