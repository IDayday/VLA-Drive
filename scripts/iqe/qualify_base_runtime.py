"""Bounded real-checkpoint A/B test; never changes a formal run or its weights."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from omegaconf import OmegaConf
from iqe.contracts import require
from iqe.io import atomic_json, file_hash, read_json
from iqe.config import load_config
from iqe.resources import qualify_distributed
from iqe.query_base import build_query_framework
from iqe.training.base_trainer import original_dataset
from iqe.training.trainer import init_distributed, scheduler_factory
from iqe.training.checkpoint import capture_rng, restore_rng, configure_language_checkpointing
from iqe.training.execution import EpochPlan, TrainingPrefetch, FrozenVisionAhead, collective_check, finite_gradients
from iqe.losses import collective_error


def rng_identity():
    state = capture_rng()
    h = hashlib.sha256(repr(state['python']).encode())
    n = state['numpy']
    h.update(n[1].tobytes()); h.update(repr((n[0], *n[2:])).encode())
    h.update(state['cpu'].numpy().tobytes())
    for tensor in state['cuda']:
        h.update(tensor.cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int, default=8)
    p.add_argument('--local-checkpoint-root')
    p.add_argument('--deterministic-backward', action='store_true', help='controlled backend for exact A/B proof, never relabel legacy nondeterministic replay')
    p.add_argument('--variant', choices=('cpu_prefetch', 'vision_ahead', 'reference_again'), default='vision_ahead')
    a = p.parse_args()
    require(4 <= a.steps <= 32, 'qualification requires 4..32 updates per variant')
    if a.deterministic_backward:
        os.environ['FLASH_ATTENTION_DETERMINISTIC']='1'
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic=True
    resources = qualify_distributed('cuda')
    rank, world = init_distributed('cuda')
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    device = f'cuda:{local_rank}'
    config = load_config(a.config)
    torch.set_num_threads(config['execution']['num_threads'])
    torch.backends.cuda.matmul.allow_tf32 = False
    # Match the running trainer's original PyTorch defaults. Changing this flag
    # changes FP32 visual convolution outputs even with matmul TF32 disabled.
    torch.backends.cudnn.allow_tf32 = True
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    require(not (out / f'rank_{rank}.json').exists(), 'qualification output already exists')
    contract = read_json(config['s0']['contract'])
    receipt = read_json(a.checkpoint + '.COMPLETE.json')
    if rank == 0:
        from iqe.training.local_checkpoint import stage_checkpoint
        staged = str(stage_checkpoint(a.checkpoint, receipt['checksum'], a.local_checkpoint_root))
        if not a.local_checkpoint_root:
            require(file_hash(a.checkpoint) == receipt['checksum'], 'checkpoint checksum')
        atomic_json(out/'LAUNCH.json', {'checkpoint': a.checkpoint, 'receipt': receipt, 'steps_per_variant': a.steps,
            'resources': resources, 'optimizer_updates_written_to_formal_run': False, 'variant': a.variant,
            'criterion': 'exact full final parameter/gradient/optimizer/RNG equality; per-step losses/IDs/RNG equal',
            'execution_hash': file_hash(Path(__file__).resolve().parents[2]/'iqe/training/execution.py'),
            'backend_flags': {'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,
                              'cudnn_tf32':torch.backends.cudnn.allow_tf32,
                              'cudnn_benchmark':torch.backends.cudnn.benchmark,
                              'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
                              'cudnn_deterministic':torch.backends.cudnn.deterministic,
                              'flash_attention_deterministic':os.getenv('FLASH_ATTENTION_DETERMINISTIC','0')}})
    if world > 1: dist.barrier()
    payload = [staged if rank == 0 else None]
    if world > 1: dist.broadcast_object_list(payload,src=0)
    state = torch.load(payload[0], map_location='cpu', mmap=True, weights_only=False)
    require(state['dependencies']['world_size'] == world, 'qualification topology differs')
    cfg = state['dependencies']['config']
    random.seed(config['seed'] + rank); np.random.seed(config['seed'] + rank); torch.manual_seed(config['seed'])
    from iqe.query_base import bind_source
    bind_source(contract['source_root'], contract['source_commit'])
    dataset = original_dataset(config, contract, allow_partial=False)
    by_id = {x['token']: i for i, x in enumerate(dataset.index)}
    fit = state['fit_scene_ids']
    from tools.ddpolicy_vehicle.training_state import epoch_batches
    from tools.foresight.student_state import optimizer_batch_counts
    from starVLA.dataloader.foresight_dataset import collate_training
    records = {}
    reference = None
    comparison = None
    for variant in ('reference', a.variant):
        # AdamW keeps CPU step tensors by reference on load_state_dict. A fresh
        # private mmap prevents the preceding A/B arm from advancing their value
        # inside the next arm's in-memory checkpoint object.
        state = torch.load(payload[0], map_location='cpu', mmap=True, weights_only=False)
        # Each arm must have the same fresh DDP bucket lifecycle, including its
        # first-iteration bucket rebuild. Reusing a reducer is not an A/B control.
        random.seed(config['seed'] + rank); np.random.seed(config['seed'] + rank); torch.manual_seed(config['seed'])
        model = build_query_framework(OmegaConf.create(contract['source_config']), contract['query_architecture'],
                                      contract['source_root'], contract['source_commit']).float().to(device)
        configure_language_checkpointing(model)
        model.train()
        params = [p for p in model.parameters() if p.requires_grad]
        wrapped = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True) if world > 1 else model
        model.load_state_dict(state['model'], strict=True)
        optimizer = torch.optim.AdamW(params, lr=cfg['lr'], weight_decay=cfg['weight_decay'],
            betas=tuple(contract['source_config']['trainer']['optimizer']['betas']))
        scheduler = scheduler_factory(optimizer, cfg)
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        optimizer.zero_grad(set_to_none=True)
        restore_rng(state['rng_by_rank'][rank])
        step = state['optimizer_step']
        epoch, offset = state['epoch'], state['offset']
        plan = EpochPlan(len(fit), cfg['global_batch_size'], config['seed'], rank, world, epoch_batches)
        records[variant] = []
        vision = FrozenVisionAhead(model) if variant == 'vision_ahead' else None
        with ThreadPoolExecutor(max_workers=config['execution']['loader_workers'] or 1) as workers:
            def prepare(address):
                started = time.perf_counter()
                samples = list(workers.map(dataset.__getitem__, [by_id[fit[i]] for i in address.mine]))
                obs, targets = collate_training(samples)
                valid = torch.arange(len(address.mine)) < address.valid_count
                for key, value in targets.items():
                    if key.endswith('_valid') or key == 'future_valid': value[~valid] = False
                targets['iqe_sample_valid'] = valid
                loaded = time.perf_counter() - started
                return obs, targets, None, loaded
            def prepare_vision(batch):
                obs, targets, _, loaded = batch
                return obs, targets, vision.prepare(obs, cfg['micro_batch_size']), loaded
            if world > 1: dist.barrier()
            torch.cuda.synchronize()
            with TrainingPrefetch(plan.iter_from(epoch, offset, a.steps), a.steps, prepare,
                                  prepare_vision if vision else None, depth=1 if variant not in ('reference','reference_again') else 0) as ahead:
                for index in range(a.steps):
                    started = time.perf_counter(); wait_start = started
                    error = None
                    try: address, batch = ahead.pop()
                    except Exception as e: error = repr(e)
                    collective_check(error, device=device)
                    obs, targets, prepared, load_seconds = batch
                    targets['iqe_sample_valid'] = targets['iqe_sample_valid'].to(device)
                    wait_seconds = time.perf_counter() - wait_start
                    if variant in ('reference','reference_again'):
                        calls = [None] * world
                        if world > 1: dist.all_gather_object(calls, int(np.ceil(len(obs)/cfg['micro_batch_size'])))
                    counts = optimizer_batch_counts(targets, torch.Generator().manual_seed(config['seed'] + step*world + rank), device)
                    counts['ego_scenes'] = len(address.entries)
                    optimizer.zero_grad(set_to_none=True)
                    logs = {}
                    with vision.activate(prepared) if vision else nullcontext():
                        for at in range(0, len(obs), cfg['micro_batch_size']):
                            end = min(len(obs), at+cfg['micro_batch_size'])
                            with wrapped.no_sync() if world > 1 and end < len(obs) else nullcontext():
                                result = wrapped(obs[at:end], {k:v[at:end] for k,v in targets.items()}, completed_updates=step, global_counts=counts)
                                collective_check(None if bool(torch.isfinite(result['loss'])) else 'nonfinite loss', device=device)
                                result['loss'].backward()
                            for k,v in result['losses'].items(): logs[k] = logs.get(k,0.) + float(v.detach())
                    finite = finite_gradients(params) if variant not in ('reference','reference_again') else not any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params)
                    collective_check(None if finite else 'nonfinite gradient', device=device)
                    norm = torch.nn.utils.clip_grad_norm_(params, cfg['grad_clip_norm'])
                    optimizer.step(); scheduler.step(); step += 1
                    if world > 1: dist.all_reduce(torch.zeros((),device=device))
                    torch.cuda.current_stream().synchronize()
                    row = {'step':step, 'seconds':time.perf_counter()-started, 'wait_seconds':wait_seconds,
                        'loader_seconds':load_seconds, 'losses':logs, 'gradient_norm':float(norm),
                        'sample_ids':[fit[i] for i in address.mine], 'rng':rng_identity(),
                        'peak_bytes':torch.cuda.max_memory_allocated()}
                    records[variant].append(row)
                    if rank == 0: print(json.dumps({'variant':variant, **row}), flush=True)
        if vision: vision.close()
        if variant == 'reference':
            reference = {'model': {k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                'grad': {k:p.grad.detach().cpu().clone() for k,p in model.named_parameters() if p.grad is not None},
                'optimizer': {k:{n:(v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v))
                                 for n,v in values.items()} for k,values in optimizer.state_dict()['state'].items()},
                'scheduler': copy.deepcopy(scheduler.state_dict()), 'rng':rng_identity()}
        else:
            mismatches = []
            max_error = 0.
            def compare(label, actual, expected):
                nonlocal max_error
                if torch.is_tensor(actual):
                    actual = actual.detach().cpu()
                    same = torch.equal(actual, expected)
                    if not same:
                        error = float((actual.float()-expected.float()).abs().max())
                        max_error = max(max_error, error)
                else: same = actual == expected
                if not same: mismatches.append(label)
            for k,v in model.state_dict().items(): compare('parameter_or_buffer/'+k,v,reference['model'][k])
            for k,p in model.named_parameters():
                if p.grad is not None: compare('gradient/'+k,p.grad,reference['grad'][k])
            for k,values in optimizer.state_dict()['state'].items():
                for n,v in values.items(): compare(f'optimizer/{k}/{n}',v,reference['optimizer'][k][n])
            require(scheduler.state_dict() == reference['scheduler'], 'scheduler changed')
            if rng_identity() != reference['rng']: mismatches.append('RNG')
            for before,after in zip(records['reference'], records[variant]):
                for key in ('losses','sample_ids','rng','gradient_norm'):
                    if before[key] != after[key]: mismatches.append(f"step/{after['step']}/{key}")
            comparison = {'exact':not mismatches, 'mismatch_count':len(mismatches), 'mismatches':mismatches[:30], 'max_abs_error':max_error}
        del optimizer, scheduler, wrapped, params, model, result, targets, counts
        torch.cuda.reset_peak_memory_stats()
        import gc
        gc.collect(); torch.cuda.empty_cache()
    report = {'status':'PASS' if comparison['exact'] else 'FAIL', 'comparison':comparison, 'records':records}
    atomic_json(out/f'rank_{rank}.json', report)
    if world > 1: dist.barrier()
    if rank == 0:
        ranks = [read_json(out/f'rank_{r}.json') for r in range(world)]
        timing = {v:{'median_s':statistics.median(row['seconds'] for r in ranks for row in r['records'][v][2:-1]),
                     'p95_s':float(np.percentile([row['seconds'] for r in ranks for row in r['records'][v][2:-1]],95))}
                  for v in records}
        result = {'status':'PASS' if all(r['status']=='PASS' for r in ranks) else 'FAIL', 'timing':timing,
            'all_ranks':ranks, 'checkpoint':a.checkpoint, 'checkpoint_hash':receipt['checksum'], 'world_size':world,
            'global_batch':cfg['global_batch_size'], 'framework_contract_hash':contract['framework_contract_hash'],
            'execution_hash':file_hash(Path(__file__).resolve().parents[2]/'iqe/training/execution.py'),
            'backend_flags':read_json(out/'LAUNCH.json')['backend_flags'],
            'warmup_excluded':2, 'pipeline_drain_excluded':1,
            'no_formal_training_writeback':True, 'no_precision_change':True, 'science':'UNTESTED'}
        atomic_json(out/'RESULT.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='all_ranks'}), flush=True)
    if dist.is_initialized(): dist.destroy_process_group()


if __name__ == '__main__': main()
