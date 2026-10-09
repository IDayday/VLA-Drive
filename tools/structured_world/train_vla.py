"""Eight-GPU original-FM/structured-world trainer with complete ZeRO recovery."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import socket
import subprocess
import sys
import time
import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch import distributed as dist, nn
import deepspeed
from omegaconf import OmegaConf
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from starVLA.model.framework.vla_structured_fgtr import VLAStructuredFGTR
from starVLA.dataloader.structured_world.dataset import StructuredNAVSIMDataset, collate_structured
from starVLA.dataloader.structured_world.nuscenes_dataset import StructuredNuScenesDataset
from tools.ddpolicy_vehicle.optimizer_safety import bounded_parameter_groups, capture_master_samples, master_update_evidence
from tools.ddpolicy_vehicle.training_state import epoch_batches
from tools.foresight.student_state import capture_rng, restore_rng, learning_rate, validate_rank_batches, optimizer_batch_counts
from tools.structured_world.recovery_boundary import verify_boundary
from tools.structured_world.training_assets import initialization_assets, file_digest, verify_checkpoint_files
from starVLA.model.modules.structured_world.execution import configure_execution
from tools.structured_world.prefetch import OrderedScenePrefetch
from tools.structured_world.resume_origin import read_origin, verify_scientific_contract


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
    temporary.replace(path)


def source_hashes():
    paths = []
    for directory in ['starVLA/model', 'starVLA/dataloader',
                      'tools/ddpolicy_vehicle', 'tools/foresight', 'third_party']:
        paths += sorted((ROOT/directory).rglob('*.py'))
    paths.append(Path(__file__))
    paths.append(ROOT/'tools/structured_world/recovery_boundary.py')
    paths.append(ROOT/'tools/structured_world/training_assets.py')
    paths.append(ROOT/'tools/structured_world/stage_timing.py')
    paths.append(ROOT/'tools/structured_world/prefetch.py')
    paths.append(ROOT/'tools/structured_world/resume_origin.py')
    record = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scope', choices=['profile', 'small_fit', 'formal'], required=True)
    parser.add_argument('--debug-updates', type=int, default=0)
    parser.add_argument('--debug-scenes', type=int, default=0,
                        help='Fixed training-only engineering subset; forbidden in formal scope')
    parser.add_argument('--micro-batch', type=int, default=2)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int, default=0)
    parser.add_argument('--save-every', type=int, default=5000)
    parser.add_argument('--geometry', type=Path)
    parser.add_argument('--geometry-identity')
    parser.add_argument('--dino-root', type=Path)
    parser.add_argument('--dino-identity')
    parser.add_argument('--image-root', type=Path)
    parser.add_argument('--profile-stages', action='store_true',
                        help='Only profile scope: measure the real model and optimizer stages')
    parser.add_argument('--execution-mode', choices=('reference', 'loss_preserving_v1'), default='reference')
    parser.add_argument('--resume-origin-run', type=Path,
                        help='Paused parent of an execution-only upgrade; remains in every recovery contract')
    args = parser.parse_args()
    if args.profile_stages and args.scope != 'profile':
        raise ValueError('Extra stage instrumentation is forbidden in formal/small-fit runs')
    policy = json.loads(Path('/mnt/project/server_dispatch_policy.json').read_text())
    hosts = policy['task_authorizations']['structured_world_fgtr_round1']['allowed_hosts']
    canonical = socket.gethostname().removesuffix('-worker-0')
    if canonical not in hosts:
        raise ValueError('Host is not authorized for this task')
    rank, world, local = int(os.environ['RANK']), int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_RANK'])
    if world != 8 or args.micro_batch not in (1, 2, 4):
        raise ValueError('One complete eight-GPU host and registered microbatch required')
    torch.cuda.set_device(local)
    # The existing framework's distributed state can initialize NCCL on import.
    if not dist.is_initialized():
        dist.init_process_group('nccl')
    if dist.get_rank() != rank or dist.get_world_size() != world:
        raise ValueError('Imported distributed state differs from torchrun ranks')
    torch.set_num_threads(2)
    random.seed(42+rank); np.random.seed(42+rank); torch.manual_seed(42+rank); torch.cuda.manual_seed_all(42+rank)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    # Warn-only determinism does not select deterministic Flash backward in
    # Torch 2.5. Mature CUDA kernels can also retain atomic rounding noise;
    # common-prefix recovery measures this rather than claiming bitwise updates.
    torch.use_deterministic_algorithms(True, warn_only=True)
    config = OmegaConf.load(args.config)
    dataset = config.structured_world.dataset
    if dataset not in ('navsim', 'nuscenes'): raise ValueError('Registered dataset required')
    if args.scope == 'formal':
        if config.structured_world.registration_status != 'frozen_after_common_calibration_and_profile' or not args.geometry or args.debug_updates or args.debug_scenes:
            raise ValueError('Formal training requires frozen common recipe and full shared geometry')
        if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
            raise ValueError('Formal training source must be a clean committed worktree')
    elif args.debug_updates <= 0:
        raise ValueError('Engineering scopes require an explicit finite update horizon')
    if dataset == 'navsim':
        data = StructuredNAVSIMDataset(args.cache, '/var/tmp/ddp-full-foresight-20260929/student_train_v1',
            dino_root=args.dino_root or '/var/tmp/ddp-full-foresight-20260929/targets/C1',
            dino_index='/mnt/project/ddp-full-foresight-study-artifacts/20260929/dino_index_v1',
            expected_dino=args.dino_identity or '7663c45b77dd711e8304e8d202644dcaa8467f7e265ff8b4c1816d59d2c131cb',
            image_root=args.image_root or '/var/tmp/ddp-full-foresight-20260929/images', allow_debug=args.scope != 'formal')
        current_identity, dino_identity = data.base.identity, data.base.dino_identity['identity']
    else:
        if not args.dino_root or not args.dino_identity: raise ValueError('Verified six-camera C1 cache required')
        data = StructuredNuScenesDataset(args.cache, dino_root=args.dino_root, expected_dino=args.dino_identity,
                                        image_root=args.image_root, allow_debug=args.scope != 'formal')
        current_identity, dino_identity = data.identity['source_current_identity'], data.dino_identity['identity']
    if args.scope == 'formal':
        expected_assets = dict(config.structured_world.formal_recipe_assets)
        observed_assets = {'label_cache_identity': data.identity['identity'],
                           'current_DINO_identity': dino_identity,
                           'shared_geometry_identity': args.geometry_identity}
        if observed_assets != expected_assets:
            raise ValueError('Formal dataset/teacher/geometry differs from the frozen common recipe')
    original_population = len(data)
    training_selection = None
    if args.debug_scenes:
        if not 32 <= args.debug_scenes < len(data):
            raise ValueError('Engineering selection requires at least one batch and fewer scenes than full population')
        indices = sorted(np.random.default_rng(42).choice(len(data), args.debug_scenes, replace=False).tolist())
        training_selection = {'population_kind': 'fixed_training_debug_subset_not_formal', 'selection_seed': 42,
            'source_scene_count': original_population, 'source_indices': indices,
            'tokens': [data.index[index]['token'] for index in indices]}
        subset = torch.utils.data.Subset(data, indices)
        subset.identity = data.identity
        data = subset
    validate_rank_batches(len(data), 32, world, args.micro_batch)
    initialization = [initialization_assets(config, args.geometry, args.geometry_identity, data.identity['identity'],
                           formal=args.scope == 'formal') if rank == 0 else None]
    dist.broadcast_object_list(initialization, src=0)
    updates_per_epoch = math.ceil(len(data)/32)
    horizon = (100000 if dataset == 'navsim' else 24*updates_per_epoch) if args.scope == 'formal' else args.debug_updates
    warmup = (5000 if dataset == 'navsim' else updates_per_epoch) if args.scope == 'formal' else max(1, horizon//10)
    contract = {'schema': 'structured_world_student_v1', 'scope': args.scope,
        'training_source_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'training_source_tree_sha256': source_hashes(), 'external_source_versions': json.loads((ROOT/'third_party/LOCK.json').read_text()),
        'external_source_addendum': json.loads((ROOT/'third_party/ADDENDUM_LOCK.json').read_text()),
        'config': OmegaConf.to_container(config, resolve=True), 'cache_identity': data.identity['identity'],
        'current_and_GT_identity': current_identity, 'DINO_identity': dino_identity,
        'shared_geometry_identity': args.geometry_identity, 'training_seed': 42, 'world_size': world,
        'initialization_assets': initialization[0],
        'effective_global_batch': 32, 'FM_repeat': 8, 'microbatch': args.micro_batch,
        'schedule_horizon': horizon, 'schedule_warmup': warmup,
        'dataset': dataset, 'scenes_per_epoch': len(data), 'updates_per_epoch': updates_per_epoch,
        'source_population_scenes': original_population, 'engineering_training_selection': training_selection,
        'observation_plan': [25000, 50000, 75000, 100000] if dataset == 'navsim' else [6, 12, 18, 24],
        'determinism': {'torch_algorithms': 'warn_only', 'cudnn_deterministic': True,
            'CUBLAS_WORKSPACE_CONFIG': os.environ['CUBLAS_WORKSPACE_CONFIG'],
            'native_CUDA_atomic_rounding': 'measured with common-prefix recovery'},
        'precision': 'FP32 parameters/master/optimizer; BF16 Qwen vision/language and original action compute; FP32 geometry/refiner; TF32 off',
        'extra_stage_instrumentation': args.profile_stages}
    contract['execution_mode'] = args.execution_mode
    resume_origin = None
    if args.resume_origin_run is not None:
        if args.scope != 'formal':
            raise ValueError('Execution ancestry is for preserved formal runs, not engineering initialization')
        resume_origin, parent_contract, parent_complete = read_origin(args.resume_origin_run)
        verify_scientific_contract(parent_contract, contract)
        contract['execution_upgrade_origin'] = resume_origin
    identity = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'checkpoints/latest').exists() and not args.resume:
        raise ValueError('Existing recovery checkpoint requires --resume; no implicit reinitialization')
    if rank == 0:
        if (args.output/'identity.json').exists() and json.loads((args.output/'identity.json').read_text())['identity'] != identity:
            raise ValueError('Run directory belongs to another scientific/source contract')
        atomic_json(args.output/'identity.json', {'identity': identity, **contract})
    dist.barrier()
    model = VLAStructuredFGTR(config).float()
    if args.geometry:
        model.load_shared_geometry(args.geometry, args.geometry_identity, allow_debug=args.scope != 'formal')
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda().train()
    execution_contract = configure_execution(model, args.execution_mode)
    if rank == 0:
        atomic_json(args.output/'execution_policy.json', execution_contract)
    rates = dict(config.trainer.learning_rate)
    groups, group_records = bounded_parameter_groups(model.named_parameters(), learning_rates=rates)
    if resume_origin is not None:
        if json.loads((args.resume_origin_run/'optimizer_groups.json').read_text()) != group_records:
            raise ValueError('Execution migration changed actual optimizer groups or base learning rates')
    ds_config = {'train_micro_batch_size_per_gpu': args.micro_batch, 'gradient_accumulation_steps': 1,
        'train_batch_size': args.micro_batch*world, 'bf16': {'enabled': False}, 'fp16': {'enabled': False},
        'gradient_clipping': float(config.trainer.gradient_clipping), 'steps_per_print': 1000000,
        'zero_optimization': {'stage': 2, 'contiguous_gradients': True, 'overlap_comm': True,
            'reduce_scatter': True, 'reduce_bucket_size': 100000000, 'allgather_bucket_size': 100000000},
        'optimizer': {'type': 'Adam', 'params': {'lr': float(rates['base']), 'betas': [.9, .95],
                     'eps': 1e-8, 'weight_decay': .001, 'adam_w_mode': True}}}
    engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=groups, config=ds_config)
    stage_timing = None
    if args.profile_stages:
        from tools.structured_world.stage_timing import TrainingStageTiming
        stage_timing = TrainingStageTiming(model, engine)
    generators = {name: torch.Generator(device='cuda').manual_seed(420000+rank+offset) for name, offset in
                  [('noise', 100), ('time', 200), ('proposal', 300), ('queries', 400)]}
    generators['counts'] = torch.Generator().manual_seed(420500+rank)
    completed, epoch, offset, exposure, elapsed = 0, 0, 0, 0, 0.
    if args.resume or resume_origin is not None:
        recovery_run = args.output if args.resume else args.resume_origin_run
        tag = (recovery_run/'checkpoints/latest').read_text().strip()
        if Path(tag).name != tag:
            raise ValueError('Unsafe checkpoint pointer')
        recovery_folder = recovery_run/'checkpoints'/tag
        complete = json.loads((recovery_folder/'COMPLETE.json').read_text())
        expected_identity = identity if args.resume else resume_origin['identity']
        if complete['identity'] != expected_identity:
            raise ValueError('Incomplete/foreign recovery checkpoint')
        verify_checkpoint_files(recovery_folder, complete, rank=rank, world=world)
        dist.barrier()
        _, state = engine.load_checkpoint(str(recovery_run/'checkpoints'), tag=tag,
            load_module_strict=True, load_optimizer_states=True, load_lr_scheduler_states=True)
        completed, epoch, offset, exposure, elapsed = [state[k] for k in ('completed', 'epoch', 'offset', 'exposure', 'elapsed')]
        restore_rng(torch.load(recovery_folder/f'rng_rank{rank}.pt', map_location='cpu', weights_only=False), generators)
        boundary = verify_boundary(engine, recovery_folder, generators)
        if rank == 0:
            atomic_json(args.output/'RESUME_BOUNDARY_VERIFIED.json', {'identity': identity,
                'completed': completed, 'epoch': epoch, 'offset': offset, 'exposure': exposure, **boundary})
    if rank == 0:
        atomic_json(args.output/'optimizer_groups.json', group_records)
    milestones = ({25000, 50000, 75000, 100000} if dataset == 'navsim' else
                  {epoch*updates_per_epoch for epoch in (6, 12, 18, 24)}) if args.scope == 'formal' else {horizon}
    def save(status):
        nonlocal elapsed
        tag = f'milestone_{completed:07d}' if completed in milestones else f'latest_{completed:07d}'
        staging = 'writing_'+tag
        checkpoint_root = args.output/'checkpoints'
        if shutil.disk_usage(args.output).free < 100*(1 << 30):
            raise RuntimeError('Storage reserve below 100 GiB; preserving existing recovery checkpoints')
        state = {'identity': identity, 'completed': completed, 'epoch': epoch, 'offset': offset,
                 'exposure': exposure, 'elapsed': elapsed, 'scheduler_horizon': horizon,
                 'shared_geometry_identity': model.shared_geometry_identity}
        start = time.monotonic()
        engine.save_checkpoint(str(checkpoint_root), tag=staging, client_state=state, save_latest=False)
        torch.save(capture_rng(generators), checkpoint_root/staging/f'rng_rank{rank}.pt')
        dist.barrier()
        owned = [checkpoint_root/staging/f'rng_rank{rank}.pt',
                 checkpoint_root/staging/f'zero_pp_rank_{rank}_mp_rank_00_optim_states.pt']
        if rank == 0: owned.append(checkpoint_root/staging/'mp_rank_00_model_states.pt')
        local_hashes = {p.name: file_digest(p) for p in owned}
        rank_hashes = [None]*world; dist.all_gather_object(rank_hashes, local_hashes)
        hashes = {name: value for record in rank_hashes for name, value in record.items()}
        save_cost = torch.tensor(time.monotonic()-start, device='cuda'); dist.all_reduce(save_cost, op=dist.ReduceOp.MAX)
        # Recovery stores training elapsed separately; checkpoint wall time is
        # recorded per complete checkpoint for total cost accounting.
        if rank == 0:
            folder = checkpoint_root/tag
            if folder.exists():
                raise ValueError('Refusing to overwrite an existing checkpoint')
            (checkpoint_root/staging).rename(folder)
            atomic_json(folder/'COMPLETE.json', {'identity': identity, 'tag': tag, 'completed': completed,
                'ranks': world, 'status': status, 'save_seconds': float(save_cost),
                'checkpoint_GPU_hours': float(save_cost)*world/3600., 'file_sha256': hashes})
            temp = checkpoint_root/'latest.tmp'; temp.write_text(tag+'\n'); temp.replace(checkpoint_root/'latest')
            rolling = sorted(checkpoint_root.glob('latest_*'))
            for old in rolling[:-2]:
                old_identity = json.loads((old/'COMPLETE.json').read_text())['identity']
                if old_identity != identity:
                    raise ValueError('Foreign checkpoint found during retention')
                shutil.rmtree(old)
        dist.barrier()
    if not completed:
        save('INITIAL')
    elif resume_origin is not None and not args.resume:
        # Own an atomic recovery point before executing the first upgraded step.
        save('RUNNING')
    optimized_execution = args.execution_mode == 'loss_preserving_v1'
    batch_cache = {}
    def batches_for(current_epoch):
        if not optimized_execution:
            return epoch_batches(len(data), 32, 42, current_epoch)
        if current_epoch not in batch_cache:
            batch_cache[current_epoch] = epoch_batches(len(data), 32, 42, current_epoch)
        for previous in list(batch_cache):
            if previous < current_epoch-1:
                del batch_cache[previous]
        return batch_cache[current_epoch]
    with ThreadPoolExecutor(max_workers=2) as pool, \
            (OrderedScenePrefetch(data, collate_structured, pool) if optimized_execution else nullcontext()) as prefetch:
        while completed < horizon:
            batches = batches_for(epoch)
            if offset >= len(batches):
                epoch += 1; offset = 0; continue
            start = time.monotonic()
            indices = batches[offset][rank::world]
            if prefetch is None:
                samples = list(pool.map(data.__getitem__, indices))
                observations, targets = collate_structured(samples)
            else:
                observations, targets = prefetch.consume((epoch, offset), indices)
                if completed+1 < horizon:
                    next_epoch, next_offset = epoch, offset+1
                    if next_offset == len(batches):
                        next_epoch, next_offset = epoch+1, 0
                    prefetch.submit((next_epoch, next_offset), batches_for(next_epoch)[next_offset][rank::world])
            data_seconds = time.monotonic()-start
            counts = optimizer_batch_counts(targets, generators['counts'], torch.device('cuda', local))
            actual_lrs = []
            for group, record in zip(engine.optimizer.param_groups, group_records):
                base = float(record['base_lr'])
                group['lr'] = learning_rate(completed, base, base*.05, warmup, horizon)
                actual_lrs.append(group['lr'])
            logs, raw, effective, query_records, activation_gradients = {}, {}, {}, [], []
            proposal_out_of_range = 0
            for at in range(0, len(indices), args.micro_batch):
                end = min(len(indices), at+args.micro_batch); boundary = end == len(indices)
                engine.set_gradient_accumulation_boundary(boundary)
                output = engine(observations[at:end], {k: v[at:end] for k, v in targets.items()},
                    completed_updates=completed, noise_generator=generators['noise'], time_generator=generators['time'],
                    proposal_generator=generators['proposal'], query_generator=generators['queries'], global_counts=counts)
                if not torch.isfinite(output['loss']):
                    raise FloatingPointError('Nonfinite full objective')
                engine.backward(output['loss'], scale_wrt_gas=False)
                if optimized_execution:
                    # Queue the unchanged optimizer before transferring detached
                    # diagnostic scalars to the host. Preserve the before/after
                    # FP32 master sample boundary and the original loss graph.
                    if boundary: before = capture_master_samples(engine.optimizer)
                    engine.step()
                    if boundary: evidence = master_update_evidence(engine.optimizer, before, torch.device('cuda', local))
                for name, value in output['losses'].items(): logs[name] = logs.get(name, 0.)+float(value.detach())
                for name, value in output['metrics']['raw'].items(): raw[name] = raw.get(name, 0.)+float(value.detach())
                for name, value in output['metrics']['effective_counts'].items():
                    effective[name] = effective.get(name, 0.)+float(value.sum())
                query_records.extend(output['metrics']['query_strata'])
                if optimized_execution:
                    # A single scalar transfer replaces three blocking transfers
                    # inside the autograd hooks. These are detached observations.
                    norms = output['metrics']['activation_gradient_norms']
                    names = sorted(norms)
                    values = torch.stack([norms[name] for name in names]).cpu().tolist() if names else []
                    activation_gradients.append(dict(zip(names, values)))
                else:
                    activation_gradients.append(output['metrics']['activation_gradient_norms'])
                proposal_out_of_range += int(output['metrics']['proposal_out_of_range'].sum())
                if not optimized_execution:
                    if boundary: before = capture_master_samples(engine.optimizer)
                    engine.step()
                    if boundary: evidence = master_update_evidence(engine.optimizer, before, torch.device('cuda', local))
                del output
            torch.cuda.synchronize()
            cost = torch.tensor(time.monotonic()-start, device='cuda'); dist.all_reduce(cost, op=dist.ReduceOp.MAX)
            stage_records = None
            if stage_timing is not None:
                local_stages = stage_timing.consume_after_synchronize()
                local_stages['data_load_and_collate_seconds'] = data_seconds
                stage_records = [None]*world
                dist.all_gather_object(stage_records, local_stages)
            elapsed += float(cost); completed += 1; exposure += len(batches[offset]); offset += 1
            keys = sorted(logs); packed = torch.tensor([logs[k] for k in keys], device='cuda'); dist.all_reduce(packed); packed /= world
            if rank == 0:
                gradient_norm = float(engine.get_global_grad_norm())
                clip_limit = float(config.trainer.gradient_clipping)
                record = {'updates': completed, 'epoch': epoch, 'offset': offset, 'scene_exposure': exposure,
                    'fractional_epochs': exposure/len(data),
                    'weighted_losses': dict(zip(keys, packed.tolist())), 'rank0_raw_losses': raw,
                    'rank0_effective_auxiliary_counts': effective, 'rank0_query_strata': query_records,
                    'rank0_activation_gradient_norms_per_microbatch': activation_gradients,
                    'rank0_proposal_out_of_range_steps': proposal_out_of_range,
                    'actual_parameter_group_lrs': actual_lrs, 'global_gradient_norm': gradient_norm,
                    'gradient_clipped': gradient_norm > clip_limit if clip_limit > 0 else False,
                    'gradient_clip_scale': min(1., clip_limit/(gradient_norm+1e-6)) if clip_limit > 0 else 1.,
                    'peak_GPU_allocated_bytes_rank0': torch.cuda.max_memory_allocated(),
                    'GPU_hours': elapsed*world/3600., 'seconds': float(cost), **evidence}
                if stage_records is not None:
                    record['per_rank_stage_timing'] = stage_records
                with (args.output/'metrics.jsonl').open('a') as stream: stream.write(json.dumps(record)+'\n')
                print(json.dumps(record), flush=True)
            if (args.output/'STOP_AFTER_CHECKPOINT').exists() or (args.stop_after and completed >= args.stop_after):
                save('PAUSED'); break
            if completed in milestones or completed % args.save_every == 0:
                save('COMPLETE' if completed == horizon else 'RUNNING')
            if completed == horizon and rank == 0:
                atomic_json(args.output/'TRAINING_COMPLETE.json', {'identity': identity, 'updates': completed,
                    'exposure': exposure, 'GPU_hours': elapsed*world/3600., 'evaluation_complete': False})
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
