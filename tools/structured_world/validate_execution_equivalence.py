"""Eight-rank real trained-model forward/gradient/RNG execution comparison.

No optimizer update or new initialization claim. The A/A replay records native
CUDA variation separately from the A/B execution change. Tolerances are fixed
before the probe, and any failure prevents formal deployment.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import json
import os
from pathlib import Path
import socket
import sys
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch import distributed as dist, nn
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from starVLA.model.framework.vla_structured_fgtr import VLAStructuredFGTR
from starVLA.model.modules.structured_world.execution import configure_execution
from starVLA.dataloader.structured_world.dataset import StructuredNAVSIMDataset, collate_structured
from tools.ddpolicy_vehicle.training_state import epoch_batches
from tools.foresight.student_state import capture_rng, restore_rng, optimizer_batch_counts
from tools.structured_world.recovery_boundary import exact
from tools.structured_world.training_assets import file_digest
from tools.structured_world.train_vla import atomic_json

GRAD_RTOL, GRAD_ATOL = 1e-4, 1e-6


def main():
    parser = argparse.ArgumentParser()
    for name in ('run', 'cache', 'dino-root', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--model-state-file', type=Path,
                        help='Optional byte-identical local copy; avoids concurrent NFS mmap page faults')
    parser.add_argument('--attention-diagnostic', choices=('native', 'math'), default='native',
                        help='Math is a separate diagnostic for native Flash backward variation, never a formal setting')
    parser.add_argument('--microbatch', type=int, choices=(2, 4), default=4)
    parser.add_argument('--optimized-mode', choices=('io_preserving_v1', 'loss_preserving_v1'), default='loss_preserving_v1')
    parser.add_argument('--precision-diagnostic', choices=('native', 'fp32'), default='native',
                        help='FP32 is a separate controlled numerical diagnostic, never a formal precision change')
    args = parser.parse_args()
    policy = json.loads(Path('/mnt/project/server_dispatch_policy.json').read_text())
    if socket.gethostname().removesuffix('-worker-0') not in policy['task_authorizations']['structured_world_fgtr_round1']['allowed_hosts']:
        raise ValueError('Unauthorized execution probe host')
    rank, world, local = [int(os.environ[name]) for name in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK')]
    if world != 8:
        raise ValueError('This probe requires the actual full eight-rank model')
    torch.cuda.set_device(local); torch.set_num_threads(2)
    if not dist.is_initialized(): dist.init_process_group('nccl')
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    identity = json.loads((args.run/'identity.json').read_text())
    tag = (args.run/'checkpoints/latest').read_text().strip()
    if Path(tag).name != tag: raise ValueError('Unsafe checkpoint pointer')
    folder = args.run/'checkpoints'/tag
    complete = json.loads((folder/'COMPLETE.json').read_text())
    if complete['identity'] != identity['identity'] or complete['status'] != 'PAUSED':
        raise ValueError('Preserved complete paused model required')
    if identity['scope'] != 'formal' or identity['dataset'] != 'navsim':
        raise ValueError('A real formal NAVSIM checkpoint is required')
    model_file = args.model_state_file or folder/'mp_rank_00_model_states.pt'
    if rank == 0:
        if file_digest(model_file) != complete['file_sha256']['mp_rank_00_model_states.pt']:
            raise ValueError('Saved model hash changed')
        args.output.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output/'REGISTRATION.json', {'checkpoint': tag, 'training_identity': identity['identity'],
            'scope': 'no-update actual trained model A/A and execution A/B', 'ranks': world, 'microbatch': args.microbatch,
            'optimized_mode': args.optimized_mode,
            'gradient_tolerance': {'rtol': GRAD_RTOL, 'atol': GRAD_ATOL},
            'forward_loss_tolerance': {'rtol': 1e-6, 'atol': 1e-7},
            'trajectory_RNG_and_buffers': 'bitwise', 'precision': identity['precision']})
        atomic_json(args.output/'ATTENTION_DIAGNOSTIC.json', {'mode': args.attention_diagnostic,
            'precision_diagnostic': args.precision_diagnostic,
            'formal_attention_backend_changed': False,
            'formal_precision_changed': False,
            'purpose': 'isolate checkpoint recomputation equivalence from native Flash backward variation'})
    dist.barrier()
    model = VLAStructuredFGTR(OmegaConf.create(identity['config'])).float()
    saved = torch.load(model_file, map_location='cpu', mmap=True, weights_only=False)
    model.load_state_dict(saved['module'], strict=True)
    epoch, offset = saved['epoch'], saved['offset']
    del saved
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda().train()
    model.inference_fp32 = args.precision_diagnostic == 'fp32'
    model.shared_geometry_identity = identity['shared_geometry_identity']
    data = StructuredNAVSIMDataset(args.cache, '/var/tmp/ddp-full-foresight-20260929/student_train_v1',
        dino_root=args.dino_root, dino_index='', expected_dino=identity['DINO_identity'],
        image_root='/var/tmp/ddp-full-foresight-20260929/images')
    if data.identity['identity'] != identity['cache_identity']: raise ValueError('Foreign data contract')
    batches = epoch_batches(len(data), 32, 42, epoch)
    if offset == len(batches):
        epoch += 1; offset = 0; batches = epoch_batches(len(data), 32, 42, epoch)
    selected_batch = batches[offset][:args.microbatch*world]
    indices = selected_batch[rank::world]
    with ThreadPoolExecutor(max_workers=2) as pool:
        observations, targets = collate_structured(list(pool.map(data.__getitem__, indices)))
    generators = {name: torch.Generator(device='cuda').manual_seed(420000+rank+shift)
        for name, shift in [('noise', 100), ('time', 200), ('proposal', 300), ('queries', 400)]}
    generators['counts'] = torch.Generator().manual_seed(420500+rank)
    start_rng = torch.load(folder/f'rng_rank{rank}.pt', map_location='cpu', weights_only=False)
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    reference_grads, reference_outputs, reference_buffers, reference_rng = {}, None, None, None
    checks = []
    for label, mode in [('reference', 'reference'), ('reference_replay', 'reference'),
                        ('optimized', args.optimized_mode)]:
        configure_execution(model, mode)
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for name, value in model.named_buffers(): value.copy_(buffers[name])
        restore_rng(start_rng, generators)
        counts = optimizer_batch_counts(targets, generators['counts'], torch.device('cuda', local))
        torch.cuda.synchronize(); begin = time.monotonic()
        from torch.nn.attention import sdpa_kernel, SDPBackend
        attention = sdpa_kernel(SDPBackend.MATH) if args.attention_diagnostic == 'math' else nullcontext()
        with attention:
            result = model(observations, targets, completed_updates=complete['completed'],
                noise_generator=generators['noise'], time_generator=generators['time'],
                proposal_generator=generators['proposal'], query_generator=generators['queries'], global_counts=counts)
            result['loss'].backward()
        torch.cuda.synchronize(); seconds = time.monotonic()-begin
        outputs = {**{key: value.detach().cpu() for key, value in result['diagnostics'].items()},
                   **{key: value.detach().cpu() for key, value in result['losses'].items()}}
        after_rng = capture_rng(generators)
        after_buffers = {name: value.detach().cpu().clone() for name, value in model.named_buffers()}
        grad_keys = {name for name, parameter in model.named_parameters() if parameter.grad is not None}
        grad_diff, violations, grad_elements, square_error, square_ref = 0., 0, 0, 0., 0.
        if reference_outputs is None:
            reference_outputs, reference_rng, reference_buffers = outputs, after_rng, after_buffers
            reference_grads = {name: parameter.grad.detach().cpu().clone() for name, parameter in model.named_parameters()
                               if parameter.grad is not None}
        else:
            exact(after_rng, reference_rng, 'post-forward/backward RNG')
            exact(after_buffers, reference_buffers, 'post-forward/backward mutable buffers')
            if grad_keys != set(reference_grads): raise AssertionError('Changed gradient parameter inventory')
            for name, value in outputs.items():
                if name in ('q0', 'q_final'):
                    torch.testing.assert_close(value, reference_outputs[name], rtol=0, atol=0)
                else:
                    torch.testing.assert_close(value, reference_outputs[name], rtol=1e-6, atol=1e-7)
            for name, parameter in model.named_parameters():
                if name not in grad_keys: continue
                actual, expected = parameter.grad.detach().cpu(), reference_grads[name]
                error = (actual-expected).abs()
                if not torch.isfinite(actual).all(): raise AssertionError('Nonfinite gradients: '+name)
                violations += int((error > GRAD_ATOL+GRAD_RTOL*expected.abs()).sum())
                grad_diff = max(grad_diff, float(error.max()))
                grad_elements += error.numel()
                square_error += float(error.double().square().sum())
                square_ref += float(expected.double().square().sum())
                del actual, error
        record = {'label': label, 'mode': mode, 'forward_and_backward_seconds': seconds,
            'gradient_parameter_count': len(grad_keys), 'gradient_elements_compared': grad_elements,
            'gradient_max_absolute_difference': grad_diff, 'gradient_tolerance_violations': violations,
            'gradient_relative_L2': (square_error/max(square_ref, 1e-300))**.5,
            'q0_q_final_RNG_and_mutable_buffers': 'bitwise',
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'passed': violations == 0}
        checks.append(record)
        print(json.dumps({'rank': rank, **record}), flush=True)
        del result
    records = [None]*world
    dist.all_gather_object(records, {'rank': rank, 'tokens': [data.index[i]['token'] for i in indices], 'checks': checks})
    passed = all(len(row['checks']) == 3 and all(check['passed'] for check in row['checks']) for row in records)
    if rank == 0:
        atomic_json(args.output/'COMPLETE.json', {'training_identity': identity['identity'], 'checkpoint': tag,
            'ranks': world, 'scene_count': len(selected_batch), 'per_rank': records, 'passed': passed,
            'attention_diagnostic': args.attention_diagnostic, 'optimized_mode': args.optimized_mode,
            'precision_diagnostic': args.precision_diagnostic,
            'limitation': 'No optimizer update here; native ZeRO update/profile and exact boundary recovery are verified separately'})
    dist.destroy_process_group()
    if not passed: raise AssertionError('Real model execution equivalence failed; formal rollout prohibited')


if __name__ == '__main__':
    main()
