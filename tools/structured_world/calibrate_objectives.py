"""Measure real component gradients on a fixed training subset, never dev PDMS.

Per-rank scene gradient norms are reported as a calibration distribution, not
misrepresented as the norm of a globally averaged optimizer gradient.
"""
import argparse
import json
import os
from pathlib import Path
import random
import socket
import sys
import numpy as np
import torch
from torch import distributed as dist
from omegaconf import OmegaConf
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from starVLA.model.framework.vla_structured_fgtr import VLAStructuredFGTR
from starVLA.dataloader.structured_world.dataset import StructuredNAVSIMDataset, collate_structured
from starVLA.dataloader.structured_world.nuscenes_dataset import StructuredNuScenesDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--geometry-identity', required=True)
    parser.add_argument('--dino-root', type=Path)
    parser.add_argument('--dino-identity')
    parser.add_argument('--samples-per-rank', type=int, default=2)
    args = parser.parse_args()
    policy = json.loads(Path('/mnt/project/server_dispatch_policy.json').read_text())
    if socket.gethostname().removesuffix('-worker-0') not in policy['task_authorizations']['structured_world_fgtr_round1']['allowed_hosts']:
        raise ValueError('Unauthorized calibration host')
    rank, world, local = [int(os.environ[name]) for name in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK')]
    if world != 8: raise ValueError('Fixed full-eight-card calibration required')
    torch.cuda.set_device(local)
    if not dist.is_initialized(): dist.init_process_group('nccl')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    random.seed(42+rank); np.random.seed(42+rank); torch.manual_seed(42+rank); torch.cuda.manual_seed_all(42+rank)
    config = OmegaConf.load(args.config)
    if config.structured_world.group not in ('G1_FULL_UNIFORM', 'G3_EVENT_LOCAL'):
        raise ValueError('Calibration requires a complete registered objective')
    if config.structured_world.dataset == 'navsim':
        data = StructuredNAVSIMDataset(args.cache, '/var/tmp/ddp-full-foresight-20260929/student_train_v1',
            dino_root='/var/tmp/ddp-full-foresight-20260929/targets/C1',
            dino_index='/mnt/project/ddp-full-foresight-study-artifacts/20260929/dino_index_v1',
            expected_dino='7663c45b77dd711e8304e8d202644dcaa8467f7e265ff8b4c1816d59d2c131cb',
            image_root='/var/tmp/ddp-full-foresight-20260929/images', allow_debug=True)
    else:
        data = StructuredNuScenesDataset(args.cache, dino_root=args.dino_root, expected_dino=args.dino_identity, allow_debug=True)
    model = VLAStructuredFGTR(config).float().cuda().train()
    named = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    indices = np.random.default_rng(42).choice(len(data), args.samples_per_rank*world, replace=False).tolist()
    if data.identity['source_current_identity']['split'] != 'train': raise ValueError('Training-only calibration required')
    rows = []
    for phase in ('random_geometry', 'prepared_geometry'):
        if phase == 'prepared_geometry':
            model.load_shared_geometry(args.geometry, args.geometry_identity, allow_debug=True)
        for selected in indices[rank::world]:
            # All task/random states are reset identically between the two
            # preparation states. No optimizer update or validation tuning occurs.
            torch.manual_seed(1042+selected); torch.cuda.manual_seed_all(1042+selected)
            generators = {name: torch.Generator(device='cuda').manual_seed(420000+selected+offset)
                          for name, offset in [('noise', 100), ('time', 200), ('proposal', 300), ('query', 400)]}
            observations, targets = collate_structured([data[selected]])
            output = model(observations, targets, completed_updates=1000,
                noise_generator=generators['noise'], time_generator=generators['time'],
                proposal_generator=generators['proposal'], query_generator=generators['query'])
            losses = output['losses']
            objectives = {'baseline_FM_plus_current_DINO': losses['original_FM']+losses['current_DINO'],
                'current_geometry': losses['current_geometry'], 'future_semantics': losses['future_semantics'],
                'query_relations': losses['local_relations'], 'refine_at_full_weight': losses['refine']}
            record = {'rank': rank, 'phase': phase, 'token': observations[0]['token'], 'objectives': {}}
            for name, value in objectives.items():
                # The unchanged mature ResNet uses reentrant checkpointing,
                # which supports backward() but rejects autograd.grad(inputs).
                model.zero_grad(set_to_none=True)
                value.backward(retain_graph=True)
                norm_squared = {}
                for parameter_name, parameter in named:
                    gradient = parameter.grad
                    if gradient is not None:
                        module = parameter_name.split('.')[0]
                        norm_squared[module] = norm_squared.get(module, 0.)+float(torch.linalg.vector_norm(gradient.detach().float()))**2
                record['objectives'][name] = {'loss': float(value.detach()),
                    'total_parameter_gradient_norm': sum(norm_squared.values())**.5,
                    'parameter_module_gradient_norms': {k: v**.5 for k, v in norm_squared.items()}}
                model.zero_grad(set_to_none=True)
            rows.append(record)
            del output, objectives, losses
            torch.cuda.empty_cache()
    all_rows = [None]*world
    dist.all_gather_object(all_rows, rows)
    if rank == 0:
        flat = [row for rank_rows in all_rows for row in rank_rows]
        report = {'scope': 'fixed_train_gradient_measurement_not_formal_results',
                  'dataset': config.structured_world.dataset, 'group': config.structured_world.group,
                  'cache_identity': data.identity['identity'], 'geometry_identity': args.geometry_identity,
                  'samples': len(indices), 'loss_weights_used': dict(config.structured_world.loss_weights),
                  'norm_definition': 'per scene/rank parameter gradient; no optimizer or validation score tuning',
                  'rows': flat}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({'measurements': len(flat), 'output': str(args.output)}), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__': main()
