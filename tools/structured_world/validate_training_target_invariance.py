"""Replace actual training targets while holding model/current inputs/RNG fixed.

Unlike adding ignored GT keys to deployment inputs, this exercises the complete
training forward and its FM, depth, geometry, event and query label paths.
It performs no optimizer update and restores mutable buffers between forwards.
"""
import argparse
import json
from pathlib import Path
import random
import socket
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.structured_world.infer_checkpoint import load_model
from tools.structured_world.build_cache import atomic_json, digest
from starVLA.dataloader.structured_world.dataset import StructuredNAVSIMDataset, collate_structured


def replace_targets(targets):
    changed = {key: value.clone() if torch.is_tensor(value) else value for key, value in targets.items()}
    changed['ego'] = -targets['ego']
    changed['current_dino'] = -targets['current_dino']
    changed['road_distance'] = -targets['road_distance']
    changed['depth'].zero_()
    occupied = targets['occupancy']
    changed['occupancy'] = torch.where(occupied == 255, occupied, 1-occupied)
    return changed


def main():
    parser = argparse.ArgumentParser()
    for name in ('run', 'cache', 'dino-root', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--scenes', type=int, default=4)
    args = parser.parse_args()
    policy = json.loads(Path('/mnt/project/server_dispatch_policy.json').read_text())
    if socket.gethostname().removesuffix('-worker-0') not in policy['task_authorizations']['structured_world_fgtr_round1']['allowed_hosts']:
        raise ValueError('Unauthorized validation host')
    torch.cuda.set_device(0); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    model, identity, checkpoint = load_model(args.run, args.tag, 'cuda')
    if identity['dataset'] != 'navsim': raise ValueError('This real-source probe requires NAVSIM labels')
    model.inference_fp32 = False
    model.train()
    data = StructuredNAVSIMDataset(args.cache, '/var/tmp/ddp-full-foresight-20260929/student_train_v1',
        dino_root=args.dino_root, dino_index='/nonexistent/future_teacher_index',
        expected_dino=identity['DINO_identity'], image_root='/var/tmp/ddp-full-foresight-20260929/images', allow_debug=True)
    if data.identity['identity'] != identity['cache_identity']: raise ValueError('Foreign probe label population')
    selection = identity.get('engineering_training_selection')
    indices = selection['source_indices'] if selection else list(range(len(data)))
    checks = []
    for index in indices[:args.scenes]:
        observations, targets = collate_structured([data[index]])
        buffers = {name: buffer.detach().clone() for name, buffer in model.named_buffers()}
        random.seed(4200+index); np.random.seed(4200+index)
        torch.manual_seed(4200+index); torch.cuda.manual_seed_all(4200+index)
        cpu_state, cuda_state = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        def forward(labels):
            for name, buffer in model.named_buffers(): buffer.copy_(buffers[name])
            torch.set_rng_state(cpu_state); torch.cuda.set_rng_state_all(cuda_state)
            generators = {name: torch.Generator(device='cuda').manual_seed(42000+index+offset)
                for offset, name in enumerate(('noise_generator', 'time_generator', 'proposal_generator', 'query_generator'))}
            with torch.no_grad():
                result = model(observations, labels, completed_updates=checkpoint['completed'], **generators)
            return result
        original = forward(targets)
        substituted = forward(replace_targets(targets))
        differences = {}
        for name in ('q0', 'q_final'):
            left, right = original['diagnostics'][name], substituted['diagnostics'][name]
            differences[name] = float((left-right).abs().max())
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        if float(original['loss']) == float(substituted['loss']):
            raise AssertionError('Substituted labels did not exercise the training objective')
        checks.append({'token': data.index[index]['token'], 'max_absolute_differences': differences,
            'original_loss': float(original['loss']), 'substituted_loss': float(substituted['loss'])})
        del original, substituted, buffers
    atomic_json(args.output, {'scope': 'actual complete training-forward label substitution; no optimizer update',
        'training_identity': identity['identity'], 'checkpoint': args.tag,
        'validator_sha256': digest(Path(__file__)), 'training_mode': True,
        'replaced': ['ego GT', 'current DINO', 'road distance', 'depth', 'current and future occupancy'],
        'held_fixed': ['current RGB/navigation/legal ego/calibration', 'parameters', 'buffers', 'FM/proposal/query RNG'],
        'tolerance': 'bitwise q0 and q_final', 'checks': checks, 'passed': True})
    print(json.dumps({'passed': True, 'scenes': len(checks), 'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
