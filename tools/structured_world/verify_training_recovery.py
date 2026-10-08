"""Run real eight-card continuous and interrupted training, then compare state."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
import numpy as np
import torch


def compare(left, right, path, report, *, exact=False):
    if torch.is_tensor(left):
        if left.shape != right.shape or left.dtype != right.dtype:
            raise AssertionError(f'{path}: tensor layout changed')
        if left.is_floating_point() and not exact:
            torch.testing.assert_close(left, right, rtol=1e-4, atol=1e-5, msg=path)
            error = float((left-right).abs().max()) if left.numel() else 0.
            report['maximum_tensor_absolute_difference'] = max(report['maximum_tensor_absolute_difference'], error)
        elif not torch.equal(left, right):
            raise AssertionError(f'{path}: discrete/RNG state changed')
        report['tensor_elements_checked'] += left.numel()
        return
    if isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        if set(left) != set(right):
            raise AssertionError(f'{path}: state keys changed')
        for name in left:
            compare(left[name], right[name], path+'/'+str(name), report, exact=exact)
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError(f'{path}: sequence length changed')
        for index, (a, b) in enumerate(zip(left, right)):
            compare(a, b, path+'/'+str(index), report, exact=exact)
    elif left != right:
        raise AssertionError(f'{path}: scalar state changed: {left} / {right}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['vla', 'geometry'], required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--imagenet', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'continuous').exists() or (args.output/'interrupted').exists():
        raise ValueError('Recovery verification needs a new output directory')
    launch = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8']
    if args.kind == 'vla':
        if not args.config: raise ValueError('VLA requires a real registered group config')
        common = launch+[str(root/'tools/structured_world/train_vla.py'), '--config', str(args.config),
            '--scope', 'profile', '--debug-updates', '4', '--save-every', '2']
    else:
        if not args.imagenet: raise ValueError('Geometry requires the public ImageNet asset')
        common = launch+[str(root/'tools/structured_world/train_geometry.py'), '--imagenet', str(args.imagenet),
            '--debug-updates', '4', '--save-every', '2']
    common += ['--cache', str(args.cache)]
    commands = [common+['--output', str(args.output/'continuous')],
        common+['--output', str(args.output/'interrupted'), '--resume']]
    (args.output/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
    env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    for index, command in enumerate(commands):
        if index == 1:
            # Resume the EXACT checkpoint at update two from the continuous
            # run. Two independent initial runs confound recovery with ordinary
            # nondeterministic CUDA backward rounding in their first two steps.
            source = args.output/'continuous'
            destination = args.output/'interrupted'
            destination.mkdir()
            shutil.copy2(source/'identity.json', destination/'identity.json')
            parents = [p/('checkpoints' if args.kind == 'vla' else '') for p in (source, destination)]
            parents[1].mkdir(exist_ok=True)
            tag = 'latest_0000002' if args.kind == 'vla' else 'checkpoint_0000002'
            shutil.copytree(parents[0]/tag, parents[1]/tag, copy_function=os.link)
            (parents[1]/'latest').write_text(tag+'\n')
        print(json.dumps({'phase': index, 'command': command}), flush=True)
        with (args.output/f'phase_{index}.log').open('w') as log:
            subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    parents = [args.output/name/('checkpoints' if args.kind == 'vla' else '') for name in ('continuous', 'interrupted')]
    folders = [p/(p/'latest').read_text().strip() for p in parents]
    report = {'kind': args.kind, 'world_size': 8, 'effective_global_batch': 32,
        'continuous_updates': 4, 'interrupted_updates': 'exact shared checkpoint at2 + resumed2', 'tensor_elements_checked': 0,
        'maximum_tensor_absolute_difference': 0., 'files_checked': [], 'RNG_comparison': 'exact',
        'floating_state_tolerance': {'relative': 1e-4, 'absolute': 1e-5},
        'tolerance_reason': 'post-update native CUDA/AMP rounding; exact restore boundary reported separately when instrumented'}
    if (args.output/'interrupted/RESUME_BOUNDARY_VERIFIED.json').exists():
        boundary = json.loads((args.output/'interrupted/RESUME_BOUNDARY_VERIFIED.json').read_text())
        if not boundary['passed']: raise AssertionError('Exact restore boundary failed')
        report['exact_restore_boundary'] = boundary
    if args.kind == 'vla':
        model = [torch.load(p/'mp_rank_00_model_states.pt', map_location='cpu', mmap=True, weights_only=False) for p in folders]
        for key in ('module', 'param_shapes', 'buffer_names', 'global_steps', 'completed', 'epoch', 'offset',
                    'exposure', 'scheduler_horizon', 'shared_geometry_identity'):
            compare(model[0][key], model[1][key], key, report)
        del model
        for rank in range(8):
            name = f'zero_pp_rank_{rank}_mp_rank_00_optim_states.pt'
            state = [torch.load(p/name, map_location='cpu', mmap=True, weights_only=False) for p in folders]
            compare(state[0], state[1], name, report)
            report['files_checked'].append(name)
            del state
    else:
        states = [torch.load(p/'model_optimizer.pt', map_location='cpu', weights_only=False) for p in folders]
        compare(states[0], states[1], 'model_optimizer', report)
        del states
    for rank in range(8):
        name = f'rng_rank{rank}.pt' if args.kind == 'vla' else f'rank{rank}.pt'
        states = [torch.load(p/name, map_location='cpu', weights_only=False) for p in folders]
        if args.kind == 'geometry':
            for key in ('completed', 'epoch', 'offset', 'exposure', 'scheduler_horizon'):
                compare(states[0][key], states[1][key], key, report)
            states = [s['rng'] for s in states]
        compare(states[0], states[1], name, report, exact=True)
        report['files_checked'].append(name)
    report['passed'] = True
    (args.output/'RECOVERY_VERIFIED.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
