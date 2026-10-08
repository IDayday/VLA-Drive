"""Separate exact checkpoint restoration from subsequent native CUDA variation.

This reads existing recovery runs; it never changes training, tolerances, or
checkpoints and never treats a failed floating comparison as a passing test.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    folders = []
    for name in ('continuous', 'interrupted'):
        root = args.run/name/'checkpoints'
        folders.append(root/(root/'latest').read_text().strip())
    report = {'exact_restore_boundary': json.loads((args.run/'interrupted/RESUME_BOUNDARY_VERIFIED.json').read_text()),
              'post_update_tolerance': {'relative': 1e-4, 'absolute': 1e-5},
              'floating_tensors': 0, 'floating_elements': 0, 'outside_tolerance_elements': 0,
              'maximum_absolute_difference': 0., 'exact_state_mismatches': [], 'largest_differences': []}
    differences = []

    def walk(a, b, path, exact=False):
        if torch.is_tensor(a):
            if a.shape != b.shape or a.dtype != b.dtype:
                report['exact_state_mismatches'].append({'path': path, 'reason': 'shape/dtype'})
                return
            if not a.is_floating_point() or exact:
                if not torch.equal(a, b):
                    report['exact_state_mismatches'].append({'path': path, 'reason': 'values'})
                return
            report['floating_tensors'] += 1
            report['floating_elements'] += a.numel()
            maximum, outside, square, scale = 0., 0, 0., 0.
            for left, right in zip(a.reshape(-1).split(2_000_000), b.reshape(-1).split(2_000_000)):
                delta = (left-right).abs()
                maximum = max(maximum, float(delta.max()) if len(delta) else 0.)
                outside += int((delta > 1e-5+1e-4*right.abs()).sum())
                square += float(delta.double().square().sum())
                scale += float(right.double().square().sum())
            report['outside_tolerance_elements'] += outside
            report['maximum_absolute_difference'] = max(report['maximum_absolute_difference'], maximum)
            if maximum:
                differences.append({'path': path, 'elements': a.numel(), 'maximum_absolute_difference': maximum,
                                    'outside_tolerance_elements': outside,
                                    'RMS_relative_to_state': (square/max(scale, 1e-30))**.5})
            return
        if isinstance(a, np.ndarray):
            if not np.array_equal(a, b): report['exact_state_mismatches'].append({'path': path, 'reason': 'numpy values'})
        elif isinstance(a, dict):
            if set(a) != set(b):
                report['exact_state_mismatches'].append({'path': path, 'reason': 'dictionary keys'})
                return
            for key in a: walk(a[key], b[key], path+'/'+str(key), exact)
        elif isinstance(a, (list, tuple)):
            if len(a) != len(b):
                report['exact_state_mismatches'].append({'path': path, 'reason': 'sequence length'})
                return
            for index, (left, right) in enumerate(zip(a, b)): walk(left, right, path+'/'+str(index), exact)
        elif hasattr(a, '__dict__'):
            walk(vars(a), vars(b), path+'/attributes', exact)
        elif a != b:
            report['exact_state_mismatches'].append({'path': path, 'reason': 'scalar values', 'left': str(a), 'right': str(b)})

    def read(name):
        return [torch.load(p/name, map_location='cpu', weights_only=False, mmap=True) for p in folders]

    models = read('mp_rank_00_model_states.pt')
    for key in ('module', 'param_shapes', 'buffer_names', 'global_steps', 'completed', 'epoch', 'offset',
                'exposure', 'scheduler_horizon', 'shared_geometry_identity'):
        walk(models[0][key], models[1][key], key)
    del models
    for rank in range(8):
        name = f'zero_pp_rank_{rank}_mp_rank_00_optim_states.pt'
        states = read(name); walk(*states, name); del states
        name = f'rng_rank{rank}.pt'
        states = read(name); walk(*states, name, exact=True); del states
    report['largest_differences'] = sorted(differences, key=lambda row: row['maximum_absolute_difference'], reverse=True)[:30]
    report['outside_tolerance_fraction'] = report['outside_tolerance_elements']/max(1, report['floating_elements'])
    report['post_update_floating_test_passed'] = not report['outside_tolerance_elements'] and not report['exact_state_mismatches']
    report['interpretation'] = ('The load boundary and all subsequent RNG/data/step comparisons are distinct checks. '
                                'A nonzero native CUDA continuation difference is reported, not hidden by enlarging tolerance.')
    (args.run/'RECOVERY_NUMERICAL_ANALYSIS.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
