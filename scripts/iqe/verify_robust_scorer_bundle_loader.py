#!/usr/bin/env python3
"""One real-observation public-loader check, without qualification runtime setup."""
import argparse
import fcntl
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from iqe.contracts import require
from iqe.data.sources import load_observation
from iqe.io import atomic_json, file_hash, read_json
from iqe.robust_scorer.bundle import load as load_bundle
from iqe.robust_scorer.config import load
from iqe.robust_scorer.qualification import compare_outputs
from iqe.specialist_study.binding import scenes_by_role
from iqe.specialist_study.resource_guard import check


def flags():
    return {'deterministic': torch.are_deterministic_algorithms_enabled(),
            'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_tf32': torch.backends.cudnn.allow_tf32,
            'cudnn_benchmark': torch.backends.cudnn.benchmark}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-samples', type=int, choices=[1], required=True)
    args = parser.parse_args()
    config = load(args.config)
    check(config, 'benchmark-system')
    folder = config.root/'qualification/public_loader'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'RUN.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source = config.source()
        scene = scenes_by_role(source)['stage_val'][0]
        bundle = config.root/'research_bundle/bundle.pt'
        reference = config.root/'qualification/REFERENCE.pt'
        receipt = read_json(reference.with_suffix('.json'))
        dependencies = {'bundle': file_hash(bundle),
                        'choice': file_hash(config.root/'MODEL_CHOICE.json')}
        require(receipt['dependencies'] == dependencies, 'different reference model')
        require(receipt['scene_id'] == scene.scene_id and
                receipt['observation_hash'] == scene.observation_hash, 'different reference observation')
        require(file_hash(reference) == receipt['checksum'], 'reference checksum changed')
        identity = dependencies | {'reference': receipt['checksum'],
            'loader_source': file_hash(ROOT/'iqe/robust_scorer/bundle.py'),
            'runtime_source': file_hash(ROOT/'iqe/specialist_study/training.py'),
            'script_source': file_hash(__file__)}
        result_path = folder/'REPORT.json'
        if result_path.exists():
            result = read_json(result_path)
            require(result['identity'] == identity, 'stale public-loader qualification')
            print(result)
            return
        # Deliberately start outside the registered execution flags. No call
        # to training.runtime or qualification.run occurs before public load.
        torch.use_deterministic_algorithms(False)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        before = flags()
        system = load_bundle(bundle, args.device)
        after = flags()
        require(after == dict(deterministic=True, matmul_tf32=False,
                              cudnn_tf32=False, cudnn_benchmark=False), 'public runtime not restored')
        expected = torch.load(reference, map_location='cpu', weights_only=True)
        with torch.no_grad():
            actual = system.shared_forward([load_observation(scene, system.adapter.contract)])
        errors = compare_outputs(actual, expected)
        result = dict(status='PASS', identity=identity, real_scenes=1,
                      scene_id=scene.scene_id, observation_hash=scene.observation_hash,
                      flags_before=before, flags_after=after, errors=errors,
                      separate_process=True, manual_runtime_setup=False,
                      calls=actual['calls'], branch_calls=actual['branch_calls'],
                      optimizer_steps=0, weights_changed=False, new_navtest=False,
                      original_reference_or_tolerances_modified=False,
                      device=args.device, hardware=torch.cuda.get_device_name()
                      if args.device.startswith('cuda') else 'CPU')
        atomic_json(result_path, result, immutable=True)
        print(result)


if __name__ == '__main__':
    main()
