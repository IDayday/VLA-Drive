"""Compare real ordered CPU input/label bytes and RNG, without using any GPU."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import pickle
import random
import sys
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from starVLA.dataloader.structured_world.dataset import StructuredNAVSIMDataset, collate_structured
from starVLA.dataloader.structured_world.nuscenes_dataset import StructuredNuScenesDataset
from tools.structured_world.prefetch import OrderedScenePrefetch


def fingerprint(value):
    if torch.is_tensor(value):
        return [str(value.dtype), list(value.shape), hashlib.sha256(value.numpy().tobytes()).hexdigest()]
    if isinstance(value, np.ndarray):
        return [str(value.dtype), list(value.shape), hashlib.sha256(value.tobytes()).hexdigest()]
    if isinstance(value, Image.Image):
        return [value.mode, value.size, hashlib.sha256(value.tobytes()).hexdigest()]
    if isinstance(value, dict):
        return {key: fingerprint(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [fingerprint(item) for item in value]
    return value


def RNG_digest():
    return hashlib.sha256(pickle.dumps((random.getstate(), np.random.get_state(), torch.random.get_rng_state().numpy()))).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ('cache', 'dino', 'images', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--current', type=Path)
    parser.add_argument('--dataset', choices=('navsim', 'nuscenes'), default='navsim')
    parser.add_argument('--DINO-identity', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.dataset == 'navsim':
        if not args.current: parser.error('NAVSIM requires --current')
        data = StructuredNAVSIMDataset(args.cache, args.current, dino_root=args.dino,
            dino_index='', expected_dino=args.DINO_identity, image_root=args.images)
    else:
        data = StructuredNuScenesDataset(args.cache, dino_root=args.dino,
            expected_dino=args.DINO_identity, image_root=args.images)
    indices = np.random.default_rng(20261009).choice(len(data), 64, replace=False).reshape(16, 4).tolist()
    before = RNG_digest()
    with ThreadPoolExecutor(max_workers=2) as pool:
        reference = [fingerprint(collate_structured(list(pool.map(data.__getitem__, selected)))) for selected in indices]
        with OrderedScenePrefetch(data, collate_structured, pool) as prefetch:
            prefetch.submit((0, 0), indices[0])
            for offset, selected in enumerate(indices):
                actual = prefetch.consume((0, offset), selected)
                if offset+1 < len(indices): prefetch.submit((0, offset+1), indices[offset+1])
                assert fingerprint(actual) == reference[offset], 'Actual current inputs/targets/order changed'
    assert before == RNG_digest(), 'Data loading advanced a training RNG'
    report = {'schema': 'real_'+args.dataset.upper()+'_CPU_prefetch_equivalence_v1', 'scenes': 64, 'ordered_batches': 16,
        'current_images_state_navigation_calibration_and_all_targets': 'bytewise identical',
        'Python_NumPy_Torch_CPU_RNG': 'exact unchanged', 'CUDA_used': False,
        'future_data_position': 'unconsumed prefetch is not an exposure; derived from saved epoch/offset',
        'cache_identity': data.identity['identity'], 'DINO_identity': data.dino_identity['identity'], 'passed': True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
