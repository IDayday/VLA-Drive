"""Offline CPU load and feature-forward check for the downloaded timm DINOv3."""
import argparse
import json
from pathlib import Path
import time

import timm
import torch
from PIL import Image
from safetensors.torch import load_file

from .download_dinov3 import REVISION, WEIGHT_SHA256
from .score_pdms import atomic_json
from starVLA.model.modules.vehicle_joint.initialization import file_sha256


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    started = time.time()
    root = Path(args.root)
    config = json.loads((root / 'config.json').read_text())
    if file_sha256(root / 'model.safetensors') != WEIGHT_SHA256:
        raise ValueError('Weight checksum mismatch')
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    model = timm.create_model(config['architecture'], pretrained=False,
                              num_classes=config['num_classes'], global_pool=config['global_pool'])
    keys = model.load_state_dict(load_file(str(root / 'model.safetensors'), device='cpu'), strict=True)
    model.eval().requires_grad_(False)
    cfg = config['pretrained_cfg']
    transform = timm.data.create_transform(
        input_size=tuple(cfg['input_size']), interpolation=cfg['interpolation'],
        mean=tuple(cfg['mean']), std=tuple(cfg['std']), crop_pct=cfg['crop_pct'], is_training=False)
    with Image.open(args.image) as image:
        pixels = transform(image.convert('RGB')).unsqueeze(0)
    rng = torch.get_rng_state().clone()
    with torch.inference_mode():
        features = model.forward_features(pixels)
        repeated = model.forward_features(pixels)
        pooled = model.forward_head(features, pre_logits=True)
    assert torch.isfinite(features).all() and torch.isfinite(pooled).all()
    assert torch.equal(features, repeated)
    assert torch.equal(rng, torch.get_rng_state())
    assert features.shape == (1, 261, 1024) and pooled.shape == (1, 1024)
    report = {
        'status': 'PASS', 'revision': REVISION, 'weight_sha256': WEIGHT_SHA256,
        'torch': torch.__version__, 'timm': timm.__version__, 'device': 'cpu', 'dtype': 'float32',
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'missing_keys': keys.missing_keys, 'unexpected_keys': keys.unexpected_keys,
        'image_sha256': file_sha256(Path(args.image)), 'input_shape': list(pixels.shape),
        'feature_shape': list(features.shape), 'embedding_shape': list(pooled.shape),
        'finite_outputs': True, 'repeated_output_exact': True, 'rng_unchanged': True,
        'all_parameters_frozen': all(not p.requires_grad for p in model.parameters()),
        'rope_policy': 'timm default FP32 periods, no BF16 truncation',
        'optimizer_updates': 0, 'GPU_used': False, 'elapsed_seconds': time.time() - started,
    }
    atomic_json(Path(args.output), report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
