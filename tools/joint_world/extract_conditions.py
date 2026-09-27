"""Rebuild frozen current-observation graph inputs; labels live in separate caches."""
import argparse
import hashlib
import io
import json
import subprocess
from pathlib import Path

import torch

from tools.structured_world.runtime import load_baseline, load_dataset, seed_all
from tools.structured_world_v1p1.budget import start, record
from starVLA.model.modules.joint_world.policy import JointTrajectoryPolicy


CACHE_FIELDS = {'schema_version', 'token', 'identity_sha256', 'input_sha256', 'native_actions',
                'actor_features', 'context', 'current_xy', 'existence', 'current_boxes', 'current_logits'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['world-checkpoint', 'graph-config', 'manifest', 'output', 'ledger', 'run-id']:
        p.add_argument('--' + key, required=True)
    p.add_argument('--limit', type=int)
    a = p.parse_args(); out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    blob = Path(a.world_checkpoint).read_bytes()
    saved = torch.load(io.BytesIO(blob), map_location='cpu', weights_only=False)
    base = saved['identity']['arguments']; cfg = saved['identity']['config']
    if cfg['provider'] != 'qwen': raise ValueError('This extractor currently supports image/Qwen conditions only')
    graph_cfg = json.loads(Path(a.graph_config).read_text())
    identity = {'schema_version': 1, 'code_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'world_checkpoint_sha256': hashlib.sha256(blob).hexdigest(), 'arguments': vars(a),
                'manifest_sha256': hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),
                'baseline_checkpoint_sha256': saved['identity']['original_checkpoint_sha256'],
                'world_config': cfg, 'sensor_contract': 'current CAM_L0 CAM_F0 CAM_R0; no augmentation',
                'frozen_upstream': True, 'targets_loaded': False}
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    start(a.ledger, a.run_id, 0, identity)
    try:
        seed_all(42); agent = load_baseline(base['checkpoint'], base['vlm']); agent.model.requires_grad_(False)
        model = JointTrajectoryPolicy(agent.model, cfg, graph_cfg).cuda().eval().requires_grad_(False)
        for key in ['reader', 'heads']: getattr(model.world, key).load_state_dict(saved[key], strict=True)
        ds = load_dataset(agent, a.manifest, base['data_root'], a.limit)
        records = []
        for index in range(len(ds)):
            if not record(a.ledger, a.run_id, 0): raise RuntimeError('GPU hour cap')
            raw = ds[index]; e = {key: raw[key] for key in ['image', 'lang', 'state', 'token']}
            fingerprint = hashlib.sha256()
            for im in e['image']: fingerprint.update(im.tobytes())
            fingerprint.update(str(e['lang']).encode()); fingerprint.update(e['state'].tobytes())
            with torch.no_grad(): native, pred, current = model.encode_current([e])
            data = {'schema_version': 1, 'token': e['token'], 'identity_sha256': identity_hash,
                    'input_sha256': fingerprint.hexdigest(), 'native_actions': native.cpu(),
                    'current_boxes': pred['boxes'].cpu(), 'current_logits': pred['logits'].cpu(),
                    **{key: value.cpu() for key, value in current.items()}}
            assert set(data) == CACHE_FIELDS
            path = out / (e['token'] + '.pt'); torch.save(data, path)
            records.append({'token': e['token'], 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            if (index + 1) % 16 == 0: print(f'Extracted {index + 1}/{len(ds)}', flush=True)
        (out / 'manifest.json').write_text(json.dumps({'identity': identity, 'identity_sha256': identity_hash,
                                                     'records': records, 'failed': 0}, indent=2))
        record(a.ledger, a.run_id, 0, 'complete')
    except BaseException:
        record(a.ledger, a.run_id, 0, 'failed'); raise


if __name__ == '__main__': main()
