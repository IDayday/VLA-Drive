"""Official Stage1 feature builder, resumable rank-local sharded extraction."""
import argparse
import os
from pathlib import Path
import sys
import time

import torch
from .assets import atomic_json, check_official, digest, read


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ('official-source', 'official-revision', 'manifest', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--limit', type=int, default=0)
    a = p.parse_args()
    source = check_official(a.official_source, a.official_revision)
    sys.path.insert(0, str(source))
    from navsim.agents.recogdrive.recogdrive_features import ReCogDriveFeatureBuilder
    from navsim.common.dataclasses import AgentInput, Cameras, Camera, EgoStatus
    rank, world, local = [int(os.environ.get(k, d)) for k, d in [('RANK', 0), ('WORLD_SIZE', 1), ('LOCAL_RANK', 0)]]
    torch.cuda.set_device(local); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    data = read(a.manifest); identity = data['identity']; rows = data['rows']
    if a.limit:
        # A real pipeline smoke must include both official splits. It never
        # publishes a full-population completion marker.
        rows = ([r for r in rows if r['split'] == 'train'][:a.limit // 2] +
                [r for r in rows if r['split'] == 'val'][:a.limit - a.limit // 2])
    root = Path(a.output); root.mkdir(parents=True, exist_ok=True)
    manifest_hash = digest(a.manifest)
    builder = ReCogDriveFeatureBuilder(cache_hidden_state=True, cache_mode=True,
        model_type='internvl', checkpoint_path=identity['stage1']['path'], device=f'cuda:{local}')
    model = builder.backbone.model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    from safetensors import safe_open
    with safe_open(Path(identity['stage1']['path']) / 'model.safetensors', framework='pt') as weights:
        if set(weights.keys()) != set(model.state_dict()):
            raise ValueError('Stage1 core tensor inventory is not strictly identical')
    signature = dict(schema='official_recogdrive_stage1_full_hidden_v1',
                     manifest_sha256=manifest_hash, official_revision=a.official_revision,
                     stage1=identity['stage1'], dtype='float32', actual_scenes=len(rows), world=world)
    ident = root / f'identity_rank{rank:02d}.json'
    if ident.exists() and read(ident) != signature:
        raise ValueError('Feature cache identity changed')
    atomic_json(ident, signature)
    done, total_bytes, begin = 0, 0, time.time()
    for row in rows[rank::world]:
        dest = root / row['token'][:2] / (row['token'] + '.pt')
        if dest.exists():
            item = torch.load(dest, map_location='cpu', weights_only=True)
            if item['token'] != row['token'] or item['manifest_sha256'] != manifest_hash:
                raise ValueError('Foreign feature cache entry')
        else:
            statuses = [EgoStatus(ego_pose=pose, ego_velocity=row['velocity'],
                ego_acceleration=row['acceleration'], driving_command=row['command']) for pose in row['history']]
            cameras = []
            for i in range(4):
                cs = Cameras(**{k: Camera() for k in Cameras.__dataclass_fields__})
                if i == 3:
                    cs.cam_f0.image = Path(row['image'])
                cameras.append(cs)
            with torch.inference_mode():
                item = builder.compute_features(AgentInput(statuses, cameras, []))
            if item['last_hidden_state'].ndim != 2 or item['last_hidden_state'].shape[-1] != 1536:
                raise ValueError('Wrong Stage1 hidden layout')
            if any(not torch.isfinite(v).all() for v in item.values()):
                raise FloatingPointError('Nonfinite official Stage1 features')
            item.update(token=row['token'], manifest_sha256=manifest_hash)
            dest.parent.mkdir(parents=True, exist_ok=True)
            temp = dest.with_suffix(f'.{rank}.tmp'); torch.save(item, temp); temp.replace(dest)
        done += 1; total_bytes += dest.stat().st_size
        if done == 1 or done % 50 == 0:
            atomic_json(root / f'progress_rank{rank:02d}.json', dict(status='RUNNING', rank=rank,
                completed=done, bytes=total_bytes, seconds=time.time()-begin,
                samples_s=done/(time.time()-begin), hidden_shape=list(item['last_hidden_state'].shape),
                peak_allocated=torch.cuda.max_memory_allocated(), real_optimizer_updates=0))
    atomic_json(root / f'COMPLETE_rank{rank:02d}.json', dict(signature=signature, scenes=done,
        bytes=total_bytes, seconds=time.time()-begin, real_optimizer_updates=0))


if __name__ == '__main__':
    main()
