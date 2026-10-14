"""Streamed initialization and checkpoint hashes; no weight/data upload."""
import hashlib
import json
from pathlib import Path


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''): digest.update(block)
    return digest.hexdigest()


def initialization_assets(config, geometry, geometry_identity, label_identity, *, formal):
    root = Path(config.framework.qwenvl.base_vlm)
    files = sorted(p for p in root.iterdir() if p.suffix in ('.json', '.safetensors', '.txt') and p.is_file())
    if not any(p.suffix == '.safetensors' for p in files): raise ValueError('Missing generic Qwen weights')
    values = {'Qwen_public_pretraining_files': {p.name: file_digest(p) for p in files},
              'ImageNet_R50_sha256': file_digest(config.structured_world.imagenet_checkpoint)}
    if geometry:
        import torch
        state = torch.load(geometry, map_location='cpu', weights_only=False)
        if state['identity'] != geometry_identity or state['label_cache_identity'] != label_identity:
            raise ValueError('Shared geometry differs from the student training population/cache')
        if state['dataset'] != config.structured_world.dataset or state['training_task'] != 'current_depth_road_occupancy_no_planner':
            raise ValueError('Foreign dataset or driving-trained geometry initialization')
        if state['ImageNet_sha256'] != values['ImageNet_R50_sha256']:
            raise ValueError('ImageNet initialization differs between perception and student')
        if formal and (state['population'] != 'full_train_population' or not 5 <= state['epochs'] <= 10):
            raise ValueError('Formal groups need five to ten complete shared training epochs')
        values['shared_geometry'] = {'sha256': file_digest(geometry), 'identity': geometry_identity,
            'training_source_sha': state['training_source_sha'], 'label_cache_identity': label_identity,
            'epochs': state['epochs'], 'actual_exposure': state['actual_exposure']}
    elif formal:
        raise ValueError('Formal geometry preparation is mandatory')
    values['identity'] = hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
    return values


def verify_checkpoint_files(folder, complete, *, rank=None, world=8):
    hashes = complete.get('file_sha256')
    if not hashes: raise ValueError('Checkpoint lacks immutable file hashes; use its historical frozen engineering reader')
    files = {p.name for p in Path(folder).glob('*.pt')}
    if files != set(hashes): raise ValueError('Checkpoint file inventory differs')
    selected = hashes if rank is None else {name: expected for name, expected in hashes.items()
        if name == f'rng_rank{rank}.pt' or name == f'rank{rank}.pt' or name == f'zero_pp_rank_{rank}_mp_rank_00_optim_states.pt'
        or (rank == 0 and name in ('mp_rank_00_model_states.pt', 'model_optimizer.pt'))}
    if not selected: raise ValueError('Missing rank checkpoint files')
    for name, expected in selected.items():
        if file_digest(Path(folder)/name) != expected: raise ValueError('Checkpoint file changed: '+name)
