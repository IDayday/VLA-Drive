"""Strict new-architecture checkpoint loaders; historical World schemas rejected."""
import copy
import json
from pathlib import Path

import torch

from .contracts import file_sha, fingerprint

IL_SCHEMA = 'qwen_world_compare_IL_v1'
RL_SCHEMA = 'qwen_world_compare_RL_v1'
SOURCE_ROOT = Path(__file__).resolve().parents[3]


def _read(path):
    return json.loads(Path(path).read_text())


def contained_file(root, name):
    root = Path(root)
    path = root/name
    if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('Unsafe or missing sealed asset: '+name)
    return path


def inspect_IL_checkpoint(path, *, verify_files='model'):
    """Reuse native inventory checks; metadata-only mode is explicitly labeled.

    Evaluations already hash model bytes. DEV parent selection can check their
    sealed model digest against the immutable native inventory without rereading
    twenty full models. Loading/export always requests model-byte verification.
    """
    root = Path(path)
    if verify_files in ('all', 'model'):
        from starVLA.rl.qwen_world_postcot.loading import inspect_checkpoint
        info = inspect_checkpoint(root, verify_files=verify_files)
    elif verify_files == 'inventory':
        seal, identity, files = _read(root/'COMPLETE'), _read(root/'identity.json'), _read(root/'files.json')
        if (seal.get('status') != 'COMPLETE' or seal.get('files_sha256') != file_sha(root/'files.json')
                or seal.get('identity_sha256') != fingerprint(identity)):
            raise ValueError('Changed native IL checkpoint seal')
        name = seal.get('model_file')
        world = identity.get('world', 0)
        if (name != 'native/mp_rank_00_model_states.pt' or name not in files or world < 1
                or len([p for p in files if p.endswith('_optim_states.pt')]) != world
                or any(f'rank{i:02d}_state.pt' not in files for i in range(world))):
            raise ValueError('Native IL recovery inventory incomplete')
        for name in files:
            contained_file(root, name)
        info = dict(path=str(root.resolve()), seal=seal, identity=identity,
                    files=files, model_sha256=files[seal['model_file']])
    else:
        raise ValueError('Unknown IL verification scope')
    identity = info['identity']
    if (identity.get('schema') != IL_SCHEMA or identity.get('architecture_id') not in ('A', 'B', 'C', 'D')
            or identity.get('uses_rl') is not False or type(identity.get('qualification')) is not bool
            or not isinstance(identity.get('model_config'), dict)
            or not isinstance(identity.get('qwen_config'), dict)
            or not isinstance(identity.get('config'), dict)
            or not isinstance(identity.get('source_files'), dict)):
        raise ValueError('Not a complete four-framework IL checkpoint identity')
    if not identity['qualification'] and identity['world'] != 8:
        raise ValueError('Formal four-framework IL uses eight ranks')
    if not 0 <= info['seal']['updates'] <= 100000:
        raise ValueError('IL update outside the fixed 100k budget')
    info['verification_scope'] = verify_files
    return info


def model_from_identity(identity, *, train_heads=True):
    from starVLA.model.modules.qwen_world_compare.model import FourFrameworkVLA
    if identity.get('schema') != IL_SCHEMA:
        raise ValueError('Cannot construct this architecture from a historical model schema')
    return FourFrameworkVLA(identity['architecture_id'], qwen_path=None,
        config=copy.deepcopy(identity['model_config']),
        checkpoint_config=copy.deepcopy(identity['qwen_config']), train_heads=train_heads)


def strict_tensor_load(model, state):
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError('Strict deployment tensor keys differ: missing='+repr(sorted(set(expected)-set(state)))
                         +'; extra='+repr(sorted(set(state)-set(expected))))
    for name, value in state.items():
        target = expected[name]
        if not isinstance(value, torch.Tensor) or value.shape != target.shape or value.dtype != target.dtype:
            raise ValueError('Strict load would change tensor shape/dtype: '+name)
        if value.is_floating_point() and value.dtype != torch.float32:
            raise ValueError('Full-framework model storage must remain FP32: '+name)
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError('Nonfinite model tensor: '+name)
    model.load_state_dict(state, strict=True)


def load_IL_checkpoint(path, device='cpu', *, deployment=False):
    info = inspect_IL_checkpoint(path, verify_files='model')
    raw = torch.load(Path(info['path'])/info['seal']['model_file'], map_location='cpu', weights_only=False)
    if not isinstance(raw, dict) or not isinstance(raw.get('module'), dict):
        raise ValueError('Native IL checkpoint has no complete module state')
    model = model_from_identity(info['identity'], train_heads=True)
    strict_tensor_load(model, raw['module'])
    if deployment:
        from .deployment import prune_deployment
        # This is a private newly loaded instance, never a live training model.
        model = prune_deployment(model, copy_model=False)
    return model.eval().requires_grad_(False).to(device), info


def inspect_RL_checkpoint(path, *, verify_files='all'):
    root = Path(path)
    if '.incomplete' in root.name:
        raise ValueError('Unpublished RL checkpoint is not a policy')
    identity, seal, files = _read(root/'identity.json'), _read(root/'COMPLETE'), _read(root/'files.json')
    body = dict(identity); saved = body.pop('sha256', None)
    world = identity.get('world', 0)
    if (fingerprint(body) != saved or identity.get('schema') != RL_SCHEMA
            or identity.get('architecture_id') not in ('A', 'B', 'C', 'D')
            or seal.get('schema') != 2 or world < 1 or seal.get('world') != world
            or seal.get('phase') not in ('READY', 'ROLLOUT_COMMITTED', 'INNER_COMMITTED', 'COMPLETE')
            or seal.get('files_sha256') != file_sha(root/'files.json')):
        raise ValueError('Invalid four-framework committed RL checkpoint')
    if (identity.get('purpose') not in ('formal', 'qualification', 'qualification_not_formal_initialization')
            or (identity['purpose'] == 'formal' and world != 8)
            or identity.get('recipe', {}).get('action_shape') != [8, 3]
            or identity['recipe'].get('prediction_type') != 'x0'
            or not 0 <= seal['updates'] <= 50000):
        raise ValueError('RL purpose, codec, prediction or fixed budget changed')
    required = {'identity.json', 'optimizer.pt', 'transaction.json'}|{f'rank{i}.pt' for i in range(world)}
    if not required.issubset(files):
        raise ValueError('RL checkpoint lacks optimizer/adapter/rank transaction coverage')
    if verify_files not in ('all', 'inventory'):
        raise ValueError('RL inspection supports all/inventory only')
    for name, sha in files.items():
        target = contained_file(root, name)
        if verify_files == 'all' and file_sha(target) != sha:
            raise ValueError('Changed RL payload: '+name)
    transaction = _read(root/'transaction.json')
    ranks = transaction.get('ranks', [])
    if len(ranks) != world:
        raise ValueError('RL transaction lacks every rank boundary')
    common = ('phase', 'updates', 'collections', 'scene_cursor', 'next_inner', 'run', 'pending')
    if any({k:r.get(k) for k in common} != {k:ranks[0].get(k) for k in common} for r in ranks):
        raise ValueError('RL ranks disagree on the committed boundary')
    if (ranks[0].get('run') != saved or ranks[0].get('phase') != seal['phase']
            or ranks[0].get('updates') != seal['updates']
            or ranks[0].get('next_inner') != seal['next_inner']):
        raise ValueError('RL boundary/source identity changed')
    return dict(path=str(root.resolve()), identity=identity, seal=seal, files=files,
                transaction=transaction, verification_scope=verify_files)


inspect_checkpoint = inspect_IL_checkpoint
load_deployment_checkpoint = lambda path, device='cpu': load_IL_checkpoint(path, device, deployment=True)
