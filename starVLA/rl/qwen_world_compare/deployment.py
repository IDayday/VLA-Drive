"""Standalone A/B/C/D IL/RL exports; prune training-only state, keep LoRA."""
import copy
import json
import inspect
import os
from pathlib import Path
import uuid

import torch

from .contracts import file_sha, fingerprint
from .loading import (IL_SCHEMA, RL_SCHEMA, contained_file, inspect_IL_checkpoint,
                      inspect_RL_checkpoint, load_IL_checkpoint, model_from_identity,
                      strict_tensor_load)
from .rl_scope import install_lora

EXPORT_SCHEMA = 'qwen_world_compare_single_trajectory_export_v1'


def prune_deployment(model, *, copy_model=True):
    """Default works on a copy; private newly loaded models can prune in place."""
    model = copy.deepcopy(model) if copy_model else model
    if model.architecture_id == 'A':
        model.decision.prune_training_components()
    elif model.architecture_id == 'B':
        model.decision.strip_training_only()
    elif model.architecture_id == 'C':
        model.decision.prune_training_components()
    elif model.architecture_id == 'D':
        model.decision.discard_training_only()
    else:
        raise ValueError('Cannot prune an unknown architecture')
    if hasattr(model, 'readouts'):
        del model.readouts
    model.train_heads = False
    state = model.state_dict()
    if any(name.startswith('readouts.') for name in state):
        raise ValueError('A deployment must not retain training readouts')
    if model.architecture_id == 'A' and any(name.startswith('decision.world_stream.') for name in state):
        raise ValueError('A training-only world stream leaked into deployment')
    if model.architecture_id == 'C' and any('.post_action_bridges.' in name or name.endswith('.post_queries') for name in state):
        raise ValueError('C reverse bridge/R queries leaked into deployment')
    if model.architecture_id == 'D' and 'decision.r_embeddings' in state:
        raise ValueError('D R query leaked into deployment')
    return model.eval().requires_grad_(False)


def load_adapter_state(model, adapters):
    wanted = {'action_expert.'+name: parameter for name, parameter in model.named_parameters()
              if name.endswith(('.lora_A', '.lora_B'))}
    if not wanted or set(adapters) != set(wanted):
        raise ValueError('RL export adapter closure differs from the exact architecture scope')
    with torch.no_grad():
        for name, target in wanted.items():
            value = adapters[name]
            if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                    or value.shape != target.shape or value.dtype != target.dtype
                    or not torch.isfinite(value).all()):
                raise ValueError('Invalid unmerged adapter tensor: '+name)
            target.copy_(value)


def _source_runtime():
    root = Path(__file__).resolve().parents[3]
    paths = list((root/'starVLA/model/modules/qwen_world_compare').glob('*.py'))
    paths += [root/name for name in (
        'starVLA/model/modules/qwen_world_postcot/actor.py',
        'starVLA/model/modules/qwen_world_postcot/memory.py',
        'starVLA/model/modules/qwen_world_postcot/flow.py',
        'starVLA/model/modules/qwen_world_postcot/qwen_encoder.py',
        'starVLA/model/modules/qwen_sourceflow/qwen_encoder.py',
        'starVLA/model/modules/qwen_sourceflow/action_codec.py',
        'starVLA/rl/qwen_world_compare/rl_scope.py')]
    return {str(path.relative_to(root)): file_sha(path) for path in sorted(set(paths))}


def _library_runtime():
    import transformers
    from transformers.models.qwen3_vl import modeling_qwen3_vl
    return dict(torch=torch.__version__, transformers=transformers.__version__,
                native_qwen_modeling_sha256=file_sha(inspect.getfile(modeling_qwen3_vl)),
                parameters='float32', model_forward='bfloat16', action_state_ODE='float32')


def inspect_export(path, *, verify_source=True):
    root = Path(path)
    meta = json.loads((root/'export.json').read_text())
    complete = json.loads((root/'COMPLETE').read_text())
    if (meta.get('schema') != EXPORT_SCHEMA or complete.get('export_sha256') != file_sha(root/'export.json')
            or meta.get('model_sha256') != file_sha(contained_file(root, 'model.pt'))
            or meta.get('architecture_id') not in ('A', 'B', 'C', 'D')
            or meta.get('prediction_type') != 'x0' or meta.get('action_shape') != [8, 3]
            or meta.get('deployment_inference') != 'single_candidate_cleanx_K10_shift5'
            or meta.get('LoRA_merge') is not False
            or meta.get('no_teacher_future_candidate_metric_or_train_heads') is not True
            or type(meta.get('qualification')) is not bool):
        raise ValueError('Changed four-framework standalone export contract')
    parent = meta.get('IL_parent', {})
    if (parent.get('identity', {}).get('schema') != IL_SCHEMA
            or parent['identity'].get('architecture_id') != meta['architecture_id']):
        raise ValueError('Export architecture/IL ancestry mismatch')
    from starVLA.rl.qwen_sourceflow.export import PROCESSOR_FILES
    if set(meta.get('processor', {})) != set(PROCESSOR_FILES):
        raise ValueError('Incomplete standalone official processor inventory')
    for name, sha in meta['processor'].items():
        if file_sha(contained_file(root/'processor', name)) != sha:
            raise ValueError('Changed standalone processor: '+name)
    if bool(meta.get('rl_lora')) != bool(meta.get('RL_parent')):
        raise ValueError('Export RL adapter/source lineage disagree')
    if meta.get('RL_parent') is not None:
        rl = meta['RL_parent']
        if (rl.get('schema') != RL_SCHEMA or rl.get('architecture_id') != meta['architecture_id']
                or rl.get('IL_parent', {}).get('model_sha256') != parent['model_sha256']):
            raise ValueError('Export RL references a different architecture or IL parent')
    expected_qualification = parent['identity']['qualification'] or bool(
        meta.get('RL_parent') and meta['RL_parent'].get('purpose') != 'formal')
    if meta['qualification'] != expected_qualification:
        raise ValueError('A qualification export cannot become a formal model')
    if verify_source and meta.get('runtime_source_files') != _source_runtime():
        raise ValueError('Export model computation source differs from its locked runtime')
    if meta.get('runtime_libraries') != _library_runtime():
        raise ValueError('Export native Qwen/library/numerical runtime changed')
    return meta


def load_native_policy(il_checkpoint, rl_checkpoint=None, device='cpu'):
    """Strict native IL + optional native adapters, without serialization/merge."""
    model, parent = load_IL_checkpoint(il_checkpoint, deployment=True)
    rl_info = rl_meta = lora_cfg = None
    if rl_checkpoint is not None:
        rl_info = inspect_RL_checkpoint(rl_checkpoint)
        rl_meta = rl_info['identity']
        if (rl_meta['architecture_id'] != parent['identity']['architecture_id']
                or rl_meta.get('IL_parent', {}).get('model_sha256') != parent['model_sha256']
                or rl_meta['IL_parent'].get('identity') != parent['identity']):
            raise ValueError('RL export must load the same own-architecture permanent IL parent')
        lora_cfg = copy.deepcopy(rl_meta['recipe']['grpo']['lora'])
        install_lora(model, r=lora_cfg['r'], alpha=lora_cfg['alpha'], dropout=lora_cfg['dropout'])
        payload = torch.load(Path(rl_checkpoint)/'optimizer.pt', map_location='cpu', weights_only=False)
        load_adapter_state(model, payload['adapters'])
    return model.eval().requires_grad_(False).to(device), dict(parent=parent, RL_info=rl_info,
                                                            RL_parent=rl_meta, rl_lora=lora_cfg)


def export_policy(il_checkpoint, output, rl_checkpoint=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Deployment export must never overwrite an existing asset')
    model, sources = load_native_policy(il_checkpoint, rl_checkpoint)
    parent, rl_info = sources['parent'], sources['RL_info']
    rl_meta, lora_cfg = sources['RL_parent'], sources['rl_lora']
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name+'.incomplete.'+uuid.uuid4().hex)
    staging.mkdir(parents=True, exist_ok=False)
    torch.save(model.state_dict(), staging/'model.pt')
    from starVLA.rl.qwen_sourceflow.export import package_processor
    processor = package_processor(staging/'processor')
    meta = dict(schema=EXPORT_SCHEMA, architecture_id=parent['identity']['architecture_id'],
        model_config=copy.deepcopy(parent['identity']['model_config']),
        qwen_config=copy.deepcopy(parent['identity']['qwen_config']), IL_parent=parent,
        IL_updates=parent['seal']['updates'], RL_parent=rl_meta, rl_lora=lora_cfg,
        model_sha256=file_sha(staging/'model.pt'), processor=processor,
        prediction_type='x0', action_shape=[8, 3],
        deployment_inference='single_candidate_cleanx_K10_shift5',
        deployment_components=model.deployment_components(),
        runtime_source_files=_source_runtime(),
        runtime_libraries=_library_runtime(),
        no_teacher_future_candidate_metric_or_train_heads=True,
        qualification=bool(parent['identity']['qualification'] or (rl_meta and rl_meta.get('purpose') != 'formal')),
        LoRA_merge=False)
    if rl_info is not None:
        meta.update(RL_checkpoint=rl_info['path'], RL_updates=rl_info['seal']['updates'],
                    RL_checkpoint_files_sha256=rl_info['seal']['files_sha256'])
    (staging/'export.json').write_text(json.dumps(meta, indent=2, allow_nan=False)+'\n')
    (staging/'COMPLETE').write_text(json.dumps(dict(export_sha256=file_sha(staging/'export.json')))+'\n')
    inspect_export(staging)
    output.parent.mkdir(parents=True, exist_ok=True)
    os.rename(staging, output)
    return meta


def load_export(path, device='cpu'):
    root = Path(path)
    meta = inspect_export(root)
    # Reconstruct and prune exactly the same full topology before strict load.
    model = model_from_identity(meta['IL_parent']['identity'], train_heads=True)
    model = prune_deployment(model, copy_model=False)
    if meta['rl_lora']:
        cfg = meta['rl_lora']
        install_lora(model, r=cfg['r'], alpha=cfg['alpha'], dropout=cfg['dropout'])
    state = torch.load(root/'model.pt', map_location='cpu', weights_only=True, mmap=True)
    strict_tensor_load(model, state)
    return model.eval().requires_grad_(False).to(device), meta


export_checkpoint = export_policy
