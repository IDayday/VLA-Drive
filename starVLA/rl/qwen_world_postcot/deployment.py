"""Strict single-trajectory exports retaining RL LoRA wrappers for BF16 parity."""
import json
from copy import deepcopy
import os
from pathlib import Path

import torch

from .contracts import file_sha, fingerprint
from .loading import load_deployment_checkpoint
from starVLA.model.modules.qwen_world_postcot.model import QwenWorldPostCoT


EXPORT_SCHEMA = 'world_postcot_single_trajectory_export_v1'


def _contained_file(root, name):
    path = root/name
    if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('Unsafe or missing sealed export/checkpoint file: '+name)
    return path


def inspect_export(path):
    root = Path(path)
    meta = json.loads((root/'export.json').read_text())
    seal = json.loads((root/'COMPLETE').read_text())
    if (meta.get('schema') != EXPORT_SCHEMA or seal.get('export_sha256') != file_sha(root/'export.json')
            or meta['model_sha256'] != file_sha(_contained_file(root, 'model.pt'))
            or meta.get('prediction_type') != 'x0' or meta.get('action_shape') != [8, 3]
            or meta.get('LoRA_merge') is not False
            or meta.get('no_teacher_future_candidate_metric_or_train_heads') is not True
            or meta.get('predictive_memory_retained') is not True):
        raise ValueError('Deployment export schema, clean-x contract or receipt changed')
    from starVLA.rl.qwen_sourceflow.export import PROCESSOR_FILES
    if set(meta['processor']) != set(PROCESSOR_FILES):
        raise ValueError('Deployment processor inventory is incomplete or unexpected')
    for name, digest in meta['processor'].items():
        if file_sha(_contained_file(root/'processor', name)) != digest:
            raise ValueError('Deployment processor changed: '+name)
    if bool(meta['rl_lora']) != bool(meta['RL_parent']):
        raise ValueError('RL adapter configuration and source lineage disagree')
    if type(meta['qualification']) is not bool:
        raise ValueError('Export requires explicit qualification status')
    expected_qualification = bool(meta['IL_parent']['identity']['qualification'])
    if meta['RL_parent'] is not None:
        expected_qualification |= meta['RL_parent'].get('purpose') != 'formal'
    if meta['qualification'] != expected_qualification:
        raise ValueError('An RL qualification must remain diagnostic after export')
    return meta


def load_lora_adapter_state(head, adapters):
    """Load the exact unmerged keys emitted by native RecipeActor checkpointing."""
    state = head.state_dict()
    wanted = {'action_expert.'+name for name, _ in head.named_parameters() if 'lora_' in name}
    if not wanted or set(adapters) != wanted:
        raise ValueError('Incomplete/unexpected LoRA adapter closure')
    with torch.no_grad():
        for name, value in adapters.items():
            key = name.removeprefix('action_expert.')
            target = state[key]
            if (not isinstance(value, torch.Tensor) or value.shape != target.shape
                    or value.dtype != target.dtype or value.dtype != torch.float32
                    or not torch.isfinite(value).all()):
                raise ValueError('LoRA tensor shape/dtype/finite contract changed: '+name)
            target.copy_(value)
    head.load_state_dict(state, strict=True)


def load_export(path, device='cpu'):
    root = Path(path)
    meta = inspect_export(root)
    model = QwenWorldPostCoT(None, config=deepcopy(meta['model_config']),
        checkpoint_config=deepcopy(meta['qwen_config']), train_heads=False)
    if meta['rl_lora']:
        from .rl_actor import install_deployment_lora
        install_deployment_lora(model.actor, meta['rl_lora'])
    state = torch.load(root/'model.pt', map_location='cpu', weights_only=True)
    target = model.state_dict()
    for name, value in state.items():
        if name in target and (value.dtype != target[name].dtype or value.shape != target[name].shape):
            raise ValueError('Strict export load would silently convert trained tensor: '+name)
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False).to(device), meta


def load_scope_trainable_state(model,adapters,adaptation):
    """Strict native optimizer payload -> inference-only policy, including memory.

    The inherited checkpoint calls this field `adapters`; for the memory arm it
    deliberately also stores every fully trained provider parameter.
    """
    from .scope_contracts import MEMORY,TIME
    if adaptation not in (MEMORY,TIME):raise ValueError('Unknown deployment scope')
    action={n:v for n,v in adapters.items() if n.startswith('action_expert.')}
    expected_memory={}
    if adaptation==MEMORY:
        for prefix in ('scene_provider','predictive_provider'):
            expected_memory.update({prefix+'.'+n:p for n,p in getattr(model,prefix).named_parameters()})
    if set(adapters)!=set(action)|set(expected_memory):raise ValueError('Incomplete/unexpected trainable memory payload')
    load_lora_adapter_state(model.actor,action)
    with torch.no_grad():
        for name,target in expected_memory.items():
            value=adapters[name]
            if (not isinstance(value,torch.Tensor) or value.shape!=target.shape or value.dtype!=torch.float32
                    or target.dtype!=value.dtype or not torch.isfinite(value).all()):
                raise ValueError('Memory deployment tensor contract changed: '+name)
            target.copy_(value)


def export_checkpoint(checkpoint, output, rl_checkpoint=None):
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    model, parent = load_deployment_checkpoint(checkpoint)
    # Model constructors may normalize Qwen config in-place. The source receipt
    # must retain the original sealed bytes, never that runtime working copy.
    from .loading import inspect_checkpoint
    sealed_parent = inspect_checkpoint(checkpoint, verify_files="model")
    if parent["model_sha256"] != sealed_parent["model_sha256"]:
        raise ValueError("Source checkpoint changed while exporting")
    parent = sealed_parent
    rlmeta = rlcfg = None
    if rl_checkpoint:
        from .rl_actor import install_deployment_lora
        root = Path(rl_checkpoint)
        rlmeta = json.loads((root/'identity.json').read_text())
        seal = json.loads((root/'COMPLETE').read_text())
        files = json.loads((root/'files.json').read_text())
        body = dict(rlmeta); saved = body.pop('sha256', None)
        if (fingerprint(body) != saved or seal.get('schema') != 2
                or seal['files_sha256'] != file_sha(root/'files.json')
                or rlmeta.get('schema') not in ('world_postcot_cleanx_canonical_RL_v1','world_postcot_cleanx_scope_RL_v1')
                or rlmeta.get('prediction_type') != 'x0'
                or seal['world'] != rlmeta['world']):
            raise ValueError('Wrong or changed clean-x native RL checkpoint schema')
        required = {'identity.json', 'optimizer.pt', 'transaction.json'} | {
            f'rank{i}.pt' for i in range(rlmeta['world'])}
        if not required.issubset(files):
            raise ValueError('RL checkpoint lacks a complete native rank/adapter receipt')
        for name, digest in files.items():
            if file_sha(_contained_file(root, name)) != digest:
                raise ValueError('Changed RL state: '+name)
        if (rlmeta['IL_parent']['model_sha256'] != parent['model_sha256']
                or rlmeta['IL_parent']['identity'] != parent['identity']):
            raise ValueError('RL and frozen full-framework IL parents differ')
        rlcfg = rlmeta['recipe']['grpo']
        install_deployment_lora(model.actor, rlcfg)
        adapters = torch.load(root/'optimizer.pt', map_location='cpu', weights_only=False)['adapters']
        if rlmeta.get('schema')=='world_postcot_cleanx_scope_RL_v1':
            load_scope_trainable_state(model,adapters,rlmeta['recipe']['adaptation'])
        else:load_lora_adapter_state(model.actor, adapters)
    staging = out.with_name(out.name+'.building')
    staging.mkdir(parents=True, exist_ok=False)
    torch.save(model.state_dict(), staging/'model.pt')
    from starVLA.rl.qwen_sourceflow.export import package_processor
    processor = package_processor(staging/'processor')
    diagnostic = bool(parent['identity']['qualification']) or bool(
        rlmeta is not None and rlmeta.get('purpose') != 'formal')
    meta = dict(schema=EXPORT_SCHEMA, model_config=parent['identity']['model_config'],
        qwen_config=parent['identity']['qwen_config'], IL_parent=parent,
        RL_parent=rlmeta, rl_lora=rlcfg, model_sha256=file_sha(staging/'model.pt'),
        processor=processor, prediction_type='x0', action_shape=[8, 3],
        deployment='current legal3 cameras/navigation/ego → student memory → clean-x K10shift5; one trajectory',
        no_teacher_future_candidate_metric_or_train_heads=True, predictive_memory_retained=True,
        qualification=diagnostic, LoRA_merge=False)
    meta['IL_updates']=parent['seal']['updates']
    if rlmeta is not None:
        meta.update(RL_checkpoint=str(root.resolve()),RL_updates=seal['updates'],RL_checkpoint_files_sha256=seal['files_sha256'])
    (staging/'export.json').write_text(json.dumps(meta, indent=2))
    (staging/'COMPLETE').write_text(json.dumps(dict(export_sha256=file_sha(staging/'export.json'))))
    # Inspect final staged bytes before publication, including processor files.
    inspect_export(staging)
    os.rename(staging, out)
    return meta
