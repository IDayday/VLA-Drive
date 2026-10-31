"""Current-best IL export + native RL adapters, with the original deploy math.

B/D's historical native IL recovery inventories need not still exist.  A
strict, protected standalone IL export supplies their complete deployed model;
the original native ancestry is retained as provenance, never used as a path
from which to restore optimizer state.
"""
import copy
import json
import os
from pathlib import Path
import uuid

import torch

from starVLA.rl.qwen_world_compare import deployment as original
from starVLA.rl.qwen_world_compare.contracts import file_sha, fingerprint
from starVLA.rl.qwen_world_compare.loading import inspect_RL_checkpoint
from starVLA.rl.qwen_world_compare.rl_scope import install_lora
from .parents import load_parent_model

EXPORT_SCHEMA = original.EXPORT_SCHEMA
RESEARCH_SCHEMA = 'ABCD_current_best_RL_research_v1'
# New exports are bound to this runtime and use every original strict check.
inspect_export = original.inspect_export
prune_deployment = original.prune_deployment


def parent_export_path(meta):
    """Resolve the *export* parent; never fall back to a retired native path."""
    lineage = meta['IL_parent'].get('export_parent', {})
    path = meta.get('parent_export') or lineage.get('path')
    if not path or not lineage.get('model_sha256') or not lineage.get('export_sha256'):
        raise ValueError('Current-best deployment needs its sealed export-parent provenance')
    if Path(path).resolve() != Path(lineage['path']).resolve():
        raise ValueError('Deployment and IL ancestry disagree on the export parent')
    return str(Path(path).resolve())


def load_export(path, device='cpu'):
    """Strict new policy, or a selected IL export's explicit narrow migration."""
    root = Path(path)
    header = json.loads((root/'export.json').read_text())
    if header.get('RL_parent') is None and header.get('research_schema') != RESEARCH_SCHEMA:
        # This is the already sealed IL parent itself, not an RL deployment.
        # load_parent_model rechecks all bytes/libraries and the one reviewed
        # source delta; no broad verify_source=False loading path is exposed.
        model, parent = load_parent_model(root, device=device)
        meta = copy.deepcopy(header)
        meta.update(IL_parent=parent, parent_export=str(root.resolve()),
                    research_schema=RESEARCH_SCHEMA,
                    parent_source_migration=parent['export_parent']['source_migration'])
        return model, meta
    return original.load_export(root, device=device)


def load_native_policy(il_checkpoint, rl_checkpoint=None, device='cpu'):
    """Load selected IL *export*, then exact unmerged native RL adapter closure."""
    model, parent = load_parent_model(il_checkpoint, device='cpu')
    rl_info = rl_meta = lora_cfg = None
    if rl_checkpoint is not None:
        rl_info = inspect_RL_checkpoint(rl_checkpoint)
        rl_meta = rl_info['identity']
        other = rl_meta.get('IL_parent', {})
        if (rl_meta['architecture_id'] != parent['identity']['architecture_id']
                or other.get('model_sha256') != parent['model_sha256']
                or other.get('identity') != parent['identity']
                or fingerprint(other.get('export_parent')) != fingerprint(parent['export_parent'])):
            raise ValueError('RL adapters must retain the exact selected own-architecture IL export parent')
        lora_cfg = copy.deepcopy(rl_meta['recipe']['grpo']['lora'])
        install_lora(model, r=lora_cfg['r'], alpha=lora_cfg['alpha'], dropout=lora_cfg['dropout'])
        payload = torch.load(Path(rl_checkpoint)/'optimizer.pt', map_location='cpu', weights_only=False)
        original.load_adapter_state(model, payload['adapters'])
    sources = dict(parent=parent, RL_info=rl_info, RL_parent=rl_meta, rl_lora=lora_cfg)
    return model.eval().requires_grad_(False).to(device), sources


def export_policy(il_checkpoint, output, rl_checkpoint=None):
    """Seal a standalone single-trajectory policy without modifying its parent."""
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
        parent_export=str(Path(il_checkpoint).resolve()), research_schema=RESEARCH_SCHEMA,
        IL_updates=parent['seal']['updates'], RL_parent=rl_meta, rl_lora=lora_cfg,
        model_sha256=file_sha(staging/'model.pt'), processor=processor,
        prediction_type='x0', action_shape=[8, 3],
        deployment_inference='single_candidate_cleanx_K10_shift5',
        deployment_components=model.deployment_components(),
        runtime_source_files=original._source_runtime(),
        runtime_libraries=original._library_runtime(),
        no_teacher_future_candidate_metric_or_train_heads=True,
        qualification=bool(parent['identity']['qualification'] or (rl_meta and rl_meta.get('purpose') != 'formal')),
        LoRA_merge=False)
    if rl_info is not None:
        meta.update(RL_checkpoint=rl_info['path'], RL_updates=rl_info['seal']['updates'],
                    RL_checkpoint_files_sha256=rl_info['seal']['files_sha256'])
    (staging/'export.json').write_text(json.dumps(meta, indent=2, allow_nan=False)+'\n')
    (staging/'COMPLETE').write_text(json.dumps(dict(export_sha256=file_sha(staging/'export.json')))+'\n')
    inspect_export(staging)
    os.rename(staging, output)
    return meta


export_checkpoint = export_policy
