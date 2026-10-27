"""Explicit user-pinned IL/RL Test observations; no training/selection grant."""
import json
from pathlib import Path
from .contracts import fingerprint


def policy_binding(parent, export):
    rl = export.get('RL_parent')
    if parent['identity'].get('qualification') is not False or export.get('qualification'):
        raise ValueError('Qualification weights cannot enter a formal Test snapshot')
    if rl is None:
        return dict(policy_kind='IL', checkpoint=parent['path'], updates=parent['seal']['updates'],
                    checkpoint_files_sha256=parent['seal']['files_sha256'], training_scope='joint_IL')
    if (rl.get('purpose') != 'formal' or rl.get('schema') not in
            ('world_postcot_cleanx_canonical_RL_v1', 'world_postcot_cleanx_scope_RL_v1')):
        raise ValueError('Unknown/nonformal native World RL identity')
    return dict(policy_kind='RL', checkpoint=export['RL_checkpoint'], updates=export['RL_updates'],
                checkpoint_files_sha256=export['RL_checkpoint_files_sha256'],
                training_scope=rl['recipe'].get('adaptation', 'ATTENTION_LORA'))


def validate_snapshot(path, *, parent, export, data_sha256, tokens, population_sha256,
                      protocol_sha256, output, seed):
    record = json.loads(Path(path).read_text()); body = dict(record)
    if body.pop('sha256', None) != fingerprint(body):
        raise ValueError('Changed user snapshot registration')
    expected = dict(schema='world_user_snapshot_NAVTEST_v1', **policy_binding(parent, export),
        IL_parent_checkpoint=parent['path'], IL_parent_model_sha256=parent['model_sha256'],
        model_sha256=export['model_sha256'], data_sha256=data_sha256,
        token_sha256=fingerprint(tokens), population_sha256=population_sha256,
        protocol_sha256=protocol_sha256, output=str(Path(output).resolve()), seed=2101,
        count=12146, prediction_type='x0', inference='single_candidate_cleanx_K10_shift5',
        used_for_parent_selection=False, used_for_actor_update=False)
    if (seed != 2101 or len(tokens) != 12146 or len(set(tokens)) != 12146
            or any(record.get(k) != v for k, v in expected.items())):
        raise ValueError('Test snapshot differs from its exact pinned model/data/protocol')
    return record


def load_native_snapshot(parent, export):
    """Restore the native trainable closure, independently of export mapping."""
    import torch
    from .loading import load_deployment_checkpoint
    from .rl_actor import CleanXRecipeActor
    from .scope_actor import ScopeRecipeActor
    from starVLA.rl.simwam_recipe.lora_adapter import restore_adapters
    model, info = load_deployment_checkpoint(parent['path'], 'cpu')
    rl = export.get('RL_parent')
    if rl is not None:
        root = Path(export['RL_checkpoint'])
        identity = json.loads((root / 'identity.json').read_text())
        if identity != rl:
            raise ValueError('Native RL identity differs from exported lineage')
        cfg = identity['recipe']
        actor = (ScopeRecipeActor(model.actor, cfg, scene_provider=model.scene_provider,
                                 predictive_provider=model.predictive_provider)
                 if identity['schema'] == 'world_postcot_cleanx_scope_RL_v1'
                 else CleanXRecipeActor(model.actor, cfg))
        payload = torch.load(root / 'optimizer.pt', map_location='cpu', weights_only=False)['adapters']
        if set(payload) != set(identity['parameter_contract']['trainable_names']):
            raise ValueError('Native trainable memory/LoRA closure is incomplete')
        if any(v.dtype != torch.float32 or not bool(torch.isfinite(v).all()) for v in payload.values()):
            raise ValueError('Native trainable tensors must be finite FP32')
        restore_adapters(actor, payload)
        model.eval().requires_grad_(False)
        del actor, payload
    return model.eval().requires_grad_(False).cuda(), info
