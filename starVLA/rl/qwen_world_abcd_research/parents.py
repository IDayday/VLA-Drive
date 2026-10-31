"""Own-architecture current-best IL exports, with explicit source migration.

No IL optimizer is needed for fresh RL. A retired native recovery inventory is
never recreated or presented as recoverable. Existing export seals stay intact.
"""
import copy
import json
from pathlib import Path

import torch

from starVLA.rl.qwen_world_compare.contracts import ROOT, file_sha, fingerprint
from starVLA.rl.qwen_world_compare import deployment as original
from starVLA.rl.qwen_world_compare.loading import model_from_identity, strict_tensor_load

SELECTION_SCHEMA = 'user_authorized_ABCD_current_NAVTEST_best_parent_selection_v1'
ADAPTER = 'starVLA/model/modules/qwen_world_compare/qwen_flow_adapter.py'
ADAPTER_MIGRATION = (
    '3f6e0abf84b5f0afed34f710ea07040516342c2e67597b6fcf0f1278ebc15043',
    'e95e1fcccd1aa37f8b095a83e0b134adbecc7350739b6701cd6b007042db0367',
)


def validate_best_selection(path, parent_export, architecture):
    value = json.loads(Path(path).read_text())
    body = dict(value); seal = body.pop('sha256', None)
    if (seal != fingerprint(body) or value.get('schema') != SELECTION_SCHEMA
            or value.get('Test_used_for_research_selection') is not True
            or value.get('immutable_for_registered_runs') is not True):
        raise ValueError('Expected sealed user-authorized current-best selection')
    selected = value['selected'][architecture]
    meta = json.loads((Path(parent_export)/'export.json').read_text())
    complete = json.loads((Path(parent_export)/'COMPLETE').read_text())
    if (Path(selected['deployment_parent']).resolve() != Path(parent_export).resolve()
            or meta['architecture_id'] != architecture or meta.get('RL_parent') is not None
            or complete['export_sha256'] != file_sha(Path(parent_export)/'export.json')
            or meta['model_sha256'] != selected['deployment_model_sha256']
            or meta['IL_parent']['model_sha256'] != selected['native_model_sha256']
            or meta['IL_parent']['seal']['updates'] != selected['updates']
            or selected.get('native_export_parity', {}).get('status') != 'PASS'):
        raise ValueError('Wrong architecture, selected checkpoint, export or parity')
    return selected


def inspect_parent_export(path):
    # Preserve every original export/file/library check, then perform our
    # narrower computation-source check instead of accepting arbitrary drift.
    meta = original.inspect_export(path, verify_source=False)
    if meta.get('qualification') is not False or meta.get('RL_parent') is not None:
        raise ValueError('Parent must be a complete formal IL export, never RL')
    expected = meta['runtime_source_files']; actual = original._source_runtime()
    added = set(actual)-set(expected)
    helper = 'starVLA/model/modules/qwen_world_compare/execution.py'
    helper_sha = '4f1cef430829f29b3c2938486c0bbc39782812be17255dbb784519d789270f22'
    if (set(expected)-set(actual) or added not in (set(), {helper})
            or (added and actual[helper] != helper_sha)):
        raise ValueError('Parent runtime source inventory changed')
    differences = {name: (expected[name], actual[name]) for name in expected if expected[name] != actual[name]}
    if differences not in ({}, {ADAPTER: ADAPTER_MIGRATION}):
        raise ValueError('Unreviewed parent execution-source change: '+repr(differences))
    # The sole permitted delta adds a checkpoint_short_suffix default=True.
    # Its True branch is the old activation-checkpoint execution; no weights,
    # shapes, masks, numerical kernels or denoising formula change.
    meta = copy.deepcopy(meta)
    meta['parent_source_migration'] = dict(
        differences=differences, added_execution_helper={name:actual[name] for name in added}, checkpoint_short_suffix=True,
        source_original_seals_unchanged=True,
        scope='one reviewed default-True activation-checkpoint execution flag; no model math/parameters change',
    )
    return meta


def load_parent_model(path, device='cpu'):
    root = Path(path)
    meta = inspect_parent_export(root)
    model = model_from_identity(meta['IL_parent']['identity'], train_heads=True)
    model = original.prune_deployment(model, copy_model=False)
    if meta['architecture_id'] == 'D':
        if getattr(model.decision, 'checkpoint_short_suffix', True) is not True:
            raise ValueError('D parent migration requires original checkpoint semantics')
    state = torch.load(root/'model.pt', map_location='cpu', weights_only=True, mmap=True)
    strict_tensor_load(model, state)
    info = copy.deepcopy(meta['IL_parent'])
    info['export_parent'] = dict(path=str(root.resolve()), model_sha256=meta['model_sha256'],
        export_sha256=file_sha(root/'export.json'), source_migration=meta['parent_source_migration'],
        model_load='strict complete deployed closure; original native recovery not required for fresh RL')
    return model.eval().requires_grad_(False).to(device), info
