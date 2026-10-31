"""Explicit lossless execution lineage; scientific identities remain strict.

The original native loader validates the OLD identity and every optimizer/RNG
shard. Only the named execution patch may differ. New checkpoints publish a new
identity; no old checkpoint/COMPLETE/rank state is edited to spoof compatibility.
"""
import copy
import json
from pathlib import Path
from .contracts import fingerprint,file_sha


SCHEMA='world_compare_execution_handoff_v1'
ALLOWED_PATCH={
    'starVLA/model/modules/qwen_world_compare/qwen_flow_adapter.py',
    'starVLA/model/modules/qwen_world_compare/execution.py',
    'starVLA/rl/qwen_world_compare/batching.py',
    'starVLA/rl/qwen_world_compare/prefetch.py',
    'starVLA/rl/qwen_world_compare/execution_migration.py',
    'tools/qwen_world_compare/train_il.py',
    'tools/qwen_world_compare/benchmark_execution.py',
    'tools/qwen_world_compare/qualify_prefetch.py',
    'tools/qwen_world_compare/analyze_execution_pair.py',
    'tools/qwen_world_compare/prepare_execution_handoff.py',
    # Reuse the already qualified operations source fcd43eb; these files do
    # not implement actor/loss/optimizer mathematics.
    'tools/qwen_world_compare/campaign.py',
    'tools/qwen_world_compare/dispatch.py',
    'tools/qwen_world_compare/profile.py',
    'tools/qwen_world_compare/qualify.py',
    'tools/qwen_world_compare/qualify_rl.py',
    'starVLA/rl/qwen_world_compare/profiling.py',
}


def scientific_identity(value):
    value=copy.deepcopy(value);value.pop('source_files',None)
    value['execution'].pop('execution_tuning',None)
    # Qualification does not get authority to alter scientific fields.
    value.pop('qualification',None)
    return value


def validate_patch(before,after,*,qualification=False):
    if scientific_identity(before)!=scientific_identity(after):
        raise ValueError('Execution patch changed scientific/model/data/optimizer/precision identity')
    if not qualification and before['qualification']!=after['qualification']:
        raise ValueError('Qualification weights cannot become formal initialization')
    old,new=before['source_files'],after['source_files']
    changes={name:dict(before=old.get(name),after=new.get(name))for name in set(old)|set(new)if old.get(name)!=new.get(name)}
    if not changes or not set(changes).issubset(ALLOWED_PATCH):
        raise ValueError('Unapproved imported dependency changed: '+repr(sorted(set(changes)-ALLOWED_PATCH)))
    if any(row['after']is None for row in changes.values()):raise ValueError('Execution migration cannot delete an imported dependency')
    return dict(sorted(changes.items()))


def validate_receipt(path,new_identity,*,resume=None):
    path=Path(path);value=json.loads(path.read_text());body=dict(value);seal=body.pop('sha256',None)
    if seal!=fingerprint(body)or value.get('schema')!=SCHEMA or value.get('status')!='PASS':
        raise ValueError('Unsealed/unqualified execution handoff')
    if new_identity!=value['new_identity']:raise ValueError('Execution receipt does not bind this exact runtime identity')
    parent=Path(value['parent_checkpoint'])
    old=json.loads((parent/'identity.json').read_text())
    if (old!=value['old_identity']or file_sha(parent/'COMPLETE')!=value['parent_COMPLETE_sha256']
            or file_sha(parent/'files.json')!=value['parent_inventory_sha256']):
        raise ValueError('Pinned original native recovery boundary changed')
    if validate_patch(old,new_identity)!=value['changes']:raise ValueError('Execution patch differs from reviewed receipt')
    if new_identity['qualification']or new_identity['world']!=8:raise ValueError('Formal handoff preserves eight-rank formal ancestry')
    for proof in value['proofs']:
        p=Path(proof['path']);actual=json.loads(p.read_text())
        if file_sha(p)!=proof['sha256']or actual.get('status')!='PASS':raise ValueError('Equivalence proof changed')
    if not value['proofs']:raise ValueError('No actual equivalence evidence')
    if resume is not None:
        actual=json.loads((Path(resume)/'identity.json').read_text())
        if actual not in(old,new_identity):raise ValueError('Resume checkpoint is outside this exact execution lineage')
        if actual==old and Path(resume).resolve()!=parent.resolve():raise ValueError('First handoff must restore the pinned committed boundary')
    return value


def restore(path,engine,new_identity,builder,receipt):
    from starVLA.rl.qwen_world_postcot import checkpoint as cp
    value=validate_receipt(receipt,new_identity,resume=path)
    actual=json.loads((Path(path)/'identity.json').read_text())
    # Keep native strict identity, tensor keys/dtypes, Adam/scheduler counters,
    # rank RNG and builder progress checks. No strict=False or field skipping.
    return cp.restore(path,engine,actual,builder)
