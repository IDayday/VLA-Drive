"""Repartition an own-run complete native FPO state between 16 and 32 ranks."""
import copy
import json
import os
from pathlib import Path
import shutil
import uuid
import torch
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.qwen_world_compare.contracts import fingerprint, file_sha
from starVLA.rl.d44_native_rl.recovery import checkpoint, material_digest
from . import migration
from .virtual_lanes import slots


def repartition(payloads, source_world, target_world):
    if len(payloads)!=source_world:raise ValueError('Missing original ranks')
    pending=payloads[0]['rollout']is not None
    if any((p['rollout']is not None)!=pending for p in payloads):
        raise ValueError('Mixed pending and READY states')
    locations={slot:(r,j)for r in range(source_world)for j,slot in enumerate(slots(r,source_world))}
    if set(locations)!=set(range(64)):raise ValueError('Original scene coverage changed')
    result=[]
    for rank in range(target_world):
        indices=[locations[s]for s in slots(rank,target_world)]
        original=payloads[indices[0][0]]
        own=copy.deepcopy(original)
        if pending:
            batch=own['rollout']
            batch['scenes']=[copy.deepcopy(payloads[r]['rollout']['scenes'][j])for r,j in indices]
            batch['digest']=material_digest(batch)
        result.append(own)
    return result


def prepare(checkpoint_path, source_code, output, *, target_world):
    cp=Path(checkpoint_path).resolve(strict=True);source=Path(source_code).resolve(strict=True)
    root=Path(output).resolve()
    if root.exists()or root.is_relative_to(cp.parent.parent):raise ValueError('New independent execution root required')
    old=migration.sealed(json.loads((cp/'identity.json').read_text()))
    source_world=old['world'];meta=old.get('performance_execution',{})
    if (source_world not in(16,32)or target_world not in(16,32)or source_world==target_world
            or old.get('qualification')is not False or meta.get('profile')!='fast'
            or 'verified_improvement'in old['recipe']):raise ValueError('Only own mature qualified FPO16/32 reshard')
    migration.validate(cp.parent.parent,old,actual_inventory=old['source_files'])
    for name,digest in old['source_files'].items():
        if not(source/name).resolve().is_relative_to(source)or file_sha(source/name)!=digest:
            raise ValueError('Original execution implementation changed')
    seal=json.loads((cp/'COMPLETE').read_text());files=json.loads((cp/'files.json').read_text())
    if seal['world']!=source_world or file_sha(cp/'files.json')!=seal['files_sha256']:
        raise ValueError('Original checkpoint seal changed')
    for name,digest in files.items():
        if Path(name).name!=name or not(cp/name).resolve().is_relative_to(cp)or file_sha(cp/name)!=digest:
            raise ValueError('Original checkpoint payload changed')
    common=torch.load(cp/'optimizer.pt',map_location='cpu',weights_only=False)
    payloads=[torch.load(cp/f'rank{r}.pt',map_location='cpu',weights_only=False)for r in range(source_world)]
    original_rows=[]
    for own in payloads:
        if own['identity_sha256']!=old['sha256']:raise ValueError('Foreign native rank')
        original_rows.append(migration.boundary(old).validate_boundary(own['state'],own['rollout'],old,common['optimizer'],common['scheduler']))
    migration.boundary(old).validate_ranks(original_rows)
    if original_rows!=json.loads((cp/'transaction.json').read_text())['ranks']:
        raise ValueError('Original rank state differs from sealed transaction')
    identity=copy.deepcopy(old);identity.pop('sha256')
    identity.update(world=target_world,source_files=migration.inventory(),performance_execution=dict(
        schema=migration.SCHEMA,profile='fast',origin_layout='fast',origin_updates=seal['updates'],
        source_checkpoint=str(cp),source_identity_sha256=old['sha256'],source_files_sha256=seal['files_sha256'],
        common_optimizer_sha256=files['optimizer.pt'],same_algorithm=True,
        reshard=dict(source_world=source_world,target_world=target_world,global_scenes=64,
                     group=16,inner_updates=2,virtual_sampler_lanes=8,pending_material_preserved=True,
                     floating_DDP_reduction_order_may_differ=True)))
    identity['sha256']=fingerprint(identity)
    staged=root.with_name(root.name+'.incomplete.'+uuid.uuid4().hex)
    dest=staged/'checkpoints'/cp.name;dest.mkdir(parents=True)
    summaries=[]
    for rank,own in enumerate(repartition(payloads,source_world,target_world)):
        own['identity_sha256']=identity['sha256']
        summaries.append(migration.boundary(identity).validate_boundary(own['state'],own['rollout'],identity,common['optimizer'],common['scheduler']))
        checkpoint.atomic_torch(dest/f'rank{rank}.pt',own)
    migration.boundary(identity).validate_ranks(summaries)
    shutil.copy2(cp/'optimizer.pt',dest/'optimizer.pt')
    if file_sha(dest/'optimizer.pt')!=files['optimizer.pt']:raise ValueError('Adam/policy payload changed')
    atomic_json(dest/'identity.json',identity)
    atomic_json(dest/'transaction.json',dict(schema=checkpoint.CHECKPOINT_CONTRACT,attempt=uuid.uuid4().hex,ranks=summaries))
    copied={p.name:file_sha(p)for p in sorted(dest.iterdir())if p.is_file()}
    atomic_json(dest/'files.json',copied)
    atomic_json(dest/'COMPLETE',dict(seal,world=target_world,files_sha256=file_sha(dest/'files.json')))
    atomic_json(staged/'identity.json',identity)
    receipt=dict(schema=migration.SCHEMA,source_identity=old,source_seal=seal,target_identity_sha256=identity['sha256'],
        native_state_preserved=True,optimizer_bytes_preserved=True,original_reference_retained=True,
        rank_global_slots={str(r):slots(r,target_world)for r in range(target_world)},
        RNG_protocol='all original virtual8 lanes preserved; per-rank global RNG copied from owner of its first source slot; no global RNG consumed by collection')
    receipt['sha256']=fingerprint(receipt);atomic_json(staged/'PERFORMANCE_EXECUTION.json',receipt)
    final=root/'checkpoints'/cp.name
    atomic_json(staged/'latest.json',dict(checkpoint=str(final),updates=seal['updates'],files_sha256=file_sha(dest/'files.json')))
    migration.validate(staged,identity);os.rename(staged,root)
    return dict(root=str(root),checkpoint=str(final),world=target_world,origin_updates=seal['updates'])
