"""Typed READY world16->16 execution upgrade; no actor, optimizer or RNG reset.

Only a new root is published. The common native payload is copied byte-for-byte,
rank RNG/virtual8/history/scene clocks are preserved, and no old journal is forged.
"""
import copy
import json
import os
from pathlib import Path
import random
import shutil
import uuid
import numpy as np
import torch
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.qwen_world_compare.contracts import file_sha,fingerprint
from starVLA.rl.qwen_world_compare.recovery_state import tensor_digest
from starVLA.rl.qwen_world_abcd_research.checkpoint import atomic_torch,CHECKPOINT_CONTRACT

SCHEMA='d46_world16_execution_upgrade_v1'
FLAG_NAMES=('fast_prefix','fuse_fm_mc','async_scoring','rotary_cache','no_grad_prefix_cache')


def flags(value):
    if not isinstance(value,dict)or set(value)!=set(FLAG_NAMES)or any(type(v)is not bool for v in value.values()):
        raise ValueError('Execution upgrade requires exactly five boolean execution flags')
    return dict(value)


def sealed(value,label):
    if not isinstance(value,dict):raise ValueError('Missing '+label)
    body=dict(value);sha=body.pop('sha256',None)
    if sha!=fingerprint(body):raise ValueError('Changed '+label+' seal')
    return value


def journal_origin(identity):
    upgrade=identity.get('execution_upgrade')
    if upgrade is None:return identity['native_world_transition']['origin_updates']
    validate_identity(identity)
    return upgrade['origin_updates']


def validate_identity(identity):
    """Pure metadata guard, also safe at every native READY/PENDING boundary."""
    value=identity.get('execution_upgrade')
    if value is None:return None
    sealed(identity,'world16 identity')
    if (value.get('schema')!=SCHEMA or identity.get('world')!=16 or identity['recipe'].get('arm')not in('D0','D3_FPO')
        or identity.get('fresh_Adam')is not False):raise ValueError('Not a typed D0/FPO world16 execution upgrade')
    flags(value.get('flags'))
    u=value.get('origin_updates');transition=identity['native_world_transition']
    if (type(u)is not int or u<transition['origin_updates']or u%identity['recipe']['inner_updates']
        or type(value.get('source_qualification'))is not bool or type(value.get('target_qualification'))is not bool
        or value['target_qualification']is not identity.get('qualification')
        or transition.get('target_qualification')is not identity.get('qualification')
        or(value['source_qualification']and not value['target_qualification'])
        or value.get('new_inventory_sha256')!=fingerprint(identity['source_files'])):
        raise ValueError('Execution upgrade origin/mode/source inventory changed')
    for name in ('source_identity_sha256','source_checkpoint_files_sha256','source_checkpoint_identity_file_sha256',
                 'source_world_transition_sha256','common_optimizer_sha256','old_inventory_sha256','new_inventory_sha256'):
        if not isinstance(value.get(name),str)or len(value[name])!=64:raise ValueError('Missing execution upgrade digest '+name)
    if not Path(value.get('source_world16_checkpoint','')).is_absolute():raise ValueError('Missing absolute source16 checkpoint')
    return value


def validate_receipt(root,identity,transition_receipt):
    """Bind the immutable root receipt to the actual sealed source16 ancestry.

    Native restore still verifies the complete *target* payload and Adam/RNG.
    This metadata-only ancestry check never reloads the old model/optimizer.
    """
    meta=validate_identity(identity)
    if meta is None:
        if transition_receipt.get('execution_upgrade_sha256')is not None:raise ValueError('Unexpected execution upgrade receipt')
        return None
    root=Path(root)
    value=sealed(json.loads((root/'EXECUTION_UPGRADE.json').read_text()),'execution upgrade receipt')
    if (value.get('schema')!=SCHEMA or value.get('execution_upgrade')!=meta
        or value.get('target_identity_sha256')!=identity['sha256']
        or transition_receipt.get('execution_upgrade_sha256')!=value['sha256']):
        raise ValueError('Execution upgrade receipt/profile differs')
    cp=Path(meta['source_world16_checkpoint'])
    old=sealed(json.loads((cp/'identity.json').read_text()),'source16 identity')
    seal=json.loads((cp/'COMPLETE').read_text());inventory=json.loads((cp/'files.json').read_text())
    if (old['sha256']!=meta['source_identity_sha256']or file_sha(cp/'identity.json')!=meta['source_checkpoint_identity_file_sha256']
        or file_sha(cp/'files.json')!=meta['source_checkpoint_files_sha256']or seal.get('files_sha256')!=meta['source_checkpoint_files_sha256']
        or inventory.get('identity.json')!=meta['source_checkpoint_identity_file_sha256']
        or inventory.get('optimizer.pt')!=meta['common_optimizer_sha256']or seal.get('world')!=16 or old.get('world')!=16
        or seal.get('contract')!=CHECKPOINT_CONTRACT or seal.get('schema')!=2
        or seal.get('phase')!='READY'or seal.get('next_inner')!=0 or seal.get('updates')!=meta['origin_updates']
        or old.get('qualification')is not meta['source_qualification']or fingerprint(old['source_files'])!=meta['old_inventory_sha256']):
        raise ValueError('Execution upgrade source16 checkpoint ancestry changed')
    source_transition=sealed(json.loads((cp.parent.parent/'WORLD_TRANSITION.json').read_text()),'source16 transition')
    if source_transition['sha256']!=meta['source_world_transition_sha256']or source_transition.get('target_identity_sha256')!=old['sha256']:
        raise ValueError('Execution upgrade original WORLD_TRANSITION changed')
    expected_transition=copy.deepcopy(source_transition);expected_transition.pop('sha256')
    expected_transition.update(target_identity_sha256=identity['sha256'],source_inventory=identity['source_files'],
        qualification=identity['qualification'],execution_upgrade_sha256=value['sha256'])
    if fingerprint(expected_transition)!=transition_receipt.get('sha256'):
        raise ValueError('Execution upgrade changed original world8 transition ancestry')
    # Only execution source/mode/fresh marker/origin profile may differ.
    expected=copy.deepcopy(old);expected.pop('sha256');expected.update(source_files=identity['source_files'],qualification=identity['qualification'],fresh_Adam=False)
    expected['native_world_transition']['target_qualification']=identity['qualification']
    expected['execution_upgrade']=meta
    if fingerprint(expected)!=identity['sha256']:
        raise ValueError('Execution upgrade changed parent/scope/recipe or original world8 ancestry')
    return value


def _rng(rng):
    """Validate saved CPU/Python/NumPy/CUDA bytes without touching global RNG."""
    from .native import tensor_sha
    if set(rng)!=set(('python','numpy','torch','cuda','rollout_generator')):raise ValueError('Incomplete native RNG bundle')
    random.Random().setstate(rng['python']);np.random.RandomState().set_state(rng['numpy'])
    for key in ('torch','rollout_generator'):
        tensor_sha(rng[key]);torch.Generator().set_state(rng[key])
    if rng['cuda']is not None:tensor_sha(rng['cuda'])
    return tensor_digest(rng)


def prepare(checkpoint,new_root,old_frozen_source,*,execution_flags,qualification=False):
    """Verify full source/native bytes and publish one independent READY root."""
    from .native import World16Boundary,source_inventory,validate_supported_recipe
    from .runtime import receipt
    cp=Path(checkpoint).resolve(strict=True);root=Path(new_root).resolve();old_source=Path(old_frozen_source).resolve(strict=True)
    if type(qualification)is not bool:raise ValueError('qualification must be boolean')
    profile=flags(execution_flags)
    if root.exists():raise FileExistsError('New execution root must not exist')
    oldroot=cp.parent.parent
    if root.is_relative_to(oldroot)or oldroot.is_relative_to(root):raise ValueError('New execution root must be independent of old run')
    if '.incomplete'in cp.name:raise ValueError('Unpublished source16 checkpoint')
    old=sealed(json.loads((cp/'identity.json').read_text()),'source16 identity')
    seal=json.loads((cp/'COMPLETE').read_text());inventory=json.loads((cp/'files.json').read_text())
    if (old.get('world')!=16 or seal.get('world')!=16 or seal.get('schema')!=2 or seal.get('contract')!=CHECKPOINT_CONTRACT
        or seal.get('phase')!='READY'or seal.get('next_inner')!=0 or seal.get('files_sha256')!=file_sha(cp/'files.json')):
        raise ValueError('Only complete world16 READY checkpoint may be upgraded')
    if old.get('qualification')and not qualification:raise ValueError('Qualification weights may never enter formal execution')
    if old['recipe'].get('arm')not in('D0','D3_FPO'):raise ValueError('Execution upgrade is only D0/FPO')
    validate_supported_recipe(old['recipe'])
    from starVLA.rl.d44_native_rl.config import validate
    validate(old['recipe'])
    tiny=(qualification and old.get('qualification_fixture')=='d46_tiny_native_world_transition_CPU_v1'
          and old.get('parent_model_sha256','').startswith('tiny_native_qualification:')and old['scope']['trainable_numel']<1000000)
    from starVLA.rl.d44_native_rl.parent import D46_MODEL_SHA
    if not tiny and old.get('parent_model_sha256')!=D46_MODEL_SHA:raise ValueError('Execution upgrade requires own exact IL46 policy parent')
    if json.loads((oldroot/'identity.json').read_text())!=old:raise ValueError('Source16 root/checkpoint identity differs')
    required={'identity.json','transaction.json','optimizer.pt'}|{f'rank{r}.pt'for r in range(16)}
    if not required<=inventory.keys():raise ValueError('Incomplete source16 native inventory')
    for name,value in inventory.items():
        if Path(name).name!=name or not name or not(cp/name).resolve().is_relative_to(cp)or file_sha(cp/name)!=value:raise ValueError('Changed/escaping source16 payload '+name)
    for name,value in old['source_files'].items():
        path=(old_source/name).resolve()
        if not path.is_relative_to(old_source)or file_sha(path)!=value:raise ValueError('Old frozen source differs: '+name)
    # Old receipt is validated against the actually hash-verified old inventory,
    # never against whichever new source happens to run this offline converter.
    previous=receipt(oldroot,old,expected_sources=old['source_files'])
    current=source_inventory()
    if not tiny:
        protected=lambda n:n.startswith('starVLA/model/')or n in('starVLA/rl/d44_native_rl/parent.py','starVLA/rl/d44_native_rl/conditions.py')
        if any(current.get(n)!=v for n,v in old['source_files'].items()if protected(n)):
            raise ValueError('Execution upgrade may not change model/parent/conditions source')
        from starVLA.rl.d44_native_rl.deployment import inspect
        inspect(cp)
    common=torch.load(cp/'optimizer.pt',map_location='cpu',weights_only=False)
    if set(common['adapters'])!=set(old['scope']['trainable_names']):raise ValueError('Incomplete common trainable closure')
    if any(p.requires_grad or p.dtype!=torch.float32 or not torch.isfinite(p).all()for p in common['adapters'].values()):raise ValueError('Invalid common trainables')
    tx=json.loads((cp/'transaction.json').read_text())
    if tx.get('schema')!=CHECKPOINT_CONTRACT or len(tx.get('ranks',[]))!=16:raise ValueError('Incomplete source16 transaction')
    payloads=[];summaries=[];rng_sha=[]
    for rank in range(16):
        own=torch.load(cp/f'rank{rank}.pt',map_location='cpu',weights_only=False)
        if own.get('identity_sha256')!=old['sha256']or own.get('rollout')is not None or own['state'].get('phase')!='READY':raise ValueError('Not exact source16 READY material')
        row=World16Boundary.validate_boundary(own['state'],None,old,common['optimizer'],common['scheduler'])
        if row!=tx['ranks'][rank]or row['updates']!=seal['updates']:raise ValueError('Source16 transaction/optimizer clock differs')
        state=own['state'];n=state['collections']*64
        if (state['scene_cursor']!=n or state['raw_scene_exposures']!=n or state['padding_scene_exposures']!=n-state['valid_scene_exposures']
            or state['new_candidates']!=n*16 or state['valid_PG_candidates']!=state['valid_scene_exposures']*16
            or state['extra_B1_queries']!=0 or state['adaptive_G32_queries']!=0):raise ValueError('Source16 scene/query clock differs')
        rng_sha.append(_rng(own['rng']));summaries.append(row);payloads.append(own)
    World16Boundary.validate_ranks(summaries)
    meta=dict(schema=SCHEMA,source_world16_checkpoint=str(cp),source_identity_sha256=old['sha256'],
        source_checkpoint_files_sha256=file_sha(cp/'files.json'),source_checkpoint_identity_file_sha256=inventory['identity.json'],
        source_world_transition_sha256=previous['sha256'],common_optimizer_sha256=inventory['optimizer.pt'],
        origin_updates=seal['updates'],flags=profile,source_qualification=old['qualification'],target_qualification=qualification,
        old_inventory_sha256=fingerprint(old['source_files']),new_inventory_sha256=fingerprint(current))
    identity=copy.deepcopy(old);identity.pop('sha256');identity.update(source_files=current,qualification=qualification,fresh_Adam=False,execution_upgrade=meta)
    identity['native_world_transition']['target_qualification']=qualification;identity['sha256']=fingerprint(identity)
    upgrade=dict(schema=SCHEMA,execution_upgrade=meta,target_identity_sha256=identity['sha256'],
        source_frozen_root=str(old_source),source_rank_files_sha256={str(r):inventory[f'rank{r}.pt']for r in range(16)},
        source_rank_RNG_sha256=rng_sha,source_virtual8_rng_sha256=summaries[0]['virtual8_rng_sha256'],
        common_optimizer_byte_copy=True,only_rank_changes=['identity_sha256','state.journal_origin_updates'],
        copied_old_journal=False,qualification_weights_never_return_to_formal=True)
    upgrade['sha256']=fingerprint(upgrade)
    transition=copy.deepcopy(previous);transition.pop('sha256');transition.update(target_identity_sha256=identity['sha256'],source_inventory=current,
        qualification=qualification,execution_upgrade_sha256=upgrade['sha256']);transition['sha256']=fingerprint(transition)
    target=root/'checkpoints'/f"update{seal['updates']:06}_inner0";attempt=root.with_name(root.name+'.incomplete.'+uuid.uuid4().hex);dest=attempt/'checkpoints'/target.name
    dest.mkdir(parents=True);rows=[]
    for rank,own in enumerate(payloads):
        own['identity_sha256']=identity['sha256'];own['state']['journal_origin_updates']=seal['updates']
        rows.append(World16Boundary.validate_boundary(own['state'],None,identity,common['optimizer'],common['scheduler']))
        atomic_torch(dest/f'rank{rank}.pt',own)
    World16Boundary.validate_ranks(rows)
    shutil.copyfile(cp/'optimizer.pt',dest/'optimizer.pt')
    if file_sha(dest/'optimizer.pt')!=inventory['optimizer.pt']:raise ValueError('Common optimizer byte copy differs')
    atomic_json(dest/'identity.json',identity);atomic_json(attempt/'identity.json',identity)
    for parent in(dest,attempt):
        atomic_json(parent/'EXECUTION_UPGRADE.json',upgrade)
        atomic_json(parent/('world_transition.json'if parent==dest else'WORLD_TRANSITION.json'),transition)
    atomic_json(dest/'transaction.json',dict(schema=CHECKPOINT_CONTRACT,attempt=uuid.uuid4().hex,ranks=rows,execution_identity=None,
        migration_receipt_sha256=None,world_transition_sha256=transition['sha256'],execution_upgrade_sha256=upgrade['sha256']))
    files={p.name:file_sha(p)for p in sorted(dest.iterdir())if p.is_file()};atomic_json(dest/'files.json',files)
    atomic_json(dest/'COMPLETE',dict(schema=2,contract=CHECKPOINT_CONTRACT,phase='READY',world=16,updates=seal['updates'],next_inner=0,files_sha256=file_sha(dest/'files.json')))
    atomic_json(attempt/'latest.json',dict(checkpoint=str(target),updates=seal['updates'],files_sha256=file_sha(dest/'files.json')))
    # Receipt/parent/profile validation before publication; native payloads were
    # validated with actual16 Boundary + source transaction above.
    receipt(attempt,identity)
    os.rename(attempt,root)
    return dict(root=str(root),checkpoint=str(target),identity_sha256=identity['sha256'],execution_upgrade_sha256=upgrade['sha256'],
        world_transition_sha256=transition['sha256'],updates=seal['updates'],qualification=qualification,common_optimizer_sha256=inventory['optimizer.pt'])
