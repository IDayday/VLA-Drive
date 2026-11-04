"""Typed execution-only continuation. Never adopt benchmark policy weights."""
import copy
import json
import os
from pathlib import Path
import shutil
import uuid

from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.qwen_world_compare.contracts import file_sha, fingerprint
from starVLA.rl.d44_native_rl.recovery import checkpoint

SCHEMA = 'd46_fpo_performance_execution_v1'


def inventory():
    from tools.d44_native_rl.train import code_identity
    result = code_identity()
    for directory in ('starVLA/rl/d46_world16', 'starVLA/rl/d46_fpo_verified',
                      'starVLA/rl/d46_fpo_engineering', 'tools/d46_fpo_engineering'):
        result.update({str(p): file_sha(p) for p in sorted(Path(directory).glob('*.py'))})
    return result


def boundary(identity):
    if identity.get('performance_execution',{}).get('reshard'):
        from .virtual_lanes import DistributedBoundary
        return DistributedBoundary()
    if identity['world'] == 16:
        from starVLA.rl.d46_world16.native import World16Boundary
        return World16Boundary()
    from starVLA.rl.d46_fpo_verified.runtime import VerifiedBoundary
    return VerifiedBoundary()


def sealed(value):
    body = copy.deepcopy(value)
    digest = body.pop('sha256', None)
    if fingerprint(body) != digest:
        raise ValueError('Changed execution identity/receipt')
    return value


def validate(root, identity, *, actual_inventory=None):
    sealed(identity)
    receipt = sealed(json.loads((Path(root) / 'PERFORMANCE_EXECUTION.json').read_text()))
    meta = identity.get('performance_execution', {})
    old = sealed(receipt['source_identity'])
    expected = copy.deepcopy(old)
    expected.pop('sha256')
    expected.update(source_files=identity['source_files'], performance_execution=meta,
                    training_run_id=old.get('training_run_id', old['sha256']))
    redistribution=meta.get('reshard')
    if redistribution:
        if (redistribution.get('source_world')!=old['world'] or redistribution.get('target_world')!=identity['world']
                or old['world']not in(16,32)or identity['world']not in(16,32)
                or redistribution.get('global_scenes')!=64 or 'verified_improvement'in old['recipe']):
            raise ValueError('Unsupported native scene redistribution')
        expected['world']=identity['world']
    if (meta.get('schema') != SCHEMA or meta.get('profile') not in ('exact', 'fast')
            or meta.get('source_identity_sha256') != old['sha256']
            or meta.get('origin_updates') != receipt['source_seal']['updates']
            or receipt['target_identity_sha256'] != identity['sha256']
            or fingerprint(expected) != identity['sha256']
            or identity['source_files'] != (inventory() if actual_inventory is None else actual_inventory)):
        raise ValueError('Execution changed scientific state, source inventory or profile')
    if old.get('qualification') is not False or identity.get('qualification') is not False:
        raise ValueError('Benchmark/qualification weights cannot initialize formal execution')
    if not receipt.get('native_state_preserved') or not receipt.get('optimizer_bytes_preserved'):
        raise ValueError('Missing native continuity evidence')
    return receipt


def prepare(source_checkpoint, source_code, output, *, profile, evidence):
    import torch
    from starVLA.rl.d44_native_rl.parent import D46_MODEL_SHA
    from starVLA.rl.qwen_world_compare.recovery_state import tensor_digest
    cp = Path(source_checkpoint).resolve(strict=True)
    old_code = Path(source_code).resolve(strict=True)
    root = Path(output).resolve()
    if root.exists() or root.is_relative_to(cp.parent.parent):
        raise ValueError('Execution root must be new and independent')
    old = sealed(json.loads((cp / 'identity.json').read_text()))
    seal = json.loads((cp / 'COMPLETE').read_text())
    files = json.loads((cp / 'files.json').read_text())
    if (old.get('qualification') is not False or old['parent_model_sha256'] != D46_MODEL_SHA
            or old['recipe']['arm'] != 'D3_FPO' or old['world'] not in (8, 16)
            or seal['world'] != old['world'] or seal['contract'] != checkpoint.CHECKPOINT_CONTRACT
            or seal['files_sha256'] != file_sha(cp / 'files.json') or profile not in ('exact', 'fast')):
        raise ValueError('Only own complete formal D46 FPO state may continue')
    required = {'optimizer.pt', 'identity.json', 'transaction.json'} | {f'rank{r}.pt' for r in range(old['world'])}
    if not required <= files.keys():
        raise ValueError('Incomplete checkpoint inventory')
    for name, digest in files.items():
        if Path(name).name != name or not (cp/name).resolve().is_relative_to(cp) or file_sha(cp/name) != digest:
            raise ValueError('Changed native payload ' + name)
    for name, digest in old['source_files'].items():
        if not (old_code/name).resolve().is_relative_to(old_code) or file_sha(old_code/name) != digest:
            raise ValueError('Changed original scientific implementation ' + name)
    evidence = Path(evidence).resolve(strict=True)
    proof = json.loads(evidence.read_text())
    rows = {r['mode']: r for r in proof.get('rows', [])}
    if (proof.get('status') != 'COMPLETE' or proof.get('world') != old['world']
            or Path(proof['source_checkpoint']).resolve() != cp or profile not in rows
            or not rows[profile]['native_roundtrip_exact']):
        raise ValueError('Missing same-checkpoint full-cycle qualification')
    if profile == 'exact' and not rows[profile]['final_parameters_bitexact_to_baseline']:
        raise ValueError('Exact profile did not preserve parameter values')
    meta = dict(schema=SCHEMA, profile=profile, origin_updates=seal['updates'],
                source_checkpoint=str(cp), source_identity_sha256=old['sha256'],
                source_files_sha256=seal['files_sha256'], common_optimizer_sha256=files['optimizer.pt'],
                evidence=str(evidence), evidence_sha256=file_sha(evidence),
                finish_legacy_pending_with_legacy_layout=True, same_algorithm=True)
    identity = copy.deepcopy(old)
    identity.pop('sha256')
    identity.update(source_files=inventory(), performance_execution=meta,
                    training_run_id=old.get('training_run_id', old['sha256']))
    identity['sha256'] = fingerprint(identity)
    staging = root.with_name(root.name + '.incomplete.' + uuid.uuid4().hex)
    target = staging / 'checkpoints' / cp.name
    target.mkdir(parents=True)
    common = torch.load(cp/'optimizer.pt', map_location='cpu', weights_only=False)
    original_summaries = json.loads((cp/'transaction.json').read_text())['ranks']
    if len(original_summaries) != old['world']:
        raise ValueError('Incomplete original distributed transaction')
    summaries = []
    runtime = boundary(identity)
    for rank in range(old['world']):
        own = torch.load(cp/f'rank{rank}.pt', map_location='cpu', weights_only=False)
        before = tensor_digest({k:v for k,v in own.items() if k != 'identity_sha256'})
        original = boundary(old).validate_boundary(own['state'], own['rollout'], old, common['optimizer'], common['scheduler'])
        if own['identity_sha256'] != old['sha256'] or original != original_summaries[rank]:
            raise ValueError('Native rank state differs from original sealed transaction')
        own['identity_sha256'] = identity['sha256']
        summaries.append(runtime.validate_boundary(own['state'], own['rollout'], identity, common['optimizer'], common['scheduler']))
        checkpoint.atomic_torch(target/f'rank{rank}.pt', own)
        restored = torch.load(target/f'rank{rank}.pt', map_location='cpu', weights_only=False)
        if tensor_digest({k:v for k,v in restored.items() if k != 'identity_sha256'}) != before:
            raise ValueError('Native rank state changed during copy')
    runtime.validate_ranks(summaries)
    shutil.copy2(cp/'optimizer.pt', target/'optimizer.pt')
    assert file_sha(target/'optimizer.pt') == files['optimizer.pt']
    atomic_json(target/'identity.json', identity)
    atomic_json(target/'transaction.json', dict(schema=checkpoint.CHECKPOINT_CONTRACT, attempt=uuid.uuid4().hex, ranks=summaries))
    copied = {p.name:file_sha(p) for p in sorted(target.iterdir()) if p.is_file()}
    atomic_json(target/'files.json', copied)
    atomic_json(target/'COMPLETE', dict(seal, files_sha256=file_sha(target/'files.json')))
    atomic_json(staging/'identity.json', identity)
    receipt = dict(schema=SCHEMA, source_identity=old, source_seal=seal, target_identity_sha256=identity['sha256'],
                   native_state_preserved=True, optimizer_bytes_preserved=True, original_reference_retained=True)
    receipt['sha256'] = fingerprint(receipt)
    atomic_json(staging/'PERFORMANCE_EXECUTION.json', receipt)
    final = root/'checkpoints'/cp.name
    atomic_json(staging/'latest.json', dict(checkpoint=str(final), updates=seal['updates'], files_sha256=file_sha(target/'files.json')))
    validate(staging, identity)
    os.rename(staging, root)
    return dict(root=str(root), checkpoint=str(final), identity=identity['sha256'], profile=profile,
                origin_updates=seal['updates'], scientific_run=identity['training_run_id'])
