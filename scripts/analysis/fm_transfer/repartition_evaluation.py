"""Republish sealed predictions for more shards, without running a model/scorer.

The old bank is immutable. Only execution identity changes: shard count and its
manifest digest. All non-identity payload values are roundtrip-checked exactly.
The unchanged qualified evaluator subsequently resumes the new bank.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import struct
import time

import numpy as np
import torch

from starVLA.rl.simwam_recipe.contracts import POPULATION, file_sha, fingerprint
from starVLA.rl.simwam_recipe.checkpoint import atomic_torch
from starVLA.rl.flow_grpo.transactions import atomic_json
from scripts.analysis.fm_transfer.rescore_saved_navtest import authenticated, validate_input_manifest


def value_digest(value):
    """Exact values/dtypes/shapes, including BF16 tensors and IEEE float bits."""
    digest = hashlib.sha256()
    def visit(x):
        if isinstance(x, torch.Tensor):
            y = x.detach().cpu().contiguous()
            digest.update(str(('tensor', str(y.dtype), tuple(y.shape))).encode())
            digest.update(y.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x, np.ndarray):
            if x.dtype.hasobject: raise TypeError('Object array is not a numerical record')
            digest.update(str(('numpy', x.dtype.str, x.shape)).encode())
            digest.update(np.ascontiguousarray(x).tobytes())
        elif isinstance(x, dict):
            digest.update(b'dict')
            for key in sorted(x): visit(key); visit(x[key])
        elif isinstance(x, (list, tuple)):
            digest.update(str((type(x).__name__, len(x))).encode())
            for item in x: visit(item)
        elif isinstance(x, float): digest.update(b'float' + struct.pack('!d', x))
        elif x is None or isinstance(x, (str, int, bool)):
            digest.update(json.dumps([type(x).__name__, x], ensure_ascii=False).encode())
        else: raise TypeError('Unqualified record type: ' + str(type(x)))
    visit(value)
    return digest.hexdigest()


def repartition_identity(old, shards):
    if not 1 <= shards <= 64 or shards == old['shards']:
        raise ValueError('A distinct supported shard count is required')
    value = copy.deepcopy(old); value.pop('sha256'); value['shards'] = shards
    value['sha256'] = fingerprint(value)
    if {k:v for k,v in old.items() if k not in ('shards', 'sha256')} != {
            k:v for k,v in value.items() if k not in ('shards', 'sha256')}:
        raise ValueError('Scientific identity changed')
    return value


def republish(path, target, old_identity, new_identity, *, replace=None):
    seal = json.loads(path.with_suffix('.json').read_text())
    if seal['identity_sha256'] != old_identity or seal['payload_sha256'] != file_sha(path):
        raise ValueError('Invalid source seal; never recompute silently')
    original = torch.load(path, map_location='cpu', weights_only=False)
    if original['identity_sha256'] != old_identity or original['token'] != seal['token'] or original['log'] != seal['log']:
        raise ValueError('Wrong source payload')
    for mode, record in original['records'].items():
        if record['status'] != 'OK' or record['metrics'] != seal['metrics'][mode]:
            raise ValueError('Invalid original score')
    changed = dict(replace or {}, identity_sha256=new_identity)
    numerical = {k:v for k,v in original.items() if k not in changed}
    before = value_digest(numerical)
    payload = dict(original, **changed)
    if target.exists() or target.with_suffix('.json').exists():
        raise FileExistsError('Never overwrite a prediction')
    atomic_torch(target, payload)
    reread = torch.load(target, map_location='cpu', weights_only=False)
    if before != value_digest({k:v for k,v in reread.items() if k not in changed}):
        raise ValueError('Numerical payload changed while repartitioning')
    result = dict(seal, identity_sha256=new_identity, payload_sha256=file_sha(target))
    if replace and 'original_prediction_sha256' in replace:
        result['original_prediction_sha256'] = replace['original_prediction_sha256']
    atomic_json(target.with_suffix('.json'), result)
    return dict(token=seal['token'], old_payload_sha256=seal['payload_sha256'],
                new_payload_sha256=result['payload_sha256'], numerical_values_sha256=before)


def migrate(evaluation, output, shards, drain_receipt, score_source=None, score_output=None):
    started = time.monotonic()
    old, new = Path(evaluation).resolve(), Path(output).resolve()
    if new.exists(): raise FileExistsError('Use a new execution bank')
    receipt = json.loads(Path(drain_receipt).read_text())
    if receipt['status'] != 'ALL_ORIGINAL_GPU_WORKERS_EXITED_AT_SCENE_BOUNDARY':
        raise ValueError('Original writers have not drained')
    identity = authenticated(old/'manifest.json')
    if receipt['banks'][str(old)]['manifest_sha256'] != file_sha(old/'manifest.json'):
        raise ValueError('Drain belongs to another bank')
    pop = json.loads(Path(POPULATION).read_text()); validate_input_manifest(identity, pop)
    ranks = [json.loads((old/f'rank{i}.json').read_text()) for i in range(identity['shards'])]
    if any(r['status'] not in ('STOPPED_AT_SCENE_BOUNDARY', 'ASSIGNED_COMPLETE') for r in ranks):
        raise ValueError('Unclosed original worker')
    target_identity = repartition_identity(identity, shards)
    old_seals = sorted((old/'scenes').glob('*/*.json'))
    tokens = [p.stem for p in old_seals]
    if len(set(tokens)) != len(tokens) or not set(tokens) <= set(pop['navtest_tokens']):
        raise ValueError('Foreign or repeated scene')
    invocations = list((old/'invocations').glob('*/rank*.json'))
    counters = [json.loads(p.read_text()) for p in invocations]
    if (not counters or any('end_unix' not in r or r['status'] not in
            ('STOPPED_AT_SCENE_BOUNDARY', 'ASSIGNED_COMPLETE') for r in counters)
            or sum(r['new'] for r in counters) != len(tokens)):
        raise ValueError('Physical invocation counters do not match committed scenes')
    new.mkdir(parents=True)
    atomic_json(new/'manifest.json', target_identity)
    migrated = []
    for seal_path in old_seals:
        path = seal_path.with_suffix('.pt')
        migrated.append(republish(path, new/path.relative_to(old), identity['sha256'], target_identity['sha256']))
    # Exact copies explicitly inherit physical costs; rank identities remain the
    # original identities. New evaluator invocations get distinct UUIDs.
    for path in invocations:
        target = new/path.relative_to(old); target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    score_receipt = None
    if (score_source is None) != (score_output is None): raise ValueError('Both score paths are required')
    if score_source:
        score_receipt = migrate_scores(Path(score_source), Path(score_output), old, new, identity, target_identity)
    record = dict(schema='sealed_Navtest_execution_repartition_v1', status='COMPLETE_METADATA_ONLY_REPARTITION',
        old_evaluation=str(old), new_evaluation=str(new), old_manifest_sha256=file_sha(old/'manifest.json'),
        new_manifest_sha256=file_sha(new/'manifest.json'), old_shards=identity['shards'], new_shards=shards,
        reused_scenes=len(migrated), migrated_payloads=migrated,
        inherited_invocations=[dict(path=str(p), sha256=file_sha(p)) for p in invocations],
        scores=score_receipt, additional_predictions=0, additional_scores=0,
        worker_implementation_changed=False, scientific_fields_changed=False,
        migration_source_sha256=file_sha(__file__), wall_seconds=time.monotonic()-started)
    atomic_json(new/'REPARTITION_COMPLETE.json', record)
    return record


def migrate_scores(old, new, old_eval, new_eval, old_identity, new_identity):
    if new.exists(): raise FileExistsError('Preserve previous scorer attempts')
    source_id = authenticated(old/'manifest.json')
    if (source_id['original_manifest_identity'] != old_identity['sha256'] or
            source_id['original_evaluation'] != str(old_eval)):
        raise ValueError('Scores belong to another prediction bank')
    attempts = [json.loads(p.read_text()) for p in (old/'attempts').glob('*.json')]
    if not attempts or any('ended_utc' not in a or a['status'] == 'RUNNING' for a in attempts):
        raise ValueError('Score writers have not exited')
    target_id = dict(source_id); target_id.pop('sha256')
    target_id.update(original_evaluation=str(new_eval), original_manifest_identity=new_identity['sha256'],
                     original_manifest_sha256=file_sha(new_eval/'manifest.json'))
    target_id['sha256'] = fingerprint(target_id)
    new.mkdir(parents=True); atomic_json(new/'manifest.json', target_id)
    rows = []
    for seal in sorted((old/'scenes').glob('*/*.json')):
        token = seal.stem; prediction = new_eval/'scenes'/token[:2]/(token+'.pt')
        prior = json.loads(seal.read_text())
        if prior['original_prediction_sha256'] != file_sha(old_eval/'scenes'/token[:2]/(token+'.pt')):
            raise ValueError('Original score/prediction relation changed')
        rows.append(republish(seal.with_suffix('.pt'), new/'scenes'/token[:2]/(token+'.pt'),
            source_id['sha256'], target_id['sha256'], replace=dict(
                original_prediction_sha256=file_sha(prediction), original_manifest_identity=new_identity['sha256'])))
    record = dict(parent_score_bank=str(old), parent_manifest_sha256=file_sha(old/'manifest.json'),
        reused_scenes=len(rows), rows=rows, inherited_attempts=attempts,
        inherited_physical_scoring_calls=sum(a['physical_scoring_calls'] for a in attempts),
        inherited_invocation_wall_seconds=sum(a['wall_seconds'] for a in attempts),
        additional_scores=0, cost_note='Inherited costs remain separate from new scorer invocations; include both in final cost report.')
    atomic_json(new/'REPARTITION_COMPLETE.json', record)
    return {k:v for k,v in record.items() if k not in ('rows', 'inherited_attempts')}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--evaluation', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--shards', type=int, required=True); parser.add_argument('--drain-receipt', required=True)
    parser.add_argument('--score-source'); parser.add_argument('--score-output')
    args = parser.parse_args(); result = migrate(**vars(args))
    print(json.dumps({k:v for k,v in result.items() if k not in ('migrated_payloads','inherited_invocations')}))


if __name__ == '__main__': main()
