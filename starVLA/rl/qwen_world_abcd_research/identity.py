"""Canonical JSON boundary for exact, persistable run identities.

The parent source migration contains tuple-valued diffs in memory. JSON stores
them as arrays. Normalize before comparison, preserving every value/hash check.
This never touches model tensors, optimizer state, RNG or scientific settings.
"""
import json
from pathlib import Path
from starVLA.rl.qwen_world_compare.contracts import fingerprint, file_sha


def canonical_json_identity(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def validate_execution_receipt(receipt, identity, execution, checkpoint):
    """Only a sealed same-science receipt can admit a new execution lineage."""
    body=dict(receipt);sha=body.pop('sha256')
    if fingerprint(body)!=sha or receipt['old_identity']!=identity:
        raise ValueError('Execution migration receipt/old identity changed')
    old=canonical_json_identity(identity);new=canonical_json_identity(receipt['new_identity'])
    for value in(old,new):
        for key in('source_files','execution_identity','sha256'):value.pop(key,None)
    if old!=new or receipt['new_identity']['execution_identity']!=execution:
        raise ValueError('Execution migration changed scientific fields or active execution')
    if not receipt.get('evidence'):raise ValueError('Execution migration requires actual evidence')
    for proof in receipt['evidence']:
        if (file_sha(proof['path'])!=proof['sha256']
                or json.loads(Path(proof['path']).read_text()).get('status')!='PASS'):
            raise ValueError('Execution migration evidence changed or failed')
    path=Path(checkpoint);transaction=json.loads((path/'transaction.json').read_text())
    if file_sha(path/'files.json')!=receipt['parent_files_sha256']:
        if (transaction.get('migration_receipt_sha256')!=sha
                or transaction.get('execution_identity')!=execution):
            raise ValueError('Checkpoint is neither audited parent nor this execution descendant')
