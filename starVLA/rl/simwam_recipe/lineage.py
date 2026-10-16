"""Explicit, evidence-bound execution migrations; never bypass source hashes."""
import json
from pathlib import Path
from .contracts import fingerprint,file_sha

SCIENTIFIC_KEYS=('recipe','upstream_commit','SFT_sha256','hard_manifest_sha256',
    'projected_condition_identity','policy_ratio','deployment','world','purpose',
    'score_protocol','parameter_contract','rng_protocol','scene_reduction','actual_optimizer')


def scientific_identity(identity):
    fields={k:identity[k] for k in SCIENTIFIC_KEYS}
    if 'data_stream_identity' in identity:fields['data_stream_identity']=identity['data_stream_identity']
    return dict(schema='simwam_scientific_v2',sha256=fingerprint(fields),fields=fields)


def execution_identity(sources,options=None):
    fields=dict(source_files=sources,module_root=str(Path.cwd().resolve()),
                checkpoint_schema=2,control_backend='Gloo; no post-fatal NCCL save',
                DDP='find_unused_parameters=True/static_graph=False/bucket25MB unchanged')
    if options is not None:fields['options']=options
    return dict(schema='simwam_execution_v2',sha256=fingerprint(fields),fields=fields)


def verify_migration(path,previous,current,resume):
    receipt=json.loads(Path(path).read_text());payload=dict(receipt);sha=payload.pop('sha256')
    if fingerprint(payload)!=sha:raise ValueError('Migration receipt changed')
    if scientific_identity(previous)!=scientific_identity(current):raise ValueError('Algorithm/config/assets changed: not an execution migration')
    if receipt['old_identity']!=previous or receipt['new_identity']!=current:
        raise ValueError('Migration is not bound to these exact source identities')
    for evidence in receipt['evidence']:
        if file_sha(evidence['path'])!=evidence['sha256']:raise ValueError('Migration equivalence evidence changed')
        if json.loads(Path(evidence['path']).read_text())['status']!='PASS':raise ValueError('Migration equivalence not passed')
    resume=Path(resume)
    if file_sha(resume/'files.json')!=receipt['parent_files_sha256']:
        transaction=resume/'transaction.json'
        if not transaction.exists() or json.loads(transaction.read_text()).get('migration_receipt_sha256')!=sha:
            raise ValueError('Checkpoint is not the audited parent or its declared execution lineage')
    return receipt
