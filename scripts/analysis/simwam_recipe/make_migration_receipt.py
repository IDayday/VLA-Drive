"""Explicit execution-only source migration, requiring matching real parity evidence.

Does not edit the checkpoint, identity, source snapshot, or start training.
"""
import argparse
import json
from pathlib import Path
from scripts.analysis.simwam_recipe.audit_checkpoints import inspect
from starVLA.rl.simwam_recipe.lineage import scientific_identity
from starVLA.rl.simwam_recipe.contracts import file_sha,fingerprint
from starVLA.rl.flow_grpo.transactions import atomic_json


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--new-identity',required=True)
    p.add_argument('--parity-evidence',required=True);p.add_argument('--recovery-evidence',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    parent=Path(a.checkpoint);audit=inspect(parent)
    if audit['status']!='COMMITTED_BOUNDARY':raise ValueError(audit)
    old=json.loads((parent/'identity.json').read_text());new=json.loads(Path(a.new_identity).read_text())
    if scientific_identity(old)!=scientific_identity(new):raise ValueError('Not an algorithm-preserving execution migration')
    parity=json.loads(Path(a.parity_evidence).read_text());recovery=json.loads(Path(a.recovery_evidence).read_text())
    if parity['status']!='PASS' or parity['source_identities']!=[old['sha256'],new['sha256']]:
        raise ValueError('Parity evidence is not for these exact old and new sources')
    if recovery['status']!='PASS' or not recovery['optimizer_adapters_scheduler_RNG_exact']:
        raise ValueError('Exact inner recovery not demonstrated')
    # Evidence must belong to the same destination, not an unrelated passing test.
    recovered=json.loads((Path(recovery['continuous'])/'identity.json').read_text())
    if recovered!=new:raise ValueError('Recovery evidence source identity differs')
    changed={k:dict(old=old['source_files'].get(k),new=new['source_files'].get(k)) for k in
        sorted(set(old['source_files'])|set(new['source_files'])) if old['source_files'].get(k)!=new['source_files'].get(k)}
    allowed={'scripts/analysis/simwam_recipe/train.py'}|{'starVLA/rl/simwam_recipe/'+n for n in
        ('checkpoint.py','collection.py','contracts.py','recovery_state.py','invocation.py','lineage.py')}
    if set(changed)-allowed:raise ValueError('Migration changes actor/sampler/loss/imported dependency outside the audited execution patch')
    result=dict(schema='simwam_execution_migration_v2',classification='B normal path unchanged; recovery transaction only',
        parent_checkpoint=str(parent),parent_files_sha256=file_sha(parent/'files.json'),
        old_identity=old,new_identity=new,scientific_identity=scientific_identity(old),source_diff=changed,
        evidence=[dict(path=str(Path(p).resolve()),sha256=file_sha(p)) for p in (a.parity_evidence,a.recovery_evidence)],
        immutable_old_source_files_preserved=True)
    result['sha256']=fingerprint(result);out=Path(a.output)
    if out.exists():raise FileExistsError('Migration receipts are immutable')
    atomic_json(out,result);print(json.dumps(dict(status='RECEIPT_CREATED',output=str(out),sha256=result['sha256'])))


if __name__=='__main__':main()
