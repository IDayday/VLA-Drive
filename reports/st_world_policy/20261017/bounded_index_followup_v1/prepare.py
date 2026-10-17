"""Prepare two bounded metadata registrations. No admission/build/teacher I/O."""
import copy, hashlib, json
from pathlib import Path
ROOT=Path('/mnt/project/st-world-policy-20261016')
PREP=ROOT/'audits/indexed_composition_preparation_v1'
SOURCE=Path('/mnt/project/VLA-Drive-st-metadata-qualified-runtime-v1-20261017')
MIB=1<<20

def load(p):
    p=Path(p)
    if p.stat().st_size>MIB:raise ValueError('control exceeds 1MiB')
    return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
    raw=(json.dumps(d,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    if len(raw)>MIB:raise ValueError('prepared control exceeds 1MiB')
    if p.exists():
        if p.read_bytes()!=raw:raise ValueError('immutable preparation conflict: '+str(p))
    else:
        p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as f:f.write(raw)
    return sha(p)

def main():
    old=ROOT/'resources/CACHE_STORAGE_BUDGET_POLICY_v6.json'
    policy=load(old)
    template=load(ROOT/'audits/metadata_migration_preparation_v1/CURRENT_INDEX_STORAGE_REGISTRATION_V2.json')
    fit=ROOT/'audits/history_data/native384_train64_cost_v1'
    fit_identity=load(fit/'identity.json')
    future=ROOT/'caches/future_metadata_qualification32_v2'
    future_complete=load(future/'COMPLETE.json')
    entries={
      'request_index':dict(writer_kind='future_index',population=32,root=ROOT/'caches/future32_request_metadata_index_v1',
        flags=['--requests',str(future/'TEACHER_REQUESTS.json')],
        request_identity=future_complete['request_identity']),
      'fit_index':dict(writer_kind='current_index',population=64,root=ROOT/'caches/original_pca_fit64_metadata_index_v1',
        flags=['--cache-root',str(fit),'--expected-cache-identity',fit_identity['identity']],
        request_identity=fit_identity['request_manifest_identity'])}
    prepared={'schema':'st_bounded_index_followup_preparation_v1','status':'PREPARED_NOT_ADMITTED',
      'source':str(SOURCE),'source_sha':'1162b955','base_policy':str(old),'base_policy_sha256':sha(old),
      'state_dir':policy['state_dir'],'additional_reserved_bytes':64*MIB,'stages':{}}
    for stage,item in entries.items():
        reg=copy.deepcopy(template)
        reg.update(writer_kind=item['writer_kind'],output_root=str(item['root']),
           storage_requirements=dict(output_bytes=14*MIB,staging_bytes=14*MIB,metadata_bytes=4*MIB))
        reg['limits'].update(max_disk_bytes=32*MIB,max_rss_bytes=2<<30,max_seconds=180)
        flags=['--expected-request-identity',item['request_identity'],
          '--contract-receipt',str(ROOT/'audits/fixed_teacher_batch64_runtime_run_v1/RESOLVED_PLAN.json'),
          '--contract-receipt-sha256',sha(ROOT/'audits/fixed_teacher_batch64_runtime_run_v1/RESOLVED_PLAN.json'),
          '--output',str(item['root']),'--max-scenes',str(item['population']),
          '--max-disk-bytes',str(28*MIB),'--max-builder-rss-bytes',str(2<<30)]
        reg['command']=[template['command'][0],'-m','tools.st_world_policy.index_teacher_metadata','build',*item['flags'],*flags]
        regpath=PREP/(stage.upper()+'_REGISTRATION.json'); regsha=save(regpath,reg)
        claim=dict(root=str(item['root']),claim_id='bounded-'+stage.replace('_','-')+'-v1',
          identity_kind='st_metadata_build_registration_v1',cache_identity=regsha,
          registration_path=str(regpath),registration_sha256=regsha,**reg['storage_requirements'])
        if any(x['root']==claim['root'] or x['claim_id']==claim['claim_id'] for x in policy['claims']):
            raise ValueError('duplicate planned claim')
        policy['claims'].append(claim)
        prepared['stages'][stage]={'registration':str(regpath),'registration_sha256':regsha,'output_root':str(item['root'])}
    # Same core, same existing claims, same 4GiB package/768GiB campaign envelope.
    # These append-only declarations need no migration or independent ledger.
    new=ROOT/'resources/CACHE_STORAGE_BUDGET_POLICY_v7_index_followup.json'
    prepared.update(policy=str(new),policy_sha256=save(new,policy))
    p=PREP/'PREPARED.json'; print(json.dumps({'prepared':str(p),'sha256':save(p,prepared),'status':prepared['status']}))
if __name__=='__main__':main()
