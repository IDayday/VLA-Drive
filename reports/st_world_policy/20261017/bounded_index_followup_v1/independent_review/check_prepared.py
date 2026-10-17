"""Read-only exact small controls/stat; no StorageBudget initialization/build."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path('/mnt/project/st-world-policy-20261016')
SOURCE=Path('/mnt/project/VLA-Drive-st-metadata-qualified-runtime-v1-20261017')
HERE=Path(__file__).parent
PREP=ROOT/'audits/indexed_composition_preparation_v1/PREPARED.json'
CONTROL=Path('/mnt/project/VLA-Drive-st-world-policy-20261016/reports/st_world_policy/20261017/bounded_index_followup_v1/run_once.py')
CP=ROOT/'audits/metadata_migration_preparation_v1/CHECKPOINT_COMMITMENTS_V1.json'
pins=[]; checks={}


def small(path,expected=None,json_value=True):
    path=Path(path)
    assert path.is_absolute() and path.resolve()==path and path.stat().st_size<=1<<20
    with path.open('rb') as stream:
        before=os.fstat(stream.fileno()); raw=stream.read((1<<20)+1); after=os.fstat(stream.fileno())
    def version(stat): return (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
    assert len(raw)==before.st_size and version(before)==version(after)==version(path.stat())
    digest=hashlib.sha256(raw).hexdigest()
    if expected is not None: assert digest==expected,(path,digest)
    pins.append(dict(path=str(path),bytes=len(raw),sha256=digest))
    return json.loads(raw) if json_value else raw


prep=small(PREP,'b2afa6c88c1b26764f7226efb87340f387d25fa20f7c7ba6e8bd32164042fd19')
old=small(prep['base_policy'],prep['base_policy_sha256'])
new=small(prep['policy'],prep['policy_sha256'])
ledger_path=Path(prep['state_dir'])/'CLAIMS.json'
ledger=small(ledger_path); ledger_sha=pins[-1]['sha256']
checks['same_exact_core']=all(old.get(k)==v==new.get(k) for k,v in ledger['core'].items())
checks['same_policy_except_two_appended_claims']={k:v for k,v in new.items() if k!='claims'}=={k:v for k,v in old.items() if k!='claims'}
checks['preserve_old_declarations']=new['claims'][:len(old['claims'])]==old['claims']
added=new['claims'][len(old['claims']):]
checks['only_two_new_roots']=len(added)==2 and {x['root'] for x in added}=={v['output_root'] for v in prep['stages'].values()}
checks['not_yet_admitted']=all(x['root'] not in ledger['claims'] and not Path(x['root']).exists() for x in added)
checks['campaign_cap_floor_unchanged']=new['cache_cap_bytes']==768<<30 and new['free_reserve_bytes']==1024<<30
checks['metadata_cap_unchanged']=new['metadata_package']==old['metadata_package'] and new['metadata_package']['maximum_bytes']==4<<30
total=lambda x:sum(x[k] for k in ('output_bytes','staging_bytes','metadata_bytes'))
checks['64MiB_addition']=sum(total(x) for x in added)==prep['additional_reserved_bytes']==64<<20
package=new['metadata_package']; baseline_ids={r['claim_id'] for r in package['baseline_legacy_reservations']}
metadata=sum(total(x) for x in new['claims'] if x['root'] not in package['baseline_claims'])+sum(x['reserved_bytes'] for x in new['legacy_reservations'] if x['claim_id'] not in baseline_ids)
aggregate=new['ledger_reserve_bytes']+sum(x['reserved_bytes'] for x in new['legacy_reservations'])+sum(total(x) for x in new['claims'])
checks['total_metadata_within_four_GiB']=metadata<=4<<30
checks['all_declared_within_campaign']=aggregate<=768<<30
fit=small(ROOT/'audits/history_data/native384_train64_cost_v1/identity.json')
future=small(ROOT/'caches/future_metadata_qualification32_v2/COMPLETE.json')
checks['future_metadata_complete_only']=future['status']=='COMPLETE' and future['tensor_disk_bytes']==0 and future['teacher_extractions']==0 and future['training_admission'] is False
for stage,item in prep['stages'].items():
    reg=small(item['registration'],item['registration_sha256'])
    row=next(x for x in added if x['root']==item['output_root'])
    checks[stage+'_typed_registration']=row['identity_kind']==reg['schema']=='st_metadata_build_registration_v1' and row['cache_identity']==row['registration_sha256']==item['registration_sha256']
    checks[stage+'_32MiB_budget']=total(row)==32<<20 and reg['storage_requirements']=={k:row[k] for k in ('output_bytes','staging_bytes','metadata_bytes')}
    limits=reg['limits']
    checks[stage+'_finite_limits']=limits==dict(max_depth=0,max_disk_bytes=32<<20,max_entries=100,max_rss_bytes=2<<30,max_seconds=180)
    checks[stage+'_runtime_contract']=reg['cwd']==str(SOURCE) and reg['callback_protocol']=='st_metadata_storage_check_v1'
    command=reg['command']; flags=dict(zip(command[4::2],command[5::2]))
    checks[stage+'_command']=command[:4]==['/root/miniconda3/envs/ddp/bin/python','-m','tools.st_world_policy.index_teacher_metadata','build'] and flags['--output']==item['output_root'] and flags['--max-disk-bytes']==str(28<<20) and flags['--max-builder-rss-bytes']==str(2<<30)
    if stage=='fit_index':
        checks[stage+'_source_binding']=reg['writer_kind']=='current_index' and flags['--max-scenes']=='64' and flags['--expected-cache-identity']==fit['identity'] and flags['--expected-request-identity']==fit['request_manifest_identity'] and flags['--cache-root']==str(ROOT/'audits/history_data/native384_train64_cost_v1')
    else:
        checks[stage+'_source_binding']=reg['writer_kind']=='future_index' and flags['--max-scenes']=='32' and flags['--expected-request-identity']==future['request_identity'] and flags['--requests']==str(ROOT/'caches/future_metadata_qualification32_v2/TEACHER_REQUESTS.json')
    small(flags['--contract-receipt'],flags['--contract-receipt-sha256'])
    for ref in reg['source_pins']: small(ref['path'],ref['sha256'],False)
small(CONTROL,json_value=False); controller_sha=pins[-1]['sha256']
checks['reviewed_controller_matches_exact_live_file']=small(HERE.parent/'run_once.py',json_value=False)==CONTROL.read_bytes()
checks['runtime_clean_and_fixed']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()=='1162b955b3bf9abb6728823a84a50c38e1a19bb8' and not subprocess.check_output(['git','status','--porcelain'],cwd=SOURCE).strip()
checkpoint=small(CP,'1bf12212b29202cd3f5cf80e73425ab5d663c778e7ba35df43a83bb2c27289ad')
checks['checkpoint_reserve_full']=checkpoint['status']=='CURRENT' and checkpoint['remaining_bytes']==1983<<30 and checkpoint['checked_at_unix']<=time.time() and checkpoint['valid_until_unix']>=time.time()+600
free=shutil.disk_usage(ROOT).free
checks['conservative_disk_floor']=free >= (768+1024+1983)<<30
dispatch=small('/mnt/project/server_dispatch_policy.json')
checks['host_still_authorized']='training-vla-zt' in dispatch['allowed_hosts']
writer_times=[]
for filename,expected in [('WRITER_OBSERVATION.json','f7ce511309fd7db03dde66219187b76392393c79b1bc46fa5aca37171ce93af7'),
                          ('TEACHER_WRITER_OBSERVATION.json','5e913c5806bd0c49341dac4cf39f2e55a9f061eacaa0448033077decfa7a7dc8')]:
    observation=small(PREP.parent/filename,expected)
    rows=observation['rows']; writer_times.extend(row['checked_unix'] for row in rows)
    checks[filename+'_fresh_empty_writers']=observation['owner']=='Lead' and len(rows)==8 and len({x['host'] for x in rows})==8 and all(row['writer_processes']==[] and 0<=time.time()-row['checked_unix']<=600 for row in rows)
    checks[filename+'_queues_explicit']=bool(observation['automatic_queues'])
checks['ledger_unchanged']=hashlib.sha256(ledger_path.read_bytes()).hexdigest()==ledger_sha
assert all(checks.values()),{k:v for k,v in checks.items() if not v}
report=dict(schema='st_bounded_index_followup_independent_controls_v1',status='PASS_PREPARED_CONTROLS_ONLY',checked_at_unix=time.time(),checks=checks,pins=pins,ledger_sha256=ledger_sha,controller_sha256=controller_sha,metadata_reserved_bytes=metadata,aggregate_declared_bytes=aggregate,filesystem_available_bytes=free,earliest_writer_observation_unix=min(writer_times),checkpoint_receipt=str(CP),checkpoint_sha256='1bf12212b29202cd3f5cf80e73425ab5d663c778e7ba35df43a83bb2c27289ad',real_ledger_mutations=0,real_builds=0,image_tensor_native_opens=0,formal_training_admission=False)
(HERE/'PREPARED_CONTROLS.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(status=report['status'],checks=len(checks),controller_sha256=controller_sha,metadata_bytes=metadata,aggregate_bytes=aggregate,free_bytes=free)))
