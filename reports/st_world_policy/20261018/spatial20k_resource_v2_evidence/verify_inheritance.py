"""Narrow resource-only evidence rebinding; no CSV/model/teacher payload read."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path('/mnt/project/st-world-policy-20261016/formal/spatial/registrations')
RUNTIME = Path('/mnt/project/VLA-Drive-st-official-runtime-v2-20261016')
OUT = Path(__file__).resolve().parent
OLD = OUT.parent/'spatial20k_actual_evidence_v1'
sys.path.insert(0, str(RUNTIME))
from tools.st_world_policy.check_formal_approval import verify_registration, file_hash

def pin(p):
    p=Path(p).resolve(); return {'path':str(p),'sha256':file_hash(p)}
def read(p): return json.loads(Path(p).read_text())
expected={'SP2':'f344cafa2352f1824a7a0d865aa8aaf2f5d680e2e328eefd4826834ffa0953f6','SP3':'ab7a0ae39d5f7d6662d6b9f2b8f01b6738f4b53d6ae20048f56e3ffff5a847b8'}
rows={}
for arm in ('SP2','SP3'):
    old_path=ROOT/f'{arm}_fulltrain_seed42_official_20k_v1/REGISTRATION.json'
    new_path=ROOT/f'{arm}_fulltrain_seed42_official_20k_v1_resource_v2/REGISTRATION.json'
    old,new=read(old_path),read(new_path)
    assert file_hash(new_path)==expected[arm]
    assert [k for k in old if old[k]!=new[k]]==['artifacts']
    assert [k for k in old['artifacts'] if old['artifacts'][k]!=new['artifacts'][k]]==['resource_lease']
    old_review=read(OLD/f'{arm}_EVIDENCE.json')
    assert old_review['reviewer_agent']=='/root/spatial20k_evidence'
    assert old_review['fingerprint']['registration_sha256']==file_hash(old_path)
    for ref in old_review['evidence_artifacts']:
        assert file_hash(ref['path'])==ref['sha256'],ref['path']
    actual=verify_registration(new_path,repo=RUNTIME)
    assert actual==read(new_path.parent/'FINGERPRINT.json')
    assert actual['run_id']==old_review['fingerprint']['run_id']
    assert '/root/spatial20k_evidence' not in new['implementer_agents']
    lease=read(new['artifacts']['resource_lease']['path'])
    cfg=read(new['artifacts']['config']['path'])
    assert lease['supersedes_unlaunched_lease']==old['artifacts']['resource_lease']
    assert lease['source_sha']==new['source_sha']
    assert lease['run_id']==new['run_id'] and lease['world_size']==new['execution']['world_size']
    assert lease['native_completed_start']==5000 and lease['native_completed_endpoint']==20000
    assert lease['maximum_optimizer_updates_this_stage']==15000
    assert lease['no_automatic_resume'] and lease['first_continuation_only']
    # This is a separate registration version for the same never-dispatched run.
    output=Path(cfg['runtime']['output_root'])/new['run_id']
    assert not output.exists(),output
    assert not (old_path.parent/'LAUNCH.json').exists()
    rows[arm]={'registration':pin(new_path),'old_registration':pin(old_path),'inherited_scientific_evidence':pin(OLD/f'{arm}_EVIDENCE.json'),'fingerprint':actual,'only_changed_artifact':'resource_lease','scientific_source_config_data_teacher_target_base_environment_execution_exactly_unchanged':True,'never_launched_output_absent_at_review':True,'new_lease':new['artifacts']['resource_lease']}
result={'schema':'st_spatial20k_resource_v2_evidence_inheritance_v1','reviewer_agent':'/root/spatial20k_evidence','created_utc':datetime.now(timezone.utc).isoformat(),'status':'PASS','arms':rows,'new_model_csv_teacher_scoring_or_bootstrap_calls':0,'source_native_registration_hash_checks':True,'resource_arithmetic_independently_approved_by_this_role':False,'launch_dispatched':False}
(OUT/'ACTUAL_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':'PASS','sha256':file_hash(OUT/'ACTUAL_CHECKS.json')}))
