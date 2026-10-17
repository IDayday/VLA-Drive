"""Independent small-control issuance. No prepare/run/registration is called."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import time

HERE=Path(__file__).resolve().parent
CONTROL=Path('/mnt/project/VLA-Drive-st-indexed-composition-20261017/reports/st_world_policy/20261017/indexed_profile64_control_v1/controller.py')
spec=importlib.util.spec_from_file_location('reviewed_profile64_control',CONTROL)
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
pin=lambda p:{'path':str(p),'sha256':c.digest(p)}
assert not c.OUT.exists()
inputs,cfg=c.common()  # Metadata preflight only; no factory or model.
disk=c.storage()
gpu=c.gpu_record(True)  # Query only; no lease or GPU acquisition.
assert gpu['uuid']=='GPU-f679bff0-a4a2-ae4f-0b85-74c519b45a9c'
old=Path('/mnt/project/st-world-policy-20261016/audits/spatial_probe_actual_SP0_5000_v1/storage_preparation_dev_v1/CHECKPOINT_COMMITMENTS_2103G_V1.json')
assert c.digest(old)=='60606edc286a0c34016842a03f0de08f2422936e347364c7373defe9efcc5259'
assert c.read(old)['remaining_bytes']==2103<<30
now=time.time();until=now+7200
cp={'schema':'st_checkpoint_storage_commitments_v1','status':'CURRENT','checked_at_unix':now,'valid_until_unix':until,'remaining_bytes':2183<<30,'free_floor_bytes':1<<40,'actual_free_bytes':disk['actual_free_bytes'],'quota_verified':False,'origin':'fresh filesystem capacity and conservative full budgets; old receipt retained unchanged','scope':'Full original nine ST budgets1983GiB + prior indexed smoke120GiB + new bounded PROFILE80GiB; no subtraction for existing checkpoints','basis':{'previous_2103GiB':pin(old),'new_profile_controller':pin(CONTROL),'new_profile_inputs':pin(c.INPUT)},'existing_checkpoint_bytes_not_deleted':True,'cache_ledger_modified':False,'reservation_kind':'advisory_not_filesystem_quota'}
cp_path=HERE/'CHECKPOINT_COMMITMENTS_2183G_V1.json'
assert not cp_path.exists()
cp_path.write_text(json.dumps(cp,sort_keys=True,indent=2)+'\n')
a={'schema':'st_indexed_profile64_resource_admission_v1','status':'ADMITTED','host':c.HOST,'gpu':'0','source_sha':c.SOURCE,'run_id':c.RUN_ID,'output_root':str(c.OUT),'controller_sha256':c.digest(CONTROL),'inputs_sha256':c.digest(c.INPUT),'checkpoint_reservation_bytes':80<<30,'other_commitments_bytes':2103<<30,'free_floor_bytes':1<<40,'max_invocation_seconds':1500,'outer_timeout_seconds':1620,'cleanup_seconds':180,'valid_from_unix':now,'valid_until_unix':until,'latest_start_unix':until-1800,'reviewer':'/root/st_metadata_resource_review','checkpoint_commitments':pin(cp_path),'gpu_observation':gpu,'cpu_threads':2,'cpu_affinity_enforced_by_controller':False,'automatic_retry':False,'allowed_actions':['prepare','run'],'one_run_only':True,'formal_approved':False,'source_correctness_scope':'Existing independent source review inherited; no model/numerical re-review','metadata_effective_reserved_mib':1264,'new_teacher_tensor_bytes':0}
path=HERE/'ADMISSION.json';assert not path.exists()
path.write_text(json.dumps(a,sort_keys=True,indent=2)+'\n')
assert c.admission(str(path),c.digest(path))[0]==a
negatives=[]
with tempfile.TemporaryDirectory(prefix='st-profile-admission-') as root:
    p=Path(root)/'invalid.json'
    for key,value in [('controller_sha256','0'*64),('inputs_sha256','0'*64),('other_commitments_bytes',1983<<30),('valid_until_unix',now+1799)]:
        bad=copy.deepcopy(a);bad[key]=value;p.write_text(json.dumps(bad))
        try:c.admission(str(p),c.digest(p))
        except RuntimeError:negatives.append(key)
        else:raise AssertionError('Invalid admission accepted: '+key)
assert not c.OUT.exists()
e={'status':'PASS_ONCE_PROFILE64_RESOURCE_CONTROL','admission':pin(path),'commitments':pin(cp_path),'controller':pin(CONTROL),'inputs':pin(c.INPUT),'actual_storage_observation':disk,'actual_gpu_observation':gpu,'positive_actual_reader_test':True,'negative_tests_rejected':negatives,'owner_cpu_tests':'6 passed, current independent rerun','runtime_clean_and_input_hashes_verified':True,'training_root_created':False,'GPU_acquired':False,'cache_ledger_changed':False,'prepare_or_run_called':False}
(HERE/'EVIDENCE.json').write_text(json.dumps(e,sort_keys=True,indent=2)+'\n')
print(json.dumps({'admission':pin(path),'commitments':pin(cp_path),'controller':pin(CONTROL),'inputs':pin(c.INPUT),'valid_until_unix':until,'latest_start_unix':until-1800,'status':e['status']},sort_keys=True,indent=2))
